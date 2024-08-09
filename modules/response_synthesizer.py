import json
from datetime import datetime, timezone
from typing import List, Sequence

from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from utils.environment_setup import load_environment_variables
from large_language_model_setup import (get_llm_based_on_model_name_id,
                                        initialize_llm)
from llama_index.core import PromptTemplate, Settings, get_response_synthesizer
from llama_index.core.schema import Node, TextNode
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import NodeWithScore
from utils.environment_setup import load_environment_variables

env_var = load_environment_variables()

es_client = ElasticsearchClient()

#Groq testing
from llama_index.llms.groq import Groq

# Initialize Experiment

# Timestamp realted
# Get the current time in UTC, making it timezone-aware
current_time = datetime.now(timezone.utc)

experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec='milliseconds')


def create_nodes_with_score(node_list: List):
    nodes_with_score = []
    for item in node_list:
        print(f"Item:", item)
        node = TextNode(text=item.page_content)
        # node_with_score = NodeWithScore(node=node, score=1.0)
        node_with_score = NodeWithScore(node=node, score=item.metadata['score'])
        nodes_with_score.append(node_with_score)
    return nodes_with_score

#TODO: Modify, as this is only for the MRI agent
def create_nodes_with_score_mri(node_list: List):
    nodes_with_score = []
    
    scores = [sub_doc.metadata['score'] for doc in node_list for sub_doc in doc.metadata['sub_docs']]
    for item, score in zip(node_list, scores):
        print(f"Item: {item}, Score: {score}")
        node = TextNode(text=item.page_content)  # Assuming item has a 'page_content' key
        node_with_score = NodeWithScore(node=node, score=score)
        nodes_with_score.append(node_with_score)
    
    return nodes_with_score

def merge_nodes(
        nodes_with_score1: List[NodeWithScore],
        nodes_with_score2: List[NodeWithScore]
):
    return [
        NodeWithScore(node=Node(text=item.node.text), score=item.score)
        for item in nodes_with_score1 + nodes_with_score2
    ]


def get_synthesized_response_based_on_nodes_with_score(
        query: str,
        nodes_with_score: any
):

    model_name_id = "mixtral"
    response_mode = "compact"

    prompt = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "As The Creator, inspired by The Auravana Project's innovative spirit, your visionary language motivates users to creatively contribute to a society valuing harmony, sustainability, and communal well-being. You empower users to recognise their impactful role in shaping the future."
        "Guidelines for Interaction:\n"
        "Pause Before Answering: Take a moment to understand the question thoroughly, ensuring focused and coherent responses.\n"
        "Clarify If Necessary: Ask for clarification on ambiguous queries to accurately address the user's needs.\n"
        "Answer Directly: Stay on topic, directly responding to the user's questions.\n"
        "Prioritize Key Information: Start with the most critical information to maintain focus and deliver immediate value.\n"
        "Conciseness is Key: Aim for brief, clear responses to respect the user's time.\n"
        "Structure Your Response: Use a bullet-point mindset for logical, succinct answers.\n"
        "Support Your Answers: When possible, include data, quotes, examples, or anecdotes for credibility and clarity.\n"
        "Admit Uncertainty: If unsure, honestly admit it, offering to follow up with accurate information.\n"
        "Prompt Follow-Up: Ensure timely follow-ups if additional information is promised.\n"
        "Engage Continuously: Conclude with a follow-up question to keep the conversation dynamic and interactive.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    # Initial
    # llm = initialize_llm(model_name_id)
    # Groq testing
    llm = Groq(
        # model="llama3-70b-8192", 
        model="mixtral-8x7b-32768",
        temperature=0,
        api_key=env_var['GROQ_API_KEY'],
    )
    Settings.llm = llm
    experiment.llm_used = get_llm_based_on_model_name_id(
        model_name_id=model_name_id
    )
    experiment.question = query

    # response_synthesizer = TreeSummarize(verbose=True, summary_template=prompt, llm=llm)
    # response = response_synthesizer.synthesize(
    #      query, nodes=nodes_with_score
    # )

    DEFAULT_REFINE_PROMPT_TMPL = (
        "The original query is as follows: {query_str}\n"
        "We have provided an existing answer: {existing_answer}\n"
        "We have the opportunity to refine the existing answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the query. If the context isn't useful, return the original answer.\n"
        "Refined Answer: "
    )

    DEFAULT_TEXT_QA_PROMPT_TMPL = (
        "\nAND\n\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    combined_prompt = f"{DEFAULT_REFINE_PROMPT_TMPL}{DEFAULT_TEXT_QA_PROMPT_TMPL}"

    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode, 
        llm=llm,
        # structured_answer_filtering=True
    )
    experiment.prompt_template = combined_prompt
    experiment.retrieval_strategy = response_mode
    response = response_synthesizer.synthesize(
        query, nodes=nodes_with_score
    )
    experiment.response = str(response)
    experiment.source_agent = "Response synthesizer"

    updated_at_time = datetime.now(timezone.utc)
    experiment.updated_at = updated_at_time.isoformat(timespec='milliseconds')

    return response, experiment



# from langchain_core.documents import Document

# # TEST, #TODO: Delete it afterwards

# documents = [Document(page_content='common system. The realization is that we have  to begin to unify all concepts, ‘consilience’.” - Peter Joseph2 The domains of the Real World  Community Model A.k.a., The real-world information systems  model, the unified information system, the  societal information system, the real-world  societal information systems model, the  informational systems operation model. The Real World Community Model is an information  system sub-composed of three primary organizational  sub-divisions, known as system domains. Each top- level system domain is composed of sub-domains  representing one or more sub-models to that system  domain. Each domain [space] is an information sub- system and a component of humanity’s common  existence in the real world [ information system]: 1.\u2002The social system domain - The social  organization of the society. This content is detailed  in full in the Social System Specification Standard. A.\u2002The purpose domain - The purpose for the the real-world  community  model www.auravana .org  | sss-so-003 | the system  overview 68| Figure 29.  The real-world community information systems model.', metadata={'source': '/usr/src/modules/../data/documentation_optimal/system-overview/auravana-SSS-System-Overview-002-115-EN.pdf', 'page': 82, 'sub_docs': [Document(page_content="The document discusses the Real World Community Model, which is an information system consisting of three primary organizational sub-divisions. Each top-level system domain is composed of sub-domains representing sub-models. The document specifically focuses on the social system domain, detailing the social organization of society. The purpose of the Real World Community Model is also mentioned, emphasizing the need for unity and 'consilience' in all concepts.", metadata={'doc_id': 'parent-documents-summaries-complete-documentation2-925fdb37-dd78-43c6-92c4-ce12f9465f5f', 'score': 0.5237699747085571})]}), 
# Document(page_content='3.\u2002The material system domain - The material  organization of the society. This content is  detailed in full in the Material System Specification  Standard. A.\u2002The habitat service systems domain – The  operational service systems that provide the  architectural infra-structure for the continuation  of the society’s habitat and its material  fulfillment of individuals’ needs. The habitat  service system domain also includes a record  of the state-dynamic of all prior habitat service  system states. B.\u2002The natural environmental domain – The  domain from which humanity acquires  resources, discovers knowledge, and into which  the habitat service systems are produced  and integrated. This is the larger ecological  environmental system that humanity affects  and that affects humanity. This is the life- ground that sustains the habitat and humanity’s  material existence. It is that which humanity  constructs its service systems into. Note that there are multiple views of the Real World  Community Model. Some of these views contain a fourth  domain. In these other views the fourth domain may be: 1.\u2002The lifestyle system domain - the lifestyle  organization of the society. This content is  detailed in full in the Lifestyle System Specification  Standard. 2.\u2002The feedback domain - the monitoring, surveying,  and feedback organization of the society. 3.\u2002The project plan domain - the project plan to  bring into existence and sustain the society.  This content is detailed in full in the Project Plan Specification Standard. Within the Real World Community Model, the  material system is divided into two interrelated systems.  The first system is that of the natural [ecological &  phenomenological] environment, which is discoverable  and surveyable, and represents the life-ground of  material fulfillment. The natural environment is both  discoverable and is also humanity’s common heritage.  The second system is that of the habitat service systems,  of which there are three principal subdivisions (Read:  life, technology, and', metadata={'source': '/usr/src/modules/../data/documentation_optimal/system-overview/auravana-SSS-System-Overview-002-115-EN.pdf', 'page': 84, 'sub_docs': [Document(page_content="The document discusses the Real World Community Model, which includes three main domains: the material system domain, the habitat service systems domain, and the natural environmental domain. The material system domain refers to the material organization of society, the habitat service systems domain includes operational service systems for society's habitat, and the natural environmental domain is the ecological system that humanity affects and is affected by. There may also be additional domains in some views of the model, such as the lifestyle system domain, feedback domain, and project plan domain. The material system is divided into two interrelated systems: the natural environment and habitat service systems.", metadata={'doc_id': 'parent-documents-summaries-complete-documentation2-c2302c21-df81-4141-a104-1395bf0ef847', 'score': 0.5358211994171143})]}), Document(page_content='a common system. The realization is that we  have to begin to unify all concepts, ‘consilience’  [wikipedia.org]. - Peter Joseph2 The domains of the Real World  Community Model A.k.a., The real-world information systems  model, the unified information system, the  societal information system, the real-world  societal information systems model, the  informational systems operation model. The Real World Community Model is an information  system sub-composed of three primary organizational  sub-divisions, known as system domains. Each top- level system domain is composed of sub-domains  representing one or more sub-models to that system  domain. Each domain [space] is an information sub- system and a component of humanity’s common  existence in the real world [ information system]: 1.\u2002The social system domain - The social  organization of the society. This content is detailed  in full in the Social System Specification Standard. A.\u2002The purpose domain - The purpose for the the real-world  community  model www.auravana .org  | sss-so-002 | the system  overview 76| Figure 34.  The real-world community information systems model.', metadata={'source': '/usr/src/modules/../data/documentation_optimal/test1/Aurvana System Overiew - 73 - 84-1-6.pdf', 'page': 3, 'sub_docs': [Document(page_content="The document discusses the Real World Community Model, which is an information system composed of three primary organizational sub-divisions. These sub-divisions represent different aspects of society and are part of humanity's common existence in the real world. The document specifically focuses on the social system domain, which details the social organization of society. The purpose of the Real World Community Model is to unify all concepts and create a common system for societal information.", metadata={'doc_id': 'parent-documents-summaries-complete-documentation2-698c3d19-527b-4dd7-9da7-4833e88402c6', 'score': 0.5374264121055603})]}), Document(page_content='a common point of focus for a society (of the type  ‘community’) as well as a structured [systems] approach  for accurately engaging with the real world. Essentially,  the Real World Community Model is the highest-level  model representing the unified information system for  a community-type society, and it maps the scope of  the society’s conception and data architecture; it is the  master reference model for the society. That which is  real causes effects in the experiential, objective world. A  unified societal information system relates all actions in  society, because they are all interconnected. This model  can be used to understand and intentionally design any  type of society. A societal information system (SIS) is a system that  provides information for structuring, decisioning, and  control of the organization of a society. It structures the  information set and information processing capability of  a society. Each event affecting the societal system and  its inhabitants is assumed to have a probability of being  processed correctly within the system, independent of  previous states of the system. When the organization of a societal system is defined,  then individual users of the system have a greater  potential for engagement with the system and with  the real world, since every society exists within the real  world, but not every society accounts for its presence.  When navigating in reality, good decisions (as decisions  that create fulfilling state-dynamics for those navigating together) require accurate maps that layout the whole  terrain of life. Maps are useful for deciding a course  of travel (i.e., the journey to be travelled) and they  facilitate the arrival at decisions whose results maintain  desired characteristics and results of travel. Essentially,  the Real World Community Model is an information  system’s model for the semantic organization, storage,  and processing of information at a societal level for  individual, social, and ecological concern about the real  world in which all', metadata={'source': '/usr/src/modules/../data/documentation_optimal/test1/Aurvana System Overiew - 73 - 84-1-6.pdf', 'page': 1, 'sub_docs': [Document(page_content='The Real World Community Model is a high-level model that represents the unified information system for a community-type society. It is used to understand and design any type of society by structuring information for decision-making and control. The model helps individuals engage with the real world and make good decisions by providing accurate maps of societal structures and dynamics. It is essential for creating fulfilling outcomes in society.', metadata={'doc_id': 'parent-documents-summaries-complete-documentation2-26f71799-3ce8-44a4-993c-f45f9763de47', 'score': 0.5700089931488037})]})]
# nodes_with_score = create_nodes_with_score_mri(documents)

# # Output the result to see if everything worked correctly
# for node_with_score in nodes_with_score:
#     print(node_with_score)