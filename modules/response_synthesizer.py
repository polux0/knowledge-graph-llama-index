from typing import List
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer
from environment_setup import load_environment_variables
from large_language_model_setup import initialize_llm
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import TreeSummarize
from typing import Sequence
from elasticsearch_service import ExperimentDocument, ElasticsearchClient
from datetime import datetime, timezone
from large_language_model_setup import get_llm_based_on_model_name_id

es_client = ElasticsearchClient()

# Initialize Experiment

# Timestamp realted
# Get the current time in UTC, making it timezone-aware
current_time = datetime.now(timezone.utc)

experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec='milliseconds')

def merge_nodes(nodes_with_score1: List[NodeWithScore], nodes_with_score2: List[NodeWithScore]): 
    return [
        NodeWithScore(node=Node(text=item.node.text), score=item.score)
        for item in nodes_with_score1 + nodes_with_score2
    ]
                
def get_synthesized_response_based_on_nodes_with_score(query: str, nodes_with_score: any):
    model_name_id="default"
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

    # Extracting the 'text' attribute from each TextNode inside NodeWithScore
    texts: Sequence[str] = [node_with_score.node.text for node_with_score in nodes_with_score]


    llm = initialize_llm(model_name_id)
    experiment.llm_used = get_llm_based_on_model_name_id(model_name_id=model_name_id)
    experiment.question = query


    # response_synthesizer = TreeSummarize(verbose=True, summary_template=prompt, llm=llm)
    # response = response_synthesizer.synthesize(
    #      query, nodes=nodes_with_score
    # )

    response_synthesizer = get_response_synthesizer(response_mode=response_mode, llm=llm)
    experiment.retrieval_strategy = response_mode
    response = response_synthesizer.synthesize(
        query, nodes=nodes_with_score
    )
    experiment.response = str(response)
    experiment.source_agent = "Response synthesizer"
    experiment.updated_at = current_time.isoformat(timespec='milliseconds')
    
    # response = response_synthesizer.synthesize(
    #     query, [texts]
    # )
    return response, experiment