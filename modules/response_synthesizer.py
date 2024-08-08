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
        print(f"Item:", )
        node = TextNode(text=item.page_content)
        node_with_score = NodeWithScore(node=node, score=1.0)
        # node_with_score = NodeWithScore(node=node, score=item.metadata['score'])
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
        model="llama3-70b-8192", 
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
