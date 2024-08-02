import logging
import sys

from format_message_with_prompt import format_message
from prompts import TEMPLATES

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def query_knowledge_graph(index,
                          query,
                          template_id="default",
                          response_mode="tree_summarize",
                          embedding_mode="hybrid", 
                          llm="default"):
    """Query the Knowledge Graph with a specified query, using a specified message template.

    Args:
        index (KnowledgeGraphIndex): The Knowledge Graph Index to query.
        query (str): The query string.
        template_id (str, optional): Identifier for the predefined message template. Defaults to "default".
        response_mode (str, optional): The mode of response from the query engine. Defaults to "tree_summarize".
        embedding_mode (str, optional): The mode of embedding to use for the query. Defaults to "hybrid".

    Returns:
        str: The response from the query.
    """
    query_engine = index.as_query_engine(include_text=True, response_mode=response_mode, embedding_mode=embedding_mode, similarity_top_k=5, llm=llm)
    
    print("!!!!!!!!!!!!!!!!Template received, knowledge graph:", template_id)
    message_template = format_message(query, template_id)
    response = query_engine.query(query)
    print("!!!!!!!!!!!!!!!!Final question, knowledge graph:", message_template)
    # logging.info(f"Logging the response nodes from knowledge graph: {response.source_nodes}")
    return response.response, response.source_nodes
    # return response.response.split("")[-1].strip()