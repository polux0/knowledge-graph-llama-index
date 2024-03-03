import logging
from prompts import TEMPLATES

def query_knowledge_graph(index, query, template_id="default", response_mode="tree_summarize", embedding_mode="hybrid"):
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
    query_engine = index.as_query_engine(include_text=True, response_mode=response_mode, embedding_mode=embedding_mode, similarity_top_k=5)
    
    # Select the template based on the template_id
    if template_id in TEMPLATES:
        message_template = TEMPLATES[template_id].format(query=query)
    else:
        logging.error(f"Invalid template_id: {template_id}. Falling back to default template.")
        message_template = TEMPLATES["default"].format(query=query)

    logging.info(f"Sending the following message to the query engine: {message_template}")
    response = query_engine.query(message_template)
    logging.info(f"Logging the whole response before stripping it: {response}")
    return response.response
    # return response.response.split("")[-1].strip()
