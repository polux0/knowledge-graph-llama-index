from llama_index import StorageContext
from llama_index.graph_stores import Neo4jGraphStore


# technical debt - modularize further so it can receive vector storage as well
def create_storage_context(neo4j_credentials, persistence_dir):
    """Create a storage context for Neo4j graph storage.

    Args:
        neo4j_credentials (dict): A dictionary containing Neo4j connection credentials.
        persistence_dir (str): The directory path for persistence storage.

    Returns:
        StorageContext: The initialized storage context.
    """
    graph_store = Neo4jGraphStore(**neo4j_credentials)
    return StorageContext.from_defaults(graph_store=graph_store, persist_dir=persistence_dir)
