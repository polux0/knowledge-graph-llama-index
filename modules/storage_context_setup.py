from llama_index.core import StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from persistence_directory_validity import check_files_in_directory
from utils.environment_setup import load_environment_variables

env_vars = load_environment_variables()

NEO4J_USERNAME = env_vars["NEO4J_USERNAME"]
NEO4J_PASSOWRD = env_vars["NEO4J_PASSOWRD"]
NEO4J_URL = env_vars["NEO4J_URL"]
NEO4J_DATABASE = env_vars["NEO4J_DATABASE"]


# technical debt - modularize further so it can receive vector storage as well
def create_storage_context(neo4j_credentials, env_vars, persistence_dir):
    """Create a storage context for Neo4j graph storage.

    Args:
        neo4j_credentials (dict): A dictionary containing Neo4j connection credentials.
        persistence_dir (str): The directory path for persistence storage.

    Returns:
        StorageContext: The initialized storage context.
    """
    # try:
    required_files = [
        "default__vector_store.json",
        "docstore.json",
        "graph_store.json",
        "image__vector_store.json",
        "index_store.json",
    ]  # Example file names
    if check_files_in_directory(persistence_dir, required_files):
        print("Persistence directory: ", persistence_dir)
        storage_context = StorageContext.from_defaults(persistence_dir)
        print("Created storage context from persistence directory...")
    else:
        NEO4J_USERNAME = env_vars["NEO4J_USERNAME"]
        NEO4J_PASSOWRD = env_vars["NEO4J_PASSOWRD"]
        NEO4J_URL = env_vars["NEO4J_URL"]
        NEO4J_DATABASE = env_vars["NEO4J_DATABASE"]
        graph_store = Neo4jGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSOWRD,
            url=NEO4J_URL,
            database=NEO4J_DATABASE,
        )
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        print("Created new storage context...")
    return storage_context
