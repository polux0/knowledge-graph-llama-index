import logging
from dotenv import load_dotenv
from environment_setup import (load_environment_variables, setup_logging)
from large_language_model_setup import initialize_llm
from embedding_model_setup import initialize_embedding_model
from data_loading import load_documents
from service_context_setup import create_service_context
from storage_context_setup import create_storage_context
from knowledge_graph_index_managment import manage_knowledge_graph_index
from querying import query_knowledge_graph

# Load environment variables and setup logging
load_dotenv()
env_vars = load_environment_variables()
setup_logging()

# Initialize LLM and Embedding model
llm = initialize_llm(env_vars['HF_TOKEN'])
embed_model = initialize_embedding_model(env_vars['HF_TOKEN'])

# Load documents
documents = load_documents("./data/real_world_community_model")

# Setup the service context
service_context = create_service_context(llm, embed_model)

# Setup the storage context
neo4j_credentials = {
    'username': env_vars['NEO4J_USERNAME'],
    'password': env_vars['NEO4J_PASSWORD'],
    'url': env_vars['NEO4J_URL'],
    'database': env_vars['NEO4J_DATABASE']
}
storage_context = create_storage_context(neo4j_credentials, './persistence/real_world_community_model_15_triplets_per_chunk_neo4j')

# Manage the Knowledge Graph Index
try:
    index = manage_knowledge_graph_index(documents, service_context, storage_context, index_loaded=True)
    print('Loading of index is finished...')
except Exception as e:
    logging.error(f"Error loading index: {e}")
    index = manage_knowledge_graph_index(documents, service_context, storage_context, index_loaded=False)
    print('Index not found, constructing one...')

# Query the knowledge graph
query = "What are the domains of the Real World Community Model?"
response = query_knowledge_graph(index, query, template_id="default")
print(response)