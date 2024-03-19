import logging
import os
from dotenv import load_dotenv
from llama_index.core import KnowledgeGraphIndex, StorageContext, load_index_from_storage
from environment_setup import (load_environment_variables, setup_logging)
from large_language_model_setup import initialize_llm
from embedding_model_modular_setup import initialize_embedding_model
from data_loading import load_documents
from service_context_setup import create_service_context
# Related to technical debt #1
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from storage_context_setup import create_storage_context
from knowledge_graph_index_managment import manage_knowledge_graph_index
# Related to technical debt #1
from querying import query_knowledge_graph
from data_path_resolver import resolve_data_path


# Load environment variables and setup logging
load_dotenv()
env_vars = load_environment_variables()
setup_logging()

# Initialize LLM and Embedding model
llm = initialize_llm(env_vars['HF_TOKEN'], model_name_id="default")
embed_model = initialize_embedding_model(env_vars['HF_TOKEN'], embedding_model_id="default")

# Load documents
documents = load_documents("../data/real_world_community_model")

# Setup the service context
service_context = create_service_context(llm, 256, embed_model, True)

# Setup the storage context
neo4j_credentials = {
    'username': env_vars['NEO4J_USERNAME'],
    'password': env_vars['NEO4J_PASSWORD'],
    'url': env_vars['NEO4J_URL'],
    'database': env_vars['NEO4J_DATABASE']
}
persistence_directory = resolve_data_path('../persistence/real_world_community_model_15_triplets_per_chunk_neo4j')

# Technical debt 1 - modularize further
graph_store = Neo4jGraphStore(
  username=os.getenv('NEO4J_USERNAME'),
  password=os.getenv('NEO4J_PASSOWRD'),
  url=os.getenv('NEO4J_URL'),
  database=os.getenv('NEO4J_DATABASE')
)

try:
  storage_context = StorageContext.from_defaults(persist_dir=persistence_directory)
  index = load_index_from_storage(storage_context=storage_context,
                                  service_context=service_context,
                                  max_triplets_per_chunk=15,
                                  include_embeddings=True)
  index_loaded = True
  print('Loading of index is finished...')
except:
  index_loaded = False
  print('Index not found, constructing one...')
if not index_loaded:
  # construct the Knowledge Graph Index
  storage_context = StorageContext.from_defaults(graph_store=graph_store)
  index = KnowledgeGraphIndex.from_documents(documents=documents,
                                             max_triplets_per_chunk=15,
                                             service_context=service_context,
                                              storage_context=storage_context,
                                             include_embeddings=False)


# Technical debt 1 - modularize further
# Print the index
# print("index: ", index)
def generate_response_based_on_knowledge_graph(query: str):
  return query_knowledge_graph(index, query, template_id="default")
# Query the knowledge graph
# query = "What are the domains of the Real World Community Model?"
# query = "What do systems need to flourish?"
# query = "Would you tell me more about societal information system"
# query = "What would be the way to construct better societies?"
# query = "Can you tell me about the key domains of Real World Community Model?"
# response = generate_response_based_on_knowledge_graph(index, query, template_id="default")
# print(response)