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
# Elasticsearch realted
from elasticsearch_service import ExperimentDocument, ElasticsearchClient

from large_language_model_setup import get_llm_based_on_model_name_id
from embedding_model_modular_setup import get_embedding_model_based_on_model_name_id
from prompts import get_template_based_on_template_id
from datetime import datetime, timezone


# Initialize the Elasticsearch client - technical debt - 
es_client = ElasticsearchClient(scheme='http', host='elasticsearch', port=9200)

# Initialize Experiment

# Timestamp realted
# Get the current time in UTC, making it timezone-aware
current_time = datetime.now(timezone.utc)

# Format the current time as an ISO 8601 string, including milliseconds
created_at = current_time.isoformat(timespec='milliseconds')
updated_at = created_at  # Initially, both timestamps will be the same

experiment = ExperimentDocument()
# Load environment variables and setup logging
load_dotenv()
env_vars = load_environment_variables()
setup_logging()

# variables
model_name_id = "default"
embedding_model_id = "default" 
chunk_size = 256
max_triplets_per_chunk = 15


# Initialize LLM and Embedding model
llm = initialize_llm(env_vars['HF_TOKEN'], model_name_id = model_name_id)
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)

embed_model = initialize_embedding_model(env_vars['HF_TOKEN'], embedding_model_id=embedding_model_id)
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(embedding_model_id)


# Load documents
documents = load_documents("../data/real_world_community_model")

# Setup the service context
service_context = create_service_context(llm, chunk_size, embed_model, True)
experiment.chunk_size = chunk_size

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
  password=os.getenv('NEO4J_PASSWORD'),
  url=os.getenv('NEO4J_URL'),
  database=os.getenv('NEO4J_DATABASE')
)

try:
  storage_context = StorageContext.from_defaults(persist_dir=persistence_directory)
  index = load_index_from_storage(storage_context=storage_context,
                                  service_context=service_context,
                                  max_triplets_per_chunk=max_triplets_per_chunk,
                                  include_embeddings=True)
  
  experiment.max_triplets_per_chunk = max_triplets_per_chunk
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
  
# Knowledge graph specific retriever: 

# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

# graph_rag_retriever = KnowledgeGraphRAGRetriever(
#     storage_context=storage_context,
#     verbose=True,
#     llm = llm
# )

# query_engine = RetrieverQueryEngine.from_args(
#     graph_rag_retriever,
# )

# Technical debt 1 - modularize further
# Print the index
# print("index: ", index)
def generate_response_based_on_knowledge_graph(query: str):
  # knowledgeGraphRagRetriever = query_engine.query(query)
  # print("response form knowledge graph, graphRagRetriever ******************************************************************: ", knowledgeGraphRagRetriever)
  experiment.question = query
  experiment.prompt_template = get_template_based_on_template_id("simon"),
  # technical debt - tree_summarize
  experiment.retrieval_strategy = "tree_summarize"
  response = query_knowledge_graph(index, query, template_id="simon")
  experiment.response = response

  # timestamp realted
  # Get the current time in UTC, making it timezone-aware
  current_time = datetime.now(timezone.utc)

  # Format the current time as an ISO 8601 string, including milliseconds
  experiment.updated_at = current_time.isoformat(timespec='milliseconds')

  es_client.save_experiment(experiment_document=experiment)

  print("response from knowledge graph ******************************************************************: ", response) 
  return response
# Query the knowledge graph
# query = "What are the domains of the Real World Community Model?"
# query = "What do systems need to flourish?"
# query = "Would you tell me more about societal information system"
# query = "What would be the way to construct better societies?"
# query = "Can you tell me about the key domains of Real World Community Model?"
# response = generate_response_based_on_knowledge_graph(index, query, template_id="default")
# print(response)