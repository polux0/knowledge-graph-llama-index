import os
from datetime import datetime, timezone

from data_loading import load_documents
from data_path_resolver import resolve_data_path
from dotenv import load_dotenv
# Elasticsearch realted
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from embedding_model_modular_setup import (
    get_embedding_model_based_on_model_name_id, initialize_embedding_model)
from environment_setup import load_environment_variables, setup_logging
from knowledge_graph_index_managment import manage_knowledge_graph_index
from large_language_model_setup import (get_llm_based_on_model_name_id,
                                        initialize_llm)
from llama_index.core import (KnowledgeGraphIndex, Settings, StorageContext,
                              load_index_from_storage)
from llama_index.graph_stores.nebula import NebulaGraphStore
# Related to technical debt #1
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from nebula_graph_client import NebulaGraphClient
from prompts import get_template_based_on_template_id
# Related to technical debt #1
from querying import query_knowledge_graph
from service_context_setup import create_service_context
from storage_context_setup import create_storage_context

client = NebulaGraphClient(
    [(os.getenv('NEBULA_URL'), int(os.getenv('NEBULA_PORT')))],
    os.getenv('NEBULA_USERNAME'),
    os.getenv('NEBULA_PASSWORD')
)
try:
    client.create_space('test')  # Create space with default parameters
    client.use_space('test')
    client.create_schema()
    print("Schema created successfully")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    client.use_space('test')
    print("Using space 'test'.")
    result = client.describe_space('test')
    if result:
        print("Schema for 'test':", result)
    else:
        print("Failed to retrieve schema for 'test'")
except Exception as e:
    print(f"Error retrieving schema: {e}")
finally:
    client.close()
# Initialize the Elasticsearch client - technical debt - 
es_client = ElasticsearchClient()

# Initialize Experiment

# Timestamp realted
# Get the current time in UTC, making it timezone-aware
current_time = datetime.now(timezone.utc)

experiment = ExperimentDocument()
# Format the current time as an ISO 8601 string, including milliseconds
experiment.created_at = current_time.isoformat(timespec='milliseconds')
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
llm = initialize_llm(model_name_id)
Settings.llm = llm
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)

embed_model = initialize_embedding_model(env_vars['HF_TOKEN'], embedding_model_id=embedding_model_id)
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(embedding_model_id)
Settings.embed_model = embed_model
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

try:
    persistence_directory = resolve_data_path('../persistence/real_world_community_model_15_triplets_per_chunk_nebula')
except Exception as e:
    print("An error occurred while resolving the data path:", e)
    persistence_directory = ""
    # You can handle the error further as needed

# Move this block inside or outside of the try/except block, depending on your logic.
# For illustration, I'm placing it outside the try/except block.
# Technical debt 1 - modularize further
# Will be removed soon
# graph_store = Neo4jGraphStore(
#   username=os.getenv('NEO4J_USERNAME'),
#   password=os.getenv('NEO4J_PASSWORD'),
#   url=os.getenv('NEO4J_URL'),
#   database=os.getenv('NEO4J_DATABASE')
# )

os.environ["NEBULA_USER"] = "root"
os.environ[
    "NEBULA_PASSWORD"
] = "password"  # replace with your password, by default it is "nebula"
os.environ["NEBULA_ADDRESS"] = os.getenv('NEBULA_URL') + ':' + os.getenv('NEBULA_PORT')
# Necessary parameters to instantiate NebulaGraph
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]
# We'll start using NebulaGraph
graph_store = NebulaGraphStore(
    space_name="test",
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
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
  index = KnowledgeGraphIndex.from_documents(
                                            documents,
                                            storage_context=storage_context,
                                            max_triplets_per_chunk=15,
                                            space_name="test",
                                            edge_types=edge_types,
                                            rel_prop_names=rel_prop_names,
                                            tags=tags,
                                            llm=llm
                                            )
  
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

  template = get_template_based_on_template_id("simon")
  print("template, that is making an issue: ", template)
  experiment.question = query
  experiment.prompt_template = template,
  # technical debt - tree_summarize
  experiment.retrieval_strategy = "tree_summarize"
  experiment.source_agent = "KGAgent"
  response = query_knowledge_graph(index, query, template)
  experiment.response = response

  # timestamp realted
  # Get the current time in UTC, making it timezone-aware
  current_time = datetime.now(timezone.utc)

  # Format the current time as an ISO 8601 string, including milliseconds
  experiment.updated_at = current_time.isoformat(timespec='milliseconds')

  es_client.save_experiment(experiment_document=experiment)

  print("response from knowledge graph ******************************************************************: ", response) 
  return response


# technical debt

def generate_response_based_on_knowledge_graph_with_debt(query: str):

  template = get_template_based_on_template_id("default")
  experiment.question = query
  experiment.prompt_template = template,
  # technical debt - tree_summarize
  experiment.retrieval_strategy = "tree_summarize"
  experiment.source_agent = "KGAgent"
  response, source_nodes = query_knowledge_graph(index, query, template, llm=llm)
  experiment.response = response

  # timestamp realted
  # Get the current time in UTC, making it timezone-aware
  current_time = datetime.now(timezone.utc)

  # Format the current time as an ISO 8601 string, including milliseconds
  experiment.updated_at = current_time.isoformat(timespec='milliseconds')

  print("response from knowledge graph ******************************************************************: ", response) 
  return response, experiment, source_nodes