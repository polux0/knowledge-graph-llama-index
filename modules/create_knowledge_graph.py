import os
from datetime import datetime, timezone

from data_loading import load_documents, load_documents_with_filnames_as_ids
from data_path_resolver import resolve_data_path, create_data_path
from dotenv import load_dotenv
# Elasticsearch realted
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from embedding_model_modular_setup import (
    get_embedding_model_based_on_model_name_id, initialize_embedding_model)
from environment_setup import load_environment_variables, setup_logging
from get_document_by_id import get_document_by_id
from large_language_model_setup import (get_llm_based_on_model_name_id,
                                        initialize_llm)
from llama_index.core import (KnowledgeGraphIndex, Settings, StorageContext,
                              load_index_from_storage)
from llama_index.graph_stores.nebula import NebulaGraphStore
from nebula_graph_client import NebulaGraphClient
from prompts import get_template_based_on_template_id
from querying import query_knowledge_graph
from service_context_setup import create_service_context
from visualize_graph import generate_network_graph

# load_dotenv()
load_dotenv()


print("NEBULA URL: ", os.getenv("NEBULA_URL"))
client = NebulaGraphClient(
    [(os.getenv("NEBULA_URL"), int(os.getenv("NEBULA_PORT")))],
    os.getenv("NEBULA_USERNAME"),
    os.getenv("NEBULA_PASSWORD"),
)

# Initialize the Elasticsearch client - technical debt -
es_client = ElasticsearchClient()

# Initialize Experiment

# Timestamp realted
# Get the current time in UTC, making it timezone-aware
current_time = datetime.now(timezone.utc)

experiment = ExperimentDocument()
# Format the current time as an ISO 8601 string, including milliseconds
experiment.created_at = current_time.isoformat(timespec="milliseconds")
# Load environment variables and setup logging
# Local development 
env_vars = load_environment_variables()
setup_logging()

# variables
model_name_id = "mixtral"
embedding_model_id = "cohere"
chunk_size = 256
max_triplets_per_chunk = 15
documents_directory = "../data/real_world_community_model_1st_half"

# Initialize LLM and Embedding model
llm = initialize_llm(model_name_id)
Settings.llm = llm
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)
embed_model = initialize_embedding_model(embedding_model_id=embedding_model_id)
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(
    embedding_model_id
)
Settings.embed_model = embed_model
# Load documents
print("Loading documents...")
documents = load_documents_with_filnames_as_ids(
    directory_path=documents_directory
)
# Setup the service context
specific_document = get_document_by_id(documents, "/home/equinox/Desktop/development/knowledge-graph-llama-index/src/modules/../data/real_world_community_model_1st_half/Aurvana System Overiew - 73 - 84-1-6.pdf_part_0")
if specific_document:
    print("Document found:", specific_document)
else:
    print("No document found with the specified ID.")

service_context = create_service_context(llm, chunk_size, embed_model, True)
experiment.chunk_size = chunk_size

database_name = ""
created_database_name = ""

try:
    database_name = client.create_space_if_not_exists(model_name_id,
                                                      embedding_model_id,
                                                      chunk_size,
                                                      max_triplets_per_chunk,
                                                      documents_directory
                                                      )

except Exception as e:
    print(f"Error while creating space: {e}")

try:
    # Use the database_name variable instead of the hardcoded string.
    persistence_directory = resolve_data_path(f"../persistence/{database_name}")
except Exception as e:
    print("An error occurred while resolving the data path:", e)
    # Also replace here when creating the data path
    create_data_path(f"../persistence/{database_name}")
    
# Have to modify this path:

os.environ["NEBULA_USER"] = os.getenv("NEBULA_USERNAME")
os.environ["NEBULA_PASSWORD"] = os.getenv("NEBULA_PASSWORD")
os.environ["NEBULA_ADDRESS"] = os.getenv("NEBULA_URL") + ":" + os.getenv("NEBULA_PORT")
# Necessary parameters to instantiate NebulaGraph
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]

graph_store = NebulaGraphStore(
    space_name=database_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

try:
    storage_context = StorageContext.from_defaults(persist_dir=resolve_data_path(f"../persistence/{database_name}"),
                                                   graph_store=graph_store)
    index = load_index_from_storage(
        storage_context=storage_context,
        service_context=service_context,
        max_triplets_per_chunk=max_triplets_per_chunk,
        include_embeddings=True,
    )

    experiment.max_triplets_per_chunk = max_triplets_per_chunk
    index_loaded = True
    print("Loading of index is finished...")
except:
    index_loaded = False
    print("Index not found, constructing one...")
if not index_loaded:
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=15,
        space_name=database_name,
        # edge_types=edge_types,
        # rel_prop_names=rel_prop_names,
        # tags=tags,
        # llm=llm,
        show_progress=True
    )
    index.storage_context.persist(persist_dir=resolve_data_path(f"../persistence/{database_name}"))
    generate_network_graph(index, database_name)
# technical debt


def generate_response_based_on_knowledge_graph_with_debt(query: str):

    template = get_template_based_on_template_id("default")
    experiment.question = query
    experiment.prompt_template = (template,)
    # technical debt - tree_summarize
    experiment.retrieval_strategy = "tree_summarize"
    experiment.source_agent = "KGAgent"
    response, source_nodes = query_knowledge_graph(index,
                                                   query,
                                                   template,
                                                   llm=llm)
    experiment.response = response

    # timestamp realted
    # Get the current time in UTC, making it timezone-aware
    current_time = datetime.now(timezone.utc)

    # Format the current time as an ISO 8601 string, including milliseconds
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")

    print(
        "response from knowledge graph ***********************************: ",
        response,
    )
    return response, experiment, source_nodes
