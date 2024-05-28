import logging
import sys

from custom_ingestion_pipeline import CustomIngestionPipeline
from data_loading import load_documents
from embedding_model_modular_setup import initialize_embedding_model
from environment_setup import load_environment_variables
from large_language_model_setup import initialize_llm
from llama_index.core import (Document, Settings, StorageContext,
                              VectorStoreIndex)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from get_database_name_based_on_parameters import (
    load_child_vector_configuration,
    load_parent_vector_configuration
)
# from llama_index.core.ingestion import IngestionPipeline
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.ingestion import IngestionCache
import os
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
import chromadb

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from datetime import datetime, timezone

# Elasticsearch realted
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from embedding_model_modular_setup import \
    get_embedding_model_based_on_model_name_id
from format_message_with_prompt import format_message
from large_language_model_setup import get_llm_based_on_model_name_id
from prompts import get_template_based_on_template_id

# Elasticsearch related
es_client = ElasticsearchClient()

# Initialize Experiment

# Get the current time in UTC, making it timezone-aware#
current_time = datetime.now(timezone.utc)

experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec="milliseconds")

# Variables parent
model_name_id = "gpt-3.5-turbo"
embedding_model_id = "openai-text-embedding-3-large"
parent_chunk_size = 2048
parent_chunk_overlap = 512

# Variables child
child_chunk_sizes = [128, 256, 512]
child_chunk_sizes_overlap = [64, 128, 256]

# Load the documents, modular function previously used for knowledge graph construction

# Smallest sample, for testing purposes
# documents_directory = "../data/real_world_community_model_1st_half"
# Complete documentation

folders = ['decision-system', 'habitat-system', 'lifestyle-system', 'material-system', 'project-execution', 'project-plan',
           'social-system', 'system-overview']
documents = []
for folder in folders:
    documents_directory = f"../data/documentation_optimal/{folder}"
    documents.extend(load_documents(documents_directory))
print("documents length: ", len(documents))

# load the documents, example from llama documentation

doc_text = "\n\n".join([d.get_content() for d in documents])
docs = [Document(text=doc_text)]

node_parser = SentenceSplitter(chunk_size=parent_chunk_size)
experiment.chunk_size = parent_chunk_size

base_nodes = node_parser.get_nodes_from_documents(docs)
# set node ids to be a constant
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"

env_vars = load_environment_variables()

# embedings
embed_model = initialize_embedding_model(embedding_model_id=embedding_model_id)
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(
    embedding_model_id
)
Settings.embed_model = embed_model

# large language model
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)
print("experiment.llm_used", experiment.llm_used)
llm = initialize_llm(model_name_id)
Settings.llm = llm

print("Trying to connect to chromaDB client...")
remote_db = chromadb.HttpClient(
    host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT")
)
print("Sucessfully connected to chromaDB client...")
print("All collections in Chroma: ", remote_db.list_collections())

# Dynamic configuration

# child_chroma_collection_name = load_child_vector_configuration(
#     embedding_model_id,
#     child_chunk_sizes,
#     child_chunk_sizes_overlap,
#     documents_directory
# )


child_chroma_collection_name = "complete-documentation-parent-child"

# remote_db.delete_collection(name=child_chroma_collection_name)

print("Child chroma collection name: ", child_chroma_collection_name)

chroma_collection_child = remote_db.get_or_create_collection(
    child_chroma_collection_name
)

print(f"Are there embeddings inside collection {chroma_collection_child.name} ?",
      f"count: {chroma_collection_child.count()}")

vector_store_child = ChromaVectorStore(
    chroma_collection=chroma_collection_child,
    ssl=False
)

# Storage context
storage_context_child = StorageContext.from_defaults(
    vector_store=vector_store_child
)

sub_chunk_sizes = [128, 256, 512]

# technical debt - create service context for this
sub_node_parsers = [
    SentenceSplitter(chunk_size=c, chunk_overlap=c / 2)
    for c in sub_chunk_sizes
]

all_nodes = []

for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_indices = [
            IndexNode.from_text_node(sub_node, base_node.node_id)
            for sub_node in sub_nodes
        ]
        all_nodes.extend(sub_indices)

    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)

print("Started making dictionaries...")
# Maybe we need to store this into database
all_nodes_dict = {n.node_id: n for n in all_nodes}
print("Finished with making dictionaries...")

# Ingest cache #2

ingest_cache_child = IngestionCache(
    cache=RedisCache.from_host_and_port(host=os.getenv("REDIS_URL"),
                                        port=os.getenv("REDIS_PORT")),
    collection=child_chroma_collection_name,
)

# necessary to create a collection for the first time

if chroma_collection_child.count() == 0:
    print("!Ingestion pipeline has started...")
    pipeline2 = CustomIngestionPipeline(
        transformations=[
            embed_model,
            ],
        vector_store=vector_store_child,
        cache=ingest_cache_child,
        # docstore=SimpleDocumentStore(),
    )
    pipeline2.run(
        documents=all_nodes,
        show_progress=True,
    )
print("Ingestion pipeline has finished...")

vector_index_chunk = VectorStoreIndex.from_vector_store(
    vector_store=vector_store_child,
    storage_context=storage_context_child,
    embed_model=embed_model,
)

vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=3,
                                                         llm=llm)

retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)
experiment.retrieval_strategy = "RecursiveRetriever - Parent Child"
question = "Can you tell me about the key domains of Real World Community Model"
# question = "Can you tell me more about the social system domain"
nodes = retriever_chunk.retrieve(
    question,
)
print("Displaying source node with parent retrieval: \n")
for node in nodes:
    print(node)

prompt_template = get_template_based_on_template_id("default")
message_template = format_message(question, prompt_template)
query_engine_chunk = RetrieverQueryEngine.from_args(retriever_chunk, llm=llm)
response = query_engine_chunk.query(message_template)
print("Response: ", response)


def generate_response_based_on_vector_embeddings_with_debt(question: str):

    experiment.question = question
    prompt_template = get_template_based_on_template_id("default")
    experiment.prompt_template = (get_template_based_on_template_id("default"),)
    print(
        "With parent-child retriever*******************************************************************\n\n: "
    )
    message_template = format_message(question, prompt_template)
    response = query_engine_chunk.query(message_template)
    experiment.response = str(response)
    experiment.source_agent = "VDBAgent"

    current_time = datetime.now(timezone.utc)
    # Format the current time as an ISO 8601 string, including milliseconds
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")

    print("Final response", str(response))
    return response, experiment, response.source_nodes
