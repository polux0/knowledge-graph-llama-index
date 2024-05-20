import logging
import sys

from custom_ingestion_pipeline import CustomIngestionPipeline
from data_loading import load_documents
from embedding_model_modular_setup import initialize_embedding_model
from environment_setup import load_environment_variables
from large_language_model_setup import initialize_llm
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from get_database_name_based_on_parameters import (
    load_child_vector_configuration,
    load_parent_vector_configuration,
)

# from llama_index.core.ingestion import IngestionPipeline
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.ingestion import IngestionCache
import os
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
import chromadb

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from datetime import datetime, timezone

# Elasticsearch realted
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from embedding_model_modular_setup import get_embedding_model_based_on_model_name_id
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

documents_directory = "../data/documentation"

documents = load_documents(documents_directory)

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

# parent_chroma_collection_name = load_parent_vector_configuration(
#     embedding_model_id,
#     parent_chunk_size,
#     parent_chunk_overlap,
#     documents_directory
# )

child_chroma_collection_name = load_child_vector_configuration(
    embedding_model_id,
    child_chunk_sizes,
    child_chunk_sizes_overlap,
    documents_directory,
)

print("Child chroma collection name: ", child_chroma_collection_name)

# chroma_collection_parent = remote_db.get_or_create_collection(
#     parent_chroma_collection_name
# )

chroma_collection_child = remote_db.get_or_create_collection(
    child_chroma_collection_name
)

# print(f"Are there embeddings inside collection {chroma_collection_parent.name} ?",
#       f"count: {chroma_collection_parent.count()}")

print(
    f"Are there embeddings inside collection {chroma_collection_child.name} ?",
    f"count: {chroma_collection_child.count()}",
)

# vector_store_parent = ChromaVectorStore(
#     chroma_collection=chroma_collection_parent
# )
vector_store_child = ChromaVectorStore(
    chroma_collection=chroma_collection_child, ssl=False
)

# Storage context parent
# storage_context_parent = StorageContext.from_defaults(
#     vector_store=vector_store_parent
# )
# Storage context
storage_context_child = StorageContext.from_defaults(vector_store=vector_store_child)

# Ingestion for parent documents
# ingest_cache_parent = IngestionCache(
#     cache=RedisCache.from_host_and_port(host=os.getenv("REDIS_URL"),
#                                         port=os.getenv("REDIS_PORT")),
#     collection=parent_chroma_collection_name,
# )

# necessary to create a collection for the first time
# if chroma_collection_parent.count() == 0:

#     pipeline = IngestionPipeline(
#         transformations=[
#             SentenceSplitter(chunk_size=parent_chunk_size,
#                              chunk_overlap=parent_chunk_overlap),
#             embed_model,
#             ],
#         vector_store=vector_store_parent,
#         cache=ingest_cache_parent,
#         # docstore=SimpleDocumentStore(),
#     )
#     pipeline.run(documents=documents)

#     base_index = VectorStoreIndex.from_vector_store(
#         vector_store=vector_store_parent,
#         storage_context=storage_context_parent,
#         embed_model=embed_model,
#     )
# else:
#     # after collection was sucessfully created

#     base_index = VectorStoreIndex.from_vector_store(
#         vector_store_parent,
#         storage_context=storage_context_parent,
#         embed_model=embed_model
#     )

# base_retriever = base_index.as_retriever(similarity_top_k=2, llm=llm)


# query_engine_base = RetrieverQueryEngine.from_args(base_retriever, llm=llm)

# response = query_engine_base.query(
#     "Can you tell me about the key domains of Real World Community Model"
# )

# print("testing")
# print("Base retrieval, response: \n")
# print(response)


# Part II Chuck References: Smaller Child Chunks Reffering to Bigger Parent Chunk

sub_chunk_sizes = [128, 256, 512]

# technical debt - create service context for this
sub_node_parsers = [
    SentenceSplitter(chunk_size=c, chunk_overlap=c / 2) for c in sub_chunk_sizes
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
    cache=RedisCache.from_host_and_port(
        host=os.getenv("REDIS_URL"), port=os.getenv("REDIS_PORT")
    ),
    collection=child_chroma_collection_name,
)

# necessary to create a collection for the first time

# if chroma_collection_child.count() == 0:
print("Ingestion pipeline has started...")
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

# else:

# after collection was sucessfully created
vector_index_chunk = VectorStoreIndex.from_vector_store(
    vector_store_child,
    storage_context=storage_context_child,
    embed_model=embed_model,
    llm=llm,
)

vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=3, llm=llm)

retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)
experiment.retrieval_strategy = "RecursiveRetriever - Parent Child"

# nodes = retriever_chunk.retrieve(
#     "Can you tell me about the key domains of Real World Community Model"
# )
# print("Parent retrievals, length: ", len(nodes))

# print("Displaying source node with parent retrieval")
# for node in nodes:
#     display_source_node(node, source_length=2000)

query_engine_chunk = RetrieverQueryEngine.from_args(retriever_chunk, llm=llm)


def generate_response_based_on_vector_embeddings_with_debt(question: str):

    experiment.question = question
    prompt_template = get_template_based_on_template_id("default")
    experiment.prompt_template = (get_template_based_on_template_id("default"),)
    print(
        "With parent-child retriever*******************************************************************\n\n: "
    )
    print("Template received, vector embeddings:", prompt_template)
    message_template = format_message(question, prompt_template)
    print("!!!!!!!!!!!!!!!!Final question, vetor embeddings:", message_template)
    response = query_engine_chunk.query(message_template)
    # logging.info(f"Logging the response nodes from a vector database: {response.source_nodes}")
    experiment.response = str(response)
    experiment.source_agent = "VDBAgent"

    current_time = datetime.now(timezone.utc)

    # Format the current time as an ISO 8601 string, including milliseconds
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")

    print("Final response", str(response))
    return response, experiment, response.source_nodes
