import logging
import redis
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
from get_database_name_based_on_parameters import load_child_vector_configuration, load_parent_vector_configuration
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.ingestion import IngestionCache
import os
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
import chromadb
from datetime import datetime, timezone
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from embedding_model_modular_setup import get_embedding_model_based_on_model_name_id
from format_message_with_prompt import format_message
from large_language_model_setup import get_llm_based_on_model_name_id
from prompts import get_template_based_on_template_id

# Initialize logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Elasticsearch client
es_client = ElasticsearchClient()

# Get the current time in UTC, making it timezone-aware
current_time = datetime.now(timezone.utc)
experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec="milliseconds")

# Variables parent
model_name_id = "default"
embedding_model_id = "openai-text-embedding-3-large"
parent_chunk_size = 2048
parent_chunk_overlap = 512
# Production
child_chroma_collection_name = "complete-documentation-parent-child"
# Local testing
# child_chroma_collection_name = "complete-documentation-parent-child1"

# Variables child
child_chunk_sizes = [128, 256, 512]
child_chunk_sizes_overlap = [64, 128, 256]

# Load the documents
# Production
folders = ['decision-system', 'habitat-system', 'lifestyle-system', 'material-system', 'project-execution', 'project-plan', 'social-system', 'system-overview']
# Local testing
# folders = ['test1']
documents = []
for folder in folders:
    documents_directory = f"../data/documentation_optimal/{folder}"
    documents.extend(load_documents(documents_directory))
logger.info(f"Documents loaded: {len(documents)}")

doc_text = "\n\n".join([d.get_content() for d in documents])
docs = [Document(text=doc_text)]

node_parser = SentenceSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
experiment.chunk_size = parent_chunk_size

base_nodes = node_parser.get_nodes_from_documents(docs)
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"

env_vars = load_environment_variables()

# Initialize embedding model
embed_model = initialize_embedding_model(embedding_model_id=embedding_model_id)
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(embedding_model_id)
Settings.embed_model = embed_model

# Initialize large language model
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)
logger.info(f"Using LLM: {experiment.llm_used}")
llm = initialize_llm(model_name_id)
Settings.llm = llm

# Connect to ChromaDB
logger.info("Connecting to ChromaDB client...")
remote_db = chromadb.HttpClient(host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT"))
logger.info("Connected to ChromaDB client.")
logger.info(f"All collections in Chroma: {remote_db.list_collections()}")

logger.info(f"Child chroma collection name: {child_chroma_collection_name}")

chroma_collection_child = remote_db.get_or_create_collection(child_chroma_collection_name)
logger.info(f"Embeddings in collection {chroma_collection_child.name}: count: {chroma_collection_child.count()}")

vector_store_child = ChromaVectorStore(chroma_collection=chroma_collection_child, ssl=False)
storage_context_child = StorageContext.from_defaults(vector_store=vector_store_child)

sub_chunk_sizes = [128, 256, 512]
sub_node_parsers = [SentenceSplitter(chunk_size=c, chunk_overlap=c // 2) for c in sub_chunk_sizes]

all_nodes = []
for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_indices = [IndexNode.from_text_node(sub_node, base_node.node_id) for sub_node in sub_nodes]
        all_nodes.extend(sub_indices)
    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)

logger.info("Started making dictionaries...")
all_nodes_dict = {n.node_id: n for n in all_nodes}
logger.info("Finished with making dictionaries...")

# Ingest cache #2
redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
        )
ingest_cache_child = IngestionCache(cache=RedisCache.from_redis_client(redis_client),
                                    collection=child_chroma_collection_name
                                    )

# Initial ingestion if necessary
if chroma_collection_child.count() == 0:
    logger.info("Ingestion pipeline has started...")
    pipeline2 = CustomIngestionPipeline(transformations=[embed_model], vector_store=vector_store_child, cache=ingest_cache_child)
    pipeline2.run(documents=all_nodes, show_progress=True)
    logger.info("Ingestion pipeline has finished...")

vector_index_chunk = VectorStoreIndex.from_vector_store(vector_store=vector_store_child, storage_context=storage_context_child, embed_model=embed_model)
vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=3, llm=llm)

retriever_chunk = RecursiveRetriever("vector", retriever_dict={"vector": vector_retriever_chunk}, node_dict=all_nodes_dict, verbose=True)
experiment.retrieval_strategy = "RecursiveRetriever - Parent Child"

question = "Can you tell me about the key domains of Real World Community Model?"
nodes = retriever_chunk.retrieve(question)

prompt_template = get_template_based_on_template_id("default")
message_template = format_message(question, prompt_template)
query_engine_chunk = RetrieverQueryEngine.from_args(retriever_chunk, llm=llm)
response = query_engine_chunk.query(message_template)
logger.info(f"Response: {response}")


def stringify_and_combine_nodes(nodes) -> str:
    combined_output = "Nodes retrieved: \n\n"
    combined_output += "\n".join([repr(node) for node in nodes])
    return combined_output


def generate_response_based_on_vector_embeddings_with_debt(question: str):
    experiment.question = question
    prompt_template = get_template_based_on_template_id("default")
    experiment.prompt_template = prompt_template
    logger.info("With parent-child retriever:")
    message_template = format_message(question, prompt_template)
    response = query_engine_chunk.query(message_template)
    experiment.response = str(response)
    experiment.source_agent = "VDBAgent"
    experiment.retrieved_nodes = stringify_and_combine_nodes(nodes)
    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    logger.info(f"Final response: {response}")
    return response, experiment, response.source_nodes

