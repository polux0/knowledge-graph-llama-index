from data_loading import load_documents
from embedding_model_modular_setup import get_embedding_model_based_on_model_name_id, initialize_embedding_model
from format_message_with_prompt import format_message
from large_language_model_setup import get_llm_based_on_model_name_id, initialize_llm
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.packs.raptor import RaptorPack
# Configuring summarization
from llama_index.packs.raptor.base import SummaryModule
# Raptor Retriever
from llama_index.packs.raptor import RaptorRetriever
# Retrieval
from llama_index.core.query_engine import RetrieverQueryEngine
# System
import os
from dotenv import load_dotenv
# Elasticsearch
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from datetime import datetime, timezone
# Prompts
from prompts import get_template_based_on_template_id
load_dotenv()

# constants
embedding_model_id = "openai-text-embedding-3-large"
large_language_model_id = "gpt-3.5-turbo"
chunk_size = 2000
chunk_overlap = 1000
retriever_mode = "tree_traversal"

# Elasticsearch related
current_time = datetime.now(timezone.utc)
elasticsearch_client = ElasticsearchClient()
experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec="milliseconds")

# ChromaDB
remote_db = chromadb.HttpClient(
    host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT")
)

documents_directory = "../data/real_world_community_model_1st_half"
# documents_directory = "../data/real_world_community_model_1st_half"
documents = load_documents(documents_directory)

# Logging variables
experiment.chunk_size = chunk_size

# TODO delete after testing
remote_db.delete_collection(name=str("raptor"))
collection = remote_db.get_or_create_collection("raptor")
vector_store = ChromaVectorStore(chroma_collection=collection)

embed_model = initialize_embedding_model(embedding_model_id=embedding_model_id)
llm = initialize_llm(model_name_id=large_language_model_id)

# Logging variables
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(
    embedding_model_id
)
experiment.llm_used = get_llm_based_on_model_name_id(large_language_model_id)


# Configuring summarization
summary_prompt = (
    "As a professional summarizer, create a concise and comprehensive summary"
    "of the provided text, be it an article, post, conversation, or passage"
    "with as much detail as possible."
)

summary_module = SummaryModule(
    llm=llm, summary_prompt=summary_prompt, num_workers=2
)

if collection.count() == 0:
    print(f"Collection {collection} not found, creating and generating embeddings... ")
    raptor_pack = RaptorPack(
        documents,
        embed_model=embed_model,
        summary_module=summary_module,
        llm=llm,
        vector_store=vector_store,
        similarity_top_k=5,
        mode=retriever_mode,  # Possibilities are compact and tree traveral
        transformations=[
            SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        ]
    )
# Retrieval
retriever = RaptorRetriever(
    [],
    embed_model=embed_model,  # used for embedding clusters
    llm=llm,  # used for generating summaries
    vector_store=vector_store,  # used for storage
    similarity_top_k=5,  # top k for each layer, or overall top-k for collapsed
    mode=retriever_mode,  # sets default mode
)
# Logging variables
experiment.retrieval_strategy = f"RaptorRetriever, mode: {retriever_mode}"

# query = "What are the domains of real world community model?"
# results = retriever.retrieve(query, retriever_mode)
# for result in results:
#     print(f"Document: {result}")

query_engine = RetrieverQueryEngine.from_args(
    retriever,
    llm=llm
)
# response = query_engine.query(query)
# print(str(response))


def generate_response_based_on_raptor_indexing_with_debt(question: str):

    experiment.question = question
    prompt_template = get_template_based_on_template_id("default")
    experiment.prompt_template = (
        get_template_based_on_template_id("default"),
    )
    message_template = format_message(question, prompt_template)
    response = query_engine.query(message_template)
    experiment.response = str(response)
    experiment.source_agent = "RaptorAgent"

    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    # Source nodes
    source_nodes = retriever.retrieve(question, retriever_mode)
    response = query_engine.query(question)

    return response, experiment, source_nodes
