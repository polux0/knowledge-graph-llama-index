from data_loading import load_documents
from embedding_model_modular_setup import initialize_embedding_model
from large_language_model_setup import initialize_llm
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

load_dotenv()

# constants

# previously
# embedding_model_id = "openai-text-embedding-3-large"
# chunk_size = 2000
# chunk_overlap = 1000
embedding_model_id = "openai-text-embedding-3-small"
large_language_model_id = "gpt-3.5-turbo"
chunk_size = 1024
chunk_overlap = 512

remote_db = chromadb.HttpClient(
    host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT")
)

documents_directory = "../data/real_world_community_model_1st_half"
documents = load_documents(documents_directory)

# TODO delete after testing
# remote_db.delete_collection(name=str("raptor"))
collection = remote_db.get_or_create_collection("raptor")
vector_store = ChromaVectorStore(chroma_collection=collection)

embed_model = initialize_embedding_model(embedding_model_id=embedding_model_id)
llm = initialize_llm(model_name_id=large_language_model_id)

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
    raptor_pack = RaptorPack(
        documents,
        embed_model=embed_model,
        summary_module=summary_module,
        llm=llm,
        vector_store=vector_store,
        similarity_top_k=5,
        mode="tree_traversal",  # Possibilities are compact and tree traveral
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
    mode="tree_traversal",  # sets default mode
)
query = "What are the domains of real world community model?"
results = retriever.retrieve(query, "tree_traversal")
for result in results:
    print(f"Document: {result}")

query_engine = RetrieverQueryEngine.from_args(
    retriever,
    llm=llm
)
response = query_engine.query(query)
print(str(response))
