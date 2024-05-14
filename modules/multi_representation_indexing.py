import uuid

from data_loading import load_documents_langchain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint

# from langchain_community.llms import HuggingFaceEndpoint
import os

# from langchain.storage import InMemoryByteStore
from langchain.storage import RedisStore
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import chromadb

load_dotenv()
repository_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Constants

chroma_collection_name = "summaries-rwcm"
redis_namespace = "parent-documents-rwcm"

# Initialize large language model
llm = HuggingFaceEndpoint(
    repo_id=repository_id,
    max_length=128,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),
)
# Initalize embeddings model
embeddings_model = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))

# Initialize chroma
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT")
)

# The storage layer for the parent documents
# TODO replace this with environment variables after manual tests
# TODO add this environment variable in github secret as well
# See if we have already store the parent documents
redis_store = RedisStore(
    redis_url="redis://localhost:6379",
    namespace=redis_namespace
)

# Get the documents
documents_directory = "../data/real_world_community_model_1st_half"
documents = load_documents_langchain(documents_directory)

chain = (
    {"document": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(
        "Summarize the following document:\n\n{document}"
    )
    | llm
    | StrOutputParser()
)

# Get or create Chroma collection
chroma_collection = chroma_client.get_or_create_collection(
    chroma_collection_name
)
# Define vector store
vectorstore = Chroma(
    client=chroma_client,
    collection_name=chroma_collection_name,
    embedding_function=embeddings_model
)
id_key = "doc_id"

# The retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=redis_store,
    id_key=id_key,
)

# See if we have already created summaries for this collection
id_key = ""
# If no, create them
if chroma_collection.count() == 0:
    print("Collection not found, creating embeddings...")
    summaries = chain.batch(documents, {"max_concurrency": 5})
    # previously
    # doc_ids = [str(uuid.uuid4()) for _ in documents]
    doc_ids = [f"{redis_namespace}-{uuid.uuid4()}" for _ in documents]
    # Documents linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
# If yes, load them
else:
    summaries = chroma_collection.get()
    print("Chroma collection is found...")

pattern = f"{redis_namespace}*"
# Keys that we are looking for to understand if the cache is empty
keys = list(redis_store.yield_keys(prefix=pattern))
if len(keys) == 0:
    # redis_store.mset(list(zip(doc_ids, documents)))
    retriever.docstore.mset(list(zip(doc_ids, documents)))
else:
    print("Cache in Redis is found...")

query = "What are domains of real world community model?"
sub_docs = vectorstore.similarity_search(query, k=1)
# print("subdocs: ", sub_docs[0])

retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
# retrieved_docs[0].page_content[0:500]
# print("retrieved docs: ", retrieved_docs)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
# Pass question to the qa_chain
result = qa_chain({"query": query})
print("Printing final answer", result["result"])
