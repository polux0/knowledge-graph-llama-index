import uuid

from data_loading import load_documents_langchain
from embedding_model_modular_setup import initialize_embedding_model
from format_message_with_prompt import format_message
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint

# from langchain_community.llms import HuggingFaceEndpoint
import os
from langchain_community.storage import RedisStore
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import chromadb
# Elasticsearch
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from datetime import datetime, timezone
# Prompts
from large_language_model_setup import get_llm_based_on_model_name_id
from prompts import get_template_based_on_template_id

load_dotenv()
repository_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Constants

chroma_collection_name = "summaries-rwcm"
redis_namespace = "parent-documents-rwcm"
chunk_size = 1024

# Elasticsearch related
current_time = datetime.now(timezone.utc)
elasticsearch_client = ElasticsearchClient()
experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec="milliseconds")

# Initialize large language model
llm = HuggingFaceEndpoint(
    repo_id=repository_id,
    # max_length=128,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),
)

# Initalize embeddings model
# embeddings_model = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Logging variables
experiment.llm_used = get_llm_based_on_model_name_id("default")
experiment.embed_model = initialize_embedding_model(embedding_model_id="openai-text-embedding-3-large")

# Initialize chroma
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT")
)

# TODO delete after testing
chroma_client.delete_collection(name=chroma_collection_name)

# Logging variables
experiment.chunk_size = ""

# The storage layer for the parent documents
redis_store = RedisStore(
    redis_url=f'redis://{os.getenv("REDIS_URL")}:{os.getenv("REDIS_PORT")}',
    namespace=redis_namespace
)

# Get the documents
documents_directory = "../data/real_world_community_model_1st_half"
documents = load_documents_langchain(documents_directory)

# Split the documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
# docs = text_splitter.split_documents(docs)

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
doc_ids = []
experiment.retrieval_strategy = f"MultiVectorRetriever, mode: Similarity search" # There is another method - mmr

# See if we have already created summaries for this collection
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
    # adding the original chunks to the vectorstore as well
    # for i, doc in enumerate(documents):
    #     doc.metadata[id_key] = doc_ids[i]
    #     retriever.vectorstore.add_documents(documents)


# If yes, load them
# else:
#     summaries = chroma_collection.get()
#     print("Chroma collection is found...")

pattern = f"{redis_namespace}*"
# Keys that we are looking for to understand if the cache is empty
keys = list(redis_store.yield_keys(prefix=pattern))
if len(keys) == 0:
    # redis_store.mset(list(zip(doc_ids, documents)))
    retriever.docstore.mset(list(zip(doc_ids, documents)))
else:
    print("Cache in Redis is found...")

# Testing purposes
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore,
#     byte_store=redis_store,
#     id_key=id_key,
# )
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
question = "What are domains of real world community model?"
sub_docs = vectorstore.similarity_search(question, k=3)
print("subdocs: \n", sub_docs)

retrieved_docs = retriever.get_relevant_documents(question, n_results=1)
# retrieved_docs[0].page_content[0:500]
print("retrieved docs: \n", retrieved_docs)
print("retrieved documents, length: " + str(len(retrieved_docs)))

response_dictionary = qa_chain({"query": question})
response = response_dictionary['result']
print("Printing final answer", response)


def generate_response_based_on_multirepresentation_indexing_with_debt(question: str):

    experiment.question = question
    # prompt_template = get_template_based_on_template_id("default")
    # experiment.prompt_template = (
    #     get_template_based_on_template_id("default"),
    # )
    experiment.prompt_template = " "
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=retriever,
                                        #    prompt=prompt_template
                                           )
    experiment.source_agent = "Multi Representation Agent"

    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    # Source nodes
    source_nodes = retriever.get_relevant_documents(question, n_results=3)
    response_dictionary = qa_chain({"query": question})
    response = response_dictionary['result']
    experiment.response = str(response)

    return response, experiment, source_nodes
