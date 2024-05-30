import uuid
import re
from data_loading import load_documents_langchain
from embedding_model_modular_setup import initialize_embedding_model
from format_message_with_prompt import format_message
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models import ChatOpenAI

import os

# Redis related
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from redis import Redis

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

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Logging
import logging
import sys


def preprocess_text(text):
    # Replace newline characters
    text = text.replace("\n", " ")
    # Use a regular expression to find and replace all Unicode characters starting with \u
    text = re.sub(r"\\u[0-9A-Fa-f]{4}", " ", text)
    return text


# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()
repository_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Constants
chunk_size = 2048
chunk_overlap = 518
# Local
# model_name_id = "default"
# embedding_model_id = "openai-text-embedding-3-large"
# chroma_collection_name = "MRITESTTTTTTTTTTT2"
# redis_namespace = "parent-documents-MRITESTTTTTTTTTTT2"
# documents_directory = "../data/documentation_optimal/test1"
# Production
model_name_id = "default"
embedding_model_id = "openai-text-embedding-3-large"
chroma_collection_name = "summaries-complete-documentation2"
redis_namespace = "parent-documents-summaries-complete-documentation2"
documents_directory = "../data/documentation_optimal"

# Elasticsearch related
current_time = datetime.now(timezone.utc)
elasticsearch_client = ElasticsearchClient()
experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec="milliseconds")

# Initialize large language model, for local testing
llm = HuggingFaceEndpoint(
    repo_id=repository_id,
    # max_length=128,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),
)

# Initialize large language model, production
# llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name_id)

# Initalize embeddings model
# embeddings_model = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Logging variables
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)
experiment.embeddings_model = initialize_embedding_model(
    embedding_model_id="openai-text-embedding-3-large"
)

# Initialize chroma
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT")
)
# TODO delete after testing
# chroma_client.delete_collection(name=chroma_collection_name)

# Logging variables
experiment.chunk_size = chunk_size
experiment.chunk_overlap = chunk_overlap

redis_client = Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    password=os.getenv("REDIS_PASSWORD"),
)
redis_store = RedisStore(client=redis_client, namespace=redis_namespace)
all_documents = load_documents_langchain(documents_directory)

# Preprocess the text to remove newline characters
for doc in all_documents:
    doc.page_content = preprocess_text(doc.page_content)
# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
documents = text_splitter.split_documents(all_documents)

chain = (
    {"document": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(
        "Summarize the following document:\n\n{document}"
    )
    | llm
    | StrOutputParser()
)

# Get or create Chroma collection
chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)
# Define vector store
vectorstore = Chroma(
    client=chroma_client,
    collection_name=chroma_collection_name,
    embedding_function=embeddings_model,
)
id_key = "doc_id"

# The retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=redis_store,
    id_key=id_key,
)
doc_ids = []
experiment.retrieval_strategy = (
    f"MultiVectorRetriever, mode: Similarity search"  # There is another method - mmr
)

# Check if redis cache is empty
pattern = f"{redis_namespace}*"
keys = list(redis_store.yield_keys(prefix=pattern))
# See if we have already created summaries for this collection
# If no, create them

if chroma_collection.count() == 0 and len(keys) == 0:
    print("MRI Collection not found, creating embeddings...")
    # Solution for processing document by document:
    for document in documents:
        print(f"Processing document {document.metadata}")
        document_list = [document]
        summaries = chain.batch(document_list, {"max_concurrency": 2})
        doc_id = f"{redis_namespace}-{uuid.uuid4()}"
        # Create summary document
        summary_doc = Document(page_content=summaries[0], metadata={id_key: doc_id})
        # Add to Chroma retriever
        retriever.vectorstore.add_documents([summary_doc])
        # Store the original document in Redis
        retriever.docstore.mset([(doc_id, document)])
        # Our own solution
        # for document in documents:
        #     summaries = chain.batch(document, {"max_concurrency": 2})
        #     doc_ids = [f"{redis_namespace}-{uuid.uuid4()}" for _ in documents]
        #     # Documents linked to summaries
        #     summary_docs = [
        #         Document(page_content=s, metadata={id_key: doc_ids[i]})
        #         for i, s in enumerate(summaries)
        #     ]
        #     retriever.vectorstore.add_documents(summary_docs)
        #     retriever.docstore.mset(list(zip(doc_ids, documents)))
        # Our own solution

        # adding the original chunks to the vectorstore as well
        # for i, doc in enumerate(documents):
        #     doc.metadata[id_key] = doc_ids[i]
        #     retriever.vectorstore.add_documents(documents)
        # Our own solution
    print("Create MRI embeddings for complete documentation...")
print("MRI Embeddings have been already created...")
print(
    f"Are there embeddings inside MRI collection {chroma_collection.name} ?",
    f"count: {chroma_collection.count()}",
)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
question = "What are domains of real world community model?"
question1 = "Domains of real world community model"
sub_docs = vectorstore.similarity_search(question, k=3)
# for docs in sub_docs:
#     print("subdocs: \n", docs)

retrieved_docs = retriever.get_relevant_documents(question1, n_results=3)
# print("Nodes retrieved: \n")
# for node in retrieved_docs:
#     print(node)
# retrieved_docs[0].page_content[0:500]
print("retrieved docs: \n", retrieved_docs)
print("retrieved documents, length: " + str(len(retrieved_docs)))

response_dictionary = qa_chain({"query": question})
response = response_dictionary["result"]
print("Printing final answer", response)


def stringify_and_combine(sub_docs, retrieved_docs) -> str:
    combined_output = "Summary documents: \n"
    combined_output += "\n".join([repr(doc) for doc in sub_docs])
    combined_output += "\n\Nodes retrieved from original documents ( linked to summaries ) : \n"
    combined_output += "\n".join([repr(doc) for doc in retrieved_docs])
    return combined_output


# Call the function and print the result
# combined_string = stringify_and_combine(sub_docs, retrieved_docs)
# print("What is going into the elasticsearch: \n", combined_string)


def generate_response_based_on_multirepresentation_indexing_with_debt(question: str):

    experiment.question = question
    experiment.prompt_template = " "
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        #    prompt=prompt_template
    )
    experiment.source_agent = "Multi Representation Agent"

    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    source_nodes = stringify_and_combine(sub_docs, retrieved_docs)
    experiment.retrieved_nodes = source_nodes
    response_dictionary = qa_chain({"query": question})
    response = response_dictionary["result"]
    experiment.response = str(response)

    return response, experiment, source_nodes
