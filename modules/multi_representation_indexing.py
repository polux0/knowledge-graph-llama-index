# System
from rewrite.utils import Utils
from utils.environment_setup import load_environment_variables
import uuid
import re
from data_loading import load_documents_langchain
from embedding_model_modular_setup import get_embedding_model_based_on_model_name_id, initialize_embedding_model
from format_message_with_prompt import format_message
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models import ChatOpenAI

# Redis related
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from redis import Redis

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
# TODO: Delete after testing, as we are trying to introduce retriever that has relevance score of documents
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# TODO: 
from rewrite.retrievers.CustomMultiVectorRetriever import CustomMultiVectorRetriever
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import chromadb

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSequence

# Elasticsearch
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from datetime import datetime, timezone


# Prompts
from langchain_core.prompts import PromptTemplate
from large_language_model_setup import get_llm_based_on_model_name_id
from prompts import get_template_based_on_template_id

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Logging
import logging
import sys

# Rerankers
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from rewrite.MessageHistoryProcessor import MessageHistoryProcessor

env_vars = load_environment_variables()


def preprocess_text(text):
    # Replace newline characters
    text = text.replace("\n", " ")
    # Use a regular expression to find and replace all Unicode characters starting with \u
    text = re.sub(r"\\u[0-9A-Fa-f]{4}", " ", text)
    return text


# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

repository_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Constants
chunk_size = 2048
chunk_overlap = 518
# Local
# model_name_id = "default"
# embedding_model_id = "openai-text-embedding-3-large"
# chroma_collection_name = "MRITESTTTTTTTTTTT3"
# redis_namespace = "parent-documents-MRITESTTTTTTTTTTT3"
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
    huggingfacehub_api_token=env_vars["HUGGING_FACE_API_KEY"],
)

# Initialize large language model, production
contextualize_llm = ChatOpenAI(openai_api_key=env_vars["OPENAI_API_KEY"], model_name="gpt-3.5-turbo")

# Initalize embeddings model
# embeddings_model = CohereEmbeddings(cohere_api_key=env_vars["COHERE_API_KEY"))
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Logging variables
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(
    model_name_id="openai-text-embedding-3-large"
)

# Initialize chroma
chroma_client = chromadb.HttpClient(
    host=env_vars["CHROMA_URL"], port=env_vars["CHROMA_PORT"]
)
# TODO delete after testing
# chroma_client.delete_collection(name=chroma_collection_name)

# Logging variables
experiment.chunk_size = chunk_size
experiment.chunk_overlap = chunk_overlap

redis_client = Redis(
    host=env_vars["REDIS_HOST"],
    port=env_vars["REDIS_PORT"],
    username=env_vars["REDIS_USERNAME"],
    password=env_vars["REDIS_PASSWORD"],
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
retriever = CustomMultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=redis_store,
    id_key=id_key,
)

# Reranker
compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
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
        # Adding the original chunks to the vectorstore as well
        # for i, doc in enumerate(documents):
        #     doc.metadata[id_key] = doc_ids[i]
        #     retriever.vectorstore.add_documents(documents)
    print("Create MRI embeddings for complete documentation...")
print("MRI Embeddings have been already created...")
print(
    f"Are there embeddings inside MRI collection {chroma_collection.name} ?",
    f"count: {chroma_collection.count()}",
)


#TODO: Move to `utils` or modify the way we are storing retrieved nodes

def stringify_and_combine(sub_docs, retrieved_docs) -> str:
    combined_output = "Summary documents: \n"
    combined_output += "\n".join([repr(doc) for doc in sub_docs])
    combined_output += "\nNodes retrieved from original documents: \n"
    combined_output += "\n".join([repr(doc) for doc in retrieved_docs])
    return combined_output

#TODO: Message history is not configured to work properly with streamlit.
#TODO: Temporairly will be removed
def generate_response_based_on_multirepresentation_indexing_with_debt(question: str):

    experiment.question = question
    experiment.prompt_template = " "
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        #    prompt=prompt_template
    )
    experiment.source_agent = "Multi Representation Agent"

    sub_docs = vectorstore.similarity_search(question, k=3)
    retrieved_docs = retriever.get_relevant_documents(question, n_results=3)

    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    source_nodes = stringify_and_combine(sub_docs, retrieved_docs)
    experiment.retrieved_nodes = source_nodes
    response_dictionary = qa_chain({"query": question})
    response = response_dictionary["result"]
    experiment.response = str(response)

    return response, experiment, source_nodes, retrieved_docs


#TODO: Code clean-up
def generate_response_based_on_multirepresentation_indexing(question: str, chat_id: int):

    retrieved_docs = compression_retriever.get_relevant_documents(question, n_results=3)
    print(f"!MRI RETRIEVED DOCUMENTS:\n", retrieved_docs)
    # print(f"!MRI RETRIEVED DOCUMENTS, Document by document: \n")
    # for doc in retrieved_docs:
    #         print(doc)

    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=(
            "You are a friendly library assistant designed to create detailed and precise responses about the standards in the Auravana Project documentation, based on the context given below. For specific definitions of models or domains, please quote directly from the context, whilst ensuring that you answer the question. If you can't find the answer, say 'I don't have the context required to answer that question - could you please rephrase it?'\n"
            "Question: {question}\n"
            "Context: {context}\n"
            "Answer:"
        )
    )

    experiment.question = question
    experiment.prompt_template = " "

    # Replacement
    rag_chain = (
        {"context": compression_retriever , "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    experiment.source_agent = "Multi Representation Agent"

    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    source_nodes = Utils.serialize_retrieved_nodes_for_the_mri_agent(retrieved_docs)
    experiment.retrieved_nodes = source_nodes

    # Replacement
    response = rag_chain.invoke(question)
    experiment.response = str(response)

    return str(response), experiment, str(source_nodes), retrieved_docs
