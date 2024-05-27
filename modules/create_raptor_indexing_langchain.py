from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

#Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

from data_loading import load_documents_langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Visualisation
import matplotlib.pyplot as plt
from large_language_model_setup import get_llm_based_on_model_name_id
from raptor_functions import recursive_embed_cluster_summarize
import tiktoken

# Generating clusters of documents
from typing import Optional
import numpy as np
import umap
# Gaussian Mixture Clustering
from sklearn.mixture import GaussianMixture
# Data frames
import pandas as pd
# Chroma
from langchain_community.vectorstores import Chroma
# Load environment variables
load_dotenv()
# Rag Chain ( Langchain )
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
# String output parser
from langchain_core.output_parsers import StrOutputParser
# ChromaDB
import chromadb
# Question and Answer chain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
# Elasticsearch
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from datetime import datetime, timezone
from embedding_model_modular_setup import initialize_embedding_model


# Constants
chunk_size = 2000
chunk_overlap = 1000
model_name_id = "gpt-3.5-turbo"
embedding_model_id = "openai-text-embedding-3-large"
# chroma_collection_name = "raptor-locll-test"
chroma_collection_name = "raptor-complete-documentation-production"

# Elasticsearch related
current_time = datetime.now(timezone.utc)
elasticsearch_client = ElasticsearchClient()
experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec="milliseconds")


# Split list into parts
def split_list_into_parts(lst, n_parts):
    """
    Splits a list into n_parts roughly equal parts.
    """
    for i in range(0, len(lst), n_parts):
        yield lst[i:i + n_parts]


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize LLM
# repository_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(
#     repo_id=repository_id,
#     temperature=0.1,
#     huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),
# )


# Initialize large language model, production
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name=model_name_id
)
# Logging variables
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)

# Initialize Embedding Model
# TODO: This should be dynamic

# OpenAI embeddings
# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Hugging face Embeddings
# embeddings_model = HuggingFaceEndpointEmbeddings(
#     model="thenlper/gte-large",
#     task="feature-extraction",
#     huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),
# )
# embeddings_model = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Logging variables
experiment.embed_model = initialize_embedding_model(embedding_model_id="openai-text-embedding-3-large")
# Logging variables
experiment.chunk_size = chunk_size
experiment.chunk_overlap = chunk_overlap

# Initialize ChromaDB
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT")
)
# chroma_client.delete_collection(name=chroma_collection_name)
# Get or create Chroma collection
chroma_collection = chroma_client.get_or_create_collection(
    chroma_collection_name
)
# Get the documents
# documents_directory = "../data/documentation"
# documents_directory = "../data/real_world_community_model_1st_half"
# documents_directory = "../data/flow_and_rwcm"
# Local
# documents_directory = "../data/test"
# Production

# name of the folders: 
folders = ['decision-system', 'habitat-system', 'lifestyle-system', 'material-system', 'project-execution', 'project-plan',
           'social-system', 'system-overview']
# folders = ['system-overview']

if chroma_collection.count() == 0:
    print("Raptor collection not found, creating embeddings...")

    # Loop through folders to load and process documents
    for folder in folders:
        documents_directory = f"../data/documentation_optimal/{folder}"

        documents_pdf = load_documents_langchain(documents_directory)
        documents_text = [d.page_content for d in documents_pdf]

        # Concatenate the content, sorted and reversed
        d_sorted = sorted(documents_pdf, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        texts_split = text_splitter.split_text(concatenated_content)

        # Batch the split texts
        batch_size = 10  # Adjust batch size according to your needs
        texts_batches = list(split_list_into_parts(texts_split, batch_size))

        # Process each batch
        for batch in texts_batches:
            # Generate embeddings and summaries
            results = recursive_embed_cluster_summarize(
                batch,
                level=1,
                n_levels=3
            )

            # Initialize all_texts with the batch
            all_texts = batch.copy()

            # Extract summaries from each level and add to all_texts
            for level in sorted(results.keys()):
                summaries = results[level][1]["summaries"].tolist()
                all_texts.extend(summaries)

            # Build the vectorstore with Chroma
            vectorstore = Chroma.from_texts(
                client=chroma_client,
                collection_name=chroma_collection_name,
                texts=all_texts,
                embedding=embeddings_model
            )
    print("Complete documentation embeddings created.")
else:
    print("Raptor collection found, loading data from it...")
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=chroma_collection_name,
        embedding_function=embeddings_model
    )

# Initialize the retriever
retriever = vectorstore.as_retriever()
experiment.retrieval_strategy = "Raptor"

# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
# Prompt
prompt = hub.pull("rlm/rag-prompt")

print("The prompt we are using: ", prompt)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
# question = "What are domains of real world community model?"
question = "Would you tell me more about artificial intelligence units?"
response = rag_chain.invoke(question)
print(str(response))


def generate_response_based_on_raptor_indexing_with_debt(question: str):

    experiment.question = question
    experiment.prompt_template = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
    experiment.source_agent = "Raptor Agent"

    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    # Source nodes
    source_nodes = retriever.get_relevant_documents(question, n_results=3)
    # Response
    response = rag_chain.invoke(question)
    experiment.response = str(response)

    return str(response), experiment, source_nodes
