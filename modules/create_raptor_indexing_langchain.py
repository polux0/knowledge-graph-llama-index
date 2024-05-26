from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

#Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from data_loading import load_documents_langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Visualisation
import matplotlib.pyplot as plt
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
# from langchain.chat_models import ChatOpenAI
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


# Constants
chunk_size = 2000
chunk_overlap = 1000
model_name_id = "gpt-3.5-turbo"
embedding_model_id = "openai-text-embedding-3-large"
# chroma_collection_name = "raptor-locll-test"
chroma_collection_name = "raptor-complete-documentation-production"

# Initialize LLM
repository_id = "mistralai/Mistral-7B-Instruct-v0.2"
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
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize ChromaDB
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_URL"), port=os.getenv("CHROMA_PORT")
)
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
documents_directory = "../data/documentation_optimal"

documents_pdf = load_documents_langchain(documents_directory)
# print("Documents: ", documents_pdf)
documents_text = [d.page_content for d in documents_pdf]
# print("Documents.text: ", documents_text)

# Split the documents into chunks
chunk_size = 2000
chunk_overlap = 1000

# Concatante the content, as it's list of strings
d_sorted = sorted(documents_pdf, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=chunk_size, chunk_overlap=chunk_overlap
# )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap 
    )
documents = text_splitter.split_documents(documents_pdf)
# print("documents length: %s" % len(documents))
# print("documents splitted: ", documents)
# for document in documents:
#     print(document.page_content)

texts_split = text_splitter.split_text(concatenated_content)
# Understand if collection is empty
if chroma_collection.count() == 0:
    # for document in documents:
    print("Raptor collection not found, creating embeddings...")
    # Build a tree
    # Was before
    # leaf_texts = documents_text
    # Testing now
    leaf_texts = texts_split
    results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

    # Initialize all_texts with leaf_texts
    all_texts = leaf_texts.copy()

    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)

    # Use all_texts to build the vectorstore with Chroma
    vectorstore = Chroma.from_texts(
        client=chroma_client,
        collection_name=chroma_collection_name,
        texts=all_texts,
        embedding=embeddings_model
        )
else:
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=chroma_collection_name,
        embedding_function=embeddings_model
    )
    print("Raptor collection found, loading data from it...")

retriever = vectorstore.as_retriever()
# TODO: Play around with RetrievalQA chain once we start loading embeddings from remote db
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Test it in RAG Chain

# Prompt
prompt = hub.pull("rlm/rag-prompt")

print("The prompt we are using: ", prompt)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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
