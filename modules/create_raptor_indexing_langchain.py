import os
from langchain_community.llms import HuggingFaceEndpoint
#Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

from data_loading import load_documents_langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

from large_language_model_setup import get_llm_based_on_model_name_id
from raptor_functions import recursive_embed_cluster_summarize
from rewrite.MessageHistoryProcessor import MessageHistoryProcessor
import tiktoken

# Generating clusters of documents
from typing import Optional
# Chroma
from langchain_community.vectorstores import Chroma
# Rag Chain ( Langchain )
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
from embedding_model_modular_setup import get_embedding_model_based_on_model_name_id, initialize_embedding_model
# Custom prompts
from langchain_core.prompts import PromptTemplate
# Testing Groq
from langchain_groq import ChatGroq
# To introduce Retriever that returns custom scores
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from utils.environment_setup import load_environment_variables
# Rerankers
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
env_vars = load_environment_variables()

# Constants

# Local
# chunk_size = 2000
# chunk_overlap = 1000
# model_name_id = "default"
# embedding_model_id = "default"
# chroma_collection_name = "raptor-locll-test1"

# Production, 1st configuration
# chunk_size = 2000
# chunk_overlap = 1000
# chroma_collection_name = "raptor-complete-documentation-production"
# model_name_id = "default"
# embedding_model_id = "openai-text-embedding-3-large"

# Production, 2nd configuration
chunk_size = 5096
chunk_overlap = 2048
chroma_collection_name = "raptor-complete-documentation-production-1"
model_name_id = "default"
embedding_model_id = "openai-text-embedding-3-large"

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


repository_id = "mistralai/Mistral-7B-Instruct-v0.3"
# Initialize large language model, local testing

llm = HuggingFaceEndpoint(
    repo_id=repository_id,
    temperature=0.1,
    huggingfacehub_api_token=env_vars["HUGGING_FACE_API_KEY"],
)

# Experiment with Groq

# os.environ["GROQ_API_KEY"] = env_vars["GROQ_API_KEY"]
# llm = ChatGroq(
#     model="llama3-70b-8192",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
# Initialize large language model, production
# llm = ChatOpenAI(
#     openai_api_key=env_vars["OPENAI_API_KEY"],
#     model_name=model_name_id
# )
# Logging variables
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)

# Initialize Embedding Model
# TODO: This should be dynamic
# Hugging face Embeddings
# embeddings_model = HuggingFaceEndpointEmbeddings(
#     model="thenlper/gte-large",
#     task="feature-extraction",
#     huggingfacehub_api_token=env_vars["HUGGING_FACE_API_KEY"],
# )
# Cohere embeddings
# embeddings_model = CohereEmbeddings(cohere_api_key=env_vars["COHERE_API_KEY"])
# OpenAI embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Logging variables
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(model_name_id=="openai-text-embedding-3-large")
experiment.chunk_size = chunk_size
experiment.chunk_overlap = chunk_overlap

# Initialize ChromaDB
chroma_client = chromadb.HttpClient(
    host=env_vars["CHROMA_URL"], port=env_vars["CHROMA_PORT"]
)
# chroma_client.delete_collection(name=chroma_collection_name)
# Get or create Chroma collection
chroma_collection = chroma_client.get_or_create_collection(
    chroma_collection_name
)

# Production
# folders = ['decision-system', 'habitat-system', 'lifestyle-system', 'material-system', 'project-execution', 'project-plan',
#            'social-system', 'system-overview']
# Local
folders = ['test1']

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
    print("Created RAPTOR embeddings for complete documenation.")
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
# Initialize the Reranker
compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
# Prompt

# Define prompt

# Previously
# prompt = hub.pull("rlm/rag-prompt")
# Custom
prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template=(
        "You are a friendly library assistant designed to create detailed and precise responses about the standards in the Auravana Project documentation, based on the context given below. For specific definitions of models or domains, please quote directly from the context, whilst ensuring that you answer the question. If you can't find the answer, say 'I don't have the context required to answer that question - could you please rephrase it?'\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    )
)
print("The prompt we are using: ", prompt)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Question
question = "Would you tell me more about artificial intelligence units?"

# Retriever related: 

# source_nodes = retriever.get_relevant_documents(
#     question,
#     n_results=3,
#     return_source_documents=True
# )


def stringify_and_combine_nodes(nodes) -> str:
    combined_output = "Nodes retrieved: \n\n"
    combined_output += "\n".join([repr(node) for node in nodes])
    return combined_output


response = rag_chain.invoke(question)
# print(str(response))


def generate_response_based_on_raptor_indexing_with_debt(question: str):

    experiment.question = question
    experiment.prompt_template = prompt.template
    experiment.source_agent = "Raptor Agent"

    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    # Source nodes
    source_nodes = retriever.get_relevant_documents(question, n_results=3)
    experiment.retrieved_nodes = stringify_and_combine_nodes(source_nodes)
    response = rag_chain.invoke(question)
    experiment.response = str(response)

    return str(response), experiment, source_nodes

#TODO: This is helper function that probably should be moved in utils
@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs

def generate_response_based_on_raptor_indexing(question: str, chat_id: int):

    experiment.question = question
    experiment.prompt_template = prompt.template
    experiment.source_agent = "Raptor Agent"

    source_nodes_with_score = compression_retriever.invoke(question)

    current_time = datetime.now(timezone.utc)
    experiment.updated_at = current_time.isoformat(timespec="milliseconds")
    # source_nodes = retriever.get_relevant_documents(question, n_results=3)
    experiment.retrieved_nodes = stringify_and_combine_nodes(source_nodes_with_score)
    response = rag_chain.invoke(question)
    experiment.response = str(response)

    return str(response), experiment, source_nodes_with_score
