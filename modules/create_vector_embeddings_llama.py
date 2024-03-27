from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from data_loading import load_documents
from embedding_model_modular_setup import initialize_embedding_model
from environment_setup import (load_environment_variables, setup_logging)
from large_language_model_setup import initialize_llm
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
import chromadb
import json

# Elasticsearch realted
from elasticsearch_service import ExperimentDocument, ElasticsearchClient

from large_language_model_setup import get_llm_based_on_model_name_id
from embedding_model_modular_setup import get_embedding_model_based_on_model_name_id
from prompts import get_template_based_on_template_id
from datetime import datetime, timezone

# Formating message related

from format_message_with_prompt import format_message

# Elasticsearch related
es_client = ElasticsearchClient(scheme='http', host='elasticsearch', port=9200)

# Initialize Experiment

# Timestamp realted
# Get the current time in UTC, making it timezone-aware
current_time = datetime.now(timezone.utc)

# Format the current time as an ISO 8601 string, including milliseconds
created_at = current_time.isoformat(timespec='milliseconds')
updated_at = created_at  # Initially, both timestamps will be the same

experiment = ExperimentDocument()

# Variables
model_name_id = "default"
embedding_model_id = "default" 
chunk_size = 256
max_triplets_per_chunk = 15

# Load the documents, modular function previously used for knowledge graph construction

documents_directory = "../data/real_world_community_model"

documents = load_documents(documents_directory)

# load the documents, example from llama documentation

# loader = PDFReader()
# documents = loader.load_data(file=Path("./data/real_world_community_model/Aurvana System Overiew - 73 - 84.pdf"))


doc_text = "\n\n".join([d.get_content() for d in documents])
docs = [Document(text=doc_text)]


node_parser = SentenceSplitter(chunk_size=1024)
experiment.chunk_size=1024

base_nodes = node_parser.get_nodes_from_documents(docs)
# set node ids to be a constant
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"
    
env_vars = load_environment_variables()

print("env_vars: {}".format(env_vars))

# embedings 
embed_model = initialize_embedding_model(hf_token=env_vars['HF_TOKEN_ANOTHER'], embedding_model_id=embedding_model_id)
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(embedding_model_id)

# large language model
experiment.llm_used = get_llm_based_on_model_name_id(model_name_id)
llm = initialize_llm(hf_token=env_vars['HF_TOKEN_ANOTHER'], model_name_id = model_name_id)

# initialize ChromaDB
remote_db = chromadb.HttpClient(host='chromadb', port=8000) 
# remote_db = chromadb.HttpClient()

print("All collections in Chroma: ", remote_db.list_collections())

chroma_collection = remote_db.get_or_create_collection('real_world_community_model')
chroma_collection_parent = remote_db.get_or_create_collection("real_world_community_model_parent")

print("Are there embeddings inside those collections? real_world_community_model, count: ", chroma_collection.count())
print("Are there embeddings inside those collections? real_world_community_model_parent, count: ", chroma_collection_parent.count())

vector_store = ChromaVectorStore(chroma_collection=chroma_collection, ssl=False)

vector_store_parent = ChromaVectorStore(chroma_collection=chroma_collection_parent)
# Storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

storage_context_parent = StorageContext.from_defaults(vector_store=vector_store_parent)

# necessary to create a collection for the first time

if chroma_collection.count() == 0:

    base_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

else:
# after collection was sucessfully created

    base_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)

base_retriever = base_index.as_retriever(similarity_top_k=2)


# defining Baseline Retriever that simply fetches the top-k raw text nodes by embedding similarity

# retrievals = base_retriever.retrieve(
#     "Can you tell me about the key domains of Real World Community Model"
# )

# # to print all nodes that are retrieved ( debugging purposes )

# print("Retrievals, length: ", len(retrievals))

# for n in retrievals:
#     display_source_node(n, source_length=1500)

query_engine_base = RetrieverQueryEngine.from_args(base_retriever, llm=llm)

# response = query_engine_base.query(
#     "Can you tell me about the key domains of Real World Community Model"
# )

# print("Base retrieval, response: \n")
# print(response)


# Part II Chuck References: Smaller Child Chunks Reffering to Bigger Parent Chunk

sub_chunk_sizes = [128, 256, 512]

# technical debt - create service context for this
sub_node_parsers = [
    SentenceSplitter(chunk_size=c, chunk_overlap=c/2) for c in sub_chunk_sizes
]

all_nodes = []

for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_indices = [
            IndexNode.from_text_node(sub_node, base_node.node_id) for sub_node in sub_nodes
        ]
        all_nodes.extend(sub_indices)

    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)


print("Started making dictionaries")
all_nodes_dict = {n.node_id: n for n in all_nodes}
print("Finished with making dictionaries")

# necessary to create a collection for the first time

if chroma_collection_parent.count() == 0:

    vector_index_chunk = VectorStoreIndex(all_nodes, storage_context=storage_context_parent,embed_model=embed_model)

else: 

# after collection was sucessfully created
    vector_index_chunk = VectorStoreIndex.from_vector_store(vector_store_parent, storage_context=storage_context_parent, embed_model=embed_model)

vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k = 3)

retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True
)
experiment.retrieval_strategy = "RecursiveRetriever - Parent Child"

# nodes = retriever_chunk.retrieve(
#     "Can you tell me about the key domains of Real World Community Model"
# )
# print("Parent retrievals, length: ", len(nodes))

# print("Displaying source node with parent retrieval")
# for node in nodes:
#     display_source_node(node, source_length=2000)

query_engine_chunk = RetrieverQueryEngine.from_args(retriever_chunk, llm=llm)

def generate_response_based_on_vector_embeddings(question:str):

    experiment.question = question
    prompt_template = get_template_based_on_template_id("simon")
    experiment.prompt_template = get_template_based_on_template_id("simon"),
    print("With parent-child retriever*******************************************************************\n\n: ")
    response = query_engine_chunk.query(format_message(question, prompt_template))
    experiment.response = str(response)
    
    current_time = datetime.now(timezone.utc)

    # Format the current time as an ISO 8601 string, including milliseconds
    experiment.updated_at = current_time.isoformat(timespec='milliseconds')

    es_client.save_experiment(experiment_document=experiment)

    print("Final response", str(response))
    return response
