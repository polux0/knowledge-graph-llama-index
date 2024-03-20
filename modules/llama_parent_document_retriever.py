print("This code has actually happened0!")
from pathlib import Path
print("This code has actually happened1!")
# from llama_index.readers.file import PDFReader
from data_loading import load_documents
print("This code has actually happened2!")
from llama_index.core.response.notebook_utils import display_source_node
print("This code has actually happened3!")
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
import json

# print("This code has actually happened!")
# try:
#     loader = PDFReader()
#     print("loader, PDFReader() is loaded...")
# except ImportError:
#     print("loader, PDFReader() could not be loaded, error:", str(ImportError))

try:
    # documents_directory = '../../data/real_world_community_model/Aurvana System Overiew - 73 - 84.pdf'
    # documents = loader.load_data(file=Path(documents_directory))
    documents = load_documents("../data/real_world_community_model")
    print("Sucessfully loaded documentation...")
except Exception as e:
    print("Error loading documentation: " + str(e))
from llama_index.core import Document

doc_text = "\n\n".join([d.get_content() for d in documents])
docs = [Document(text=doc_text)]

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode

node_parser = SentenceSplitter(chunk_size=1024)

base_nodes = node_parser.get_nodes_from_documents(docs)
# set node ids to be a constant
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"

# Our own embedding model

from embedding_model_setup import initialize_embedding_model
from environment_setup import (load_environment_variables, setup_logging)

env_vars = load_environment_variables()

embed_model = initialize_embedding_model(env_vars['HF_TOKEN'], embedding_model_id="default")

# Our own LLM

from large_language_model_setup import initialize_llm

llm = initialize_llm(env_vars['HF_TOKEN'], model_name_id="default")

#Define Baseline Retriever

base_index = VectorStoreIndex(base_nodes, embed_model=e)
base_retriever = base_index.as_retriever(similarity_top_k=2)

# Defining a baseline retriever that simply fetches the top-k raw text nodes by embedding similarity

retrievals = base_retriever.retrieve(
    "Can you tell me more about real world community model"
)

for n in retrievals:
    display_source_node(n, source_length=1500)


