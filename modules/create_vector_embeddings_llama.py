from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from data_loading import load_documents
from embedding_model_modular_setup import initialize_embedding_model
from environment_setup import (load_environment_variables, setup_logging)
from large_language_model_setup import initialize_llm
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
import json


# load the documents, previously used for knowledge graph construction

# documents = load_documents("../data/real_world_community_model")

loader = PDFReader()
documents = loader.load_data(file=Path("./data/real_world_community_model/Aurvana System Overiew - 73 - 84.pdf"))


doc_text = "\n\n".join([d.get_content() for d in documents])
docs = [Document(text=doc_text)]


node_parser = SentenceSplitter(chunk_size=1024)

base_nodes = node_parser.get_nodes_from_documents(docs)
# set node ids to be a constant
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"

# enviornment variables
    
env_vars = load_environment_variables()

# embedings 
embed_model = initialize_embedding_model(env_vars['HF_TOKEN'], embedding_model_id="default")

# large language model
llm = initialize_llm(env_vars['HF_TOKEN'], model_name_id="default")

try:
    storage_context = StorageContext.from_defaults(persist_dir="./persistence/vectors") 
    # load index
    base_index = load_index_from_storage(storage_context, embed_model=embed_model)
    index_loaded = True
    print('Loading of index is finished...')
except:
    index_loaded = False
    print('Index not found, constructing one...')
if not index_loaded:
    # rebuild storage context
    base_index = VectorStoreIndex(base_nodes, embed_model=embed_model)
    base_index.storage_context.persist('./persistence/vectors')

base_retriever = base_index.as_retriever(similarity_top_k=2)


# defining Baseline Retriever that simply fetches the top-k raw text nodes by embedding similarity

retrievals = base_retriever.retrieve(
    "Can you tell me about the key domains of Real World Community Model"
)

# to print all nodes that are retrieved ( debugging purposes )

# print("displaying retrievals")

for n in retrievals:
    display_source_node(n, source_length=1500)

query_engine_base = RetrieverQueryEngine.from_args(base_retriever, llm=llm)

response = query_engine_base.query(
    "Can you tell me about the key domains of Real World Community Model"
)

print(response)


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

all_nodes_dict = {n.node_id: n for n in all_nodes}

vector_index_chunk = VectorStoreIndex(all_nodes, embed_model=embed_model)
vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k = 3)

retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True
)

nodes = retriever_chunk.retrieve(
    "Can you tell me about the key domains of Real World Community Model"
)
# for node in nodes:
#     display_source_node(node, source_length=2000)

query_engine_chunk = RetrieverQueryEngine.from_args(retriever_chunk, llm=llm)

response = query_engine_chunk.query(
    "Can you tell me about the key domains of Real World Community Model"
)
print("With parent-child retriever enabled****************************************************************")
print(str(response))