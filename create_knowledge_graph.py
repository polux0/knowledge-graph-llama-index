import logging
import sys
import os

# import dependenices
from dotenv import load_dotenv
from llama_index import (SimpleDirectoryReader,
LLMPredictor,
ServiceContext,
KnowledgeGraphIndex)
from llama_index.graph_stores import SimpleGraphStore, Neo4jGraphStore
import neo4j
from llama_index.storage.storage_context import StorageContext
from llama_index import load_index_from_storage
from llama_index.llms import HuggingFaceInferenceAPI
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings import LangchainEmbedding
from pyvis.network import Network

# load environment variables from .env file
load_dotenv()

# accessing the environment variables
HF_TOKEN = os.getenv('HUGGING_FACE_API_KEY')

# neo4j environment variables
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSOWRD = os.getenv('NEO4J_PASSOWRD')
NEO4J_URL = os.getenv('NEO4J_URL')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')


print('Neo4j version')
print(neo4j.__version__)

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# setup the LLM
llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN
)

# setup up Embedding model 

embed_model = LangchainEmbedding(
  HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN,model_name="thenlper/gte-large")
)

# load the data
documents = SimpleDirectoryReader("./data/real_world_community_model").load_data()
print('Loaded the data...')

# setup the service context

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model,
    chunk_overlap=True
)

#setup the storage context

graph_store = Neo4jGraphStore(
  username=NEO4J_USERNAME,
  password=NEO4J_PASSOWRD,
  url=NEO4J_URL,
  database=NEO4J_DATABASE
)

print('Constructing the Knowledge Graph Index...')

# load the Knowlege Graph Index ( if it's not first application )
try:
  storage_context = StorageContext.from_defaults(persist_dir='./persistence/real_world_community_model_15_triplets_per_chunk_neo4j')
  index = load_index_from_storage(storage_context=storage_context,
                                  service_context=service_context,
                                  max_triplets_per_chunk=15,
                                  include_embeddings=True)
  index_loaded = True
  print('Loading of index is finished...')
except:
  index_loaded = False
  print('Index not found, constructing one...')
if not index_loaded:
  # construct the Knowledge Graph Index
  
  storage_context = StorageContext.from_defaults(graph_store=graph_store)
  index = KnowledgeGraphIndex.from_documents(documents=documents,
                                             max_triplets_per_chunk=15,
                                             service_context=service_context,
                                             storage_context=storage_context,
                                             include_embeddings=False)
  
  

print('Construction of index is finished...')

# persist the Knowledge Graph Index so we don't have to recreate it again once we want to play with it

storage_context.persist('./persistence/real_world_community_model_15_triplets_per_chunk_neo4j')

# technical debt - create script from here that loads from persisted index

# query the knowledge graph by building Query Engine

# Related to questions about flow

# query = "What does it feel like to be in a state of the flow?"
# query = 'What is the ‘flow’ cycle?'
# query = 'What are the five performance enhancing chemicals that are released during a flow state?'

# query = "What is reality?"

# query = "What is Real world community model?"
# Answer, using Mistral-7B-Instruct-v0.2 as LLM
# The Real World Community Model is a concept described in the Aurvana System Overview document. It refers to a model of a community as a population of people navigating together in common, surrounded by their environment, and includes the direction domain (goals and other directional components), data domain (data collected from the environment and data output as the result of information processing), knowledge domain (logical integration of observations and relationships into common knowledge, social approach, and the value system and reasoning), and values domain (society's value system and reasoning that supports the fulfillment of needs and orients the total alignment with the common direction of intent). The model is dynamically simulatable and adaptive, and the whole of the real world must be modeled for optimal societal structuring.

query = "What are the domains of the Real World Commuity Model?"


# Related to questions about `real world community model`

query_engine = index.as_query_engine(include_text=True,
                                     response_mode ="tree_summarize",
                                     embedding_mode="hybrid",
                                     similarity_top_k=5,)
#
message_template =f"""<|system|>Please check if the following pieces of context has any mention of the keywords provided in the Question.
                                If not then don't know the answer, just say that you don't know.
                                Stop there.
                                Please do not try to make up an answer.</s>
<|user|>
Question: {query}
Helpful Answer:
</s>"""
#
response = query_engine.query(message_template)
#
print(response.response.split("<|assistant|>")[-1].strip())

# visualizing the graph
g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("real_world_community_model.html")