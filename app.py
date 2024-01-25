import logging
import sys
import os

# import dependenices
from dotenv import load_dotenv
from llama_index import (SimpleDirectoryReader,
LLMPredictor,
ServiceContext,
KnowledgeGraphIndex)
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import HuggingFaceInferenceAPI
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings import LangchainEmbedding
from pyvis.network import Network

# load environment variables from .env file
load_dotenv()

# accessing the environment variables
HF_TOKEN = os.getenv('HUGGING_FACE_API_KEY')

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# setup the LLM
llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN
)

# setup up Embedding model 

embed_model = LangchainEmbedding(
  HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN,model_name="thenlper/gte-large")
)

# load the data
documents = SimpleDirectoryReader("./data").load_data()
print('Loaded the data...')

# setup the service context

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

#setup the storage context

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

print('Constructing the Knowledge Graph Index...')

# construct the Knowlege Graph Index

index = KnowledgeGraphIndex.from_documents(documents=documents,
                                           max_triplets_per_chunk=3,
                                           service_context=service_context,
                                           storage_context=storage_context,
                                           include_embeddings=True)


print('Construction of index is finished...')

# persist the Knowledge Graph Index so we don't have to recreate it again once we want to play with it

storage_context.persist('./persistence')

# technical debt - create script from here that loads from persisted index

# query the knowledge graph by building Query Engine

query = "What does it feel like to be in a state of the flow?"
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
