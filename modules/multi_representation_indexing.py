import uuid

from data_loading import load_documents, load_documents_langchain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
# from langchain_community.llms import HuggingFaceEndpoint
import os
from langchain.storage import InMemoryByteStore
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

load_dotenv()
repository_id = "mistralai/Mistral-7B-Instruct-v0.2"


llm = HuggingFaceEndpoint(
    repo_id=repository_id,
    max_length=128,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv('HUGGING_FACE_API_KEY')
)

# Get the documents
documents_directory = "../data/real_world_community_model_1st_half"
documents = load_documents_langchain(documents_directory)

chain = (
    {"document": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{document}")
    | llm
    | StrOutputParser()
)

summaries = chain.batch(documents, {"max_concurrency": 5})

# print("Summaries: ", summaries)

embeddings_model = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))

# The vectorstore to use to index the child chunks
# TODO Possibility to create this dynamically based on parameters
# That we have to define
vectorstore = Chroma(collection_name="summaries",
                     embedding_function=embeddings_model)


# The storage layer for the parent documents
# TODO See if it is possible to integrate Redis Vector Store from Llama Index or find Langchain alternative
store = InMemoryByteStore()

id_key = "doc_id"

# The retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in documents]

# Documents linked to summaries
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, documents)))

query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query, k=1)
# print("subdocs: ", sub_docs[0])

retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
retrieved_docs[0].page_content[0:500]
# print("retrieved docs: ", retrieved_docs)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever
)

# Pass question to the qa_chain
question = "What are domains of real world community model?"
result = qa_chain({"query": question})
# print("Printing final answer", result["result"])