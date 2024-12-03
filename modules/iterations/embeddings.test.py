
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, 
    api_url=api_url,
    model_name="bge-large-en-v1.5"
)
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=environment)

loader = PyPDFDirectoryLoader("data")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

vectordb = Pinecone.from_documents(chunks, embeddings, index_name=index_name, namespace=namespace)