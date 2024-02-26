from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings import LangchainEmbedding

def initialize_embedding_model(model_name, hf_token):
    """Initialize the embedding model using Langchain Embedding.

    Args:
        hf_token (str): The Hugging Face API token.

    Returns:
        LangchainEmbedding: The initialized embedding model.
    """
    return LangchainEmbedding(HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name=model_name))