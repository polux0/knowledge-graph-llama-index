import logging
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.bridge.langchain import langchain
from embedding_models import EMBEDDING_MODELS
def initialize_embedding_model(hf_token, embedding_model_id):
    """Initialize the embedding model using Langchain Embedding.

    Args:
        hf_token (str): The Hugging Face API token.

    Returns:
        LangchainEmbedding: The initialized embedding model.
    """
    # technical debt, move to embedding_models.py
    if embedding_model_id in EMBEDDING_MODELS:
        model_name = EMBEDDING_MODELS[embedding_model_id]
    else:
        logging.error(f"Invalid embedding_model_id: {embedding_model_id}. Falling back to default embedding model name.")
        model_name = EMBEDDING_MODELS["default"]
    return HuggingFaceEmbedding(model_name=model_name)

