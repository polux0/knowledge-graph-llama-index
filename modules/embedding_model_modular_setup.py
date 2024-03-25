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
    return HuggingFaceInferenceAPIEmbeddings(model_name=model_name, api_key=hf_token)

def get_embedding_model_based_on_model_name_id(model_name_id: str):
    """
    Retrieve the model name corresponding to the given model name ID.

    This function checks if the provided model_name_id is present in the predefined
    dictionary of embedding models (EMBEDDING_MODELS). If found, it returns the corresponding
    model name. If the model_name_id is not found, it logs an error and returns the default
    model name specified in EMBEDDING_MODELS.

    Args:
        model_name_id (str): The identifier for the desired embedding model.

    Returns:
        str: The name of the embedding model corresponding to the given identifier.
             If the identifier is not found, returns the name of the default model.
    """
    if model_name_id in EMBEDDING_MODELS:
        model_name = EMBEDDING_MODELS[model_name_id]
    else:
        logging.error(f"Invalid model_name_id: {model_name_id}. Falling back to default model name.")
        model_name = EMBEDDING_MODELS["default"]
    return model_name