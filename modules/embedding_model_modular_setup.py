import logging

from embedding_models import EMBEDDING_MODELS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from huggingface_hub import login
from environment_setup import load_environment_variables
import os
env_vars = load_environment_variables()


def initialize_embedding_model(hf_token, embedding_model_id):
    """Initialize the embedding model using Langchain Embedding.

    Args:
        hf_token (str): The Hugging Face API token.

    Returns:
        LangchainEmbedding: The initialized embedding model.
    """
    # technical debt, move to embedding_models.py
    login(token=env_vars['HF_TOKEN'])
    if embedding_model_id in EMBEDDING_MODELS:
        if embedding_model_id == 'cohere':
            os.environ["COHERE_API_KEY"] = env_vars['COHERE_API_KEY']
            model_name = CohereEmbedding(
                cohere_api_key=env_vars['COHERE_API_KEY'],
                model_name="embed-english-v3.0",
                input_type="search_query",
            )
            return model_name
        else:
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