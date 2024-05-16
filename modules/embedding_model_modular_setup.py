import logging

from embedding_models import EMBEDDING_MODELS
from llama_index.embeddings.huggingface import (
            HuggingFaceInferenceAPIEmbedding,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from huggingface_hub import login
from environment_setup import load_environment_variables
import os
env_vars = load_environment_variables()


def initialize_embedding_model(embedding_model_id):
    """Initialize the embedding model using Langchain Embedding.

    Args:
        embedding_model_id (str): Hugging face embedding model name

    Returns:
        HuggingFaceInferenceAPIEmbedding: The initialized embedding model.
    """
    # technical debt, move to embedding_models.py
    hugging_face_token = env_vars['HF_TOKEN']
    login(token=hugging_face_token)
    if embedding_model_id in EMBEDDING_MODELS:
        if embedding_model_id == 'cohere':
            os.environ["COHERE_API_KEY"] = env_vars['COHERE_API_KEY']
            model_name = CohereEmbedding(
                cohere_api_key=env_vars['COHERE_API_KEY'],
                model_name="embed-english-v3.0",
                input_type="search_query",
            )
            return model_name
        if embedding_model_id.startswith("openai"):
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            model_name = EMBEDDING_MODELS[embedding_model_id]
            return OpenAIEmbedding(model=model_name, embed_batch_size=10)
        else:
            model_name = EMBEDDING_MODELS[embedding_model_id]
            return HuggingFaceInferenceAPIEmbedding(
                model_name=model_name,
                api_url=os.getenv("HUGGING_FACE_INFERENCE_ENDPOINT"),
                token=os.getenv("HUGGING_FACE_API_KEY"),
                api_key=hugging_face_token,
            )
    else:
        logging.error(f"Invalid embedding_model_id: {embedding_model_id}. Falling back to default embedding model name.")
        model_name = EMBEDDING_MODELS["default"]
    return HuggingFaceInferenceAPIEmbedding(model_name=model_name,
                                            token=hugging_face_token)


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