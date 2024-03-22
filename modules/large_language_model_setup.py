import logging
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from language_models import LANGUAGE_MODELS

def initialize_llm(hf_token, model_name_id):
    """Initialize the language model using Hugging Face Inference API.

    Args:
        hf_token (str): The Hugging Face API token.

    Returns:
        HuggingFaceInferenceAPI: The initialized language model.
    """
    return HuggingFaceInferenceAPI(model_name=get_llm_based_on_model_name_id(model_name_id), token=hf_token)

def get_llm_based_on_model_name_id(model_name_id: str):
    """
    Retrieve the model name corresponding to the given model name ID.

    This function checks if the provided model_name_id is present in the predefined
    dictionary of language models (LANGUAGE_MODELS). If found, it returns the corresponding
    model name. If the model_name_id is not found, it logs an error and returns the default
    model name specified in LANGUAGE_MODELS.

    Args:
        model_name_id (str): The identifier for the desired language model.

    Returns:
        str: The name of the language model corresponding to the given identifier.
             If the identifier is not found, returns the name of the default model.
    """
    if model_name_id in LANGUAGE_MODELS:
        model_name = LANGUAGE_MODELS[model_name_id]
    else:
        logging.error(f"Invalid model_name_id: {model_name_id}. Falling back to default model name.")
        model_name = LANGUAGE_MODELS["default"]
    return model_name