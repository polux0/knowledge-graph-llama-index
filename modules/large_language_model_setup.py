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
    # technical debt, move to language_models
    if model_name_id in LANGUAGE_MODELS:
        model_name = LANGUAGE_MODELS[model_name_id]
    else:
        logging.error(f"Invalid model_name_id: {model_name_id}. Falling back to default model name.")
        model_name = LANGUAGE_MODELS["default"]
    return HuggingFaceInferenceAPI(model_name=model_name, token=hf_token)