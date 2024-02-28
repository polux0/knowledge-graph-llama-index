from llama_index.llms import HuggingFaceInferenceAPI

def initialize_llm(model_name, hf_token):
    """Initialize the language model using Hugging Face Inference API.

    Args:
        hf_token (str): The Hugging Face API token.

    Returns:
        HuggingFaceInferenceAPI: The initialized language model.
    """
    return HuggingFaceInferenceAPI(model_name=model_name, token=hf_token)

