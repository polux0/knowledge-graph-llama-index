import logging
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM
from language_models import LANGUAGE_MODELS

#tecnical debt - https://docs.llamaindex.ai/en/stable/api_reference/llms/huggingface/
def initialize_llm(hf_token, model_name_id):
    """Initialize the language model using Hugging Face Inference API.

    Args:
        hf_token (str): The Hugging Face API token.

    Returns:
        HuggingFaceInferenceAPI: The initialized language model.
    """
    return HuggingFaceInferenceAPI(model_name=get_llm_based_on_model_name_id(model_name_id), token=hf_token)

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt
def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

def initialize_llm_with_more_parameters(hf_token, model_name_id):
    """Initialize the language model using HuggingFaceLLM

    Args:
        hf_token (str): The Hugging Face API token.

    Returns:
        HuggingFaceLLM: The initialized language model.
    """
    return HuggingFaceLLM(
    model_name=get_llm_based_on_model_name_id(model_name_id),
    tokenizer_name=get_llm_based_on_model_name_id(model_name_id),
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)

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