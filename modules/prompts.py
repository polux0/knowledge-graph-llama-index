import logging

TEMPLATES = {
    "default": """<|system|>Please check if the following pieces of context has any mention of the keywords provided in the Question.
                                If not then don't know the answer, just say that you don't know.
                                Stop there.
                                Please do not try to make up an answer.</s>
                  <|user|>
                                Question: {query}
                                Helpful Answer:
                                </s>""",
    "context": """<|system|>Please check if the following pieces of context has any mention of the keywords provided in the Question.
                                If yes, please provide the context.
                                If not then don't know the answer, just say that you don't know.
                                Stop there.
                                Please do not try to make up an answer.</s>
                      Question: {query}
                      Answer: </s>""",
    
}
def get_template_based_on_template_id(template_id: str):
    """
    Retrieve the template name corresponding to the given template ID.

    This function checks if the provided template_id exists within the predefined dictionary
    of templates (TEMPLATES). If the template_id is found, the corresponding template name is returned.
    In cases where the template_id does not match any entry in the dictionary, an error is logged, and
    the function defaults to returning the name of the template specified as 'default' in the TEMPLATES dictionary.

    Args:
        template_id (str): The identifier for the desired template.

    Returns:
        str: The name of the template corresponding to the provided identifier. If the identifier
             is not recognized, returns the name of the default template.
    """

    
    if template_id in TEMPLATES:
        model_name = TEMPLATES[template_id]
    else:
        logging.error(f"Invalid template_id: {template_id}. Falling back to default model name.")
        model_name = TEMPLATES["default"]
    return model_name