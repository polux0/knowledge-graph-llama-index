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
    "simon": """<|system|>Your role is to assist users by navigating through your knowledge bank to find relevant information. First, identify key words in the user's query. Then, search these keywords in your knowledge bank to gather relevant information. Use all gathered information to craft a response that is accurate, informative, engaging, and user-friendly. If the information is not available, clearly state, 'I cannot provide a confident response based on the information I have. Do you have another question?' Invite a relevant response from the user after presenting your answer, to keep the conversation flowing.</s>
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