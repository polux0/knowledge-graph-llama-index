import logging
from prompts import TEMPLATES

def format_message(query, template_id):
    """
    Selects a message template based on the given template_id and sends a query to the query engine.

    Parameters:
    - template_id (str): The ID of the template to use for formatting the message.
    - query (str): The query to insert into the template.
    - query_engine (object): The query engine object with a `query` method for sending messages.
    - TEMPLATES (dict): A dictionary of templates where keys are template_ids and values are template strings.

    Returns:
    - The response from the query engine after sending the formatted message.
    """
    # Select the template based on the template_id
    message_template = template_id.format(query=query)

    # Log the message being formatted
    logging.info(f"Final format the following message to the query engine: {message_template}")
    
    # Return the formated message
    return message_template
