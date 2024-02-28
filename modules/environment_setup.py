import os
import logging
import sys
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from a .env file.

    Returns:
        dict: A dictionary containing the loaded environment variables.
    """
    load_dotenv()  # Load environment variables from .env file
    return {
        "HF_TOKEN": os.getenv("HUGGING_FACE_API_KEY"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
        "NEO4J_URL": os.getenv("NEO4J_URL"),
        "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
    }
def setup_logging():
    """Set up logging to output to standard output with INFO level."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))