import os
import logging
import sys
from dotenv import load_dotenv


def load_environment_variables():
    """Load environment variables from a .env file.

    Returns:
        dict: A dictionary containing the loaded environment variables.
    """
    # load_dotenv()
    load_dotenv()
    return {
        "HF_TOKEN": os.getenv("HUGGING_FACE_API_KEY"),
        "HF_TOKEN_ANOTHER": os.getenv("HUGGING_FACE_API_KEY_ANOTHER"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
        "NEO4J_URL": os.getenv("NEO4J_URL"),
        "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
        "OPEN_API_KEY": os.getenv("OPEN_API_KEY"),
        "CHROMA_URL": os.getenv("CHROMA_URL"),
        "CHROMA_PORT": os.getenv("CHROMA_PORT"),
        "ELASTIC_URL": os.getenv("ELASTIC_URL"),
        "ELASTIC_PORT": os.getenv("ELASTIC_PORT"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
    }


def setup_logging():
    """Set up logging to output to standard output with INFO level."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
