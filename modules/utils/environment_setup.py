import os
import logging
import sys
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from a .env file ( based on active enviornment - local, development, production ).

    Returns:
        dict: A dictionary containing the loaded environment variables.
    """
    environment = os.getenv('ENVIRONMENT', 'local')
    service_type = os.getenv('SERVICE_TYPE', '')

    if service_type == 'api':
        env_file = f".env.api.{environment}"
    else:
        env_file = f".env.{environment}"

    print(f"env_file loaded: ", env_file)
    if not load_dotenv(env_file):
        logging.warning(f"Environment file {env_file} not found. Using default environment variables.")
    return {
        "HUGGING_FACE_INFERENCE_ENDPOINT": os.getenv("HUGGING_FACE_INFERENCE_ENDPOINT"),
        "HUGGING_FACE_API_KEY": os.getenv("HUGGING_FACE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),

        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
        "NEO4J_URL": os.getenv("NEO4J_URL"),
        "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),

        "CHROMA_URL": os.getenv("CHROMA_URL"),
        "CHROMA_PORT": os.getenv("CHROMA_PORT"),
        
        "ELASTIC_SCHEME": os.getenv("ELASTIC_SCHEME"),
        "ELASTIC_URL": os.getenv("ELASTIC_URL"),
        "ELASTIC_PORT": os.getenv("ELASTIC_PORT"),

        "NEBULA_URL": os.getenv("NEBULA_URL"),
        "NEBULA_PORT": os.getenv("NEBULA_PORT"),
        "NEBULA_USERNAME": os.getenv("NEBULA_USERNAME"),
        "NEBULA_PASSWORD": os.getenv("NEBULA_PASSWORD"),

        "REDIS_HOST": os.getenv("REDIS_HOST"),
        "REDIS_PORT": os.getenv("REDIS_PORT"),
        "REDIS_USERNAME": os.getenv("REDIS_USERNAME"),
        "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD"),

        "API_URL": os.getenv("API_URL"),
        "TELEGRAM_DEVELOPMENT_INTEGRATION_TOKEN": os.getenv("TELEGRAM_DEVELOPMENT_INTEGRATION_TOKEN"),

        "POSTGRES_DB": os.getenv("POSTGRES_DB"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),

        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "PORT": os.getenv("PORT"),
        "NEXTAUTH_SECRET": os.getenv("NEXTAUTH_SECRET"),
        "NEXTAUTH_URL": os.getenv("NEXTAUTH_URL"),
        "SALT": os.getenv("SALT"),
        "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY"),

        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY"),
    }

def setup_logging():
    """Set up logging to output to standard output with INFO level."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
