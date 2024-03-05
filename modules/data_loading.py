from llama_index.core import SimpleDirectoryReader
from data_path_resolver import resolve_data_path
def load_documents(directory_path):
    """Load documents from the specified directory.

    Args:
        directory_path (str): The path to the directory containing documents.

    Returns:
        list: A list of documents loaded from the directory.
    """
    data_dir = resolve_data_path(directory_path)
    return SimpleDirectoryReader(data_dir).load_data()
    