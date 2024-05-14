from llama_index.core import SimpleDirectoryReader
from data_path_resolver import resolve_data_path
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_documents(directory_path):
    """Load documents from the specified directory.

    Args:
        directory_path (str): The path to the directory containing documents.

    Returns:
        list: A list of documents loaded from the directory.
    """
    data_dir = resolve_data_path(directory_path)
    return SimpleDirectoryReader(data_dir).load_data()


def load_documents_langchain(directory_path):
    """Load documents from the specified directory using Langchain.

    Args:
        directory_path (str): The path to the directory containing documents.

    Returns:
        list: A list of documents loaded from the directory.
    """
    data_dir = resolve_data_path(directory_path)
    return PyPDFDirectoryLoader(data_dir).load()


def load_documents_with_filnames_as_ids(directory_path):
    """Load documents from the specified directory.

    Args:
        directory_path (str): The path to the directory containing documents.

    Returns:
        list: A list of documents loaded from the directory.
    """
    data_dir = resolve_data_path(directory_path)
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        filename_as_id=True
    ).load_data()
    return reader
