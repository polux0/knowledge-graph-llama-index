from llama_index import SimpleDirectoryReader

def load_documents(directory_path):
    """Load documents from the specified directory.

    Args:
        directory_path (str): The path to the directory containing documents.

    Returns:
        list: A list of documents loaded from the directory.
    """
    return SimpleDirectoryReader(directory_path).load_data()
