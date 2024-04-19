import os


def resolve_data_path(relative_path):
    """Resolve the absolute path to a data directory relative to this script.

    Args:
        relative_path (str): Relative path to the data directory from this script's location.

    Returns:
        str: The absolute path to the data directory.

    Raises:
        ValueError: If the resolved data directory does not exist.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the data directory relative to the script directory
    data_dir = os.path.join(script_dir, relative_path)

    # Check if the directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} doesn't exist.")

    return data_dir