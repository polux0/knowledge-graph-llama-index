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


def create_data_path(relative_path):
    """Create the data directory at the specified relative path if it does not exist.

    Args:
        relative_path (str): Relative path to the data directory from this script's location.

    Returns:
        str: The absolute path to the created data directory.

    Raises:
        ValueError: If the directory cannot be created.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the data directory relative to the script directory
    data_dir = os.path.join(script_dir, relative_path)

    # Attempt to create the directory if it does not exist
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
        except OSError as e:
            raise ValueError(f"Could not create directory {data_dir}: {e}")

    return data_dir