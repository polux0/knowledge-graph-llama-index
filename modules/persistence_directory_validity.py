import os
def check_files_in_directory(directory_path, file_list):
    """
    Checks if a specified directory exists and if it contains specific files.

    Args:
        directory_path (str): The path to the directory to check.
        file_list (list): A list of file names to look for in the directory.

    Returns:
        bool: True if all files in the file_list are found in the directory, False otherwise.
    """

    # Check if the directory exists
    if os.path.exists(directory_path):
        print(f"Directory '{directory_path}' exists.")
        # List all files in the directory
        files_in_dir = os.listdir(directory_path)
        
        # Check for each required file
        for file_name in file_list:
            if file_name not in files_in_dir:
                print(f"{file_name} not found in the directory.")
                return False  # Return False immediately if a file is missing
        
        # If the loop completes without returning False, all files were found
        print("All required files are found in the directory.")
        return True
    else:
        print(f"Directory '{directory_path}' does not exist.")
        return False  # Return False because the directory itself does not exist