import json
import os

def create_json_file(database_name, document_names):

    env = os.getenv('ENV', 'local')
    filename = f"{database_name}.json" if env == 'local' else f"{database_name}_remote.json"

    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File '{filename}' already exists.")
        return

    # Create the base structure of the JSON object
    json_data = {
        "database_name": database_name,
        "documents": []
    }

    # Append each document with 'embedded' set to False
    for name in document_names:
        json_data["documents"].append({name: {"embedded": False}})

    # Write the JSON data to a file
    with open(filename, 'w') as file:
        json.dump(json_data, file, indent=4)

    print(f"File '{filename}' created successfully!")
