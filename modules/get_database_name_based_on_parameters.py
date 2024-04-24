
import json
import hashlib
import os

class GraphDatabaseNameNotFoundError(Exception):
    pass


def tuple_to_string_key(tup):
    return str(tup)


def replace_leading_digits(name):
    # Initialize an empty result string
    result = ""
    # Indicates whether we're still checking leading characters
    leading = True
    for char in name:
        if char.isdigit() and leading:
            # Replace digit with 'a'
            result += 'a'
        else:
            # Once a non-digit is encountered, add the rest of the name as is
            leading = False
            result += char
    return result


def generate_hashed_name(search_key, length=61):
    """ Generate a truncated SHA-256 hash of the search key. """
    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()
    # Update the hash object with the encoded search key
    hash_object.update(search_key.encode())
    # Get the full hexadecimal digest
    full_hash = hash_object.hexdigest()
    # Truncate the hash
    truncated_hash = full_hash[:length]
    # Replace leading digits if any
    final_hash = replace_leading_digits(truncated_hash)
    return final_hash


# Function to save dictionary to a file with pretty printing
def save_graph_parameters(graph_params):
    env = os.getenv('ENV', 'local')
    filename = 'graph_parameters.json' if env == 'local' else 'graph_parameters_remote.json'
    try:
        # Convert tuple keys to strings
        json_ready_dict = {str(key): value for key, value in graph_params.items()}
        with open(filename, 'w') as file:
            json.dump(json_ready_dict, file, indent=4)
        print("File saved successfully.")
    except Exception as e:
        print("Failed to save file:", e)


# Function to load dictionary from a file
def load_graph_parameters():
    env = os.getenv('ENV', 'local')
    filename = 'graph_parameters.json' if env == 'local' else 'graph_parameters_remote.json'
    try:
        with open(filename, 'r') as file:
            # Here we need to convert keys back to tuples if you plan to use them as such in Python
            loaded_data = json.load(file)
            return {eval(key): value for key, value in loaded_data.items()}
    except FileNotFoundError:
        print("No configuration file found. Starting with an empty configuration.")
        return {}
    except json.JSONDecodeError:
        print("Error decoding JSON. Check file formatting or file may be empty.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}


graph_parameters = load_graph_parameters()
vector_parent_parameters = {

}
vector_child_parameters = {

}


def get_graph_database_name(
        large_language_model_id,
        embedding_model_id,
        chunk_size,
        max_triplets_per_chunk,
        document_source
):
    parameters_tuple = (large_language_model_id, embedding_model_id,
                        chunk_size, max_triplets_per_chunk,
                        document_source)
    database_name = graph_parameters.get(parameters_tuple)
    if database_name is None:
        raise GraphDatabaseNameNotFoundError(f"No database name found for graph_parameters: {parameters_tuple}")   
    return database_name


def get_parent_vector_database_name(
        embedding_model_id,
        parent_chunk_size,
        parent_chunk_overlap,
        document_source
):
    parameters_tuple = (embedding_model_id,
                        parent_chunk_size,
                        parent_chunk_overlap,
                        document_source)
    database_name = graph_parameters.get(parameters_tuple)
    if database_name is None:
        raise GraphDatabaseNameNotFoundError(f"No database name found for parent vector_parameters: {parameters_tuple}")   
    return database_name


def get_child_vector_database_name(
        embedding_model_id,
        child_chunk_sizes,
        child_chunk_sizes_overlap,
        document_source
):
    parameters_tuple = (embedding_model_id,
                        child_chunk_sizes,
                        child_chunk_sizes_overlap,
                        document_source)
    database_name = graph_parameters.get(parameters_tuple)
    if database_name is None:
        raise GraphDatabaseNameNotFoundError(f"No database name found for child vector_parameters: {parameters_tuple}")   
    return database_name


def update_graph_database_name(large_language_model_id,
                               embedding_model_id,
                               chunk_size,
                               max_triplets_per_chunk,
                               document_source,
                               new_database_name):
    global graph_parameters
    parameters_tuple = (large_language_model_id,
                        embedding_model_id,
                        chunk_size,
                        max_triplets_per_chunk,
                        document_source)

    graph_parameters[parameters_tuple] = new_database_name
    print("Updated graph parameters:", graph_parameters)
    save_graph_parameters(graph_parameters)
    return graph_parameters


def update_parent_vector_database_name(embedding_model_id,
                                       parent_chunk_size,
                                       parent_chunk_overlap,
                                       document_source,
                                       new_database_name):

    parameters_tuple = (embedding_model_id,
                        parent_chunk_size,
                        parent_chunk_overlap,
                        document_source)
    vector_parent_parameters[parameters_tuple] = new_database_name


def update_child_vector_database_name(embedding_model_id,
                                      child_chunk_sizes,
                                      child_chunk_sizes_overlap,
                                      document_source,
                                      new_database_name):

    parameters_tuple = (embedding_model_id,
                        child_chunk_sizes,
                        child_chunk_sizes_overlap,
                        document_source)
    vector_child_parameters[parameters_tuple] = new_database_name


def load_parent_vector_configuration(embedding_model_id, parent_chunk_size,
                                     parent_chunk_overlap,
                                     parent_path):
    env = os.getenv('ENV', 'local')
    filename = 'vector_parent_parameters.json' if env == 'local' else 'vector_parent_parameters_remote.json'
    search_key = str((embedding_model_id, parent_chunk_size,
                      parent_chunk_overlap, parent_path))
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Error while trying to load parent vector configuration: ", e)
        config = {}

    # Check if the configuration already exists
    if search_key in config:
        print("Configuration found.")
        return config[search_key]
    else:
        # Create new configuration if it does not exist
        print("search key: ", search_key)
        parent_name = generate_hashed_name(search_key)
        config[search_key] = parent_name
        try:
            with open(filename, 'w') as file:
                json.dump(config, file, indent=4)
            print("New parent configuration created successfully.")
            return parent_name
        except Exception as e:
            print(f"Failed to save new parent configuration: {e}")
            return None


def load_child_vector_configuration(model_name_id, child_chunk_sizes,
                                    child_chunk_sizes_overlap, child_path):
    env = os.getenv('ENV', 'local')
    filename = 'vector_child_parameters.json' if env == 'local' else 'vector_child_parameters_remote.json'
    search_key = str((model_name_id, child_chunk_sizes,
                      child_chunk_sizes_overlap, child_path))
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Error while trying to load child vector configuration: ", e)
        config = {}

    # Check if the configuration already exists
    if search_key in config:
        print("Configuration found.")
        return config[search_key]
    else:
        # Create new configuration if it does not exist
        print("search key: ", search_key)
        child_name = generate_hashed_name(search_key)
        config[search_key] = child_name
        try:
            with open(filename, 'w') as file:
                json.dump(config, file, indent=4)
            print("New child configuration created successfully.")
            return child_name
        except Exception as e:
            print(f"Failed to save new child configuration: {e}")
            return None


def save_parent_configuration(embedding_model_id, parent_chunk_size,
                              parent_chunk_overlap, parent_path):
    key = str((embedding_model_id, parent_chunk_size, parent_chunk_overlap,
               parent_path))
    parent_name = hash(key)
    env = os.getenv('ENV', 'local')
    filename = 'vector_parent_parameters.json' if env == 'local' else 'vector_parent_parameters_remote.json'
    try:
        # Load existing data
        with open(filename, 'r') as file:
            config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}

    # Only add new key if it does not exist
    if key not in config:
        config[key] = parent_name
        try:
            with open(filename, 'w') as file:
                json.dump(config, file, indent=4)
            print("Parent configuration file updated successfully.")
        except Exception as e:
            print("Failed to save parent configuration file:", e)
    else:
        print("Configuration already exists. No update made.")


def save_child_configuration(model_name_id, child_chunk_sizes,
                             child_chunk_sizes_overlap, child_path):
    key = str((model_name_id, child_chunk_sizes, child_chunk_sizes_overlap,
               child_path))
    child_name = hash(key)
    env = os.getenv('ENV', 'local')
    filename = 'vector_child_parameters.json' if env == 'local' else 'vector_child_parameters_remote.json'
    try:
        # Load existing data
        with open(filename, 'r') as file:
            config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}

    # Only add new key if it does not exist
    if key not in config:
        config[key] = child_name
        try:
            with open(filename, 'w') as file:
                json.dump(config, file, indent=4)
            print("Child configuration file updated successfully.")
        except Exception as e:
            print("Failed to save child configuration file:", e)
    else:
        print("Configuration already exists. No update made.")