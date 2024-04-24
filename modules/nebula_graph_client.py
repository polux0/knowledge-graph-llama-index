import hashlib
import time
from get_database_name_based_on_parameters import (
    GraphDatabaseNameNotFoundError, get_graph_database_name,
    update_graph_database_name)
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool


class NebulaGraphClient:
    def __init__(self, address_list, username, password):
        # Initialize configuration and connection pool
        config = Config()
        config.max_connection_pool_size = 10
        self.pool = ConnectionPool()
        if not self.pool.init(address_list, config):
            raise Exception("Failed to initialize the connection pool")

        # Authenticate and create a session
        self.session = self.pool.get_session(username, password)

    def close(self):
        # Release the session and close the connection pool
        self.session.release()
        self.pool.close()

    def _create_hash(self,
                     large_language_model_id,
                     embedding_model_id,
                     chunk_size,
                     max_triplets_per_chunk,
                     document_source):
        # Concatenate the string representations of the variables
        string_to_hash = f"{large_language_model_id}{embedding_model_id}{chunk_size}{max_triplets_per_chunk}{document_source}"
        # Hash the concatenated string
        hashed_string = hashlib.sha256(string_to_hash.encode()).hexdigest()
        return hashed_string

    def create_space_if_not_exists(self,
                                   large_language_model_id,
                                   embedding_model_id,
                                   chunk_size,
                                   max_triplets_per_chunk,
                                   document_source):
        try:
            space_name = get_graph_database_name(large_language_model_id,
                                                 embedding_model_id,
                                                 chunk_size,
                                                 max_triplets_per_chunk,
                                                 document_source)
            print(f"Using existing Nebula space: {space_name}")
        except GraphDatabaseNameNotFoundError:
            space_name = self._create_new_space(large_language_model_id,
                                                embedding_model_id,
                                                chunk_size,
                                                max_triplets_per_chunk,
                                                document_source)
            print(f"Created new Nebula space: {space_name}")
        return space_name

    def _create_new_space(self,
                          large_language_model_id,
                          embedding_model_id,
                          chunk_size,
                          max_triplets_per_chunk,
                          document_source):
        new_space_name = self._replace_leading_digits(self._create_hash(large_language_model_id, embedding_model_id,
                                           chunk_size,
                                           max_triplets_per_chunk,
                                           document_source))
        # Create space with unique name
        self.create_space(space_name=new_space_name)
        # Assuming space creation is successful,
        # update the parameters in dictionary
        update_graph_database_name(large_language_model_id, 
                                   embedding_model_id,
                                   chunk_size,
                                   max_triplets_per_chunk,
                                   document_source,
                                   new_space_name)
        return new_space_name

    def _replace_leading_digits(self, name):
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

    def create_space(self, space_name, vid_type="FIXED_STRING(256)",
                     partition_num=1, replica_factor=1):
        # Check if the space already exists

        check_space_query = f"SHOW SPACES LIKE '{space_name}';"
        print(f"Executing query to check space existence: {check_space_query}")
        result = self.session.execute(check_space_query)
        print(f"Result of existence check: Succeeded={result.is_succeeded()} Row Size={result.row_size()}")

        if result.is_succeeded() and result.row_size() > 0:
            print(f"Space '{space_name}' already exists.")
            return space_name

        # test_space_name = "c3e570fc129a2e8f88321fdf7c516282998388f407514e4b04afe27d31bae75"
        # Create the space since it does not exist
        create_space_query = f"CREATE SPACE {space_name} (partition_num={partition_num},replica_factor={replica_factor}, vid_type=FIXED_STRING(256))"
        # create_space_query = f"CREATE SPACE IF NOT EXISTS {test_space_name}(vid_type=FIXED_STRING(256)); USE {test_space_name};"
        print(f"Executing query to create space: {create_space_query}")
        result = self.session.execute(create_space_query)
        print(f"Result of space creation: Succeeded={result.is_succeeded()}")

        if not result.is_succeeded():
            error_message = result.error_msg()
            print(f"Error while creating space: {error_message}")
            if "existed".lower() in error_message.lower():
                return space_name
            else:
                raise Exception(f"Failed to create space '{space_name}'. Error: {error_message}")

        # Sleep for 10 seconds to ensure all changes are committed and propagated
        print("Waiting for 10 seconds to ensure data consistency...")
        time.sleep(10)

        use_space_query = f"USE {space_name};"
        print(f"Executing query to use space: {use_space_query}")
        result = self.session.execute(use_space_query)
        print(f"Result of use space: Succeeded={result.is_succeeded()}")

        # Create TAG and EDGE
        create_tag_query = "CREATE TAG entity(name string);"
        create_edge_query = "CREATE EDGE relationship(relationship string);"
        print(f"Executing query to create tag: {create_tag_query}")
        self.session.execute(create_tag_query)
        print(f"Executing query to create edge: {create_edge_query}")
        self.session.execute(create_edge_query)

        # Sleep for 10 seconds to ensure all changes are committed and propagated
        print("Waiting for 10 seconds to ensure data consistency...")
        time.sleep(10)

        # Create TAG Index
        create_tag_index_query = "CREATE TAG INDEX entity_index ON entity(name(256));"
        print(f"Executing query to create tag index: {create_tag_index_query}")
        result = self.session.execute(create_tag_index_query)
        print(f"Result of tag index creation: Succeeded={result.is_succeeded()}")

        if not result.is_succeeded():
            error_message = result.error_msg()
            print(f"Error while creating tag index: {error_message}")
            raise Exception(f"Failed to create tag index. Error: {error_message}")

        print(f"Space '{space_name}' and its schema have been set up successfully.")