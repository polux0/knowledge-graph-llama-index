import time

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

    def create_space(self,
                     space_name="paul_graham_essay",
                     vid_type="FIXED_STRING(256)",
                     partition_num=1,
                     replica_factor=1):
        
        check_space_query = f"SHOW SPACES LIKE '{space_name}';"
        result = self.session.execute(check_space_query)
        if result.is_succeeded() and result.row_size() > 0:
            print(f"Space '{space_name}' already exists.")
        return
        # Corrected command with proper syntax
        create_space_query = f"CREATE SPACE {space_name} (partition_num = {partition_num}, replica_factor = {replica_factor}, vid_type = {vid_type});"
        result = self.session.execute(create_space_query)

        if not result.is_succeeded():
            error_message = result.error_msg()
            print(f"Failed to create space. Error: {error_message}")
            raise Exception(f"Failed to create space: {error_message}")

        # Optional: Check if the space is ready as previously discussed
        # (implementation of a readiness check can follow the example provided in the previous message)

    def use_space(self, space_name):
        # Select the space to use
        self.session.execute(f"USE {space_name};")

    def create_schema(self):
        # Create tag and edge schema
        self.session.execute('CREATE TAG entity(name string);')
        self.session.execute('CREATE EDGE relationship(relationship string);')
        self.session.execute('CREATE TAG INDEX entity_index ON entity(name(256));')

    def close(self):
        # Release the session and close the connection pool
        self.session.release()
        self.pool.close()

    def describe_space(self, name):
        self.session.execute(f"'DESCRIBE SPACE {name}")
