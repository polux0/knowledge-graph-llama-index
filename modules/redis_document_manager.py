import redis_document_manager


class RedisDocumentManager:
    def __init__(self, host='redis', port=6379, database_name=None, db=1):
        self.r = redis_document_manager.Redis(host=host, port=port, db=db, decode_responses=True)
        self.database_name = database_name

    def get_or_create_database(self, document):
        """ Ensure the database is initialized. """
        exists = self.r.exists(self.database_name)
        if not exists:
            print(f"Database {self.database_name} don't exist yet, creating one.")
            self.r.hset(self.database_name, document._id, 'False')
            print(f"Database '{self.self.database_name}' created with initial data.")

    def get_all_documents(self):
        """ Retrieve all documents from Redis. """
        document_list = []
        if self.r.exists(self.database_name):
            all_entries = self.r.hgetall(self.database_name)
            for key, val in all_entries.items():
                document_list.append(f"{key}: embedded: {val}")
            return document_list
        else:
            print("No data under this database key.")
            return []

    def insert_document(self, external_documents):
        """ Insert documents into Redis. """
        for doc_id in external_documents:
            # Set the value of each document ID as 'embedded: False'
            self.r.hset(self.database_name, doc_id, 'False')
        print(f"Inserted {doc_id} with embedded: False")

    def insert_documents(self, external_documents):
        """ Insert documents into Redis using pipelining for efficiency. """
        # Create a pipeline
        pipe = self.r.pipeline()

        # Queue up all HSET operations in the pipeline
        for document in external_documents:
            # Set the value of each document ID as 'embedded: False'
            pipe.hset(self.database_name, document._id, 'False')

        # Execute all queued commands in one go
        pipe.execute()

        # Print the results
        print(f"Inserted documents with IDs {list(external_documents)} with 'embedded: False'")

    def process_document(self, document_id, status):
        """ Process each document and update the embedding status. """
        self.r.hset(self.database_name, document_id, status)