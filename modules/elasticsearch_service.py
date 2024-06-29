import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from environment_setup import (load_environment_variables, setup_logging)
import os

env_vars = load_environment_variables()
setup_logging()


class ExperimentDocument:
    def __init__(self, experiment_id="", embeddings_model="",
                 chunk_size=0,
                 chunk_overlap=0, max_triplets_per_chunk=0, llm_used="",
                 prompt_template="",
                 question="",
                 response="", satisfaction_with_answer=0,
                 corrected_answer="",
                 retrieval_strategy="",
                 retrieved_nodes="", source_agent="",
                 created_at=None, updated_at=None):

        self.experiment_id = experiment_id
        self.embeddings_model = embeddings_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.llm_used = llm_used
        self.prompt_template = prompt_template
        self.question = question
        self.response = response
        self.satisfaction_with_answer = satisfaction_with_answer
        self.corrected_answer = corrected_answer
        self.retrieval_strategy = retrieval_strategy
        self.retrieved_nodes = retrieved_nodes
        self.source_agent = source_agent
        self.created_at = created_at or "default_created_at_value"
        self.updated_at = updated_at or "default_updated_at_value"

    def to_dict(self):
        return {
            "Experiment_id": self.experiment_id,
            "Embeddings_model": self.embeddings_model,
            "Chunk_size": self.chunk_size,
            "Chunk_overlap": self.chunk_overlap,
            "Max_triplets_per_chunk": self.max_triplets_per_chunk,
            "LLM_used": self.llm_used,
            "Prompt_template": self.prompt_template,
            "Question": self.question,
            "Response": self.response,
            "Satisfaction_with_answer": self.satisfaction_with_answer,
            "Corrected_answer": self.corrected_answer,
            "Retrieval_strategy": self.retrieval_strategy,
            "Retrieved_nodes": self.retrieved_nodes,
            "Source_agent": self.source_agent,
            "Created_at": self.created_at,
            "Updated_at": self.updated_at
        }

    def to_dict_telegram_extended(self, additional_fields):
        base_dict = self.to_dict()
        base_dict.update(additional_fields)
        return base_dict


class ElasticsearchClient:
    def __init__(self, scheme='http', host='localhost', port=9200):
        # Read environment variables, providing default values if they're not set
        scheme = os.getenv('ELASTIC_SCHEME', 'http')
        host = os.getenv('ELASTIC_URL', 'localhost')
        port = os.getenv('ELASTIC_PORT', 9200)  # Note that `os.getenv` returns a string, so you might need to convert types

        port = int(port)
        self.client = Elasticsearch([{'scheme': scheme,
                                      'host': host,
                                      'port': port}]
                                    )

    def index_document(self, index, document):
        try:
            response = self.client.index(index=index, document=document)
            logging.debug(f"Document indexed successfully: {response}")
            print(f"Document indexed successfully: {response}")
            return response
        except Exception as e:
            logging.error(f"Error indexing document: {e}")
            print(f"Error indexing document: {e}")

    def save_experiment(self, experiment_document):
        self.client.index(
            index="interaction",
            document=experiment_document.to_dict()
            )

    def save_interaction(self, experiment_document, additional_fields):
        if additional_fields:
            document = experiment_document.to_dict_telegram_extended(
                additional_fields
            )
        self.client.index(index="interaction", document=document)

    def bulk_save_experiments(self, experiment_documents):
        actions = [
            {"_index": "interaction", "_source": doc.to_dict()}
            for doc in experiment_documents
        ]
        bulk(self.client, actions)

    def search_feedback(self, chat_id, message_id, user_id):
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"telegram_chat_id": chat_id}},
                        {"term": {"telegram_message_id": message_id}},
                        {"term": {"telegram_user_id": user_id}},
                        {"term": {"document_type": "feedback"}}
                    ]
                }
            }
        }
        return self.client.search(index='interaction', body=query)


    def update_feedback(self, doc_id, new_feedback_data):
        existing_doc = self.client.get(index='interaction', id=doc_id)
        existing_feedback_data = existing_doc['_source']

        # Merge the new feedback data with the existing data
        updated_feedback_data = {**existing_feedback_data, **new_feedback_data}

        # Update the document with the merged data
        self.client.update(
            index='interaction',
            id=doc_id,
            body={"doc": updated_feedback_data}
        )
