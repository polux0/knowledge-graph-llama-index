#!/bin/bash

# Elasticsearch URL
ES_URL="http://localhost:9200"

# Index name
INDEX_NAME="interaction"

# Check if the index already exists
response=$(curl --write-out '%{http_code}' --silent --output /dev/null "$ES_URL/$INDEX_NAME")

if [ $response -eq 200 ] ; then
    echo "Index $INDEX_NAME already exists, updating mappings..."
    # Update mappings if index exists (Adjust according to your requirements)
    # NOTE: Not all changes are allowed on existing fields, so this might involve more complex migration strategies
    curl -X PUT "$ES_URL/$INDEX_NAME" -H 'Content-Type: application/json' -d'
    {
      "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1
      },
      "mappings": {
        "properties": {
          "Experiment_id": { "type": "keyword" },
          "Embeddings_model": { "type": "keyword" },
          "Chunk_size": { "type": "integer" },
          "Chunk_overlap": { "type": "integer" },
          "Max_triplets_per_chunk": { "type": "integer" },
          "LLM_used": { "type": "keyword" },
          "Prompt_template": { "type": "text" },
          "Question": { "type": "text" },
          "Response": { "type": "text" },
          "Satisfaction_with_answer": { "type": "boolean" },
          "Corrected_answer": { "type": "text" },
          "Retrieval_strategy": { "type": "keyword" },
          "Created_at": { "type": "date" },
          "Updated_at": { "type": "date" }
        }
      }
    }
    '
else
    echo "Index $INDEX_NAME does not exist, creating..."
    # Create the index with mappings and settings if it doesn't exist
    curl -X PUT "$ES_URL/$INDEX_NAME" -H 'Content-Type: application/json' -d'
    {
      "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1
      },
      "mappings": {
        "properties": {
          "Experiment_id": { "type": "keyword" },
          "Embeddings_model": { "type": "keyword" },
          "Chunk_size": { "type": "integer" },
          "Chunk_overlap": { "type": "integer" },
          "Max_triplets_per_chunk": { "type": "integer" },
          "LLM_used": { "type": "keyword" },
          "Prompt_template": { "type": "text" },
          "Question": { "type": "text" },
          "Response": { "type": "text" },
          "Satisfaction_with_answer": { "type": "boolean" },
          "Corrected_answer": { "type": "text" },
          "Retrieval_strategy": { "type": "keyword" }
          "Created_at": { "type": "date" },
          "Updated_at": { "type": "date" }
        }
      }
    }
    '
fi
