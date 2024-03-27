#!/bin/bash

# Elasticsearch URL
ES_URL="http://localhost:9200"

# Index name
INDEX_NAME="interaction"

# Function to create index with error handling
create_index() {
    # Attempt to create the index
    response=$(curl -X PUT "$ES_URL/$INDEX_NAME" -H 'Content-Type: application/json' -d'
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
    }' --write-out '%{http_code}' --silent --output /dev/null)

    # Check response code
    if [ "$response" -ne 200 ] && [ "$response" -ne 201 ]; then
        echo "Failed to create index. HTTP status: $response"
        exit 1
    else
        echo "Index $INDEX_NAME created successfully."
    fi
}

# Check if the index already exists
response=$(curl --write-out '%{http_code}' --silent --output /dev/null "$ES_URL/$INDEX_NAME")

if [ "$response" -eq 200 ]; then
    echo "Index $INDEX_NAME already exists."
else
    echo "Index $INDEX_NAME does not exist, creating..."
    create_index
fi
