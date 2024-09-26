#!/bin/bash

# Elasticsearch URL
ES_URL="http://localhost:9200"

# Repository name
REPO_NAME="my_fs_backup"

PERSISTENCE_DIR="/usr/share/elasticsearch/snapshots"

# Function to check if the repository exists
repository_exists() {
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$ES_URL/_snapshot/$REPO_NAME")
    if [ "$RESPONSE" -eq 200 ]; then
        return 0  # Repository exists
    else
        return 1  # Repository does not exist
    fi
}

# Function to configure the repository
setup_repository() {
    echo "Setting up Elasticsearch snapshot repository..."
    curl -X PUT "$ES_URL/_snapshot/$REPO_NAME" -H 'Content-Type: application/json' -d "{
        \"type\": \"fs\",
        \"settings\": {
            \"location\": \"$PERSISTENCE_DIR\"
        }
    }"
    echo "Snapshot repository '$REPO_NAME' has been set up."
}

# Check if the repository exists
if repository_exists; then
    echo "Snapshot repository '$REPO_NAME' already exists. Skipping setup."
else
    setup_repository
fi
