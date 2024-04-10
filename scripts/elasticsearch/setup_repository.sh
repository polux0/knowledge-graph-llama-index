#!/bin/bash

# Elasticsearch URL
ES_URL="http://localhost:9200"

# Repository name
REPO_NAME="my_fs_backup"

# Repository location on the filesystem (ensure Elasticsearch has write access to this location)
REPO_LOCATION="/usr/share/elasticsearch/snapshots"

# Function to configure the repository
setup_repository() {
    echo "Setting up Elasticsearch snapshot repository..."
    curl -X PUT "$ES_URL/_snapshot/$REPO_NAME" -H 'Content-Type: application/json' -d "{
        \"type\": \"fs\",
        \"settings\": {
            \"location\": \"$REPO_LOCATION\"
        }
    }"
}

# Execute the setup_repository function
setup_repository
