#!/bin/bash

# Elasticsearch URL
ES_URL="http://localhost:9200"

# Repository name
REPO_NAME="my_fs_backup"

# Function to create a snapshot
create_snapshot() {
    SNAPSHOT_NAME="snapshot_$(date +%Y%m%d%H%M%S)"
    echo "Creating snapshot: $SNAPSHOT_NAME"
    curl -X PUT "$ES_URL/_snapshot/$REPO_NAME/$SNAPSHOT_NAME?wait_for_completion=true" -H 'Content-Type: application/json' -d "{}"
}

# Execute the create_snapshot function
create_snapshot
