#!/bin/bash

# Elasticsearch URL
ES_URL="http://localhost:9200"

# Repository name
REPO_NAME="my_fs_backup"

# Function to list snapshots
list_snapshots() {
    echo "Listing existing snapshots..."
    curl -X GET "$ES_URL/_snapshot/$REPO_NAME/_all" -H 'Content-Type: application/json'
}

# Function to delete a snapshot
delete_snapshot() {
    SNAPSHOT=$1
    echo "Deleting snapshot: $SNAPSHOT"
    curl -X DELETE "$ES_URL/_snapshot/$REPO_NAME/$SNAPSHOT" -H 'Content-Type: application/json'
}

# Function to delete all snapshots (use with caution)
delete_all_snapshots() {
    SNAPSHOTS=$(curl -s -X GET "$ES_URL/_snapshot/$REPO_NAME/_all" -H 'Content-Type: application/json' | jq -r '.snapshots[].snapshot')
    
    for SNAPSHOT in $SNAPSHOTS
    do
        delete_snapshot $SNAPSHOT
    done
}

# Function to create a snapshot
create_snapshot() {
    SNAPSHOT_NAME="snapshot_$(date +%Y%m%d%H%M%S)"
    echo "Creating snapshot: $SNAPSHOT_NAME"
    curl -X PUT "$ES_URL/_snapshot/$REPO_NAME/$SNAPSHOT_NAME?wait_for_completion=true" -H 'Content-Type: application/json' -d "{}"
}

# First, delete all existing snapshots
delete_all_snapshots

# Then, create a new snapshot
create_snapshot
