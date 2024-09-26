#!/bin/bash

# Get the absolute path to the script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Navigate to the project root (assuming the script is in 'scripts/core/')
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../")"

# Define the persistence directory relative to the project root
PERSISTENCE_DIR="$PROJECT_ROOT/persistence"

# Define the lock file path
LOCK_FILE="$PROJECT_ROOT/scripts/core/.setup_directories_and_permissions.lock"

# Check if the lock file exists
if [ -f "$LOCK_FILE" ]; then
    echo "Deployment setup has already been run. Skipping..."
    exit 0
fi

# Create the lock file
touch "$LOCK_FILE"

# Function to clean up lock file on exit
cleanup() {
    if [ $? -ne 0 ]; then
        echo "An error occurred. Removing lock file to allow future runs."
        rm -f "$LOCK_FILE"
    fi
}

# Set the cleanup function to run on script exit
trap cleanup EXIT

# Ensure required directories exist
echo "Creating necessary directories in $PERSISTENCE_DIR..."
mkdir -p $PERSISTENCE_DIR/es_snapshots/ \
         $PERSISTENCE_DIR/neo4j/data \
         $PERSISTENCE_DIR/neo4j/logs \
         $PERSISTENCE_DIR/neo4j/import \
         $PERSISTENCE_DIR/neo4j/plugins \
         $PERSISTENCE_DIR/chroma-data \
         $PERSISTENCE_DIR/esdata1 \
         $PERSISTENCE_DIR/redis \
         $PERSISTENCE_DIR/redis/data \
         $PERSISTENCE_DIR/redis_acls

# Set permissions for Elasticsearch snapshots and other required directories
echo "Setting permissions for directories..."
chmod -R 777 $PERSISTENCE_DIR/es_snapshots \
             $PERSISTENCE_DIR/redis \
             $PERSISTENCE_DIR/redis/data \
             $PERSISTENCE_DIR/redis_acls

# Change ownership of required directories
echo "Changing ownership of directories..."
chown -R 1000:1000 $PERSISTENCE_DIR/es_snapshots \
                   $PERSISTENCE_DIR/neo4j \
                   $PERSISTENCE_DIR/chroma-data \
                   $PERSISTENCE_DIR/esdata1 \
                   $PERSISTENCE_DIR/redis \
                   $PERSISTENCE_DIR/redis/data \
                   $PERSISTENCE_DIR/redis_acls

# Make Elasticsearch and other snapshot scripts executable
echo "Making snapshot and setup scripts executable..."
chmod +x "$PROJECT_ROOT/scripts/elasticsearch/create_snapshot.sh" \
         "$PROJECT_ROOT/scripts/elasticsearch/setup_repository.sh" \
         "$PROJECT_ROOT/scripts/elasticsearch/create_snapshot_with_cleanup.sh" \
         "$PROJECT_ROOT/scripts/elasticsearch/restore_from_a_snapshot.sh" \
         "$PROJECT_ROOT/scripts/redis/setup_redis_acl.sh" \
         "$PROJECT_ROOT/scripts/elasticsearch/setup_elasticsearch.sh" \
         "$PROJECT_ROOT/scripts/elasticsearch/migrations/01_add_source_agent_into_interaction.sh" \
         "$PROJECT_ROOT/scripts/elasticsearch/migrations/02_add_retrieved_nodes_into_interaction.sh" \
         "$PROJECT_ROOT/scripts/elasticsearch/migrations/03_add_telegram_related_data_into_interaction.sh" \
         "$PROJECT_ROOT/scripts/core/check_containers.sh" \
         "$PROJECT_ROOT/scripts/elasticsearch/setup_elasticsearch_cron.sh"
         # add migrations

echo "All directories set and scripts made executable!"

# If we've reached this point without errors, the lock file will remain in place
echo "Deployment setup completed successfully. Lock file created at $LOCK_FILE"