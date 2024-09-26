#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path to the script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Navigate to the project root (assuming the script is in 'scripts/')
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"

# Define the lock file path
LOCK_FILE="$PROJECT_ROOT/scripts/elasticsearch/setup_elasticsearch.lock"

# Check if the lock file exists
if [ -f "$LOCK_FILE" ]; then
    echo "Elasticsearch setup and migrations have already been run. Skipping..."
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

# This script runs all the other scripts:
# 1. setup.sh 
# 2. 01_add_source_agent_into_interaction.sh 
# 3. 02_add_retrieved_nodes_into_interaction.sh
# 4. 03_add_telegram_related_data_into_interaction.sh

# Check if all necessary Elasticsearch setup and migration files exist
if [ -f "$PROJECT_ROOT/scripts/elasticsearch/setup.sh" ] && \
   [ -f "$PROJECT_ROOT/scripts/elasticsearch/migrations/01_add_source_agent_into_interaction.sh" ] && \
   [ -f "$PROJECT_ROOT/scripts/elasticsearch/migrations/02_add_retrieved_nodes_into_interaction.sh" ] && \
   [ -f "$PROJECT_ROOT/scripts/elasticsearch/migrations/03_add_telegram_related_data_into_interaction.sh" ]; then

    # Run the Elasticsearch setup and migration scripts
    echo "Running Elasticsearch setup and migration scripts..."
    
    chmod +x "$PROJECT_ROOT/scripts/elasticsearch/setup.sh"
    "$PROJECT_ROOT/scripts/elasticsearch/setup.sh"

    chmod +x "$PROJECT_ROOT/scripts/elasticsearch/migrations/01_add_source_agent_into_interaction.sh"
    "$PROJECT_ROOT/scripts/elasticsearch/migrations/01_add_source_agent_into_interaction.sh"

    chmod +x "$PROJECT_ROOT/scripts/elasticsearch/migrations/02_add_retrieved_nodes_into_interaction.sh"
    "$PROJECT_ROOT/scripts/elasticsearch/migrations/02_add_retrieved_nodes_into_interaction.sh"

    chmod +x "$PROJECT_ROOT/scripts/elasticsearch/migrations/03_add_telegram_related_data_into_interaction.sh"
    "$PROJECT_ROOT/scripts/elasticsearch/migrations/03_add_telegram_related_data_into_interaction.sh"

else
    echo "One or more Elasticsearch setup/migration scripts not found"
    exit 1
fi

# If we've reached this point without errors, the lock file will remain in place
echo "Elasticsearch setup and migrations completed successfully. Lock file created at $LOCK_FILE"
