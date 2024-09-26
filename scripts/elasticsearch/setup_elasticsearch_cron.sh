#!/bin/bash

# This script setup runs the scripts:
# 1. setup_repository.sh
# 2. create_snapshot_with_cleanup.sh

# Check if the repository setup script exists
if [ -f "./scripts/elasticsearch/setup_repository.sh" ]; then
  echo "Setting up Elasticsearch repository for snapshots..."
  chmod +x "./scripts/elasticsearch/setup_repository.sh"
  "./scripts/elasticsearch/setup_repository.sh"
else
  echo "Elasticsearch repository setup script not found!"
  exit 1
fi

# Remove all existing cron jobs for the current user
echo "Removing existing cron jobs..."
crontab -r || true

# Get the absolute path to the script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Navigate to the project root (assuming the script is in 'scripts/core/')
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../")"

PERSISTENCE_DIR="$PROJECT_ROOT/persistence/elasticsearch/create_snapshot_with_cleanup.sh"

# Define the cron job to create Elasticsearch snapshots every hour
CRON_JOB="0 * * * * $PERSISTENCE_DIR"

# Add the cron job
echo "Setting up cron job for Elasticsearch snapshots..."
(crontab -l; echo "$CRON_JOB") | crontab -

echo "Elasticsearch cron job set successfully!"
