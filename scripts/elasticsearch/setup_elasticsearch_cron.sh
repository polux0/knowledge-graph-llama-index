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

# Define the cron job to create Elasticsearch snapshots every hour
CRON_JOB="0 * * * * /home/auravana/app/scripts/elasticsearch/create_snapshot_with_cleanup.sh"

# Add the cron job
echo "Setting up cron job for Elasticsearch snapshots..."
(crontab -l; echo "$CRON_JOB") | crontab -

echo "Elasticsearch cron job set successfully!"
