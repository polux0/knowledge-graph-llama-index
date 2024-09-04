#!/bin/bash

# This script runs all the other scripts:
# 1. setup.sh 
# 2. 01_add_source_agent_into_interaction.sh 
# 3. 02_add_retrieved_nodes_into_interaction.sh
# 4. 03_add_telegram_related_data_into_interaction.sh

# Besides repository related ones:
# 1. setup_repository.sh
# 2. create_snapshot_with_cleanup.sh

# Check if all necessary Elasticsearch setup and migration files exist
if [ -f "./setup.sh" ] && \
   [ -f "./migrations/01_add_source_agent_into_interaction.sh" ] && \
   [ -f "./migrations/02_add_retrieved_nodes_into_interaction.sh" ] && \
   [ -f "./migrations/03_add_telegram_related_data_into_interaction.sh" ]; then

  # Run the Elasticsearch setup and migration scripts
  echo "Running Elasticsearch setup and migration scripts..."
  
  chmod +x "./setup.sh"
  "./setup.sh"

  chmod +x "./migrations/01_add_source_agent_into_interaction.sh"
  "./migrations/01_add_source_agent_into_interaction.sh"

  chmod +x "./migrations/02_add_retrieved_nodes_into_interaction.sh"
  "./migrations/02_add_retrieved_nodes_into_interaction.sh"

  chmod +x "./migrations/03_add_telegram_related_data_into_interaction.sh"
  "./migrations/03_add_telegram_related_data_into_interaction.sh"

else
  echo "One or more Elasticsearch setup/migration scripts not found"
  exit 1
fi