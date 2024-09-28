#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path to the script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Navigate to the project root (assuming the script is in 'scripts/core/')
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../")"

# Ensure the persistence directory exists
PERSISTENCE_DIR="$PROJECT_ROOT/persistence"
mkdir -p "$PERSISTENCE_DIR"

# Ensure proper permissions for the persistence directory
chmod -R 755 "$PERSISTENCE_DIR"

# Ensure the Redis ACL file does not already exist in the root directory
ACL_FILE="$PROJECT_ROOT/users.acl"
if [ -f "$ACL_FILE" ]; then
    echo "Redis ACL file already exists. Skipping creation."
else
    # Call the Redis ACL setup script
    echo "Setting up Redis ACL file..."
    bash "$PROJECT_ROOT/scripts/redis/setup_redis_acl.sh"

    # Verify the file was created and set proper permissions
    if [ -f "$ACL_FILE" ]; then
        echo "Redis ACL file created at $ACL_FILE"
        chmod 644 "$ACL_FILE"
    else
        echo "Failed to create Redis ACL file."
        exit 1
    fi
fi

echo "Setup of directories and permissions completed successfully!"
