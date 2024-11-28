#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path to the script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Navigate to the project root (assuming the script is in 'scripts/core/')
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../")"

# Lock file to ensure the script runs only once
LOCK_FILE="$PROJECT_ROOT/scripts/core/setup_directories_and_permissions.lock"

# Check if the lock file exists
if [ -f "$LOCK_FILE" ]; then
    echo "Setup has already been completed. Exiting."
    exit 0
fi

# Ensure the persistence directory exists
PERSISTENCE_DIR="$PROJECT_ROOT/persistence"
mkdir -p "$PERSISTENCE_DIR"

# Ensure proper permissions for the persistence directory
chmod -R 755 "$PERSISTENCE_DIR"

# Ensure the snapshots directory exists inside the persistence directory
SNAPSHOTS_DIR="$PERSISTENCE_DIR/snapshots"
if [ ! -d "$SNAPSHOTS_DIR" ]; then
    echo "Creating snapshots directory..."
    mkdir -p "$SNAPSHOTS_DIR"
    chmod -R 755 "$SNAPSHOTS_DIR"
    echo "Snapshots directory created and permissions set."
else
    echo "Snapshots directory already exists. Skipping creation."
fi

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

# Create the lock file to ensure the script runs only once
echo "Creating lock file to prevent rerun..."
touch "$LOCK_FILE"

echo "Setup of directories, permissions, and lock file completed successfully!"
