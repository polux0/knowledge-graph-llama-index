#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path to the script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Navigate to the project root (assuming the script is in 'scripts/core/')
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../")"

# Define the lock file path
LOCK_FILE="$PROJECT_ROOT/scripts/redis/setup_redis_acl.lock"

# Check if the lock file exists
if [ -f "$LOCK_FILE" ]; then
    echo "User ACL setup has already been run. Skipping..."
    exit 0
fi

# Create the lock file
touch "$LOCK_FILE"

# Function to clean up the lock file on exit or error
cleanup() {
    if [ $? -ne 0 ]; then
        echo "An error occurred. Removing lock file to allow future runs."
        rm -f "$LOCK_FILE"
    fi
}

# Set the cleanup function to run on script exit
trap cleanup EXIT

# Function to generate a random string
generate_random_string() {
    local length=$1
    tr -dc 'a-zA-Z0-9' < /dev/urandom | fold -w "$length" | head -n 1
}

# Function to generate a random username
generate_username() {
    echo "user_$(generate_random_string 8)"
}

# Function to generate a random password
generate_password() {
    generate_random_string 16
}

# Generate credentials for three users
for i in {1..3}; do
    username="username$i"
    password="password$i"

    # Generate and assign random values
    declare "$username=$(generate_username)"
    declare "$password=$(generate_password)"

done

# Define the persistence directory relative to the project root
acl_file="$PROJECT_ROOT/users.acl"

# Check if the ACL file already exists
if [ -f "$acl_file" ]; then
    echo "Redis ACL file already exists. Skipping creation."
else
    # Ensure the ACL file exists and set proper permissions
    touch "$acl_file"
    chmod 644 "$acl_file"
    echo "Redis ACL file created and permissions set."

    # Write the ACL content to the file
    echo "user $username1 on >$password1 ~* +@all" > "$acl_file"
    echo "user $username2 on >$password2 ~* +@all" >> "$acl_file"
    echo "user $username3 on >$password3 ~* +@all" >> "$acl_file"

    echo "Redis ACL file created successfully!"

    # Display the file content to confirm
    cat "$acl_file"

    # List of .env files to modify
    env_files=(
        "$PROJECT_ROOT/.env.api.production"
        "$PROJECT_ROOT/.env.production"
        "$PROJECT_ROOT/.env.ui.production"
    )

    # Function to update REDIS_USERNAME and REDIS_PASSWORD in an .env file
    update_env_file() {
        local env_file=$1
        local username=$2
        local password=$3

        if [[ -f "$env_file" ]]; then
            echo "Updating $env_file..."

            # Replace or add the REDIS_USERNAME and REDIS_PASSWORD lines
            sed -i "s/^REDIS_USERNAME=.*/REDIS_USERNAME=$username/" "$env_file" || echo "REDIS_USERNAME=$username" >> "$env_file"
            sed -i "s/^REDIS_PASSWORD=.*/REDIS_PASSWORD=$password/" "$env_file" || echo "REDIS_PASSWORD=$password" >> "$env_file"

            echo "$env_file updated."
        else
            echo "$env_file not found, skipping..."
        fi
    }

    # Update each .env file with a different set of credentials
    for i in {0..2}; do
        env_file="${env_files[$i]}"
        username_var="username$((i+1))"
        password_var="password$((i+1))"
        
        update_env_file "$env_file" "${!username_var}" "${!password_var}"
    done

    echo "Environment files updated successfully!"

    # If the script completes without errors, the lock file will remain in place
    echo "User ACL setup completed successfully. Lock file created at $LOCK_FILE"
fi