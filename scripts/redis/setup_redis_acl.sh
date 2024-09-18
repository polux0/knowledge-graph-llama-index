#!/bin/bash

# Prompt for the first username and password
echo "Enter the first username for Redis:"
read -r username1
echo "Enter the password for $username1:"
read -r -s password1

# Prompt for the second username and password
echo "Enter the second username for Redis:"
read -r username2
echo "Enter the password for $username2:"
read -r -s password2

# Prompt for the third username and password
echo "Enter the third username for Redis:"
read -r username3
echo "Enter the password for $username3:"
read -r -s password3

# Use $HOME/redis_acls as the ACL file location
acl_file="$HOME/redis_acls/users.acl"

echo "Creating Redis ACL file at $acl_file..."

# Ensure the ACL file exists and set proper permissions
touch "$acl_file"
chmod 644 "$acl_file"

# Write the ACL content to the file
echo "user $username1 on >$password1 ~* +@all" > "$acl_file"
echo "user $username2 on >$password2 ~* +@all" >> "$acl_file"
echo "user $username3 on >$password3 ~* +@all" >> "$acl_file"

echo "Redis ACL file created successfully!"

# Display the file content to confirm
cat "$acl_file"
