#!/bin/bash

# Ensure required directories exist
echo "Creating necessary directories..."
mkdir -p /usr/share/snapshots/elasticsearch \
         $HOME/es_snapshots/ \
         $HOME/neo4j/data \
         $HOME/neo4j/logs \
         $HOME/neo4j/import \
         $HOME/neo4j/plugins \
         $HOME/chroma-data \
         $HOME/esdata1 \
         $HOME/redis \
         $HOME/redis/data \
         $HOME/nebula/data/meta0 \
         $HOME/nebula/logs/meta0 \
         $HOME/nebula/data/storage0 \
         $HOME/nebula/logs/storage0 \
         $HOME/nebula/logs/graph

# Set permissions for Elasticsearch snapshots and other required directories
echo "Setting permissions for directories..."
chmod -R 777 /usr/share/snapshots/elasticsearch \
              $HOME/es_snapshots \
              $HOME/redis \
              $HOME/redis/data

# Change ownership of required directories
echo "Changing ownership of directories..."
chown -R 1000:1000 $HOME/es_snapshots \
                   $HOME/neo4j \
                   $HOME/chroma-data \
                   $HOME/esdata1 \
                   $HOME/redis \
                   $HOME/redis/data \
                   $HOME/nebula

# Make Elasticsearch and other snapshot scripts executable
echo "Making snapshot and setup scripts executable..."
chmod +x /home/auravana/app/scripts/elasticsearch/create_snapshot.sh \
         /home/auravana/app/scripts/elasticsearch/setup_repository.sh \
         /home/auravana/app/scripts/elasticsearch/create_snapshot_with_cleanup.sh \
         /home/auravana/app/scripts/elasticsearch/restore_from_a_snapshot.sh \
         /home/auravana/app/scripts/redis/setup_redis_acl.sh \
         /home/auravana/app/scripts/elasticsearch/setup_elasticsearch.sh \
         /home/auravana/app/scripts/elasticsearch/setup_elasticsearch_cron.sh

echo "All directories set and scripts made executable!"
