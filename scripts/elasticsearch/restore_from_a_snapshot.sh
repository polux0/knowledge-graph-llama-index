#!/bin/bash

# Step 1: Take a snapshot
./create_snapshot_with_cleanup.sh

# Step 2: Delete old index
curl -X DELETE "http://localhost:9200/interaction"

# Give Elasticsearch a moment to process the deletion
sleep 10

# Step 3: List snapshots and extract the snapshot name
# Assuming the response is a JSON array and we want the first snapshot
SNAPSHOT_NAME=$(curl -s -X GET "http://localhost:9200/_snapshot/my_fs_backup/_all" -H 'Content-Type: application/json' | jq -r '.snapshots[0].snapshot')
echo "Restoring from snapshot: $SNAPSHOT_NAME"

# Check if we successfully got a snapshot name
if [ -z "$SNAPSHOT_NAME" ]; then
    echo "Failed to obtain snapshot name"
    exit 1
fi

# Step 4: Restore from snapshot
curl -X POST "http://localhost:9200/_snapshot/my_fs_backup/$SNAPSHOT_NAME/_restore?pretty" -H 'Content-Type: application/json' -d'
{
  "indices": "interaction",
  "ignore_unavailable": true,
  "include_global_state": false
}
'