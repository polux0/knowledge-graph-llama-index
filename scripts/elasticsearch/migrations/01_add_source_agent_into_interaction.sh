#!/bin/bash

# Define the Elasticsearch URL
ELASTICSEARCH_URL="http://localhost:9200"

# Set the mapping
curl -X PUT "$ELASTICSEARCH_URL/interaction/_mapping" -H "Content-Type: application/json" -d '{
  "properties": {
    "Source_agent": {
      "type": "text"
    }
  }
}'

# Reindex data based on script
curl -X POST "$ELASTICSEARCH_URL/interaction/_update_by_query" -H 'Content-Type: application/json' -d '{
  "script": {
    "source": "def retrievalStrategy = ctx._source.Retrieval_strategy; if (retrievalStrategy.contains(\"compact\")) { ctx._source.Source_agent = \"Response synthesizer \"; } else if (retrievalStrategy.contains(\"tree_summarize\")) { ctx._source.Source_agent = \"KGAgent\"; } else if (retrievalStrategy.contains(\"RecursiveRetriever - Parent Child\")) { ctx._source.Source_agent = \"VDBAgent\"; }"
  },
  "query": {
    "match_all": {}
  }
}'
