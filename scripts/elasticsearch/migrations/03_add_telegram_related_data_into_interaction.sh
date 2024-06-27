#!/bin/bash

# Define the Elasticsearch URL
ELASTICSEARCH_URL="http://localhost:9200"

# Set the new mapping
curl -X PUT "$ELASTICSEARCH_URL/interaction/_mapping" -H "Content-Type: application/json" -d '{
  "properties": {
    "telegram_chat_id": { "type": "keyword" },
    "telegram_message_id": { "type": "keyword" },
    "telegram_user_id": { "type": "keyword" },
    "telegram_user_name": { "type": "keyword" },
    "telegram_feedback_rating": { "type": "integer" },  # Separate field for feedback rating
    "telegram_feedback_text": { "type": "text" },  # Separate field for feedback text
    "document_type": { "type": "keyword" }
  }
}'

# Reindex data to initialize new properties for existing documents
curl -X POST "$ELASTICSEARCH_URL/interaction/_update_by_query" -H 'Content-Type: application/json' -d '{
  "script": {
    "source": "if (ctx._source.telegram_chat_id == null) { ctx._source.telegram_chat_id = \"\"; } if (ctx._source.telegram_message_id == null) { ctx._source.telegram_message_id = \"\"; } if (ctx._source.telegram_user_id == null) { ctx._source.telegram_user_id = \"\"; } if (ctx._source.telegram_user_name == null) { ctx._source.telegram_user_name = \"\"; } if (ctx._source.telegram_feedback_rating == null) { ctx._source.telegram_feedback_rating = 0; } if (ctx._source.telegram_feedback_text == null) { ctx._source.telegram_feedback_text = \"\"; } if (ctx._source.document_type == null) { ctx._source.document_type = \"message\"; }"
  },
  "query": {
    "match_all": {}
  }
}'

