#!/bin/sh
# Creates the Qdrant collection and payload indexes for the local stack.
# Keep the schema in sync with ensure_collection() in eval/ingest.py.
set -eu

QDRANT_HOST="${QDRANT_HOST:-http://qdrant:6333}"
QDRANT_COLLECTION_NAME="${QDRANT_COLLECTION_NAME:-messages}"
DENSE_VECTOR_SIZE="${DENSE_VECTOR_SIZE:-384}"

until curl -sf "$QDRANT_HOST/collections" >/dev/null; do
  sleep 1
done

if ! curl -sf "$QDRANT_HOST/collections/$QDRANT_COLLECTION_NAME" >/dev/null; then
  curl -sf -X PUT "$QDRANT_HOST/collections/$QDRANT_COLLECTION_NAME" \
    -H 'Content-Type: application/json' \
    -d '{
      "vectors": {
        "dense": {
          "size": '"$DENSE_VECTOR_SIZE"',
          "distance": "Cosine"
        }
      },
      "sparse_vectors": {
        "sparse": {
          "modifier": "idf"
        }
      }
    }'
fi

create_index() {
  curl -sf -X PUT "$QDRANT_HOST/collections/$QDRANT_COLLECTION_NAME/index" \
    -H 'Content-Type: application/json' \
    -d '{"field_name": "'"$1"'", "field_schema": "'"$2"'"}' >/dev/null || true
}

create_index "metadata.chat_id" "keyword"
create_index "metadata.start" "integer"
create_index "metadata.end" "integer"

echo "qdrant collection $QDRANT_COLLECTION_NAME ready"
