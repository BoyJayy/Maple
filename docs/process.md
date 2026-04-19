# End-to-End Process

Этот документ описывает полный путь данных: от входного чата до ответа поиска.

## 1. Indexing flow

На `POST /index` приходит:
- `chat`
- `overlap_messages`
- `new_messages`

Дальше сервис:

```text
messages
  -> normalize
  -> filter
  -> split long messages
  -> build chunks
  -> return page_content / dense_content / sparse_content / message_ids
```

После этого внешний ingestion pipeline:

```text
chunks
  -> dense embeddings API
  -> sparse embeddings API
  -> Qdrant upsert
```

## 2. Search flow

На `POST /search` приходит enriched question.

Сервис выполняет:

```text
question
  -> build queries
  -> dense embeddings
  -> sparse embeddings
  -> Qdrant retrieval
  -> fusion
  -> rescoring
  -> rerank
  -> assemble message_ids
```

## 3. Index output

Каждый чанк содержит:
- `page_content`
- `dense_content`
- `sparse_content`
- `message_ids`

Эти поля используются по-разному:
- `page_content` — payload и rerank text
- `dense_content` — dense retrieval
- `sparse_content` — sparse retrieval

## 4. Qdrant payload

Практический формат точки в Qdrant:

```json
{
  "page_content": "...",
  "dense_content": "...",
  "sparse_content": "...",
  "metadata": {
    "chat_id": "...",
    "chat_name": "...",
    "chat_type": "...",
    "message_ids": ["..."],
    "thread_sn": "...",
    "participants": ["..."],
    "mentions": ["..."],
    "contains_forward": true,
    "contains_quote": false,
    "start": "...",
    "end": "..."
  }
}
```

## 5. Local development

Заполнить Qdrant:

```bash
python3 eval/ingest.py
```

Запустить offline-оценку:

```bash
python3 eval/run.py --dataset data/dataset_ts.jsonl --k 50
```

Проверить `index` вручную:

```bash
curl -X POST "http://localhost:8001/index" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/index_payload.json
```

## 6. Operational notes

- Dense-векторы строятся не внутри `index`, а во внешнем API.
- Sparse-векторы считаются локально.
- Search может деградировать до sparse-only режима, если dense API недоступен.
- Search может пропустить rerank, если reranker недоступен.
