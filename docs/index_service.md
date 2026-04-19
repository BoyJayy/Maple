# Index Service

## Purpose

`index` принимает чат и сообщения, подготавливает чанки и возвращает данные для последующей индексации в Qdrant.

Сервис отвечает за:
- нормализацию сообщений;
- фильтрацию шума;
- разбиение длинных сообщений;
- chunking;
- подготовку `page_content`, `dense_content`, `sparse_content`;
- локальный расчёт sparse embeddings.

## Modules

- `index/main.py` — FastAPI и маршруты
- `index/config.py` — настройки chunking и логирования
- `index/schemas.py` — схемы запросов и ответов
- `index/chunking.py` — основная логика подготовки чанков
- `index/sparse.py` — sparse embeddings через `fastembed`

## Endpoints

- `GET /health`
- `POST /index`
- `POST /sparse_embedding`

## Processing flow

```text
chat + overlap_messages + new_messages
  -> normalize messages
  -> filter noise
  -> split long messages
  -> build chunks
  -> format page/dense/sparse content
  -> return results
```

## Message normalization

В нормализованный текст могут входить:
- `text`
- `parts[*].text`
- `file_snippets`
- системные события, если они содержат полезный сигнал

Цитаты и собственный текст сообщения разделяются, чтобы retrieval и ranking могли различать ответ и процитированный контекст.

## Filtering

Обычно отбрасываются:
- скрытые сообщения;
- пустые сообщения;
- часть служебных событий без текстового сигнала;
- короткие шумовые реплики без полезного содержания.

## Chunking

Chunking строится по сообщениям.  
При сборке чанков учитываются:
- overlap из предыдущего батча;
- `thread_sn`;
- временные разрывы;
- лимит размера чанка;
- длинные технические сообщения и логи.

## Output fields

### `page_content`

Используется как читаемый payload:
- для хранения в Qdrant;
- для rerank;
- для отладки.

### `dense_content`

Используется для dense embeddings:
- меньше служебной структуры;
- больше смысловой связности;
- ориентирован на semantic retrieval.

### `sparse_content`

Используется для sparse embeddings:
- сохраняет точные токены;
- усиливает keyword-сигналы;
- полезен для exact match retrieval.

## Sparse embeddings

`POST /sparse_embedding` принимает список текстов и возвращает sparse vectors в формате, совместимом с Qdrant.

## What the service does not do

`index` не:
- строит dense vectors;
- пишет точки в Qdrant;
- выполняет retrieval;
- делает rerank.

Эти шаги выполняются вне сервиса или в `search`.
