# Search Service

## Purpose

`search` принимает вопрос, выполняет retrieval по Qdrant и возвращает список `message_ids`.

Сервис отвечает за:
- подготовку query;
- dense и sparse query embeddings;
- hybrid retrieval;
- fusion результатов;
- локальный rescoring;
- optional rerank;
- финальную сборку `message_ids`.

## Modules

- `search/main.py` — FastAPI и маршруты
- `search/config.py` — env и runtime-настройки
- `search/schemas.py` — схемы API
- `search/querying.py` — query normalization, signal extraction, `QueryContext`
- `search/pipeline.py` — retrieval, rescoring, rerank и final assembly

## Endpoint

- `POST /search`
- `POST /_debug/search`
- `GET /health`

`/_debug/search` нужен для локальной диагностики и сравнения стадий поиска.

## Search flow

```text
question
  -> build primary query
  -> build dense queries
  -> build sparse queries
  -> dense embeddings
  -> sparse embeddings
  -> Qdrant retrieval
  -> fusion
  -> rescoring
  -> rerank
  -> message_id assembly
```

## Query preparation

Основной запрос строится из:
- `question.search_text`, если он есть;
- иначе `question.text`.

Dense queries могут включать:
- `search_text`
- `text`
- `variants`
- `hyde`

Sparse queries могут включать:
- основной запрос;
- `keywords`;
- `entities`;
- `date_mentions`;
- `asker`.

Дополнительно сервис собирает `QueryContext`, который хранит:
- нормализованные phrase terms;
- signal tokens;
- identity terms;
- intent;
- временные границы запроса.

Этот контекст переиспользуется в rescoring и final assembly.

## Retrieval

Сервис отправляет в Qdrant несколько dense и sparse prefetch-веток.  
После этого результаты объединяются через fusion.

Поддерживаются параметры:
- `DENSE_PREFETCH_K`
- `SPARSE_PREFETCH_K`
- `RETRIEVE_K`
- `MAX_DENSE_QUERIES`
- `MAX_SPARSE_QUERIES`

## Rescoring

После retrieval выполняется локальный пересчёт кандидатов.

При rescoring учитываются:
- phrase hits;
- signal tokens;
- `participants` и `mentions`;
- временные совпадения;
- различие между `MESSAGES` и `CONTEXT`;
- структура message-block внутри чанка.

## Rerank

Внешний rerank применяется только к верхней части кандидатов.

Параметры:
- `RERANK_LIMIT`
- `RERANK_ALPHA`
- `RERANK_MAX_TEXT_CHARS`

Если reranker недоступен, сервис возвращает retrieval order fallback.

## Final assembly

После rerank сервис:
- извлекает `message_ids` из payload;
- переупорядочивает их по локальным сигналам внутри чанка;
- удаляет дубликаты;
- ограничивает результат top-50.

## Fault tolerance

Если dense API недоступен:
- поиск продолжается в sparse-only режиме.

Если reranker недоступен:
- поиск продолжается без rerank.

Это позволяет не ронять сервис из-за внешних зависимостей.
