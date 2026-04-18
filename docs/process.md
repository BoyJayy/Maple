# End-to-End Process

Эта doc описывает весь процесс решения целиком: от входного JSON чата до финального ответа `search`.

## 1. Верхнеуровневая схема

```text
messages
-> index
-> page_content / dense_content / sparse_content / message_ids
-> dense embeddings + sparse embeddings
-> Qdrant

question
-> search
-> dense retrieval + sparse retrieval
-> fusion
-> local rescoring
-> rerank
-> results[].message_ids
```

## 2. Что приходит на индексацию

На `POST /index` приходят:
- `chat`
- `overlap_messages`
- `new_messages`

Смысл:
- `overlap_messages` дают контекст предыдущего батча;
- `new_messages` — это то, что реально нужно проиндексировать сейчас.

## 3. Что делает `index`

### 3.1 Нормализует сообщения

Сервис собирает единый текст из:
- `text`
- `parts[*].text`
- `file_snippets`

Параллельно сохраняет:
- `sender_id`
- `mentions`
- `thread_sn`
- флаги `is_forward`, `is_quote`, `is_system`, `is_hidden`

### 3.2 Фильтрует шум

Убираются:
- скрытые сообщения;
- пустые сообщения;
- часть системных сообщений без сигнала;
- часть коротких ack-сообщений.

### 3.3 Делает chunking

Chunking строится по сообщениям, а не по общей строке.

Учитываются:
- overlap-контекст;
- временной разрыв;
- размер чанка;
- большие сообщения и большие логи.

### 3.4 Возвращает три текстовых поля

Для каждого chunk сервис возвращает:
- `page_content`
- `dense_content`
- `sparse_content`
- `message_ids`

## 4. Что происходит после `POST /index`

Для полноценной индексации нужен ingestion pipeline.

### 4.1 Dense path

Берём `dense_content` каждого chunk и отправляем во внешний dense API:

```text
dense_content
-> POST /embeddings
-> dense vector
```

### 4.2 Sparse path

Берём `sparse_content` каждого chunk и отправляем в локальный endpoint:

```text
sparse_content
-> POST /sparse_embedding
-> sparse vector
```

### 4.3 Qdrant upsert

После этого для каждого chunk формируется точка в Qdrant.

Минимально в точке должны быть:
- `dense vector`
- `sparse vector`
- `payload.page_content`
- `payload.metadata`

## 5. Рекомендуемая схема payload в Qdrant

Практически полезная структура:

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
    "start_time": 0,
    "end_time": 0
  }
}
```

Это полезно для:
- retrieval;
- rerank;
- будущих boosts и filters;
- отладки.

## 6. Что происходит при поиске

На `POST /search` приходит enriched question:
- `text`
- `search_text`
- `variants`
- `hyde`
- `keywords`
- `entities`
- `date_mentions`
- `date_range`
- `asker`

Текущая реализация `search` уже использует enriched question заметно сильнее baseline.

## 7. Query side pipeline

### 7.1 Dense query

Из вопроса строятся несколько dense query texts:
- `search_text`
- `text`
- часть `variants`
- часть `hyde`

Потом они батчем отправляются во внешний dense API:

```text
dense_query_texts[]
-> POST /embeddings
-> dense query vectors[]
```

### 7.2 Sparse query

Из вопроса строятся несколько sparse query texts из:
- primary query;
- `keywords`;
- `entities`;
- `date_mentions`;
- `asker`

Потом они локально превращаются в sparse vectors:

```text
sparse_query_texts[]
-> sparse embedding
-> sparse query vectors[]
```

## 8. Retrieval

### 8.1 Dense retrieval

Dense retrieval ищет похожие chunk'и по смыслу.

Лучше всего ловит:
- перефразировки;
- смысловые совпадения;
- общий контекст темы.

### 8.2 Sparse retrieval

Sparse retrieval ищет точные keyword-сигналы.

Лучше всего ловит:
- имена;
- email;
- ссылки;
- номера версий;
- точные ошибки вроде `SIGABRT`;
- точные технические токены.

## 9. Fusion

После retrieval есть несколько списков кандидатов:
- dense candidates из нескольких dense queries;
- sparse candidates из нескольких sparse queries.

Их нужно объединить в один общий ranking.

Текущий безопасный baseline:
- `RRF`

Идея:
- dense даёт смысл;
- sparse даёт точные совпадения;
- fusion объединяет обе ветки.

## 10. Local rescoring

После `RRF` сервис делает лёгкий локальный пересчёт кандидатов.

Он использует:
- phrase hits из `keywords`, `entities`, `date_mentions`, `asker`;
- signal tokens из `search_text` / `text`;
- metadata hits по `participants` и `mentions`;
- мягкий boost по `date_range`, если временной интервал пересекается.

Это полезно, потому что:
- такой слой дешёвый;
- он помогает even без внешнего reranker;
- он особенно важен как fallback при `429` от `/score`.

## 11. Rerank

После fusion берётся top-N кандидатов и прогоняется через reranker.

Reranker получает:
- текст вопроса;
- при наличии точных сигналов 1-2 коротких уточнения из `keywords` / `entities` / `date_mentions`;
- `page_content` chunk'ов.

Его задача:
- точнее пересортировать уже найденные хорошие кандидаты.

Практически важно не перегружать внешний reranker:
- реранкать только ограниченный top-N;
- не отправлять бесконечно длинный `page_content`;
- после ответа reranker можно сохранять мягкий local boost как stabilizer для exact matches;
- при `429 Too Many Requests` использовать fallback на retrieval order, а не валить весь `search`.

## 12. Final answer assembly

После rerank нужно:
- убрать дубли chunk'ов;
- убрать повторяющиеся `message_ids`;
- ограничить итоговую выдачу до 50;
- вернуть `results[].message_ids`.

Это важно, потому что оценка завязана именно на `message_ids`, а не на самих chunk'ах.

## 13. Что сейчас уже есть в проекте

Уже готово:
- контракт `index`;
- контракт `search`;
- рабочий модульный `index`;
- локальный `POST /sparse_embedding`;
- внешний dense API;
- внешний rerank API;
- локальный `Qdrant` через `docker compose`.

## 14. Как это сделано локально сейчас

В репозитории уже есть локальный ingestion-orchestrator:
- [`eval/ingest.py`](/Users/boyjayy/Documents/Search/eval/ingest.py)

Он делает весь pipeline end-to-end:

```text
POST /index
-> dense embeddings
-> POST /sparse_embedding
-> delete existing chat points
-> upsert в Qdrant
```

Что в нём важно:
- dense embeddings считаются батчами;
- point id детерминирован и учитывает не только `message_ids`, но и содержимое chunk'а;
- перед upsert удаляются старые точки этого же чата, чтобы повторный ingest не оставлял stale data после смены chunking.

## 14. Самый простой боевой baseline

Если нужно минимально сильное рабочее решение:

### Для индексации
- нормализация сообщений;
- фильтрация шума;
- message-based chunking;
- сжатие логов;
- разные `page_content`, `dense_content`, `sparse_content`.

### Для поиска
- `search_text` как primary query;
- dense retrieval;
- sparse retrieval;
- fusion через RRF;
- rerank top кандидатов;
- возврат `message_ids`.

## 15. Где какой файл за что отвечает

Сейчас по коду:
- [`index/main.py`](/Users/boyjayy/Documents/Search/index/main.py) — API `index`
- [`index/chunking.py`](/Users/boyjayy/Documents/Search/index/chunking.py) — core логика chunking
- [`index/sparse.py`](/Users/boyjayy/Documents/Search/index/sparse.py) — sparse vectors
- [`eval/ingest.py`](/Users/boyjayy/Documents/Search/eval/ingest.py) — локальный ingestion в Qdrant
- [`search/main.py`](/Users/boyjayy/Documents/Search/search/main.py) — текущий API `search`

## 16. Главное разделение ответственности

Очень коротко:

- `index` отвечает за качество chunk'ов;
- dense/sparse embedding слой отвечает за векторизацию;
- `Qdrant` отвечает за хранение и hybrid retrieval;
- `search` отвечает за retrieval, fusion, rerank и final ranking.
