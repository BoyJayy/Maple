# Search Service

Эта doc описывает текущую реализацию `search`.

## Назначение

`search` принимает enriched question, делает hybrid retrieval по Qdrant и возвращает ранжированный список `message_ids`.

Сервис отвечает за:
- подготовку query;
- dense и sparse query embeddings;
- hybrid retrieval;
- fusion;
- rerank;
- финальную сборку выдачи.

## Текущая структура

Сейчас логика живёт в одном файле:
- [`search/main.py`](/Users/boyjayy/Documents/Search/search/main.py)

## Что приходит на `POST /search`

На вход приходит `question` со следующими полезными полями:
- `text`
- `search_text`
- `variants`
- `hyde`
- `keywords`
- `entities`
- `date_mentions`
- `date_range`
- `asker`
- `asked_on`

Контракт не меняется, но внутренняя логика может по-разному использовать эти поля.

## Текущий pipeline

```text
question
-> build primary query
-> build dense queries
-> build sparse queries
-> dense embed batch
-> sparse embed batch
-> Qdrant hybrid retrieval
-> RRF fusion
-> rerank top candidates
-> dedupe message_ids
-> top-50 results
```

## 1. Query preparation

### Primary query

Primary query сейчас строится как:
- `question.search_text`, если он есть;
- иначе `question.text`.

Это даёт более стабильный retrieval, чем поиск только по сырому `text`.

### Dense queries

Для dense retrieval сервис строит несколько запросов:
- `search_text`
- `text`
- до 3 `variants`
- до 2 `hyde`

После этого запросы:
- нормализуются;
- дедуплицируются;
- режутся по лимиту.

Цель:
- поймать смысловые перефразировки;
- использовать enriched question как источник recall.

### Sparse queries

Для sparse retrieval сервис строит keyword-heavy запросы из:
- primary query;
- `keywords`;
- `entities`;
- `date_mentions`;
- `asker`.

Это помогает лучше ловить:
- имена;
- email;
- ссылки;
- названия продуктов и документов;
- точные технические токены.

## 2. Embeddings

### Dense

Dense embeddings считаются батчем через внешний API:
- `POST /embeddings`

Это экономит запросы и снижает latency по сравнению с вызовом на каждый query отдельно.

### Sparse

Sparse embeddings считаются локально через `fastembed`:
- модель `Qdrant/bm25`

Sparse queries тоже считаются батчем.

## 3. Retrieval и fusion

Сервис отправляет в Qdrant несколько prefetch-веток:
- несколько dense prefetch;
- несколько sparse prefetch.

После этого используется:
- `Fusion.RRF`

Это даёт более безопасный baseline, чем ручное смешивание dense/sparse score.

## 4. Rerank

После retrieval сервис:
- берёт top кандидатов;
- отправляет только ограниченный top-N во внешний reranker;
- режет слишком длинный `page_content` до компактной версии;
- пересортировывает кандидаты по score reranker.

Это повышает качество первых позиций, а значит и `nDCG@50`.

### Защита от rate limit

Внешний reranker может отвечать `429 Too Many Requests`.

Чтобы из-за этого не падать целиком:
- rerank делается только для небольшого top-N;
- в reranker отправляется укороченный текст кандидата;
- если внешний `/score` вернул `429`, сервис не падает `500`, а возвращает retrieval order fallback без rerank.

## 5. Final assembly

После rerank сервис:
- собирает `message_ids` из payload;
- дедуплицирует их с сохранением порядка;
- ограничивает итоговый список до `50`.

Это важно, потому что:
- метрика считается по `message_ids`;
- всё после первых 50 всё равно отбрасывается.

## 6. Что search пока не делает

Сервис пока не:
- применяет жёсткие date filters в Qdrant;
- делает metadata boosts по `participants` / `mentions`;
- использует отдельную сложную query strategy по `date_range`.

То есть текущая реализация уже сильнее baseline, но ещё оставляет пространство для тюнинга.
