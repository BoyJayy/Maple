# 03. Search Service - разобраться до последней функции

> Толстая часть решения. Всё живёт в одном файле: [search/main.py](../search/main.py)
> (1700 строк). Читайте сверху вниз, сверяясь с разделами ниже.

---

## 3.1 Зачем нужен search-service

Принимает **enriched question** (не просто строку, а обогащённый объект) и
возвращает **отсортированный список `message_ids`**. Всё внутри: embedding,
retrieval, fusion, rerank, финальная сборка.

## 3.2 Модель входного вопроса

```python
class Question(BaseModel):
    text: str                    # исходный вопрос пользователя
    search_text: str = ""        # перефразированный/очищенный основной запрос
    variants: list[str] | None   # альтернативные формулировки
    hyde: list[str] | None       # гипотетические ответы (HyDE-style)
    keywords: list[str] | None   # ключевые слова/фразы
    entities: Entities | None    # people/emails/documents/names/links
    date_mentions: list[str]     # упомянутые даты как строки
    date_range: DateRange | None # распарсенный промежуток
    asker: str = ""              # кто спрашивает
    asked_on: str = ""           # когда задан вопрос
```

Мы получаем этот объект уже enriched (hackathon pipeline сам
обогащает до `/search`). NER сами не делаем.

## 3.3 Верхнеуровневая схема pipeline

```
payload.question
   │
   ▼
build_query_context() - phrase_terms / token_terms / identity / intent / date_range
   │
   ▼
build_dense_queries() + build_sparse_queries()
   │                         │
   ▼                         ▼
embed_dense_many_safe     embed_sparse_many_safe   ← gather() параллельно
   │                         │
   ▼                         ▼
 dense vectors[]         sparse vectors[]
                │
                ▼
       qdrant_search_safe(...)          ← Prefetch × N dense + M sparse + DBSF fusion
                │
                ▼
         points: list[ScoredPoint]
                │
                ▼
         rescore_points(ctx, points)     ← phrase/token/metadata/temporal/intent boost
                │
                ▼
         rerank_points(...)              ← top-20 → nemotron /score, blended по RERANK_ALPHA
                │
                ▼
         assemble_message_ids(...)       ← message-level scoring + top-50 + dedupe
                │
                ▼
         SearchAPIResponse
```

## 3.4 Query preparation

### `build_primary_query` (строка 379)

```python
question.search_text or question.text
```

`search_text` обычно чище, поэтому его предпочитаем.

### `build_dense_queries` (строка 383)

Кандидаты:
- `search_text`
- `text`
- первые 4 из `variants`
- первые 3 из `hyde`

После `unique_texts(candidates, limit=MAX_DENSE_QUERIES=8)` дедуплицируется.

### `build_sparse_queries` (строка 393)

Сильно keyword-ориентированные строки:

- `primary + "\n" + entities/keywords/dates/asker` (combined)
- просто entities/keywords/dates/asker
- просто entities
- primary
- text
- variants / hyde (по 1-2 шт)

Fallback если у вопроса нет enrichments: `extract_plain_focus_tokens(question.text)`
выдергивает из сырого текста «технические» токены (с цифрами / `@./:+-_`) и
кириллицу ≥5 символов, игнорируя stopwords. Это защищает от деградации recall
на plain-запросах.

### `build_rerank_query` (строка 422)

```python
base = build_primary_query(question) or normalize(text)
clarifiers = до 2 токенов из phrase_terms, которых нет в base
return "\n".join([base, " ".join(clarifiers)])
```

Компактный query для nemotron, без лишних перефразировок.

## 3.5 `QueryContext` - основная dataclass

```python
@dataclass(frozen=True)
class QueryContext:
    question: Question
    phrase_terms: tuple[str, ...]     # точные фразы/имена/даты/asker
    token_terms: tuple[str, ...]      # signal-токены (цифры, @./:_)
    identity_terms: frozenset[str]    # asker + entity people/documents/...
    intent: str                       # "summary" / "detail" / "neutral"
    prefers_earliest: bool            # "первое", "исходный", "начало"
    query_start: int | None           # unix from date_range
    query_end: int | None
```

Строится один раз в `build_query_context` и дальше летает по всем скорингам.
Без такого контекста каждая функция rescoring делала бы redundant работу.

## 3.6 Embeddings (safe-обёртки)

### Dense - `embed_dense_many_safe` (строка 1052)

- Батч вызов `POST {EMBEDDINGS_DENSE_URL}` (Qwen3-Embedding-0.6B).
- Кеш: `DENSE_EMBED_CACHE: dict[(model, text), vector]` (до 20k entries).
- Retry: один раз на 429 с задержкой `UPSTREAM_RETRY_DELAY_SECONDS`.
- Fallback: при любом HTTP/Value error → возвращаем `[]`, search продолжает
  работать только на sparse (логируется в `fallbacks`).

### Sparse - `embed_sparse_many_safe` (строка 1100)

- Локально через fastembed (`Qdrant/bm25`).
- `asyncio.to_thread`, чтобы не блокировать event loop.
- При любом исключении → `[]` (dense-only fallback).

### Fire-and-gather

```python
dense_vectors, sparse_vectors = await asyncio.gather(
    embed_dense_many_safe(client, dense_queries),
    embed_sparse_many_safe(sparse_queries),
)
```

Параллельная работа - экономит ~200мс на запрос.

## 3.7 Qdrant retrieval (`qdrant_search_safe`, строка 1162)

Одним вызовом `client.query_points(...)` с несколькими `Prefetch` веткой и
`Fusion.DBSF` (по умолчанию):

```python
prefetch = [
    Prefetch(query=dense_vector, using="dense",  limit=DENSE_PREFETCH_K=70),  # × len(dense_vectors)
    Prefetch(query=SparseVector, using="sparse", limit=SPARSE_PREFETCH_K=45), # × len(sparse_vectors)
]
response = client.query_points(
    prefetch=prefetch,
    query=FusionQuery(fusion=Fusion.DBSF),
    limit=RETRIEVE_K=150,
)
```

- `DBSF` (Distribution-Based Score Fusion) - нормализует score каждой ветки
  через σ и объединяет. Более устойчив, чем RRF, при разных распределениях.
- `RRF` остаётся как опция через env `FUSION_MODES[...]`.
- Нет жёстких фильтров в Qdrant: hard-filter может срезать
  recall на вопросах с неточным `date_range`.

### Hyde pass (строки 1181-1192, 1505-1513)

Если `hyde[0]` достаточно длинный (≥ `HYDE_MIN_SIGNATURE=40` chars), то
ставим его первым dense-query и дополнительно делаем **отдельный single-query
search** на HyDE-векторе (`HYDE_PASS_LIMIT=60` точек). Это добавляет
«семантически похожие» кандидаты поверх fusion.

## 3.8 Local rescoring (`rescore_points`, `compute_local_boost`)

Лёгкий boost без внешних вызовов. Компоненты (`compute_local_boost`, строка 962):

```
boost = message_score                            # phrase + token hits в MESSAGES
      + min(context_score * 0.2, 0.05)           # мягкий вклад CONTEXT
      + best_block_score                         # лучший message-block, до 0.28
      + score_intent_alignment                   # summary vs detail
      + score_metadata_signals                   # participants ∩ asker/entities
      + score_temporal_signal                    # date_range overlap → 0.06
      - context_penalty                          # если попали только в CONTEXT: -0.08
```

### Почему MESSAGES важнее CONTEXT

Когда overlap из предыдущего чанка случайно совпадает с query, retrieval
подтягивает «соседний» чанк выше правильного. Мы штрафуем такой случай
(`context_penalty=0.08`) и бустим попадания именно в MESSAGES-секцию.

### Quote-aware scoring

Если в message-block стоит флаг `quote` и попадание в `Quoted message:` части,
то quoted_score умножается на 0.2 и capped на 0.05 → своя реплика ценится
в ~5× сильнее цитаты. Это защищает от:
- ответа с цитатой, который «тянет в себя» весь исходный вопрос;
- live-update, который цитирует старый анонс.

### Intent alignment (опционально, через `INTENT_ALIGNMENT_WEIGHT`)

- summary вопросы (`"где ..."`, `"в каком документе ..."`) → boost
  кандидатам с `документ`/`http` → +0.12 × weight.
- detail вопросы (`"какой ..."`, `"как ..."`) → boost короткому
  содержательному ответу, penalty длинным summary-like → ±0.08 × weight.

По умолчанию `INTENT_ALIGNMENT_WEIGHT=0.0` (не портит baseline recall), но
доступен как ручной knob для sweep.

## 3.9 Rerank (`rerank_points`, строка 1394)

- Берём `points[:RERANK_LIMIT=20]`.
- `build_rerank_target(point)` переставляет `MESSAGES → CONTEXT` и
  обрезает до `RERANK_MAX_TEXT_CHARS=1200`.
- Батч в `POST {RERANKER_URL}` с моделью `nvidia/llama-nemotron-rerank-1b-v2`.
- Кеш: `RERANK_SCORE_CACHE: dict[(model, query, candidate_text), score]`.
- Blend с retrieval rank:

```python
retrieval_rank_score = 1.0 - (index / total_candidates)
blended = RERANK_ALPHA * (rerank_score + local_boost) + (1-RERANK_ALPHA) * retrieval_rank_score
```

`RERANK_ALPHA=0.2` означает: prefetch-порядок - якорь, reranker его слегка
корректирует. Это recall-safe и всё ещё бустит первые позиции.

### Fallbacks (все → retrieval order, без 500)

- 429 → короткий retry → fallback.
- Любой HTTP error → fallback.
- `len(scores) != len(candidates)` → fallback.

## 3.10 Final assembly (`assemble_message_ids`, строка 1304)

Единственное место, где мы переходим с уровня chunk'ов на уровень
отдельных `message_ids` - то, по чему считается метрика.

Алгоритм:

1. Для каждого point извлекаем `message_ids` из `metadata` и `blocks` из
   `page_content` (MESSAGES-секция, разбитая по `\n\n[YYYY-MM-DD ...`).
2. Если `len(blocks) == len(message_ids)`, скорим каждый блок отдельно:
   - `score_message_block` - phrase/token/quote-aware + intent + "earliest" boost.
   - `point_bonus = max(0, 0.18 - point_index * 0.004)` - дополнительный вес
     по месту в выдаче.
3. Если счёт блоков не совпал (technical fragment, split) - fallback:
   `reorder_message_ids_for_point` со слабыми phrase-hits.
4. Сортируем все scored-сообщения, дедуплицируем с сохранением порядка.
5. Режем до `FINAL_MESSAGE_LIMIT=50`.

### Early-exit

После `ASSEMBLY_EXIT_AFTER=120` points мы проверяем: если лучший возможный
block_score + следующий point_bonus уже ≤ текущий top-50 threshold - выходим.
Это спасает latency на больших прогонах без потерь качества.

### `prefers_earliest`

Если вопрос содержит «первое / исходный / начало / first / earliest» -
добавляем маленький linear boost раннему message внутри блока, чтобы
самый первый вопрос в треде побеждал своих последователей.

## 3.11 Наблюдаемость

### Per-request trace (`PipelineTrace`, строка 628)

На каждый `/search` формируется JSON-лог:

```json
{
  "event": "search",
  "request_id": "a1b2c3d4e5f6",
  "question": "какие сроки по проекту...",
  "total_ms": 842.31,
  "stages_ms": {"embed": 120.5, "qdrant": 230.7, "rescore": 45.1, "rerank": 410.2, "assemble": 35.8},
  "counts": {"dense_queries": 5, "sparse_queries": 7, "retrieval": 150, "reranked": 20},
  "fallbacks": [],
  "errors": [],
  "status": "ok"
}
```

### `/metrics` - Prometheus

- `search_requests_total{status=ok|degraded|error}`
- `search_request_duration_seconds` (histogram)
- `search_stage_duration_seconds{stage=embed|qdrant|...}` (per-stage histogram)
- `search_fallbacks_total{stage=dense|sparse|rerank|qdrant|...}`
- `search_errors_total{cls=ExceptionClassName}`

Тот же SLA, что у любого prod search-сервиса.

### OpenTelemetry (`OTEL_ENABLED=1`)

`FastAPIInstrumentor` + `HTTPXClientInstrumentor` + ручные span'ы per stage
(`embed / qdrant / rescore / rerank / assemble`). Экспорт через OTLP или console.

## 3.12 Endpoint'ы

| Endpoint                 | Назначение                                                    |
| ------------------------ | ------------------------------------------------------------- |
| `GET /health`            | Healthcheck                                                   |
| `GET /metrics`           | Prometheus endpoint                                           |
| `POST /search`           | Основной контракт (не менять)                                 |
| `POST /_debug/search`    | Внутренний: возвращает per-stage predictions + PipelineTrace  |

`/_debug/search` принимает query-параметры:
- `no_rescore=true` - отключить local rescoring
- `no_rerank=true` - отключить внешний reranker
- `fusion=rrf|dbsf`
- `max_dense=N`, `max_sparse=N`

Именно этот endpoint использует `eval/run.py --stages` для per-stage отчёта.

## 3.13 Все env-параметры (reference)

| Env                             | Default | Роль                                                          |
| ------------------------------- | ------- | ------------------------------------------------------------- |
| `HOST`/`PORT`                   | 0.0.0.0 / 8003 | сеть                                                   |
| `API_KEY` **или** `OPEN_API_LOGIN+PASSWORD` | - | аутентификация в upstream API                              |
| `EMBEDDINGS_DENSE_URL`          | -       | POST /embeddings URL                                          |
| `RERANKER_URL`                  | -       | POST /score URL                                               |
| `QDRANT_URL`                    | -       | URL Qdrant                                                    |
| `QDRANT_COLLECTION_NAME`        | evaluation | имя коллекции                                             |
| `QDRANT_DENSE_VECTOR_NAME`      | dense   | название dense-вектора в схеме                                |
| `QDRANT_SPARSE_VECTOR_NAME`     | sparse  | название sparse-вектора                                       |
| `DENSE_PREFETCH_K`              | 70      | prefetch-limit каждого dense-prefetch                         |
| `SPARSE_PREFETCH_K`             | 45      | prefetch-limit каждого sparse-prefetch                        |
| `RETRIEVE_K`                    | 150     | финальный limit после fusion                                  |
| `RERANK_LIMIT`                  | 20      | сколько реранкать                                             |
| `RERANK_ALPHA`                  | 0.2     | blend-вес rerank score vs retrieval order                     |
| `RERANK_MAX_TEXT_CHARS`         | 1200    | обрезка кандидатов перед /score                               |
| `FINAL_MESSAGE_LIMIT`           | 50      | top-K message_ids в ответе (≤50 по ТЗ)                        |
| `MAX_DENSE_QUERIES`             | 8       | сколько dense-queries максимум                                |
| `MAX_SPARSE_QUERIES`            | 8       | сколько sparse-queries максимум                               |
| `HYDE_MIN_SIGNATURE`            | 40      | минимальная длина hyde[0] для отдельного pass                 |
| `HYDE_PASS_LIMIT`               | 60      | limit у hyde-pass                                             |
| `UPSTREAM_CACHE_MAX_ITEMS`      | 20000   | LRU max для dense + rerank кешей                              |
| `UPSTREAM_MAX_RETRIES`          | 1       | retry на 429                                                  |
| `UPSTREAM_RETRY_DELAY_SECONDS`  | 0.25    | базовая задержка retry                                        |
| `INTENT_ALIGNMENT_WEIGHT`       | 0.0     | вес intent boost                                              |
| `OTEL_ENABLED`                  | -       | включить tracing                                              |
| `OTEL_EXPORTER`                 | otlp    | `otlp` или `console`                                          |
| `OTEL_SERVICE_NAME`             | search-service | имя сервиса в trace                                    |

---

## 3.x Worked example - трассировка одного запроса

Проследим, что именно происходит по времени и по данным для запроса:

```json
{
  "question": {
    "text": "где писали про SIGABRT на M1?",
    "search_text": "SIGABRT M1 MacBook Air",
    "keywords": ["SIGABRT", "M1"],
    "entities": {"names": ["MacBook Air", "M1"]},
    "asker": "alice@example.com",
    "variants": ["падает с SIGABRT на apple silicon"],
    "hyde": ["На MacBook Air M1 desktop client падает со SIGABRT при старте. Нужно посмотреть stacktrace."]
  }
}
```

### Шаг 1 - `build_query_context` (~1ms)

```python
QueryContext(
  primary_query="SIGABRT M1 MacBook Air",       # из search_text
  phrase_terms=frozenset(["SIGABRT", "M1", "MacBook Air"]),
  token_terms=frozenset(["sigabrt", "m1", "macbook", "air", "где", "писали", "про", "на"]),
  identity_terms=frozenset(["alice@example.com", "macbook air", "m1"]),
  signal_terms=frozenset(["sigabrt", "m1"]),    # токены с цифрой/символами
  date_range=None,
  prefers_earliest=False,
  intent="detail",                              # вопрос «где/как/какой»
)
```

### Шаг 2 - Query expansion

```python
dense_query_texts = [
  "SIGABRT M1 MacBook Air",                                    # search_text
  "где писали про SIGABRT на M1?",                             # text
  "падает с SIGABRT на apple silicon",                         # variants[0]
  "На MacBook Air M1 desktop client падает со SIGABRT при старте...",      # hyde[0] — длиннее 40 → в HyDE pass
]  # итого 3 dense-query (hyde вынесен в отдельный pass)

sparse_query_texts = [
  "SIGABRT M1 MacBook Air",
  "где писали про SIGABRT на M1?",
  "SIGABRT",                    # keywords[0]
  "M1",                         # keywords[1]
  "MacBook Air",                # entities.names[0]
  "M1",                         # entities.names[1] — дедуп
  "alice@example.com",          # asker
  "падает с SIGABRT на apple silicon",  # variants[0]
]  # дедуп → 7 sparse queries
```

### Шаг 3 - `dense_embed_safe` batch (~200ms → ~5ms if cached)

Первый прогон: real upstream call → `POST /embeddings` с батчем из 3 текстов →
`[vec1, vec2, vec3]` (shape `3×1024`). Кладём в LRU cache.
Повторный прогон тем же запросом: cache hit → ~5ms.

### Шаг 4 - sparse embeddings (~5ms)

`fastembed.SparseTextEmbedding` локально для 7 текстов → `[{indices, values}, ...]`.

### Шаг 5 - `qdrant_search` hybrid fusion (~60ms)

```python
prefetch = [
  Prefetch(query=dense_vec[0], using="dense",  limit=70),  # 3 branches
  Prefetch(query=dense_vec[1], using="dense",  limit=70),
  Prefetch(query=dense_vec[2], using="dense",  limit=70),
  Prefetch(query=sparse_vec[0], using="sparse", limit=45),  # 7 branches
  Prefetch(query=sparse_vec[1], using="sparse", limit=45),
  # ... ещё 5 sparse
]  # итого 10 prefetch → DBSF fusion → top-150

result = client.query_points(
    "evaluation",
    query=FusionQuery(fusion=Fusion.DBSF),
    prefetch=prefetch,
    limit=150,
)
# → 150 ScoredPoint (chunk_id, score, payload)
```

### Шаг 6 - HyDE pass (~30ms, if hyde present)

Отдельный `query_points` с hyde-embedding → 60 кандидатов → объединение с основным retrieval.

### Шаг 7 - `local_rescoring` (~40ms, 150 кандидатов)

Для каждого кандидата:
```python
local_boost = (
    score_text_signals(page_content, phrase_terms, token_terms, signal_terms)
    + score_metadata_signals(metadata, identity_terms)
    + score_temporal_signal(metadata, date_range)
    + score_context_penalty(page_content, phrase_terms)
    + (INTENT_ALIGNMENT_WEIGHT * score_intent_alignment(page_content, intent))
)
```

Пример для chunk'а с `SIGABRT` в MESSAGES + participants=[alice,bob]:
- phrase hits (`SIGABRT`, `M1`) → +0.10
- metadata (asker в participants) → +0.04
- temporal (no date_range) → 0
- context penalty (match в MESSAGES, не в CONTEXT) → 0
- intent alignment off → 0
- total → `local_boost = 0.14`, прибавляется к retrieval score.

### Шаг 8 - `rerank_safe` top-20 (~300ms)

Top-20 по rescored-score передаются в `POST /score` cross-encoder'у.
Результат - массив из 20 rerank scores ∈ [0, 1].

Blended score:
```python
blended = 0.2 * rerank_score + 0.8 * retrieval_rank_score
```

### Шаг 9 - `assemble_message_ids` (~5ms)

Для top-N (обычно 30-40 chunk'ов) считаем per-message score внутри каждого chunk'а
(`score_message_block`), мержим, сортируем, дедуп, cut до `FINAL_MESSAGE_LIMIT=50`.

### Итог - response

```json
{
  "message_ids": [
    "4555555555555555555",
    "4555555555555555556",
    ...
  ],
  "status": "ok",
  "trace": {
    "total_ms": 641,
    "stages_ms": {
      "build_context": 1,
      "dense_embed": 205,
      "sparse_embed": 5,
      "qdrant_search": 63,
      "hyde_pass": 28,
      "rescore": 41,
      "rerank": 293,
      "assemble": 5
    },
    "counts": {"prefetch_candidates": 150, "after_rescore": 150, "after_rerank": 20, "final": 50},
    "fallbacks": [],
    "status": "ok"
  }
}
```

### Что изменится при fallback'ах

| Сценарий              | Что меняется в трассе                                              |
| --------------------- | ------------------------------------------------------------------ |
| `dense API 429`       | `dense_embed_safe` → `fallbacks: ["dense_embed"]`, dense branches пустые, sparse-only retrieval, rerank работает. `total_ms ~400ms` |
| `reranker 429`        | `rerank_safe` → `fallbacks: ["rerank"]`, blended = retrieval_rank. `total_ms ~250ms` |
| `Qdrant down`         | `qdrant_search` → `fallbacks: ["qdrant"]`, `status: "degraded"`, `message_ids: []`, но 200 OK |

---

## 3.20 Приложение: API reference - все функции search-сервиса

Карта функций `search/main.py`, сгруппированных по роли в pipeline'е. Если про функцию нет ни слова в основной части документа - ищите её здесь.

### Роуты и lifecycle

| Функция                 | Что делает |
| ----------------------- | ---------- |
| `health`                | `GET /health → 200` для liveness-check тестирующей системы. |
| `search`                | `POST /search(SearchAPIRequest)` - основной endpoint. Вызывает `run_search_pipeline`, возвращает `message_ids` + `status`. |
| `search_debug`          | `POST /search_debug` - расширенный ответ с `trace/stages`, используется `eval/run.py --stages`. Тот же pipeline, другой контракт. |
| `metrics`               | `GET /metrics` - Prometheus exposition format (`request_count`, `stage_duration_seconds`, `fallback_total`). |
| `lifespan`              | FastAPI lifespan: на старте - `get_sparse_model()` прогрев, `httpx.AsyncClient` + `AsyncQdrantClient` pool; на shutdown - `.aclose()`. |
| `exception_handler`     | Глобальный 500-хендлер: логирует traceback, возвращает JSON вместо HTML. |
| `main`                  | uvicorn entry-point, читает `HOST`/`PORT`. |
| `validate_required_env` | На старте проверяет наличие всех env (`QDRANT_URL`, `API_KEY`, `EMBEDDINGS_DENSE_URL`, `RERANKER_URL`, ...). Без них `RuntimeError` сразу, а не в первом запросе. |
| `setup_tracing`         | Если `OTEL_ENABLED=1` - инструментирует FastAPI+httpx через OpenTelemetry, создаёт span per stage. Если выключено - no-op. |

### Env helpers

| Функция                       | Что делает |
| ----------------------------- | ---------- |
| `getenv_int(name, default)`   | Читает int из env с fallback; ловит `ValueError`, падает в default с предупреждением. |
| `getenv_float(name, default)` | То же для float. Используется для `RERANK_ALPHA`, `PHRASE_BOOST`, всех порогов. |

### Query context - строим представление вопроса

| Функция                              | Что делает |
| ------------------------------------ | ---------- |
| `build_query_context(question)`      | Собирает `QueryContext`: `phrase_terms`, `token_terms`, `identity_terms`, `intent`, `prefers_earliest`, `query_start/end`. Вход для всего ranking-слоя. |
| `normalize_query_text(text)`         | `re.sub(r"\s+", " ", text).strip()` - убирает multiline whitespace. Точка нормализации для всех сравнений. |
| `normalize_terms(values)`            | `lower()` + `unique_texts()` - lowercased deduped список. Готовит `identity_terms`. |
| `unique_texts(texts, limit=None)`    | Дедуп с сохранением порядка + нормализация whitespace. Базовый helper. |
| `collect_entity_terms(entities)`     | Флэтит `entities.people/emails/documents/names/links` в один dedup'нутый список. |
| `build_primary_query(question)`      | `search_text or text` - выбирает основной текст вопроса для dense/sparse single-query branch. |
| `build_dense_queries(question)`      | Собирает до 8 вариантов: `search_text`, `text`, 4 варианта из `variants`, 3 из `hyde`. Это multi-query dense branch. |
| `build_sparse_queries(question)`     | Собирает до 7 вариантов: `search_text`, `text`, `keywords`, entities. Sparse не нужны `hyde`/`variants` - они парафразы, не keywords. |
| `build_rerank_query(question)`       | Строит `label` для rerank endpoint - обычно `text` с keywords/entities склеенными. |
| `question_has_enrichments(q)`        | True, если в вопросе есть `keywords/entities/date_mentions/asker`. Сигнал «можно доверять enrichments». |
| `extract_signal_tokens(text)`        | Токенайзер: `TOKEN_RE.findall` + фильтр по длине ≥4 или наличию цифр/спецсимволов. Основа token_terms. |
| `extract_plain_focus_tokens(text)`   | Fallback: если у вопроса нет `keywords/entities`, вытащить содержательные токены из `text` с фильтром по `PLAIN_QUERY_STOPWORDS`. |
| `build_phrase_terms(question)`       | Phrase-terms для rescore: `keywords + entity_terms + date_mentions + asker`; если пусто - fallback на `extract_plain_focus_tokens`. |
| `build_query_signal_tokens(q)`       | Token-terms для rescore: токены из `primary_query`, `text`, `keywords`, entities, первых 2 `variants/hyde`. |
| `detect_query_intent(question)`      | Возвращает `"summary"/"detail"/"neutral"` по стартовым маркерам (`"кратко про"`, `"расскажи"`, `"что известно о"`, ...). Влияет на `score_intent_alignment`. |
| `is_summary_like_text(text)`         | Эвристика «этот chunk выглядит как summary»: содержит `"итого"/"TL;DR"/"в двух словах"` или ссылку на документ. |
| `is_summary_like_text_lowered(t)`    | Lowered-версия для внутренних вызовов (не normalize'ит заново). |
| `query_prefers_earliest_message(q)`  | Ищет в вопросе маркеры `"первое"/"исходное"/"самое раннее"/"first"` - если есть, в `reorder_message_ids_for_point` бустим ранние блоки. |
| `parse_timestamp(value, end_of_day)` | Универсальный parser: принимает `int/float`, epoch-as-string, `YYYY-MM-DD`, full ISO. `end_of_day=True` добавляет `23:59:59`. Для `date_range.from/to`. |

### Upstream-взаимодействие (dense, rerank, Qdrant)

| Функция                                             | Что делает |
| --------------------------------------------------- | ---------- |
| `get_upstream_request_kwargs()`                     | Строит kwargs для `httpx.AsyncClient.post`: заголовок `Authorization: Bearer <API_KEY>`, таймаут `UPSTREAM_TIMEOUT`. |
| `post_upstream_json(client, url, payload, purpose)` | Обёртка над `client.post` с retry на 429: экспоненциальная задержка, `UPSTREAM_MAX_RETRIES` попыток, подробный лог. Используется для dense и rerank. |
| `cache_set(cache, key, value)`                      | Bounded insert в LRU-dict: при превышении `UPSTREAM_CACHE_MAX_ITEMS` выкидывает первый (FIFO). Используется для `DENSE_EMBED_CACHE`, `RERANK_SCORE_CACHE`. |
| `embed_dense_many(client, texts)`                   | Пакетный POST на `/embeddings` (Qwen3). Кэшированные - возвращает из памяти, остальные батчем просит у upstream. |
| `embed_dense_many_safe(client, texts)`              | Обёртка: ловит 429/timeout/network, переводит branch в `fallbacks: ["dense_embed"]`, возвращает пустой список - хендлер сверху переключится на sparse-only. |
| `embed_sparse_many(texts)`                          | Синхронно (через `asyncio.to_thread`) вызывает `get_sparse_model().embed(texts)`. Локально, без сети. |
| `embed_sparse_many_safe(texts)`                     | Обёртка на случай `fastembed` исключения - `fallbacks: ["sparse_embed"]`, дальше dense-only. |
| `get_sparse_model()`                                | `lru_cache(1)` над `SparseTextEmbedding("Qdrant/bm25")`. Модель грузится один раз на процесс. |
| `qdrant_search(client, dense, sparse, fusion)`      | Строит `QueryRequest` с prefetch branches (multi-query dense + multi-query sparse), вызывает `client.query_points` с `FusionQuery`. Возвращает `points`. |
| `qdrant_search_safe(...)`                           | Retry + fallback: если `fusion=DBSF` падает - пробуем `RRF`; если и это падает - dense-only; при полном провале - пустой список + `fallbacks: ["qdrant"]`. |
| `qdrant_hyde_pass(client, hyde_vec)`                | Отдельный Qdrant `query_points` **только для HyDE-вектора** - limit `HYDE_PASS_LIMIT`, dense-only. Результаты мержатся с основным retrieval на этапе rescore. |
| `get_rerank_scores(client, label, targets)`         | Постит `{text_1: label, text_2: targets}` на rerank-API, парсит `data[].score`, кэширует через `RERANK_SCORE_CACHE`. Ядро rerank-стадии. |

### Rescoring - локальная постобработка retrieval'а

| Функция                                   | Что делает |
| ----------------------------------------- | ---------- |
| `get_point_payload(point)`                | Безопасный геттер: `point.payload or {}`, проверяет `isinstance(dict)`. |
| `get_point_metadata(point)`               | `payload["metadata"] or {}` - откуда берутся `chat_id`, `thread_sn`, `start/end`, `participants`. |
| `get_point_text(point)`                   | Возвращает lowercased+normalize'нутый `page_content` целиком. Для глобального phrase-matching. |
| `get_point_context_text(point)`           | Только CONTEXT-секция page_content'а (lowercased). Используется для `context_penalty`. |
| `get_point_message_text(point)`           | Только MESSAGES-секция. Для boоста - если матч в MESSAGES, это сильный сигнал. |
| `split_page_sections(page_content)`       | Парсит `CHAT:...\nCONTEXT:...\nMESSAGES:...` → `(context, messages)`. Ключевой парсер для всего rescore'а. |
| `extract_message_blocks(page_content)`    | Берёт MESSAGES секцию, режет по `MESSAGE_BLOCK_SPLIT_RE` (двойной `\n` между блоками), возвращает список блоков. Один блок ≈ одно исходное сообщение. |
| `split_block_header_body(block)`          | Первая строка блока = header (`[timestamp \| sender \| flags]`), остальное = body. |
| `block_has_flag(block, flag)`             | Парсит `flags` из header'а (`quote`, `forward`, `system`, `thread=...`, `part=x/y`), проверяет наличие. |
| `split_quoted_text(text)`                 | Если в тексте `>>> QUOTE_MARKER` - делит на `(own_text, quoted_text)`. Для quote-downweight. |
| `collapse_message_blocks(blocks)`         | Склеивает блоки с флагом `part=1/N`, `part=2/N`... обратно в один блок, соответствующий одному исходному message_id. |
| `score_text_signals(text, phrase, token)` | Ядро phrase+token boost: `+0.07` за phrase с цифрой/спецсимволом, `+0.04` за обычный phrase, `+0.01` за token; cap'ы 0.24 и 0.08. |
| `score_message_block(ctx, block, idx)`    | Считает score per-message-block с учётом quote-downweight, prefers_earliest, intent-alignment. Используется в `assemble_message_ids`. |
| `score_best_message_block(ctx, point)`    | Максимум `score_message_block` по всем блокам chunk'а. Chunk-level сигнал: хоть в одном message есть прямой hit. |
| `score_metadata_signals(ctx, point)`      | `+0.06`, если `participants/thread_sn/asker` матчат entity_terms/identity_terms вопроса. |
| `score_temporal_signal(ctx, point)`       | `+0.06`, если `[start, end]` chunk'а пересекает `query_start/query_end` из `date_range`. Soft boost - не hard filter. |
| `score_intent_alignment(q, text)`         | `+0.12/-0.08` в зависимости от того, совпадает ли `intent` вопроса (`summary/detail`) с характером chunk'а. |
| `compute_local_boost(ctx, point)`         | Главная rescoring-функция: суммирует `score_text_signals(message_text)` + `score_best_message_block` + `metadata` + `temporal` + `intent_alignment` − `context_penalty`. Это `+0.14`-типа буст, который уезжает в `blended_score`. |
| `rescore_points(ctx, points)`             | Применяет `compute_local_boost` к каждой точке, возвращает `(reordered_points, boost_map)`. |

### Rerank и сборка ответа

| Функция                                                | Что делает |
| ------------------------------------------------------ | ---------- |
| `build_rerank_target(point)`                           | Строит текст для rerank: `MESSAGES:\n...\n\nCONTEXT:\n...` - именно MESSAGES сверху, чтобы cross-encoder не отвлекался на overlap. |
| `trim_rerank_text(text, limit)`                        | Режет до `RERANK_MAX_TEXT_CHARS`, добавляет `" ..."`. Защита от OOM на rerank-API. |
| `rerank_points(client, ctx, query, points, boost_map)` | Берёт top-`RERANK_LIMIT` (20), зовёт `get_rerank_scores`, считает `blended = α*rerank + (1-α)*rank_score`, сортирует. Остальной хвост идёт после. |
| `extract_message_ids(point)`                           | Берёт `payload.metadata.message_ids`, приводит к `[str]`. |
| `dedupe_message_ids(ids, limit)`                       | Сохраняя порядок, убирает дубли + cut до `limit` (обычно `FINAL_MESSAGE_LIMIT=50`). |
| `reorder_message_ids_for_point(ctx, point)`            | Если в chunk'е несколько сообщений - переупорядочивает их по `score_text_signals` на per-message-block уровне (+ буст ранних, если `prefers_earliest`). |
| `assemble_message_ids(ctx, points)`                    | Финальная сборка: для каждого top-N чанка зовёт `reorder_message_ids_for_point`, складывает в один список, `dedupe_message_ids`. Это и есть `response.message_ids`. |
| `run_search_pipeline(payload)`                         | Оркестратор: build_context → embed dense/sparse → qdrant_search → hyde_pass → rescore → rerank → assemble. Логирует `stages_ms`, эмитит Prometheus метрики, возвращает `SearchAPIResponse`. |

### Как читать эту таблицу

- Любая строка = одна функция в рантайме. Helpers, которые не влияют на ranking (`normalize_query_text`, `unique_texts`), указаны для полноты.
- Порядок групп примерно совпадает с порядком вызова в pipeline'е.
- Если в основной части документа функция уже описана (например, `compute_local_boost` в §3.7) - здесь одна строка-референс, не дубль логики.
- Полная цепочка `/search`: `search → run_search_pipeline → build_query_context → embed_{dense,sparse}_many_safe → qdrant_search_safe → qdrant_hyde_pass → rescore_points → rerank_points → assemble_message_ids`.
