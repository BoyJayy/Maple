# 01. Архитектура решения (для человека)

> Верхнеуровневая карта проекта. Прочитав её, вы сможете за 2 минуты
> объяснить жюри, что мы делаем, зачем и из каких кусков состоит система.

---

## 1.1 Задача одним предложением

Построить **поисковую систему по чат-сообщениям**: на вход - enriched-вопрос, на
выход - отсортированный список `message_ids` сообщений, которые отвечают на вопрос.
Метрика считается по `message_ids`, не по тексту.

## 1.2 Что делает пользователь ТЗ

- Индексирует чат через `POST /index` (index-service строит chunks).
- Выполняет поиск через `POST /search` (search-service возвращает message_ids).
- Оценивается по `Recall@50 * 0.8 + nDCG@50 * 0.2` (см. [eval/metrics.py](../eval/metrics.py)).

## 1.3 Две коробки

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ index-service│ ──> │   Qdrant     │ <── │search-service│
│  (chunking + │     │ hybrid vector│     │ (retrieval + │
│  sparse emb.)│     │   database   │     │fusion+rerank)│
└──────────────┘     └──────────────┘     └──────────────┘
        ▲                                         │
        │                                         │
    messages                                   question
    (chat payload)                            (enriched)
```

Оба сервиса - **FastAPI + uvicorn**, Python 3.13, slim-image.
Qdrant - отдельным контейнером (`qdrant/qdrant:v1.14.1`).

## 1.4 Решения архитектуры

| Решение                                     | Почему                                                              |
| ------------------------------------------- | ------------------------------------------------------------------- |
| Два сервиса, а не один                      | Так требует ТЗ: разные контракты, разные образы                     |
| **Hybrid retrieval (dense + sparse)**       | Dense ловит смысл, sparse - точные токены (имена, ссылки, ошибки)   |
| Три разных текста на chunk                  | `page_content` - для rerank/отладки, `dense_content` - для смысла, `sparse_content` - для BM25 keyword-match |
| Sparse считается **локально** (fastembed)   | Не зависим от внешнего API, нет rate limit                          |
| Dense считается **во внешнем API**          | Так решил хакатон - Qwen3-Embedding-0.6B                            |
| Qdrant `Query API` с `Fusion.DBSF`          | Безопасный baseline, не требует тюнинга весов между dense и sparse  |
| Rerank top-N через nvidia nemotron          | Поднимает первые позиции → это бустит `nDCG@50`                     |
| `RERANK_ALPHA = 0.2` (blend с retrieval)    | Reranker не перетирает хороший prefetch, recall остаётся стабильным |
| Graceful degradation при 429                | Если упал dense → идём sparse-only. Если упал rerank → возвращаем retrieval order. Никогда не 500 |
| Кеши dense embeddings + rerank scores       | Снижают riskpack 429 и ускоряют повторные вопросы                   |

## 1.5 Компоненты и где они лежат в коде

```
Hackathon-Search-Engine-1/
├── index/                    # FastAPI index-service
│   ├── main.py               # /health, /index, /sparse_embedding
│   ├── schemas.py            # Pydantic: IndexAPIRequest/Response
│   ├── chunking.py           # вся логика: нормализация + chunking + форматирование
│   ├── sparse.py             # fastembed BM25 + кеш
│   ├── config.py             # env-параметры (thresholds, limits)
│   └── Dockerfile / Makefile / requirements.txt
├── search/                   # FastAPI search-service
│   ├── main.py               # весь pipeline поиска (1700 строк)
│   ├── Dockerfile / Makefile / requirements.txt
├── eval/                     # локальная оценка решения
│   ├── ingest.py             # orchestrator: /index → dense → sparse → Qdrant
│   ├── run.py                # гоняет dataset → Recall@K, nDCG@K
│   └── metrics.py            # формула score
├── scripts/                  # dev/QA инструменты
│   ├── chunking_diagnostic.py  # видит ли chunker все сообщения, размер чанков
│   ├── sweep_chunking.py       # coord-descent по параметрам chunker'а
│   ├── loadtest.py             # async p50/p95/p99 для /search
│   ├── chaostest.py            # kill upstream → assert 200
│   ├── ab_qdrant.py            # A/B сравнение конфигов Qdrant
│   └── morph_diag.py           # диагностика морфологии для stopword filtering
├── data/                     # тестовые чаты и synthetic eval
│   └── dataset_ts_chat.json    # локальный пример чата
├── docker-compose.yml        # qdrant + index + search
└── docs/                     # ← вы здесь
```

## 1.6 Граница ответственности каждого сервиса

```
index  - отвечает за КАЧЕСТВО ЧАНКОВ
search - отвечает за КАЧЕСТВО РАНЖИРОВАНИЯ
Qdrant - отвечает за ХРАНЕНИЕ и PREFETCH
```

Полезная mental model: если упал `nDCG`, сначала смотрите на `search`;
если упал `Recall` - скорее всего проблема в chunking'е или в ingestion'е.

## 1.7 Потоки данных (два независимых flow)

### Индексация (offline, batch-style)

```
chat payload
  → POST /index  (index-service строит chunks)
  → dense embedding batch  (внешний Qwen3 /embeddings)
  → sparse embedding batch (локальный /sparse_embedding через fastembed)
  → upsert в Qdrant
```

### Поиск (online, per-request)

```
enriched question
  → build_dense_queries + build_sparse_queries
  → embed_dense_many_safe + embed_sparse_many_safe   (параллельно asyncio.gather)
  → Qdrant Query API (DBSF/RRF fusion)
  → local rescoring (phrase/token boost + metadata)
  → rerank top-20 (nemotron /score)
  → assemble_message_ids → top-50
```

## 1.8 Observability (что добавили сверху)

- `/metrics` - Prometheus: request counts, per-stage histogram, fallback counters, error counts.
- Structured per-request JSON log: `request_id`, `total_ms`, `stages_ms`, `counts`, `fallbacks`, `status`.
- OpenTelemetry tracing (опционально, по `OTEL_ENABLED=1`): span per stage (embed / qdrant / rescore / rerank / assemble).
- `chaostest.py` - автоматически проверяет, что при смерти любого upstream сервис возвращает 200.

## 1.9 Что мы НЕ делаем (и почему осознанно)

- Не строим свою dense модель - ТЗ даёт Qwen3 через внешний API.
- Не делаем жёстких date-фильтров в Qdrant - мягкий boost по `date_range` работает лучше на смешанных вопросах.
- Не делаем жёстких filters по participants/mentions - те же соображения: soft boost > hard filter для recall.
- Не используем дорогой cross-encoder на всём retrieval - rerank только top-20, иначе 429 и latency.
- Не пересобираем index перед каждым поиском - ingestion это offline-pipeline, а не runtime зависимость.

---

## 1.10 Пример - что куда идёт (end-to-end)

Допустим, у пользователя есть локальный чат из 1000 сообщений и вопрос:

> «где писали про SIGABRT на M1?»

### Фаза 1 - ingestion (один раз, offline)

```text
local chat corpus  (1000 msgs)
         │
         ▼
 POST /index  ──►  index-service
         │
         │   normalize → filter → split → chunk → format
         │
         ▼
 ~120 chunks с {page_content, dense_content, sparse_content, message_ids}
         │
         ├──► POST /embeddings (Qwen3) ──► dense vectors (1024-dim × 120)
         └──► POST /sparse_embedding     ──► sparse vectors (BM25 × 120)
                                                  │
                                                  ▼
                                      Qdrant upsert (120 points)
                                      collection: evaluation
```

### Фаза 2 - поиск (на каждый вопрос, runtime)

```text
question = {"text": "где писали про SIGABRT на M1?",
            "search_text": "SIGABRT M1 MacBook",
            "keywords": ["SIGABRT", "M1"],
            "entities": {"names": ["MacBook Air", "M1"]}}
         │
         ▼
 POST /search  ──►  search-service
         │
         ├── build_query_context()        ~1ms    (phrase/token_terms, identity_terms)
         ├── dense_embed_safe() × 8       ~200ms  (cached, 8 query variants)
         ├── sparse_embed_safe() × 8      ~5ms    (local fastembed)
         ├── qdrant_search() hybrid       ~60ms   (16 prefetch → DBSF fusion → top-150)
         ├── hyde_pass() if hyde present  ~30ms   (отдельный dense pass)
         ├── local_rescoring()            ~40ms   (phrase/entity/temporal/context)
         ├── rerank_safe() top-20         ~300ms  (nemotron cross-encoder)
         └── assemble_message_ids()       ~5ms    (chunk → message-blocks → top-50)
         │
         ▼
 {"message_ids": ["4555555555555555555", ...], "status": "ok"}
         │
         ├── log JSON { request_id, total_ms: 641, stages_ms: {...}, status: "ok" }
         ├── Prometheus: search_request_duration_seconds, search_stage_duration_seconds{stage=...}
         └── OTEL span per stage (если OTEL_ENABLED=1)
```

### Что отвечает за что

| Стадия                | Сервис          | За что отвечает                                           |
| --------------------- | --------------- | --------------------------------------------------------- |
| `POST /index`         | index-service   | Нарезка на чанки + подготовка 3 текстов                   |
| `POST /sparse_embedding` | index-service | Локально считает BM25-векторы                             |
| dense vectors         | external API    | Qwen3-0.6B, 1024-dim, нам доступно через `EMBEDDINGS_DENSE_URL` |
| Qdrant upsert/query   | Qdrant          | Хранение + hybrid retrieval через `query_points` API      |
| `POST /search`        | search-service  | Оркестрация: context → retrieval → rescore → rerank → assemble |
| `POST /score` (rerank) | external API   | Nemotron cross-encoder, top-20                            |

Главная идея: offline-ingestion строит векторы один раз,
online-search гоняет быстрые branches и rerank только на top-20. Если падает любой
external API - сервис не падает, а переходит в degraded mode (sparse-only или
retrieval-order).
