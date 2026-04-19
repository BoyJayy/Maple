# 04. End-to-End Data Flow - на конкретном примере

> Здесь прослеживается одно сообщение от сырого JSON чата до `message_id`
> в ответе `/search`. Цель - научиться разворачивать это в голове и рисовать
> на доске во время защиты.

---

## 4.1 Сцена: 3 сообщения в чате «Atlas Hub»

Допустим, пришёл такой кусок чата:

```json
{
  "chat": {"id": "chat-42", "name": "Atlas Hub", "type": "group", "sn": "ah"},
  "new_messages": [
    {
      "id": "m-1", "time": 1714744800, "sender_id": "alice@x",
      "thread_sn": "T1", "text": "Всем привет! Когда планируется релиз SDK 3.2?",
      "parts": null, "mentions": [],
      "is_system": false, "is_hidden": false, "is_forward": false, "is_quote": false,
      "file_snippets": "", "member_event": null
    },
    {
      "id": "m-2", "time": 1714744900, "sender_id": "bob@x",
      "thread_sn": "T1", "text": "Релиз SDK 3.2 запланирован на 15 мая.",
      "parts": null, "mentions": [],
      "is_system": false, "is_hidden": false, "is_forward": false, "is_quote": false,
      "file_snippets": "", "member_event": null
    },
    {
      "id": "m-3", "time": 1714745000, "sender_id": "alice@x",
      "thread_sn": "T1", "text": "ок",
      "parts": null, "mentions": [],
      "is_system": false, "is_hidden": false, "is_forward": false, "is_quote": false,
      "file_snippets": "", "member_event": null
    }
  ],
  "overlap_messages": []
}
```

## 4.2 Stage 1 - `POST /index`

### normalize

Все три сообщения проходят через `render_message`:
- `m-1` → `"Всем привет! Когда планируется релиз SDK 3.2?"`
- `m-2` → `"Релиз SDK 3.2 запланирован на 15 мая."`
- `m-3` → `"ок"`

### filter (`is_message_searchable`)

- `m-3` = `"ок"` ∈ `SHORT_ACK_MESSAGES`, нет `file_snippets`/mentions/forward/quote → выбрасывается.
- `m-1`, `m-2` проходят.

### chunking

Оба сообщения в одном треде (`T1`), между ними 100 секунд (< 3ч),
размер `page_content` маленький - всё влезает в один чанк.

`m-1` - заголовок-greeting, но `starts_new_topic(m-1)` возвращает `True`
(`"всем привет"` + длина) → overlap сбрасывается (у нас его всё равно нет).

### output

```json
{
  "results": [
    {
      "page_content":
        "CHAT: Atlas Hub\n\nCHAT_TYPE: group\n\nCHAT_ID: chat-42\n\n"
        "MESSAGES:\n\n"
        "[2024-05-03 10:00:00 UTC | alice@x | thread=T1]\n"
        "Всем привет! Когда планируется релиз SDK 3.2?\n\n"
        "[2024-05-03 10:01:40 UTC | bob@x | thread=T1]\n"
        "Релиз SDK 3.2 запланирован на 15 мая.",
      "dense_content":
        "chat Atlas Hub\n\nchat_type group\n\n"
        "Всем привет! Когда планируется релиз SDK 3.2?\n\n"
        "Релиз SDK 3.2 запланирован на 15 мая.",
      "sparse_content":
        "Atlas Hub\ngroup\nchat-42\n"
        "Всем привет! Когда планируется релиз SDK 3.2?\nsender: alice@x\n\n"
        "Релиз SDK 3.2 запланирован на 15 мая.\nsender: bob@x",
      "message_ids": ["m-1", "m-2"]
    }
  ]
}
```

Видно:
- `page_content` содержит timestamps и flags, читается глазами.
- `dense_content` - smooth, без служебки.
- `sparse_content` - компактный keyword-mix, с явными `sender:`.

## 4.3 Stage 2 - ingestion orchestrator (`eval/ingest.py`)

Три батча вызовов:

1. `POST /index` → получили 1 chunk (см. выше).
2. `POST /sparse_embedding` с `texts=[chunk.sparse_content]` →
   `vectors=[{indices:[...], values:[...]}]`.
3. Dense API (внешний) `POST /embeddings` с
   `{"model":"Qwen/Qwen3-Embedding-0.6B","input":[chunk.dense_content]}` →
   `{"data":[{"index":0,"embedding":[0.012, -0.033, ...]}]}` (1024 float).

Формируется стабильный point ID:

```python
uuid5(NAMESPACE, f"{chat_id}:m-1,m-2:<sha1(page_content)>")
```

Это детерминированный ID: при повторной индексации того же чанка → тот же UUID,
upsert идемпотентный.

Потом `qc.delete(...)` по `metadata.chat_id == "chat-42"` и upsert:

```python
PointStruct(
    id=<uuid>,
    vector={"dense": [...1024 floats...], "sparse": SparseVector(...)},
    payload={
        "page_content": "CHAT: Atlas Hub\n\n...",
        "metadata": {
            "chat_name": "Atlas Hub", "chat_type": "group", "chat_id": "chat-42",
            "chat_sn": "ah", "thread_sn": "T1",
            "message_ids": ["m-1", "m-2"],
            "start": "1714744800", "end": "1714744900",
            "participants": ["alice@x", "bob@x"],
            "mentions": [],
            "contains_forward": False, "contains_quote": False,
        },
    },
)
```

## 4.4 Stage 3 - `POST /search`

Предположим, приходит вопрос:

```json
{
  "question": {
    "text": "Когда выйдет SDK 3.2?",
    "search_text": "Дата релиза SDK 3.2",
    "variants": ["когда запланирован релиз SDK 3.2"],
    "hyde": ["Релиз SDK 3.2 ожидается в мае 2024 года."],
    "keywords": ["SDK 3.2", "релиз"],
    "entities": {"people": [], "documents": [], "names": ["SDK 3.2"], "links": []},
    "date_mentions": ["май"],
    "date_range": {"from": "2024-04-01", "to": "2024-06-01"},
    "asker": "carol@x",
    "asked_on": "2024-05-02"
  }
}
```

### 4.4.1 `build_query_context`

```
phrase_terms  = ("sdk 3.2", "релиз", "май", "carol@x")
token_terms   = ("sdk", "3.2", "релиз", "дата", "запланирован", ...)
identity_terms= {"carol@x", "sdk 3.2"}
intent        = "neutral"   (нет "где/в каком документе/какой")
prefers_earliest = False
query_start=1711929600, query_end=1717200000
```

### 4.4.2 `build_dense_queries`

```
[
  "Дата релиза SDK 3.2",                        # search_text
  "Когда выйдет SDK 3.2?",                      # text
  "когда запланирован релиз SDK 3.2",          # variant
  "Релиз SDK 3.2 ожидается в мае 2024 года.",  # hyde[0] (HYDE_MIN_SIGNATURE=40 OK)
]
```

Внутри `run_search_pipeline` hyde[0] переставляется в начало и отдельным
pass'ом уходит в Qdrant.

### 4.4.3 `build_sparse_queries`

```
[
  "Дата релиза SDK 3.2\nSDK 3.2 релиз май carol@x",    # combined
  "SDK 3.2 релиз май carol@x",                          # exact_focus
  "SDK 3.2",                                            # entity_terms
  "Дата релиза SDK 3.2",                                # primary
  "Когда выйдет SDK 3.2?",                              # text
  "когда запланирован релиз SDK 3.2",                   # variant
  "Релиз SDK 3.2 ожидается в мае 2024 года.",           # hyde
]
```

### 4.4.4 Embeddings

Параллельно:
- Dense: `POST /embeddings` с 4 текстами → кеш пополняется.
- Sparse: локально `fastembed.embed(7 text)` → 7 SparseVector.

### 4.4.5 Qdrant retrieval

```python
prefetch = [
    Prefetch(query=dv, using="dense",  limit=70) for dv in dense_vectors (4 шт)
] + [
    Prefetch(query=sv, using="sparse", limit=45) for sv in sparse_vectors (7 шт)
]
# Итого 11 prefetch веток, fusion DBSF, limit 150
```

Наш chunk `[m-1, m-2]` по идее ловится:
- dense-веткой (semantic similarity на «Релиз SDK 3.2»);
- sparse-веткой (`SDK 3.2`, `релиз`).

Fusion даёт высокий combined score → chunk попадает в top.

### 4.4.6 Local rescoring

Для нашего chunk:

```
message_text = "[... alice@x ...] всем привет ... когда планируется релиз sdk 3.2?
                [... bob@x ...] релиз sdk 3.2 запланирован на 15 мая."
phrase_hits:
  "sdk 3.2"  → +0.07 (digit → special signal)
  "релиз"    → +0.04
  "май"      → +0.04
  "carol@x"  → 0 (нет в тексте)
token_hits:
  "sdk", "3.2", "релиз", "дата" → 4 × 0.01 = +0.04

best_block_score: для блока m-2 (len ≤ 220) ещё +0.03

metadata signals: participants=[alice@x, bob@x], identity={carol@x, sdk 3.2}
  → intersection = ∅, 0

temporal: query_range=[2024-04-01, 2024-06-01], start=1714744800 (2024-05-03)
  → попадает, +0.06

context_penalty: не нашли только в CONTEXT, 0.

Total boost ≈ 0.04 + 0.04 + 0.07 + 0.03 + 0.06 = ~0.24
```

Сильный буст поверх fusion score → chunk остаётся в топе.

### 4.4.7 Rerank (top-20)

`build_rerank_query`:
```
Дата релиза SDK 3.2
sdk 3.2 релиз
```

`build_rerank_target(point)` переставляет в `MESSAGES → CONTEXT`:
```
MESSAGES:
[... alice@x ...] всем привет ...
[... bob@x ...] релиз sdk 3.2 запланирован на 15 мая.
```

Nemotron-rerank-1b-v2 возвращает высокий `score`. Blended:

```
retrieval_rank_score = 1.0 - index/20  (если chunk был #0 → 1.0)
rerank_score = nemotron_score + local_boost
blended = 0.2 * rerank_score + 0.8 * retrieval_rank_score
```

Chunk остаётся на первом месте.

### 4.4.8 Assemble message_ids

Для нашего chunk `len(message_ids) == len(blocks) == 2` → per-message scoring:

```
m-1 (alice, "всем привет ... когда планируется ..."):
  phrase hits: "sdk 3.2" → +0.07
  token hits: "sdk", "3.2", "релиз" → +0.03
  len ≤ 220 → +0.03
  block_score ≈ 0.13, + point_bonus (0.18) = 0.31

m-2 (bob, "релиз sdk 3.2 запланирован на 15 мая"):
  phrase hits: "sdk 3.2" → +0.07, "релиз" → +0.04, "май" → +0.04
  token hits: "sdk", "3.2", "релиз", "запланирован", "15" → +0.05
  len ≤ 220 → +0.03
  block_score ≈ 0.23, + point_bonus (0.18) = 0.41
```

`m-2` обгоняет `m-1` (в нём **ответ**), даже несмотря на то, что `m-1`
идёт первым в chunk'е.

### 4.4.9 Response

```json
{"results": [{"message_ids": ["m-2", "m-1"]}]}
```

Только один chunk → один результирующий `SearchAPIItem`.

## 4.5 Что произойдёт при сбое upstream

### Dense API 429

- `embed_dense_many_safe` ловит `HTTPStatusError(429)`, пишет WARN, возвращает `[]`.
- `dense_vectors = []`, `sparse_vectors = [...]`.
- `qdrant_search_safe` вызывается только со sparse-ветками.
- Retrieval работает, но только по keyword-матчам.
- PipelineTrace: `fallbacks=["dense:empty_or_error"]`, `status=degraded`.
- HTTP 200, клиент получает результат (может быть чуть хуже recall).

### Reranker 429

- `rerank_points` ловит 429, пишет WARN, возвращает `points` как есть.
- `points is reranked` → `True` (object identity) → trace пишет fallback.
- Finalize идёт по retrieval-order.
- HTTP 200.

### Qdrant упал

- `qdrant_search_safe` ловит любое exception, возвращает `None`.
- `best_points = None` → early return с `results=[]`.
- HTTP 200 с пустым ответом лучше, чем 500.

Это проверяется `scripts/chaostest.py` - он перезапускает search-service
с битыми URL для каждого upstream по очереди и проверяет, что:
- `/search` отвечает 200;
- в `/metrics` нужный `search_fallbacks_total{stage=...}` инкрементировался.

## 4.6 Timing breakdown (real numbers из loadtest.py)

Типичный one-shot `/search` с прогретыми кешами:

```
embed   ~ 120ms   (dense batch через внешний API)
qdrant  ~ 200ms   (11 prefetch + DBSF fusion + RETRIEVE_K=150)
rescore ~ 40ms    (чистый python по 150 point'ам)
rerank  ~ 400ms   (nemotron batch 20 candidates)
assemble~ 30ms
------------------
total   ~ 800ms   p50
```

При `429` на dense: -120ms embed и возможный loss recall, но p50 не страдает.
При `429` на rerank: -400ms total; nDCG падает относительно blended, но
recall@50 не меняется.
