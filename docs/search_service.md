# Search Service

The `search` service accepts a question, runs hybrid retrieval against Qdrant and returns ranked `message_ids`.

The current implementation is local-first:
- dense embeddings are computed locally with `fastembed`;
- sparse embeddings are computed locally with `fastembed`;
- the optional reranker is a local cross-encoder, also loaded with `fastembed`.

## Responsibilities

- prepare the primary search query;
- build dense and sparse query variants;
- extract a time range from `date_range` / `date_mentions`;
- compute query embeddings locally;
- fetch candidates from Qdrant (optionally filtered by time);
- combine dense and sparse retrieval with fusion;
- optionally apply point-level local rescoring;
- optionally rerank top candidates with a local cross-encoder;
- assemble and score final `message_ids`.

## Module structure

- `search/main.py` — FastAPI app and endpoints
- `search/config.py` — runtime settings
- `search/schemas.py` — API models
- `search/querying.py` — query preparation
- `search/pipeline.py` — embeddings, retrieval, fusion, rescoring and assembly

## Endpoints

- `GET /health` — liveness
- `GET /ready` — readiness: the Qdrant collection exists and its dense vector
  size matches `DENSE_VECTOR_SIZE` (catches a model switch without reingest)
- `POST /search`
- `POST /_debug/search` (`fusion` defaults to the service `FUSION_MODE`)

## Search flow

```text
question
  -> build search context (queries, exact terms, stems, time range)
  -> build dense queries
  -> build sparse queries
  -> local embeddings
  -> Qdrant hybrid retrieval (+ optional time filter)
  -> fusion
  -> optional point-level rescoring
  -> optional cross-encoder rerank
  -> message-level scoring and assembly
```

## Query preparation

The primary query is built from:
- `question.search_text`, if non-empty after normalization;
- otherwise `question.text`.

Additional signal comes from:
- `keywords`
- `entities`
- `date_mentions`
- `variants`
- `asker`

The service extracts a compact set of exact terms and uses them in two places:
- as a compact sparse query;
- as local exact match signals during rescoring.

Term matching is morphology aware: tokens are stemmed with Snowball
(Russian for Cyrillic tokens, English otherwise) and compared on word
boundaries, so `релиз` matches `релизе` but `код` does not match `кодекс`.
Tokens with digits or identifier punctuation (emails, links, versions)
are matched verbatim; sentence punctuation glued to token edges is stripped,
and compound identifiers also match by their word parts (`plan` hits
`release-plan.docx`).

Exact-term extraction prioritizes high-signal sources (entities, then
keywords, then date mentions, then question text) and drops a small set of
Russian/English filler words, so the 12-term budget is not consumed by
`подскажи`, `пожалуйста`, `что` and similar tokens before entities are
reached.

A valid structured `date_range` is authoritative. `date_mentions` are parsed
only when the structured range is absent or invalid, so a precise day is not
accidentally widened by a looser year or month mention.

## Time filter

If the question carries `date_range` or ISO-like dates inside `date_mentions`
(`YYYY`, `YYYY-MM`, `YYYY-MM-DD`), the service builds a time window, widens it
by `TIME_FILTER_MARGIN_SECONDS` and builds a range condition over
`metadata.start` / `metadata.end`. A date-only upper bound is treated as the
inclusive end of that day. The feature is controlled by
`TIME_FILTER_ENABLED` and requires integer `start` / `end` payload fields plus
their payload indexes.

`TIME_FILTER_MODE=hard` is the quality-tested default. It applies the range to
every dense and sparse branch. If the filtered search returns zero points, the
service retries once without the filter, so an inaccurate date does not turn
into an empty response.

With `TIME_FILTER_BOUNDS_GUARD_ENABLED=1`, the service lazily caches the
collection-wide earliest `metadata.start` and latest `metadata.end`. A query
range that cannot overlap that envelope skips filtering before retrieval. The
cache is single-flight, uses success/retry TTLs, and fails open: a bounds lookup
failure leaves normal hard filtering and its zero-result fallback in place.

`TIME_FILTER_MODE=soft` remains available for experiments. It preserves all
normal unfiltered branches and adds filtered copies of the strongest branches
as a fusion signal. Their count and limit are controlled by
`TIME_FILTER_SOFT_DENSE_QUERIES`, `TIME_FILTER_SOFT_SPARSE_QUERIES` and
`TIME_FILTER_SOFT_PREFETCH_K`. It is not the default because the dialogues
benchmark ranked worse with the additional fusion branches.

## Dense queries

Dense queries are built from a small set of normalized strings, typically:
- primary query;
- raw `question.text`;
- the first variant, if present;
- compact exact term query.

The number of dense queries is limited by `MAX_DENSE_QUERIES`.

## Sparse queries

Sparse queries are built from:
- exact terms;
- primary query;
- raw `question.text`;
- the first variant, if present.

The number of sparse queries is limited by `MAX_SPARSE_QUERIES`.

## Retrieval

The service embeds query variants locally, then sends them to Qdrant as multiple prefetch branches:
- dense branches use the `dense` vector field;
- sparse branches use the `sparse` vector field.

The result sets are merged with fusion:
- `dbsf`
- or `rrf`

Main parameters:
- `DENSE_PREFETCH_K`
- `SPARSE_PREFETCH_K`
- `RETRIEVE_K`
- `FUSION_MODE`

## Rescoring

Point-level rescoring is controlled by `POINT_RESCORE_ENABLED` and is disabled
by default. The committed evaluation sets ranked slightly better without it,
and skipping the pass also lowers CPU cost.

Signals include:
- exact term hits in the message block;
- exact term hits in the context block;
- exact term hits in metadata such as participants and mentions;
- original point rank after fusion.

The rescore works on fusion **ranks**, not raw fusion scores, so its weights
behave the same under `dbsf` and `rrf`. All weights are configurable:
`RESCORE_RANK_BONUS_MAX`, `RESCORE_RANK_BONUS_STEP`,
`RESCORE_MESSAGE_HIT_WEIGHT`, `RESCORE_CONTEXT_HIT_WEIGHT`,
`RESCORE_METADATA_HIT_WEIGHT`.

These `RESCORE_*` values are used only when point rescoring is enabled. Final
message assembly still applies message-block scoring through `ASSEMBLE_*`;
`no_rescore=true` skips only the optional point-level stage.

## Reranker

An optional cross-encoder rerank stage runs after fusion or point rescoring. It
is disabled by default and controlled by:
- `RERANK_ENABLED=1`
- `RERANK_MODEL_NAME` (default `jinaai/jina-reranker-v2-base-multilingual`)
- `RERANK_TOP_K` — how many top candidates are reranked
- `RERANK_MAX_DOC_CHARS` — per-document text budget

The reranker scores `(primary_query, chunk messages)` pairs locally and
reorders the top candidates; the rest keep their order. It adds noticeable
latency and a ~1.1 GB model download, so enable it deliberately and measure
with `eval/run.py --stages`.

Reranking is fail-open: a model or runtime failure is logged and the fused
candidates continue to final assembly. A warmup failure does not stop service
startup; reranking stays disabled until the next restart.

## Final assembly

The final response is built from payload `message_ids`.

Per message ordering inside a chunk uses `metadata.message_blocks`
(`{message_id, text}` pairs written at index time), so the mapping stays
correct even when one message is split into several fragments. For old
points without `message_blocks` the service falls back to parsing
`page_content`, and then to plain `message_ids` order.

The final list is:
- deduplicated;
- limited by `FINAL_MESSAGE_LIMIT`;
- returned as `results[].message_ids`.

## Debug endpoint

`POST /_debug/search` returns both the final output and intermediate stage outputs.

Useful query parameters:
- `fusion=dbsf`
- `fusion=rrf`
- `max_dense=1`
- `max_sparse=2`
- `no_rescore=true`
- `no_rerank=true`

This endpoint is intended for local analysis and A/B testing.
The `rescored` stage appears only when `POINT_RESCORE_ENABLED=1` and point
rescoring was not skipped. The `reranked` stage appears only when reranking is
enabled and succeeds.

## Default models

- dense: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- sparse: `Qdrant/bm25` with `language=russian` (Snowball stemming; Latin tokens pass through unchanged)
- reranker (optional): `jinaai/jina-reranker-v2-base-multilingual`

All can be overridden with environment variables. `SPARSE_MODEL_LANGUAGE`
must match the value used at index time — changing it requires reingesting.

Switching the dense model (for example to `intfloat/multilingual-e5-large`,
the strongest multilingual dense model available in `fastembed`) requires:
- the same `DENSE_MODEL_NAME` in search and ingest;
- a matching `DENSE_VECTOR_SIZE` (1024 for e5-large) and a fresh collection;
- E5 prefixes are applied automatically (`query: ` / `passage: `);
  override with `DENSE_QUERY_PREFIX` / `DENSE_DOCUMENT_PREFIX` if needed.

Models are warmed up at service start, so the first request does not pay
the model load cost.
