# Architecture

`Maple` has three runtime parts:
- `index`
- `search`
- `Qdrant`

At a high level the flow looks like this:

```text
chat payload
  -> index
  -> chunks + metadata
  -> dense and sparse vectors
  -> Qdrant

question
  -> search
  -> dense and sparse retrieval
  -> fusion + rescoring
  -> message_ids
```

## Components

### index

`index` accepts raw chat data and turns it into searchable chunks.

Responsibilities:
- normalize messages;
- filter empty or noisy entries;
- split long messages;
- preserve overlap between batches;
- build `page_content`, `dense_content` and `sparse_content`;
- provide sparse vectors through `/sparse_embedding`.

### search

`search` accepts a question and builds a small set of dense and sparse queries.

Responsibilities:
- prepare the primary query;
- extract exact terms from question text, keywords and entities;
- compute local dense and sparse embeddings;
- query Qdrant with both retrieval modes;
- merge results with fusion;
- apply lightweight local rescoring;
- return final `message_ids`.

### Qdrant

Qdrant stores one point per chunk. Each point contains:
- a dense vector;
- a sparse vector;
- `page_content`;
- metadata with `message_ids`, per-message `message_blocks`, participants,
  mentions and integer time bounds (`start` / `end`).

Payload indexes: `metadata.chat_id` (keyword), `metadata.start` and
`metadata.end` (integer, used by the search-time date filter).

## Content fields

Each chunk contains three text views.

### page_content

Human readable text used for:
- payload storage;
- debugging;
- block level message assembly.

### dense_content

Compact semantic text used for dense embeddings.

### sparse_content

Token preserving text used for sparse embeddings and exact term matching.

## Retrieval model

The search pipeline is hybrid:
- dense retrieval finds semantic matches;
- sparse retrieval finds exact terms, names, links and identifiers;
- an optional time filter narrows candidates when the question mentions dates;
- fusion combines both result sets;
- rescoring (rank-based, morphology-aware term matching) gives a small boost
  to points with stronger exact matches in the message body or metadata;
- an optional local cross-encoder reranker reorders the top candidates.

## Local first setup

The project does not depend on external APIs.

Models used by default:
- dense: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- sparse: `Qdrant/bm25` (`language=russian`)
- reranker (optional, off by default): `jinaai/jina-reranker-v2-base-multilingual`

All are loaded locally through `fastembed`.

## Repository structure

### index

- `main.py` — FastAPI app and routes
- `config.py` — chunking settings
- `schemas.py` — request and response models
- `chunking.py` — normalization and chunk building
- `sparse.py` — local sparse embedding wrapper

### search

- `main.py` — FastAPI app and routes
- `config.py` — runtime settings
- `schemas.py` — API models
- `querying.py` — query preparation and exact term extraction
- `pipeline.py` — embeddings, Qdrant retrieval, fusion and rescoring

### eval

- `ingest.py` — fills Qdrant from a local dataset
- `run.py` — runs offline evaluation against `search`
- `metrics.py` — evaluation metrics
