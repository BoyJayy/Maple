import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = os.getenv("RERANKER_URL")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
REQUIRED_ENV_VARS = [
    "EMBEDDINGS_DENSE_URL",
    "RERANKER_URL",
    "QDRANT_URL",
]
 
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")


def validate_required_env() -> None:
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")

    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError("Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set")

    missing_env_vars = [
        name for name in REQUIRED_ENV_VARS if os.getenv(name) is None or os.getenv(name) == ""
    ]
    if not missing_env_vars:
        return

    logger.error("Empty required env vars: %s", ", ".join(missing_env_vars))
    raise RuntimeError(f"Empty required env vars: {', '.join(missing_env_vars)}")


validate_required_env()


def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}

    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return kwargs


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class Entities(BaseModel):
    people: list[str] | None = None
    emails: list[str] | None = None
    documents: list[str] | None = None
    names: list[str] | None = None
    links: list[str] | None = None


class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = None
    hyde: list[str] | None = None
    keywords: list[str] | None = None
    entities: Entities | None = None
    date_mentions: list[str] | None = None
    date_range: DateRange | None = None
    search_text: str = ""


class SearchAPIRequest(BaseModel):
    question: Question


class SearchAPIItem(BaseModel):
    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]


class DenseEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class DenseEmbeddingResponse(BaseModel):
    data: list[DenseEmbeddingItem]


class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]

# Метадата чанков в Qdrant'e, по которой вы можете фильтровать
class ChunkMetadata(BaseModel):
    chat_name: str
    chat_type: str # channel, group, private, thread
    chat_id: str
    chat_sn: str
    thread_sn: str | None = None
    message_ids: list[str]
    start: str
    end: str
    participants: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    contains_forward: bool = False
    contains_quote: bool = False


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=API_KEY,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.1.0", lifespan=lifespan)


DENSE_PREFETCH_K = 25
SPARSE_PREFETCH_K = 35
RETRIEVE_K = 40
RERANK_LIMIT = 8
RERANK_MAX_TEXT_CHARS = 1200
FINAL_MESSAGE_LIMIT = 50
MAX_DENSE_QUERIES = 5
MAX_SPARSE_QUERIES = 3
WHITESPACE_RE = re.compile(r"\s+")


def normalize_query_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def unique_texts(texts: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for text in texts:
        normalized = normalize_query_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        items.append(normalized)
        if limit is not None and len(items) >= limit:
            break
    return items


def collect_entity_terms(entities: Entities | None) -> list[str]:
    if entities is None:
        return []

    values = [
        *(entities.people or []),
        *(entities.emails or []),
        *(entities.documents or []),
        *(entities.names or []),
        *(entities.links or []),
    ]
    return unique_texts(values)


def build_primary_query(question: Question) -> str:
    return normalize_query_text(question.search_text or question.text)


def build_dense_queries(question: Question) -> list[str]:
    candidates = [
        question.search_text,
        question.text,
        *((question.variants or [])[:3]),
        *((question.hyde or [])[:2]),
    ]
    return unique_texts(candidates, limit=MAX_DENSE_QUERIES)


def build_sparse_queries(question: Question) -> list[str]:
    primary = build_primary_query(question)
    entity_terms = collect_entity_terms(question.entities)
    focus_terms = unique_texts(
        [
            *(question.keywords or []),
            *entity_terms,
            *(question.date_mentions or []),
            question.asker,
        ]
    )
    exact_focus = " ".join(focus_terms)
    combined = "\n".join(part for part in [primary, exact_focus] if part)
    candidates = [
        combined,
        exact_focus,
        question.text,
    ]
    return unique_texts(candidates, limit=MAX_SPARSE_QUERIES)


def build_rerank_query(question: Question) -> str:
    return normalize_query_text(question.text or build_primary_query(question))


def dedupe_message_ids(message_ids: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []
    for message_id in message_ids:
        if message_id in seen:
            continue
        seen.add(message_id)
        unique_ids.append(message_id)
        if len(unique_ids) >= limit:
            break
    return unique_ids


def trim_rerank_text(text: str, *, limit: int) -> str:
    normalized = normalize_query_text(text)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()} ..."


async def embed_dense_many(
    client: httpx.AsyncClient,
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []

    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": texts,
        },
    )
    response.raise_for_status()

    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")

    return [item.embedding for item in sorted(payload.data, key=lambda item: item.index)]


async def embed_sparse_many(texts: list[str]) -> list[SparseVector]:
    if not texts:
        return []

    vectors = await asyncio.to_thread(lambda: list(get_sparse_model().embed(texts)))
    if not vectors:
        raise ValueError("Sparse embedding response is empty")

    return [
        SparseVector(
            indices=[int(index) for index in item.indices.tolist()],
            values=[float(value) for value in item.values.tolist()],
        )
        for item in vectors
    ]


async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
) -> Any | None:
    prefetch: list[models.Prefetch] = [
        models.Prefetch(
            query=dense_vector,
            using=QDRANT_DENSE_VECTOR_NAME,
            limit=DENSE_PREFETCH_K,
        )
        for dense_vector in dense_vectors
    ]
    prefetch.extend(
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            ),
            using=QDRANT_SPARSE_VECTOR_NAME,
            limit=SPARSE_PREFETCH_K,
        )
        for sparse_vector in sparse_vectors
    )

    if not prefetch:
        return None

    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
    )

    if not response.points:
        return None

    return response.points


def extract_message_ids(point: Any) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    message_ids = metadata.get("message_ids") or []

    return [str(message_id) for message_id in message_ids]


async def get_rerank_scores(
    client: httpx.AsyncClient,
    label: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []

    # Rerank endpoint возвращает score для пары query -> candidate text.
    response = await client.post(
        RERANKER_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": label,
            "text_2": targets,
        },
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data") or []

    return [float(sample["score"]) for sample in data]


async def rerank_points(
    client: httpx.AsyncClient,
    query: str,
    points: list[Any],
) -> list[Any]:
    if not points:
        return []

    rerank_candidates = points[:RERANK_LIMIT]
    untouched_tail = points[RERANK_LIMIT:]
    rerank_targets = [
        trim_rerank_text(str((point.payload or {}).get("page_content") or ""), limit=RERANK_MAX_TEXT_CHARS)
        for point in rerank_candidates
    ]

    try:
        scores = await get_rerank_scores(client, query, rerank_targets)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            logger.warning(
                "Reranker rate-limited the request; using retrieval order fallback for %s candidates",
                len(rerank_candidates),
            )
            return points
        raise

    reranked_candidates = [
        point
        for _, point in sorted(
            zip(scores, rerank_candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    return reranked_candidates + untouched_tail


# Ваш сервис должен имплементировать оба этих метода
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    primary_query = build_primary_query(payload.question)
    if not primary_query:
        raise HTTPException(status_code=400, detail="question.text is required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    dense_queries = build_dense_queries(payload.question)
    sparse_queries = build_sparse_queries(payload.question)

    dense_vectors, sparse_vectors = await asyncio.gather(
        embed_dense_many(client, dense_queries),
        embed_sparse_many(sparse_queries),
    )
    best_points = await qdrant_search(qdrant, dense_vectors, sparse_vectors)

    if best_points is None:
        return SearchAPIResponse(results=[])

    rerank_query = build_rerank_query(payload.question)
    best_points = await rerank_points(client, rerank_query, list(best_points))

    message_ids: list[str] = []
    for point in best_points:
        message_ids += extract_message_ids(point)

    final_message_ids = dedupe_message_ids(message_ids, limit=FINAL_MESSAGE_LIMIT)
    if not final_message_ids:
        return SearchAPIResponse(results=[])

    return SearchAPIResponse(
        results=[SearchAPIItem(message_ids=final_message_ids)]
    )


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    detail = str(exc) or repr(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return JSONResponse(status_code=500, content={"detail": detail})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
