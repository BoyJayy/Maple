import asyncio
import re
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient, models

from config import (
    DENSE_MODEL_NAME,
    DENSE_PREFETCH_K,
    FINAL_MESSAGE_LIMIT,
    FUSION_MODE,
    QDRANT_COLLECTION_NAME,
    QDRANT_DENSE_VECTOR_NAME,
    QDRANT_SPARSE_VECTOR_NAME,
    RETRIEVE_K,
    SPARSE_MODEL_NAME,
    SPARSE_PREFETCH_K,
    logger,
)
from querying import SearchContext, build_search_context, dedupe_message_ids, normalize_text
from schemas import SearchAPIRequest, SparseVector


MESSAGE_BLOCK_SPLIT_RE = re.compile(r"\n\n(?=\[\d{4}-\d{2}-\d{2} )")


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    logger.info("Loading dense model %s", DENSE_MODEL_NAME)
    return TextEmbedding(model_name=DENSE_MODEL_NAME)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


async def embed_dense(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    def _run() -> list[list[float]]:
        return [vector.tolist() for vector in get_dense_model().embed(texts)]

    return await asyncio.to_thread(_run)


async def embed_sparse(texts: list[str]) -> list[SparseVector]:
    if not texts:
        return []

    def _run() -> list[SparseVector]:
        vectors: list[SparseVector] = []
        for vector in get_sparse_model().embed(texts):
            vectors.append(
                SparseVector(
                    indices=[int(index) for index in vector.indices.tolist()],
                    values=[float(value) for value in vector.values.tolist()],
                )
            )
        return vectors

    return await asyncio.to_thread(_run)


async def qdrant_search(
    qdrant_client: AsyncQdrantClient,
    *,
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
    fusion: str,
) -> list[Any]:
    prefetch: list[models.Prefetch] = []
    for dense_vector in dense_vectors:
        prefetch.append(
            models.Prefetch(
                query=dense_vector,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
            )
        )
    for sparse_vector in sparse_vectors:
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=SPARSE_PREFETCH_K,
            )
        )

    if not prefetch:
        return []

    fusion_mode = models.Fusion.DBSF if fusion.lower() == "dbsf" else models.Fusion.RRF
    response = await qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=fusion_mode),
        limit=RETRIEVE_K,
        with_payload=True,
    )
    return list(response.points)


def get_payload(point: Any) -> dict[str, Any]:
    payload = getattr(point, "payload", None) or {}
    return payload if isinstance(payload, dict) else {}


def get_metadata(point: Any) -> dict[str, Any]:
    metadata = get_payload(point).get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def extract_message_ids(point: Any) -> list[str]:
    return [str(message_id) for message_id in (get_metadata(point).get("message_ids") or [])]


def split_sections(page_content: str) -> tuple[str, str]:
    if "MESSAGES:" not in page_content:
        return "", page_content
    before_messages, messages = page_content.split("MESSAGES:", 1)
    context = ""
    if "CONTEXT:" in before_messages:
        _, context = before_messages.split("CONTEXT:", 1)
    return normalize_text(context).lower(), normalize_text(messages).lower()


def extract_message_blocks(page_content: str) -> list[str]:
    if "MESSAGES:" not in page_content:
        return []
    messages_text = page_content.split("MESSAGES:", 1)[1].strip()
    return [block.strip() for block in MESSAGE_BLOCK_SPLIT_RE.split(messages_text) if block.strip()]


def count_term_hits(text: str, exact_terms: tuple[str, ...]) -> int:
    lowered = normalize_text(text).lower()
    return sum(1 for term in exact_terms if term and term in lowered)


def score_point(ctx: SearchContext, point: Any, *, rank: int) -> float:
    page_content = str(get_payload(point).get("page_content") or "")
    context_text, message_text = split_sections(page_content)
    metadata = get_metadata(point)
    metadata_text = " ".join(
        [
            *[str(item) for item in (metadata.get("participants") or [])],
            *[str(item) for item in (metadata.get("mentions") or [])],
        ]
    ).lower()

    base_score = float(getattr(point, "score", 0.0) or 0.0)
    message_hits = count_term_hits(message_text, ctx.exact_terms)
    context_hits = count_term_hits(context_text, ctx.exact_terms)
    metadata_hits = count_term_hits(metadata_text, ctx.exact_terms)
    rank_bonus = max(0.0, 0.2 - rank * 0.005)

    return base_score + rank_bonus + (message_hits * 0.04) + (context_hits * 0.01) + (metadata_hits * 0.02)


def rescore_points(ctx: SearchContext, points: list[Any]) -> list[Any]:
    scored = [
        (score_point(ctx, point, rank=index), -index, point)
        for index, point in enumerate(points)
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [point for _, _, point in scored]


def assemble_message_ids(ctx: SearchContext, points: list[Any], *, limit: int) -> list[str]:
    scored_messages: list[tuple[float, int, int, str]] = []

    for point_rank, point in enumerate(points):
        message_ids = extract_message_ids(point)
        if not message_ids:
            continue

        blocks = extract_message_blocks(str(get_payload(point).get("page_content") or ""))
        point_bonus = max(0.0, 0.2 - point_rank * 0.005)

        if len(blocks) == len(message_ids):
            for block_index, (message_id, block) in enumerate(zip(message_ids, blocks, strict=True)):
                block_score = point_bonus + (count_term_hits(block, ctx.exact_terms) * 0.05) - (block_index * 0.01)
                scored_messages.append((block_score, -point_rank, -block_index, message_id))
            continue

        for block_index, message_id in enumerate(message_ids):
            fallback_score = point_bonus - (block_index * 0.01)
            scored_messages.append((fallback_score, -point_rank, -block_index, message_id))

    scored_messages.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    ordered_message_ids = [message_id for _, _, _, message_id in scored_messages]
    return dedupe_message_ids(ordered_message_ids, limit=limit)


async def run_search_pipeline(
    qdrant_client: AsyncQdrantClient,
    payload: SearchAPIRequest,
    *,
    collect_stages: bool = False,
    fusion: str | None = None,
    max_dense: int | None = None,
    max_sparse: int | None = None,
    skip_rescore: bool = False,
) -> tuple[list[str], dict[str, list[str]]]:
    ctx = build_search_context(payload.question)
    if not ctx.primary_query:
        raise ValueError("question.text is required")

    dense_queries = list(ctx.dense_queries)
    sparse_queries = list(ctx.sparse_queries)
    if max_dense is not None:
        dense_queries = dense_queries[: max(0, max_dense)]
    if max_sparse is not None:
        sparse_queries = sparse_queries[: max(0, max_sparse)]

    dense_vectors, sparse_vectors = await asyncio.gather(
        embed_dense(dense_queries),
        embed_sparse(sparse_queries),
    )
    points = await qdrant_search(
        qdrant_client,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        fusion=fusion or FUSION_MODE,
    )
    if not points:
        return [], {}

    stages: dict[str, list[str]] = {}
    if collect_stages:
        stages["retrieval"] = assemble_message_ids(ctx, points, limit=FINAL_MESSAGE_LIMIT)

    if not skip_rescore:
        points = rescore_points(ctx, points)
        if collect_stages:
            stages["rescored"] = assemble_message_ids(ctx, points, limit=FINAL_MESSAGE_LIMIT)

    final_message_ids = assemble_message_ids(ctx, points, limit=FINAL_MESSAGE_LIMIT)
    return final_message_ids, stages
