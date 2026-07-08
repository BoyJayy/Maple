import asyncio
import re
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient, models

from config import (
    ASSEMBLE_BLOCK_HIT_WEIGHT,
    ASSEMBLE_BLOCK_INDEX_PENALTY,
    DENSE_MODEL_NAME,
    DENSE_PREFETCH_K,
    DENSE_QUERY_PREFIX,
    FINAL_MESSAGE_LIMIT,
    FUSION_MODE,
    QDRANT_COLLECTION_NAME,
    QDRANT_DENSE_VECTOR_NAME,
    QDRANT_SPARSE_VECTOR_NAME,
    RERANK_ENABLED,
    RERANK_MAX_DOC_CHARS,
    RERANK_MODEL_NAME,
    RERANK_TOP_K,
    RESCORE_CONTEXT_HIT_WEIGHT,
    RESCORE_MESSAGE_HIT_WEIGHT,
    RESCORE_METADATA_HIT_WEIGHT,
    RESCORE_RANK_BONUS_MAX,
    RESCORE_RANK_BONUS_STEP,
    RETRIEVE_K,
    SPARSE_MODEL_LANGUAGE,
    SPARSE_MODEL_NAME,
    SPARSE_PREFETCH_K,
    TIME_FILTER_ENABLED,
    logger,
)
from querying import (
    SearchContext,
    build_search_context,
    count_stem_hits,
    dedupe_message_ids,
    normalize_text,
)
from schemas import SearchAPIRequest, SparseVector


MESSAGE_BLOCK_SPLIT_RE = re.compile(r"\n\n(?=\[\d{4}-\d{2}-\d{2} )")


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    logger.info("Loading dense model %s", DENSE_MODEL_NAME)
    return TextEmbedding(model_name=DENSE_MODEL_NAME)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading sparse model %s (language=%s)", SPARSE_MODEL_NAME, SPARSE_MODEL_LANGUAGE)
    if SPARSE_MODEL_NAME == "Qdrant/bm25":
        return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME, language=SPARSE_MODEL_LANGUAGE)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@lru_cache(maxsize=1)
def get_reranker() -> Any:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    logger.info("Loading reranker model %s", RERANK_MODEL_NAME)
    return TextCrossEncoder(model_name=RERANK_MODEL_NAME)


async def embed_dense(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    prefixed = [f"{DENSE_QUERY_PREFIX}{text}" for text in texts]

    def _run() -> list[list[float]]:
        return [vector.tolist() for vector in get_dense_model().embed(prefixed)]

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


def build_time_filter(ctx: SearchContext) -> models.Filter | None:
    if not TIME_FILTER_ENABLED or ctx.time_range is None:
        return None
    start, end = ctx.time_range
    return models.Filter(
        must=[
            models.FieldCondition(key="metadata.start", range=models.Range(lte=end)),
            models.FieldCondition(key="metadata.end", range=models.Range(gte=start)),
        ]
    )


async def qdrant_search(
    qdrant_client: AsyncQdrantClient,
    *,
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
    fusion: str,
    query_filter: models.Filter | None = None,
) -> list[Any]:
    prefetch: list[models.Prefetch] = []
    for dense_vector in dense_vectors:
        prefetch.append(
            models.Prefetch(
                query=dense_vector,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
                filter=query_filter,
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
                filter=query_filter,
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


def extract_stored_blocks(point: Any) -> list[tuple[str, str]]:
    """Return (message_id, text) pairs stored by the index service, if present."""
    blocks = get_metadata(point).get("message_blocks")
    if not isinstance(blocks, list):
        return []
    pairs: list[tuple[str, str]] = []
    for block in blocks:
        if not isinstance(block, dict):
            return []
        message_id = str(block.get("message_id") or "")
        if not message_id:
            return []
        pairs.append((message_id, str(block.get("text") or "")))
    return pairs


def split_sections(page_content: str) -> tuple[str, str]:
    if "MESSAGES:" not in page_content:
        return "", page_content
    before_messages, messages = page_content.split("MESSAGES:", 1)
    context = ""
    if "CONTEXT:" in before_messages:
        _, context = before_messages.split("CONTEXT:", 1)
    return normalize_text(context).lower(), normalize_text(messages).lower()


def extract_messages_section(page_content: str) -> str:
    if "MESSAGES:" not in page_content:
        return page_content
    return page_content.split("MESSAGES:", 1)[1].strip()


def extract_message_blocks(page_content: str) -> list[str]:
    if "MESSAGES:" not in page_content:
        return []
    messages_text = page_content.split("MESSAGES:", 1)[1].strip()
    return [block.strip() for block in MESSAGE_BLOCK_SPLIT_RE.split(messages_text) if block.strip()]


def rank_bonus(rank: int) -> float:
    return max(0.0, RESCORE_RANK_BONUS_MAX - rank * RESCORE_RANK_BONUS_STEP)


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

    message_hits = count_stem_hits(message_text, ctx.exact_stems)
    context_hits = count_stem_hits(context_text, ctx.exact_stems)
    metadata_hits = count_stem_hits(metadata_text, ctx.exact_stems)

    return (
        rank_bonus(rank)
        + (message_hits * RESCORE_MESSAGE_HIT_WEIGHT)
        + (context_hits * RESCORE_CONTEXT_HIT_WEIGHT)
        + (metadata_hits * RESCORE_METADATA_HIT_WEIGHT)
    )


def rescore_points(ctx: SearchContext, points: list[Any]) -> list[Any]:
    scored = [
        (score_point(ctx, point, rank=index), -index, point)
        for index, point in enumerate(points)
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [point for _, _, point in scored]


async def rerank_points(ctx: SearchContext, points: list[Any]) -> list[Any]:
    if len(points) <= 1:
        return points

    head = points[:RERANK_TOP_K]
    tail = points[RERANK_TOP_K:]
    documents = [
        extract_messages_section(str(get_payload(point).get("page_content") or ""))[:RERANK_MAX_DOC_CHARS]
        for point in head
    ]

    def _run() -> list[float]:
        return [float(score) for score in get_reranker().rerank(ctx.primary_query, documents)]

    scores = await asyncio.to_thread(_run)
    reranked = [
        point
        for _, _, point in sorted(
            ((score, -index, point) for index, (score, point) in enumerate(zip(scores, head, strict=True))),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
    ]
    return [*reranked, *tail]


def assemble_message_ids(ctx: SearchContext, points: list[Any], *, limit: int) -> list[str]:
    scored_messages: list[tuple[float, int, int, str]] = []

    for point_rank, point in enumerate(points):
        message_ids = extract_message_ids(point)
        if not message_ids:
            continue

        point_bonus = rank_bonus(point_rank)

        stored_blocks = extract_stored_blocks(point)
        if stored_blocks:
            for block_index, (message_id, block_text) in enumerate(stored_blocks):
                block_score = (
                    point_bonus
                    + (count_stem_hits(block_text, ctx.exact_stems) * ASSEMBLE_BLOCK_HIT_WEIGHT)
                    - (block_index * ASSEMBLE_BLOCK_INDEX_PENALTY)
                )
                scored_messages.append((block_score, -point_rank, -block_index, message_id))
            continue

        blocks = extract_message_blocks(str(get_payload(point).get("page_content") or ""))
        if len(blocks) == len(message_ids):
            for block_index, (message_id, block) in enumerate(zip(message_ids, blocks, strict=True)):
                block_score = (
                    point_bonus
                    + (count_stem_hits(block, ctx.exact_stems) * ASSEMBLE_BLOCK_HIT_WEIGHT)
                    - (block_index * ASSEMBLE_BLOCK_INDEX_PENALTY)
                )
                scored_messages.append((block_score, -point_rank, -block_index, message_id))
            continue

        for block_index, message_id in enumerate(message_ids):
            fallback_score = point_bonus - (block_index * ASSEMBLE_BLOCK_INDEX_PENALTY)
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
    skip_rerank: bool = False,
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
    query_filter = build_time_filter(ctx)
    points = await qdrant_search(
        qdrant_client,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        fusion=fusion or FUSION_MODE,
        query_filter=query_filter,
    )
    if not points and query_filter is not None:
        # The time filter is a precision optimization, not a correctness gate:
        # question dates may refer to content rather than message time, and
        # collections ingested before the integer start/end migration do not
        # match Range conditions at all. Zero filtered hits -> retry unfiltered.
        logger.warning("Time-filtered search returned no points, retrying without time filter")
        points = await qdrant_search(
            qdrant_client,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            fusion=fusion or FUSION_MODE,
            query_filter=None,
        )
    if not points:
        return [], {}

    # Rescoring and assembly stem every candidate's text: CPU-bound, so keep
    # them off the event loop (text_stems memoizes repeats across stages).
    stages: dict[str, list[str]] = {}
    if collect_stages:
        stages["retrieval"] = await asyncio.to_thread(
            assemble_message_ids, ctx, points, limit=FINAL_MESSAGE_LIMIT
        )

    if not skip_rescore:
        points = await asyncio.to_thread(rescore_points, ctx, points)
        if collect_stages:
            stages["rescored"] = await asyncio.to_thread(
                assemble_message_ids, ctx, points, limit=FINAL_MESSAGE_LIMIT
            )

    if RERANK_ENABLED and not skip_rerank:
        points = await rerank_points(ctx, points)
        if collect_stages:
            stages["reranked"] = await asyncio.to_thread(
                assemble_message_ids, ctx, points, limit=FINAL_MESSAGE_LIMIT
            )

    final_message_ids = await asyncio.to_thread(
        assemble_message_ids, ctx, points, limit=FINAL_MESSAGE_LIMIT
    )
    return final_message_ids, stages
