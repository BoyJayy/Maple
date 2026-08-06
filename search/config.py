import logging
import os


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "messages")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")

DENSE_MODEL_NAME = os.getenv(
    "DENSE_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DENSE_VECTOR_SIZE = int(os.getenv("DENSE_VECTOR_SIZE", "384"))
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")
# BM25 stemmer/stopwords language; must match the value used at index time.
SPARSE_MODEL_LANGUAGE = os.getenv("SPARSE_MODEL_LANGUAGE", "russian")


def default_dense_query_prefix(model_name: str) -> str:
    # E5 models expect "query: " on queries; the matching "passage: " document
    # prefix lives in eval/ingest.py, which is the only place documents are embedded.
    return "query: " if "e5" in model_name.lower() else ""


DENSE_QUERY_PREFIX = os.getenv("DENSE_QUERY_PREFIX", default_dense_query_prefix(DENSE_MODEL_NAME))

FUSION_MODE = os.getenv("FUSION_MODE", "dbsf")
DENSE_PREFETCH_K = int(os.getenv("DENSE_PREFETCH_K", "40"))
SPARSE_PREFETCH_K = int(os.getenv("SPARSE_PREFETCH_K", "40"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "60"))
MAX_DENSE_QUERIES = int(os.getenv("MAX_DENSE_QUERIES", "3"))
MAX_SPARSE_QUERIES = int(os.getenv("MAX_SPARSE_QUERIES", "3"))
FINAL_MESSAGE_LIMIT = int(os.getenv("FINAL_MESSAGE_LIMIT", "50"))

# Point-level rescoring is optional. The balanced default leaves fused point
# order intact: on both committed eval sets that is slightly better and avoids
# a CPU-heavy pass. Message-level scoring in assemble_message_ids remains on.
POINT_RESCORE_ENABLED = os.getenv("POINT_RESCORE_ENABLED", "0") == "1"

# Point rescoring works on fusion ranks (not raw scores), so weights do not depend on FUSION_MODE.
RESCORE_RANK_BONUS_MAX = float(os.getenv("RESCORE_RANK_BONUS_MAX", "0.2"))
RESCORE_RANK_BONUS_STEP = float(os.getenv("RESCORE_RANK_BONUS_STEP", "0.005"))
RESCORE_MESSAGE_HIT_WEIGHT = float(os.getenv("RESCORE_MESSAGE_HIT_WEIGHT", "0.04"))
RESCORE_CONTEXT_HIT_WEIGHT = float(os.getenv("RESCORE_CONTEXT_HIT_WEIGHT", "0.01"))
RESCORE_METADATA_HIT_WEIGHT = float(os.getenv("RESCORE_METADATA_HIT_WEIGHT", "0.02"))
ASSEMBLE_BLOCK_HIT_WEIGHT = float(os.getenv("ASSEMBLE_BLOCK_HIT_WEIGHT", "0.05"))
ASSEMBLE_BLOCK_INDEX_PENALTY = float(os.getenv("ASSEMBLE_BLOCK_INDEX_PENALTY", "0.01"))

# Time filter built from question.date_range / date_mentions. Hard filtering is
# the quality-tested default; the collection-bounds guard prevents clearly
# disjoint ranges from suppressing otherwise relevant candidates.
TIME_FILTER_ENABLED = os.getenv("TIME_FILTER_ENABLED", "1") == "1"
TIME_FILTER_MARGIN_SECONDS = int(os.getenv("TIME_FILTER_MARGIN_SECONDS", "86400"))
TIME_FILTER_MODE = os.getenv("TIME_FILTER_MODE", "hard").lower()
if TIME_FILTER_MODE not in {"hard", "soft"}:
    raise ValueError("TIME_FILTER_MODE must be 'hard' or 'soft'")
TIME_FILTER_BOUNDS_GUARD_ENABLED = os.getenv("TIME_FILTER_BOUNDS_GUARD_ENABLED", "1") == "1"
TIME_FILTER_BOUNDS_CACHE_SECONDS = float(os.getenv("TIME_FILTER_BOUNDS_CACHE_SECONDS", "60"))
TIME_FILTER_BOUNDS_RETRY_SECONDS = float(os.getenv("TIME_FILTER_BOUNDS_RETRY_SECONDS", "5"))
TIME_FILTER_SOFT_DENSE_QUERIES = int(os.getenv("TIME_FILTER_SOFT_DENSE_QUERIES", "1"))
TIME_FILTER_SOFT_SPARSE_QUERIES = int(os.getenv("TIME_FILTER_SOFT_SPARSE_QUERIES", "1"))
TIME_FILTER_SOFT_PREFETCH_K = int(os.getenv("TIME_FILTER_SOFT_PREFETCH_K", "20"))

# Optional cross-encoder reranker over the current top candidates.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0") == "1"
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "jinaai/jina-reranker-v2-base-multilingual")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "30"))
RERANK_MAX_DOC_CHARS = int(os.getenv("RERANK_MAX_DOC_CHARS", "2000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")
