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


def default_dense_prefixes(model_name: str) -> tuple[str, str]:
    if "e5" in model_name.lower():
        return "query: ", "passage: "
    return "", ""


_DEFAULT_QUERY_PREFIX, _DEFAULT_DOCUMENT_PREFIX = default_dense_prefixes(DENSE_MODEL_NAME)
DENSE_QUERY_PREFIX = os.getenv("DENSE_QUERY_PREFIX", _DEFAULT_QUERY_PREFIX)
DENSE_DOCUMENT_PREFIX = os.getenv("DENSE_DOCUMENT_PREFIX", _DEFAULT_DOCUMENT_PREFIX)

FUSION_MODE = os.getenv("FUSION_MODE", "dbsf")
DENSE_PREFETCH_K = int(os.getenv("DENSE_PREFETCH_K", "40"))
SPARSE_PREFETCH_K = int(os.getenv("SPARSE_PREFETCH_K", "40"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "60"))
MAX_DENSE_QUERIES = int(os.getenv("MAX_DENSE_QUERIES", "3"))
MAX_SPARSE_QUERIES = int(os.getenv("MAX_SPARSE_QUERIES", "3"))
FINAL_MESSAGE_LIMIT = int(os.getenv("FINAL_MESSAGE_LIMIT", "50"))

# Rescoring works on fusion ranks (not raw scores), so weights do not depend on FUSION_MODE.
RESCORE_RANK_BONUS_MAX = float(os.getenv("RESCORE_RANK_BONUS_MAX", "0.2"))
RESCORE_RANK_BONUS_STEP = float(os.getenv("RESCORE_RANK_BONUS_STEP", "0.005"))
RESCORE_MESSAGE_HIT_WEIGHT = float(os.getenv("RESCORE_MESSAGE_HIT_WEIGHT", "0.04"))
RESCORE_CONTEXT_HIT_WEIGHT = float(os.getenv("RESCORE_CONTEXT_HIT_WEIGHT", "0.01"))
RESCORE_METADATA_HIT_WEIGHT = float(os.getenv("RESCORE_METADATA_HIT_WEIGHT", "0.02"))
ASSEMBLE_BLOCK_HIT_WEIGHT = float(os.getenv("ASSEMBLE_BLOCK_HIT_WEIGHT", "0.05"))
ASSEMBLE_BLOCK_INDEX_PENALTY = float(os.getenv("ASSEMBLE_BLOCK_INDEX_PENALTY", "0.01"))

# Time filter built from question.date_range / date_mentions; applied to Qdrant prefetch.
TIME_FILTER_ENABLED = os.getenv("TIME_FILTER_ENABLED", "1") == "1"
TIME_FILTER_MARGIN_SECONDS = int(os.getenv("TIME_FILTER_MARGIN_SECONDS", "86400"))

# Optional cross-encoder reranker over the rescored top candidates.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0") == "1"
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "jinaai/jina-reranker-v2-base-multilingual")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "30"))
RERANK_MAX_DOC_CHARS = int(os.getenv("RERANK_MAX_DOC_CHARS", "2000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")
