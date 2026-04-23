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

FUSION_MODE = os.getenv("FUSION_MODE", "dbsf")
DENSE_PREFETCH_K = int(os.getenv("DENSE_PREFETCH_K", "40"))
SPARSE_PREFETCH_K = int(os.getenv("SPARSE_PREFETCH_K", "40"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "60"))
MAX_DENSE_QUERIES = int(os.getenv("MAX_DENSE_QUERIES", "3"))
MAX_SPARSE_QUERIES = int(os.getenv("MAX_SPARSE_QUERIES", "3"))
FINAL_MESSAGE_LIMIT = int(os.getenv("FINAL_MESSAGE_LIMIT", "50"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")
