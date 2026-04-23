import logging
import os


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1600"))
OVERLAP_MESSAGE_COUNT = int(os.getenv("OVERLAP_MESSAGE_COUNT", "2"))
MAX_TIME_GAP_SECONDS = int(os.getenv("MAX_TIME_GAP_SECONDS", str(3 * 60 * 60)))

LONG_MESSAGE_CHAR_THRESHOLD = int(os.getenv("LONG_MESSAGE_CHAR_THRESHOLD", "1400"))
LONG_MESSAGE_LINE_THRESHOLD = int(os.getenv("LONG_MESSAGE_LINE_THRESHOLD", "30"))
TECHNICAL_PREVIEW_LINES = int(os.getenv("TECHNICAL_PREVIEW_LINES", "18"))
TECHNICAL_PREVIEW_CHARS = int(os.getenv("TECHNICAL_PREVIEW_CHARS", "1800"))
SPLIT_MESSAGE_CHAR_THRESHOLD = int(os.getenv("SPLIT_MESSAGE_CHAR_THRESHOLD", "1200"))
SPLIT_SEGMENT_TARGET_CHARS = int(os.getenv("SPLIT_SEGMENT_TARGET_CHARS", "650"))

SHORT_ACK_MESSAGES = {
    "+",
    "++",
    "ага",
    "да",
    "нет",
    "ок",
    "окей",
    "понял",
    "спасибо",
    "ясно",
    "yes",
    "no",
    "ok",
    "thanks",
    "thx",
}

SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "2"))

TECHNICAL_TRACE_MARKERS = (
    "traceback",
    "exception",
    "stack trace",
    "goroutine ",
    "runtime.",
    "pc=",
    "sigabrt",
    "panic:",
    ".go:",
    ".py:",
)
