from contextlib import asynccontextmanager
from typing import Any
import asyncio

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from chunking import build_chunks
from config import HOST, PORT, UVICORN_WORKERS, logger
from schemas import IndexAPIRequest, IndexAPIResponse, SparseEmbeddingRequest
from sparse import embed_sparse_texts, get_sparse_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(get_sparse_model)
    logger.info("Sparse model warmed up")
    yield


app = FastAPI(title="Index Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    # Sync handler: FastAPI runs it in a worker thread, so CPU-bound
    # chunking of large payloads does not block the event loop.
    return IndexAPIResponse(
        results=build_chunks(
            payload.data.chat,
            payload.data.overlap_messages,
            payload.data.new_messages,
        )
    )


@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest) -> dict[str, Any]:
    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=UVICORN_WORKERS,
    )


if __name__ == "__main__":
    main()
