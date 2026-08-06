import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from config import (
    DENSE_VECTOR_SIZE,
    HOST,
    PORT,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_DENSE_VECTOR_NAME,
    QDRANT_URL,
    RERANK_ENABLED,
    logger,
)
from pipeline import (
    CollectionTimeBoundsCache,
    get_dense_model,
    get_reranker,
    get_sparse_model,
    run_search_pipeline,
)
from schemas import SearchAPIItem, SearchAPIRequest, SearchAPIResponse


async def warmup_models() -> bool:
    await asyncio.gather(
        asyncio.to_thread(get_dense_model),
        asyncio.to_thread(get_sparse_model),
    )
    if not RERANK_ENABLED:
        logger.info("Models warmed up")
        return False

    try:
        await asyncio.to_thread(get_reranker)
    except Exception:
        # Dense+sparse retrieval remains fully usable when the optional model
        # cannot be downloaded or initialized. Keep it disabled until restart
        # instead of retrying the expensive load on every request.
        logger.exception("Reranker warmup failed; continuing without reranking")
        return False

    logger.info("Models and reranker warmed up")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    app.state.time_bounds_cache = CollectionTimeBoundsCache(app.state.qdrant)
    app.state.reranker_available = await warmup_models()
    try:
        yield
    finally:
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    try:
        exists = await app.state.qdrant.collection_exists(QDRANT_COLLECTION_NAME)
        if not exists:
            return JSONResponse(
                status_code=503,
                content={"status": f"collection {QDRANT_COLLECTION_NAME} not found"},
            )
        info = await app.state.qdrant.get_collection(QDRANT_COLLECTION_NAME)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "qdrant unavailable", "detail": str(exc)})

    vectors = getattr(info.config.params, "vectors", None) or {}
    dense_params = vectors.get(QDRANT_DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
    dense_size = getattr(dense_params, "size", None)
    if dense_size is not None and dense_size != DENSE_VECTOR_SIZE:
        return JSONResponse(
            status_code=503,
            content={
                "status": "dense vector size mismatch",
                "detail": f"collection has {dense_size}, service configured for {DENSE_VECTOR_SIZE}; "
                "reingest with RESET_COLLECTION=1 after changing the dense model",
            },
        )
    return JSONResponse(content={"status": "ok"})


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    try:
        final_message_ids, _ = await run_search_pipeline(
            app.state.qdrant,
            payload,
            time_bounds_cache=app.state.time_bounds_cache,
            skip_rerank=not app.state.reranker_available,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not final_message_ids:
        return SearchAPIResponse(results=[])

    return SearchAPIResponse(
        results=[SearchAPIItem(message_ids=final_message_ids)]
    )


@app.post("/_debug/search")
async def search_debug(
    payload: SearchAPIRequest,
    no_rescore: bool = False,
    no_rerank: bool = False,
    fusion: str | None = None,
    max_dense: int | None = None,
    max_sparse: int | None = None,
) -> dict[str, Any]:
    try:
        final_message_ids, stages = await run_search_pipeline(
            app.state.qdrant,
            payload,
            time_bounds_cache=app.state.time_bounds_cache,
            skip_rescore=no_rescore,
            skip_rerank=no_rerank or not app.state.reranker_available,
            collect_stages=True,
            fusion=fusion,
            max_dense=max_dense,
            max_sparse=max_sparse,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not final_message_ids:
        return {"final": [], "stages": stages}

    return {"final": final_message_ids, "stages": stages}


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
