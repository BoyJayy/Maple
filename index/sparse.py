from functools import lru_cache

from config import FASTEMBED_CACHE_PATH, SPARSE_MODEL_LANGUAGE, SPARSE_MODEL_NAME, logger
from schemas import SparseVector


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding

    logger.info(
        "Loading sparse model %s (language=%s) from cache %s",
        SPARSE_MODEL_NAME,
        SPARSE_MODEL_LANGUAGE,
        FASTEMBED_CACHE_PATH,
    )
    if SPARSE_MODEL_NAME == "Qdrant/bm25":
        return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME, language=SPARSE_MODEL_LANGUAGE)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    vectors: list[dict[str, list[int] | list[float]]] = []

    for item in model.embed(texts):
        vectors.append(
            {
                "indices": item.indices.tolist(),
                "values": item.values.tolist(),
            }
        )

    return vectors
