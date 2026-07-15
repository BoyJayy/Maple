import asyncio

import main


def test_reranker_warmup_failure_keeps_search_models_available(monkeypatch):
    monkeypatch.setattr(main, "RERANK_ENABLED", True)
    monkeypatch.setattr(main, "get_dense_model", lambda: object())
    monkeypatch.setattr(main, "get_sparse_model", lambda: object())

    def broken_reranker():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(main, "get_reranker", broken_reranker)

    assert asyncio.run(main.warmup_models()) is False
