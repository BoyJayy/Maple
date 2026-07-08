import asyncio

import pipeline
from qdrant_client import models
from schemas import SearchAPIRequest


class FakeResponse:
    def __init__(self, points):
        self.points = points


class FakeQdrant:
    """Returns points only for unfiltered prefetches, emulating a collection
    where the time filter matches nothing (old string payloads, wrong dates)."""

    def __init__(self, points, *, empty_when_filtered: bool = True):
        self._points = points
        self._empty_when_filtered = empty_when_filtered
        self.calls: list[bool] = []

    async def query_points(self, *, collection_name, prefetch, query, limit, with_payload):
        filtered = any(item.filter is not None for item in prefetch)
        self.calls.append(filtered)
        if filtered and self._empty_when_filtered:
            return FakeResponse([])
        return FakeResponse(self._points)


class FakePoint:
    def __init__(self, payload):
        self.payload = payload
        self.score = 0.0


def make_point(message_id: str, text: str) -> FakePoint:
    block = f"[2023-11-14 22:13:00 UTC | user@example.com]\n{text}"
    return FakePoint(
        {
            "page_content": f"CHAT: c\n\nCHAT_TYPE: group\n\nCHAT_ID: chat-1\n\nMESSAGES:\n\n{block}",
            "metadata": {
                "message_ids": [message_id],
                "message_blocks": [{"message_id": message_id, "text": block}],
                "participants": [],
                "mentions": [],
            },
        }
    )


def stub_embeddings(monkeypatch):
    async def fake_dense(texts):
        return [[0.1, 0.2] for _ in texts]

    async def fake_sparse(texts):
        return []

    monkeypatch.setattr(pipeline, "embed_dense", fake_dense)
    monkeypatch.setattr(pipeline, "embed_sparse", fake_sparse)


def test_pipeline_falls_back_to_unfiltered_search_on_zero_hits(monkeypatch):
    stub_embeddings(monkeypatch)
    points = [make_point("m1", "обсуждали релиз"), make_point("m2", "другое")]
    client = FakeQdrant(points)
    payload = SearchAPIRequest.model_validate(
        {"question": {"text": "Что решили про релиз?", "date_mentions": ["2026-04-01"]}}
    )

    final_ids, _ = asyncio.run(pipeline.run_search_pipeline(client, payload))

    # First call carried the time filter and found nothing; retry went unfiltered.
    assert client.calls == [True, False]
    assert "m1" in final_ids and "m2" in final_ids


def test_pipeline_without_dates_queries_unfiltered_once(monkeypatch):
    stub_embeddings(monkeypatch)
    client = FakeQdrant([make_point("m1", "текст")])
    payload = SearchAPIRequest.model_validate({"question": {"text": "просто вопрос"}})

    final_ids, _ = asyncio.run(pipeline.run_search_pipeline(client, payload))

    assert client.calls == [False]
    assert final_ids == ["m1"]


def test_pipeline_collects_rerank_stage(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "RERANK_ENABLED", True)

    class FakeReranker:
        def rerank(self, query, documents):
            # Reverse order: last document is most relevant.
            return list(range(len(documents)))

    monkeypatch.setattr(pipeline, "get_reranker", lambda: FakeReranker())
    points = [make_point(f"m{index}", "одинаковый текст") for index in range(3)]
    client = FakeQdrant(points, empty_when_filtered=False)
    payload = SearchAPIRequest.model_validate({"question": {"text": "одинаковый текст"}})

    final_ids, stages = asyncio.run(
        pipeline.run_search_pipeline(client, payload, collect_stages=True)
    )

    assert set(stages) == {"retrieval", "rescored", "reranked"}
    assert final_ids[0] == "m2"


def test_rerank_points_reorders_top_and_keeps_tail(monkeypatch):
    class FakeReranker:
        def rerank(self, query, documents):
            return [float(index) for index in range(len(documents))]

    monkeypatch.setattr(pipeline, "get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(pipeline, "RERANK_TOP_K", 2)

    ctx = pipeline.build_search_context(
        __import__("schemas").Question(text="вопрос про что-нибудь")
    )
    points = [make_point(f"m{index}", "текст") for index in range(4)]

    reranked = asyncio.run(pipeline.rerank_points(ctx, points))

    # Head [m0, m1] reversed by scores, tail [m2, m3] untouched.
    ids = [point.payload["metadata"]["message_ids"][0] for point in reranked]
    assert ids == ["m1", "m0", "m2", "m3"]


def test_rerank_points_short_circuits_single_point():
    ctx = pipeline.build_search_context(__import__("schemas").Question(text="вопрос"))
    points = [make_point("m1", "текст")]
    assert asyncio.run(pipeline.rerank_points(ctx, points)) is points


def test_build_time_filter_range_values():
    ctx = pipeline.build_search_context(
        __import__("schemas").Question(text="что было", date_mentions=["2023-05-12"])
    )
    time_filter = pipeline.build_time_filter(ctx)
    assert isinstance(time_filter, models.Filter)
