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

    def __init__(self, points, *, empty_when_filtered: bool = True, time_bounds=None):
        self._points = points
        self._empty_when_filtered = empty_when_filtered
        self._time_bounds = time_bounds
        self.calls: list[bool] = []
        self.prefetch_filters: list[list[bool]] = []
        self.scroll_calls: list[tuple[str, str]] = []

    async def query_points(self, *, collection_name, prefetch, query, limit, with_payload):
        filters = [item.filter is not None for item in prefetch]
        self.prefetch_filters.append(filters)
        hard_filtered = bool(filters) and all(filters)
        self.calls.append(hard_filtered)
        if hard_filtered and self._empty_when_filtered:
            return FakeResponse([])
        return FakeResponse(self._points)

    async def scroll(
        self,
        *,
        collection_name,
        limit,
        order_by,
        with_payload,
        with_vectors,
    ):
        self.scroll_calls.append((order_by.key, order_by.direction.value))
        if self._time_bounds is None:
            return [], None
        start, end = self._time_bounds
        key = order_by.key.rsplit(".", 1)[-1]
        value = start if key == "start" else end
        return [FakePoint({"metadata": {key: value}})], None


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
    monkeypatch.setattr(pipeline, "TIME_FILTER_MODE", "hard")
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


def test_pipeline_skips_point_rescore_when_disabled(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "POINT_RESCORE_ENABLED", False)
    client = FakeQdrant([make_point("m1", "первое"), make_point("m2", "второе")])
    payload = SearchAPIRequest.model_validate({"question": {"text": "просто вопрос"}})

    final_ids, stages = asyncio.run(
        pipeline.run_search_pipeline(client, payload, collect_stages=True)
    )

    assert "rescored" not in stages
    assert final_ids == stages["retrieval"]


def test_pipeline_can_enable_point_rescore(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "POINT_RESCORE_ENABLED", True)
    client = FakeQdrant([make_point("m1", "первое"), make_point("m2", "второе")])
    payload = SearchAPIRequest.model_validate({"question": {"text": "просто вопрос"}})

    _, stages = asyncio.run(
        pipeline.run_search_pipeline(client, payload, collect_stages=True)
    )

    assert "rescored" in stages


def test_pipeline_collects_rerank_stage(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "RERANK_ENABLED", True)
    monkeypatch.setattr(pipeline, "POINT_RESCORE_ENABLED", True)

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


def test_pipeline_soft_time_boost_keeps_unfiltered_candidates(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "TIME_FILTER_MODE", "soft")
    points = [make_point("m1", "обсуждали релиз")]
    client = FakeQdrant(points)
    payload = SearchAPIRequest.model_validate(
        {"question": {"text": "Что решили про релиз?", "date_mentions": ["2026-04-01"]}}
    )

    final_ids, _ = asyncio.run(pipeline.run_search_pipeline(client, payload))

    assert final_ids == ["m1"]
    assert client.calls == [False]
    assert any(client.prefetch_filters[0])
    assert not all(client.prefetch_filters[0])


def test_pipeline_skips_disjoint_time_filter_using_cached_bounds(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "TIME_FILTER_MODE", "hard")
    points = [make_point("m1", "обсуждали релиз")]
    # Collection is in 2023; the question asks about 2026.
    client = FakeQdrant(points, time_bounds=(1_672_531_200, 1_704_067_199))
    cache = pipeline.CollectionTimeBoundsCache(client)
    payload = SearchAPIRequest.model_validate(
        {"question": {"text": "Что решили про релиз?", "date_mentions": ["2026-04-01"]}}
    )

    final_ids, _ = asyncio.run(
        pipeline.run_search_pipeline(client, payload, time_bounds_cache=cache)
    )

    assert final_ids == ["m1"]
    assert client.calls == [False]
    assert sorted(client.scroll_calls) == [
        ("metadata.end", "desc"),
        ("metadata.start", "asc"),
    ]


def test_time_bounds_cache_is_reused(monkeypatch):
    monkeypatch.setattr(pipeline, "TIME_FILTER_MODE", "hard")
    client = FakeQdrant([], time_bounds=(1_672_531_200, 1_704_067_199))
    cache = pipeline.CollectionTimeBoundsCache(client, success_ttl_seconds=60)

    async def load_twice():
        return await cache.get(), await cache.get()

    first, second = asyncio.run(load_twice())

    assert first == second == (1_672_531_200, 1_704_067_199)
    assert len(client.scroll_calls) == 2


def test_time_bounds_ttl_starts_after_slow_lookup():
    now = [0.0]

    class SlowBoundsQdrant(FakeQdrant):
        async def scroll(self, **kwargs):
            now[0] += 10.0
            return await super().scroll(**kwargs)

    client = SlowBoundsQdrant([], time_bounds=(1_672_531_200, 1_704_067_199))
    cache = pipeline.CollectionTimeBoundsCache(
        client,
        success_ttl_seconds=5,
        clock=lambda: now[0],
    )

    async def load_twice():
        return await cache.get(), await cache.get()

    first, second = asyncio.run(load_twice())

    assert first == second == (1_672_531_200, 1_704_067_199)
    assert len(client.scroll_calls) == 2


def test_pipeline_keeps_hard_filter_when_bounds_overlap(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "TIME_FILTER_MODE", "hard")
    points = [make_point("m1", "обсуждали релиз")]
    client = FakeQdrant(
        points,
        empty_when_filtered=False,
        time_bounds=(0, 4_000_000_000),
    )
    cache = pipeline.CollectionTimeBoundsCache(client)
    payload = SearchAPIRequest.model_validate(
        {"question": {"text": "Что решили про релиз?", "date_mentions": ["2026-04-01"]}}
    )

    final_ids, _ = asyncio.run(
        pipeline.run_search_pipeline(client, payload, time_bounds_cache=cache)
    )

    assert final_ids == ["m1"]
    assert client.calls == [True]


def test_time_bounds_failure_fails_open_to_hard_filter(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "TIME_FILTER_MODE", "hard")

    class BrokenBoundsQdrant(FakeQdrant):
        async def scroll(self, **kwargs):
            raise RuntimeError("bounds unavailable")

    points = [make_point("m1", "обсуждали релиз")]
    client = BrokenBoundsQdrant(points, empty_when_filtered=False)
    cache = pipeline.CollectionTimeBoundsCache(client)
    payload = SearchAPIRequest.model_validate(
        {"question": {"text": "Что решили про релиз?", "date_mentions": ["2026-04-01"]}}
    )

    final_ids, _ = asyncio.run(
        pipeline.run_search_pipeline(client, payload, time_bounds_cache=cache)
    )

    assert final_ids == ["m1"]
    assert client.calls == [True]


def test_reranker_failure_falls_back_to_fused_results(monkeypatch):
    stub_embeddings(monkeypatch)
    monkeypatch.setattr(pipeline, "RERANK_ENABLED", True)
    monkeypatch.setattr(pipeline, "POINT_RESCORE_ENABLED", False)

    async def broken_reranker(ctx, points):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(pipeline, "rerank_points", broken_reranker)
    client = FakeQdrant([make_point("m1", "релиз")], empty_when_filtered=False)
    payload = SearchAPIRequest.model_validate({"question": {"text": "релиз"}})

    final_ids, stages = asyncio.run(
        pipeline.run_search_pipeline(client, payload, collect_stages=True)
    )

    assert final_ids == ["m1"]
    assert set(stages) == {"retrieval"}
