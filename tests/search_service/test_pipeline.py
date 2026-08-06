from pipeline import (
    assemble_message_ids,
    build_time_filter,
    extract_message_blocks,
    extract_stored_blocks,
    rescore_points,
    score_point,
    split_sections,
)
from querying import SearchContext, build_search_context
from schemas import Question


class FakePoint:
    def __init__(self, payload: dict, score: float = 0.0):
        self.payload = payload
        self.score = score


def make_context(text: str = "Что обсуждали про релиз?") -> SearchContext:
    return build_search_context(Question(text=text))


def page_content(blocks: list[str], *, context: str = "") -> str:
    parts = ["CHAT: Test Chat", "CHAT_TYPE: group", "CHAT_ID: chat-1"]
    if context:
        parts.extend(["CONTEXT:", context])
    parts.append("MESSAGES:")
    parts.extend(blocks)
    return "\n\n".join(parts)


def make_point(
    blocks: list[tuple[str, str]],
    *,
    score: float = 0.0,
    stored_blocks: bool = True,
    context: str = "",
) -> FakePoint:
    rendered = [f"[2023-11-14 22:13:00 UTC | user@example.com]\n{text}" for _, text in blocks]
    metadata = {
        "message_ids": list(dict.fromkeys(message_id for message_id, _ in blocks)),
        "participants": ["user@example.com"],
        "mentions": [],
    }
    if stored_blocks:
        metadata["message_blocks"] = [
            {"message_id": message_id, "text": rendered_text}
            for (message_id, _), rendered_text in zip(blocks, rendered, strict=True)
        ]
    return FakePoint({"page_content": page_content(rendered, context=context), "metadata": metadata}, score=score)


def test_split_sections_separates_context_and_messages():
    content = page_content(["[2023-11-14 22:13:00 UTC | u]\nтело"], context="контекстное сообщение")
    context_text, message_text = split_sections(content)
    assert "контекстное" in context_text
    assert "тело" in message_text
    assert "контекстное" not in message_text


def test_extract_message_blocks_from_page_content():
    content = page_content(
        [
            "[2023-11-14 22:13:00 UTC | u]\nпервое",
            "[2023-11-14 22:14:00 UTC | u]\nвторое",
        ]
    )
    blocks = extract_message_blocks(content)
    assert len(blocks) == 2
    assert "первое" in blocks[0]
    assert "второе" in blocks[1]


def test_extract_stored_blocks_rejects_malformed_metadata():
    point = make_point([("m1", "текст")])
    point.payload["metadata"]["message_blocks"] = [{"text": "нет id"}]
    assert extract_stored_blocks(point) == []
    point.payload["metadata"]["message_blocks"] = "not-a-list"
    assert extract_stored_blocks(point) == []


def test_score_point_is_independent_of_fusion_score_scale():
    ctx = make_context()
    low = make_point([("m1", "обсуждали релиз")], score=0.01)
    high = make_point([("m1", "обсуждали релиз")], score=42.0)
    assert score_point(ctx, low, rank=0) == score_point(ctx, high, rank=0)


def test_score_point_rewards_term_hits_with_morphology():
    ctx = make_context("Что обсуждали про релиз?")
    with_hit = make_point([("m1", "вчера говорили о релизе")])
    without_hit = make_point([("m1", "беседа о погоде")])
    assert score_point(ctx, with_hit, rank=0) > score_point(ctx, without_hit, rank=0)


def test_rescore_points_promotes_matching_point():
    ctx = make_context("Что обсуждали про релиз?")
    points = [make_point([(f"m{index}", "нерелевантный текст")]) for index in range(5)]
    matching = make_point([("m-hit", "детали релиза и планы")])
    reordered = rescore_points(ctx, [*points, matching])
    assert reordered[0] is matching


def test_assemble_prefers_blocks_with_term_hits():
    ctx = make_context("Что обсуждали про релиз?")
    point = make_point([("m1", "про обед"), ("m2", "про релиз и деплой"), ("m3", "про погоду")])
    message_ids = assemble_message_ids(ctx, [point], limit=10)
    assert message_ids[0] == "m2"
    assert set(message_ids) == {"m1", "m2", "m3"}


def test_assemble_works_for_fragmented_messages():
    ctx = make_context("Что обсуждали про релиз?")
    # Two fragments share one message id: the legacy len(blocks) == len(ids)
    # invariant breaks, but stored blocks keep the mapping exact.
    point = make_point([("m1", "часть один"), ("m1", "часть два про релиз"), ("m2", "другое")])
    message_ids = assemble_message_ids(ctx, [point], limit=10)
    assert message_ids[0] == "m1"
    assert set(message_ids) == {"m1", "m2"}


def test_assemble_falls_back_to_page_content_parsing():
    ctx = make_context("Что обсуждали про релиз?")
    point = make_point([("m1", "про обед"), ("m2", "про релиз")], stored_blocks=False)
    message_ids = assemble_message_ids(ctx, [point], limit=10)
    assert message_ids[0] == "m2"


def test_assemble_dedupes_across_points():
    ctx = make_context()
    first = make_point([("m1", "текст"), ("m2", "текст")])
    second = make_point([("m2", "текст"), ("m3", "текст")])
    message_ids = assemble_message_ids(ctx, [first, second], limit=10)
    assert sorted(message_ids) == ["m1", "m2", "m3"]


def test_build_time_filter_absent_without_range():
    assert build_time_filter(make_context()) is None


def test_build_time_filter_uses_overlap_condition():
    ctx = build_search_context(Question(text="что было", date_mentions=["2023-05-12"]))
    time_filter = build_time_filter(ctx)
    assert time_filter is not None
    keys = [condition.key for condition in time_filter.must]
    assert keys == ["metadata.start", "metadata.end"]
    start_condition, end_condition = time_filter.must
    assert start_condition.range.lte is not None
    assert end_condition.range.gte is not None
    assert end_condition.range.gte < start_condition.range.lte
