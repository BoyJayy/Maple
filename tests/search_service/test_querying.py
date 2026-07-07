from config import MAX_DENSE_QUERIES, MAX_SPARSE_QUERIES, TIME_FILTER_MARGIN_SECONDS
from querying import (
    build_search_context,
    count_stem_hits,
    dedupe_message_ids,
    extract_exact_terms,
    extract_time_range,
    normalize_text,
    stem_token,
)
from schemas import DateRange, Question


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  раз \n два\tтри ") == "раз два три"


def test_stem_token_joins_russian_word_forms():
    assert stem_token("релизе") == stem_token("релиза") == stem_token("релиз")


def test_stem_token_keeps_identifiers_verbatim():
    assert stem_token("go1.18") == "go1.18"
    assert stem_token("user@corp.com") == "user@corp.com"


def test_count_stem_hits_matches_russian_morphology():
    ctx = build_search_context(Question(text="Что обсуждали про релиз?"))
    assert count_stem_hits("вчера говорили о релизе и деплое", ctx.exact_stems) >= 1


def test_count_stem_hits_respects_word_boundaries():
    stems = (stem_token("код"),)
    assert count_stem_hits("в кодексе написано", stems) == 0
    assert count_stem_hits("посмотри код в репозитории", stems) == 1


def test_extract_exact_terms_filters_short_tokens():
    question = Question(text="Го на обед в 12", keywords=["Go", "1.18"])
    terms = extract_exact_terms(question)
    assert "1.18" in terms
    assert "го" not in terms
    assert "go" not in terms
    assert "12" in terms


def test_extract_exact_terms_is_limited():
    question = Question(text=" ".join(f"термин{index}" for index in range(30)))
    assert len(extract_exact_terms(question)) <= 12


def test_extract_time_range_from_date_mentions():
    question = Question(text="что было", date_mentions=["обсуждали 2023-05"])
    time_range = extract_time_range(question)
    assert time_range is not None
    start, end = time_range
    # 2023-05-01 00:00 UTC and 2023-05-31 23:59 UTC, widened by the margin.
    assert start == 1682899200 - TIME_FILTER_MARGIN_SECONDS
    assert end == 1685577600 - 1 + TIME_FILTER_MARGIN_SECONDS


def test_extract_time_range_from_date_range():
    question = Question(
        text="что было",
        date_range=DateRange.model_validate({"from": "2023-05-01", "to": "2023-05-02"}),
    )
    time_range = extract_time_range(question)
    assert time_range is not None
    start, end = time_range
    assert start < end
    assert end - start >= 86400


def test_extract_time_range_absent_without_dates():
    assert extract_time_range(Question(text="просто вопрос")) is None


def test_extract_time_range_ignores_garbage():
    assert extract_time_range(Question(text="q", date_mentions=["в марте", "скоро"])) is None


def test_build_search_context_limits_and_stems():
    question = Question(
        text="Что обсуждали про Go 1.18?",
        search_text="Go 1.18",
        keywords=["Go", "1.18"],
        variants=["Разговор про Go 1.18", "лишний вариант"],
    )
    ctx = build_search_context(question)
    assert ctx.primary_query == "Go 1.18"
    assert 0 < len(ctx.dense_queries) <= MAX_DENSE_QUERIES
    assert 0 < len(ctx.sparse_queries) <= MAX_SPARSE_QUERIES
    assert len(ctx.exact_stems) == len(set(ctx.exact_stems))
    assert "1.18" in ctx.exact_terms


def test_dedupe_message_ids_keeps_order_and_limit():
    assert dedupe_message_ids(["a", "b", "a", "c", "d"], limit=3) == ["a", "b", "c"]
