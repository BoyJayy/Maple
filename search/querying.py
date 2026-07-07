import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import snowballstemmer

from config import MAX_DENSE_QUERIES, MAX_SPARSE_QUERIES, TIME_FILTER_MARGIN_SECONDS
from schemas import Entities, Question


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[\w@./:+-]+", re.UNICODE)
CYRILLIC_RE = re.compile(r"[а-яё]")
# Tokens with digits or identifier punctuation (emails, links, versions) are matched verbatim.
VERBATIM_TOKEN_RE = re.compile(r"[\d@./:+-]")
DATE_MENTION_RE = re.compile(r"\b(19\d{2}|20\d{2})(?:-(0[1-9]|1[0-2])(?:-([0-3]\d))?)?\b")

_RUSSIAN_STEMMER = snowballstemmer.stemmer("russian")
_ENGLISH_STEMMER = snowballstemmer.stemmer("english")


@dataclass(frozen=True)
class SearchContext:
    primary_query: str
    dense_queries: tuple[str, ...]
    sparse_queries: tuple[str, ...]
    exact_terms: tuple[str, ...]
    exact_stems: tuple[str, ...]
    time_range: tuple[int, int] | None


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def stem_token(token: str) -> str:
    if VERBATIM_TOKEN_RE.search(token):
        return token
    if CYRILLIC_RE.search(token):
        return _RUSSIAN_STEMMER.stemWord(token)
    return _ENGLISH_STEMMER.stemWord(token)


def text_stems(text: str) -> set[str]:
    return {stem_token(token) for token in TOKEN_RE.findall(text.lower())}


def count_stem_hits(text: str, stems: tuple[str, ...]) -> int:
    if not stems or not text:
        return 0
    found = text_stems(text)
    return sum(1 for stem in stems if stem in found)


def unique_texts(items: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if limit is not None and len(result) >= limit:
            break
    return result


def collect_entity_terms(entities: Entities | None) -> list[str]:
    if entities is None:
        return []
    return unique_texts(
        [
            *(entities.people or []),
            *(entities.emails or []),
            *(entities.documents or []),
            *(entities.names or []),
            *(entities.links or []),
        ]
    )


def build_primary_query(question: Question) -> str:
    return normalize_text(question.search_text or question.text)


def extract_exact_terms(question: Question) -> list[str]:
    text_candidates = [
        build_primary_query(question),
        question.text,
        *(question.keywords or []),
        *collect_entity_terms(question.entities),
        *(question.date_mentions or []),
        question.asker,
    ]
    terms: list[str] = []
    for text in text_candidates:
        for token in TOKEN_RE.findall(normalize_text(text).lower()):
            if len(token) >= 3 or any(ch.isdigit() for ch in token) or any(ch in token for ch in "@./:+-_"):
                terms.append(token)
    return unique_texts(terms, limit=12)


def _date_bounds(year: int, month: int | None, day: int | None) -> tuple[int, int]:
    if month is None:
        start = datetime(year, 1, 1, tzinfo=UTC)
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    elif day is None:
        start = datetime(year, month, 1, tzinfo=UTC)
        end = datetime(year + (1 if month == 12 else 0), month % 12 + 1, 1, tzinfo=UTC)
    else:
        start = datetime(year, month, day, tzinfo=UTC)
        end = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp()) - 1


def _parse_datetime(value: str) -> int | None:
    try:
        parsed = datetime.fromisoformat(normalize_text(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def extract_time_range(question: Question) -> tuple[int, int] | None:
    bounds: list[tuple[int, int]] = []

    if question.date_range is not None:
        start = _parse_datetime(question.date_range.from_)
        end = _parse_datetime(question.date_range.to)
        if start is not None and end is not None and start <= end:
            bounds.append((start, end))

    for mention in question.date_mentions or []:
        for match in DATE_MENTION_RE.finditer(str(mention)):
            year = int(match.group(1))
            month = int(match.group(2)) if match.group(2) else None
            day = int(match.group(3)) if match.group(3) else None
            try:
                bounds.append(_date_bounds(year, month, day))
            except ValueError:
                continue

    if not bounds:
        return None
    start = min(item[0] for item in bounds) - TIME_FILTER_MARGIN_SECONDS
    end = max(item[1] for item in bounds) + TIME_FILTER_MARGIN_SECONDS
    return start, end


def build_search_context(question: Question) -> SearchContext:
    primary_query = build_primary_query(question)
    exact_terms = extract_exact_terms(question)
    exact_query = " ".join(exact_terms)

    dense_queries = unique_texts(
        [
            primary_query,
            question.text,
            *((question.variants or [])[:1]),
            exact_query,
        ],
        limit=MAX_DENSE_QUERIES,
    )
    sparse_queries = unique_texts(
        [
            exact_query,
            primary_query,
            question.text,
            *((question.variants or [])[:1]),
        ],
        limit=MAX_SPARSE_QUERIES,
    )

    return SearchContext(
        primary_query=primary_query,
        dense_queries=tuple(dense_queries),
        sparse_queries=tuple(sparse_queries),
        exact_terms=tuple(exact_terms),
        exact_stems=tuple(dict.fromkeys(stem_token(term) for term in exact_terms)),
        time_range=extract_time_range(question),
    )


def dedupe_message_ids(message_ids: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for message_id in message_ids:
        if message_id in seen:
            continue
        seen.add(message_id)
        result.append(message_id)
        if len(result) >= limit:
            break
    return result
