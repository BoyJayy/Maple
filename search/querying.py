import re
from dataclasses import dataclass

from config import MAX_DENSE_QUERIES, MAX_SPARSE_QUERIES
from schemas import Entities, Question


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[\w@./:+-]+", re.UNICODE)


@dataclass(frozen=True)
class SearchContext:
    primary_query: str
    dense_queries: tuple[str, ...]
    sparse_queries: tuple[str, ...]
    exact_terms: tuple[str, ...]


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


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
