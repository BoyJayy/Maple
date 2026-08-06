import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import snowballstemmer

from config import MAX_DENSE_QUERIES, MAX_SPARSE_QUERIES, TIME_FILTER_MARGIN_SECONDS
from schemas import Entities, Question


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[\w@./:+-]+", re.UNICODE)
CYRILLIC_RE = re.compile(r"[а-яё]")
# Tokens with digits or identifier punctuation (emails, links, versions) are matched verbatim.
VERBATIM_TOKEN_RE = re.compile(r"[\d@./:+-]")
TOKEN_EDGE_PUNCT = ".:/+-@"
TOKEN_INNER_SPLIT_RE = re.compile(r"[@./:+_-]+")
DATE_MENTION_RE = re.compile(r"\b(19\d{2}|20\d{2})(?:-(0[1-9]|1[0-2])(?:-([0-3]\d))?)?\b")

_RUSSIAN_STEMMER = snowballstemmer.stemmer("russian")
_ENGLISH_STEMMER = snowballstemmer.stemmer("english")

# Filler words that otherwise eat the exact-term budget without carrying
# retrieval signal. Applied only to plain word tokens, never to identifiers.
EXACT_TERM_STOPWORDS = frozenset(
    {
        # ru: interrogatives, pronouns, prepositions
        "что", "чего", "чем", "как", "какой", "какая", "какое", "какие", "каких",
        "кто", "кого", "кому", "где", "куда", "когда", "почему", "зачем",
        "это", "эта", "этот", "эти", "этом", "того", "тот", "том", "так",
        "все", "всё", "всех", "они", "оно", "она", "его", "еще", "ещё",
        "мне", "нам", "вам", "нас", "вас", "наш", "ваш", "мы", "вы",
        "про", "для", "при", "под", "над", "без", "или", "если", "чтобы",
        "был", "была", "было", "были", "есть", "нет", "уже", "только",
        "который", "которая", "которые", "которых",
        # ru: politeness / meta-verbs typical for questions over chat history
        "подскажи", "подскажите", "пожалуйста", "скажи", "скажите",
        "напомни", "напомните", "расскажи", "расскажите",
        "обсуждали", "обсудили", "говорили", "писали", "решили", "итоге",
        # en
        "the", "and", "for", "with", "from", "that", "this", "these", "those",
        "what", "when", "where", "which", "who", "how", "why",
        "did", "does", "was", "were", "are", "you", "our", "their",
        "please", "tell", "about", "discuss", "discussed",
    }
)


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


def iter_tokens(text: str) -> list[str]:
    # Strip sentence punctuation glued to token edges ("релиза." -> "релиза"),
    # keep identifier punctuation inside ("release-plan.docx", "1.18").
    tokens = []
    for raw in TOKEN_RE.findall(text.lower()):
        token = raw.strip(TOKEN_EDGE_PUNCT)
        if token:
            tokens.append(token)
    return tokens


def stem_token(token: str) -> str:
    if VERBATIM_TOKEN_RE.search(token):
        return token
    if CYRILLIC_RE.search(token):
        return _RUSSIAN_STEMMER.stemWord(token)
    return _ENGLISH_STEMMER.stemWord(token)


@lru_cache(maxsize=4096)
def text_stems(text: str) -> frozenset[str]:
    stems: set[str] = set()
    for token in iter_tokens(text):
        stems.add(stem_token(token))
        if VERBATIM_TOKEN_RE.search(token):
            # Compound identifiers also expose their word parts, so the term
            # "plan" still hits "release-plan.docx" and "релиз" hits "пост-релиз".
            for part in TOKEN_INNER_SPLIT_RE.split(token):
                if len(part) >= 3 and not part.isdigit():
                    stems.add(stem_token(part))
    return frozenset(stems)


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
    # A whitespace-only search_text is not a usable override.
    return normalize_text(question.search_text) or normalize_text(question.text)


def extract_exact_terms(question: Question) -> list[str]:
    # Highest-signal sources first: the 12-term budget must not be exhausted
    # by question-text filler before entities and keywords are reached.
    text_candidates = [
        *collect_entity_terms(question.entities),
        *(question.keywords or []),
        *(question.date_mentions or []),
        build_primary_query(question),
        question.text,
        question.asker,
    ]
    terms: list[str] = []
    for text in text_candidates:
        for token in iter_tokens(normalize_text(text)):
            if VERBATIM_TOKEN_RE.search(token) or "_" in token:
                terms.append(token)
            elif len(token) >= 3 and token not in EXACT_TERM_STOPWORDS:
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


def _parse_datetime(value: str, *, end_of_day: bool = False) -> int | None:
    normalized = normalize_text(value)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    timestamp = int(parsed.timestamp())
    # A date-only upper bound means "up to the end of that day", not midnight.
    if end_of_day and len(normalized) <= 10:
        timestamp += 86400 - 1
    return timestamp


def extract_time_range(question: Question) -> tuple[int, int] | None:
    bounds: list[tuple[int, int]] = []

    if question.date_range is not None:
        start = _parse_datetime(question.date_range.from_)
        end = _parse_datetime(question.date_range.to, end_of_day=True)
        if start is not None and end is not None and start <= end:
            # An explicit structured range is authoritative. Mixing it with
            # looser date_mentions can accidentally widen a precise day into
            # a month or year and make the filter ineffective.
            bounds.append((start, end))

    if not bounds:
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
