"""Три версии контента чанка под три модели (Этап 2 роадмапа).

dense_content  — чистая семантика: только тексты, без служебных префиксов.
Совпадает с baseline main.py::build_chunks. Шумовые заголовки
засорили бы эмбеддинг → специально не добавляем.

sparse_content — keyword dump: тексты + sender_id + mentions. BM25 ловит
запросы вида "кто alice@corp" или "@bob говорил про X"
даже если в исходном тексте сендер не упомянут словесно.

page_content   — structured dialog: `[YYYY-MM-DD HH:MM | sender_id]` + текст
построчно. Даёт реранкеру timestamp + автора для каждого
сообщения — помогает judging'у "кто кого спрашивает", "когда".
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas import MessageLike


def build_dense_content(messages: list[tuple[MessageLike, str]]) -> str:
    return "\n".join(text for _, text in messages)


def build_sparse_content(messages: list[tuple[MessageLike, str]]) -> str:
    lines: list[str] = []
    for m, text in messages:
        lines.append(text)
        if m.sender_id:
            lines.append(m.sender_id)
        if m.mentions:
            lines.extend(m.mentions)
    return "\n".join(lines)


def build_page_content(messages: list[tuple[MessageLike, str]]) -> str:
    lines: list[str] = []
    for m, text in messages:
        header = _format_header(m)
        lines.append(header)
        lines.append(text)
    return "\n".join(lines)


def _format_header(m: MessageLike) -> str:
    ts = datetime.fromtimestamp(m.time, tz=UTC).strftime("%Y-%m-%d %H:%M") if m.time else "?"
    if m.sender_id:
        return f"[{ts} | {m.sender_id}]"
    return f"[{ts}]"
