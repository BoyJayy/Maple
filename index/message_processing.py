"""Фильтрация и рендер одного сообщения в plain text.

Parity с index/main.py::render_message / keep_message.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas import MessageLike


def keep_message(m: MessageLike) -> bool:
    if m.is_hidden:
        return False
    return not m.is_system


def render_message(message: MessageLike) -> str:
    parts: list[str] = []
    if message.text:
        parts.append(message.text.strip())
    if message.parts:
        for part in message.parts:
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                parts.append(part_text.strip())
    if message.file_snippets:
        parts.append(message.file_snippets.strip())
    return "\n".join(p for p in parts if p)
