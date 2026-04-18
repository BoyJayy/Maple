"""Protocols для модульного chunking.

Не Pydantic — чтобы chunking.py не зависел от FastAPI и умел принимать
и Message-инстансы из main.py, и dict'ы, и mock-объекты в тестах.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class MessageLike(Protocol):
    id: str
    thread_sn: str | None
    time: int
    text: str
    file_snippets: str
    parts: list[dict[str, Any]] | None
    is_system: bool
    is_hidden: bool


@dataclass(frozen=True)
class ChunkResult:
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]
