"""Message-boundary chunking.

Alternative модульная реализация, parity с index/main.py::build_chunks.
Пока НЕ используется main.py — он держит свою inline-версию. Этот модуль
разрабатывается параллельно; когда стабилизируется + пройдёт eval — main.py
переключим на импорт отсюда.

Алгоритм:
  1. filter (is_hidden / is_system / пустой текст) + render
  2. sort по (time, id) — стабильно, воспроизводимо
  3. sequential scan:
       hard boundary: thread change  |  gap > TIME_GAP_SECONDS
       soft boundary: len > UPPER_CHARS И len >= LOWER_CHARS
       при граничном переходе: последние OVERLAP_MESSAGES → хвост нового чанка
       (только если boundary мягкая; при hard boundary overlap НЕ переносится)
  4. emit только чанки, в которых есть хотя бы одно new_message
"""
from __future__ import annotations

from config import DEFAULT_CONFIG, ChunkingConfig
from content_builders import (
    build_dense_content,
    build_page_content,
    build_sparse_content,
)
from message_processing import keep_message, render_message
from schemas import ChunkResult, MessageLike


def build_chunks(
    overlap_messages: list[MessageLike],
    new_messages: list[MessageLike],
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> list[ChunkResult]:
    all_messages = overlap_messages + new_messages

    rendered: list[tuple[MessageLike, str]] = []
    for m in all_messages:
        if not keep_message(m):
            continue
        text = render_message(m)
        if not text:
            continue
        rendered.append((m, text))
    rendered.sort(key=lambda pair: (pair[0].time, pair[0].id))

    new_ids = {m.id for m in new_messages}

    groups = _group_messages(rendered, config)

    chunks: list[ChunkResult] = []
    for group in groups:
        if not any(m.id in new_ids for m, _ in group):
            continue
        rendered_texts = [t for _, t in group]
        chunks.append(
            ChunkResult(
                page_content=build_page_content(rendered_texts),
                dense_content=build_dense_content(rendered_texts),
                sparse_content=build_sparse_content(rendered_texts),
                message_ids=[m.id for m, _ in group],
            )
        )
    return chunks


def _group_messages(
    rendered: list[tuple[MessageLike, str]],
    config: ChunkingConfig,
) -> list[list[tuple[MessageLike, str]]]:
    groups: list[list[tuple[MessageLike, str]]] = []
    current: list[tuple[MessageLike, str]] = []
    current_len = 0

    for m, text in rendered:
        if current:
            prev = current[-1][0]
            same_thread = (m.thread_sn or "") == (prev.thread_sn or "")
            gap = (m.time - prev.time) if (m.time and prev.time) else 0
            hard_boundary = (not same_thread) or gap > config.time_gap_seconds
            size_boundary = current_len + len(text) > config.upper_chars

            if hard_boundary or (size_boundary and current_len >= config.lower_chars):
                groups.append(current)
                if hard_boundary or config.overlap_messages <= 0:
                    tail: list[tuple[MessageLike, str]] = []
                else:
                    tail = current[-config.overlap_messages :]
                current = list(tail)
                current_len = sum(len(t) for _, t in current)
        current.append((m, text))
        current_len += len(text)

    if current:
        groups.append(current)

    return groups
