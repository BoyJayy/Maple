"""Три версии контента чанка под три модели.

Сейчас все три идентичны — baseline parity с main.py.
Разделение (dense=семантика, sparse=keywords, page=structured) —
Этап 2 роадмапа. Тогда сюда добавятся реальные различия.
"""
from __future__ import annotations


def build_page_content(rendered_messages: list[str]) -> str:
    return "\n".join(rendered_messages)


def build_dense_content(rendered_messages: list[str]) -> str:
    return "\n".join(rendered_messages)


def build_sparse_content(rendered_messages: list[str]) -> str:
    return "\n".join(rendered_messages)
