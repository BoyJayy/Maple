"""Chunking configuration.

Значения подобраны эмпирически sweep'ом на Go Nova.json.
См. docs/ml_description.md § Chunking, секция «Как подбирали параметры».
Менять вместе с пересчётом eval.
"""
from __future__ import annotations

from dataclasses import dataclass

LOWER_CHARS = 400
UPPER_CHARS = 1600
TIME_GAP_SECONDS = 3600
OVERLAP_MESSAGES = 2


@dataclass(frozen=True)
class ChunkingConfig:
    lower_chars: int = LOWER_CHARS
    upper_chars: int = UPPER_CHARS
    time_gap_seconds: int = TIME_GAP_SECONDS
    overlap_messages: int = OVERLAP_MESSAGES


DEFAULT_CONFIG = ChunkingConfig()
