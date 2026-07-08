"""Retrieval metrics: Recall@K, nDCG@K, MRR@K and bootstrap CIs."""
from __future__ import annotations

import math
import random


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for msg_id in top_k if msg_id in relevant)
    return hits / len(relevant)


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, msg_id in enumerate(predicted[:k]):
        if msg_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    for i, msg_id in enumerate(predicted[:k]):
        if msg_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def score(recall_avg: float, ndcg_avg: float) -> float:
    # Recall dominates: the consumer feeds the returned messages to an LLM,
    # so missing a relevant message hurts more than imperfect ordering.
    return recall_avg * 0.8 + ndcg_avg * 0.2


def bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-question metric values."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(size)) / size
        for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    low_index = int(alpha * (n_resamples - 1))
    high_index = int((1.0 - alpha) * (n_resamples - 1))
    return means[low_index], means[high_index]
