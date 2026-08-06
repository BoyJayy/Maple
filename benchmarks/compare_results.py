#!/usr/bin/env python3
"""Create a paired comparison from two benchmark_search.py result files."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable


INTENT_SUFFIXES = ("_fix", "_owner", "_doc", "_signal", "_checkpoint")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_dataset(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text()
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def select_samples(result: dict[str, Any], concurrency: int, round_number: int) -> dict[str, dict[str, Any]]:
    mode = next((item for item in result["modes"] if item["concurrency"] == concurrency), None)
    if mode is None:
        raise ValueError(f"{result['label']} has no concurrency={concurrency} mode")
    selected = [sample for sample in mode["samples"] if sample["round"] == round_number]
    samples = {sample["id"]: sample for sample in selected}
    if len(samples) != len(selected):
        raise ValueError(f"duplicate query ids in {result['label']} round {round_number}")
    return samples


def recall(predicted: list[str], relevant: set[str], k: int) -> float:
    return len(set(predicted[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, message_id in enumerate(predicted[:k])
        if message_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal


def mrr(predicted: list[str], relevant: set[str], k: int) -> float:
    for rank, message_id in enumerate(predicted[:k], start=1):
        if message_id in relevant:
            return 1.0 / rank
    return 0.0


def best_rank(predicted: list[str], relevant: set[str], missing_rank: int) -> int:
    ranks = [rank for rank, message_id in enumerate(predicted, start=1) if message_id in relevant]
    return min(ranks, default=missing_rank)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def topic_cluster(query_id: str) -> str:
    for suffix in INTENT_SUFFIXES:
        if query_id.endswith(suffix):
            return query_id[: -len(suffix)]
    return query_id


def cluster_bootstrap_ci(
    deltas: dict[str, float],
    clusters: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = {}
    for query_id, delta in deltas.items():
        grouped.setdefault(clusters[query_id], []).append(delta)
    cluster_ids = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sampled_values = [
            value
            for _cluster_index in range(len(cluster_ids))
            for value in grouped[rng.choice(cluster_ids)]
        ]
        estimates.append(statistics.mean(sampled_values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def metric_rows(
    left_samples: dict[str, dict[str, Any]],
    right_samples: dict[str, dict[str, Any]],
    query_ids: list[str],
    ks: list[int],
    clusters: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    functions: dict[str, Callable[[list[str], set[str], int], float]] = {
        "recall": recall,
        "ndcg": ndcg,
        "mrr": mrr,
    }
    output: dict[str, dict[str, dict[str, float]]] = {}
    for k in ks:
        output[str(k)] = {}
        for metric_name, metric_function in functions.items():
            left_values: dict[str, float] = {}
            right_values: dict[str, float] = {}
            for query_id in query_ids:
                relevant = set(left_samples[query_id]["relevant"])
                left_values[query_id] = metric_function(left_samples[query_id]["predicted"], relevant, k)
                right_values[query_id] = metric_function(right_samples[query_id]["predicted"], relevant, k)
            deltas = {query_id: right_values[query_id] - left_values[query_id] for query_id in query_ids}
            ci_low, ci_high = cluster_bootstrap_ci(
                deltas,
                clusters,
                resamples=resamples,
                seed=seed + k + sum(ord(character) for character in metric_name),
            )
            output[str(k)][metric_name] = {
                "left": statistics.mean(left_values.values()),
                "right": statistics.mean(right_values.values()),
                "delta": statistics.mean(deltas.values()),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--ks", default="1,3,5,10,20,50")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    left = load_json(args.left)
    right = load_json(args.right)
    if left["dataset"]["sha256"] != right["dataset"]["sha256"]:
        parser.error("dataset hashes differ")
    if left["corpus"]["sha256"] != right["corpus"]["sha256"]:
        parser.error("corpus hashes differ")
    if left["config"]["question_mode"] != right["config"]["question_mode"]:
        parser.error("question modes differ")

    left_samples = select_samples(left, args.concurrency, args.round)
    right_samples = select_samples(right, args.concurrency, args.round)
    if set(left_samples) != set(right_samples):
        parser.error("query id sets differ")
    query_ids = sorted(left_samples)
    if any(left_samples[query_id]["relevant"] != right_samples[query_id]["relevant"] for query_id in query_ids):
        parser.error("ground-truth ids differ")

    dataset_path = args.dataset or Path(left["dataset"]["path"])
    entries = load_dataset(dataset_path)
    entry_by_id = {str(entry["id"]): entry for entry in entries}
    if set(entry_by_id) != set(query_ids):
        parser.error("dataset entries do not match result samples")

    positives = [query_id for query_id in query_ids if left_samples[query_id]["relevant"]]
    negatives = [query_id for query_id in query_ids if not left_samples[query_id]["relevant"]]
    clusters = {query_id: topic_cluster(query_id) for query_id in positives}
    ks = [int(value) for value in args.ks.split(",")]
    metrics = metric_rows(
        left_samples,
        right_samples,
        positives,
        ks,
        clusters,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    missing_rank = max(ks) + 1
    left_ranks = {
        query_id: best_rank(
            left_samples[query_id]["predicted"],
            set(left_samples[query_id]["relevant"]),
            missing_rank,
        )
        for query_id in positives
    }
    right_ranks = {
        query_id: best_rank(
            right_samples[query_id]["predicted"],
            set(right_samples[query_id]["relevant"]),
            missing_rank,
        )
        for query_id in positives
    }
    rank_comparison = {
        "right_wins": sum(right_ranks[query_id] < left_ranks[query_id] for query_id in positives),
        "ties": sum(right_ranks[query_id] == left_ranks[query_id] for query_id in positives),
        "left_wins": sum(right_ranks[query_id] > left_ranks[query_id] for query_id in positives),
        "left_median_rank": statistics.median(left_ranks.values()),
        "right_median_rank": statistics.median(right_ranks.values()),
        "missing_rank": missing_rank,
    }

    categories: dict[str, list[str]] = {}
    for query_id in positives:
        categories.setdefault(str(entry_by_id[query_id].get("category") or "uncategorized"), []).append(query_id)
    category_metrics = {
        category: {
            "n": len(category_query_ids),
            "metrics": metric_rows(
                left_samples,
                right_samples,
                category_query_ids,
                ks,
                {query_id: query_id for query_id in category_query_ids},
                resamples=args.bootstrap_resamples,
                seed=args.seed,
            ),
        }
        for category, category_query_ids in sorted(categories.items())
    }

    negative_results = {
        "n": len(negatives),
        "left_no_result_rate": (
            statistics.mean(not left_samples[query_id]["predicted"] for query_id in negatives)
            if negatives
            else None
        ),
        "right_no_result_rate": (
            statistics.mean(not right_samples[query_id]["predicted"] for query_id in negatives)
            if negatives
            else None
        ),
    }

    output = {
        "left": {"label": left["label"], "commit": left["commit"], "path": str(args.left)},
        "right": {"label": right["label"], "commit": right["commit"], "path": str(args.right)},
        "dataset_sha256": left["dataset"]["sha256"],
        "corpus_sha256": left["corpus"]["sha256"],
        "question_mode": left["config"]["question_mode"],
        "positive_queries": len(positives),
        "negative_queries": negative_results,
        "cluster_count": len(set(clusters.values())),
        "bootstrap_resamples": args.bootstrap_resamples,
        "metrics": metrics,
        "rank_comparison": rank_comparison,
        "categories": category_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")

    print(f"{left['label']} -> {right['label']}: {len(positives)} positive queries")
    print(
        f"rank wins/ties/losses = {rank_comparison['right_wins']}/"
        f"{rank_comparison['ties']}/{rank_comparison['left_wins']}"
    )
    for k in ks:
        recall_row = metrics[str(k)]["recall"]
        mrr_row = metrics[str(k)]["mrr"]
        print(
            f"K={k:<2} recall {recall_row['left']:.4f}->{recall_row['right']:.4f} "
            f"(d={recall_row['delta']:+.4f}, CI [{recall_row['delta_ci95_low']:+.4f}, "
            f"{recall_row['delta_ci95_high']:+.4f}]); MRR d={mrr_row['delta']:+.4f}"
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
