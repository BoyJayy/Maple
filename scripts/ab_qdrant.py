"""A/B test Qdrant fusion and query-count knobs on a small eval subset.

Usage:
    python3 scripts/ab_qdrant.py --dataset data/Dataset_sweep_questions.jsonl --limit 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from metrics import ndcg_at_k, recall_at_k, score

SEARCH_URL = "http://localhost:8002"


def load_dataset(path: Path) -> list[dict]:
    raw = path.read_text()
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def run_config(entries: list[dict], params: dict, k: int) -> tuple[float, float, float, float]:
    t0 = time.time()
    http = httpx.Client(timeout=120.0)
    qs = "&".join(f"{key}={val}" for key, val in params.items())
    url = f"{SEARCH_URL}/_debug/search?{qs}"
    recalls: list[float] = []
    ndcgs: list[float] = []
    for entry in entries:
        gt = set(entry["answer"]["message_ids"])
        response = http.post(url, json={"question": entry["question"]})
        response.raise_for_status()
        predicted = response.json().get("final") or []
        recalls.append(recall_at_k(predicted, gt, k))
        ndcgs.append(ndcg_at_k(predicted, gt, k))
    elapsed = time.time() - t0
    recall_avg = statistics.mean(recalls)
    ndcg_avg = statistics.mean(ndcgs)
    return recall_avg, ndcg_avg, score(recall_avg, ndcg_avg), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    entries = load_dataset(args.dataset)[: args.limit]
    print(f"N={len(entries)}  K={args.k}")

    # Query counts are capped by MAX_DENSE_QUERIES/MAX_SPARSE_QUERIES (default 3),
    # so meaningful variants go below the cap, not above it.
    configs = [
        ("rrf, full", {"fusion": "rrf"}),
        ("dbsf, full", {"fusion": "dbsf"}),
        ("rrf, 1+1", {"fusion": "rrf", "max_dense": 1, "max_sparse": 1}),
        ("dbsf, 1+1", {"fusion": "dbsf", "max_dense": 1, "max_sparse": 1}),
        ("rrf, 2+2", {"fusion": "rrf", "max_dense": 2, "max_sparse": 2}),
        ("dbsf, 2+2", {"fusion": "dbsf", "max_dense": 2, "max_sparse": 2}),
        ("dbsf, full, no_rescore", {"fusion": "dbsf", "no_rescore": "true"}),
    ]

    header = f"{'config':<25} {'R@'+str(args.k):<10} {'nDCG@'+str(args.k):<10} {'score':<10} {'time':<8}"
    print(header)
    print("-" * len(header))
    for label, params in configs:
        recall_avg, ndcg_avg, total_score, elapsed = run_config(entries, params, args.k)
        print(f"{label:<25} {recall_avg:<10.4f} {ndcg_avg:<10.4f} {total_score:<10.4f} {elapsed:<8.1f}")


if __name__ == "__main__":
    main()
