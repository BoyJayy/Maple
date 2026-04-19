"""Run eval dataset through search service, compute Recall@K and nDCG@K.

Usage:
    python eval/run.py --dataset eval/dataset.jsonl
    python eval/run.py --dataset eval/dataset.jsonl --stages

With --stages, hits /_debug/search to report metrics at each pipeline phase:
    retrieval -> rescored -> reranked -> final
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import httpx
from metrics import ndcg_at_k, recall_at_k, score

SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8002")


def extract_ids(results: list[dict]) -> list[str]:
    return [mid for item in results for mid in (item.get("message_ids") or [])]


def load_dataset(path: Path) -> list[dict]:
    raw = path.read_text()
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def run(dataset_path: Path, k: int, verbose: bool, stages: bool) -> None:
    entries = load_dataset(dataset_path)
    http = httpx.Client(timeout=120.0)

    stage_scores: dict[str, list[tuple[float, float]]] = {}
    misses: list[dict] = []
    endpoint = "/_debug/search" if stages else "/search"

    qs_parts: list[str] = []
    if os.getenv("NO_RESCORE"):
        qs_parts.append("no_rescore=true")
    if os.getenv("NO_RERANK"):
        qs_parts.append("no_rerank=true")
    qs = ("?" + "&".join(qs_parts)) if qs_parts else ""

    for entry in entries:
        qid = entry["id"]
        question = entry["question"]
        gt = set(entry["answer"]["message_ids"])

        r = http.post(f"{SEARCH_URL}{endpoint}{qs}", json={"question": question})
        r.raise_for_status()
        body = r.json()

        if stages:
            stage_predictions: dict[str, list[str]] = dict(body.get("stages") or {})
            stage_predictions["final"] = body.get("final") or []
        else:
            stage_predictions = {"final": extract_ids(body.get("results") or [])}

        for stage_name, predicted in stage_predictions.items():
            r_k = recall_at_k(predicted, gt, k)
            n_k = ndcg_at_k(predicted, gt, k)
            stage_scores.setdefault(stage_name, []).append((r_k, n_k))

            if stage_name == "final" and r_k < 1.0:
                missed = gt - set(predicted[:k])
                misses.append({"id": qid, "recall": r_k, "missed": sorted(missed)})

            if verbose and stage_name == "final":
                print(f"  {qid}  R@{k}={r_k:.3f}  nDCG@{k}={n_k:.3f}  '{question['text'][:60]}'")

    print()
    print(f"N = {len(entries)}")
    stage_order = ["retrieval", "rescored", "reranked", "final"]
    ordered = [name for name in stage_order if name in stage_scores]
    ordered += [name for name in stage_scores if name not in stage_order]

    header = f"{'stage':<12} {'Recall@'+str(k):<12} {'nDCG@'+str(k):<12} {'score':<10}"
    print(header)
    print("-" * len(header))
    for stage_name in ordered:
        recalls = [r for r, _ in stage_scores[stage_name]]
        ndcgs = [n for _, n in stage_scores[stage_name]]
        recall_avg = statistics.mean(recalls) if recalls else 0.0
        ndcg_avg = statistics.mean(ndcgs) if ndcgs else 0.0
        s = score(recall_avg, ndcg_avg)
        print(f"{stage_name:<12} {recall_avg:<12.4f} {ndcg_avg:<12.4f} {s:<10.4f}")

    if misses:
        print(f"\nMisses ({len(misses)}):")
        for m in misses[:20]:
            print(f"  {m['id']}  R={m['recall']:.3f}  missed={m['missed']}")
        if len(misses) > 20:
            print(f"  ... and {len(misses) - 20} more")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="path to JSONL dataset")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--stages", action="store_true", help="hit /_debug/search and report per-stage metrics")
    args = p.parse_args()
    run(args.dataset, args.k, args.verbose, args.stages)


if __name__ == "__main__":
    main()
