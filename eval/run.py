"""Run eval dataset through search service, compute Recall@K, nDCG@K and MRR@K.

Usage:
    python eval/run.py --dataset eval/dataset.jsonl
    python eval/run.py --dataset eval/dataset.jsonl --stages
    python eval/run.py --dataset eval/dataset.jsonl --ks 10,50
    python eval/run.py --dataset eval/dataset.jsonl --save-baseline eval/baseline.json
    python eval/run.py --dataset eval/dataset.jsonl --baseline eval/baseline.json

With --stages, hits /_debug/search to report metrics at each pipeline phase:
    retrieval -> rescored -> reranked -> final

Dataset entries may carry optional fields:
    "category": "semantic" | "exact" | "date" | ... — final-stage metrics are
        additionally broken down per category;
    empty answer.message_ids marks a negative question (no answer in corpus);
        such entries are counted but excluded from ranking metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import httpx
from metrics import bootstrap_ci, mrr_at_k, ndcg_at_k, recall_at_k, score

SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8002")
STAGE_ORDER = ["retrieval", "rescored", "reranked", "final"]


def extract_ids(results: list[dict]) -> list[str]:
    return [mid for item in results for mid in (item.get("message_ids") or [])]


def load_dataset(path: Path) -> list[dict]:
    raw = path.read_text()
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_baseline(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def baseline_stage_metrics(baseline: dict, stage: str, k: int) -> dict | None:
    """Supports both the legacy flat format (single k) and the multi-k format."""
    stages = baseline.get("stages", {})
    entry = stages.get(stage)
    if entry is None:
        return None
    if "ks" in baseline:
        return entry.get(str(k))
    if baseline.get("k") in (None, k):
        return entry
    return None


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def run(
    dataset_path: Path,
    ks: list[int],
    verbose: bool,
    stages: bool,
    baseline_path: Path | None,
    save_baseline_path: Path | None,
) -> None:
    entries = load_dataset(dataset_path)
    http = httpx.Client(timeout=120.0)

    qs_parts: list[str] = []
    if os.getenv("NO_RESCORE"):
        qs_parts.append("no_rescore=true")
    if os.getenv("NO_RERANK"):
        qs_parts.append("no_rerank=true")
    # Only /_debug/search understands the toggles, so their presence forces
    # the debug endpoint even without --stages.
    use_debug = stages or bool(qs_parts)
    endpoint = "/_debug/search" if use_debug else "/search"
    qs = ("?" + "&".join(qs_parts)) if qs_parts else ""

    primary_k = ks[0]
    # stage -> k -> list of (recall, ndcg, mrr) per question
    stage_scores: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
    # category -> list of (recall, ndcg, mrr) at primary_k, final stage only
    category_scores: dict[str, list[tuple[float, float, float]]] = {}
    misses: list[dict] = []
    negatives = 0

    for entry in entries:
        qid = entry["id"]
        question = entry["question"]
        gt = {str(message_id) for message_id in entry["answer"]["message_ids"]}

        r = http.post(f"{SEARCH_URL}{endpoint}{qs}", json={"question": question})
        r.raise_for_status()
        body = r.json()

        if use_debug:
            stage_predictions: dict[str, list[str]] = dict(body.get("stages") or {}) if stages else {}
            stage_predictions["final"] = body.get("final") or []
        else:
            stage_predictions = {"final": extract_ids(body.get("results") or [])}

        if not gt:
            negatives += 1
            continue

        for stage_name, predicted in stage_predictions.items():
            per_stage = stage_scores.setdefault(stage_name, {})
            for k in ks:
                per_stage.setdefault(k, []).append(
                    (recall_at_k(predicted, gt, k), ndcg_at_k(predicted, gt, k), mrr_at_k(predicted, gt, k))
                )

        final_predicted = stage_predictions["final"]
        r_k = recall_at_k(final_predicted, gt, primary_k)
        n_k = ndcg_at_k(final_predicted, gt, primary_k)
        m_k = mrr_at_k(final_predicted, gt, primary_k)
        category_scores.setdefault(str(entry.get("category") or "uncategorized"), []).append((r_k, n_k, m_k))

        if r_k < 1.0:
            missed = gt - set(final_predicted[:primary_k])
            misses.append({"id": qid, "recall": r_k, "missed": sorted(missed)})

        if verbose:
            print(
                f"  {qid}  R@{primary_k}={r_k:.3f}  nDCG@{primary_k}={n_k:.3f}  "
                f"MRR@{primary_k}={m_k:.3f}  '{question['text'][:60]}'"
            )

    scored_n = len(entries) - negatives
    print()
    print(f"N = {scored_n}" + (f"  (+{negatives} negative questions excluded)" if negatives else ""))

    ordered = [name for name in STAGE_ORDER if name in stage_scores]
    ordered += [name for name in stage_scores if name not in STAGE_ORDER]

    baseline = load_baseline(baseline_path) if baseline_path else {}
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for k in ks:
        header = f"{'stage':<12} {'Recall@'+str(k):<12} {'nDCG@'+str(k):<12} {'MRR@'+str(k):<12} {'score':<10}"
        print(header)
        print("-" * len(header))
        for stage_name in ordered:
            rows = stage_scores[stage_name][k]
            recall_avg = mean([r for r, _, _ in rows])
            ndcg_avg = mean([n for _, n, _ in rows])
            mrr_avg = mean([m for _, _, m in rows])
            s = score(recall_avg, ndcg_avg)
            summary.setdefault(stage_name, {})[str(k)] = {
                "recall": recall_avg,
                "ndcg": ndcg_avg,
                "mrr": mrr_avg,
                "score": s,
            }
            print(f"{stage_name:<12} {recall_avg:<12.4f} {ndcg_avg:<12.4f} {mrr_avg:<12.4f} {s:<10.4f}")
        if "final" in stage_scores and scored_n >= 10:
            rows = stage_scores["final"][k]
            recall_low, recall_high = bootstrap_ci([r for r, _, _ in rows])
            ndcg_low, ndcg_high = bootstrap_ci([n for _, n, _ in rows])
            print(
                f"{'final 95% CI':<12} [{recall_low:.4f}, {recall_high:.4f}]"
                f"          [{ndcg_low:.4f}, {ndcg_high:.4f}]"
            )
        print()

    if len(category_scores) > 1:
        header = f"{'category':<16} {'N':<5} {'Recall@'+str(primary_k):<12} {'nDCG@'+str(primary_k):<12} {'MRR@'+str(primary_k):<12}"
        print(header)
        print("-" * len(header))
        for category, rows in sorted(category_scores.items()):
            print(
                f"{category:<16} {len(rows):<5} {mean([r for r, _, _ in rows]):<12.4f} "
                f"{mean([n for _, n, _ in rows]):<12.4f} {mean([m for _, _, m in rows]):<12.4f}"
            )
        print()

    if baseline:
        print(f"Delta vs baseline ({baseline_path}):")
        for stage_name in ordered:
            for k in ks:
                base = baseline_stage_metrics(baseline, stage_name, k)
                if not base:
                    continue
                current = summary[stage_name][str(k)]
                deltas = "  ".join(
                    f"{metric}={current[metric] - base.get(metric, 0.0):+.4f}"
                    for metric in ("recall", "ndcg", "mrr", "score")
                )
                print(f"  {stage_name}@{k:<4} {deltas}")
        print()

    if save_baseline_path:
        if len(ks) == 1:
            payload = {
                "dataset": str(dataset_path),
                "k": primary_k,
                "n": scored_n,
                "stages": {stage: metrics[str(primary_k)] for stage, metrics in summary.items()},
            }
        else:
            payload = {"dataset": str(dataset_path), "ks": ks, "n": scored_n, "stages": summary}
        save_baseline_path.write_text(json.dumps(payload, indent=2))
        print(f"Baseline saved to {save_baseline_path}")

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
    p.add_argument("--ks", type=str, default=None, help="comma-separated list, e.g. 10,50 (overrides --k)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--stages", action="store_true", help="hit /_debug/search and report per-stage metrics")
    p.add_argument("--baseline", type=Path, default=None, help="baseline JSON to compare against")
    p.add_argument("--save-baseline", type=Path, default=None, help="write current results as baseline JSON")
    args = p.parse_args()

    ks = [int(item) for item in args.ks.split(",")] if args.ks else [args.k]

    baseline_path = args.baseline
    if baseline_path is None:
        default_baseline = Path("eval/baseline.json")
        if default_baseline.is_file():
            baseline_path = default_baseline

    run(args.dataset, ks, args.verbose, args.stages, baseline_path, args.save_baseline)


if __name__ == "__main__":
    main()
