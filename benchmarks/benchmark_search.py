#!/usr/bin/env python3
"""Benchmark Maple search quality and HTTP latency with no third-party deps.

The runner deliberately keeps one HTTP connection per worker so localhost TCP
setup is not counted for every query. Each dataset query is sent once per round;
round order is shuffled with a fixed seed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import platform
import random
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class Target:
    scheme: str
    host: str
    port: int
    path: str


_THREAD_LOCAL = threading.local()


def parse_target(raw_url: str) -> Target:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"unsupported URL: {raw_url}")
    default_port = 443 if parsed.scheme == "https" else 80
    path = parsed.path or "/search"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return Target(parsed.scheme, parsed.hostname, parsed.port or default_port, path)


def new_connection(target: Target, timeout: float) -> http.client.HTTPConnection:
    connection_type = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    return connection_type(target.host, target.port, timeout=timeout)


def thread_connection(target: Target, timeout: float) -> http.client.HTTPConnection:
    connection = getattr(_THREAD_LOCAL, "connection", None)
    connection_target = getattr(_THREAD_LOCAL, "target", None)
    if connection is None or connection_target != target:
        connection = new_connection(target, timeout)
        _THREAD_LOCAL.connection = connection
        _THREAD_LOCAL.target = target
    return connection


def post_json(
    target: Target,
    payload: dict[str, Any],
    *,
    timeout: float,
    connection: http.client.HTTPConnection | None = None,
) -> tuple[dict[str, Any], float, int, int, http.client.HTTPConnection]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    connection = connection or thread_connection(target, timeout)
    started = time.perf_counter_ns()
    retries = 0
    while True:
        try:
            connection.request("POST", target.path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            break
        except (ConnectionError, http.client.HTTPException, OSError):
            if retries >= 1:
                raise
            retries += 1
            connection.close()
            connection = new_connection(target, timeout)
            if getattr(_THREAD_LOCAL, "connection", None) is not None:
                _THREAD_LOCAL.connection = connection
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if response.status >= 400:
        raise RuntimeError(f"HTTP {response.status}: {response_body.decode(errors='replace')[:500]}")
    return json.loads(response_body), elapsed_ms, len(response_body), retries, connection


def load_entries(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text()
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def extract_ids(body: dict[str, Any]) -> list[str]:
    if "final" in body:
        raw_ids = body.get("final") or []
    else:
        raw_ids = [
            message_id
            for item in (body.get("results") or [])
            for message_id in (item.get("message_ids") or [])
        ]
    return list(dict.fromkeys(str(message_id) for message_id in raw_ids))


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    return len(set(predicted[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, message_id in enumerate(predicted[:k])
        if message_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal


def mrr_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    for rank, message_id in enumerate(predicted[:k], start=1):
        if message_id in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(samples: list[dict[str, Any]], wall_seconds: float) -> dict[str, float]:
    values = [sample["latency_ms"] for sample in samples]
    return {
        "requests": len(values),
        "wall_seconds": wall_seconds,
        "throughput_qps": len(values) / wall_seconds if wall_seconds else 0.0,
        "mean_ms": statistics.mean(values) if values else 0.0,
        "stdev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values, default=0.0),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values, default=0.0),
        "mean_response_bytes": statistics.mean(sample["response_bytes"] for sample in samples) if samples else 0.0,
        "retries": sum(sample["retries"] for sample in samples),
        "error_rate": 0.0,
    }


def quality_summary(samples: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    positives = [sample for sample in samples if sample["relevant"]]
    negatives = [sample for sample in samples if not sample["relevant"]]
    summary: dict[str, Any] = {}
    for k in ks:
        recalls = []
        ndcgs = []
        mrrs = []
        for sample in positives:
            predicted = sample["predicted"]
            relevant = set(sample["relevant"])
            recalls.append(recall_at_k(predicted, relevant, k))
            ndcgs.append(ndcg_at_k(predicted, relevant, k))
            mrrs.append(mrr_at_k(predicted, relevant, k))
        summary[str(k)] = {
            "recall": statistics.mean(recalls) if recalls else 0.0,
            "ndcg": statistics.mean(ndcgs) if ndcgs else 0.0,
            "mrr": statistics.mean(mrrs) if mrrs else 0.0,
            "score_80r_20n": (
                statistics.mean(recalls) * 0.8 + statistics.mean(ndcgs) * 0.2
                if recalls
                else 0.0
            ),
        }
    summary["counts"] = {"positive": len(positives), "negative": len(negatives)}
    summary["negative_queries"] = {
        "no_result_rate": (
            statistics.mean(not sample["predicted"] for sample in negatives)
            if negatives
            else None
        ),
        "false_positive_rate": (
            statistics.mean(bool(sample["predicted"]) for sample in negatives)
            if negatives
            else None
        ),
    }
    return summary


def execute_one(
    entry: dict[str, Any],
    target: Target,
    timeout: float,
    question_mode: str,
    connection: http.client.HTTPConnection | None = None,
) -> tuple[dict[str, Any], http.client.HTTPConnection]:
    question = entry["question"]
    if question_mode == "text-only":
        question = {"text": question["text"]}
    body, latency_ms, response_bytes, retries, connection = post_json(
        target,
        {"question": question},
        timeout=timeout,
        connection=connection,
    )
    return (
        {
            "id": str(entry["id"]),
            "latency_ms": latency_ms,
            "response_bytes": response_bytes,
            "retries": retries,
            "predicted": extract_ids(body),
            "relevant": [str(message_id) for message_id in entry["answer"]["message_ids"]],
        },
        connection,
    )


def run_round(
    entries: list[dict[str, Any]],
    target: Target,
    *,
    concurrency: int,
    timeout: float,
    question_mode: str,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    if concurrency == 1:
        connection = new_connection(target, timeout)
        try:
            samples = []
            for entry in entries:
                sample, connection = execute_one(entry, target, timeout, question_mode, connection)
                samples.append(sample)
        finally:
            connection.close()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            samples = list(
                pool.map(
                    lambda entry: execute_one(entry, target, timeout, question_mode)[0],
                    entries,
                )
            )
    return samples, time.perf_counter() - started


def benchmark_mode(
    entries: list[dict[str, Any]],
    target: Target,
    *,
    concurrency: int,
    rounds: int,
    seed: int,
    timeout: float,
    ks: list[int],
    question_mode: str,
) -> dict[str, Any]:
    all_samples: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    total_wall = 0.0
    for round_index in range(rounds):
        ordered = list(entries)
        random.Random(seed + round_index).shuffle(ordered)
        samples, wall_seconds = run_round(
            ordered,
            target,
            concurrency=concurrency,
            timeout=timeout,
            question_mode=question_mode,
        )
        for sample in samples:
            sample["round"] = round_index + 1
        all_samples.extend(samples)
        total_wall += wall_seconds
        round_summaries.append(
            {
                "round": round_index + 1,
                "latency": latency_summary(samples, wall_seconds),
                "quality": quality_summary(samples, ks),
            }
        )
        print(
            f"  c={concurrency} round={round_index + 1}: "
            f"p50={round_summaries[-1]['latency']['p50_ms']:.1f}ms "
            f"p95={round_summaries[-1]['latency']['p95_ms']:.1f}ms "
            f"qps={round_summaries[-1]['latency']['throughput_qps']:.2f}"
        )
    return {
        "concurrency": concurrency,
        "rounds": round_summaries,
        "aggregate": {
            "latency": latency_summary(all_samples, total_wall),
            "quality": quality_summary(all_samples, ks),
        },
        "samples": all_samples,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8002/search")
    parser.add_argument("--dataset", type=Path, default=Path("data/Dataset_main_questions.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/Dataset_main.json"))
    parser.add_argument("--index-points", type=int, default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", default="1,3,5,10,20,50")
    parser.add_argument("--concurrencies", default="1,8")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--dataset-warmup-rounds", type=int, default=1)
    parser.add_argument("--question-mode", choices=("full", "text-only"), default="full")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    entries = load_entries(args.dataset)
    ks = [int(value) for value in args.ks.split(",")]
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    target = parse_target(args.url)
    if not entries:
        parser.error("dataset is empty")
    if not ks or any(k <= 0 for k in ks):
        parser.error("all --ks values must be positive")
    if not concurrencies or any(value <= 0 for value in concurrencies):
        parser.error("all --concurrencies values must be positive")
    if args.rounds <= 0 or args.warmups < 0 or args.dataset_warmup_rounds < 0:
        parser.error("rounds must be positive and warmup counts non-negative")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if not args.corpus.is_file():
        parser.error(f"corpus does not exist: {args.corpus}")

    warmup_payload = {"question": {"text": "технический прогрев поисковой модели"}}
    warmup_samples = []
    warmup_retries = []
    warmup_connection = new_connection(target, args.timeout)
    try:
        for _ in range(args.warmups):
            _, elapsed_ms, _, retries, warmup_connection = post_json(
                target,
                warmup_payload,
                timeout=args.timeout,
                connection=warmup_connection,
            )
            warmup_samples.append(elapsed_ms)
            warmup_retries.append(retries)
    finally:
        warmup_connection.close()
    print(f"{args.label}: warmups ms = {', '.join(f'{value:.1f}' for value in warmup_samples)}")

    dataset_warmups = []
    for warmup_index in range(args.dataset_warmup_rounds):
        ordered = list(entries)
        random.Random(args.seed - warmup_index - 1).shuffle(ordered)
        samples, wall_seconds = run_round(
            ordered,
            target,
            concurrency=1,
            timeout=args.timeout,
            question_mode=args.question_mode,
        )
        summary = latency_summary(samples, wall_seconds)
        dataset_warmups.append(summary)
        print(
            f"  dataset warmup={warmup_index + 1}: "
            f"p50={summary['p50_ms']:.1f}ms p95={summary['p95_ms']:.1f}ms"
        )

    modes = [
        benchmark_mode(
            entries,
            target,
            concurrency=concurrency,
            rounds=args.rounds,
            seed=args.seed,
            timeout=args.timeout,
            ks=ks,
            question_mode=args.question_mode,
        )
        for concurrency in concurrencies
    ]

    result = {
        "schema_version": 1,
        "label": args.label,
        "commit": args.commit,
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "target_url": args.url,
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "queries": len(entries),
        },
        "corpus": {
            "path": str(args.corpus),
            "sha256": sha256(args.corpus),
            "index_points": args.index_points,
        },
        "config": {
            "ks": ks,
            "concurrencies": concurrencies,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "dataset_warmup_rounds": args.dataset_warmup_rounds,
            "question_mode": args.question_mode,
            "seed": args.seed,
            "timeout": args.timeout,
        },
        "warmup_latency_ms": warmup_samples,
        "warmup_retries": warmup_retries,
        "dataset_warmups": dataset_warmups,
        "modes": modes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
