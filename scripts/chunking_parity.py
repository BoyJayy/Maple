"""Parity test: new modular chunking vs baseline inline build_chunks.

Feeds все сообщения из data/Go Nova.json как new_messages в обе реализации
и сравнивает output побайтово (message_ids + content).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "index"))

import main as baseline
import chunking as new


def load_messages():
    data = json.loads((REPO / "data" / "Go Nova.json").read_text())
    return [baseline.Message(**m) for m in data["messages"]]


def main() -> int:
    messages = load_messages()
    print(f"loaded {len(messages)} messages from Go Nova.json")

    baseline_chunks = baseline.build_chunks([], messages)
    new_chunks = new.build_chunks([], messages)

    print(f"baseline: {len(baseline_chunks)} chunks")
    print(f"new     : {len(new_chunks)} chunks")

    if len(baseline_chunks) != len(new_chunks):
        print("DIFF: chunk count mismatch")
        return 1

    mismatches = 0
    for i, (b, n) in enumerate(zip(baseline_chunks, new_chunks, strict=True)):
        if b.message_ids != n.message_ids:
            mismatches += 1
            print(f"  chunk {i}: message_ids differ")
            print(f"    baseline: {b.message_ids}")
            print(f"    new     : {n.message_ids}")
        if b.page_content != n.page_content:
            mismatches += 1
            print(f"  chunk {i}: page_content differ (baseline={len(b.page_content)}, new={len(n.page_content)})")
        if b.dense_content != n.dense_content:
            mismatches += 1
            print(f"  chunk {i}: dense_content differ")
        if b.sparse_content != n.sparse_content:
            mismatches += 1
            print(f"  chunk {i}: sparse_content differ")

    if mismatches == 0:
        print("PARITY OK — all chunks identical")
        return 0
    print(f"PARITY FAIL — {mismatches} mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
