"""Structural parity test.

Сравнивает модульный chunking (index/chunking.py) с baseline inline
build_chunks (index/main.py) на data/Go Nova.json.

После Этапа 2 полный content-parity НЕ ожидается:
  message_ids   — должны совпадать (structural)
  dense_content — должен совпадать (clean semantics, без префиксов)
  page_content  — намеренно отличается: добавлены [timestamp | sender_id] headers
  sparse_content — намеренно отличается: добавлены sender_id и mentions

Скрипт фейлится, только если расходится structure (message_ids/dense).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "index"))

import chunking as new
import main as baseline


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
        print("FAIL: chunk count mismatch")
        return 1

    structural_fail = 0
    intentional_diff = 0
    for i, (b, n) in enumerate(zip(baseline_chunks, new_chunks, strict=True)):
        if b.message_ids != n.message_ids:
            structural_fail += 1
            print(f"  chunk {i}: message_ids differ (STRUCTURAL)")
        if b.dense_content != n.dense_content:
            structural_fail += 1
            print(f"  chunk {i}: dense_content differ (STRUCTURAL — должно быть identical)")
        if b.page_content != n.page_content:
            intentional_diff += 1
        if b.sparse_content != n.sparse_content:
            intentional_diff += 1

    print()
    if structural_fail == 0:
        print("STRUCTURAL PARITY OK")
    else:
        print(f"STRUCTURAL FAIL — {structural_fail} mismatches")

    print(f"intentional content divergence: {intentional_diff} (page/sparse enrichment)")

    if baseline_chunks:
        print()
        print("=== Пример chunk[0] — дельта ===")
        n = new_chunks[0]
        print("--- dense_content (baseline == new):")
        print(n.dense_content[:200])
        print()
        print("--- page_content (new, с timestamp + sender):")
        print(n.page_content[:300])
        print()
        print("--- sparse_content (new, с sender_id + mentions):")
        print(n.sparse_content[:300])

    return 0 if structural_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
