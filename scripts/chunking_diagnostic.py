"""Diagnostic: запустить chunking на Go Nova.json и распечатать статистику.

Смотрим: размер чанков в символах, число сообщений в чанке, time gaps
между соседями, coverage (сколько сообщений попало хотя бы в один чанк),
дублирование (среднее число чанков на сообщение).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "index"))

import chunking as new
import main as baseline


def load_messages():
    data = json.loads((REPO / "data" / "Go Nova.json").read_text())
    return data, [baseline.Message(**m) for m in data["messages"]]


def histogram(values, bins):
    counts = Counter()
    for v in values:
        for i, (lo, hi) in enumerate(bins):
            if lo <= v < hi:
                counts[i] += 1
                break
    return counts


def main() -> int:
    data, messages = load_messages()
    kept = [m for m in messages if new.keep_message(m)]
    print(f"messages total : {len(messages)}")
    print(f"messages kept  : {len(kept)}  (dropped system/hidden)")
    print(f"thread count   : {len({m.thread_sn or '__root__' for m in kept})}")

    chunks = new.build_chunks([], messages)
    print(f"chunks         : {len(chunks)}")
    print()

    sizes = [len(c.page_content) for c in chunks]
    msg_counts = [len(c.message_ids) for c in chunks]

    print("=== Размер чанка (символы) ===")
    print(f"  min={min(sizes)}  max={max(sizes)}  mean={statistics.mean(sizes):.0f}  median={statistics.median(sizes):.0f}")
    size_bins = [(0, 400), (400, 800), (800, 1200), (1200, 1600), (1600, 2000), (2000, 10000)]
    size_hist = histogram(sizes, size_bins)
    for i, (lo, hi) in enumerate(size_bins):
        bar = "█" * size_hist.get(i, 0)
        print(f"  [{lo:>5}..{hi:<5}) {size_hist.get(i, 0):>2} {bar}")

    print()
    print("=== Сообщений в чанке ===")
    print(f"  min={min(msg_counts)}  max={max(msg_counts)}  mean={statistics.mean(msg_counts):.1f}  median={statistics.median(msg_counts):.1f}")
    msg_bins = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 100)]
    msg_hist = histogram(msg_counts, msg_bins)
    for i, (lo, hi) in enumerate(msg_bins):
        label = f"[{lo}..{hi})"
        bar = "█" * msg_hist.get(i, 0)
        print(f"  {label:<12} {msg_hist.get(i, 0):>2} {bar}")

    print()
    print("=== Coverage ===")
    msg_appear = Counter()
    for c in chunks:
        for mid in c.message_ids:
            msg_appear[mid] += 1
    kept_ids = {m.id for m in kept}
    covered = sum(1 for mid in kept_ids if msg_appear[mid] > 0)
    uncovered = kept_ids - set(msg_appear)
    dup_ratio = sum(msg_appear.values()) / len(kept_ids) if kept_ids else 0.0
    print(f"  kept msg covered : {covered}/{len(kept_ids)}  ({100 * covered / len(kept_ids):.0f}%)")
    print(f"  dup ratio        : {dup_ratio:.2f}x  (среднее число чанков на сообщение)")
    if uncovered:
        print(f"  uncovered        : {sorted(uncovered)[:5]}...")

    print()
    print("=== Time gaps между соседними (kept) сообщениями ===")
    kept_sorted = sorted(kept, key=lambda m: (m.time, m.id))
    gaps = [kept_sorted[i + 1].time - kept_sorted[i].time for i in range(len(kept_sorted) - 1)]
    if gaps:
        print(f"  min={min(gaps)}s  max={max(gaps)}s  median={statistics.median(gaps):.0f}s")
        gap_bins = [(0, 60), (60, 600), (600, 3600), (3600, 86400), (86400, 10**9)]
        labels = ["<1m", "1-10m", "10m-1h", "1h-1d", ">1d"]
        gap_hist = histogram(gaps, gap_bins)
        for i, label in enumerate(labels):
            bar = "█" * gap_hist.get(i, 0)
            print(f"  {label:<8} {gap_hist.get(i, 0):>2} {bar}")

    print()
    print("=== Чанки — preview ===")
    for i, c in enumerate(chunks):
        first = c.message_ids[0][-6:]
        last = c.message_ids[-1][-6:]
        preview = c.page_content[:60].replace("\n", " ")
        print(f"  {i:2} msgs={len(c.message_ids)} chars={len(c.page_content):4} [{first}..{last}]  {preview!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
