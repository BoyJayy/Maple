"""Build a chat-shaped JSON from dataset_ts.jsonl for local chunker diagnostics.

This is only a dev tool. It lets eval data go through the real /index chunker
path instead of the synthetic eval/ingest shortcut.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CHAT = {
    "id": "ts_synth://dataset_ts",
    "name": "Synthetic TS Chat",
    "sn": "ts_synth://dataset_ts",
    "type": "group",
    "is_public": False,
    "members_count": 3,
    "members": None,
}

SENDERS = ["alice@ts.local", "bob@ts.local", "carol@ts.local"]


def build(dataset_path: Path, out_path: Path, *, base_time: int, time_step: int) -> None:
    by_mid: dict[str, str] = {}
    with dataset_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            answer = entry.get("answer") or {}
            mids = [str(mid) for mid in (answer.get("message_ids") or []) if mid]
            text = str(answer.get("text") or "").strip()
            if not mids or not text:
                continue
            for mid in mids:
                by_mid.setdefault(mid, text)

    ordered = sorted(by_mid, key=lambda mid: int(mid.split("_")[1]))
    messages = []
    for mid in ordered:
        idx = int(mid.split("_")[1])
        messages.append(
            {
                "id": mid,
                "thread_sn": None,
                "time": base_time + idx * time_step,
                "text": by_mid[mid],
                "sender_id": SENDERS[idx % len(SENDERS)],
                "file_snippets": "",
                "parts": [],
                "mentions": [],
                "member_event": None,
                "is_system": False,
                "is_hidden": False,
                "is_forward": False,
                "is_quote": False,
            }
        )

    payload = {"chat": CHAT, "messages": messages}
    out_path.write_text(json.dumps(payload, ensure_ascii=False))
    span = messages[-1]["time"] - messages[0]["time"] if messages else 0
    print(f"wrote {out_path}  msgs={len(messages)}  span_seconds={span}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset_ts.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/dataset_ts_chat.json"))
    parser.add_argument("--base-time", type=int, default=1_700_000_000)
    parser.add_argument(
        "--time-step",
        type=int,
        default=600,
        help="seconds per id unit; gaps of 7/11 units become 7*step / 11*step",
    )
    args = parser.parse_args()
    build(args.dataset, args.out, base_time=args.base_time, time_step=args.time_step)


if __name__ == "__main__":
    main()
