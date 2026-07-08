"""Build an eval corpus from the Den4ikAI/russian_dialogues_2 HF dataset (MIT).

Downloads dialogues through the HF datasets-server REST API (no extra deps),
filters them, groups them into topic buckets so the corpus contains thematic
distractors, assigns synthetic senders/mentions/threads and deterministic
timestamps, and writes:
- a chat payload in the Dataset_main.json format (ready for eval/ingest.py);
- a manifest with per-dialogue ids/dates/topics for authoring questions.

Realism knobs (all deterministic under --seed):
- topic buckets: several dialogues per topic, so retrieval has to pick THE
  conversation, not the only one about cars;
- 2-3 speakers per dialogue, occasional @mentions filled into `mentions`;
- some dialogue pairs are interleaved inside one time window (one of the two
  gets a thread_sn), exercising thread- and gap-based chunk boundaries;
- rare synthetic stack-trace messages exercise technical compression.

Timestamps are synthetic, which makes date-anchored eval questions verifiable:
the ground truth for "что обсуждали 3 мая" is known by construction.

Usage:
    python3 scripts/convert_hf_dialogues.py --count 300
    python3 scripts/convert_hf_dialogues.py --count 40 --no-structure  # v1-like
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "Den4ikAI/russian_dialogues_2"

MIN_TURNS = 6
MAX_TURNS = 20
MIN_TURN_CHARS = 15
MAX_TURN_CHARS = 600
MAX_SCAN_ROWS = 20000

# Coarse filter: dialogues with obscene or sensitive lexicon are skipped.
BLOCKED_RE = re.compile(
    r"ху[йеёя*]|пизд|еба[тлн]|заеб|уеб|бля|мудак|пид[ао]р|долбо[её]б|сук[аи]\b|насрать|говн",
    re.IGNORECASE,
)

TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "auto": re.compile(r"машин|авто\b|двигател|пробег|бензин|вариатор|коробк|колес|водител", re.IGNORECASE),
    "tech": re.compile(r"компьютер|видеокарт|материнк|клавиатур|виндо|процессор|смартфон|телефон|приложени|интернет", re.IGNORECASE),
    "pets": re.compile(r"\bкот\b|кошк|котан|котей|собак|щенк|лоток|животн", re.IGNORECASE),
    "money": re.compile(r"цен[аыу]|рубл|стоит|зарплат|банк|кредит|вклад|курс", re.IGNORECASE),
    "games": re.compile(r"игр[аеы]|скайрим|дот[ауе]|квест|сервер|лвл|маунт|перса", re.IGNORECASE),
    "movies": re.compile(r"фильм|сериал|кино|актер|актрис|смотрел", re.IGNORECASE),
    "food": re.compile(r"еда|мяс[оа]|готовит|рецепт|свинин|говядин|шаурм|пельмен", re.IGNORECASE),
    "health": re.compile(r"врач|болит|больниц|здоров|зрение|лечени|таблетк", re.IGNORECASE),
}

SENDER_POOL = [
    "anna.smirnova@corp.local",
    "boris.petrov@corp.local",
    "vera.kuznetsova@corp.local",
    "grigory.orlov@corp.local",
    "daria.volkova@corp.local",
    "egor.sokolov@corp.local",
    "zhanna.popova@corp.local",
    "ivan.fedorov@corp.local",
]

FIRST_NAMES = {
    "anna.smirnova@corp.local": "Анна",
    "boris.petrov@corp.local": "Борис",
    "vera.kuznetsova@corp.local": "Вера",
    "grigory.orlov@corp.local": "Григорий",
    "daria.volkova@corp.local": "Дарья",
    "egor.sokolov@corp.local": "Егор",
    "zhanna.popova@corp.local": "Жанна",
    "ivan.fedorov@corp.local": "Иван",
}

FAKE_TRACES = [
    "Опять упало при деплое:\nTraceback (most recent call last):\n  File \"app/worker.py\", line 214, in process\n    result = handler(payload)\n  File \"app/handlers.py\", line 88, in handler\n    return json.loads(raw)[\"data\"]\nKeyError: 'data'",
    "Смотрите что в логах:\npanic: runtime error: invalid memory address or nil pointer dereference\ngoroutine 42 [running]:\nmain.processBatch(0x0?)\n\t/srv/app/batch.go:117 +0x2f\nmain.main()\n\t/srv/app/main.go:31 +0x9c",
]

BASE_TIME = int(datetime(2024, 5, 1, 9, 0, tzinfo=UTC).timestamp())


def classify_topic(turns: list[str]) -> str:
    joined = " ".join(turns)
    best_topic, best_hits = "misc", 0
    for topic, pattern in TOPIC_PATTERNS.items():
        hits = len(pattern.findall(joined))
        if hits > best_hits:
            best_topic, best_hits = topic, hits
    return best_topic if best_hits >= 2 else "misc"


def fetch_pool(target: int, offset: int) -> dict[str, list[list[str]]]:
    """Scan the dataset and bucket acceptable dialogues by topic until we have
    roughly 2x the target (gives the selector room for distractor balance)."""
    pool: dict[str, list[list[str]]] = {}
    total = 0
    cursor = offset
    with httpx.Client(timeout=60.0) as http:
        while total < target * 2 and cursor < offset + MAX_SCAN_ROWS:
            for attempt in range(8):
                response = http.get(
                    API_URL,
                    params={
                        "dataset": DATASET,
                        "config": "default",
                        "split": "train",
                        "offset": cursor,
                        "length": 100,
                    },
                )
                if response.status_code == 429:
                    retry_after = float(response.headers.get("retry-after") or 2**attempt)
                    print(f"  429 at offset {cursor}, sleeping {retry_after:.0f}s")
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                break
            else:
                raise RuntimeError(f"rate-limited too long at offset {cursor}")
            time.sleep(0.4)
            rows = response.json().get("rows", [])
            if not rows:
                break
            cursor += len(rows)
            for row in rows:
                sample = row.get("row", {}).get("sample")
                if not isinstance(sample, list):
                    continue
                turns = [str(turn).strip() for turn in sample if str(turn).strip()]
                if not (MIN_TURNS <= len(turns) <= MAX_TURNS):
                    continue
                if any(len(turn) < MIN_TURN_CHARS or len(turn) > MAX_TURN_CHARS for turn in turns):
                    continue
                if BLOCKED_RE.search(" ".join(turns)):
                    continue
                pool.setdefault(classify_topic(turns), []).append(turns)
                total += 1
    return pool


def select_dialogues(pool: dict[str, list[list[str]]], count: int, rng: random.Random) -> list[tuple[str, list[str]]]:
    """Prefer topics with many dialogues: dense buckets are the distractors."""
    selected: list[tuple[str, list[str]]] = []
    topical = [(topic, items) for topic, items in pool.items() if topic != "misc"]
    topical.sort(key=lambda item: len(item[1]), reverse=True)

    for topic, items in topical:
        take = min(len(items), max(4, count // max(1, len(topical) + 2)))
        for turns in items[:take]:
            selected.append((topic, turns))
    for turns in pool.get("misc", []):
        if len(selected) >= count:
            break
        selected.append(("misc", turns))

    selected = selected[:count]
    rng.shuffle(selected)
    return selected


def assign_speakers(turns: list[str], rng: random.Random) -> list[tuple[str, str]]:
    """Two main speakers alternate; sometimes a third one interjects."""
    speakers = rng.sample(SENDER_POOL, 3)
    third_turn = rng.randrange(2, len(turns)) if rng.random() < 0.3 and len(turns) > 3 else None
    assigned = []
    for index, turn in enumerate(turns):
        sender = speakers[2] if index == third_turn else speakers[index % 2]
        assigned.append((sender, turn))
    return assigned


def build_corpus(
    dialogues: list[tuple[str, list[str]]],
    seed: int,
    *,
    structure: bool = True,
) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    chat = {
        "id": "hf://russian_dialogues_2",
        "name": "HF Dialogues Eval Corpus",
        "sn": "hf://russian_dialogues_2",
        "type": "group",
        "is_public": False,
        "members_count": len(SENDER_POOL),
        "members": [],
    }

    messages: list[dict] = []
    manifest: list[dict] = []
    current_time = BASE_TIME
    index = 0

    while index < len(dialogues):
        # 4..30 hours between sessions: exceeds MAX_TIME_GAP_SECONDS, so each
        # session lands in its own chunk window.
        current_time += rng.randint(4 * 3600, 30 * 3600)

        interleave = structure and rng.random() < 0.25 and index + 1 < len(dialogues)
        session = [dialogues[index]]
        if interleave:
            session.append(dialogues[index + 1])
        index += len(session)

        session_entries: list[dict] = []
        for slot, (topic, turns) in enumerate(session):
            assigned = assign_speakers(turns, rng) if structure else [
                (sender, turn)
                for sender, turn in zip(
                    [rng.sample(SENDER_POOL, 2)[i % 2] for i in range(len(turns))], turns, strict=True
                )
            ]
            thread_sn = f"thread-{len(manifest):03d}" if interleave and slot == 1 and rng.random() < 0.5 else None
            session_entries.append(
                {
                    "topic": topic,
                    "assigned": assigned,
                    "thread_sn": thread_sn,
                    "dialogue_index": len(manifest) + slot,
                }
            )

        # Weave session dialogues into one timeline block by block.
        cursors = [0] * len(session_entries)
        dialogue_messages: list[list[dict]] = [[] for _ in session_entries]
        while any(cursors[i] < len(session_entries[i]["assigned"]) for i in range(len(session_entries))):
            for slot, entry in enumerate(session_entries):
                block = rng.randint(1, 3)
                for _ in range(block):
                    if cursors[slot] >= len(entry["assigned"]):
                        break
                    sender, text = entry["assigned"][cursors[slot]]
                    current_time += rng.randint(30, 180)
                    mentions: list[str] = []
                    if structure and rng.random() < 0.12:
                        others = [s for s, _ in entry["assigned"] if s != sender]
                        if others:
                            mentioned = rng.choice(others)
                            text = f"@{FIRST_NAMES[mentioned]}, {text}"
                            mentions = [mentioned]
                    message = {
                        "id": f"dlg{entry['dialogue_index']:03d}_msg{cursors[slot]:02d}",
                        "time": current_time,
                        "text": text,
                        "sender_id": sender,
                        "file_snippets": "",
                        "thread_sn": entry["thread_sn"],
                        "parts": None,
                        "mentions": mentions,
                        "member_event": None,
                        "is_system": False,
                        "is_hidden": False,
                        "is_forward": False,
                        "is_quote": False,
                    }
                    messages.append(message)
                    dialogue_messages[slot].append(message)
                    cursors[slot] += 1

        # Occasionally a technical trace lands at the session tail.
        if structure and rng.random() < 0.05:
            current_time += rng.randint(30, 180)
            trace_sender = rng.choice(SENDER_POOL)
            messages.append(
                {
                    "id": f"dlg{session_entries[0]['dialogue_index']:03d}_trace",
                    "time": current_time,
                    "text": rng.choice(FAKE_TRACES),
                    "sender_id": trace_sender,
                    "file_snippets": "",
                    "thread_sn": None,
                    "parts": None,
                    "mentions": [],
                    "member_event": None,
                    "is_system": False,
                    "is_hidden": False,
                    "is_forward": False,
                    "is_quote": False,
                }
            )

        for slot, entry in enumerate(session_entries):
            block_messages = dialogue_messages[slot]
            manifest.append(
                {
                    "dialogue": entry["dialogue_index"],
                    "topic": entry["topic"],
                    "date": datetime.fromtimestamp(block_messages[0]["time"], tz=UTC).strftime("%Y-%m-%d"),
                    "thread_sn": entry["thread_sn"],
                    "interleaved": len(session_entries) > 1,
                    "participants": sorted({message["sender_id"] for message in block_messages}),
                    "messages": [
                        {"id": message["id"], "sender": message["sender_id"], "text": message["text"]}
                        for message in block_messages
                    ],
                }
            )

    return {"chat": chat, "messages": messages}, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300, help="dialogues to keep after filtering")
    parser.add_argument("--offset", type=int, default=0, help="dataset row offset to start from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-structure", action="store_true", help="disable mentions/threads/interleaving")
    parser.add_argument("--output", type=Path, default=Path("data/Dataset_dialogues.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/Dataset_dialogues_manifest.json"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool = fetch_pool(args.count, args.offset)
    by_topic = {topic: len(items) for topic, items in sorted(pool.items(), key=lambda item: -len(item[1]))}
    print(f"pool: {sum(by_topic.values())} dialogues, topics: {by_topic}")

    dialogues = select_dialogues(pool, args.count, rng)
    if len(dialogues) < args.count:
        print(f"warning: only {len(dialogues)} dialogues passed the filters")

    payload, manifest = build_corpus(dialogues, args.seed, structure=not args.no_structure)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    interleaved = sum(1 for entry in manifest if entry["interleaved"])
    with_mentions = sum(1 for message in payload["messages"] if message["mentions"])
    print(f"dialogues: {len(manifest)}, messages: {len(payload['messages'])}")
    print(f"interleaved dialogues: {interleaved}, messages with mentions: {with_mentions}")
    print(f"corpus:   {args.output}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
