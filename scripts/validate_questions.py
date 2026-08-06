"""Validate an eval questions JSONL against its corpus. Pure offline checks.

Usage:
    python3 scripts/validate_questions.py \
        --corpus data/Dataset_dialogues.json \
        --questions data/Dataset_dialogues_questions.jsonl \
        [--clean-output data/questions_clean.jsonl]

Checks:
- referenced message_ids exist in the corpus;
- non-negative questions have a non-empty answer text;
- date-anchored questions: date_range/date_mentions cover the GT messages'
  actual dates (they are synthetic, so this is exact);
- verbatim leak: if most of the question's content words appear in a single
  GT message, sparse retrieval finds it trivially — flagged as a warning;
- duplicate ids and unknown categories.

Exit code 1 if any hard error is found. --clean-output writes only valid
entries (leaky ones are kept: they are a warning, not an error).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

KNOWN_CATEGORIES = {"semantic", "exact", "date", "participant", "multihop", "negative", "uncategorized"}
WORD_RE = re.compile(r"[\wёЁ]+", re.UNICODE)
DATE_RE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?")
STOP = {
    "что", "как", "про", "кто", "где", "когда", "почему", "какой", "какая", "какие",
    "это", "был", "была", "было", "или", "для", "при", "чем", "чём", "его", "она",
    "они", "мы", "вы", "не", "на", "в", "и", "с", "по", "из", "у", "за", "до", "от",
}


def content_words(text: str) -> set[str]:
    return {word.lower() for word in WORD_RE.findall(text) if len(word) >= 3 and word.lower() not in STOP}


def parse_dates(question: dict) -> list[str]:
    found: list[str] = []
    date_range = question.get("date_range") or {}
    for value in (date_range.get("from"), date_range.get("to")):
        if value:
            found.append(str(value)[:10])
    for mention in question.get("date_mentions") or []:
        match = DATE_RE.search(str(mention))
        if match:
            found.append(match.group(0))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--clean-output", type=Path, default=None)
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text())
    messages_by_id = {message["id"]: message for message in corpus["messages"]}

    errors: list[str] = []
    warnings: list[str] = []
    valid_entries: list[dict] = []
    seen_ids: set[str] = set()
    per_category: dict[str, int] = {}

    for line_number, line in enumerate(args.questions.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc})")
            continue

        qid = str(entry.get("id") or f"line-{line_number}")
        entry_errors: list[str] = []

        if qid in seen_ids:
            entry_errors.append("duplicate id")
        seen_ids.add(qid)

        category = str(entry.get("category") or "uncategorized")
        if category not in KNOWN_CATEGORIES:
            warnings.append(f"{qid}: unknown category '{category}'")

        question = entry.get("question") or {}
        if not str(question.get("text") or "").strip():
            entry_errors.append("empty question.text")

        answer = entry.get("answer") or {}
        gt_ids = [str(message_id) for message_id in (answer.get("message_ids") or [])]
        missing = [message_id for message_id in gt_ids if message_id not in messages_by_id]
        if missing:
            entry_errors.append(f"unknown message_ids: {missing}")

        if category == "negative":
            if gt_ids:
                entry_errors.append("negative question must have empty message_ids")
        else:
            if not gt_ids:
                entry_errors.append("empty message_ids on a non-negative question")
            if not str(answer.get("text") or "").strip():
                entry_errors.append("empty answer.text")

        gt_messages = [messages_by_id[message_id] for message_id in gt_ids if message_id in messages_by_id]

        question_dates = parse_dates(question)
        if question_dates and gt_messages:
            gt_days = {
                datetime.fromtimestamp(message["time"], tz=UTC).strftime("%Y-%m-%d")
                for message in gt_messages
            }
            covered = set()
            for date_text in question_dates:
                day = datetime.fromisoformat(date_text if len(date_text) == 10 else date_text + "-01")
                for delta in (-1, 0, 1):
                    covered.add((day + timedelta(days=delta)).strftime("%Y-%m-%d"))
                if len(date_text) == 7:
                    covered.update(
                        f"{date_text}-{dom:02d}" for dom in range(1, 32)
                    )
            uncovered = gt_days - covered
            if uncovered:
                entry_errors.append(f"question dates {question_dates} do not cover GT days {sorted(uncovered)}")

        if gt_messages and category != "negative":
            words = content_words(str(question.get("text") or ""))
            if words:
                for message in gt_messages:
                    overlap = len(words & content_words(message["text"])) / len(words)
                    if overlap > 0.7:
                        warnings.append(f"{qid}: verbatim leak ({overlap:.0%} of question words in {message['id']})")
                        break

        if entry_errors:
            errors.extend(f"{qid}: {item}" for item in entry_errors)
        else:
            per_category[category] = per_category.get(category, 0) + 1
            valid_entries.append(entry)

    print(f"valid: {len(valid_entries)}  errors: {len(errors)}  warnings: {len(warnings)}")
    print("per category:", dict(sorted(per_category.items())))
    for message in errors[:30]:
        print(f"  ERROR   {message}")
    for message in warnings[:30]:
        print(f"  WARNING {message}")

    if args.clean_output:
        args.clean_output.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in valid_entries) + "\n"
        )
        print(f"clean file: {args.clean_output}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
