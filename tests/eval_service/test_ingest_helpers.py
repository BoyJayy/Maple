import json

from ingest import (
    default_document_prefix,
    is_synthetic_eval_jsonl,
    iter_synthetic_answers,
    stable_chunk_id,
)


def make_chunk(message_ids, page_content="text"):
    return {"message_ids": message_ids, "page_content": page_content}


def test_stable_chunk_id_is_deterministic():
    chunk = make_chunk(["m1", "m2"])
    assert stable_chunk_id("chat-1", chunk) == stable_chunk_id("chat-1", chunk)


def test_stable_chunk_id_ignores_message_order():
    assert stable_chunk_id("chat-1", make_chunk(["m2", "m1"])) == stable_chunk_id(
        "chat-1", make_chunk(["m1", "m2"])
    )


def test_stable_chunk_id_differs_by_chat_and_content():
    chunk = make_chunk(["m1"])
    assert stable_chunk_id("chat-1", chunk) != stable_chunk_id("chat-2", chunk)
    assert stable_chunk_id("chat-1", chunk) != stable_chunk_id(
        "chat-1", make_chunk(["m1"], page_content="other")
    )


def test_default_document_prefix_for_e5_models():
    assert default_document_prefix("intfloat/multilingual-e5-large") == "passage: "
    assert default_document_prefix("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") == ""


def test_is_synthetic_eval_jsonl(tmp_path):
    synthetic = tmp_path / "questions.jsonl"
    synthetic.write_text(json.dumps({"question": {"text": "q"}, "answer": {"message_ids": ["m1"], "text": "a"}}) + "\n")
    assert is_synthetic_eval_jsonl(synthetic)

    plain = tmp_path / "data.json"
    plain.write_text(json.dumps({"chat": {}, "messages": []}))
    assert not is_synthetic_eval_jsonl(plain)


def test_iter_synthetic_answers_dedupes_message_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    entries = [
        {"question": {"text": "q1"}, "answer": {"message_ids": ["m1", "m2"], "text": "первый ответ"}},
        {"question": {"text": "q2"}, "answer": {"message_ids": ["m2", "m3"], "text": "второй ответ"}},
        {"question": {"text": "q3"}, "answer": {"message_ids": [], "text": "без id"}},
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")

    answers = list(iter_synthetic_answers(path))
    ids = [message_id for message_id, _ in answers]
    assert ids == ["m1", "m2", "m3"]
    assert answers[2][1] == "второй ответ"
