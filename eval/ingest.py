from __future__ import annotations

import hashlib
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

INDEX_URL = os.getenv("INDEX_URL", "http://localhost:8001")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "messages")
DENSE_MODEL_NAME = os.getenv(
    "DENSE_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DENSE_SIZE = int(os.getenv("DENSE_VECTOR_SIZE", "384"))
DATA_PATH = Path(os.getenv("DATA_PATH", "data/Dataset_main.json"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
DELETE_EXISTING_CHAT_POINTS = os.getenv("DELETE_EXISTING_CHAT_POINTS", "1") == "1"
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "0") == "1"
# Route the synthetic JSONL corpus through the index service so chunking is
# exercised too, instead of writing one hand-made chunk per answer.
SYNTHETIC_VIA_INDEX = os.getenv("SYNTHETIC_VIA_INDEX", "0") == "1"


def default_document_prefix(model_name: str) -> str:
    return "passage: " if "e5" in model_name.lower() else ""


DENSE_DOCUMENT_PREFIX = os.getenv("DENSE_DOCUMENT_PREFIX", default_document_prefix(DENSE_MODEL_NAME))

_CHUNK_ID_NAMESPACE = uuid.UUID("6f8c3a1e-0000-0000-0000-000000000001")


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL_NAME)


def stable_chunk_id(chat_id: str, chunk: dict) -> str:
    page_hash = hashlib.sha1(chunk["page_content"].encode("utf-8"), usedforsecurity=False).hexdigest()
    key = f"{chat_id}:" + ",".join(sorted(chunk["message_ids"])) + f":{page_hash}"
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, key))


def ensure_payload_indexes(qc: QdrantClient, name: str) -> None:
    # Keep in sync with scripts/qdrant_init.sh.
    qc.create_payload_index(name, field_name="metadata.chat_id", field_schema=models.PayloadSchemaType.KEYWORD)
    qc.create_payload_index(name, field_name="metadata.start", field_schema=models.PayloadSchemaType.INTEGER)
    qc.create_payload_index(name, field_name="metadata.end", field_schema=models.PayloadSchemaType.INTEGER)


def ensure_collection(qc: QdrantClient, name: str, dense_size: int) -> None:
    # Keep in sync with scripts/qdrant_init.sh.
    if not qc.collection_exists(name):
        qc.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )
    ensure_payload_indexes(qc, name)


def recreate_collection(qc: QdrantClient, name: str, dense_size: int) -> None:
    if qc.collection_exists(name):
        qc.delete_collection(name)
    ensure_collection(qc, name, dense_size)


def delete_existing_chat_points(qc: QdrantClient, collection_name: str, chat_id: str) -> None:
    qc.delete(
        collection_name=collection_name,
        wait=True,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.chat_id",
                        match=models.MatchValue(value=chat_id),
                    ),
                ],
            ),
        ),
    )


def embed_dense_batch(texts: list[str]) -> list[list[float]]:
    prefixed = [f"{DENSE_DOCUMENT_PREFIX}{text}" for text in texts]
    return [vector.tolist() for vector in get_dense_model().embed(prefixed)]


def build_metadata(chat: dict, chunk: dict, messages_by_id: dict) -> dict:
    msg_ids = chunk["message_ids"]
    messages = [messages_by_id[message_id] for message_id in msg_ids if message_id in messages_by_id]
    if not messages:
        raise ValueError(f"chunk has no resolvable messages: {msg_ids[:3]}")

    times = [message["time"] for message in messages]
    participants = sorted({message["sender_id"] for message in messages})
    mentions = sorted({mention for message in messages for mention in (message.get("mentions") or [])})
    return {
        "chat_name": chat["name"],
        "chat_type": chat["type"],
        "chat_id": chat["id"],
        "chat_sn": chat["sn"],
        "thread_sn": next((message.get("thread_sn") for message in messages if message.get("thread_sn")), None),
        "message_ids": msg_ids,
        "message_blocks": chunk.get("message_blocks") or [],
        # Integers: the search service filters on these with Qdrant range conditions.
        "start": min(times),
        "end": max(times),
        "participants": participants,
        "mentions": mentions,
        "contains_forward": any(message.get("is_forward") for message in messages),
        "contains_quote": any(message.get("is_quote") for message in messages),
    }


def load_index_payload(data_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    data = json.loads(data_path.read_text())
    chat = data["chat"]
    messages = data["messages"]
    messages_by_id = {message["id"]: message for message in messages}
    return chat, messages, messages_by_id


def build_synthetic_chat(data_path: Path) -> dict[str, Any]:
    return {
        "id": f"synthetic://{data_path.stem}",
        "name": f"Synthetic Eval Corpus {data_path.stem}",
        "sn": f"synthetic://{data_path.stem}",
        "type": "group",
        "is_public": False,
        "members_count": 0,
        "members": [],
    }


def iter_synthetic_answers(data_path: Path):
    seen_ids: set[str] = set()
    with data_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            answer = entry.get("answer") or {}
            message_ids = [str(message_id) for message_id in (answer.get("message_ids") or []) if message_id]
            answer_text = str(answer.get("text") or "").strip()
            if not message_ids or not answer_text:
                continue
            for message_id in message_ids:
                if message_id in seen_ids:
                    continue
                seen_ids.add(message_id)
                yield message_id, answer_text


def load_synthetic_messages(data_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Synthetic corpus as raw messages, so /index runs the real chunking pipeline."""
    chat = build_synthetic_chat(data_path)
    messages: list[dict[str, Any]] = []
    base_time = 1_700_000_000

    for offset, (message_id, answer_text) in enumerate(iter_synthetic_answers(data_path), start=1):
        messages.append(
            {
                "id": message_id,
                "time": base_time + offset,
                "sender_id": "synthetic@eval.local",
                "text": answer_text,
                "file_snippets": "",
                "thread_sn": None,
                "mentions": [],
                "is_system": False,
                "is_hidden": False,
                "is_forward": False,
                "is_quote": False,
            }
        )

    messages_by_id = {message["id"]: message for message in messages}
    return chat, messages, messages_by_id


def load_synthetic_eval_chunks(data_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    chat = build_synthetic_chat(data_path)

    chunks: list[dict[str, Any]] = []
    messages_by_id: dict[str, dict[str, Any]] = {}
    base_time = 1_700_000_000

    for offset, (message_id, answer_text) in enumerate(iter_synthetic_answers(data_path), start=1):
        timestamp = base_time + offset
        messages_by_id[message_id] = {
            "id": message_id,
            "time": timestamp,
            "sender_id": "synthetic@eval.local",
            "mentions": [],
            "thread_sn": None,
            "is_forward": False,
            "is_quote": False,
        }
        message_block_text = (
            f"[2023-11-14 22:13:{timestamp % 60:02d} UTC | synthetic@eval.local]\n{answer_text}"
        )
        chunks.append(
            {
                "page_content": (
                    f"CHAT: {chat['name']}\n\n"
                    f"CHAT_TYPE: {chat['type']}\n\n"
                    f"CHAT_ID: {chat['id']}\n\n"
                    "MESSAGES:\n\n"
                    f"{message_block_text}"
                ),
                "dense_content": f"chat {chat['name']}\n\nchat_type {chat['type']}\n\n{answer_text}",
                "sparse_content": (
                    f"{chat['name']}\n{chat['type']}\n{chat['id']}\n"
                    f"{answer_text}\n\nsender: synthetic@eval.local"
                ),
                "message_ids": [message_id],
                "message_blocks": [{"message_id": message_id, "text": message_block_text}],
            }
        )

    return chat, chunks, messages_by_id


def is_synthetic_eval_jsonl(data_path: Path) -> bool:
    if data_path.suffix.lower() != ".jsonl":
        return False
    with data_path.open() as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return False
    try:
        first = json.loads(first_line)
    except json.JSONDecodeError:
        return False
    return "question" in first and "answer" in first


def main() -> None:
    http = httpx.Client(timeout=300.0)

    synthetic_mode = is_synthetic_eval_jsonl(DATA_PATH)
    if synthetic_mode and not SYNTHETIC_VIA_INDEX:
        print(f"[1/4] build synthetic corpus from {DATA_PATH}")
        chat, chunks, messages_by_id = load_synthetic_eval_chunks(DATA_PATH)
        print(f"      -> {len(chunks)} synthetic chunks")
    else:
        if synthetic_mode:
            chat, messages, messages_by_id = load_synthetic_messages(DATA_PATH)
        else:
            chat, messages, messages_by_id = load_index_payload(DATA_PATH)
        print(f"[1/4] POST /index ({len(messages)} messages)")
        response = http.post(
            f"{INDEX_URL}/index",
            json={"data": {"chat": chat, "overlap_messages": [], "new_messages": messages}},
        )
        response.raise_for_status()
        chunks = response.json()["results"]
        print(f"      -> {len(chunks)} chunks")

    print("[2/4] POST /sparse_embedding")
    response = http.post(
        f"{INDEX_URL}/sparse_embedding",
        json={"texts": [chunk["sparse_content"] for chunk in chunks]},
    )
    response.raise_for_status()
    sparse_vectors = response.json()["vectors"]

    print(f"[3/4] dense embed locally ({len(chunks)} in batches of {BATCH_SIZE})")
    dense_vectors: list[list[float]] = []
    for index in range(0, len(chunks), BATCH_SIZE):
        batch = [chunk["dense_content"] for chunk in chunks[index : index + BATCH_SIZE]]
        dense_vectors.extend(embed_dense_batch(batch))
        print(f"      batch {index // BATCH_SIZE + 1} done")

    print(f"[4/4] Qdrant upsert -> {COLLECTION}")
    qdrant = QdrantClient(url=QDRANT_URL)
    if RESET_COLLECTION:
        recreate_collection(qdrant, COLLECTION, DENSE_SIZE)
    else:
        ensure_collection(qdrant, COLLECTION, DENSE_SIZE)
    if DELETE_EXISTING_CHAT_POINTS and not RESET_COLLECTION:
        delete_existing_chat_points(qdrant, COLLECTION, chat["id"])

    points: list[models.PointStruct] = []
    for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors, strict=True):
        if not chunk["message_ids"]:
            continue
        points.append(
            models.PointStruct(
                id=stable_chunk_id(chat["id"], chunk),
                vector={
                    "dense": dense_vector,
                    "sparse": models.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                },
                payload={
                    "page_content": chunk["page_content"],
                    "metadata": build_metadata(chat, chunk, messages_by_id),
                },
            )
        )

    result = qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
    print(f"done. {len(points)} points status={result.status}")


if __name__ == "__main__":
    main()
