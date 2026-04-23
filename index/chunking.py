from dataclasses import dataclass, replace
from datetime import UTC, datetime
import re

from config import (
    LONG_MESSAGE_CHAR_THRESHOLD,
    LONG_MESSAGE_LINE_THRESHOLD,
    MAX_CHUNK_CHARS,
    MAX_TIME_GAP_SECONDS,
    OVERLAP_MESSAGE_COUNT,
    SHORT_ACK_MESSAGES,
    SPLIT_MESSAGE_CHAR_THRESHOLD,
    SPLIT_SEGMENT_TARGET_CHARS,
    TECHNICAL_PREVIEW_CHARS,
    TECHNICAL_PREVIEW_LINES,
    TECHNICAL_TRACE_MARKERS,
    logger,
)
from schemas import Chat, IndexAPIItem, Message


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class NormalizedMessage:
    id: str
    time: int
    sender_id: str
    thread_sn: str | None
    text: str
    mentions: list[str]
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool
    fragment_index: int = 1
    fragment_count: int = 1


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def join_sections(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part).strip()


def extract_message_parts(message: Message) -> tuple[list[str], list[str]]:
    direct_parts: list[str] = []
    quoted_parts: list[str] = []
    for part in message.parts or []:
        part_text = normalize_text(part.get("text") or "")
        if not part_text:
            continue
        media_type = str(part.get("mediaType") or "").lower()
        if media_type == "quote":
            quoted_parts.append(part_text)
        else:
            direct_parts.append(part_text)
    return direct_parts, quoted_parts


def render_member_event(message: Message) -> str:
    event = message.member_event or {}
    event_type = str(event.get("type") or "").strip()
    members = [str(member).strip() for member in (event.get("members") or []) if str(member).strip()]
    if not event_type:
        return ""
    if members:
        return f"{event_type}: {', '.join(members)}"
    return event_type


def render_message(message: Message) -> str:
    direct_parts, quoted_parts = extract_message_parts(message)
    sections = [
        normalize_text(message.text),
        *direct_parts,
        *(f"Quoted message:\n{part}" for part in quoted_parts),
        render_member_event(message),
        normalize_text(message.file_snippets),
    ]
    return join_sections(sections)


def normalize_message(message: Message, *, is_overlap: bool) -> NormalizedMessage:
    return NormalizedMessage(
        id=message.id,
        time=message.time,
        sender_id=message.sender_id,
        thread_sn=message.thread_sn,
        text=render_message(message),
        mentions=[item.strip() for item in (message.mentions or []) if item.strip()],
        is_system=message.is_system,
        is_hidden=message.is_hidden,
        is_forward=message.is_forward,
        is_quote=message.is_quote,
    )


def is_message_searchable(message: NormalizedMessage) -> bool:
    if message.is_hidden:
        return False
    if not (message.text or message.mentions):
        return False

    compact_text = normalize_text(message.text).lower()
    if (
        compact_text in SHORT_ACK_MESSAGES
        and not message.mentions
        and not message.is_forward
        and not message.is_quote
    ):
        return False
    return True


def is_technical_message(text: str) -> bool:
    lowered = text.lower()
    return (
        len(text) >= LONG_MESSAGE_CHAR_THRESHOLD
        or len(text.splitlines()) >= LONG_MESSAGE_LINE_THRESHOLD
        or any(marker in lowered for marker in TECHNICAL_TRACE_MARKERS)
    )


def compress_technical_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    lines = normalized.splitlines()
    if len(lines) > TECHNICAL_PREVIEW_LINES:
        head = max(1, TECHNICAL_PREVIEW_LINES // 2)
        tail = max(1, TECHNICAL_PREVIEW_LINES - head - 1)
        omitted = max(0, len(lines) - head - tail)
        lines = [
            *lines[:head],
            f"... [{omitted} lines omitted] ...",
            *lines[-tail:],
        ]

    preview = "\n".join(lines)
    if len(preview) > TECHNICAL_PREVIEW_CHARS:
        omitted = len(preview) - TECHNICAL_PREVIEW_CHARS
        preview = preview[:TECHNICAL_PREVIEW_CHARS].rstrip() + f"\n... [truncated {omitted} chars]"
    return preview.strip()


def split_long_text(text: str, *, target_chars: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.split("\n") if part.strip()]
    atomic_parts: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= target_chars:
            atomic_parts.append(paragraph)
            continue

        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(paragraph) if part.strip()]
        if len(sentences) <= 1:
            sentences = [paragraph[index : index + target_chars] for index in range(0, len(paragraph), target_chars)]
        atomic_parts.extend(sentences)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0
    for part in atomic_parts:
        separator = 2 if current_parts else 0
        if current_parts and current_length + separator + len(part) > target_chars:
            chunks.append("\n".join(current_parts).strip())
            current_parts = [part]
            current_length = len(part)
        else:
            current_parts.append(part)
            current_length += len(part) + separator
    if current_parts:
        chunks.append("\n".join(current_parts).strip())
    return [chunk for chunk in chunks if chunk]


def split_message_for_chunking(message: NormalizedMessage) -> list[NormalizedMessage]:
    if is_technical_message(message.text):
        return [replace(message, text=compress_technical_text(message.text))]
    if len(message.text) <= SPLIT_MESSAGE_CHAR_THRESHOLD:
        return [message]

    segments = split_long_text(message.text, target_chars=SPLIT_SEGMENT_TARGET_CHARS)
    if len(segments) <= 1:
        return [message]

    return [
        replace(
            message,
            text=segment,
            fragment_index=index,
            fragment_count=len(segments),
        )
        for index, segment in enumerate(segments, start=1)
    ]


def format_timestamp(unix_time: int) -> str:
    return datetime.fromtimestamp(unix_time, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_header(message: NormalizedMessage) -> str:
    flags: list[str] = []
    if message.thread_sn:
        flags.append(f"thread={message.thread_sn}")
    if message.is_forward:
        flags.append("forward")
    if message.is_quote:
        flags.append("quote")
    if message.is_system:
        flags.append("system")
    if message.fragment_count > 1:
        flags.append(f"part={message.fragment_index}/{message.fragment_count}")

    header = f"[{format_timestamp(message.time)} | {message.sender_id}"
    if flags:
        header += f" | {', '.join(flags)}"
    header += "]"
    return header


def format_page_message(message: NormalizedMessage) -> str:
    body_parts = [message.text] if message.text else []
    if message.mentions:
        body_parts.append(f"Mentions: {', '.join(message.mentions)}")
    return f"{build_header(message)}\n{join_sections(body_parts)}".strip()


def format_dense_message(message: NormalizedMessage) -> str:
    return join_sections([message.text, f"mentions: {', '.join(message.mentions)}" if message.mentions else ""])


def format_sparse_message(message: NormalizedMessage) -> str:
    parts = [message.text, f"sender: {message.sender_id}"]
    if message.mentions:
        parts.append(" ".join(message.mentions))
    if message.is_forward:
        parts.append("forwarded")
    if message.is_quote:
        parts.append("quoted")
    return join_sections(parts)


def estimate_page_size(message: NormalizedMessage) -> int:
    return len(format_page_message(message)) + 2


def should_flush_chunk(current_chunk: list[NormalizedMessage], next_message: NormalizedMessage, current_size: int) -> bool:
    if not current_chunk:
        return False

    previous_message = current_chunk[-1]
    if previous_message.thread_sn and next_message.thread_sn and previous_message.thread_sn != next_message.thread_sn:
        return True
    if next_message.time - previous_message.time > MAX_TIME_GAP_SECONDS:
        return True
    return current_size + estimate_page_size(next_message) > MAX_CHUNK_CHARS


def select_overlap_context(messages: list[NormalizedMessage]) -> list[NormalizedMessage]:
    context: list[NormalizedMessage] = []
    last_time: int | None = None
    for message in reversed(messages):
        if context and last_time is not None and last_time - message.time > MAX_TIME_GAP_SECONDS:
            break
        if len(context) >= OVERLAP_MESSAGE_COUNT:
            break
        context.append(message)
        last_time = message.time
    context.reverse()
    return context


def build_page_content(chat: Chat, context_messages: list[NormalizedMessage], chunk_messages: list[NormalizedMessage]) -> str:
    parts = [
        f"CHAT: {chat.name}",
        f"CHAT_TYPE: {chat.type}",
        f"CHAT_ID: {chat.id}",
    ]
    if context_messages:
        parts.append("CONTEXT:")
        parts.extend(format_page_message(message) for message in context_messages)
    parts.append("MESSAGES:")
    parts.extend(format_page_message(message) for message in chunk_messages)
    return "\n\n".join(parts).strip()


def build_dense_content(chat: Chat, context_messages: list[NormalizedMessage], chunk_messages: list[NormalizedMessage]) -> str:
    messages = [
        formatted
        for message in [*context_messages, *chunk_messages]
        if (formatted := format_dense_message(message))
    ]
    return "\n\n".join([f"chat {chat.name}", f"chat_type {chat.type}", *messages]).strip()


def build_sparse_content(chat: Chat, context_messages: list[NormalizedMessage], chunk_messages: list[NormalizedMessage]) -> str:
    messages = [
        formatted
        for message in [*context_messages, *chunk_messages]
        if (formatted := format_sparse_message(message))
    ]
    return "\n".join([chat.name, chat.type, chat.id, *messages]).strip()


def build_chunk_item(chat: Chat, context_messages: list[NormalizedMessage], chunk_messages: list[NormalizedMessage]) -> IndexAPIItem:
    message_ids = list(dict.fromkeys(message.id for message in chunk_messages))
    return IndexAPIItem(
        page_content=build_page_content(chat, context_messages, chunk_messages),
        dense_content=build_dense_content(chat, context_messages, chunk_messages),
        sparse_content=build_sparse_content(chat, context_messages, chunk_messages),
        message_ids=message_ids,
    )


def build_chunks(chat: Chat, overlap_messages: list[Message], new_messages: list[Message]) -> list[IndexAPIItem]:
    overlap = [
        segment
        for message in overlap_messages
        for segment in split_message_for_chunking(normalize_message(message, is_overlap=True))
        if is_message_searchable(segment)
    ]
    new = [
        segment
        for message in new_messages
        for segment in split_message_for_chunking(normalize_message(message, is_overlap=False))
        if is_message_searchable(segment)
    ]

    overlap.sort(key=lambda item: (item.time, item.id))
    new.sort(key=lambda item: (item.time, item.id))

    if not new:
        logger.info("Index build produced no searchable messages")
        return []

    chunks: list[IndexAPIItem] = []
    context = select_overlap_context(overlap)
    current_chunk: list[NormalizedMessage] = []
    current_size = sum(estimate_page_size(message) for message in context)

    for message in new:
        if should_flush_chunk(current_chunk, message, current_size):
            chunks.append(build_chunk_item(chat, context, current_chunk))
            context = select_overlap_context([*context, *current_chunk])
            current_chunk = []
            current_size = sum(estimate_page_size(item) for item in context)

        current_chunk.append(message)
        current_size += estimate_page_size(message)

    if current_chunk:
        chunks.append(build_chunk_item(chat, context, current_chunk))

    logger.info(
        "Built %s chunks from %s new messages",
        len(chunks),
        len(new),
    )
    return chunks
