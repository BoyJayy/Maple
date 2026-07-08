from chunking import (
    build_chunks,
    build_header,
    compress_technical_text,
    is_message_searchable,
    is_technical_message,
    normalize_message,
    normalize_text,
    select_overlap_context,
    should_flush_chunk,
    split_long_text,
    split_message_for_chunking,
)
from config import (
    MAX_CHUNK_CHARS,
    MAX_TIME_GAP_SECONDS,
    OVERLAP_MESSAGE_COUNT,
    SPLIT_MESSAGE_CHAR_THRESHOLD,
    SPLIT_SEGMENT_TARGET_CHARS,
    TECHNICAL_PREVIEW_LINES,
)
from schemas import Chat, Message


def make_message(message_id: str, *, time: int = 1_700_000_000, text: str = "обычное сообщение", **overrides) -> Message:
    payload = {
        "id": message_id,
        "time": time,
        "text": text,
        "sender_id": "user@example.com",
        "file_snippets": "",
        "is_system": False,
        "is_hidden": False,
        "is_forward": False,
        "is_quote": False,
    }
    payload.update(overrides)
    return Message(**payload)


def make_chat() -> Chat:
    return Chat(id="chat-1", name="Test Chat", sn="chat-1", type="group")


def normalized(message: Message):
    return normalize_message(message)


def test_normalize_text_strips_blank_lines_and_spaces():
    assert normalize_text("  раз  \n\n  два \n") == "раз\nдва"


def test_short_ack_is_not_searchable():
    assert not is_message_searchable(normalized(make_message("m1", text="ок")))


def test_short_ack_with_mentions_is_searchable():
    message = make_message("m1", text="ок", mentions=["ivan@example.com"])
    assert is_message_searchable(normalized(message))


def test_hidden_message_is_not_searchable():
    message = make_message("m1", is_hidden=True)
    assert not is_message_searchable(normalized(message))


def test_technical_message_detected_by_marker():
    assert is_technical_message("Traceback (most recent call last):\n  ...")


def test_compress_technical_text_keeps_head_and_tail():
    lines = [f"line {index}" for index in range(TECHNICAL_PREVIEW_LINES + 20)]
    compressed = compress_technical_text("\n".join(lines))
    assert "line 0" in compressed
    assert lines[-1] in compressed
    assert "lines omitted" in compressed


def test_split_long_text_respects_target():
    sentences = " ".join(f"Это предложение номер {index}." for index in range(80))
    parts = split_long_text(sentences, target_chars=200)
    assert len(parts) > 1
    assert all(len(part) <= 200 for part in parts)


def test_split_long_text_hard_slices_unbreakable_text():
    text = "х" * 900
    parts = split_long_text(text, target_chars=200)
    assert len(parts) == 5
    assert all(len(part) <= 200 for part in parts)


def test_split_message_marks_fragments():
    sentence = "Это достаточно длинное предложение для теста фрагментов. "
    text = (sentence * (SPLIT_MESSAGE_CHAR_THRESHOLD // len(sentence) + 2)).strip()
    assert len(text) > SPLIT_MESSAGE_CHAR_THRESHOLD
    fragments = split_message_for_chunking(normalized(make_message("m1", text=text)))
    assert len(fragments) > 1
    assert [fragment.fragment_index for fragment in fragments] == list(range(1, len(fragments) + 1))
    assert all(fragment.fragment_count == len(fragments) for fragment in fragments)
    assert all(len(fragment.text) <= SPLIT_SEGMENT_TARGET_CHARS for fragment in fragments)
    assert "part=1/" in build_header(fragments[0])


def test_should_flush_on_thread_switch():
    previous = normalized(make_message("m1", thread_sn="t1"))
    following = normalized(make_message("m2", thread_sn="t2"))
    assert should_flush_chunk([previous], following, current_size=10)


def test_should_flush_on_time_gap():
    previous = normalized(make_message("m1", time=1_700_000_000))
    following = normalized(make_message("m2", time=1_700_000_000 + MAX_TIME_GAP_SECONDS + 1))
    assert should_flush_chunk([previous], following, current_size=10)


def test_should_flush_on_size_overflow():
    previous = normalized(make_message("m1"))
    following = normalized(make_message("m2"))
    assert should_flush_chunk([previous], following, current_size=MAX_CHUNK_CHARS)


def test_select_overlap_context_limits_count():
    messages = [normalized(make_message(f"m{index}", time=1_700_000_000 + index)) for index in range(10)]
    context = select_overlap_context(messages)
    assert len(context) == OVERLAP_MESSAGE_COUNT
    assert [message.id for message in context] == ["m8", "m9"]


def test_build_chunks_produces_aligned_message_blocks():
    chat = make_chat()
    messages = [
        make_message(f"m{index}", time=1_700_000_000 + index * 60, text=f"Сообщение о релизе номер {index}")
        for index in range(5)
    ]
    chunks = build_chunks(chat, [], messages)
    assert chunks
    for chunk in chunks:
        assert chunk.message_ids
        block_ids = [block.message_id for block in chunk.message_blocks]
        assert set(block_ids) == set(chunk.message_ids)
        assert len(chunk.message_blocks) >= len(chunk.message_ids)
        for block in chunk.message_blocks:
            assert block.text.startswith("[")
    all_ids = [message_id for chunk in chunks for message_id in chunk.message_ids]
    assert all_ids == [f"m{index}" for index in range(5)]


def test_build_chunks_splits_on_time_gap():
    chat = make_chat()
    messages = [
        make_message("m1", time=1_700_000_000),
        make_message("m2", time=1_700_000_000 + MAX_TIME_GAP_SECONDS + 100),
    ]
    chunks = build_chunks(chat, [], messages)
    assert len(chunks) == 2
    assert chunks[0].message_ids == ["m1"]
    assert chunks[1].message_ids == ["m2"]
    # The second chunk carries the first message as overlap context.
    assert "CONTEXT:" in chunks[1].page_content


def test_build_chunks_ignores_noise_only_input():
    chat = make_chat()
    messages = [make_message("m1", text="ок"), make_message("m2", text="+")]
    assert build_chunks(chat, [], messages) == []
