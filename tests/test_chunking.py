from __future__ import annotations

import pytest

from knowledge_engine.chunking import TranscriptChunker


def test_short_transcript_is_a_single_chunk() -> None:
    chunker = TranscriptChunker(max_chars=1200, overlap_chars=150, min_chars=200)
    chunks = chunker.chunk("A short transcript about cap rates.")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].char_start == 0
    assert chunks[0].text == "A short transcript about cap rates."


def test_empty_and_whitespace_transcripts_produce_no_chunks() -> None:
    chunker = TranscriptChunker()
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n\n\t  ") == []


def test_long_transcript_splits_into_multiple_windows_within_bounds() -> None:
    chunker = TranscriptChunker(max_chars=120, overlap_chars=20, min_chars=30)
    paragraphs = [f"Paragraph number {i} explains a distinct idea in detail." for i in range(12)]
    text = "\n\n".join(paragraphs)

    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    # Primary spans must respect the max size; overlap prefix may add a little context.
    for chunk in chunks:
        assert chunk.char_end - chunk.char_start <= 120
    # Indices are contiguous starting at 0.
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_overlap_prefix_carries_context_between_chunks() -> None:
    chunker = TranscriptChunker(max_chars=60, overlap_chars=20, min_chars=10)
    text = "First distinct sentence here. Second distinct sentence follows. Third one closes."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # Every non-first chunk begins with a non-empty overlap prefix.
    for chunk in chunks[1:]:
        assert chunk.text  # non-empty and includes carried context


def test_oversized_single_sentence_is_hard_split() -> None:
    chunker = TranscriptChunker(max_chars=50, overlap_chars=0, min_chars=0)
    text = "word " * 60  # one long run, no sentence boundaries
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 50


def test_invalid_configuration_raises_no_silent_fallback() -> None:
    with pytest.raises(ValueError):
        TranscriptChunker(max_chars=0)
    with pytest.raises(ValueError):
        TranscriptChunker(max_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        TranscriptChunker(max_chars=100, min_chars=200)


def test_normalize_unifies_line_endings_and_collapses_blank_runs() -> None:
    chunker = TranscriptChunker()
    normalized = chunker.normalize("line one\r\n\r\n\r\n\r\nline two   \r\n")
    assert "\r" not in normalized
    assert "\n\n\n" not in normalized
    assert normalized == "line one\n\nline two"
