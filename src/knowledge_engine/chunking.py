"""Deterministic, structure-aware transcript chunking.

This module is intentionally free of any external dependency or network call.
Chunking is a *bounded* problem (per the locked architecture) and therefore lives
entirely in deterministic code — never in the LLM layer, and with NO fallbacks:
invalid configuration raises immediately.

Offsets (`char_start`, `char_end`) are reported against the *normalized* transcript
text returned by :meth:`TranscriptChunker.normalize`, not the raw input, because
normalization collapses whitespace and unifies line endings. Callers that need to
map back to the raw source should normalize first and keep that normalized copy.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_INLINE_WS = re.compile(r"[ \t]+")


class Chunk(BaseModel):
    """A single contiguous window of transcript text.

    ``text`` may include a short overlap prefix carried from the previous chunk to
    preserve context across boundaries. ``char_start``/``char_end`` describe only the
    *primary* (non-overlap) span of this chunk within the normalized transcript.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    text: str
    char_start: int
    char_end: int
    token_estimate: int


class TranscriptChunker:
    """Split a transcript into overlapping, sentence-aware windows.

    Args:
        max_chars: Hard upper bound on the primary content of a chunk.
        overlap_chars: Number of trailing characters from the previous chunk to
            prepend to the next chunk as context. Must be ``< max_chars``.
        min_chars: A trailing chunk shorter than this is merged into its predecessor
            so we never emit a stray sliver.
    """

    def __init__(self, *, max_chars: int = 1200, overlap_chars: int = 150, min_chars: int = 200) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be >= 0 and < max_chars")
        if not 0 <= min_chars <= max_chars:
            raise ValueError("min_chars must be between 0 and max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars

    def normalize(self, text: str) -> str:
        """Unify line endings, trim per-line trailing whitespace, collapse blank runs."""
        unified = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [_INLINE_WS.sub(" ", line).rstrip() for line in unified.split("\n")]
        collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        return collapsed.strip()

    def chunk(self, text: str) -> list[Chunk]:
        normalized = self.normalize(text)
        if not normalized:
            return []

        windows = self._pack_windows(self._segments(normalized))

        chunks: list[Chunk] = []
        for index, (body, start) in enumerate(windows):
            prefix = self._overlap_prefix(chunks)
            full = f"{prefix} {body}".strip() if prefix else body
            chunks.append(
                Chunk(
                    index=index,
                    text=full,
                    char_start=start,
                    char_end=start + len(body),
                    token_estimate=max(1, len(full) // 4),
                )
            )
        return chunks

    def _overlap_prefix(self, chunks: list[Chunk]) -> str:
        if not self.overlap_chars or not chunks:
            return ""
        tail = chunks[-1].text[-self.overlap_chars :]
        # Start the overlap on a word boundary so we do not split a token.
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1 :]
        return tail.strip()

    def _segments(self, text: str) -> list[tuple[str, int]]:
        """Break normalized text into paragraph/sentence pieces with offsets."""
        pieces: list[tuple[str, int]] = []
        cursor = 0
        for raw_para in _PARAGRAPH_SPLIT.split(text):
            para = raw_para.strip()
            if not para:
                continue
            offset = text.find(para, cursor)
            if offset == -1:
                offset = cursor
            cursor = offset + len(para)

            if len(para) <= self.max_chars:
                pieces.append((para, offset))
                continue

            for sentence, soffset in self._split_sentences(para, offset):
                if len(sentence) <= self.max_chars:
                    pieces.append((sentence, soffset))
                else:
                    pieces.extend(self._hard_split(sentence, soffset))
        return pieces

    def _split_sentences(self, paragraph: str, base_offset: int) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []
        cursor = 0
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            local = paragraph.find(sentence, cursor)
            if local == -1:
                local = cursor
            cursor = local + len(sentence)
            result.append((sentence, base_offset + local))
        return result

    def _hard_split(self, sentence: str, base_offset: int) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []
        for start in range(0, len(sentence), self.max_chars):
            piece = sentence[start : start + self.max_chars].strip()
            if piece:
                result.append((piece, base_offset + start))
        return result

    def _pack_windows(self, pieces: list[tuple[str, int]]) -> list[tuple[str, int]]:
        windows: list[tuple[str, int]] = []
        current = ""
        current_offset: int | None = None

        for piece, offset in pieces:
            if current and len(current) + 1 + len(piece) > self.max_chars:
                windows.append((current, current_offset or 0))
                current = ""
                current_offset = None
            if not current:
                current, current_offset = piece, offset
            else:
                current = f"{current} {piece}"

        if current:
            windows.append((current, current_offset or 0))

        # Fold a too-small trailing window back into its predecessor.
        if len(windows) >= 2 and len(windows[-1][0]) < self.min_chars:
            last_text, _ = windows.pop()
            prev_text, prev_offset = windows.pop()
            windows.append((f"{prev_text} {last_text}", prev_offset))

        return windows
