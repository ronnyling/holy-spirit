"""LLM-backed claim extraction (Build Order layer: automate the claims substrate).

BOUNDED vs UNBOUNDED split:
  * UNBOUNDED (LLM): read raw transcript text, propose atomic claims + the
    attribute ("slot") each claim is about.
  * BOUNDED (this module): strict JSON parsing + Pydantic validation. Anything
    malformed is dropped, never guessed. No evidence is ever fabricated, so
    every extracted claim enters the pipeline as UNVERIFIED and must earn
    promotion through the normal evidence gate. The human-in-the-loop gates are
    untouched.

Cold-start discipline: the prompt instructs the model to extract only what is
explicitly stated (no inference, no outside knowledge). Proposed slots merely
increment observation counts; slot promotion still requires human confirmation.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from .chunking import TranscriptChunker
from .contracts import ClaimDraft

_SYSTEM_PROMPT = (
    "You extract atomic factual claims from an expert transcript for a "
    "knowledge base. Rules:\n"
    "1. Extract ONLY claims explicitly stated in the transcript. Do not infer, "
    "summarize loosely, or add outside knowledge.\n"
    "2. Each claim must be a single, self-contained statement.\n"
    "3. For each claim, propose 0-3 short snake_case 'slots' naming the "
    "attribute the claim is about (e.g. cap_rate, entry_signal, herb_dosage).\n"
    "4. Do NOT invent evidence, sources, numbers, or confidence.\n"
    "5. If the transcript states nothing factual, return an empty array.\n"
    "Return STRICT JSON only: a JSON array of objects with keys "
    '"statement" (string) and "observed_slots" (array of strings). '
    "No prose, no markdown fences."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class SupportsComplete(Protocol):
    def complete_sync(self, *, system: str, user: str) -> str: ...


class ClaimExtractor:
    """Turns raw transcript text into validated ClaimDraft objects via the LLM.

    Extraction is chunk-aware so it SCALES to long transcripts: the text is split
    into bounded, sentence-aware windows (deterministic, BOUNDED layer) and each
    window is sent to the LLM separately. This keeps every model call's input and
    output small — the real reason a single whole-transcript call intermittently
    returned empty/truncated content on large inputs. Claims are then aggregated
    and de-duplicated across the overlapping windows by normalized statement text.
    """

    def __init__(self, client: SupportsComplete, *, chunker: TranscriptChunker | None = None) -> None:
        self._client = client
        self._chunker = chunker or TranscriptChunker()

    def extract(self, *, domain: str, entity_name: str, transcript_text: str) -> list[ClaimDraft]:
        """Extract claims across all chunks. Returns [] only for empty input.

        Per-chunk model output is parsed by the bounded layer; a validly-parsed
        empty array legitimately means "no claims in this window". Empty/truncated
        model responses raise upstream (see llm._content_or_raise) rather than
        silently yielding nothing — no fallbacks.
        """
        chunks = self._chunker.chunk(transcript_text)
        if not chunks:
            return []

        drafts: list[ClaimDraft] = []
        seen: set[str] = set()
        for chunk in chunks:
            user = (
                f"Domain: {domain}\n"
                f"Entity (topic): {entity_name}\n"
                "Transcript excerpt:\n"
                f"{chunk.text}\n"
            )
            raw = self._client.complete_sync(system=_SYSTEM_PROMPT, user=user)
            for draft in self._parse(raw):
                key = draft.statement.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                drafts.append(draft)
        return drafts

    @staticmethod
    def _parse(raw: str) -> list[ClaimDraft]:
        """Deterministically parse model output into ClaimDrafts. Bounded layer."""
        payload = _extract_json_array(raw)
        if payload is None:
            return []

        drafts: list[ClaimDraft] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            statement = item.get("statement")
            if not isinstance(statement, str) or not statement.strip():
                continue
            raw_slots = item.get("observed_slots", [])
            slots = [s.strip() for s in raw_slots if isinstance(s, str) and s.strip()] if isinstance(raw_slots, list) else []
            drafts.append(
                ClaimDraft(
                    statement=statement.strip(),
                    slot_name=slots[0] if slots else None,
                    observed_slots=slots,
                    evidence=[],  # never fabricated — stays UNVERIFIED
                    notes="extracted by LLM",
                )
            )
        return drafts


def _extract_json_array(raw: str) -> list | None:
    """Pull a JSON array out of model output, tolerating code fences/prose."""
    if not raw:
        return None
    candidate = raw.strip()

    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    # If there is surrounding prose, isolate the outermost [ ... ].
    if not candidate.startswith("["):
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None
