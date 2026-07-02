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
    """Turns raw transcript text into validated ClaimDraft objects via the LLM."""

    def __init__(self, client: SupportsComplete) -> None:
        self._client = client

    def extract(self, *, domain: str, entity_name: str, transcript_text: str) -> list[ClaimDraft]:
        """Extract claims. Returns [] on empty text or unparseable model output."""
        text = transcript_text.strip()
        if not text:
            return []

        user = (
            f"Domain: {domain}\n"
            f"Entity (topic): {entity_name}\n"
            "Transcript:\n"
            f"{text}\n"
        )
        raw = self._client.complete_sync(system=_SYSTEM_PROMPT, user=user)
        return self._parse(raw)

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
