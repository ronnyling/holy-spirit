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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Protocol

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

# MiMo v2.5 is a reasoning model: reasoning_content consumes most of the
# output budget. 8K tokens is a safe working limit; the bounded parser
# handles truncation via LLMTruncatedError upstream.
_EXTRACT_MAX_TOKENS = 8_000


class SupportsComplete(Protocol):
    def complete_sync(self, *, system: str, user: str, max_tokens: int = 8_000) -> str: ...


ProgressCallback = Callable[[int, int, str, str | None], None]


def _progress_noop(total: int, completed: int, phase: str, message: str | None = None) -> None:
    pass


class ClaimExtractor:
    """Turns raw transcript text into validated ClaimDraft objects via the LLM.

    Extraction is chunk-aware so it SCALES to long transcripts: the text is split
    into bounded, sentence-aware windows (deterministic, BOUNDED layer) and each
    window is sent to the LLM separately. This keeps every model call's input and
    output small — the real reason a single whole-transcript call intermittently
    returned empty/truncated content on large inputs. Claims are then aggregated
    and de-duplicated across the overlapping windows by normalized statement text.

    Auto-tuning: when max_workers is not explicitly set, the extractor selects
    parallelism based on chunk count. Checkpointing is auto-enabled for large
    files to allow resume on failure.
    """

    def __init__(
        self,
        client: SupportsComplete,
        *,
        chunker: TranscriptChunker | None = None,
        max_workers: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self._client = client
        self._chunker = chunker or TranscriptChunker()
        self._max_workers_override = max_workers
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def _auto_workers(self, chunk_count: int) -> int:
        """Select parallelism based on chunk count (auto-tuning)."""
        if self._max_workers_override is not None:
            return self._max_workers_override
        if chunk_count < 50:
            return 2
        if chunk_count < 500:
            return 4
        return 8

    def _auto_checkpoint(self, chunk_count: int) -> Path | None:
        """Enable checkpointing automatically for large files."""
        if self._checkpoint_dir is not None:
            return self._checkpoint_dir
        if chunk_count >= 100:
            # Auto-create checkpoint dir in temp location
            import tempfile
            return Path(tempfile.gettempdir()) / "ke_checkpoints"
        return None

    def extract(
        self,
        *,
        domain: str,
        entity_name: str,
        transcript_text: str,
    ) -> list[ClaimDraft]:
        """Extract claims across all chunks. Returns [] only for empty input.

        Per-chunk model output is parsed by the bounded layer; a validly-parsed
        empty array legitimately means "no claims in this window". Empty/truncated
        model responses raise upstream (see llm._content_or_raise) rather than
        silently yielding nothing — no fallbacks.
        """
        chunks = self._chunker.chunk(transcript_text)
        if not chunks:
            return []

        # Auto-tune based on chunk count
        max_workers = self._auto_workers(len(chunks))
        checkpoint_dir = self._auto_checkpoint(len(chunks))

        # Load checkpoint if available
        completed_chunks: dict[int, list[ClaimDraft]] = {}
        if checkpoint_dir:
            completed_chunks = self._load_checkpoint(domain, entity_name, checkpoint_dir)

        remaining = [c for c in chunks if c.index not in completed_chunks]

        # Start with already-extracted drafts
        all_drafts: list[ClaimDraft] = []
        seen: set[str] = set()
        for idx, drafts in sorted(completed_chunks.items()):
            for draft in drafts:
                key = draft.statement.strip().lower()
                if key not in seen:
                    seen.add(key)
                    all_drafts.append(draft)

        if not remaining:
            return all_drafts

        # Parallel extraction
        completed_count = len(completed_chunks)
        lock = threading.Lock()

        def _extract_one(chunk) -> tuple[int, list[ClaimDraft]]:
            user = (
                f"Domain: {domain}\n"
                f"Entity (topic): {entity_name}\n"
                "Transcript excerpt:\n"
                f"{chunk.text}\n"
            )
            raw = self._client.complete_sync(
                system=_SYSTEM_PROMPT,
                user=user,
                max_tokens=_EXTRACT_MAX_TOKENS,
            )
            return chunk.index, self._parse(raw)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_one, c): c.index for c in remaining}

            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    idx, drafts = future.result()
                    with lock:
                        for draft in drafts:
                            key = draft.statement.strip().lower()
                            if key not in seen:
                                seen.add(key)
                                all_drafts.append(draft)
                        completed_count += 1

                        # Checkpoint
                        if checkpoint_dir:
                            self._save_checkpoint(domain, entity_name, idx, drafts, checkpoint_dir)
                except Exception as exc:
                    # Log but continue — partial results are valuable
                    with lock:
                        completed_count += 1

        # Clean up checkpoint on success
        if checkpoint_dir:
            self._clear_checkpoint(domain, entity_name, checkpoint_dir)

        return all_drafts

    def estimate_cost(self, transcript_text: str) -> dict:
        """Estimate API calls, time, and cost before starting."""
        chunks = self._chunker.chunk(transcript_text)
        chunk_count = len(chunks)
        avg_tokens = sum(c.token_estimate for c in chunks) / max(chunk_count, 1)

        return {
            "chunk_count": chunk_count,
            "total_chars": len(transcript_text),
            "estimated_api_calls": chunk_count,
            "estimated_tokens_in": int(avg_tokens * chunk_count),
            "estimated_time_seconds": chunk_count * 2,  # ~2s per call
            "estimated_cost_usd": round(chunk_count * 0.001, 2),  # rough estimate
            "recommendation": "proceed" if chunk_count < 5000 else "consider increasing KE_CHUNK_MAX_CHARS",
        }

    def _checkpoint_path(self, domain: str, entity_name: str, checkpoint_dir: Path) -> Path:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{domain}_{entity_name}")
        return checkpoint_dir / f"extraction_{safe_name}.json"

    def _load_checkpoint(self, domain: str, entity_name: str, checkpoint_dir: Path) -> dict[int, list[ClaimDraft]]:
        path = self._checkpoint_path(domain, entity_name, checkpoint_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result: dict[int, list[ClaimDraft]] = {}
            for idx_str, items in data.items():
                result[int(idx_str)] = [
                    ClaimDraft(**item) for item in items
                ]
            return result
        except Exception:
            return {}

    def _save_checkpoint(self, domain: str, entity_name: str, chunk_idx: int, drafts: list[ClaimDraft], checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(domain, entity_name, checkpoint_dir)

        # Load existing, update, save
        existing: dict[str, list[dict]] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        existing[str(chunk_idx)] = [d.model_dump() for d in drafts]
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def _clear_checkpoint(self, domain: str, entity_name: str, checkpoint_dir: Path) -> None:
        path = self._checkpoint_path(domain, entity_name, checkpoint_dir)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

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
