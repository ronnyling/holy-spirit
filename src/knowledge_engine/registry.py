"""Transcript harbouring, documentation, and housekeeping.

The :class:`TranscriptRegistry` is the durable "front door" of the knowledge engine.
When a transcript arrives it is:

1. **Harboured** — the raw text is written verbatim under ``transcripts/``.
2. **Documented** — a markdown document (metadata + chunk map) is written under
   ``documents/``.
3. **Housekept** — a content SHA-256 makes ingestion idempotent (the same text is
   never stored twice) and a JSON manifest indexes every transcript for retrieval,
   auditing, and reproducibility.

This module is deterministic and offline. There are NO fallbacks: a corrupt manifest
or an unwritable directory raises immediately rather than silently degrading.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .chunking import Chunk, TranscriptChunker
from .documentation import render_document

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "untitled"


class TranscriptRecord(BaseModel):
    """Immutable index entry describing one harboured transcript."""

    model_config = ConfigDict(extra="forbid")

    transcript_id: str
    domain: str
    entity_name: str
    source_kind: str
    source_id: str
    source_ref: str | None = None
    sha256: str
    char_count: int
    chunk_count: int
    raw_path: str
    document_path: str
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Processing status: "harboured" = file written, "processing" = extraction running,
    # "complete" = all steps done. Only "complete" is treated as a true duplicate.
    processing_status: str = "harboured"


class HarbourResult(BaseModel):
    """Outcome of a harbour call — the record plus whether it was newly created."""

    model_config = ConfigDict(extra="forbid")

    record: TranscriptRecord
    created: bool
    chunks: list[Chunk] = Field(default_factory=list)
    # True if this content was already fully processed (safe to skip extraction)
    already_complete: bool = False


class TranscriptRegistry:
    """Filesystem-backed registry that harbours, documents, and housekeeps transcripts."""

    def __init__(self, root: Path | str, *, chunker: TranscriptChunker | None = None) -> None:
        self.root = Path(root)
        self.transcripts_dir = self.root / "transcripts"
        self.documents_dir = self.root / "documents"
        self.manifest_path = self.root / "manifest.json"
        self.chunker = chunker or TranscriptChunker()

        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        self._by_id: dict[str, TranscriptRecord] = {}
        self._by_hash: dict[str, str] = {}
        self._load_manifest()

    @staticmethod
    def content_hash(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    def find_by_hash(self, digest: str) -> TranscriptRecord | None:
        transcript_id = self._by_hash.get(digest)
        return self._by_id.get(transcript_id) if transcript_id else None

    def get(self, transcript_id: str) -> TranscriptRecord | None:
        return self._by_id.get(transcript_id)

    def all(self) -> list[TranscriptRecord]:
        return sorted(self._by_id.values(), key=lambda record: record.ingested_at)

    def harbour(
        self,
        *,
        text: str,
        domain: str,
        entity_name: str,
        source_kind: str,
        source_id: str,
        source_ref: str | None = None,
    ) -> HarbourResult:
        """Store, document, and index a transcript. Idempotent by content hash."""
        normalized = self.chunker.normalize(text)
        if not normalized:
            raise ValueError("cannot harbour an empty transcript")

        digest = self.content_hash(normalized)
        existing = self.find_by_hash(digest)
        if existing is not None:
            # Only skip if processing was fully completed. If interrupted mid-process,
            # allow re-processing to finish the remaining steps.
            if existing.processing_status == "complete":
                return HarbourResult(record=existing, created=False, chunks=self.chunker.chunk(normalized), already_complete=True)
            # Otherwise, reset for a fresh processing run.
            existing.processing_status = "harboured"
            self._save_manifest()
            return HarbourResult(record=existing, created=False, chunks=self.chunker.chunk(normalized), already_complete=False)

        chunks = self.chunker.chunk(normalized)
        transcript_id = f"{_slug(domain)}-{_slug(entity_name)}-{digest[:12]}"
        ingested_at = datetime.now(timezone.utc)

        raw_path = self.transcripts_dir / f"{transcript_id}.txt"
        document_path = self.documents_dir / f"{transcript_id}.md"

        raw_path.write_text(normalized, encoding="utf-8")
        document_path.write_text(
            render_document(
                transcript_id=transcript_id,
                domain=domain,
                entity_name=entity_name,
                source_kind=source_kind,
                source_id=source_id,
                source_ref=source_ref,
                sha256=digest,
                char_count=len(normalized),
                ingested_at=ingested_at.isoformat(),
                chunks=chunks,
            ),
            encoding="utf-8",
        )

        record = TranscriptRecord(
            transcript_id=transcript_id,
            domain=domain,
            entity_name=entity_name,
            source_kind=source_kind,
            source_id=source_id,
            source_ref=source_ref,
            sha256=digest,
            char_count=len(normalized),
            chunk_count=len(chunks),
            raw_path=str(raw_path),
            document_path=str(document_path),
            ingested_at=ingested_at,
        )

        self._by_id[transcript_id] = record
        self._by_hash[digest] = transcript_id
        self._save_manifest()

        return HarbourResult(record=record, created=True, chunks=chunks)

    def mark_complete(self, transcript_id: str) -> None:
        """Mark a transcript as fully processed (extraction + embedding done)."""
        record = self._by_id.get(transcript_id)
        if record is None:
            return
        record.processing_status = "complete"
        self._save_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for entry in raw.get("transcripts", []):
            record = TranscriptRecord.model_validate(entry)
            self._by_id[record.transcript_id] = record
            self._by_hash[record.sha256] = record.transcript_id

    def _save_manifest(self) -> None:
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "transcripts": [record.model_dump(mode="json") for record in self.all()],
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
