from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_engine.registry import TranscriptRegistry

_SAMPLE = (
    "A 6% cap rate suits a stable suburban asset in a balanced market.\n\n"
    "Financing assumptions must hold for the exit to work.\n\n"
    "The strategy depends on holding through one full market cycle."
)


def test_harbour_writes_raw_document_and_manifest(tmp_path: Path) -> None:
    registry = TranscriptRegistry(tmp_path)
    result = registry.harbour(
        text=_SAMPLE,
        domain="real estate",
        entity_name="Cap Rate Rules",
        source_kind="external_doc",
        source_id="doc-1",
    )

    assert result.created is True
    record = result.record

    raw = Path(record.raw_path)
    document = Path(record.document_path)
    assert raw.exists() and raw.suffix == ".txt"
    assert document.exists() and document.suffix == ".md"

    doc_text = document.read_text(encoding="utf-8")
    assert "# Transcript — Cap Rate Rules" in doc_text
    assert record.sha256 in doc_text
    assert "## Chunks" in doc_text

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["transcripts"]) == 1
    assert manifest["transcripts"][0]["transcript_id"] == record.transcript_id


def test_identical_content_is_not_duplicated(tmp_path: Path) -> None:
    registry = TranscriptRegistry(tmp_path)
    first = registry.harbour(
        text=_SAMPLE,
        domain="real estate",
        entity_name="Cap Rate Rules",
        source_kind="external_doc",
        source_id="doc-1",
    )
    second = registry.harbour(
        text=_SAMPLE + "\n\n",  # normalizes to identical content
        domain="real estate",
        entity_name="Cap Rate Rules",
        source_kind="external_doc",
        source_id="doc-1-again",
    )

    assert first.created is True
    assert second.created is False
    assert second.record.transcript_id == first.record.transcript_id
    assert len(registry.all()) == 1


def test_manifest_is_reloaded_across_registry_instances(tmp_path: Path) -> None:
    first = TranscriptRegistry(tmp_path)
    result = first.harbour(
        text=_SAMPLE,
        domain="trading",
        entity_name="Breakout Setup",
        source_kind="external_doc",
        source_id="doc-1",
    )

    reopened = TranscriptRegistry(tmp_path)
    assert reopened.get(result.record.transcript_id) is not None
    assert reopened.find_by_hash(result.record.sha256) is not None
    # Idempotent even after reload.
    again = reopened.harbour(
        text=_SAMPLE,
        domain="trading",
        entity_name="Breakout Setup",
        source_kind="external_doc",
        source_id="doc-2",
    )
    assert again.created is False


def test_empty_transcript_raises_no_fallback(tmp_path: Path) -> None:
    registry = TranscriptRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.harbour(
            text="   \n\n  ",
            domain="tcm",
            entity_name="Qi",
            source_kind="user",
            source_id="s-1",
        )
