from __future__ import annotations

from pathlib import Path

from knowledge_engine import (
    ClaimDraft,
    EvidenceDraft,
    KnowledgeEngine,
    TranscriptInput,
    TranscriptRegistry,
)


def _transcript(source_id: str) -> TranscriptInput:
    return TranscriptInput(
        domain="real estate",
        entity_name="Cap Rate Rules",
        transcript_text=(
            "A 6% cap rate suits a stable suburban asset in a balanced market.\n\n"
            "Financing assumptions must hold for the exit to work over one cycle."
        ),
        source_kind="external_doc",
        source_id=source_id,
        claim_drafts=[
            ClaimDraft(
                statement="A 6% cap rate fits a stable suburban asset.",
                slot_name="cap_rate",
                observed_slots=["cap_rate"],
                evidence=[EvidenceDraft(source_kind="external_doc", source_id="doc-1", credibility=0.9)],
            )
        ],
    )


def test_engine_harbours_documents_and_housekeeps_on_ingest(tmp_path: Path) -> None:
    registry = TranscriptRegistry(tmp_path)
    engine = KnowledgeEngine(registry=registry)

    outcome = engine.ingest_transcript(_transcript("doc-1"))

    assert outcome.transcript_id is not None
    assert outcome.transcript_created is True
    assert outcome.chunk_count and outcome.chunk_count >= 1

    record = registry.get(outcome.transcript_id)
    assert record is not None
    assert Path(record.raw_path).exists()
    assert Path(record.document_path).exists()

    # Re-ingesting identical content is housekept: no duplicate harbour.
    again = engine.ingest_transcript(_transcript("doc-2"))
    assert again.transcript_created is False
    assert again.transcript_id == outcome.transcript_id
    assert len(registry.all()) == 1


def test_engine_without_registry_still_processes_claims(tmp_path: Path) -> None:
    engine = KnowledgeEngine()
    outcome = engine.ingest_transcript(_transcript("doc-1"))
    # No registry wired -> no harbour metadata, but claim processing is unaffected.
    assert outcome.transcript_id is None
    assert outcome.confirmed_claim_ids
