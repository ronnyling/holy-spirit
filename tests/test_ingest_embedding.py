"""Ingestion embedding wiring + content-dedup skip.

Verifies that new claims are embedded during ingest (batched, model warmed then
released) and that re-ingesting identical content under a DIFFERENT source_id/
file name short-circuits the whole pipeline (no re-extraction, no re-embedding).
"""

from __future__ import annotations

from pathlib import Path

from knowledge_engine import (
    ClaimDraft,
    EvidenceDraft,
    KnowledgeEngine,
    TranscriptInput,
    TranscriptRegistry,
)
from knowledge_engine.embeddings import EmbeddingClient


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
                evidence=[
                    EvidenceDraft(source_kind="external_doc", source_id="doc-1", credibility=0.9)
                ],
            )
        ],
    )


def _mock_embedder(monkeypatch, calls: list[dict]) -> EmbeddingClient:
    def fake_post(self, url, payload, headers=None):
        calls.append(payload)
        return {"embeddings": [[0.1] * 1024 for _ in payload["input"]]}

    monkeypatch.setattr(EmbeddingClient, "_post_sync", fake_post)
    return EmbeddingClient(
        provider="ollama",
        model="bge-m3",
        api_key="",
        base_url="http://localhost:11434",
        dimensions=1024,
    )


def test_ingest_embeds_new_claims_and_releases_model(monkeypatch) -> None:
    calls: list[dict] = []
    engine = KnowledgeEngine(embedding_client=_mock_embedder(monkeypatch, calls))

    outcome = engine.ingest_transcript(_transcript("doc-1"))

    claims = list(engine.store.claims.values())
    assert claims and all(c.embedding is not None and len(c.embedding) == 1024 for c in claims)
    assert any("embedded" in note for note in outcome.notes)

    # The embed happened inside warm() at the 1k context...
    assert calls[0]["keep_alive"] == "5m"
    assert calls[0]["options"]["num_ctx"] == 1024
    # ...and the session released the model on exit (keep_alive 0).
    assert calls[-1]["keep_alive"] == 0


def test_duplicate_content_skips_extraction_and_embedding(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    registry = TranscriptRegistry(tmp_path)
    engine = KnowledgeEngine(registry=registry, embedding_client=_mock_embedder(monkeypatch, calls))

    first = engine.ingest_transcript(_transcript("doc-1"))
    assert first.transcript_created is True
    calls_after_first = len(calls)
    assert calls_after_first > 0  # first ingest embedded

    # Same text, different source_id / file name -> recognised as duplicate.
    second = engine.ingest_transcript(_transcript("doc-2"))

    assert second.transcript_created is False
    assert second.transcript_id == first.transcript_id
    assert len(registry.all()) == 1
    assert len(calls) == calls_after_first  # NO new embed calls on the duplicate
    assert any("duplicate" in note.lower() for note in second.notes)


def test_ingest_without_embedder_leaves_claims_unembedded() -> None:
    engine = KnowledgeEngine()
    engine.ingest_transcript(_transcript("doc-1"))
    claims = list(engine.store.claims.values())
    assert claims and all(c.embedding is None for c in claims)
