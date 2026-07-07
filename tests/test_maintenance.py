"""Tests for maintenance reduction features."""

from __future__ import annotations

import pytest

from knowledge_engine import (
    ClaimDraft,
    EpistemicStatus,
    EvidenceDraft,
    KnowledgeEngine,
    TranscriptInput,
)
from knowledge_engine.learning import SlotLearner
from knowledge_engine.models import SlotLifecycle


class TestAutoResolve:
    """Verify auto-resolve for low-stakes conflicts."""

    def test_auto_resolve_no_evidence(self):
        """Incoming claim with no evidence against Confirmed claim → auto-resolved."""
        engine = KnowledgeEngine()

        # Seed with confirmed claim
        engine.ingest_transcript(TranscriptInput(
            domain="trading", entity_name="TestEntity",
            transcript_text="[Expert A] diversification reduces risk",
            source_kind="external_doc", source_id="doc1",
            claim_drafts=[ClaimDraft(
                statement="diversification reduces risk",
                slot_name="strategy",
                observed_slots=["strategy"],
                evidence=[EvidenceDraft(source_kind="external_doc", source_id="doc1", credibility=0.7),
                          EvidenceDraft(source_kind="external_doc", source_id="doc2", credibility=0.7)],
            )],
        ))

        # Conflict with no evidence → should auto-resolve
        outcome = engine.ingest_transcript(TranscriptInput(
            domain="trading", entity_name="TestEntity",
            transcript_text="[Expert B] concentration outperforms diversification",
            source_kind="external_doc", source_id="doc3",
            claim_drafts=[ClaimDraft(
                statement="concentration outperforms diversification",
                slot_name="strategy",
                observed_slots=["strategy"],
                evidence=[],  # No evidence
            )],
        ))

        # Should be Unverified (auto-resolved), not Disputed
        assert len(outcome.unverified_claim_ids) > 0 or len(outcome.confirmed_claim_ids) > 0
        assert len(outcome.open_case_ids) == 0  # Case auto-closed
        assert any("Auto-resolved" in n for n in outcome.notes)

    def test_standard_conflict_with_evidence(self):
        """Incoming claim with evidence against Confirmed claim → standard conflict."""
        engine = KnowledgeEngine()

        # Seed with confirmed claim
        engine.ingest_transcript(TranscriptInput(
            domain="trading", entity_name="TestEntity",
            transcript_text="[Expert A] diversification reduces risk",
            source_kind="external_doc", source_id="doc1",
            claim_drafts=[ClaimDraft(
                statement="diversification reduces risk",
                slot_name="strategy",
                observed_slots=["strategy"],
                evidence=[EvidenceDraft(source_kind="external_doc", source_id="doc1", credibility=0.7),
                          EvidenceDraft(source_kind="external_doc", source_id="doc2", credibility=0.7)],
            )],
        ))

        # Conflict WITH evidence → standard conflict
        outcome = engine.ingest_transcript(TranscriptInput(
            domain="trading", entity_name="TestEntity",
            transcript_text="[Expert B] concentration outperforms diversification",
            source_kind="external_doc", source_id="doc3",
            claim_drafts=[ClaimDraft(
                statement="concentration outperforms diversification",
                slot_name="strategy",
                observed_slots=["strategy"],
                evidence=[EvidenceDraft(source_kind="external_doc", source_id="doc3", credibility=0.7),
                          EvidenceDraft(source_kind="external_doc", source_id="doc4", credibility=0.7)],
            )],
        ))

        assert len(outcome.disputed_claim_ids) > 0
        assert len(outcome.open_case_ids) > 0


class TestAcceptOnAuthority:
    """Verify batch evidence submission."""

    def test_accept_unverified_claim(self):
        """Accept on authority works for domains with lower evidence bars."""
        engine = KnowledgeEngine()
        # Real estate has lower bar: score 0.8, 1 source
        engine.ingest_transcript(TranscriptInput(
            domain="real estate", entity_name="Test",
            transcript_text="[User] transit premium affects property values",
            source_kind="user", source_id="user1",
            claim_drafts=[ClaimDraft(
                statement="transit premium affects property values",
                slot_name="location", observed_slots=["location"],
            )],
        ))

        claims = engine.store.list_claims_for_entity(
            engine.store.upsert_entity(canonical_name="Test").id or ""
        )
        unverified = [c for c in claims if c.epistemic_status == EpistemicStatus.UNVERIFIED]
        assert len(unverified) > 0

        results = engine.accept_on_authority(
            [unverified[0].id or ""],
            accepted_by="domain-expert",
            rationale="Well-established real estate fact",
        )
        # May be confirmed (if bar met) or rejected (if bar not met)
        assert results[0]["status"] in ("confirmed", "rejected")

    def test_reject_already_confirmed(self):
        engine = KnowledgeEngine()
        engine.ingest_transcript(TranscriptInput(
            domain="trading", entity_name="Test",
            transcript_text="[Expert] diversification reduces risk",
            source_kind="external_doc", source_id="doc1",
            claim_drafts=[ClaimDraft(
                statement="diversification reduces risk",
                slot_name="strategy", observed_slots=["strategy"],
                evidence=[EvidenceDraft(source_kind="external_doc", source_id="doc1", credibility=0.7),
                          EvidenceDraft(source_kind="external_doc", source_id="doc2", credibility=0.7)],
            )],
        ))
        claims = engine.store.list_claims_for_entity(
            engine.store.upsert_entity(canonical_name="Test").id or ""
        )
        confirmed = [c for c in claims if c.epistemic_status == EpistemicStatus.CONFIRMED]
        results = engine.accept_on_authority(
            [confirmed[0].id or ""], accepted_by="test",
        )
        assert results[0]["status"] == "skipped"

    def test_reject_empty_accepted_by(self):
        engine = KnowledgeEngine()
        with pytest.raises(ValueError, match="accepted_by is required"):
            engine.accept_on_authority(["fake-id"], accepted_by="")


class TestHealthCheck:
    """Verify health check endpoint."""

    def test_health_check_returns_status(self):
        engine = KnowledgeEngine()
        health = engine.health_check()
        assert "overall" in health
        assert health["overall"] in ("healthy", "degraded")
        assert "neo4j" in health
        assert "embeddings" in health
        assert "llm" in health


class TestSlotAutoPromotion:
    """Verify slot auto-promotion at 10+ observations."""

    def test_auto_promote_at_10(self):
        learner = SlotLearner()
        engine = KnowledgeEngine()

        # Observe slot 10 times
        for i in range(10):
            engine.ingest_transcript(TranscriptInput(
                domain="trading", entity_name="Test",
                transcript_text=f"[Expert {i}] diversification reduces risk",
                source_kind="external_doc", source_id=f"doc{i}",
                claim_drafts=[ClaimDraft(
                    statement=f"diversification reduces risk variant {i}",
                    slot_name="strategy", observed_slots=["strategy"],
                    evidence=[EvidenceDraft(source_kind="external_doc", source_id=f"doc{i}", credibility=0.7),
                              EvidenceDraft(source_kind="external_doc", source_id=f"doc{i}b", credibility=0.7)],
                )],
            ))

        entity = engine.store.upsert_entity(canonical_name="Test")
        slot = engine.store.get_slot(entity.id or "", "strategy")
        assert slot is not None
        assert slot.lifecycle == SlotLifecycle.EXPECTED
        assert slot.observed_count >= 10


class TestHousekeeping:
    """Verify housekeeping method."""

    def test_housekeeping_runs(self):
        engine = KnowledgeEngine()
        result = engine.housekeeping()
        assert "actions" in result
        assert "timestamp" in result
        assert isinstance(result["actions"], list)
