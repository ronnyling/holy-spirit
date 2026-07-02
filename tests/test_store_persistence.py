"""Tests for in-memory store JSON persistence (CLI/UAT path)."""

from __future__ import annotations

from knowledge_engine import (
    ClaimDraft,
    EvidenceDraft,
    KnowledgeEngine,
    TranscriptInput,
)
from knowledge_engine.models import EpistemicStatus, SlotLifecycle
from knowledge_engine.store import KnowledgeStore


def _ingest_sample(engine: KnowledgeEngine) -> str:
    outcome = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name="Cap Rate Rules",
            transcript_text="A 6% cap rate suits a stable suburban asset.",
            source_kind="external_doc",
            source_id="doc-1",
            claim_drafts=[
                ClaimDraft(
                    statement="A 6% cap rate suits a stable suburban asset.",
                    slot_name="cap_rate",
                    observed_slots=["cap_rate"],
                    evidence=[
                        EvidenceDraft(
                            source_kind="external_doc",
                            source_id="doc-1",
                            credibility=0.9,
                        )
                    ],
                )
            ],
        )
    )
    return outcome.entity_id


def test_save_load_round_trip(tmp_path):
    store = KnowledgeStore()
    engine = KnowledgeEngine(store=store)
    entity_id = _ingest_sample(engine)

    state = tmp_path / "state.json"
    store.save(state)

    reloaded = KnowledgeStore.load(state)
    assert reloaded.snapshot() == store.snapshot()
    assert entity_id in reloaded.entities
    assert len(reloaded.claims) == 1
    assert len(reloaded.evidence) == 1
    # Slot index is rebuilt so slot lookups still work after reload.
    slot = reloaded.get_slot(entity_id, "cap_rate")
    assert slot is not None
    assert slot.name == "cap_rate"


def test_load_missing_file_returns_empty(tmp_path):
    store = KnowledgeStore.load(tmp_path / "does_not_exist.json")
    assert store.snapshot()["claims"] == 0


def test_reloaded_store_supports_more_ingest_and_confirm(tmp_path):
    # Persisted state must remain a fully functional store, not a dead snapshot.
    store = KnowledgeStore()
    engine = KnowledgeEngine(store=store)
    entity_id = _ingest_sample(engine)
    state = tmp_path / "state.json"
    store.save(state)

    reloaded = KnowledgeStore.load(state)
    engine2 = KnowledgeEngine(store=reloaded)
    # Observe the same slot twice more, then it can be confirmed to CANDIDATE.
    for _ in range(2):
        reloaded.observe_slot(entity_id, "cap_rate")
    slot = reloaded.confirm_slot(
        entity_id, "cap_rate", target=SlotLifecycle.CANDIDATE, confirmed_by="tester"
    )
    assert slot.lifecycle == SlotLifecycle.CANDIDATE
    assert engine2.state_snapshot()["entities"] == 1
