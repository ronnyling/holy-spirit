from __future__ import annotations

import pytest

from knowledge_engine import ClaimDraft, EvidenceDraft, GapKind, KnowledgeEngine, SlotLifecycle, TranscriptInput


def test_beta_tcm_slot_learning_and_gap_prompt() -> None:
    engine = KnowledgeEngine()

    for index in range(5):
        outcome = engine.ingest_transcript(
            TranscriptInput(
                domain="tcm",
                entity_name="TCM Tongue Assessment",
                transcript_text="This note records tongue coating observations.",
                source_kind="user",
                source_id=f"tcm-{index + 1}",
                claim_drafts=[
                    ClaimDraft(
                        statement="This note records tongue coating observations.",
                        slot_name="tongue_coating",
                        observed_slots=["tongue_coating"],
                    )
                ],
            )
        )

    assert outcome.slot_suggestions[0].suggested_lifecycle == SlotLifecycle.CANDIDATE.value

    engine.confirm_slot("TCM Tongue Assessment", "tongue_coating", confirmed_by="tester", target=SlotLifecycle.EXPECTED)

    missing = engine.ingest_transcript(
        TranscriptInput(
            domain="tcm",
            entity_name="TCM Tongue Assessment",
            transcript_text="The follow-up note omits the coating detail.",
            source_kind="user",
            source_id="tcm-6",
            claim_drafts=[
                ClaimDraft(
                    statement="The follow-up note adds pulse discussion.",
                    slot_name="pulse_quality",
                    observed_slots=["pulse_quality"],
                )
            ],
        )
    )

    assert any(flag.kind == GapKind.STRUCTURAL and flag.slot_name == "tongue_coating" for flag in missing.gap_flags)


def test_beta_real_estate_conflict_memory_reuses_precedent() -> None:
    engine = KnowledgeEngine()

    first = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name="Suburban Cap Rate Guidance",
            transcript_text="A 6% cap rate is acceptable for a stable suburban asset.",
            source_kind="external_doc",
            source_id="re-1",
            claim_drafts=[
                ClaimDraft(
                    statement="A 6% cap rate is acceptable for a stable suburban asset.",
                    slot_name="cap_rate",
                    observed_slots=["cap_rate"],
                    evidence=[EvidenceDraft(source_kind="external_doc", source_id="re-1", credibility=0.9)],
                )
            ],
        )
    )
    assert first.confirmed_claim_ids

    second = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name="Suburban Cap Rate Guidance",
            transcript_text="A 7% cap rate is acceptable for a stable suburban asset.",
            source_kind="external_doc",
            source_id="re-2",
            claim_drafts=[
                ClaimDraft(
                    statement="A 7% cap rate is acceptable for a stable suburban asset.",
                    slot_name="cap_rate",
                    observed_slots=["cap_rate"],
                    evidence=[EvidenceDraft(source_kind="external_doc", source_id="re-2", credibility=0.9)],
                )
            ],
        )
    )
    assert second.conflict_summaries
    first_case_id = second.conflict_summaries[0].case_id
    assert first_case_id is not None

    third = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name="Suburban Cap Rate Guidance",
            transcript_text="A 7.1% cap rate is acceptable for a stable suburban asset.",
            source_kind="external_doc",
            source_id="re-3",
            claim_drafts=[
                ClaimDraft(
                    statement="A 7.1% cap rate is acceptable for a stable suburban asset.",
                    slot_name="cap_rate",
                    observed_slots=["cap_rate"],
                    evidence=[EvidenceDraft(source_kind="external_doc", source_id="re-3", credibility=0.9)],
                )
            ],
        )
    )
    assert third.conflict_summaries
    reopened = engine.store.get_resolution_case(third.open_case_ids[0])
    assert reopened.reopened_from_case_id == first_case_id


def test_beta_trading_evidence_gating_and_cycle_prevention() -> None:
    engine = KnowledgeEngine()

    promoted = engine.ingest_transcript(
        TranscriptInput(
            domain="trading",
            entity_name="Breakout Regime Notes",
            transcript_text="Breakout setups are valid in a trending regime.",
            source_kind="external_doc",
            source_id="trade-1",
            claim_drafts=[
                ClaimDraft(
                    statement="Breakout setups are valid in a trending regime.",
                    slot_name="regime",
                    observed_slots=["regime"],
                    evidence=[
                        EvidenceDraft(source_kind="external_doc", source_id="trade-1", credibility=0.8),
                        EvidenceDraft(source_kind="user", source_id="trade-session-1", credibility=0.4),
                    ],
                )
            ],
        )
    )
    assert promoted.confirmed_claim_ids
    claim_a_id = promoted.confirmed_claim_ids[0]

    pending = engine.ingest_transcript(
        TranscriptInput(
            domain="trading",
            entity_name="Breakout Regime Notes",
            transcript_text="Pullback setups are also valid in the same regime.",
            source_kind="user",
            source_id="trade-session-2",
            claim_drafts=[
                ClaimDraft(
                    statement="Pullback setups are also valid in the same regime.",
                    slot_name="regime",
                    observed_slots=["regime"],
                )
            ],
        )
    )
    claim_b_id = pending.claim_ids[0]

    confirmed_b = engine.promote_claim(
        claim_b_id,
        [
            EvidenceDraft(source_kind="internal_wiki", source_id=claim_a_id, credibility=1.0),
            EvidenceDraft(source_kind="user", source_id="trade-session-2b", credibility=0.4),
        ],
    )
    assert confirmed_b.epistemic_status.value == "Confirmed"

    with pytest.raises(ValueError, match="cycle"):
        engine.evidence_ledger.attach_internal_support(
            source_claim_id=claim_b_id,
            target_claim_id=claim_a_id,
            store=engine.store,
        )
