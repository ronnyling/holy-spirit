from __future__ import annotations

import pytest

from knowledge_engine import (
    ClaimDraft,
    ConflictSummary,
    EvidenceDraft,
    GapKind,
    KnowledgeEngine,
    SlotLifecycle,
    TranscriptInput,
)


def build_transcript(
    *,
    domain: str,
    entity_name: str,
    source_kind: str,
    source_id: str,
    statement: str,
    slot_name: str,
    evidence: list[EvidenceDraft] | None = None,
    observed_slots: list[str] | None = None,
) -> TranscriptInput:
    return TranscriptInput(
        domain=domain,
        entity_name=entity_name,
        transcript_text=statement,
        source_kind=source_kind,  # type: ignore[arg-type]
        source_id=source_id,
        claim_drafts=[
            ClaimDraft(
                statement=statement,
                slot_name=slot_name,
                observed_slots=observed_slots or [slot_name],
                evidence=evidence or [],
            )
        ],
    )


def test_slot_learning_requires_confirmation_to_be_expected() -> None:
    engine = KnowledgeEngine()
    transcript = build_transcript(
        domain="real estate",
        entity_name="Real Estate Playbook",
        source_kind="user",
        source_id="session-1",
        statement="This playbook includes exit planning.",
        slot_name="exit_plan",
    )

    third_outcome = None
    for index in range(5):
        outcome = engine.ingest_transcript(transcript.model_copy(update={"source_id": f"session-{index + 1}"}))
        if index == 2:
            third_outcome = outcome

    assert third_outcome is not None
    assert third_outcome.slot_suggestions[0].suggested_lifecycle == SlotLifecycle.CANDIDATE.value

    slot = engine.store.get_slot(third_outcome.entity_id, "exit_plan")
    assert slot is not None
    assert slot.lifecycle == SlotLifecycle.OBSERVED

    confirmed_slot = engine.confirm_slot(
        "Real Estate Playbook",
        "exit_plan",
        confirmed_by="tester",
        target=SlotLifecycle.EXPECTED,
    )
    assert confirmed_slot["lifecycle"] == SlotLifecycle.EXPECTED.value
    assert engine.store.get_slot(third_outcome.entity_id, "exit_plan").lifecycle == SlotLifecycle.EXPECTED


def test_structural_gap_is_flagged_before_conflict() -> None:
    engine = KnowledgeEngine()
    for index in range(5):
        engine.ingest_transcript(
            build_transcript(
                domain="real estate",
                entity_name="Real Estate Playbook",
                source_kind="user",
                source_id=f"session-{index + 1}",
                statement="This playbook includes exit planning.",
                slot_name="exit_plan",
            )
        )

    engine.confirm_slot("Real Estate Playbook", "exit_plan", confirmed_by="tester", target=SlotLifecycle.EXPECTED)

    outcome = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name="Real Estate Playbook",
            transcript_text="This follow-up omits the exit plan detail.",
            source_kind="user",
            source_id="session-6",
            claim_drafts=[
                ClaimDraft(
                    statement="The strategy depends on market timing.",
                    slot_name="strategy",
                    observed_slots=["strategy"],
                )
            ],
        )
    )

    assert any(flag.kind == GapKind.STRUCTURAL and flag.slot_name == "exit_plan" for flag in outcome.gap_flags)
    assert outcome.conflict_summaries == []


def test_conflict_opens_case_and_reuses_precedent() -> None:
    engine = KnowledgeEngine()
    first = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name="Cap Rate Rules",
            transcript_text="A 6% cap rate fits a stable suburban asset.",
            source_kind="external_doc",
            source_id="doc-1",
            claim_drafts=[
                ClaimDraft(
                    statement="A 6% cap rate fits a stable suburban asset.",
                    slot_name="cap_rate",
                    observed_slots=["cap_rate"],
                    evidence=[
                        EvidenceDraft(source_kind="external_doc", source_id="doc-1", credibility=0.9),
                    ],
                )
            ],
        )
    )
    assert first.confirmed_claim_ids

    second = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name="Cap Rate Rules",
            transcript_text="A 7% cap rate fits a stable suburban asset.",
            source_kind="external_doc",
            source_id="doc-2",
            claim_drafts=[
                ClaimDraft(
                    statement="A 7% cap rate fits a stable suburban asset.",
                    slot_name="cap_rate",
                    observed_slots=["cap_rate"],
                    evidence=[
                        EvidenceDraft(source_kind="external_doc", source_id="doc-2", credibility=0.9),
                    ],
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
            entity_name="Cap Rate Rules",
            transcript_text="A 7.2% cap rate fits a stable suburban asset.",
            source_kind="external_doc",
            source_id="doc-3",
            claim_drafts=[
                ClaimDraft(
                    statement="A 7.2% cap rate fits a stable suburban asset.",
                    slot_name="cap_rate",
                    observed_slots=["cap_rate"],
                    evidence=[
                        EvidenceDraft(source_kind="external_doc", source_id="doc-3", credibility=0.9),
                    ],
                )
            ],
        )
    )
    assert third.conflict_summaries
    reopened = engine.store.get_resolution_case(third.open_case_ids[0])
    assert reopened.reopened_from_case_id == first_case_id


def test_internal_evidence_cycle_detection_blocks_support_loop() -> None:
    engine = KnowledgeEngine()

    claim_a = engine.ingest_transcript(
        TranscriptInput(
            domain="trading",
            entity_name="Trend Breakout Setup",
            transcript_text="The breakout setup is valid in a trending regime.",
            source_kind="external_doc",
            source_id="doc-a",
            claim_drafts=[
                ClaimDraft(
                    statement="The breakout setup is valid in a trending regime.",
                    slot_name="regime",
                    observed_slots=["regime"],
                    evidence=[
                        EvidenceDraft(source_kind="external_doc", source_id="doc-a", credibility=0.8),
                        EvidenceDraft(source_kind="user", source_id="session-a", credibility=0.4),
                    ],
                )
            ],
        )
    )
    assert claim_a.confirmed_claim_ids
    claim_a_id = claim_a.confirmed_claim_ids[0]

    claim_b = engine.ingest_transcript(
        TranscriptInput(
            domain="trading",
            entity_name="Trend Breakout Setup",
            transcript_text="The pullback setup is valid in a trending regime.",
            source_kind="user",
            source_id="session-b",
            claim_drafts=[
                ClaimDraft(
                    statement="The pullback setup is valid in a trending regime.",
                    slot_name="regime",
                    observed_slots=["regime"],
                )
            ],
        )
    )
    claim_b_id = claim_b.claim_ids[0]

    promoted_b = engine.promote_claim(
        claim_b_id,
        [
            EvidenceDraft(source_kind="internal_wiki", source_id=claim_a_id, credibility=1.0),
            EvidenceDraft(source_kind="user", source_id="session-b2", credibility=0.4),
        ],
    )
    assert promoted_b.epistemic_status.value == "Confirmed"

    with pytest.raises(ValueError, match="cycle"):
        engine.evidence_ledger.attach_internal_support(
            source_claim_id=claim_b_id,
            target_claim_id=claim_a_id,
            store=engine.store,
        )
