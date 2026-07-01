from knowledge_engine.models import (
    Claim,
    Entity,
    Evidence,
    EpistemicStatus,
    Provenance,
    ResolutionCase,
    Slot,
    SlotLifecycle,
)


def test_claim_defaults_to_unverified_status():
    claim = Claim(entity_id="entity-1", statement="TCM theory varies by lineage.")

    assert claim.epistemic_status == EpistemicStatus.UNVERIFIED
    assert claim.version == 1
    assert claim.provenance == []
    assert claim.slot_name is None


def test_provenance_accepts_user_source_kind():
    provenance = Provenance(source_kind="user", source_id="user-1", source_ref="session-7")

    assert provenance.source_kind == "user"
    assert provenance.source_ref == "session-7"


def test_slot_defaults_to_observed_lifecycle():
    slot = Slot(entity_id="entity-1", name="contraindications")

    assert slot.lifecycle == SlotLifecycle.OBSERVED
    assert slot.observed_count == 0
    assert slot.version == 1


def test_resolution_case_starts_open_and_versioned():
    case = ResolutionCase(conflict_signature="similarity-123")

    assert case.is_open is True
    assert case.version == 1
    assert case.conflicting_claim_ids == []


def test_evidence_requires_a_claim_and_credibility_score():
    evidence = Evidence(
        claim_id="claim-1",
        source_kind="external_doc",
        source_id="doc-9",
        credibility=0.85,
    )

    assert evidence.claim_id == "claim-1"
    assert evidence.credibility == 0.85
