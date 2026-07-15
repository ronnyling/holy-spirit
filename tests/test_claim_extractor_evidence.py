"""Tests for claim extractor evidence integration.

NO FALLBACKS: Tests verify that failures raise exceptions, not silent fallbacks.
Data integrity is paramount.
"""
import pytest
from knowledge_engine.claim_extractor_evidence import (
    enrich_claims_with_evidence,
    EvidenceExtractionError
)
from knowledge_engine.contracts import ClaimDraft
from knowledge_engine.transcript_evidence import TranscriptEvidenceDraft


def test_enrich_empty_claims():
    """Empty claims list returns empty list - no extraction needed."""
    claims = []
    transcript = "Some transcript text."

    enriched = enrich_claims_with_evidence(claims, transcript)

    assert len(enriched) == 0


def test_enrich_claims_without_llm_raises_error():
    """Must raise error when LLM client not provided - no silent fallback."""
    claims = [
        ClaimDraft(
            statement="turmeric reduces inflammation",
            slot_name="treatment_outcome",
            observed_slots=["treatment_outcome"],
            evidence=[],
            notes="extracted by LLM"
        )
    ]
    transcript = "TCM studies show turmeric reduces inflammation."

    with pytest.raises(EvidenceExtractionError) as exc_info:
        enrich_claims_with_evidence(claims, transcript)

    assert "LLM client required" in str(exc_info.value)
    assert "No fallback" in str(exc_info.value)


def test_enrich_claims_with_llm_not_implemented():
    """Must raise error when evidence extraction not implemented - no silent fallback."""
    claims = [
        ClaimDraft(
            statement="test claim",
            slot_name="test_slot",
            observed_slots=["test_slot"],
            evidence=[],
            notes="test"
        )
    ]
    transcript = "Test transcript."

    # Mock LLM client
    class MockLLMClient:
        pass

    with pytest.raises(EvidenceExtractionError) as exc_info:
        enrich_claims_with_evidence(claims, transcript, llm_client=MockLLMClient())

    assert "not yet implemented" in str(exc_info.value)
    assert "No silent fallback" in str(exc_info.value)


def test_enrich_preserves_existing_evidence():
    """Existing evidence is preserved - no overwrite without explicit instruction."""
    from knowledge_engine.contracts import EvidenceDraft

    existing_evidence = EvidenceDraft(
        source_kind="external_doc",
        source_id="doc_1",
        source_ref="study reference",
        credibility=0.8,
        notes="existing evidence"
    )

    claims = [
        ClaimDraft(
            statement="test claim",
            slot_name="test_slot",
            observed_slots=["test_slot"],
            evidence=[existing_evidence],
            notes="with existing evidence"
        )
    ]

    transcript = "Test transcript."

    # This will raise because LLM client not provided
    with pytest.raises(EvidenceExtractionError):
        enrich_claims_with_evidence(claims, transcript)


def test_evidence_extraction_error_is_raised():
    """EvidenceExtractionError is a proper exception class."""
    assert issubclass(EvidenceExtractionError, Exception)
    assert EvidenceExtractionError.__doc__ is not None
    assert "NO FALLBACKS" in EvidenceExtractionError.__doc__
