"""Tests for claim extractor evidence integration.

NO FALLBACKS: Tests verify that failures raise exceptions, not silent fallbacks.
"""
import pytest
from knowledge_engine.claim_extractor_evidence import (
    enrich_claims_with_evidence,
    EvidenceExtractionError,
    _parse_evidence_response,
    _validate_evidence_for_claim
)
from knowledge_engine.contracts import ClaimDraft, EvidenceDraft
from knowledge_engine.transcript_evidence import TranscriptEvidenceDraft


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str):
        self.response = response
        self.call_count = 0

    def complete_sync(self, *, system: str, user: str, max_tokens: int = 4000) -> str:
        self.call_count += 1
        return self.response


class FailingLLMClient:
    """Mock LLM client that always fails."""

    def complete_sync(self, *, system: str, user: str, max_tokens: int = 4000) -> str:
        raise RuntimeError("LLM service unavailable")


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


def test_enrich_claims_with_valid_evidence():
    """Successfully enrich claims with evidence from LLM."""
    claims = [
        ClaimDraft(
            statement="turmeric reduces inflammation",
            slot_name="treatment_outcome",
            observed_slots=["treatment_outcome"],
            evidence=[],
            notes="extracted by LLM"
        )
    ]
    transcript = "TCM studies show turmeric reduces inflammation by 40% in joint pain patients."

    mock_response = '''[
        {
            "statement": "40% reduction in inflammation observed",
            "source_reference": "paragraph 2, sentence 3",
            "source_quality": "academic",
            "conditions": ["joint pain patients", "12-week period"],
            "measurement_method": "randomized controlled trial",
            "confidence_indicator": "medium",
            "has_quantification": true,
            "has_time_period": true,
            "has_sample_size": false,
            "has_primary_source": true
        }
    ]'''

    llm_client = MockLLMClient(mock_response)
    enriched = enrich_claims_with_evidence(claims, transcript, llm_client)

    assert len(enriched) == 1
    assert len(enriched[0].evidence) == 1
    assert enriched[0].evidence[0].credibility > 0
    assert llm_client.call_count == 1


def test_enrich_claims_with_empty_evidence():
    """Claims with no supporting evidence get empty evidence list."""
    claims = [
        ClaimDraft(
            statement="unsupported claim",
            slot_name="test",
            observed_slots=["test"],
            evidence=[],
            notes="test"
        )
    ]
    transcript = "Transcript with no relevant information."

    mock_response = "[]"
    llm_client = MockLLMClient(mock_response)

    enriched = enrich_claims_with_evidence(claims, transcript, llm_client)

    assert len(enriched) == 1
    assert len(enriched[0].evidence) == 0


def test_enrich_claims_llm_failure():
    """Must raise error when LLM call fails - no silent fallback."""
    claims = [
        ClaimDraft(
            statement="test claim",
            slot_name="test",
            observed_slots=["test"],
            evidence=[],
            notes="test"
        )
    ]
    transcript = "Test transcript."

    llm_client = FailingLLMClient()

    with pytest.raises(EvidenceExtractionError) as exc_info:
        enrich_claims_with_evidence(claims, transcript, llm_client)

    assert "LLM call failed" in str(exc_info.value)


def test_parse_evidence_response_valid():
    """Parse valid JSON response into evidence drafts."""
    response = '''[
        {
            "statement": "test evidence",
            "source_quality": "academic",
            "conditions": ["condition1"],
            "measurement_method": "RCT",
            "confidence_indicator": "high",
            "has_quantification": true,
            "has_time_period": false,
            "has_sample_size": false,
            "has_primary_source": true
        }
    ]'''

    drafts = _parse_evidence_response(response)

    assert len(drafts) == 1
    assert drafts[0].statement == "test evidence"
    assert drafts[0].source_quality == "academic"


def test_parse_evidence_response_invalid_json():
    """Must raise error for invalid JSON - no silent fallback."""
    response = "not valid json"

    with pytest.raises(EvidenceExtractionError) as exc_info:
        _parse_evidence_response(response)

    assert "Failed to parse" in str(exc_info.value)


def test_parse_evidence_response_missing_statement():
    """Must raise error for missing statement field - no silent fallback."""
    response = '[{"source_quality": "academic"}]'

    with pytest.raises(EvidenceExtractionError) as exc_info:
        _parse_evidence_response(response)

    assert "missing required 'statement' field" in str(exc_info.value)


def test_parse_evidence_response_invalid_source_quality():
    """Must raise error for invalid source_quality - no silent fallback."""
    response = '[{"statement": "test", "source_quality": "invalid"}]'

    with pytest.raises(EvidenceExtractionError) as exc_info:
        _parse_evidence_response(response)

    assert "Invalid source_quality" in str(exc_info.value)


def test_parse_evidence_response_invalid_confidence():
    """Must raise error for invalid confidence_indicator - no silent fallback."""
    response = '[{"statement": "test", "source_quality": "academic", "confidence_indicator": "invalid"}]'

    with pytest.raises(EvidenceExtractionError) as exc_info:
        _parse_evidence_response(response)

    assert "Invalid confidence_indicator" in str(exc_info.value)


def test_validate_evidence_for_claim_valid():
    """Valid evidence passes validation."""
    evidence = [
        TranscriptEvidenceDraft(
            statement="test evidence",
            source_reference="paragraph 1",
            source_quality="academic",
            source_quality_score=0.9,
            confidence_indicator="high",
            confidence_score=0.85
        )
    ]
    claim = ClaimDraft(
        statement="test claim",
        slot_name="test",
        observed_slots=["test"]
    )

    # Should not raise
    _validate_evidence_for_claim(evidence, claim)


def test_validate_evidence_for_claim_missing_source():
    """Must raise error for missing source_reference - no silent fallback."""
    evidence = [
        TranscriptEvidenceDraft(
            statement="test evidence",
            source_reference="",  # Empty source reference
            source_quality="academic",
            source_quality_score=0.9,
            confidence_indicator="high",
            confidence_score=0.85
        )
    ]
    claim = ClaimDraft(
        statement="test claim",
        slot_name="test",
        observed_slots=["test"]
    )

    with pytest.raises(EvidenceExtractionError) as exc_info:
        _validate_evidence_for_claim(evidence, claim)

    assert "missing source_reference" in str(exc_info.value)


def test_evidence_extraction_error_is_raised():
    """EvidenceExtractionError is a proper exception class."""
    assert issubclass(EvidenceExtractionError, Exception)
    assert EvidenceExtractionError.__doc__ is not None
    assert "NO FALLBACKS" in EvidenceExtractionError.__doc__
