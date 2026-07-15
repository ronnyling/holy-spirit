import pytest
from knowledge_engine.evidence_draft import EvidenceDraft

def test_evidence_draft_creation():
    draft = EvidenceDraft(
        statement="40% reduction in inflammation",
        source_reference="paragraph 3, sentence 2",
        source_quality="academic",
        source_quality_score=0.85,
        conditions=["joint pain patients", "12-week period"],
        measurement_method="randomized controlled trial",
        methodology_score=0.9,
        confidence_indicator="medium",
        confidence_score=0.6,
        has_quantification=True,
        has_time_period=True,
        has_sample_size=False,
        has_primary_source=True,
        contradicts=[],
        supports=[],
        document_id="doc_123",
        transcript_id="transcript_456",
        extraction_method="llm_hybrid"
    )
    assert draft.statement == "40% reduction in inflammation"
    assert draft.source_quality == "academic"
    assert draft.confidence_score == 0.6

def test_evidence_draft_from_llm_output():
    llm_output = {
        "statement": "turmeric helps joint pain",
        "source_quality": "anecdotal",
        "conditions": ["joint pain"],
        "measurement_method": "observational",
        "confidence_indicator": "low"
    }
    draft = EvidenceDraft.from_llm_output(
        llm_output,
        document_id="doc_123",
        transcript_id="transcript_456"
    )
    assert draft.statement == "turmeric helps joint pain"
    assert draft.source_quality == "anecdotal"
    assert draft.extraction_method == "llm_hybrid"

def test_evidence_draft_confidence_calculation():
    draft = EvidenceDraft(
        statement="test",
        source_reference="ref",
        source_quality="academic",
        source_quality_score=0.9,
        conditions=["condition1"],
        measurement_method="RCT",
        methodology_score=0.95,
        confidence_indicator="high",
        confidence_score=0.85,
        has_quantification=True,
        has_time_period=True,
        has_sample_size=True,
        has_primary_source=True,
        contradicts=[],
        supports=[],
        document_id="doc_1",
        transcript_id="transcript_1",
        extraction_method="llm_hybrid"
    )
    # LLM score: 0.9*0.4 + 0.95*0.3 + 0.85*0.3 = 0.36 + 0.285 + 0.255 = 0.9
    # Code score: 0.2 + 0.2 + 0.1 + 0.1 + 0.1 + 0.1 = 0.8
    # Final: 0.9*0.7 + 0.8*0.3 = 0.63 + 0.24 = 0.87
    assert draft.calculate_confidence() == pytest.approx(0.87, abs=0.01)
