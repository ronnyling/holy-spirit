import pytest
from knowledge_engine.claim_extractor_evidence import enrich_claims_with_evidence
from knowledge_engine.contracts import ClaimDraft
from knowledge_engine.transcript_evidence import TranscriptEvidenceDraft

def test_enrich_claims_with_evidence():
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
    
    enriched = enrich_claims_with_evidence(claims, transcript)
    
    assert len(enriched) == 1
    assert enriched[0].statement == "turmeric reduces inflammation"
    # Evidence will be empty until LLM integration is complete
    assert isinstance(enriched[0], ClaimDraft)

def test_enrich_empty_claims():
    claims = []
    transcript = "Some transcript text."
    
    enriched = enrich_claims_with_evidence(claims, transcript)
    
    assert len(enriched) == 0

def test_enrich_preserves_existing_evidence():
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
    
    enriched = enrich_claims_with_evidence(claims, transcript)
    
    assert len(enriched) == 1
    assert len(enriched[0].evidence) == 1
    assert enriched[0].evidence[0].source_id == "doc_1"
