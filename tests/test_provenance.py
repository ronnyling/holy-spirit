import pytest
from knowledge_engine.provenance import ProvenanceChain

def test_provenance_chain_creation():
    chain = ProvenanceChain(
        claim_id="claim_123",
        evidence_ids=["ev_1", "ev_2"],
        document_ids=["doc_1"],
        transcript_ids=["transcript_1"],
        claim_metadata={"statement": "test claim"},
        evidence_metadata=[{"statement": "evidence 1"}],
        document_metadata=[{"filename": "doc.md"}],
        transcript_metadata=[{"text": "original text"}]
    )
    assert chain.claim_id == "claim_123"
    assert len(chain.evidence_ids) == 2

def test_provenance_chain_to_dict():
    chain = ProvenanceChain(
        claim_id="claim_123",
        evidence_ids=["ev_1"],
        document_ids=["doc_1"],
        transcript_ids=["transcript_1"],
        claim_metadata={},
        evidence_metadata=[],
        document_metadata=[],
        transcript_metadata=[]
    )
    d = chain.to_dict()
    assert d["claim_id"] == "claim_123"
    assert isinstance(d, dict)

def test_provenance_chain_from_store_data():
    # Mock store data
    store_data = {
        "claim": {"id": "claim_123", "statement": "test"},
        "evidence": [{"id": "ev_1", "statement": "evidence 1"}],
        "documents": [{"id": "doc_1", "filename": "doc.md"}],
        "transcripts": [{"id": "trans_1", "text": "original"}]
    }
    chain = ProvenanceChain.from_store_data("claim_123", store_data)
    assert chain.claim_id == "claim_123"
    assert len(chain.evidence_ids) == 1
