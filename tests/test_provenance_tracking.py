"""Tests for provenance tracking in KnowledgeStore.

NO FALLBACKS: Tests verify that failures raise exceptions, not silent fallbacks.
"""
import pytest
from knowledge_engine.store import KnowledgeStore
from knowledge_engine.provenance import ProvenanceChain
from knowledge_engine.models import Claim, Evidence, EpistemicStatus


def test_provenance_chain_creation():
    """ProvenanceChain can be created with required fields."""
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
    """ProvenanceChain can be serialized to dict."""
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
    """ProvenanceChain can be created from store data."""
    store_data = {
        "claim": {"id": "claim_123", "statement": "test"},
        "evidence": [{"id": "ev_1", "statement": "evidence 1"}],
        "documents": [{"id": "doc_1", "filename": "doc.md"}],
        "transcripts": [{"id": "trans_1", "text": "original"}]
    }
    chain = ProvenanceChain.from_store_data("claim_123", store_data)
    assert chain.claim_id == "claim_123"
    assert len(chain.evidence_ids) == 1


def test_store_get_provenance_for_nonexistent_claim():
    """Must raise KeyError for nonexistent claim - no silent fallback."""
    store = KnowledgeStore()

    with pytest.raises(KeyError):
        store.get_provenance("nonexistent_claim_id")


def test_store_get_provenance_for_claim_without_evidence():
    """Provenance chain returned with empty evidence for claim with no evidence."""
    store = KnowledgeStore()

    claim = Claim(
        id="claim_1",
        statement="test claim",
        entity_id="entity_1",
        epistemic_status=EpistemicStatus.UNVERIFIED
    )
    store.add_claim(claim)

    chain = store.get_provenance("claim_1")

    assert chain.claim_id == "claim_1"
    assert len(chain.evidence_ids) == 0
    assert len(chain.document_ids) == 0
    assert len(chain.transcript_ids) == 0
