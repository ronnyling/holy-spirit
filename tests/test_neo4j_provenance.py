"""Tests for Neo4j provenance tracking.

NO FALLBACKS: Tests verify that failures raise exceptions, not silent fallbacks.

These tests require a running Neo4j instance. Run with:
    KE_NEO4J_URI=bolt://localhost:7687 pytest tests/test_neo4j_provenance.py -v
"""
import os
import pytest
from knowledge_engine.graph.neo4j_store import KnowledgeGraphStore
from knowledge_engine.models import Claim, Evidence, Entity, EpistemicStatus

# Skip all tests if Neo4j is not available
NEO4J_URI = os.environ.get("KE_NEO4J_URI")
pytestmark = pytest.mark.skipif(
    not NEO4J_URI,
    reason="KE_NEO4J_URI not set - skipping Neo4j integration tests"
)


@pytest.fixture
def neo4j_store():
    """Create a Neo4j store for testing."""
    store = KnowledgeGraphStore(NEO4J_URI)
    yield store
    # Cleanup test data
    store._run("MATCH (n) WHERE n.id STARTS WITH 'test_' DELETE n")


def test_get_provenance_for_existing_claim(neo4j_store):
    """Provenance chain returned for existing claim."""
    # Create test entity
    entity = Entity(
        id="test_entity_001",
        canonical_name="Test Entity",
        aliases=[],
        description="Test entity for provenance"
    )
    neo4j_store.upsert_entity(entity.canonical_name)
    entity_id = neo4j_store._run(
        "MATCH (e:Entity {canonical_name: $name}) RETURN e.id",
        name=entity.canonical_name
    )[0]["e"]

    # Create test claim
    claim = Claim(
        id="test_claim_001",
        entity_id=entity_id,
        statement="Test claim statement",
        epistemic_status=EpistemicStatus.UNVERIFIED
    )
    neo4j_store.add_claim(claim)

    # Get provenance
    provenance = neo4j_store.get_provenance("test_claim_001")

    assert provenance["claim_id"] == "test_claim_001"
    assert "evidence_ids" in provenance
    assert "document_ids" in provenance
    assert "transcript_ids" in provenance
    assert "claim_metadata" in provenance


def test_get_provenance_for_nonexistent_claim(neo4j_store):
    """Must raise ValueError for nonexistent claim - no silent fallback."""
    with pytest.raises(ValueError) as exc_info:
        neo4j_store.get_provenance("nonexistent_claim_id")

    assert "not found in Neo4j" in str(exc_info.value)
    assert "No silent fallback" in str(exc_info.value)


def test_get_provenance_with_evidence(neo4j_store):
    """Provenance chain includes evidence when present."""
    # Create test entity
    entity = Entity(
        id="test_entity_002",
        canonical_name="Test Entity 2",
        aliases=[],
        description="Test entity for provenance with evidence"
    )
    neo4j_store.upsert_entity(entity.canonical_name)
    entity_id = neo4j_store._run(
        "MATCH (e:Entity {canonical_name: $name}) RETURN e.id",
        name=entity.canonical_name
    )[0]["e"]

    # Create test claim
    claim = Claim(
        id="test_claim_002",
        entity_id=entity_id,
        statement="Test claim with evidence",
        epistemic_status=EpistemicStatus.UNVERIFIED
    )
    neo4j_store.add_claim(claim)

    # Create test evidence
    evidence = Evidence(
        id="test_evidence_001",
        claim_id="test_claim_002",
        source_kind="external_doc",
        source_id="doc_001",
        source_ref="Test reference",
        credibility=0.8,
        linked_claim_ids=["test_claim_002"]
    )
    neo4j_store.add_evidence(evidence)

    # Get provenance
    provenance = neo4j_store.get_provenance("test_claim_002")

    assert provenance["claim_id"] == "test_claim_002"
    assert "test_evidence_001" in provenance["evidence_ids"]
    assert "doc_001" in provenance["document_ids"]
