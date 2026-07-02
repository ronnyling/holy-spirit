"""Integration tests for the Neo4j graph store.

Skipped unless a real Neo4j is reachable (``KE_NEO4J_URI``) and the ``neo4j``
driver is installed. This is test-infrastructure gating, NOT a runtime fallback:
in production the store always requires a live database.

To run locally:
    docker compose up -d
    set KE_NEO4J_URI=bolt://localhost:7687   (PowerShell: $env:KE_NEO4J_URI=...)
    set KE_NEO4J_PASSWORD=knowledge-engine
    python -m pytest tests/test_graph_neo4j.py
"""

from __future__ import annotations

import os
import random
import time
import uuid

import pytest

neo4j = pytest.importorskip("neo4j", reason="neo4j driver not installed")

from knowledge_engine.graph.neo4j_store import GraphCycleError, KnowledgeGraphStore
from knowledge_engine.models import Claim, EpistemicStatus, Entity

_URI = os.environ.get("KE_NEO4J_URI")
# The claim vector index is a single shared object on the (community-edition)
# `neo4j` database, so its dimension must agree with everything else that touches
# it — notably the CLI, which embeds with bge-m3 (1024). Hardcoding a different
# dimension here would collide with that index (and, before the dimension guard,
# silently corrupt it). Track the configured embedding dimension instead.
_DIM = int(os.environ.get("KE_EMBEDDING_DIMENSIONS", "1024"))

pytestmark = pytest.mark.skipif(
    not _URI, reason="KE_NEO4J_URI not set; Neo4j integration tests skipped"
)


@pytest.fixture()
def store() -> KnowledgeGraphStore:
    s = KnowledgeGraphStore(
        uri=_URI or "",
        user=os.environ.get("KE_NEO4J_USER", "neo4j"),
        password=os.environ.get("KE_NEO4J_PASSWORD", "knowledge-engine"),
        database=os.environ.get("KE_NEO4J_DATABASE", "neo4j"),
        embedding_dimensions=_DIM,
    )
    s.verify()
    s.apply_schema()
    yield s
    s.close()


def _entity(name: str) -> Entity:
    return Entity(id=f"ent-{uuid.uuid4().hex[:8]}", canonical_name=name)


def _claim(entity_id: str, statement: str) -> Claim:
    return Claim(
        id=f"clm-{uuid.uuid4().hex[:8]}",
        entity_id=entity_id,
        statement=statement,
        epistemic_status=EpistemicStatus.CONFIRMED,
        embedding=[0.1] * _DIM,
    )


def test_upsert_entity_and_claim_roundtrip(store: KnowledgeGraphStore) -> None:
    ent = _entity("Cap Rate Rules")
    store.upsert_entity(ent)
    clm = _claim(ent.id or "", "A 6% cap rate fits a stable suburban asset.")
    store.add_claim(clm)
    counts = store.counts()
    assert counts["Entity"] >= 1
    assert counts["Claim"] >= 1


def test_internal_support_cycle_is_rejected(store: KnowledgeGraphStore) -> None:
    ent = _entity("Cycle Guard")
    store.upsert_entity(ent)
    a = _claim(ent.id or "", "A")
    b = _claim(ent.id or "", "B")
    store.add_claim(a)
    store.add_claim(b)
    store.add_internal_support(source_claim_id=a.id or "", target_claim_id=b.id or "")
    with pytest.raises(GraphCycleError):
        store.add_internal_support(
            source_claim_id=b.id or "", target_claim_id=a.id or ""
        )


def test_vector_search_returns_similar_claim(store: KnowledgeGraphStore) -> None:
    ent = _entity("Vector Search")
    store.upsert_entity(ent)
    # A unique embedding so this claim is the strict nearest neighbour of its own
    # query vector, regardless of other claims accumulated in the local test DB.
    embedding = [random.random() for _ in range(_DIM)]
    clm = Claim(
        id=f"clm-{uuid.uuid4().hex[:8]}",
        entity_id=ent.id or "",
        statement="Financing assumptions must hold across one cycle.",
        epistemic_status=EpistemicStatus.CONFIRMED,
        embedding=embedding,
    )
    store.add_claim(clm)
    store.await_indexes()

    # Neo4j vector indexes are eventually consistent: a freshly committed node
    # becomes searchable shortly after commit, not within the write transaction.
    # Poll (bounded) until read-your-writes holds.
    top_id = None
    for _ in range(50):
        hits = store.find_similar_claims(embedding=embedding, k=1)
        if hits and hits[0].claim_id == clm.id:
            top_id = hits[0].claim_id
            break
        time.sleep(0.2)
    assert top_id == clm.id
