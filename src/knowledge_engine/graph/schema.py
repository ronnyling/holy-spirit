"""Cypher schema for the knowledge graph.

Pure and deterministic: every function returns Cypher strings and never touches a
database. This keeps the graph *contract* unit-testable independently of any
running Neo4j instance.

Graph model (locked)
--------------------
Nodes:  Entity, Claim, Evidence, ResolutionCase, Slot, Transcript, Chunk
Edges:
    (:Claim)-[:ABOUT]->(:Entity)
    (:Evidence)-[:SUPPORTS]->(:Claim)
    (:Claim)-[:CONFLICTS_WITH]->(:Claim)
    (:ResolutionCase)-[:RESOLVES]->(:Claim)
    (:Claim)-[:DERIVED_FROM]->(:Chunk)
    (:Chunk)-[:PART_OF]->(:Transcript)
    (:Transcript)-[:ABOUT]->(:Entity)
    (:Entity)-[:ALIAS_OF]->(:Entity)
    (:Slot)-[:OF]->(:Entity)

The internal-evidence rule (a Confirmed claim may support another claim) is
represented as (:Claim)-[:SUPPORTS]->(:Claim); cycle detection is a native graph
traversal (see :func:`cycle_probe_cypher`).
"""

from __future__ import annotations

# --- Node labels --------------------------------------------------------------
NODE_ENTITY = "Entity"
NODE_CLAIM = "Claim"
NODE_EVIDENCE = "Evidence"
NODE_RESOLUTION_CASE = "ResolutionCase"
NODE_SLOT = "Slot"
NODE_TRANSCRIPT = "Transcript"
NODE_CHUNK = "Chunk"

# --- Relationship types -------------------------------------------------------
REL_ABOUT = "ABOUT"
REL_SUPPORTS = "SUPPORTS"
REL_CONFLICTS_WITH = "CONFLICTS_WITH"
REL_RESOLVES = "RESOLVES"
REL_DERIVED_FROM = "DERIVED_FROM"
REL_PART_OF = "PART_OF"
REL_ALIAS_OF = "ALIAS_OF"
REL_OF = "OF"

# --- Vector index -------------------------------------------------------------
CLAIM_VECTOR_INDEX = "claim_embedding_index"
_ALLOWED_SIMILARITY = ("cosine", "euclidean")


UNIQUENESS_CONSTRAINTS: tuple[str, ...] = (
    f"CREATE CONSTRAINT entity_id IF NOT EXISTS "
    f"FOR (n:{NODE_ENTITY}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT claim_id IF NOT EXISTS "
    f"FOR (n:{NODE_CLAIM}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT evidence_id IF NOT EXISTS "
    f"FOR (n:{NODE_EVIDENCE}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT resolution_case_id IF NOT EXISTS "
    f"FOR (n:{NODE_RESOLUTION_CASE}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT slot_id IF NOT EXISTS "
    f"FOR (n:{NODE_SLOT}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT transcript_id IF NOT EXISTS "
    f"FOR (n:{NODE_TRANSCRIPT}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT chunk_id IF NOT EXISTS "
    f"FOR (n:{NODE_CHUNK}) REQUIRE n.id IS UNIQUE",
)


def vector_index_cypher(
    *,
    dimensions: int,
    name: str = CLAIM_VECTOR_INDEX,
    label: str = NODE_CLAIM,
    property_name: str = "embedding",
    similarity: str = "cosine",
) -> str:
    """Return the DDL for Neo4j's native vector index on claim embeddings.

    Raises ``ValueError`` on invalid configuration rather than silently
    coercing — consistent with the no-fallback rule.
    """

    if dimensions <= 0:
        raise ValueError(f"vector dimensions must be positive, got {dimensions}")
    if similarity not in _ALLOWED_SIMILARITY:
        raise ValueError(
            f"similarity must be one of {_ALLOWED_SIMILARITY}, got {similarity!r}"
        )
    return (
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.{property_name}) "
        f"OPTIONS {{ indexConfig: {{ "
        f"`vector.dimensions`: {dimensions}, "
        f"`vector.similarity_function`: '{similarity}' "
        f"}} }}"
    )


def schema_statements(*, dimensions: int, similarity: str = "cosine") -> list[str]:
    """All DDL needed to initialise a fresh graph: constraints + vector index."""

    return [
        *UNIQUENESS_CONSTRAINTS,
        vector_index_cypher(dimensions=dimensions, similarity=similarity),
    ]


def cycle_probe_cypher() -> str:
    """Cypher that detects whether adding SUPPORTS(source -> target) would cycle.

    Returns a query with parameters ``$source_id`` and ``$target_id`` that yields
    a row iff ``target`` already reaches ``source`` through existing SUPPORTS
    edges (i.e. the new edge would close a cycle). Cycle detection is a native
    graph concern — kept in the database, not reimplemented in Python.
    """

    return (
        "MATCH (target:Claim {id: $target_id}), (source:Claim {id: $source_id}) "
        "RETURN EXISTS { (target)-[:SUPPORTS*1..]->(source) } AS creates_cycle"
    )
