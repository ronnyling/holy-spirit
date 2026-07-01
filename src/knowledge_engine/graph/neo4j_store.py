"""Neo4j-backed knowledge graph store.

Real driver code against Neo4j 5 (native property graph + native vector index).
There is **no in-memory fallback**: construction opens a live connection and
methods run Cypher. Callers that cannot reach Neo4j get a hard error, by design.

Local verification note
------------------------
The pure Cypher DDL in :mod:`knowledge_engine.graph.schema` is unit-tested. The
behaviour of *this* module is exercised by integration tests that are skipped
unless ``KE_NEO4J_URI`` points at a reachable database (see
``tests/test_graph_neo4j.py`` and ``docker-compose.yml``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from neo4j import Driver, GraphDatabase

from ..models import Claim, Entity, Evidence, ResolutionCase, Slot
from . import schema


class GraphCycleError(RuntimeError):
    """Raised when an internal-support edge would create an evidence cycle."""


@dataclass(frozen=True)
class SimilarClaim:
    claim_id: str
    statement: str
    score: float


def _provenance_json(claim: Claim) -> str:
    return json.dumps([p.model_dump(mode="json") for p in claim.provenance])


class KnowledgeGraphStore:
    """Persistence + retrieval over the knowledge graph.

    Parameters mirror the ``KE_NEO4J_*`` environment variables. The vector index
    dimension must match the embedding provider's output dimension.
    """

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        embedding_dimensions: int,
    ) -> None:
        if not uri:
            raise ValueError("Neo4j URI is required (no fallback store exists)")
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        self._database = database
        self._embedding_dimensions = embedding_dimensions
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    # -- lifecycle -------------------------------------------------------------
    def close(self) -> None:
        self._driver.close()

    def verify(self) -> None:
        """Fail loudly if the database is unreachable."""

        self._driver.verify_connectivity()

    def apply_schema(self, *, similarity: str = "cosine", await_seconds: int = 300) -> None:
        statements = schema.schema_statements(
            dimensions=self._embedding_dimensions, similarity=similarity
        )
        with self._driver.session(database=self._database) as session:
            for statement in statements:
                session.run(statement).consume()
            # Constraints and the native vector index populate asynchronously.
            # Block until they are online so the first query never races them.
            session.run("CALL db.awaitIndexes($seconds)", seconds=await_seconds).consume()

    def await_indexes(self, *, seconds: int = 300) -> None:
        """Block until all indexes are online and have processed pending updates.

        Vector indexes are eventually consistent: a node written in one
        transaction becomes searchable shortly after commit. Callers that need a
        read-your-writes guarantee (e.g. tests) call this after writing.
        """

        self._run("CALL db.awaitIndexes($seconds)", seconds=seconds)

    def _run(self, cypher: str, /, **params):
        with self._driver.session(database=self._database) as session:
            return list(session.run(cypher, **params))

    # -- entities --------------------------------------------------------------
    def upsert_entity(self, entity: Entity) -> None:
        self._run(
            "MERGE (e:Entity {id: $id}) "
            "SET e.canonical_name = $canonical_name, "
            "    e.aliases = $aliases, "
            "    e.description = $description, "
            "    e.version = $version",
            id=entity.id,
            canonical_name=entity.canonical_name,
            aliases=entity.aliases,
            description=entity.description,
            version=entity.version,
        )

    def add_alias_edge(self, *, alias_entity_id: str, canonical_entity_id: str) -> None:
        self._run(
            "MATCH (a:Entity {id: $alias_id}), (c:Entity {id: $canonical_id}) "
            "MERGE (a)-[:ALIAS_OF]->(c)",
            alias_id=alias_entity_id,
            canonical_id=canonical_entity_id,
        )

    # -- claims ----------------------------------------------------------------
    def add_claim(self, claim: Claim) -> None:
        self._run(
            "MATCH (e:Entity {id: $entity_id}) "
            "MERGE (c:Claim {id: $id}) "
            "SET c.statement = $statement, "
            "    c.slot_name = $slot_name, "
            "    c.epistemic_status = $status, "
            "    c.provenance = $provenance, "
            "    c.embedding = $embedding, "
            "    c.version = $version, "
            "    c.tags = $tags "
            "MERGE (c)-[:ABOUT]->(e)",
            id=claim.id,
            entity_id=claim.entity_id,
            statement=claim.statement,
            slot_name=claim.slot_name,
            status=str(claim.epistemic_status),
            provenance=_provenance_json(claim),
            embedding=claim.embedding,
            version=claim.version,
            tags=claim.tags,
        )

    def set_claim_status(self, *, claim_id: str, status: str) -> None:
        self._run(
            "MATCH (c:Claim {id: $id}) SET c.epistemic_status = $status",
            id=claim_id,
            status=status,
        )

    def set_claim_embedding(self, *, claim_id: str, embedding: list[float]) -> None:
        if len(embedding) != self._embedding_dimensions:
            raise ValueError(
                f"embedding length {len(embedding)} != index dimension "
                f"{self._embedding_dimensions}"
            )
        self._run(
            "MATCH (c:Claim {id: $id}) SET c.embedding = $embedding",
            id=claim_id,
            embedding=embedding,
        )

    def add_conflict(self, *, claim_id_a: str, claim_id_b: str) -> None:
        self._run(
            "MATCH (a:Claim {id: $a}), (b:Claim {id: $b}) "
            "MERGE (a)-[:CONFLICTS_WITH]->(b)",
            a=claim_id_a,
            b=claim_id_b,
        )

    # -- evidence --------------------------------------------------------------
    def add_evidence(self, evidence: Evidence) -> None:
        self._run(
            "MATCH (c:Claim {id: $claim_id}) "
            "MERGE (ev:Evidence {id: $id}) "
            "SET ev.source_kind = $source_kind, "
            "    ev.source_id = $source_id, "
            "    ev.source_ref = $source_ref, "
            "    ev.credibility = $credibility, "
            "    ev.notes = $notes "
            "MERGE (ev)-[:SUPPORTS]->(c)",
            id=evidence.id,
            claim_id=evidence.claim_id,
            source_kind=evidence.source_kind,
            source_id=evidence.source_id,
            source_ref=evidence.source_ref,
            credibility=evidence.credibility,
            notes=evidence.notes,
        )

    def would_create_cycle(self, *, source_claim_id: str, target_claim_id: str) -> bool:
        """True iff SUPPORTS(source -> target) would close a cycle. Native probe."""

        rows = self._run(
            schema.cycle_probe_cypher(),
            source_id=source_claim_id,
            target_id=target_claim_id,
        )
        return bool(rows and rows[0]["creates_cycle"])

    def add_internal_support(
        self, *, source_claim_id: str, target_claim_id: str
    ) -> None:
        """Link a Confirmed claim as internal evidence for another claim.

        Guarded by native cycle detection: a claim can never justify itself,
        directly or transitively.
        """

        if source_claim_id == target_claim_id:
            raise GraphCycleError("a claim cannot support itself")
        if self.would_create_cycle(
            source_claim_id=source_claim_id, target_claim_id=target_claim_id
        ):
            raise GraphCycleError(
                f"internal support {source_claim_id} -> {target_claim_id} "
                "would create an evidence cycle"
            )
        self._run(
            "MATCH (s:Claim {id: $source}), (t:Claim {id: $target}) "
            "MERGE (s)-[:SUPPORTS]->(t)",
            source=source_claim_id,
            target=target_claim_id,
        )

    # -- slots -----------------------------------------------------------------
    def upsert_slot(self, slot: Slot) -> None:
        self._run(
            "MATCH (e:Entity {id: $entity_id}) "
            "MERGE (s:Slot {id: $id}) "
            "SET s.name = $name, "
            "    s.description = $description, "
            "    s.lifecycle = $lifecycle, "
            "    s.observed_count = $observed_count, "
            "    s.candidate_count = $candidate_count, "
            "    s.expected_count = $expected_count, "
            "    s.version = $version "
            "MERGE (s)-[:OF]->(e)",
            id=slot.id,
            entity_id=slot.entity_id,
            name=slot.name,
            description=slot.description,
            lifecycle=str(slot.lifecycle),
            observed_count=slot.observed_count,
            candidate_count=slot.candidate_count,
            expected_count=slot.expected_count,
            version=slot.version,
        )

    # -- resolution cases ------------------------------------------------------
    def add_resolution_case(self, case: ResolutionCase) -> None:
        self._run(
            "MERGE (rc:ResolutionCase {id: $id}) "
            "SET rc.conflict_signature = $signature, "
            "    rc.research_notes = $research_notes, "
            "    rc.decision = $decision, "
            "    rc.rationale = $rationale, "
            "    rc.version = $version, "
            "    rc.is_open = $is_open",
            id=case.id,
            signature=case.conflict_signature,
            research_notes=case.research_notes,
            decision=case.decision,
            rationale=case.rationale,
            version=case.version,
            is_open=case.is_open,
        )
        for claim_id in case.conflicting_claim_ids:
            self._run(
                "MATCH (rc:ResolutionCase {id: $case_id}), (c:Claim {id: $claim_id}) "
                "MERGE (rc)-[:RESOLVES]->(c)",
                case_id=case.id,
                claim_id=claim_id,
            )

    # -- transcripts & chunks --------------------------------------------------
    def upsert_transcript(
        self,
        *,
        transcript_id: str,
        entity_id: str,
        domain: str,
        source_kind: str,
        source_id: str,
        sha256: str,
        char_count: int,
        chunk_count: int,
    ) -> None:
        self._run(
            "MATCH (e:Entity {id: $entity_id}) "
            "MERGE (t:Transcript {id: $id}) "
            "SET t.domain = $domain, "
            "    t.source_kind = $source_kind, "
            "    t.source_id = $source_id, "
            "    t.sha256 = $sha256, "
            "    t.char_count = $char_count, "
            "    t.chunk_count = $chunk_count "
            "MERGE (t)-[:ABOUT]->(e)",
            id=transcript_id,
            entity_id=entity_id,
            domain=domain,
            source_kind=source_kind,
            source_id=source_id,
            sha256=sha256,
            char_count=char_count,
            chunk_count=chunk_count,
        )

    def add_chunk(
        self,
        *,
        chunk_id: str,
        transcript_id: str,
        index: int,
        text: str,
        char_start: int,
        char_end: int,
    ) -> None:
        self._run(
            "MATCH (t:Transcript {id: $transcript_id}) "
            "MERGE (ch:Chunk {id: $id}) "
            "SET ch.index = $index, "
            "    ch.text = $text, "
            "    ch.char_start = $char_start, "
            "    ch.char_end = $char_end "
            "MERGE (ch)-[:PART_OF]->(t)",
            id=chunk_id,
            transcript_id=transcript_id,
            index=index,
            text=text,
            char_start=char_start,
            char_end=char_end,
        )

    def link_claim_to_chunk(self, *, claim_id: str, chunk_id: str) -> None:
        self._run(
            "MATCH (c:Claim {id: $claim_id}), (ch:Chunk {id: $chunk_id}) "
            "MERGE (c)-[:DERIVED_FROM]->(ch)",
            claim_id=claim_id,
            chunk_id=chunk_id,
        )

    # -- retrieval -------------------------------------------------------------
    def find_similar_claims(
        self, *, embedding: list[float], k: int = 5, min_score: float = 0.0
    ) -> list[SimilarClaim]:
        """Vector search over claim embeddings using the native vector index."""

        if len(embedding) != self._embedding_dimensions:
            raise ValueError(
                f"query embedding length {len(embedding)} != index dimension "
                f"{self._embedding_dimensions}"
            )
        rows = self._run(
            "CALL db.index.vector.queryNodes($index, $k, $embedding) "
            "YIELD node, score "
            "WHERE score >= $min_score "
            "RETURN node.id AS id, node.statement AS statement, score",
            index=schema.CLAIM_VECTOR_INDEX,
            k=k,
            embedding=embedding,
            min_score=min_score,
        )
        return [
            SimilarClaim(claim_id=r["id"], statement=r["statement"], score=r["score"])
            for r in rows
        ]

    # -- diagnostics -----------------------------------------------------------
    def counts(self) -> dict[str, int]:
        labels = (
            schema.NODE_ENTITY,
            schema.NODE_CLAIM,
            schema.NODE_EVIDENCE,
            schema.NODE_RESOLUTION_CASE,
            schema.NODE_SLOT,
            schema.NODE_TRANSCRIPT,
            schema.NODE_CHUNK,
        )
        result: dict[str, int] = {}
        for label in labels:
            rows = self._run(f"MATCH (n:{label}) RETURN count(n) AS n")
            result[label] = int(rows[0]["n"]) if rows else 0
        return result
