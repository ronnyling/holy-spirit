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

    def vector_search_claims(
        self,
        *,
        embedding: list[float],
        domain: str | None = None,
        epistemic_status: str | None = None,
        k: int = 20,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Vector search over claims with optional domain and status filters.

        Uses the native vector index for approximate nearest neighbor search.
        """
        if len(embedding) != self._embedding_dimensions:
            raise ValueError(
                f"query embedding length {len(embedding)} != index dimension "
                f"{self._embedding_dimensions}"
            )

        # Fetch more than needed, then filter in Cypher
        fetch_k = k * 3 if domain or epistemic_status else k

        cypher = (
            "CALL db.index.vector.queryNodes($index, $fetch_k, $embedding) "
            "YIELD node, score "
            "WHERE score >= $min_score"
        )
        params: dict = {
            "index": schema.CLAIM_VECTOR_INDEX,
            "fetch_k": fetch_k,
            "embedding": embedding,
            "min_score": min_score,
        }

        if domain:
            cypher += " AND $domain IN node.tags"
            params["domain"] = domain
        if epistemic_status:
            cypher += " AND node.epistemic_status = $status"
            params["status"] = epistemic_status

        cypher += (
            " RETURN node.id AS id, node.statement AS statement, "
            "       node.epistemic_status AS status, node.tags AS tags, "
            "       node.slot_name AS slot_name, node.version AS version, "
            "       score "
            "ORDER BY score DESC LIMIT $limit"
        )
        params["limit"] = k

        rows = self._run(cypher, **params)
        return [
            {
                "claim_id": r["id"],
                "statement": r["statement"],
                "epistemic_status": r["status"],
                "tags": r["tags"],
                "slot_name": r["slot_name"],
                "version": r["version"],
                "similarity": r["score"],
            }
            for r in rows
        ]

    def vector_search_entities(
        self,
        *,
        embedding: list[float],
        domain: str | None = None,
        k: int = 20,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Semantic search: find claims nearest to query, return their entities."""
        if len(embedding) != self._embedding_dimensions:
            raise ValueError(
                f"query embedding length {len(embedding)} != index dimension "
                f"{self._embedding_dimensions}"
            )

        fetch_k = k * 5  # need extra to deduplicate entities

        cypher = (
            "CALL db.index.vector.queryNodes($index, $fetch_k, $embedding) "
            "YIELD node, score "
            "WHERE score >= $min_score"
        )
        params: dict = {
            "index": schema.CLAIM_VECTOR_INDEX,
            "fetch_k": fetch_k,
            "embedding": embedding,
            "min_score": min_score,
        }

        if domain:
            cypher += " AND $domain IN node.tags"
            params["domain"] = domain

        cypher += (
            " MATCH (node)-[:ABOUT]->(e:Entity) "
            " RETURN e.id AS entity_id, e.canonical_name AS name, "
            "        e.description AS description, e.version AS version, "
            "        score "
            " ORDER BY score DESC LIMIT $limit"
        )
        params["limit"] = k

        rows = self._run(cypher, **params)
        seen: set[str] = set()
        results: list[dict] = []
        for r in rows:
            eid = r["entity_id"]
            if eid in seen:
                continue
            seen.add(eid)
            results.append({
                "entity_id": eid,
                "canonical_name": r["name"],
                "description": r["description"],
                "version": r["version"],
                "relevance_score": r["score"],
            })
        return results

    def get_entity_with_details(self, entity_id: str) -> dict | None:
        """Return entity + all its claims, slots, and evidence as a nested dict."""
        rows = self._run(
            "MATCH (e:Entity {id: $id}) "
            "OPTIONAL MATCH (c:Claim)-[:ABOUT]->(e) "
            "OPTIONAL MATCH (ev:Evidence)-[:SUPPORTS]->(c) "
            "OPTIONAL MATCH (s:Slot)-[:OF]->(e) "
            "RETURN e AS entity, "
            "       collect(DISTINCT c) AS claims, "
            "       collect(DISTINCT ev) AS evidence, "
            "       collect(DISTINCT s) AS slots",
            id=entity_id,
        )
        if not rows or not rows[0]["entity"]:
            return None

        r = rows[0]
        entity = dict(r["entity"])
        claims = [dict(c) for c in r["claims"] if c.get("id")]
        evidence = [dict(ev) for ev in r["evidence"] if ev.get("id")]
        slots = [dict(s) for s in r["slots"] if s.get("id")]

        return {
            "entity": entity,
            "claims": claims,
            "evidence": evidence,
            "slots": slots,
        }

    def get_entity_by_name(self, entity_name: str) -> dict | None:
        """Find entity by canonical name (case-insensitive) and return full details."""
        rows = self._run(
            "MATCH (e:Entity) "
            "WHERE toLower(e.canonical_name) = toLower($name) "
            "RETURN e.id AS id LIMIT 1",
            name=entity_name,
        )
        if not rows:
            return None
        return self.get_entity_with_details(rows[0]["id"])

    def get_claim_detail(self, claim_id: str) -> dict | None:
        """Return full claim details with provenance and evidence chain."""
        rows = self._run(
            "MATCH (c:Claim {id: $id}) "
            "OPTIONAL MATCH (ev:Evidence)-[:SUPPORTS]->(c) "
            "OPTIONAL MATCH (c)-[:ABOUT]->(e:Entity) "
            "RETURN c AS claim, e AS entity, collect(DISTINCT ev) AS evidence",
            id=claim_id,
        )
        if not rows or not rows[0]["claim"]:
            return None

        r = rows[0]
        claim = dict(r["claim"])
        entity = dict(r["entity"]) if r["entity"] else None
        evidence = [dict(ev) for ev in r["evidence"] if ev.get("id")]

        return {
            "claim": claim,
            "entity": entity,
            "evidence": evidence,
        }

    def search_by_domain(
        self, *, domain: str, epistemic_status: str | None = None, limit: int = 50
    ) -> dict:
        """All confirmed knowledge in a domain — entities, claims, slots, cases."""
        status_filter = epistemic_status or "Confirmed"

        # Entities with claims in this domain
        entity_rows = self._run(
            "MATCH (c:Claim)-[:ABOUT]->(e:Entity) "
            "WHERE $domain IN c.tags AND c.epistemic_status = $status "
            "RETURN DISTINCT e.id AS id, e.canonical_name AS name, "
            "       e.description AS description, e.version AS version "
            "LIMIT $limit",
            domain=domain,
            status=status_filter,
            limit=limit,
        )

        claim_rows = self._run(
            "MATCH (c:Claim) "
            "WHERE $domain IN c.tags AND c.epistemic_status = $status "
            "RETURN c.id AS id, c.statement AS statement, c.slot_name AS slot_name, "
            "       c.epistemic_status AS status, c.version AS version "
            "LIMIT $limit",
            domain=domain,
            status=status_filter,
            limit=limit,
        )

        slot_rows = self._run(
            "MATCH (s:Slot)-[:OF]->(e:Entity) "
            "MATCH (c:Claim)-[:ABOUT]->(e) "
            "WHERE $domain IN c.tags "
            "RETURN DISTINCT s.id AS id, s.name AS name, s.lifecycle AS lifecycle, "
            "       s.observed_count AS observed_count, s.version AS version "
            "LIMIT $limit",
            domain=domain,
            limit=limit,
        )

        case_rows = self._run(
            "MATCH (rc:ResolutionCase)-[:RESOLVES]->(c:Claim) "
            "WHERE $domain IN c.tags "
            "RETURN DISTINCT rc.id AS id, rc.conflict_signature AS signature, "
            "       rc.decision AS decision, rc.rationale AS rationale, "
            "       rc.is_open AS is_open, rc.version AS version "
            "LIMIT $limit",
            domain=domain,
            limit=limit,
        )

        return {
            "domain": domain,
            "entities": [dict(r) for r in entity_rows],
            "claims": [dict(r) for r in claim_rows],
            "slots": [dict(r) for r in slot_rows],
            "resolution_cases": [dict(r) for r in case_rows],
        }

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
