"""Knowledge Engine MCP Server.

Exposes the evidence-gated knowledge graph as MCP tools.
Requires Neo4j 5 with native vector index — no in-memory fallback.

Run:
    python server.py

Tools:
  ingest_transcript       — Harbour a transcript and run the full pipeline
  confirm_slot            — Human-in-the-loop slot promotion
  list_pending_promotions — Retrieve the slot promotion queue
  promote_claim           — Evidence-gated claim promotion
  resolve_case            — Record conflict resolution
  state_snapshot          — JSON snapshot of engine state
  search_claims           — Vector search over claims
  search_entities         — Vector search: find entities by semantic similarity
  get_entity              — Full entity details with claims, slots, evidence
  get_claim               — Claim details with provenance and evidence chain
  search_by_domain        — All confirmed knowledge in a domain
  explore_experience      — World knowledge discerned through system experience

Resource:
  knowledge://state   — JSON snapshot of current engine state
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .bootstrap import build_engine_from_env
from .contracts import EvidenceDraft, TranscriptInput
from .models import SlotLifecycle

engine = build_engine_from_env()

mcp_server = FastMCP(
    "knowledge-engine",
    instructions=(
        "Evidence-gated knowledge graph for transcripts, claims, entities, "
        "and resolution cases. Requires Neo4j 5 with native vector index."
    ),
)


# -- ingestion tools -----------------------------------------------------------


@mcp_server.tool()
def ingest_transcript(transcript: TranscriptInput) -> dict[str, Any]:
    """Harbour a transcript and run the full pipeline.

    Returns entity, harbour metadata, gaps, conflicts, and claim outcomes.
    """
    return engine.ingest_transcript(transcript).model_dump()


@mcp_server.tool()
def confirm_slot(
    entity_name: str,
    slot_name: str,
    confirmed_by: str,
    target: SlotLifecycle = SlotLifecycle.EXPECTED,
) -> dict[str, Any]:
    """Promote a slot from the pending queue to Candidate or Expected lifecycle.

    Retrieve the queue first with ``list_pending_promotions`` to see what
    needs review. Confirming removes the item from the queue.
    target: CANDIDATE (3+ observations) or EXPECTED (5+ observations).
    """
    return engine.confirm_slot(entity_name, slot_name, confirmed_by, target)


@mcp_server.tool()
def list_pending_promotions() -> dict[str, Any]:
    """Return all pending slot promotion candidates from the queue.

    The queue accumulates automatically as transcripts are ingested and the
    slot learner detects threshold crossings. Review items here, then call
    ``confirm_slot`` to promote or leave them pending.
    """
    items = engine.list_pending_promotions()
    return {"pending": items, "count": len(items)}


@mcp_server.tool()
def promote_claim(claim_id: str, evidence: list[EvidenceDraft]) -> dict[str, Any]:
    """Promote a claim's epistemic status when evidence gates are satisfied.

    Evidence is evaluated against per-domain gates (score + source count).
    """
    claim = engine.promote_claim(claim_id, evidence)
    return claim.model_dump()


@mcp_server.tool()
def resolve_case(case_id: str, decision: str, rationale: str) -> dict[str, Any]:
    """Record a decision and rationale on a ResolutionCase.

    Cases are versioned and reopenable. The decision becomes reusable memory
    for future similar conflicts.
    """
    return engine.resolve_case(case_id, decision=decision, rationale=rationale)


@mcp_server.tool()
def state_snapshot() -> dict[str, Any]:
    """Return a JSON snapshot of current engine state.

    Counts: entities, claims, confirmed claims, open cases, slots, evidence.
    """
    return engine.state_snapshot()


# -- query tools (vector search) -----------------------------------------------


@mcp_server.tool()
def search_claims(
    query: str,
    domain: str | None = None,
    epistemic_status: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Vector search over claims using semantic similarity.

    Finds claims whose content is semantically similar to the query text.
    Optional filters: domain (e.g. 'trading', 'tcm', 'real_estate'),
    epistemic_status (e.g. 'Confirmed', 'Unverified', 'Disputed').
    """
    return engine.search_claims(query, domain=domain, epistemic_status=epistemic_status, limit=limit)


@mcp_server.tool()
def search_entities(
    query: str,
    domain: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Vector search: find entities whose claims are semantically similar to query.

    Returns deduplicated entities ranked by relevance score.
    Optional filter: domain tag.
    """
    return engine.search_entities(query, domain=domain, limit=limit)


@mcp_server.tool()
def get_entity(
    entity_id: str | None = None,
    entity_name: str | None = None,
) -> dict[str, Any]:
    """Full entity details with all claims, slots, and evidence.

    Provide either entity_id or entity_name. Returns the complete subgraph
    for the entity including provenance chains.
    """
    return engine.get_entity(entity_id=entity_id, entity_name=entity_name)


@mcp_server.tool()
def get_claim(claim_id: str) -> dict[str, Any]:
    """Full claim details with provenance, evidence chain, and linked entity."""
    return engine.get_claim(claim_id)


@mcp_server.tool()
def search_by_domain(domain: str, limit: int = 50) -> dict[str, Any]:
    """All confirmed knowledge in a domain.

    Returns entities, claims, slots, and resolution cases tagged with the
    given domain (e.g. 'trading', 'tcm', 'real_estate').
    """
    return engine.search_by_domain(domain, limit=limit)


@mcp_server.tool()
def list_domains() -> dict[str, Any]:
    """List all knowledge domains that have been ingested into the system.

    Returns a sorted list of domain tags alongside summary counts so consumers
    can discover what experience the engine has accumulated before querying.
    """
    from .graph.neo4j_store import KnowledgeGraphStore
    from .policy import list_policy_domains

    if isinstance(engine.store, KnowledgeGraphStore):
        graph_domains = engine.store.list_domains()
    else:
        # JSON store — derive from claim tags.
        graph_domains = sorted({
            t for c in engine.store.claims.values() for t in c.tags if t
        })

    policy_domains = list_policy_domains()
    return {
        "ingested_domains": graph_domains,
        "policy_domains": policy_domains,
        "total": len(graph_domains),
    }


@mcp_server.tool()
def explore_domain(domain: str, limit: int = 50) -> dict[str, Any]:
    """Full epistemic state of a domain — the system's accumulated experience.

    Unlike ``search_by_domain`` (which returns only Confirmed claims), this tool
    returns the complete picture: confirmed knowledge, contested claims, pending
    evidence, resolution precedents, and the emergent slot schema.  This is the
    'experience pack' a downstream agent should load before acting as an expert
    in the given domain.
    """
    from .graph.neo4j_store import KnowledgeGraphStore
    from .models import EpistemicStatus

    if not isinstance(engine.store, KnowledgeGraphStore):
        return {"error": "explore_domain requires the Neo4j graph store"}

    store = engine.store
    confirmed = store.search_by_domain(domain=domain, epistemic_status="Confirmed", limit=limit)
    disputed  = store.search_by_domain(domain=domain, epistemic_status="Disputed",  limit=limit)
    unverified = store.search_by_domain(domain=domain, epistemic_status="Unverified", limit=limit)

    snap = store.snapshot()
    return {
        "domain": domain,
        "summary": {
            "confirmed_claims": len(confirmed.get("claims", [])),
            "unverified_claims": len(unverified.get("claims", [])),
            "disputed_claims": len(disputed.get("claims", [])),
            "open_cases": snap.get("open_cases", 0),
        },
        "confirmed": confirmed,
        "unverified": unverified,
        "disputed": disputed,
    }


@mcp_server.tool()
def find_cross_domain_patterns(
    domains: list[str] | None = None,
    min_similarity: float = 0.7,
    limit: int = 20,
) -> dict[str, Any]:
    """Find semantically similar claims that span different knowledge domains.

    Useful for discovering macro-level connections — e.g. a TCM concept about
    inflammation that resembles a trading concept about market stress, or a
    real-estate cash-flow rule that mirrors a dividend-investing principle.

    Returns pairs of claims from disjoint domain sets, ranked by cosine similarity.
    """
    from .graph.neo4j_store import KnowledgeGraphStore

    if not isinstance(engine.store, KnowledgeGraphStore):
        return {"error": "find_cross_domain_patterns requires the Neo4j graph store"}

    patterns = engine.store.find_cross_domain_patterns(
        domains=domains,
        min_similarity=min_similarity,
        limit=limit,
    )
    return {"patterns": patterns, "count": len(patterns)}


@mcp_server.tool()
def explore_experience(query: str, domain: str | None = None) -> dict[str, Any]:
    """Synthesize world knowledge discerned through accumulated system experience.

    Unlike ``search_claims`` (which retrieves matching facts), this tool builds
    a three-part response:
      [WORLD VIEW]        What is commonly understood about this topic.
      [EXPERIENCE]        What the knowledge base adds, corrects, or confirms,
                          with each point labelled by epistemic status.
      [DISCERNED POSITION] The practical synthesis \u2014 where experience departs
                          from world knowledge, experience leads.

    The LLM is strictly constrained to the provided experience claims; it cannot
    invent knowledge not present in the graph. When the system is cold (no
    experience yet), the world view is returned as-is with a note.
    """
    return engine.explore_experience(query, domain=domain)


# -- resources -----------------------------------------------------------------


@mcp_server.resource("knowledge://state")
def knowledge_state() -> str:
    return json.dumps(engine.state_snapshot(), indent=2, sort_keys=True)


# -- entry point ---------------------------------------------------------------


def main() -> None:
    mcp_server.run()
