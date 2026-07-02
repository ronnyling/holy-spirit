"""Knowledge Engine MCP Server.

Exposes the evidence-gated knowledge graph as MCP tools.
Requires Neo4j 5 with native vector index — no in-memory fallback.

Run:
    python server.py

Tools:
  ingest_transcript   — Harbour a transcript and run the full pipeline
  confirm_slot        — Human-in-the-loop slot promotion
  promote_claim       — Evidence-gated claim promotion
  resolve_case        — Record conflict resolution
  state_snapshot      — JSON snapshot of engine state
  search_claims       — Vector search over claims
  search_entities     — Vector search: find entities by semantic similarity
  get_entity          — Full entity details with claims, slots, evidence
  get_claim           — Claim details with provenance and evidence chain
  search_by_domain    — All confirmed knowledge in a domain

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
    """Human-in-the-loop promotion of a slot lifecycle.

    target: CANDIDATE (3+ observations) or EXPECTED (5+ observations).
    Requires confirmed_by to identify the human approver.
    """
    return engine.confirm_slot(entity_name, slot_name, confirmed_by, target)


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


# -- resources -----------------------------------------------------------------


@mcp_server.resource("knowledge://state")
def knowledge_state() -> str:
    return json.dumps(engine.state_snapshot(), indent=2, sort_keys=True)


# -- entry point ---------------------------------------------------------------


def main() -> None:
    mcp_server.run()
