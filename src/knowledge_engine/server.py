from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from .contracts import EvidenceDraft, TranscriptInput
from .engine import KnowledgeEngine
from .models import SlotLifecycle

engine = KnowledgeEngine()
mcp = MCPServer("knowledge-engine")


@mcp.tool()
def ingest_transcript(transcript: TranscriptInput) -> dict[str, Any]:
    return engine.ingest_transcript(transcript).model_dump()


@mcp.tool()
def confirm_slot(entity_name: str, slot_name: str, confirmed_by: str, target: SlotLifecycle = SlotLifecycle.EXPECTED) -> dict[str, Any]:
    return engine.confirm_slot(entity_name, slot_name, confirmed_by, target)


@mcp.tool()
def promote_claim(claim_id: str, evidence: list[EvidenceDraft]) -> dict[str, Any]:
    claim = engine.promote_claim(claim_id, evidence)
    return claim.model_dump()


@mcp.tool()
def resolve_case(case_id: str, decision: str, rationale: str) -> dict[str, Any]:
    return engine.resolve_case(case_id, decision=decision, rationale=rationale)


@mcp.tool()
def state_snapshot() -> dict[str, Any]:
    return engine.state_snapshot()


@mcp.resource("knowledge://state")
def knowledge_state() -> str:
    return json.dumps(engine.state_snapshot(), indent=2, sort_keys=True)


def main() -> None:
    mcp.run()
