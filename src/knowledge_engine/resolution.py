from __future__ import annotations

from .conflicts import ConflictMatch
from .models import ResolutionCase
from .store import KnowledgeStore
from .utils import normalize_text


class ResolutionMemory:
    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def open_case(self, conflict: ConflictMatch) -> ResolutionCase:
        precedent = self.store.find_similar_case(conflict.conflict_signature)
        case = ResolutionCase(
            conflict_signature=conflict.conflict_signature,
            conflicting_claim_ids=[claim_id for claim_id in conflict.existing_claim_ids if claim_id],
            evidence_ids=[],
            research_notes=precedent.research_notes if precedent else None,
            decision=None,
            rationale=None,
            reopened_from_case_id=precedent.id if precedent else None,
            is_open=True,
        )
        return self.store.add_resolution_case(case)

    def resolve_case(self, case_id: str, decision: str, rationale: str) -> ResolutionCase:
        case = self.store.get_resolution_case(case_id)
        case.decision = decision
        case.rationale = rationale
        case.is_open = False
        case.version += 1
        return case

    def reopen_signature(self, entity_id: str, slot_name: str, existing_statement: str, incoming_statement: str) -> ResolutionCase | None:
        signature = f"{entity_id}:{slot_name}:{normalize_text(existing_statement)} => {normalize_text(incoming_statement)}"
        return self.store.find_similar_case(signature)
