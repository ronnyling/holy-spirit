from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .models import Claim
from .utils import normalize_text, text_similarity


class ConflictMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict_signature: str
    incoming_claim_id: str
    existing_claim_ids: list[str] = Field(default_factory=list)
    existing_statements: list[str] = Field(default_factory=list)
    message: str


class ConflictDetector:
    def detect(self, canonical_claims: list[Claim], incoming_claim: Claim) -> list[ConflictMatch]:
        matches: list[ConflictMatch] = []
        for existing in canonical_claims:
            if not self._same_aspect(existing, incoming_claim):
                continue
            if existing.statement.strip().lower() == incoming_claim.statement.strip().lower():
                continue
            score = text_similarity(existing.statement, incoming_claim.statement)
            if score < 0.35:
                continue
            signature = (
                f"{incoming_claim.entity_id}:{incoming_claim.slot_name or 'general'}:"
                f"{normalize_text(existing.statement)} => {normalize_text(incoming_claim.statement)}"
            )
            matches.append(
                ConflictMatch(
                    conflict_signature=signature,
                    incoming_claim_id=incoming_claim.id or "",
                    existing_claim_ids=[existing.id or ""],
                    existing_statements=[existing.statement],
                    message=f"Claim conflicts with canonical statement in slot '{incoming_claim.slot_name or 'general'}'.",
                )
            )
        return matches

    def _same_aspect(self, left: Claim, right: Claim) -> bool:
        left_slot = (left.slot_name or "general").strip().lower()
        right_slot = (right.slot_name or "general").strip().lower()
        return left_slot == right_slot
