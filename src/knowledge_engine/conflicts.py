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


# Semantic opposition pairs — partial word matches (prefix matching).
# When two claims in the same slot contain opposing keywords, they conflict
# regardless of text similarity.
_OPPOSITION_PAIRS: list[tuple[str, str]] = [
    ("diversif", "concentrat"),
    ("increase", "decrease"),
    ("always", "never"),
    ("best", "worst"),
    ("outperform", "underperform"),
    ("more", "less"),
    ("higher", "lower"),
    ("positive", "negative"),
    ("bullish", "bearish"),
    ("buy", "sell"),
    ("long", "short"),
    ("risk", "safe"),
    ("volatile", "stable"),
    ("growth", "decline"),
    ("profit", "loss"),
]


def _has_keyword_opposition(left: str, right: str) -> bool:
    """Check if two statements contain semantically opposing keywords."""
    left_lower = left.lower()
    right_lower = right.lower()
    for word_a, word_b in _OPPOSITION_PAIRS:
        if (word_a in left_lower and word_b in right_lower) or \
           (word_b in left_lower and word_a in right_lower):
            return True
    return False


class ConflictDetector:
    def detect(self, canonical_claims: list[Claim], incoming_claim: Claim) -> list[ConflictMatch]:
        matches: list[ConflictMatch] = []
        for existing in canonical_claims:
            if not self._same_aspect(existing, incoming_claim):
                continue
            if existing.statement.strip().lower() == incoming_claim.statement.strip().lower():
                continue

            # Two paths to conflict detection:
            # 1. Keyword opposition — semantically opposite claims conflict
            #    regardless of text similarity (fast, no LLM needed)
            # 2. Text similarity — paraphrase conflicts (existing behavior)
            is_opposition = _has_keyword_opposition(existing.statement, incoming_claim.statement)
            score = text_similarity(existing.statement, incoming_claim.statement)

            if not is_opposition and score < 0.35:
                continue

            message = (
                f"Claim conflicts with existing statement in slot "
                f"'{incoming_claim.slot_name or 'general'}'."
            )
            if is_opposition:
                message += " (semantic opposition detected)"

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
                    message=message,
                )
            )
        return matches

    def _same_aspect(self, left: Claim, right: Claim) -> bool:
        left_slot = (left.slot_name or "general").strip().lower()
        right_slot = (right.slot_name or "general").strip().lower()
        return left_slot == right_slot
