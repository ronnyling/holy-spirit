from __future__ import annotations

from .contracts import GapFlag, GapKind, ClaimDraft, TranscriptInput
from .logical_gaps import LogicalGapDetector
from .models import Claim, Evidence, Slot
from .utils import needs_context


class GapDetector:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logical_detector = LogicalGapDetector(llm_client=llm_client)

    def structural_gaps(self, entity_id: str, observed_slot_names: set[str], expected_slots: list[Slot]) -> list[GapFlag]:
        gaps: list[GapFlag] = []
        observed_lookup = {name.strip().lower() for name in observed_slot_names}

        for slot in expected_slots:
            if slot.name.strip().lower() in observed_lookup:
                continue

            question = f"Please provide the expected {slot.name} for this topic."
            if slot.description:
                question = f"Please provide the expected {slot.name}: {slot.description}"

            gaps.append(
                GapFlag(
                    kind=GapKind.STRUCTURAL,
                    entity_id=entity_id,
                    slot_name=slot.name,
                    question=question,
                    rationale="A learned expected slot is missing from this transcript.",
                )
            )

        return gaps

    def semantic_gaps(self, entity_id: str, transcript: TranscriptInput, claim_draft: ClaimDraft) -> list[GapFlag]:
        gaps: list[GapFlag] = []
        if claim_draft.evidence:
            return gaps

        if not needs_context(claim_draft.statement):
            return gaps

        question = self._question_for_domain(transcript.domain, claim_draft)
        gaps.append(
            GapFlag(
                kind=GapKind.SEMANTIC,
                entity_id=entity_id,
                slot_name=claim_draft.slot_name or "context",
                question=question,
                rationale="The claim is directional but lacks context, conditions, or supporting evidence.",
            )
        )
        return gaps

    def _question_for_domain(self, domain: str, claim_draft: ClaimDraft) -> str:
        lowered = domain.strip().lower()
        if lowered in {"trading", "stock trading"}:
            return "What market regime, backtest evidence, or failure mode supports this claim?"
        if lowered in {"real estate", "real_estate"}:
            return "Which jurisdiction, market cycle, capital stack, or exit assumption does this claim depend on?"
        if lowered in {"tcm", "traditional chinese medicine"}:
            return "Which lineage, classical source, or corroborating practitioner supports this interpretation?"
        if claim_draft.slot_name:
            return f"What evidence or conditions support the {claim_draft.slot_name} claim?"
        return "What evidence or conditions support this claim?"

    def logical_gaps(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
        """Detect logical fallacies in claims and evidence."""
        return self.logical_detector.detect(claims, evidence)
