from __future__ import annotations

from .contracts import GapFlag, GapKind
from .models import Claim, Evidence


class LogicalGapDetector:
    """Detect logical weaknesses in reasoning, not just missing evidence."""

    def detect(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
        """Analyze claims and evidence for logical fallacies."""
        gaps = []
        gaps.extend(self._detect_circular_reasoning(claims, evidence))
        gaps.extend(self._detect_cherry_picking(claims, evidence))
        gaps.extend(self._detect_over_generalization(claims, evidence))
        return gaps

    def _detect_circular_reasoning(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
        """Detect circular evidence chains: A supports B supports C supports A."""
        gaps = []

        # Build adjacency map: claim_id -> list of claim_ids it supports
        supports_map: dict[str, set[str]] = {}
        for ev in evidence:
            if ev.linked_claim_ids:
                target_claim_id = ev.claim_id
                for source_id in ev.linked_claim_ids:
                    if source_id not in supports_map:
                        supports_map[source_id] = set()
                    supports_map[source_id].add(target_claim_id)

        # DFS cycle detection
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node: str, path: list[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in supports_map.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, path + [neighbor]):
                        return True
                elif neighbor in rec_stack:
                    cycle_claims = [neighbor] + list(path)
                    gaps.append(GapFlag(
                        kind=GapKind.LOGICAL,
                        entity_id="",
                        slot_name="reasoning_chain",
                        question=f"Circular reasoning detected: {' -> '.join(cycle_claims)}",
                        severity="high",
                        rationale=f"Circular reasoning: claim chain forms a loop {' -> '.join(cycle_claims)}"
                    ))
                    return True

            rec_stack.remove(node)
            return False

        claim_ids = {c.id for c in claims}
        for claim_id in claim_ids:
            if claim_id not in visited:
                has_cycle(claim_id, [claim_id])

        return gaps

    def _detect_cherry_picking(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
        """Detect when only favorable evidence is cited (ignoring unfavorable)."""
        gaps = []

        # Group evidence by claim_id
        evidence_by_claim: dict[str, list[Evidence]] = {}
        for ev in evidence:
            if ev.claim_id not in evidence_by_claim:
                evidence_by_claim[ev.claim_id] = []
            evidence_by_claim[ev.claim_id].append(ev)

        for claim in claims:
            claim_evidence = evidence_by_claim.get(claim.id, [])
            if len(claim_evidence) < 2:
                continue

            credibilities = [ev.credibility for ev in claim_evidence]
            min_cred = min(credibilities)
            max_cred = max(credibilities)

            # Cherry-picking: large credibility spread suggests selective citation
            if max_cred - min_cred >= 0.5:
                gaps.append(GapFlag(
                    kind=GapKind.LOGICAL,
                    entity_id=claim.entity_id,
                    slot_name=claim.slot_name or "general",
                    question="Is all evidence for this claim favorable? Consider counter-evidence.",
                    severity="medium",
                    rationale=f"Cherry-picking: evidence credibility ranges from {min_cred:.2f} to {max_cred:.2f}, suggesting selective citation"
                ))

        return gaps

    def _detect_over_generalization(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
        """Detect universal claims ('all', 'always', 'never') with limited evidence."""
        gaps = []

        universal_keywords = ["all", "always", "never", "every", "none", "universally"]
        min_evidence_for_universal = 5  # Threshold for universal claims

        evidence_count: dict[str, int] = {}
        for ev in evidence:
            evidence_count[ev.claim_id] = evidence_count.get(ev.claim_id, 0) + 1

        for claim in claims:
            statement_lower = claim.statement.lower()

            # Check if claim is universal
            is_universal = any(keyword in statement_lower for keyword in universal_keywords)

            if is_universal:
                count = evidence_count.get(claim.id, 0)
                if count < min_evidence_for_universal:
                    gaps.append(GapFlag(
                        kind=GapKind.LOGICAL,
                        entity_id=claim.entity_id,
                        slot_name=claim.slot_name or "general",
                        question=f"Universal claim has only {count} evidence items. Is this sufficient?",
                        severity="medium",
                        rationale=f"Over-generalization: universal claim '{claim.statement[:50]}...' supported by only {count} evidence items (threshold: {min_evidence_for_universal})"
                    ))

        return gaps
