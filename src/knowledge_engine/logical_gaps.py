from __future__ import annotations

from .contracts import GapFlag, GapKind
from .models import Claim, Evidence


class LogicalGapDetector:
    """Detect logical weaknesses in reasoning, not just missing evidence."""

    def detect(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
        """Analyze claims and evidence for logical fallacies."""
        gaps = []
        gaps.extend(self._detect_circular_reasoning(claims, evidence))
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
