from __future__ import annotations

from collections import defaultdict, deque
from pydantic import BaseModel, ConfigDict, Field

from .contracts import EvidenceDraft
from .models import Claim, Evidence, EpistemicStatus
from .policy import get_domain_policy
from .store import KnowledgeStore


class EvidenceEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    source_count: int
    can_confirm: bool
    reasons: list[str] = Field(default_factory=list)


class EvidenceLedger:
    def __init__(self) -> None:
        self._support_graph: dict[str, set[str]] = defaultdict(set)

    def evaluate(self, domain: str, evidence_drafts: list[EvidenceDraft], store: KnowledgeStore, target_claim: Claim) -> EvidenceEvaluation:
        policy = get_domain_policy(domain)
        reasons: list[str] = []
        score = 0.0
        source_ids: set[str] = set()

        for draft in evidence_drafts:
            source_ids.add(draft.source_id)
            score += draft.credibility

            if draft.source_kind == "internal_wiki":
                try:
                    source_claim = store.load_claim(draft.source_id)
                except (KeyError, Exception):
                    reasons.append(f"internal source claim {draft.source_id} does not exist")
                    continue
                if source_claim.epistemic_status != EpistemicStatus.CONFIRMED:
                    reasons.append(f"internal source claim {draft.source_id} is not confirmed")
                    continue
                if self._would_create_cycle(source_claim.id or "", target_claim.id or ""):
                    reasons.append("internal evidence would create a dependency cycle")

        if len(source_ids) < policy.minimum_sources:
            reasons.append(f"domain policy requires at least {policy.minimum_sources} supporting sources")

        if score < policy.minimum_score:
            reasons.append(f"domain policy requires an evidence score of at least {policy.minimum_score:.1f}")

        can_confirm = not reasons
        return EvidenceEvaluation(score=score, source_count=len(source_ids), can_confirm=can_confirm, reasons=reasons)

    def record_for_claim(self, claim: Claim, evidence_drafts: list[EvidenceDraft], store: KnowledgeStore) -> list[Evidence]:
        recorded: list[Evidence] = []
        for draft in evidence_drafts:
            evidence = Evidence(
                claim_id=claim.id or "",
                source_kind=draft.source_kind,
                source_id=draft.source_id,
                source_ref=draft.source_ref,
                credibility=draft.credibility,
                notes=draft.notes,
                linked_claim_ids=draft.linked_claim_ids or [claim.id or ""],
            )
            if evidence.source_kind == "internal_wiki":
                self.attach_internal_support(source_claim_id=evidence.source_id, target_claim_id=claim.id or "", store=store)
            recorded.append(store.add_evidence(evidence))
        return recorded

    def attach_internal_support(self, source_claim_id: str, target_claim_id: str, store: KnowledgeStore) -> None:
        try:
            source_claim = store.load_claim(source_claim_id)
        except (KeyError, Exception):
            raise ValueError(f"unknown source claim: {source_claim_id}")
        if source_claim.epistemic_status != EpistemicStatus.CONFIRMED:
            raise ValueError(f"source claim {source_claim_id} is not confirmed")
        if self._would_create_cycle(source_claim_id, target_claim_id):
            raise ValueError("internal evidence would create a dependency cycle")
        self._support_graph[source_claim_id].add(target_claim_id)

    def _would_create_cycle(self, source_claim_id: str, target_claim_id: str) -> bool:
        if source_claim_id == target_claim_id:
            return True
        queue: deque[str] = deque([target_claim_id])
        visited: set[str] = set()
        while queue:
            current = queue.popleft()
            if current == source_claim_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._support_graph.get(current, set()))
        return False
