from __future__ import annotations

from typing import Any
from uuid import uuid4

from .conflicts import ConflictDetector, ConflictMatch
from .contracts import ClaimDraft, ConflictSummary, EvidenceDraft, GapFlag, SlotSuggestion, TranscriptInput, TranscriptOutcome
from .evidence import EvidenceLedger
from .gaps import GapDetector
from .learning import SlotLearner
from .models import Claim, EpistemicStatus, Provenance, SlotLifecycle
from .policy import get_domain_policy
from .registry import HarbourResult, TranscriptRegistry
from .resolution import ResolutionMemory
from .store import KnowledgeStore


class KnowledgeEngine:
    def __init__(self, *, registry: TranscriptRegistry | None = None) -> None:
        self.store = KnowledgeStore()
        self.slot_learner = SlotLearner()
        self.gap_detector = GapDetector()
        self.conflict_detector = ConflictDetector()
        self.resolution_memory = ResolutionMemory(self.store)
        self.evidence_ledger = EvidenceLedger()
        # When a registry is wired (production / e2e path), every ingested transcript
        # is harboured, documented, and housekept before any claim processing.
        self.registry = registry

    def ingest_transcript(self, transcript: TranscriptInput) -> TranscriptOutcome:
        harbour: HarbourResult | None = None
        if self.registry is not None:
            harbour = self.registry.harbour(
                text=transcript.transcript_text,
                domain=transcript.domain,
                entity_name=transcript.entity_name,
                source_kind=transcript.source_kind,
                source_id=transcript.source_id,
                source_ref=transcript.source_ref,
            )

        entity = self.store.upsert_entity(canonical_name=transcript.entity_name)
        claim_ids: list[str] = []
        new_claims: list[Claim] = []
        confirmed_claim_ids: list[str] = []
        unverified_claim_ids: list[str] = []
        disputed_claim_ids: list[str] = []
        slot_suggestions: list[SlotSuggestion] = []
        gap_flags: list[GapFlag] = []
        conflict_summaries: list[ConflictSummary] = []
        open_case_ids: list[str] = []
        notes: list[str] = []
        observed_slot_names: set[str] = set()

        for draft in transcript.claim_drafts:
            claim = self._build_claim(transcript, entity.id or "", draft)
            claim = self.store.add_claim(claim)
            claim_ids.append(claim.id or "")
            new_claims.append(claim)

            for slot_name in draft.observed_slots:
                observed_slot_names.add(slot_name)
                observation = self.slot_learner.observe(self.store, entity.id or "", slot_name)
                if observation.suggested_lifecycle is not None:
                    slot_suggestions.append(
                        SlotSuggestion(
                            slot_name=observation.slot.name,
                            current_lifecycle=observation.slot.lifecycle.value,
                            suggested_lifecycle=observation.suggested_lifecycle.value,
                            observed_count=observation.slot.observed_count,
                            reason=observation.reason or "slot crossed a learned threshold",
                        )
                    )

            gap_flags.extend(self.gap_detector.semantic_gaps(entity.id or "", transcript, draft))
            if draft.evidence:
                try:
                    recorded_evidence = self.evidence_ledger.record_for_claim(claim, draft.evidence, self.store)
                    notes.extend(self._recorded_evidence_notes(claim, recorded_evidence))
                except ValueError as exc:
                    notes.append(str(exc))

        gap_flags.extend(
            self.gap_detector.structural_gaps(
                entity.id or "",
                observed_slot_names,
                self.store.get_expected_slots(entity.id or ""),
            )
        )

        canonical_claims = self.store.list_canonical_claims(entity.id or "")
        for claim, draft in zip(new_claims, transcript.claim_drafts):
            matches = self.conflict_detector.detect(canonical_claims, claim)
            if matches:
                claim.epistemic_status = EpistemicStatus.DISPUTED
                disputed_claim_ids.append(claim.id or "")
                for match in matches:
                    case = self.resolution_memory.open_case(match)
                    open_case_ids.append(case.id or "")
                    conflict_summaries.append(
                        ConflictSummary(
                            conflict_signature=match.conflict_signature,
                            incoming_claim_id=match.incoming_claim_id,
                            existing_claim_ids=match.existing_claim_ids,
                            case_id=case.id,
                            message=match.message,
                        )
                    )
                continue

            can_confirm = self._can_confirm_claim(transcript.domain, transcript.source_kind, claim, draft, gap_flags)
            if can_confirm:
                claim.epistemic_status = EpistemicStatus.CONFIRMED
                confirmed_claim_ids.append(claim.id or "")
            else:
                claim.epistemic_status = EpistemicStatus.UNVERIFIED
                unverified_claim_ids.append(claim.id or "")

        return TranscriptOutcome(
            entity_id=entity.id or "",
            transcript_id=harbour.record.transcript_id if harbour else None,
            transcript_created=harbour.created if harbour else None,
            chunk_count=harbour.record.chunk_count if harbour else None,
            claim_ids=claim_ids,
            confirmed_claim_ids=confirmed_claim_ids,
            unverified_claim_ids=unverified_claim_ids,
            disputed_claim_ids=disputed_claim_ids,
            slot_suggestions=slot_suggestions,
            gap_flags=gap_flags,
            conflict_summaries=conflict_summaries,
            open_case_ids=open_case_ids,
            canonical_claim_ids=[claim.id or "" for claim in self.store.list_canonical_claims(entity.id or "")],
            notes=notes,
        )

    def confirm_slot(self, entity_name: str, slot_name: str, confirmed_by: str, target: SlotLifecycle) -> dict[str, Any]:
        entity = self.store.upsert_entity(canonical_name=entity_name)
        slot = self.store.confirm_slot(entity.id or "", slot_name, target=target, confirmed_by=confirmed_by)
        return {
            "slot_id": slot.id,
            "entity_id": entity.id,
            "name": slot.name,
            "lifecycle": slot.lifecycle.value,
            "observed_count": slot.observed_count,
            "candidate_count": slot.candidate_count,
            "expected_count": slot.expected_count,
        }

    def promote_claim(self, claim_id: str, evidence: list[EvidenceDraft]) -> Claim:
        claim = self.store.load_claim(claim_id)
        evaluation = self.evidence_ledger.evaluate(self._claim_domain(claim), evidence, self.store, claim)
        if not evaluation.can_confirm:
            raise ValueError("claim does not meet the evidence gate: " + "; ".join(evaluation.reasons))
        self.evidence_ledger.record_for_claim(claim, evidence, self.store)
        claim.epistemic_status = EpistemicStatus.CONFIRMED
        return claim

    def resolve_case(self, case_id: str, decision: str, rationale: str) -> dict[str, Any]:
        case = self.resolution_memory.resolve_case(case_id, decision=decision, rationale=rationale)
        return {
            "case_id": case.id,
            "decision": case.decision,
            "rationale": case.rationale,
            "version": case.version,
            "is_open": case.is_open,
            "reopened_from_case_id": case.reopened_from_case_id,
        }

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "entities": len(self.store.entities),
            "claims": len(self.store.claims),
            "confirmed_claims": len([claim for claim in self.store.claims.values() if claim.epistemic_status == EpistemicStatus.CONFIRMED]),
            "open_cases": len([case for case in self.store.resolution_cases.values() if case.is_open]),
            "slots": len(self.store.slots),
            "evidence": len(self.store.evidence),
        }

    def _build_claim(self, transcript: TranscriptInput, entity_id: str, draft: ClaimDraft) -> Claim:
        return Claim(
            id=uuid4().hex,
            entity_id=entity_id,
            statement=draft.statement,
            slot_name=draft.slot_name,
            epistemic_status=EpistemicStatus.UNVERIFIED,
            provenance=[
                Provenance(
                    source_kind=transcript.source_kind,
                    source_id=transcript.source_id,
                    source_ref=transcript.source_ref,
                    notes=transcript.notes,
                )
            ],
            tags=[transcript.domain],
        )

    def _claims_for_entity(self, entity_id: str) -> list[Claim]:
        return [claim for claim in self.store.claims.values() if claim.entity_id == entity_id]

    def _can_confirm_claim(
        self,
        domain: str,
        source_kind: str,
        claim: Claim,
        draft: ClaimDraft,
        gap_flags: list[GapFlag],
    ) -> bool:
        if source_kind == "user":
            return False
        if any(flag.entity_id == claim.entity_id for flag in gap_flags):
            return False
        if not draft.evidence:
            return False
        evaluation = self.evidence_ledger.evaluate(domain, draft.evidence, self.store, claim)
        return evaluation.can_confirm

    def _recorded_evidence_notes(self, claim: Claim, evidence: list[Any]) -> list[str]:
        if not evidence:
            return []
        return [f"recorded {len(evidence)} evidence item(s) for claim {claim.id}"]

    def _claim_domain(self, claim: Claim) -> str:
        if claim.tags:
            return claim.tags[0]
        return "default"
