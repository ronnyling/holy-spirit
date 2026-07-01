from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .models import Claim, Entity, Evidence, EpistemicStatus, ResolutionCase, Slot, SlotLifecycle
from .utils import normalize_text, stable_signature, text_similarity


class KnowledgeStore:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.claims: dict[str, Claim] = {}
        self.evidence: dict[str, Evidence] = {}
        self.slots: dict[str, Slot] = {}
        self.resolution_cases: dict[str, ResolutionCase] = {}
        self._slot_index: dict[tuple[str, str], str] = {}

    def upsert_entity(
        self,
        canonical_name: str,
        aliases: list[str] | None = None,
        description: str | None = None,
    ) -> Entity:
        normalized = canonical_name.strip().lower()
        for entity in self.entities.values():
            if entity.canonical_name.strip().lower() == normalized:
                return entity
            if normalized in {alias.strip().lower() for alias in entity.aliases}:
                return entity

        entity = Entity(
            id=uuid4().hex,
            canonical_name=canonical_name,
            aliases=aliases or [],
            description=description,
        )
        self.entities[entity.id] = entity
        return entity

    def add_claim(self, claim: Claim) -> Claim:
        if not claim.id:
            claim.id = uuid4().hex
        self.claims[claim.id] = claim
        return claim

    def add_evidence(self, evidence: Evidence) -> Evidence:
        if not evidence.id:
            evidence.id = uuid4().hex
        self.evidence[evidence.id] = evidence
        return evidence

    def load_claim(self, claim_id: str) -> Claim:
        return self.claims[claim_id]

    def list_claims_for_entity(self, entity_id: str) -> list[Claim]:
        return [claim for claim in self.claims.values() if claim.entity_id == entity_id]

    def list_canonical_claims(self, entity_id: str) -> list[Claim]:
        return [
            claim
            for claim in self.claims.values()
            if claim.entity_id == entity_id and claim.epistemic_status == EpistemicStatus.CONFIRMED
        ]

    def observe_slot(self, entity_id: str, slot_name: str, description: str | None = None) -> Slot:
        key = (entity_id, slot_name.strip().lower())
        slot_id = self._slot_index.get(key)
        if slot_id is None:
            slot = Slot(
                id=uuid4().hex,
                entity_id=entity_id,
                name=slot_name,
                description=description,
            )
            self._slot_index[key] = slot.id
            self.slots[slot.id] = slot
        else:
            slot = self.slots[slot_id]
            if description and not slot.description:
                slot.description = description

        slot.observed_count += 1
        slot.last_observed_at = datetime.now(timezone.utc)
        return slot

    def get_slot(self, entity_id: str, slot_name: str) -> Slot | None:
        slot_id = self._slot_index.get((entity_id, slot_name.strip().lower()))
        return self.slots.get(slot_id) if slot_id else None

    def get_slots_for_entity(self, entity_id: str) -> list[Slot]:
        return [slot for slot in self.slots.values() if slot.entity_id == entity_id]

    def get_expected_slots(self, entity_id: str) -> list[Slot]:
        return [
            slot
            for slot in self.get_slots_for_entity(entity_id)
            if slot.lifecycle == SlotLifecycle.EXPECTED
        ]

    def confirm_slot(self, entity_id: str, slot_name: str, target: SlotLifecycle, confirmed_by: str) -> Slot:
        if not confirmed_by.strip():
            raise ValueError("human confirmation is required to promote a slot")

        slot = self.get_slot(entity_id, slot_name)
        if slot is None:
            raise ValueError(f"unknown slot: {slot_name}")

        if target == SlotLifecycle.CANDIDATE:
            if slot.observed_count < 3:
                raise ValueError("slot needs at least 3 observations before candidate promotion")
            slot.lifecycle = SlotLifecycle.CANDIDATE
            slot.candidate_count += 1
            return slot

        if target == SlotLifecycle.EXPECTED:
            if slot.observed_count < 5:
                raise ValueError("slot needs at least 5 observations before expected promotion")
            slot.lifecycle = SlotLifecycle.EXPECTED
            slot.expected_count += 1
            return slot

        if target == SlotLifecycle.RETIRED:
            slot.lifecycle = SlotLifecycle.RETIRED
            slot.retired_at = datetime.now(timezone.utc)
            return slot

        slot.lifecycle = target
        return slot

    def add_resolution_case(self, case: ResolutionCase) -> ResolutionCase:
        if not case.id:
            case.id = uuid4().hex
        self.resolution_cases[case.id] = case
        return case

    def get_resolution_case(self, case_id: str) -> ResolutionCase:
        return self.resolution_cases[case_id]

    def find_similar_case(self, conflict_signature: str, threshold: float = 0.8) -> ResolutionCase | None:
        best_case: ResolutionCase | None = None
        best_score = threshold
        for case in self.resolution_cases.values():
            score = text_similarity(case.conflict_signature, conflict_signature)
            if score >= best_score:
                best_case = case
                best_score = score
        return best_case

    def snapshot(self) -> dict[str, int]:
        return {
            "entities": len(self.entities),
            "claims": len(self.claims),
            "canonical_claims": len(
                [
                    claim
                    for claim in self.claims.values()
                    if claim.epistemic_status == EpistemicStatus.CONFIRMED
                ]
            ),
            "evidence": len(self.evidence),
            "slots": len(self.slots),
            "resolution_cases": len(self.resolution_cases),
        }
