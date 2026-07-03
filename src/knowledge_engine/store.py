from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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
        # Keyed by "{entity_id}:{slot_name_lower}" — persists suggestions across
        # ingest calls so they accumulate rather than being silently dropped.
        self.pending_promotions: dict[str, dict] = {}

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

    def set_claim_status(self, *, claim_id: str, status: str) -> None:
        """No-op: JSON store mutates the Claim object in-place via dict reference."""

    def set_claim_embedding(self, *, claim_id: str, embedding: list[float]) -> None:
        """No-op: JSON store mutates the Claim object in-place via dict reference."""

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

    def queue_slot_promotion(
        self,
        entity_id: str,
        entity_name: str,
        slot_name: str,
        current_lifecycle: str,
        suggested_lifecycle: str,
        observed_count: int,
        reason: str,
    ) -> None:
        """Persist a slot promotion suggestion so it survives across calls."""
        key = f"{entity_id}:{slot_name.strip().lower()}"
        self.pending_promotions[key] = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "slot_name": slot_name,
            "current_lifecycle": current_lifecycle,
            "suggested_lifecycle": suggested_lifecycle,
            "observed_count": observed_count,
            "reason": reason,
        }

    def list_pending_slot_promotions(self) -> list[dict]:
        """Return all pending slot promotion candidates."""
        return list(self.pending_promotions.values())

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
        elif target == SlotLifecycle.EXPECTED:
            if slot.observed_count < 5:
                raise ValueError("slot needs at least 5 observations before expected promotion")
            slot.lifecycle = SlotLifecycle.EXPECTED
            slot.expected_count += 1
        elif target == SlotLifecycle.RETIRED:
            slot.lifecycle = SlotLifecycle.RETIRED
            slot.retired_at = datetime.now(timezone.utc)
        else:
            slot.lifecycle = target

        # Clear from the promotion queue once the user has acted.
        key = f"{entity_id}:{slot_name.strip().lower()}"
        self.pending_promotions.pop(key, None)
        return slot

    def add_resolution_case(self, case: ResolutionCase) -> ResolutionCase:
        if not case.id:
            case.id = uuid4().hex
        self.resolution_cases[case.id] = case
        return case

    def get_resolution_case(self, case_id: str) -> ResolutionCase:
        return self.resolution_cases[case_id]

    def list_open_resolution_cases(self) -> list[ResolutionCase]:
        return [c for c in self.resolution_cases.values() if c.is_open]

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

    # -- persistence -----------------------------------------------------------
    # JSON round-trip for the CLI/UAT path so an ingested corpus survives across
    # separate `ingest` and `ask` invocations. This is local UAT persistence,
    # NOT the production graph store.
    def to_dict(self) -> dict:
        return {
            "entities": {k: v.model_dump(mode="json") for k, v in self.entities.items()},
            "claims": {k: v.model_dump(mode="json") for k, v in self.claims.items()},
            "evidence": {k: v.model_dump(mode="json") for k, v in self.evidence.items()},
            "slots": {k: v.model_dump(mode="json") for k, v in self.slots.items()},
            "resolution_cases": {
                k: v.model_dump(mode="json") for k, v in self.resolution_cases.items()
            },
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeStore":
        store = cls()
        source = Path(path)
        if not source.is_file():
            return store
        data = json.loads(source.read_text(encoding="utf-8"))
        store.entities = {k: Entity.model_validate(v) for k, v in data.get("entities", {}).items()}
        store.claims = {k: Claim.model_validate(v) for k, v in data.get("claims", {}).items()}
        store.evidence = {k: Evidence.model_validate(v) for k, v in data.get("evidence", {}).items()}
        store.slots = {k: Slot.model_validate(v) for k, v in data.get("slots", {}).items()}
        store.resolution_cases = {
            k: ResolutionCase.model_validate(v)
            for k, v in data.get("resolution_cases", {}).items()
        }
        # Rebuild the (entity_id, slot_name) -> slot_id index from loaded slots.
        store._slot_index = {
            (slot.entity_id, slot.name.strip().lower()): slot.id
            for slot in store.slots.values()
            if slot.id
        }
        return store
