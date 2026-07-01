from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .models import Slot, SlotLifecycle
from .store import KnowledgeStore


class SlotObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: Slot
    suggested_lifecycle: SlotLifecycle | None = None
    reason: str | None = None


class SlotLearner:
    def __init__(self, candidate_threshold: int = 3, expected_threshold: int = 5) -> None:
        self.candidate_threshold = candidate_threshold
        self.expected_threshold = expected_threshold

    def observe(self, store: KnowledgeStore, entity_id: str, slot_name: str, description: str | None = None) -> SlotObservation:
        slot = store.observe_slot(entity_id, slot_name, description=description)

        if slot.lifecycle == SlotLifecycle.RETIRED:
            return SlotObservation(slot=slot, reason="slot is retired")

        if slot.lifecycle == SlotLifecycle.OBSERVED and slot.observed_count >= self.candidate_threshold:
            return SlotObservation(
                slot=slot,
                suggested_lifecycle=SlotLifecycle.CANDIDATE,
                reason="slot has crossed the candidate threshold",
            )

        if slot.lifecycle == SlotLifecycle.CANDIDATE and slot.observed_count >= self.expected_threshold:
            return SlotObservation(
                slot=slot,
                suggested_lifecycle=SlotLifecycle.EXPECTED,
                reason="slot has crossed the expected threshold",
            )

        return SlotObservation(slot=slot)
