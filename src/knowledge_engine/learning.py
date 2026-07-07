from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .models import Slot, SlotLifecycle
from .store import KnowledgeStore


class SlotObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: Slot
    suggested_lifecycle: SlotLifecycle | None = None
    auto_promoted: bool = False
    reason: str | None = None


class SlotLearner:
    def __init__(
        self,
        candidate_threshold: int = 3,
        expected_threshold: int = 5,
        auto_promote_threshold: int = 10,
    ) -> None:
        self.candidate_threshold = candidate_threshold
        self.expected_threshold = expected_threshold
        self.auto_promote_threshold = auto_promote_threshold

    def observe(self, store: KnowledgeStore, entity_id: str, slot_name: str, description: str | None = None) -> SlotObservation:
        slot = store.observe_slot(entity_id, slot_name, description=description)

        if slot.lifecycle == SlotLifecycle.RETIRED:
            return SlotObservation(slot=slot, reason="slot is retired")

        # Auto-promote to Expected at 10+ observations — high-confidence pattern.
        # No human confirmation needed; the evidence of repetition is sufficient.
        if slot.observed_count >= self.auto_promote_threshold:
            if slot.lifecycle in (SlotLifecycle.OBSERVED, SlotLifecycle.CANDIDATE):
                store.confirm_slot(
                    entity_id, slot_name,
                    target=SlotLifecycle.EXPECTED,
                    confirmed_by="auto-promote",
                )
                # Re-read the slot after promotion
                promoted = store.get_slot(entity_id, slot_name)
                if promoted:
                    return SlotObservation(
                        slot=promoted,
                        suggested_lifecycle=SlotLifecycle.EXPECTED,
                        auto_promoted=True,
                        reason=f"auto-promoted: {promoted.observed_count} observations "
                               f"(threshold: {self.auto_promote_threshold})",
                    )

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
