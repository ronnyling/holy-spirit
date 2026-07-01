from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EpistemicStatus(StrEnum):
    CONFIRMED = "Confirmed"
    UNVERIFIED = "Unverified"
    UNKNOWN = "Unknown"
    UNVERIFIABLE = "Unverifiable"
    DISPUTED = "Disputed"
    RETRACTED = "Retracted"


class SlotLifecycle(StrEnum):
    OBSERVED = "Observed"
    CANDIDATE = "Candidate"
    EXPECTED = "Expected"
    RETIRED = "Retired"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_kind: Literal["internal_wiki", "external_doc", "user"]
    source_id: str
    source_ref: str | None = None
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = None


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    version: int = 1


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    entity_id: str
    statement: str
    slot_name: str | None = None
    epistemic_status: EpistemicStatus = EpistemicStatus.UNVERIFIED
    provenance: list[Provenance] = Field(default_factory=list)
    embedding: list[float] | None = None
    version: int = 1
    tags: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    claim_id: str
    source_kind: Literal["internal_wiki", "external_doc", "user"]
    source_id: str
    source_ref: str | None = None
    credibility: float = Field(ge=0.0, le=1.0)
    notes: str | None = None
    linked_claim_ids: list[str] = Field(default_factory=list)


class ResolutionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    conflict_signature: str
    conflicting_claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    research_notes: str | None = None
    decision: str | None = None
    rationale: str | None = None
    version: int = 1
    reopened_from_case_id: str | None = None
    is_open: bool = True


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    entity_id: str
    name: str
    description: str | None = None
    lifecycle: SlotLifecycle = SlotLifecycle.OBSERVED
    observed_count: int = 0
    candidate_count: int = 0
    expected_count: int = 0
    retired_at: datetime | None = None
    last_observed_at: datetime | None = None
    version: int = 1
