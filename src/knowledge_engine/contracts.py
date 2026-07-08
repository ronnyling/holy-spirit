from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GapKind(StrEnum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    LOGICAL = "logical"


class EvidenceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_kind: Literal["internal_wiki", "external_doc", "user"]
    source_id: str
    source_ref: str | None = None
    credibility: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str | None = None
    linked_claim_ids: list[str] = Field(default_factory=list)


class ClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    slot_name: str | None = None
    observed_slots: list[str] = Field(default_factory=list)
    evidence: list[EvidenceDraft] = Field(default_factory=list)
    notes: str | None = None


class TranscriptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = ""
    entity_name: str = ""
    transcript_text: str
    source_kind: Literal["internal_wiki", "external_doc", "user"] = "external_doc"
    source_id: str
    source_ref: str | None = None
    evidence_credibility: float = Field(default=0.5, ge=0.0, le=1.0)
    claim_drafts: list[ClaimDraft] = Field(default_factory=list)
    notes: str | None = None


class SlotSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_name: str
    current_lifecycle: str
    suggested_lifecycle: str
    observed_count: int
    reason: str


class GapFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: GapKind
    entity_id: str
    slot_name: str
    question: str
    severity: Literal["low", "medium", "high"] = "high"
    rationale: str | None = None


class ConflictSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict_signature: str
    incoming_claim_id: str
    existing_claim_ids: list[str] = Field(default_factory=list)
    case_id: str | None = None
    message: str


class TranscriptOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    transcript_id: str | None = None
    transcript_created: bool | None = None
    chunk_count: int | None = None
    claim_ids: list[str] = Field(default_factory=list)
    confirmed_claim_ids: list[str] = Field(default_factory=list)
    unverified_claim_ids: list[str] = Field(default_factory=list)
    disputed_claim_ids: list[str] = Field(default_factory=list)
    slot_suggestions: list[SlotSuggestion] = Field(default_factory=list)
    gap_flags: list[GapFlag] = Field(default_factory=list)
    conflict_summaries: list[ConflictSummary] = Field(default_factory=list)
    open_case_ids: list[str] = Field(default_factory=list)
    canonical_claim_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Proactive conflict support: when conflicts are detected, the system
    # gathers evidence and generates a prompt for the user.
    conflict_prompt: str | None = None
    gathered_evidence: list[dict] | None = None


class ExperienceResponse(BaseModel):
    """Result of explore_experience(): world knowledge discerned through system experience."""

    model_config = ConfigDict(extra="forbid")

    query: str
    domain: str | None = None
    world_knowledge: str
    experience_claims: list[dict] = Field(default_factory=list)
    synthesis: str
    confirmed_count: int = 0
    unverified_count: int = 0
    disputed_count: int = 0
    experience_available: bool = True
