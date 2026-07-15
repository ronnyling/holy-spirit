"""Integration module for enriching claims with transcript evidence.

NO FALLBACKS: This module must either succeed completely or fail loudly.
Data integrity is paramount for traceability and replicability in the
knowledge engine's white paper.

Uses LLM config system to get appropriate client:
- HIGH priority (evidence extraction): MiMo API
- Falls back to Ollama if MiMo not available (with warning)
"""
import json
import re
from typing import List, Protocol, runtime_checkable

from .contracts import ClaimDraft, EvidenceDraft
from .transcript_evidence import TranscriptEvidenceDraft
from .extraction_prompt import EVIDENCE_SYSTEM_PROMPT, build_evidence_extraction_prompt
from .llm_config import TaskType, get_llm_client


class EvidenceExtractionError(Exception):
    """NO FALLBACKS: Raised when evidence extraction fails. Break fast, break loud.

    Data integrity is paramount for traceability, replicability, and the white paper.
    Never silently degrade or return partial results.
    """
    pass


@runtime_checkable
class SupportsComplete(Protocol):
    def complete_sync(self, *, system: str, user: str, max_tokens: int = 4000) -> str: ...


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_array(raw: str) -> list | None:
    """Pull a JSON array out of model output, tolerating code fences/prose."""
    if not raw:
        return None
    candidate = raw.strip()

    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    # If there is surrounding prose, isolate the outermost [ ... ].
    if not candidate.startswith("["):
        start = candidate.find("[")
        if start == -1:
            return None
        candidate = candidate[start:]

    if not candidate.endswith("]"):
        end = candidate.rfind("]")
        if end == -1:
            return None
        candidate = candidate[:end + 1]

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


def _parse_evidence_response(raw: str) -> List[TranscriptEvidenceDraft]:
    """Parse LLM response into TranscriptEvidenceDraft objects.

    NO FALLBACKS: Raises EvidenceExtractionError on parse failure.
    """
    payload = _extract_json_array(raw)
    if payload is None:
        raise EvidenceExtractionError(
            f"Failed to parse evidence response as JSON array. "
            f"Raw response: {raw[:500]}"
        )

    evidence_drafts = []
    for item in payload:
        if not isinstance(item, dict):
            raise EvidenceExtractionError(
                f"Invalid evidence item type: {type(item)}. Expected dict."
            )

        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise EvidenceExtractionError(
                f"Evidence missing required 'statement' field. Item: {item}"
            )

        # Validate source_quality
        source_quality = item.get("source_quality", "unknown")
        valid_qualities = {"academic", "anecdotal", "commercial", "unknown"}
        if source_quality not in valid_qualities:
            raise EvidenceExtractionError(
                f"Invalid source_quality: {source_quality}. "
                f"Must be one of: {valid_qualities}"
            )

        # Validate confidence_indicator
        confidence_indicator = item.get("confidence_indicator", "medium")
        valid_indicators = {"high", "medium", "low", "uncertain"}
        if confidence_indicator not in valid_indicators:
            raise EvidenceExtractionError(
                f"Invalid confidence_indicator: {confidence_indicator}. "
                f"Must be one of: {valid_indicators}"
            )

        evidence_drafts.append(TranscriptEvidenceDraft.from_llm_output(item))

    return evidence_drafts


def _validate_evidence_for_claim(
    evidence_drafts: List[TranscriptEvidenceDraft],
    claim: ClaimDraft
) -> None:
    """Validate evidence is appropriate for the claim.

    NO FALLBACKS: Raises EvidenceExtractionError on validation failure.
    """
    if not evidence_drafts:
        # Empty evidence is valid - claim may not have supporting evidence
        return

    for ev in evidence_drafts:
        # Validate confidence score is reasonable
        confidence = ev.calculate_confidence()
        if confidence < 0.0 or confidence > 1.0:
            raise EvidenceExtractionError(
                f"Invalid confidence score: {confidence}. Must be between 0.0 and 1.0."
            )

        # Validate source_reference exists
        if not ev.source_reference:
            raise EvidenceExtractionError(
                f"Evidence missing source_reference. Statement: {ev.statement}"
            )


def enrich_claims_with_evidence(
    claims: List[ClaimDraft],
    transcript_text: str,
    llm_client: SupportsComplete | None = None
) -> List[ClaimDraft]:
    """Enrich claims with evidence extracted from transcript.

    NO FALLBACKS: Raises EvidenceExtractionError on any failure.
    Data integrity is paramount.

    Uses LLM config system:
    - If llm_client provided, use it directly
    - Otherwise, get client from config (HIGH priority → MiMo API)

    Args:
        claims: Claims to enrich with evidence
        transcript_text: Original transcript text
        llm_client: LLM client for evidence extraction (optional, uses config if None)

    Returns:
        Claims enriched with evidence

    Raises:
        EvidenceExtractionError: If evidence extraction fails
    """
    if not claims:
        return claims

    # Get LLM client from config if not provided
    if llm_client is None:
        try:
            llm_client = get_llm_client(TaskType.EVIDENCE_EXTRACTION)
        except RuntimeError as e:
            raise EvidenceExtractionError(
                f"LLM client required for evidence extraction. {e}"
            )

    enriched_claims = []

    for claim in claims:
        # Build evidence extraction prompt
        prompt = build_evidence_extraction_prompt(
            claim.statement,
            transcript_text
        )

        # Call LLM
        try:
            raw_response = llm_client.complete_sync(
                system=EVIDENCE_SYSTEM_PROMPT,
                user=prompt,
                max_tokens=4000
            )
        except Exception as e:
            raise EvidenceExtractionError(
                f"LLM call failed for claim '{claim.statement}': {e}"
            )

        # Parse response - NO FALLBACKS
        evidence_drafts = _parse_evidence_response(raw_response)

        # Validate evidence - NO FALLBACKS
        _validate_evidence_for_claim(evidence_drafts, claim)

        # Convert TranscriptEvidenceDraft to EvidenceDraft for storage
        evidence_objects = []
        for ev_draft in evidence_drafts:
            # Map source_quality to credibility score
            quality_scores = {
                "academic": 0.9,
                "commercial": 0.5,
                "anecdotal": 0.3,
                "unknown": 0.2
            }
            credibility = quality_scores.get(ev_draft.source_quality, 0.5)

            # Adjust credibility based on confidence
            confidence_multipliers = {
                "high": 1.0,
                "medium": 0.8,
                "low": 0.5,
                "uncertain": 0.3
            }
            credibility *= confidence_multipliers.get(ev_draft.confidence_indicator, 0.5)

            evidence_obj = EvidenceDraft(
                source_kind="external_doc",
                source_id=f"transcript_evidence_{len(evidence_objects)}",
                source_ref=ev_draft.source_reference,
                credibility=credibility,
                notes=f"Extracted by LLM: {ev_draft.statement}"
            )
            evidence_objects.append(evidence_obj)

        # Create enriched claim
        enriched_claim = ClaimDraft(
            statement=claim.statement,
            slot_name=claim.slot_name,
            observed_slots=claim.observed_slots,
            evidence=evidence_objects,
            notes=claim.notes
        )
        enriched_claims.append(enriched_claim)

    return enriched_claims
