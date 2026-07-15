"""Integration module for enriching claims with transcript evidence.

NO FALLBACKS: This module must either succeed completely or fail loudly.
Data integrity is paramount for traceability and replicability in the
knowledge engine's white paper.
"""
from typing import List
from .contracts import ClaimDraft
from .transcript_evidence import TranscriptEvidenceDraft


class EvidenceExtractionError(Exception):
    """NO FALLBACKS: Raised when evidence extraction fails. Break fast, break loud.

    Data integrity is paramount for traceability, replicability, and the white paper.
    Never silently degrade or return partial results.
    """
    pass


def enrich_claims_with_evidence(
    claims: List[ClaimDraft],
    transcript_text: str,
    llm_client=None
) -> List[ClaimDraft]:
    """Enrich claims with evidence extracted from transcript.

    NO FALLBACKS: Raises EvidenceExtractionError on any failure.
    Data integrity is paramount.

    Args:
        claims: Claims to enrich with evidence
        transcript_text: Original transcript text
        llm_client: LLM client for evidence extraction (required for enrichment)

    Returns:
        Claims enriched with evidence

    Raises:
        EvidenceExtractionError: If evidence extraction fails
    """
    if not claims:
        return claims

    if llm_client is None:
        raise EvidenceExtractionError(
            "LLM client required for evidence extraction. "
            "No fallback to empty evidence allowed - "
            "this would compromise data integrity."
        )

    # Future implementation:
    # 1. For each claim, build evidence extraction prompt
    # 2. Call LLM to extract evidence
    # 3. Parse LLM output into TranscriptEvidenceDraft objects
    # 4. Link evidence to claims
    # 5. Return enriched claims

    raise EvidenceExtractionError(
        "Evidence extraction not yet implemented. "
        "No silent fallback allowed - must either extract evidence "
        "or raise an error to maintain data integrity."
    )
