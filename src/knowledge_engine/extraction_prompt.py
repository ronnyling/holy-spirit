def build_enhanced_extraction_prompt(transcript_text: str) -> str:
    """Build LLM prompt for enhanced claim + evidence extraction."""
    return f"""Extract claims AND supporting evidence from this transcript.

For each claim, also extract:
- Evidence statements that support it
- Source quality (academic/anecdotal/commercial/unknown)
- Conditions under which the claim applies
- How the measurement was obtained
- Confidence level (high/medium/low/uncertain)

Return JSON:
{{
  "claims": [
    {{
      "statement": "claim statement",
      "slot_name": "slot_name",
      "evidence": [
        {{
          "statement": "evidence statement",
          "source_quality": "academic",
          "conditions": ["condition1", "condition2"],
          "measurement_method": "randomized controlled trial",
          "confidence_indicator": "medium"
        }}
      ]
    }}
  ]
}}

Transcript:
{transcript_text}
"""


EVIDENCE_SYSTEM_PROMPT = """You extract evidence from transcripts to support existing claims.

Rules:
1. Extract ONLY evidence explicitly stated in the transcript
2. Each evidence must directly support the given claim
3. Include source quality assessment (academic/anecdotal/commercial/unknown)
4. Include conditions under which the evidence applies
5. Include measurement methodology if available
6. Include confidence indicator based on evidence strength
7. Do NOT fabricate evidence - if no supporting evidence exists, return empty array

Return STRICT JSON only: a JSON array of evidence objects with keys:
- "statement" (string): The evidence statement
- "source_quality" (string): academic/anecdotal/commercial/unknown
- "conditions" (array of strings): When this evidence applies
- "measurement_method" (string): How the measurement was obtained
- "confidence_indicator" (string): high/medium/low/uncertain
- "has_quantification" (boolean): Contains numbers/percentages
- "has_time_period" (boolean): Specifies duration
- "has_sample_size" (boolean): Mentions participants
- "has_primary_source" (boolean): References original study

No prose, no markdown fences."""


def build_evidence_extraction_prompt(claim_statement: str, transcript_text: str) -> str:
    """Build prompt to extract evidence for a specific claim."""
    return f"""Extract evidence from this transcript that supports the given claim.

Claim: {claim_statement}

Rules:
- Extract ONLY evidence explicitly stated in the transcript
- Evidence must directly support the claim
- If no supporting evidence exists, return an empty array
- Include source quality, conditions, and methodology when available

Transcript:
{transcript_text}

Return STRICT JSON array of evidence objects."""
