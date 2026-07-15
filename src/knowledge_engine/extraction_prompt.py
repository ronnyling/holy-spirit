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
