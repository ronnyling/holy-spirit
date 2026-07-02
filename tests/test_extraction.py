"""Tests for LLM-backed claim extraction.

The LLM is the UNBOUNDED layer, so it is faked here: a stub client returns a
fixed string and we assert the BOUNDED parsing/validation and the engine wiring.
No network calls.
"""

from __future__ import annotations

from knowledge_engine import (
    ClaimExtractor,
    EpistemicStatus,
    KnowledgeEngine,
    TranscriptInput,
)


class StubClient:
    """Stands in for MiMoClient. Returns whatever string it is given."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete_sync(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


def test_parse_clean_json_array():
    stub = StubClient(
        '[{"statement": "A 6% cap rate suits a stable suburban asset.", '
        '"observed_slots": ["cap_rate"]}, '
        '{"statement": "Financing assumptions must hold across one cycle.", '
        '"observed_slots": []}]'
    )
    extractor = ClaimExtractor(stub)
    drafts = extractor.extract(
        domain="real estate",
        entity_name="Cap Rate Rules",
        transcript_text="some transcript text",
    )
    assert len(drafts) == 2
    assert drafts[0].statement.startswith("A 6% cap rate")
    assert drafts[0].observed_slots == ["cap_rate"]
    assert drafts[0].slot_name == "cap_rate"
    # Extraction never fabricates evidence.
    assert drafts[0].evidence == []
    assert drafts[1].observed_slots == []
    assert drafts[1].slot_name is None


def test_parse_tolerates_code_fences_and_prose():
    stub = StubClient(
        'Here are the claims:\n```json\n'
        '[{"statement": "Qi stagnation causes rib-side distension.", '
        '"observed_slots": ["pattern"]}]\n```\nDone.'
    )
    drafts = ClaimExtractor(stub).extract(
        domain="tcm", entity_name="Liver Qi", transcript_text="t"
    )
    assert len(drafts) == 1
    assert drafts[0].observed_slots == ["pattern"]


def test_malformed_output_yields_no_claims():
    # Bounded layer refuses to guess: unparseable -> empty, never fabricated.
    for bad in ["not json at all", "", "{}", '{"statement": "x"}', "[1, 2, 3]"]:
        drafts = ClaimExtractor(StubClient(bad)).extract(
            domain="trading", entity_name="X", transcript_text="t"
        )
        assert drafts == [], bad


def test_empty_transcript_skips_llm():
    stub = StubClient("[]")
    drafts = ClaimExtractor(stub).extract(
        domain="trading", entity_name="X", transcript_text="   "
    )
    assert drafts == []
    assert stub.calls == []  # LLM not called for empty text


def test_engine_uses_extractor_when_no_drafts():
    stub = StubClient(
        '[{"statement": "Buy-and-hold rental is a core strategy.", '
        '"observed_slots": ["strategy"]}]'
    )
    engine = KnowledgeEngine(extractor=ClaimExtractor(stub))
    outcome = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name="Malaysia Residential Strategy",
            transcript_text="Expert says buy-and-hold rental is a core strategy.",
            source_kind="external_doc",
            source_id="doc-1",
        )
    )
    assert len(outcome.claim_ids) == 1
    # No evidence extracted -> must be UNVERIFIED, not auto-confirmed.
    assert outcome.confirmed_claim_ids == []
    assert len(outcome.unverified_claim_ids) == 1
    assert any("LLM extracted" in n for n in outcome.notes)
    assert len(stub.calls) == 1


def test_hand_authored_drafts_take_precedence_over_extractor():
    stub = StubClient('[{"statement": "should not be used", "observed_slots": []}]')
    engine = KnowledgeEngine(extractor=ClaimExtractor(stub))
    from knowledge_engine import ClaimDraft

    outcome = engine.ingest_transcript(
        TranscriptInput(
            domain="trading",
            entity_name="Breakout",
            transcript_text="raw text",
            source_kind="user",
            source_id="u-1",
            claim_drafts=[ClaimDraft(statement="Manual claim.", observed_slots=[])],
        )
    )
    assert len(outcome.claim_ids) == 1
    assert stub.calls == []  # extractor NOT called when drafts provided


def test_engine_without_extractor_is_unchanged():
    # Default path: no extractor, no drafts -> zero claims (today's behavior).
    engine = KnowledgeEngine()
    outcome = engine.ingest_transcript(
        TranscriptInput(
            domain="trading",
            entity_name="Breakout",
            transcript_text="lots of text but no drafts",
            source_kind="external_doc",
            source_id="doc-1",
        )
    )
    assert outcome.claim_ids == []
