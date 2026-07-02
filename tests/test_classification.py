"""Tests for LLM-backed domain classification.

The LLM is the UNBOUNDED layer, so it is stubbed. We assert the BOUNDED mapping:
only the fixed known domains are ever returned, and anything else -> None
(UNKNOWN), which the caller flags for a human instead of guessing. Domain drives
the per-domain evidence gate, so a wrong guess must never happen silently.
"""

from __future__ import annotations

from knowledge_engine import DomainClassifier


class StubClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete_sync(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


def test_classifies_exact_labels():
    for label in ("trading", "real estate", "tcm"):
        result = DomainClassifier(StubClient(label)).classify(transcript_text="some text")
        assert result == label


def test_tolerates_surrounding_text_and_case():
    result = DomainClassifier(StubClient("This is TRADING.")).classify(transcript_text="t")
    assert result == "trading"


def test_real_estate_not_shadowed_by_trading():
    result = DomainClassifier(StubClient("real estate")).classify(transcript_text="t")
    assert result == "real estate"


def test_known_aliases_map_to_canonical():
    cases = {
        "traditional chinese medicine": "tcm",
        "chinese medicine": "tcm",
        "property investing": "real estate",
        "stock trading": "trading",
    }
    for raw, expected in cases.items():
        assert DomainClassifier(StubClient(raw)).classify(transcript_text="t") == expected


def test_unknown_returns_none_never_guesses():
    for raw in ("UNKNOWN", "crypto", "cooking", "", "   ", "42"):
        assert DomainClassifier(StubClient(raw)).classify(transcript_text="t") is None


def test_empty_transcript_skips_llm():
    stub = StubClient("trading")
    assert stub.calls == []
    assert DomainClassifier(stub).classify(transcript_text="   ") is None
    assert stub.calls == []  # no LLM call for empty input


def test_only_a_sample_is_sent():
    stub = StubClient("trading")
    long_text = "word " * 5000  # ~25k chars
    DomainClassifier(stub).classify(transcript_text=long_text)
    # Classification samples the head, not the whole document.
    assert len(stub.calls) == 1
    assert len(stub.calls[0]["user"]) < 3000
