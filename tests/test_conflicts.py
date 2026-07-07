"""Tests for keyword opposition in conflict detection."""

from __future__ import annotations

import pytest

from knowledge_engine.conflicts import ConflictDetector, _has_keyword_opposition
from knowledge_engine.models import Claim, EpistemicStatus


class TestKeywordOpposition:
    """Verify semantic opposition detection."""

    def test_diversification_vs_concentration(self):
        assert _has_keyword_opposition(
            "Diversification reduces portfolio risk",
            "Concentration outperforms diversified portfolios",
        )

    def test_increase_vs_decrease(self):
        assert _has_keyword_opposition(
            "Interest rates will increase next quarter",
            "Interest rates will decrease next quarter",
        )

    def test_always_vs_never(self):
        assert _has_keyword_opposition(
            "Momentum always works in trending markets",
            "Momentum never works in ranging markets",
        )

    def test_bullish_vs_bearish(self):
        assert _has_keyword_opposition(
            "The outlook is bullish for tech stocks",
            "The outlook is bearish for tech stocks",
        )

    def test_no_opposition(self):
        assert not _has_keyword_opposition(
            "Diversification reduces portfolio risk",
            "Diversification improves risk-adjusted returns",
        )

    def test_different_domains_no_opposition(self):
        assert not _has_keyword_opposition(
            "Diversification is good for stocks",
            "Real estate markets are growing",
        )


class TestConflictDetectorWithOpposition:
    """Verify conflict detection catches semantic opposition."""

    def _make_claim(self, statement: str, slot: str = "strategy") -> Claim:
        return Claim(
            id="test-claim",
            entity_id="test-entity",
            statement=statement,
            slot_name=slot,
            epistemic_status=EpistemicStatus.CONFIRMED,
        )

    def test_opposition_detected_despite_low_similarity(self):
        detector = ConflictDetector()
        existing = self._make_claim(
            "Diversification reduces portfolio risk by spreading across asset classes"
        )
        incoming = self._make_claim(
            "Concentration outperforms because diversification dilutes returns"
        )
        matches = detector.detect([existing], incoming)
        assert len(matches) == 1
        assert "semantic opposition" in matches[0].message

    def test_paraphrase_still_detected(self):
        detector = ConflictDetector()
        existing = self._make_claim(
            "Momentum strategies work best in trending markets with ADX above 25"
        )
        incoming = self._make_claim(
            "Momentum strategies perform optimally in trending markets when ADX exceeds 25"
        )
        matches = detector.detect([existing], incoming)
        assert len(matches) == 1

    def test_different_slots_no_conflict(self):
        detector = ConflictDetector()
        existing = self._make_claim("Diversification reduces risk", slot="risk")
        incoming = self._make_claim("Concentration outperforms", slot="returns")
        matches = detector.detect([existing], incoming)
        assert len(matches) == 0

    def test_exact_match_skipped(self):
        detector = ConflictDetector()
        existing = self._make_claim("Diversification reduces risk")
        incoming = self._make_claim("Diversification reduces risk")
        matches = detector.detect([existing], incoming)
        assert len(matches) == 0
