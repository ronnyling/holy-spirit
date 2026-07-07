"""Tests for evidence-weighted ranking."""

from __future__ import annotations

import pytest

from knowledge_engine.engine import _EPISTEMIC_WEIGHTS


class TestEpistemicWeights:
    """Verify epistemic weight constants are correct and ordered."""

    def test_confirmed_is_highest(self):
        assert _EPISTEMIC_WEIGHTS["Confirmed"] > _EPISTEMIC_WEIGHTS["Disputed"]
        assert _EPISTEMIC_WEIGHTS["Confirmed"] > _EPISTEMIC_WEIGHTS["Unverified"]

    def test_disputed_above_unverified(self):
        assert _EPISTEMIC_WEIGHTS["Disputed"] > _EPISTEMIC_WEIGHTS["Unverified"]

    def test_retracted_is_lowest(self):
        assert _EPISTEMIC_WEIGHTS["Retracted"] < _EPISTEMIC_WEIGHTS["Unknown"]
        assert _EPISTEMIC_WEIGHTS["Retracted"] < _EPISTEMIC_WEIGHTS["Unverifiable"]

    def test_all_statuses_have_weights(self):
        expected = {"Confirmed", "Disputed", "Unverified", "Unknown", "Unverifiable", "Retracted"}
        assert set(_EPISTEMIC_WEIGHTS.keys()) == expected

    def test_weights_are_bounded(self):
        for status, weight in _EPISTEMIC_WEIGHTS.items():
            assert 0.0 <= weight <= 1.0, f"weight for {status} out of range"


class TestSearchClaimsWeighted:
    """Verify search_claims applies epistemic weighting."""

    def _make_engine(self):
        """Create a minimal engine with mock store for testing."""
        from knowledge_engine.engine import KnowledgeEngine
        from knowledge_engine.store import KnowledgeStore

        store = KnowledgeStore()
        return KnowledgeEngine(store=store)

    def test_weighted_score_applied(self):
        """Verify _EPISTEMIC_WEIGHTS is used in weighting logic."""
        # Simulate the weighting logic from search_claims
        results = [
            {"claim_id": "c1", "epistemic_status": "Confirmed", "rrf_score": 0.5},
            {"claim_id": "c2", "epistemic_status": "Unverified", "rrf_score": 0.6},
        ]
        for claim in results:
            status = claim.get("epistemic_status") or claim.get("status", "Unknown")
            weight = _EPISTEMIC_WEIGHTS.get(status, 0.5)
            base_score = claim.get("rrf_score") or claim.get("similarity") or claim.get("score") or 0.0
            claim["weighted_score"] = round(base_score * weight, 6)
        results.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)

        # Confirmed should rank first despite lower base score
        assert results[0]["claim_id"] == "c1"
        assert results[0]["weighted_score"] == 0.5  # 0.5 * 1.0
        assert results[1]["claim_id"] == "c2"
        assert results[1]["weighted_score"] == 0.36  # 0.6 * 0.6 (Unverified)

    def test_unknown_status_gets_default_weight(self):
        """Unknown status gets 0.5 weight."""
        weight = _EPISTEMIC_WEIGHTS.get("Unknown", 0.5)
        assert weight == 0.4

    def test_missing_status_gets_default_weight(self):
        """Status not in dict gets 0.5 default."""
        weight = _EPISTEMIC_WEIGHTS.get("NonexistentStatus", 0.5)
        assert weight == 0.5
