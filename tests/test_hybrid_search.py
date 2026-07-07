"""Tests for hybrid search (vector + full-text RRF merge)."""

from __future__ import annotations

import pytest

from knowledge_engine.graph.neo4j_store import _rrf_merge


class TestRrfMerge:
    """Unit tests for Reciprocal Rank Fusion merge logic."""

    def test_empty_inputs(self):
        result = _rrf_merge([], [], alpha=0.7, limit=10)
        assert result == []

    def test_vector_only(self):
        rows = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.9},
            {"id": "c2", "statement": "beta", "status": "Unverified",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.7},
        ]
        result = _rrf_merge(rows, [], alpha=0.7, limit=10)
        assert len(result) == 2
        assert result[0]["claim_id"] == "c1"
        assert result[1]["claim_id"] == "c2"
        assert result[0]["source"] == "vector"
        assert result[0]["vector_score"] == 0.9
        assert result[0]["ft_score"] is None

    def test_fulltext_only(self):
        rows = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 5.0},
            {"id": "c2", "statement": "beta", "status": "Unverified",
             "tags": [], "slot_name": None, "version": 1, "result_score": 3.0},
        ]
        result = _rrf_merge([], rows, alpha=0.7, limit=10)
        assert len(result) == 2
        assert result[0]["claim_id"] == "c1"
        assert result[0]["source"] == "fulltext"
        assert result[0]["vector_score"] is None
        assert result[0]["ft_score"] == 5.0

    def test_hybrid_merge_prefers_vector_top(self):
        """When a claim appears in both lists, it should combine scores."""
        vector = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.9},
        ]
        ft = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 5.0},
            {"id": "c2", "statement": "beta", "status": "Unverified",
             "tags": [], "slot_name": None, "version": 1, "result_score": 3.0},
        ]
        result = _rrf_merge(vector, ft, alpha=0.7, limit=10)
        assert len(result) == 2
        # c1 appears in both — combined score should be highest
        assert result[0]["claim_id"] == "c1"
        assert result[0]["source"] == "hybrid"
        assert result[0]["ft_score"] == 5.0
        assert result[0]["vector_score"] == 0.9

    def test_limit_respected(self):
        vector = [
            {"id": f"c{i}", "statement": f"claim {i}", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.9 - i * 0.1}
            for i in range(5)
        ]
        result = _rrf_merge(vector, [], alpha=0.7, limit=3)
        assert len(result) == 3

    def test_alpha_zero_is_pure_fulltext(self):
        vector = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.9},
        ]
        ft = [
            {"id": "c2", "statement": "beta", "status": "Unverified",
             "tags": [], "slot_name": None, "version": 1, "result_score": 5.0},
        ]
        result = _rrf_merge(vector, ft, alpha=0.0, limit=10)
        # With alpha=0, fulltext rank determines order
        assert result[0]["claim_id"] == "c2"

    def test_alpha_one_is_pure_vector(self):
        vector = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.9},
        ]
        ft = [
            {"id": "c2", "statement": "beta", "status": "Unverified",
             "tags": [], "slot_name": None, "version": 1, "result_score": 5.0},
        ]
        result = _rrf_merge(vector, ft, alpha=1.0, limit=10)
        # With alpha=1, vector rank determines order
        assert result[0]["claim_id"] == "c1"

    def test_rrf_score_is_between_0_and_1(self):
        vector = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.9},
            {"id": "c2", "statement": "beta", "status": "Unverified",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.7},
        ]
        ft = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 5.0},
        ]
        result = _rrf_merge(vector, ft, alpha=0.7, limit=10)
        for item in result:
            assert 0.0 <= item["rrf_score"] <= 1.0

    def test_deduplication_across_sources(self):
        """Same claim appearing in both vector and fulltext should be merged."""
        vector = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 0.9},
        ]
        ft = [
            {"id": "c1", "statement": "alpha", "status": "Confirmed",
             "tags": [], "slot_name": None, "version": 1, "result_score": 5.0},
        ]
        result = _rrf_merge(vector, ft, alpha=0.7, limit=10)
        claim_ids = [r["claim_id"] for r in result]
        assert claim_ids.count("c1") == 1
