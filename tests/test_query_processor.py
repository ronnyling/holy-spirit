"""Tests for query processor (expansion + decomposition)."""

from __future__ import annotations

import pytest

from knowledge_engine.query_processor import ProcessedQuery, QueryProcessor


class _FakeLLM:
    """Fake LLM that returns responses in sequence."""

    def __init__(self, responses: list[str]):
        self._resps = list(responses)
        self._call_count = 0

    def complete_sync(self, **kwargs) -> str:
        if self._call_count < len(self._resps):
            resp = self._resps[self._call_count]
            self._call_count += 1
            return resp
        return ""


class TestProcessedQuery:
    """Verify ProcessedQuery data structure."""

    def test_all_queries_deduplicates(self):
        pq = ProcessedQuery(
            original="what is cap rate",
            expansions=["capitalization rate", "what is cap rate"],
            sub_queries=["cap rate in Shanghai"],
        )
        queries = pq.all_queries
        assert len(queries) == 3  # original + 1 expansion + 1 sub_query
        assert queries[0] == "what is cap rate"

    def test_all_queries_empty(self):
        pq = ProcessedQuery(original="test query")
        assert pq.all_queries == ["test query"]

    def test_llm_calls_used_default(self):
        pq = ProcessedQuery(original="q", expansions=["a", "b"], sub_queries=["c"])
        assert pq.llm_calls_used == 0  # default value

    def test_llm_calls_used_set(self):
        pq = ProcessedQuery(original="q", expansions=["a", "b"], sub_queries=["c"], llm_calls_used=3)
        assert pq.llm_calls_used == 3


class TestQueryProcessorInit:
    """Verify QueryProcessor construction and from_env."""

    def test_from_env_defaults_off(self, monkeypatch):
        monkeypatch.delenv("KE_QUERY_EXPAND", raising=False)
        monkeypatch.delenv("KE_QUERY_DECOMPOSE", raising=False)
        qp = QueryProcessor.from_env()
        assert qp.expand is False
        assert qp.decompose is False

    def test_from_env_enables_expand(self, monkeypatch):
        monkeypatch.setenv("KE_QUERY_EXPAND", "true")
        monkeypatch.delenv("KE_QUERY_DECOMPOSE", raising=False)
        llm = _FakeLLM(["response"])
        qp = QueryProcessor.from_env(llm)
        assert qp.expand is True
        assert qp.decompose is False

    def test_from_env_enables_both(self, monkeypatch):
        monkeypatch.setenv("KE_QUERY_EXPAND", "true")
        monkeypatch.setenv("KE_QUERY_DECOMPOSE", "true")
        monkeypatch.setenv("KE_QUERY_MAX_LLM_CALLS", "5")
        llm = _FakeLLM(["response"])
        qp = QueryProcessor.from_env(llm)
        assert qp.expand is True
        assert qp.decompose is True
        assert qp.max_calls == 5

    def test_requires_llm_for_expand(self):
        qp = QueryProcessor(None, expand=True, decompose=True)
        # Without LLM, expand and decompose should be disabled
        assert qp.expand is False
        assert qp.decompose is False

    def test_with_llm_enables_features(self):
        llm = _FakeLLM(["test"])
        qp = QueryProcessor(llm, expand=True, decompose=True)
        assert qp.expand is True
        assert qp.decompose is True


class TestQueryProcessorProcess:
    """Verify query processing logic."""

    def test_no_processing_when_disabled(self):
        qp = QueryProcessor(None, expand=False, decompose=False)
        result = qp.process("what is cap rate")
        assert result.original == "what is cap rate"
        assert result.expansions == []
        assert result.sub_queries == []
        assert result.llm_calls_used == 0

    def test_expand_generates_variants(self):
        llm = _FakeLLM(["capitalization rate\nyield on cost"])
        qp = QueryProcessor(llm, expand=True, decompose=False, max_calls=3)
        result = qp.process("what is cap rate", domain="real estate")
        assert len(result.expansions) == 2
        assert "capitalization rate" in result.expansions
        # llm_calls_used counts the number of expansions returned (1 LLM call)
        assert result.llm_calls_used == 2

    def test_decompose_splits_query(self):
        llm = _FakeLLM(["cap rate trends\nvacancy rates in Shanghai"])
        qp = QueryProcessor(llm, expand=False, decompose=True, max_calls=3)
        result = qp.process("what are cap rate trends and vacancy rates in Shanghai")
        assert len(result.sub_queries) == 2
        # llm_calls_used counts the number of sub-queries returned
        assert result.llm_calls_used == 2

    def test_budget_respected(self):
        llm = _FakeLLM(["synonym1\nsynonym2", "sub1\nsub2"])
        qp = QueryProcessor(llm, expand=True, decompose=True, max_calls=2)
        result = qp.process("complex query")
        # Both expand and decompose should run within budget
        assert result.llm_calls_used <= 4  # 2 expansions + 2 sub_queries

    def test_raises_on_llm_error(self):
        """LLM errors must propagate — no silent degradation."""
        class FailingLLM:
            def complete_sync(self, **kwargs):
                raise RuntimeError("LLM unavailable")
        qp = QueryProcessor(FailingLLM(), expand=True, decompose=True)
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            qp.process("test query")

    def test_empty_expansion_filtered(self):
        llm = _FakeLLM(["\n\n"])
        qp = QueryProcessor(llm, expand=True, decompose=False)
        result = qp.process("test query")
        assert result.expansions == []

    def test_original_query_in_all_queries(self):
        llm = _FakeLLM(["synonym1"])
        qp = QueryProcessor(llm, expand=True, decompose=False)
        result = qp.process("my query")
        assert result.original in result.all_queries
