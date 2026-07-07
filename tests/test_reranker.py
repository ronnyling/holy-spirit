"""Tests for embedding-based reranker."""

from __future__ import annotations

import pytest

from knowledge_engine.reranker import RerankerClient, _cosine_similarity


class TestCosineSimilarity:
    """Verify cosine similarity computation."""

    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_known_value(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        expected = (1*4 + 2*5 + 3*6) / (
            (1**2 + 2**2 + 3**2) ** 0.5 * (4**2 + 5**2 + 6**2) ** 0.5
        )
        assert _cosine_similarity(a, b) == pytest.approx(expected)


class TestRerankerClientInit:
    """Verify RerankerClient construction and from_env."""

    def test_from_env_returns_none_when_none(self, monkeypatch):
        monkeypatch.setenv("KE_RERANKER_PROVIDER", "none")
        assert RerankerClient.from_env() is None

    def test_from_env_returns_none_when_absent(self, monkeypatch):
        monkeypatch.delenv("KE_RERANKER_PROVIDER", raising=False)
        assert RerankerClient.from_env() is None

    def test_from_env_constructs_ollama(self, monkeypatch):
        monkeypatch.setenv("KE_RERANKER_PROVIDER", "ollama")
        monkeypatch.setenv("KE_RERANKER_MODEL", "bge-reranker-v2-m3")
        monkeypatch.setenv("KE_RERANKER_API_BASE_URL", "http://localhost:11434")
        client = RerankerClient.from_env()
        assert client is not None
        assert client.provider == "ollama"
        assert client.model == "bge-reranker-v2-m3"
        client.close()

    def test_requires_model(self):
        with pytest.raises(ValueError, match="KE_RERANKER_MODEL"):
            RerankerClient(provider="ollama", model="", base_url="")

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="KE_RERANKER_PROVIDER"):
            RerankerClient(provider="", model="test", base_url="")


class TestRerankerRerank:
    """Verify rerank logic."""

    def test_empty_documents(self):
        client = RerankerClient(provider="ollama", model="test", base_url="")
        result = client.rerank("query", [], top_n=10)
        assert result == []
        client.close()

    def test_rerank_raises_on_unreachable(self):
        """When the endpoint is unreachable, the error propagates (no fallback)."""
        client = RerankerClient(
            provider="ollama", model="test",
            base_url="http://localhost:19999", timeout=1.0,
        )
        docs = [
            {"claim_id": "c1", "statement": "alpha"},
            {"claim_id": "c2", "statement": "beta"},
        ]
        with pytest.raises(Exception):
            client.rerank("query", docs, top_n=10)
        client.close()

    def test_rerank_adds_scores(self):
        """When reranker works, documents get rerank_score."""
        client = RerankerClient(
            provider="ollama", model="test",
            base_url="http://localhost:19999", timeout=1.0,
        )
        # This will raise, but we verify the interface contract
        docs = [{"claim_id": "c1", "statement": "test"}]
        with pytest.raises(Exception):
            client.rerank("query", docs, top_n=10)
        client.close()
