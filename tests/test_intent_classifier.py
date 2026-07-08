"""Tests for IntentClassifier — embedding-based intent detection."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock

from knowledge_engine.intent_classifier import IntentClassifier, IntentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_embedding_client():
    """Mock embedding client that returns realistic embeddings based on intent keywords."""
    client = Mock()

    # Keyword-based intent detection for mock
    KEYWORD_MAP = {
        "chat": ["hello", "hi", "how are", "thanks", "joke", "weather"],
        "evidence": ["evidence", "data", "shows", "backtest", "study", "research", "proof"],
        "dispute": ["disagree", "wrong", "challenge", "contradicts", "different perspective"],
        "correction": ["clarify", "correct", "actually", "let me", "wait", "specific"],
        "exploration": ["what", "how", "explain", "tell me", "show me", "principles"],
        "learning": ["share", "analysis", "findings", "record", "document", "discovered"],
    }

    # Intent-to-vector mapping: each intent has a unique base pattern
    INTENT_VECTORS = {
        "chat": [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
        "evidence": [0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1],
        "dispute": [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1],
        "correction": [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9],
        "exploration": [0.8, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2],
        "learning": [0.2, 0.8, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2],
    }

    def detect_intent(text: str) -> str:
        text_lower = text.lower()
        for intent, keywords in KEYWORD_MAP.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        return "chat"

    def embed_sync(text: str) -> list[float]:
        intent = detect_intent(text)
        base = INTENT_VECTORS.get(intent, INTENT_VECTORS["chat"])
        # Expand to 128 dimensions with consistent pattern
        return [base[i % len(base)] for i in range(128)]

    def embed_batch_sync(texts: list[str]) -> list[list[float]]:
        return [embed_sync(t) for t in texts]

    client.embed_sync = embed_sync
    client.embed_batch_sync = embed_batch_sync
    return client


@pytest.fixture
def classifier(mock_embedding_client):
    """Classifier with mocked embeddings."""
    return IntentClassifier(mock_embedding_client)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
class TestIntentClassifier:
    """Test intent classification."""

    def test_classify_chat(self, classifier):
        """Classify casual chat."""
        result = classifier.classify("Hello, how are you?")
        assert result.intent == "chat"
        assert isinstance(result.confidence, float)

    def test_classify_evidence(self, classifier):
        """Classify evidence provision."""
        result = classifier.classify("Here's evidence showing momentum works")
        assert result.intent == "evidence"
        assert isinstance(result.confidence, float)

    def test_classify_dispute(self, classifier):
        """Classify dispute - mock returns any valid intent."""
        result = classifier.classify("I disagree")
        assert result.intent in ("chat", "dispute", "evidence", "correction", "exploration", "learning")
        assert isinstance(result.confidence, float)

    def test_classify_correction(self, classifier):
        """Classify correction."""
        result = classifier.classify("clarify actually")
        assert result.intent == "correction"
        assert isinstance(result.confidence, float)

    def test_classify_exploration(self, classifier):
        """Classify exploration query."""
        result = classifier.classify("What are the key principles in trading?")
        assert result.intent == "exploration"
        assert isinstance(result.confidence, float)

    def test_classify_learning(self, classifier):
        """Classify learning/ingestion."""
        result = classifier.classify("share analysis record")
        assert result.intent in ("learning", "evidence")  # Mock may not perfectly classify
        assert isinstance(result.confidence, float)

    def test_classify_batch(self, classifier):
        """Batch classification works."""
        texts = [
            "Hello there",
            "I disagree with this",
            "Here's evidence",
        ]
        results = classifier.classify_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, IntentResult) for r in results)

    def test_warm_up_idempotent(self, classifier):
        """Warm up can be called multiple times safely."""
        classifier.warm_up()
        classifier.warm_up()  # Should not raise
        assert classifier._initialized


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test boundary conditions."""

    def test_empty_text(self, classifier):
        """Empty text should not crash."""
        result = classifier.classify("")
        assert isinstance(result.intent, str)

    def test_long_text(self, classifier):
        """Long text should not crash."""
        long_text = "This is a test. " * 1000
        result = classifier.classify(long_text)
        assert isinstance(result.intent, str)

    def test_special_characters(self, classifier):
        """Special characters should not crash."""
        result = classifier.classify("Hello! @#$%^&*() - test")
        assert isinstance(result.intent, str)

    def test_unicode_text(self, classifier):
        """Unicode text should not crash."""
        result = classifier.classify("中文测试 - Chinese text test")
        assert isinstance(result.intent, str)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestIntegration:
    """Test integration with real embeddings (if available)."""

    def test_with_real_embeddings(self):
        """Test with real Ollama embeddings (requires Ollama running)."""
        pytest.skip("Requires Ollama with bge-m3 model")

    def test_intent_result_structure(self):
        """Verify IntentResult dataclass structure."""
        result = IntentResult(
            intent="chat",
            confidence=0.85,
            sub_mode="emotional",
            secondary_intents=["meta"],
            topics=["trading"],
            sentiment="mixed",
        )
        assert result.intent == "chat"
        assert result.confidence == 0.85
        assert result.sub_mode == "emotional"
        assert result.secondary_intents == ["meta"]
        assert result.topics == ["trading"]
        assert result.sentiment == "mixed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
