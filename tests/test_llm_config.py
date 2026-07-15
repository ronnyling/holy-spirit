"""Tests for LLM configuration router.

NO FALLBACKS: Tests verify explicit behavior, no silent failures.
"""
import os
import pytest
from knowledge_engine.llm_config import (
    LLMConfig,
    OllamaClient,
    TaskType,
    TaskPriority,
    TASK_PRIORITIES,
    get_llm_client,
)


def test_task_priorities_defined():
    """All task types have defined priorities."""
    for task_type in TaskType:
        assert task_type in TASK_PRIORITIES, f"Missing priority for {task_type}"


def test_high_priority_tasks():
    """High-priority tasks are correctly identified."""
    high_priority_tasks = [
        TaskType.CLAIM_EXTRACTION,
        TaskType.EVIDENCE_EXTRACTION,
        TaskType.CONFLICT_INTERPRETATION,
        TaskType.EXPERIENCE_SYNTHESIS,
        TaskType.SEMANTIC_GAP_DETECTION,
        TaskType.CLAIM_RECONCILIATION,
        TaskType.DOMAIN_CLASSIFICATION,
    ]
    for task in high_priority_tasks:
        assert TASK_PRIORITIES[task] == TaskPriority.HIGH, f"{task} should be HIGH priority"


def test_low_priority_tasks():
    """Low-priority tasks are correctly identified."""
    low_priority_tasks = [
        TaskType.UNSTATED_ASSUMPTIONS,
        TaskType.EVIDENCE_QUALITY_ASSESSMENT,
        TaskType.SIMPLE_CLASSIFICATION,
    ]
    for task in low_priority_tasks:
        assert TASK_PRIORITIES[task] == TaskPriority.LOW, f"{task} should be LOW priority"


def test_embedding_tasks():
    """Embedding tasks are correctly identified."""
    embedding_tasks = [
        TaskType.TEXT_SIMILARITY,
    ]
    for task in embedding_tasks:
        assert TASK_PRIORITIES[task] == TaskPriority.EMBEDDING, f"{task} should be EMBEDDING priority"


def test_config_no_clients():
    """Config with no clients raises error for all tasks."""
    config = LLMConfig(mimo_client=None, ollama_client=None, embedding_client=None)

    # Test HIGH and LOW priority tasks
    for task_type in [TaskType.CLAIM_EXTRACTION, TaskType.UNSTATED_ASSUMPTIONS]:
        with pytest.raises(RuntimeError) as exc_info:
            config.get_client(task_type)
        assert "No client available" in str(exc_info.value) or "requires MiMo" in str(exc_info.value)

    # Test EMBEDDING tasks (different error message)
    with pytest.raises(RuntimeError) as exc_info:
        config.get_client(TaskType.TEXT_SIMILARITY)
    assert "requires embedding client" in str(exc_info.value)


def test_config_mimo_only():
    """Config with only MiMo client works for HIGH tasks, falls back for LOW."""
    class MockMimoClient:
        def complete_sync(self, *, system, user, max_tokens=8000):
            return "mock response"

    config = LLMConfig(mimo_client=MockMimoClient(), ollama_client=None)

    # HIGH priority tasks should work
    client = config.get_client(TaskType.CLAIM_EXTRACTION)
    assert client is not None

    # LOW priority tasks should fall back to MiMo
    client = config.get_client(TaskType.UNSTATED_ASSUMPTIONS)
    assert client is not None


def test_config_ollama_only():
    """Config with only Ollama client works for LOW tasks, raises for HIGH."""
    class MockOllamaClient:
        def complete_sync(self, *, system, user, max_tokens=8000):
            return "mock response"

    config = LLMConfig(mimo_client=None, ollama_client=MockOllamaClient())

    # LOW priority tasks should work
    client = config.get_client(TaskType.UNSTATED_ASSUMPTIONS)
    assert client is not None

    # HIGH priority tasks should raise error
    with pytest.raises(RuntimeError) as exc_info:
        config.get_client(TaskType.CLAIM_EXTRACTION)
    assert "requires MiMo" in str(exc_info.value)


def test_config_both_clients():
    """Config with both clients routes correctly."""
    class MockMimoClient:
        def complete_sync(self, *, system, user, max_tokens=8000):
            return "mimo response"

    class MockOllamaClient:
        def complete_sync(self, *, system, user, max_tokens=8000):
            return "ollama response"

    config = LLMConfig(mimo_client=MockMimoClient(), ollama_client=MockOllamaClient())

    # HIGH priority → MiMo
    client = config.get_client(TaskType.CLAIM_EXTRACTION)
    assert client.complete_sync(system="", user="") == "mimo response"

    # LOW priority → Ollama
    client = config.get_client(TaskType.UNSTATED_ASSUMPTIONS)
    assert client.complete_sync(system="", user="") == "ollama response"


def test_has_client():
    """has_client returns correct boolean."""
    class MockClient:
        def complete_sync(self, *, system, user, max_tokens=8000):
            return "response"

    config_with_mimo = LLMConfig(mimo_client=MockClient(), ollama_client=None)
    config_without = LLMConfig(mimo_client=None, ollama_client=None)

    assert config_with_mimo.has_client(TaskType.CLAIM_EXTRACTION) is True
    assert config_without.has_client(TaskType.CLAIM_EXTRACTION) is False


def test_get_status():
    """get_status returns correct information."""
    class MockMimoClient:
        model = "mimo-v2.5"

    config = LLMConfig(mimo_client=MockMimoClient(), ollama_client=None)
    status = config.get_status()

    assert status["mimo"]["configured"] is True
    assert status["mimo"]["model"] == "mimo-v2.5"
    assert status["ollama"]["configured"] is False
    assert "task_routing" in status


def test_ollama_client_from_env():
    """OllamaClient can be created from environment."""
    # Save original env
    orig_model = os.environ.get("KE_OLLAMA_SLM_MODEL")
    orig_url = os.environ.get("KE_OLLAMA_BASE_URL")

    try:
        os.environ["KE_OLLAMA_SLM_MODEL"] = "test-model"
        os.environ["KE_OLLAMA_BASE_URL"] = "http://localhost:11434"

        client = OllamaClient.from_env()
        assert client is not None
        assert client.model == "test-model"
    finally:
        # Restore env
        if orig_model is None:
            os.environ.pop("KE_OLLAMA_SLM_MODEL", None)
        else:
            os.environ["KE_OLLAMA_SLM_MODEL"] = orig_model
        if orig_url is None:
            os.environ.pop("KE_OLLAMA_BASE_URL", None)
        else:
            os.environ["KE_OLLAMA_BASE_URL"] = orig_url


def test_ollama_client_disabled():
    """OllamaClient.from_env returns None when disabled."""
    orig = os.environ.get("KE_OLLAMA_DISABLED")
    try:
        os.environ["KE_OLLAMA_DISABLED"] = "true"
        client = OllamaClient.from_env()
        assert client is None
    finally:
        if orig is None:
            os.environ.pop("KE_OLLAMA_DISABLED", None)
        else:
            os.environ["KE_OLLAMA_DISABLED"] = orig


def test_invalid_priority_raises_error():
    """Unknown priority raises ValueError."""
    config = LLMConfig(mimo_client=None, ollama_client=None)

    with pytest.raises(ValueError) as exc_info:
        config.get_client_for_priority("invalid")
    assert "Unknown priority" in str(exc_info.value)
