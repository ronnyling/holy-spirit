"""LLM Configuration Router.

Routes LLM tasks to appropriate backends based on priority:
- HIGH priority: MiMo API (external, high-quality)
- LOW priority: Ollama (local, cheap)

Modules call get_llm_client(task_type) to get the appropriate client.

NO FALLBACKS: Raises errors if required client not configured.
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from .llm import MiMoClient


class TaskPriority(Enum):
    """Task priority levels for LLM routing."""
    HIGH = "high"      # MiMo API - critical tasks requiring high quality
    LOW = "low"        # Ollama - routine tasks, cost-sensitive


class TaskType(Enum):
    """Task types mapped to priorities."""
    # HIGH priority tasks (use MiMo API)
    CLAIM_EXTRACTION = "claim_extraction"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    CONFLICT_INTERPRETATION = "conflict_interpretation"
    EXPERIENCE_SYNTHESIS = "experience_synthesis"
    SEMANTIC_GAP_DETECTION = "semantic_gap_detection"
    CLAIM_RECONCILIATION = "claim_reconciliation"
    DOMAIN_CLASSIFICATION = "domain_classification"

    # LOW priority tasks (use Ollama)
    UNSTATED_ASSUMPTIONS = "unstated_assumptions"
    EVIDENCE_QUALITY_ASSESSMENT = "evidence_quality_assessment"
    SIMPLE_CLASSIFICATION = "simple_classification"
    TEXT_SIMILARITY = "text_similarity"


# Task to priority mapping
TASK_PRIORITIES: dict[TaskType, TaskPriority] = {
    # HIGH priority - MiMo API
    TaskType.CLAIM_EXTRACTION: TaskPriority.HIGH,
    TaskType.EVIDENCE_EXTRACTION: TaskPriority.HIGH,
    TaskType.CONFLICT_INTERPRETATION: TaskPriority.HIGH,
    TaskType.EXPERIENCE_SYNTHESIS: TaskPriority.HIGH,
    TaskType.SEMANTIC_GAP_DETECTION: TaskPriority.HIGH,
    TaskType.CLAIM_RECONCILIATION: TaskPriority.HIGH,
    TaskType.DOMAIN_CLASSIFICATION: TaskPriority.HIGH,

    # LOW priority - Ollama
    TaskType.UNSTATED_ASSUMPTIONS: TaskPriority.LOW,
    TaskType.EVIDENCE_QUALITY_ASSESSMENT: TaskPriority.LOW,
    TaskType.SIMPLE_CLASSIFICATION: TaskPriority.LOW,
    TaskType.TEXT_SIMILARITY: TaskPriority.LOW,
}


@runtime_checkable
class SupportsComplete(Protocol):
    """Protocol for LLM clients that support complete_sync."""
    def complete_sync(self, *, system: str, user: str, max_tokens: int = 8000) -> str: ...


class OllamaClient:
    """Local Ollama LLM client for low-priority tasks.

    Uses Ollama's /api/generate endpoint for text generation.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> "OllamaClient | None":
        """Construct from environment, or return None if Ollama not configured."""
        if os.environ.get("KE_OLLAMA_DISABLED", "").lower() == "true":
            return None

        model = os.environ.get("KE_OLLAMA_LLM_MODEL", "llama3.2")
        base_url = os.environ.get("KE_OLLAMA_BASE_URL", "http://localhost:11434")

        return cls(model=model, base_url=base_url)

    def complete_sync(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> str:
        """Synchronous chat completion via Ollama."""
        import httpx

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")


class LLMConfig:
    """Central LLM configuration and client routing.

    Provides appropriate LLM client based on task type and priority.
    """

    def __init__(
        self,
        mimo_client: MiMoClient | None = None,
        ollama_client: OllamaClient | None = None,
    ):
        self.mimo_client = mimo_client
        self.ollama_client = ollama_client

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLMConfig from environment variables."""
        mimo_client = MiMoClient.from_env()
        ollama_client = OllamaClient.from_env()
        return cls(mimo_client=mimo_client, ollama_client=ollama_client)

    def get_client(self, task_type: TaskType) -> SupportsComplete:
        """Get appropriate LLM client for task type.

        NO FALLBACKS: Raises RuntimeError if required client not configured.

        Args:
            task_type: Type of task requiring LLM

        Returns:
            LLM client configured for this task type

        Raises:
            RuntimeError: If required client not configured
        """
        priority = TASK_PRIORITIES.get(task_type)

        if priority == TaskPriority.HIGH:
            if self.mimo_client is None:
                raise RuntimeError(
                    f"Task '{task_type.value}' requires MiMo API (HIGH priority). "
                    f"Set KE_MIMO_API_KEY to configure. "
                    f"No fallback to Ollama allowed for high-priority tasks."
                )
            return self.mimo_client

        elif priority == TaskPriority.LOW:
            if self.ollama_client is not None:
                return self.ollama_client
            elif self.mimo_client is not None:
                # Fall back to MiMo if Ollama not available (but log warning)
                import logging
                logging.warning(
                    f"Ollama not configured for task '{task_type.value}'. "
                    f"Using MiMo API instead. This increases cost."
                )
                return self.mimo_client
            else:
                raise RuntimeError(
                    f"Task '{task_type.value}' requires LLM client. "
                    f"Configure either KE_MIMO_API_KEY (MiMo) or Ollama. "
                    f"No client available."
                )

        else:
            raise ValueError(f"Unknown task priority for: {task_type}")

    def get_client_for_priority(self, priority: TaskPriority) -> SupportsComplete:
        """Get LLM client for specific priority level.

        NO FALLBACKS: Raises RuntimeError if required client not configured.
        """
        if priority == TaskPriority.HIGH:
            if self.mimo_client is None:
                raise RuntimeError(
                    "HIGH priority tasks require MiMo API. "
                    "Set KE_MIMO_API_KEY. No fallback allowed."
                )
            return self.mimo_client

        elif priority == TaskPriority.LOW:
            if self.ollama_client is not None:
                return self.ollama_client
            elif self.mimo_client is not None:
                import logging
                logging.warning(
                    "Ollama not configured for LOW priority tasks. "
                    "Using MiMo API instead."
                )
                return self.mimo_client
            else:
                raise RuntimeError(
                    "No LLM client configured. "
                    "Set KE_MIMO_API_KEY or configure Ollama."
                )

        else:
            raise ValueError(f"Unknown priority: {priority}")

    def has_client(self, task_type: TaskType) -> bool:
        """Check if a client is available for task type."""
        try:
            self.get_client(task_type)
            return True
        except RuntimeError:
            return False

    def get_status(self) -> dict:
        """Get status of all LLM clients."""
        return {
            "mimo": {
                "configured": self.mimo_client is not None,
                "model": self.mimo_client.model if self.mimo_client else None,
            },
            "ollama": {
                "configured": self.ollama_client is not None,
                "model": self.ollama_client.model if self.ollama_client else None,
            },
            "task_routing": {
                task.value: priority.value
                for task, priority in TASK_PRIORITIES.items()
            }
        }


# Global config instance
_config: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """Get or create global LLM config."""
    global _config
    if _config is None:
        _config = LLMConfig.from_env()
    return _config


def get_llm_client(task_type: TaskType) -> SupportsComplete:
    """Convenience function to get LLM client for task type.

    NO FALLBACKS: Raises RuntimeError if required client not configured.
    """
    return get_llm_config().get_client(task_type)
