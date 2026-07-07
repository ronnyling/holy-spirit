"""Embedding-based reranker for retrieval precision.

Uses Ollama's /api/embed endpoint (same as EmbeddingClient) to embed query
and documents, then computes cosine similarity locally. No fallbacks —
fails loudly on transport or model errors.

Configuration:
    KE_RERANKER_PROVIDER (ollama | none)
    KE_RERANKER_MODEL (e.g. bona/bge-reranker-v2-m3)
    KE_RERANKER_API_BASE_URL
    KE_RERANKER_KEEP_ALIVE
"""

from __future__ import annotations

import logging
import math
import os

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Deterministic cosine similarity. No fallbacks."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RerankerClient:
    """Embedding-based reranking via Ollama /api/embed.

    Embeds query and documents using the same endpoint as EmbeddingClient,
    then computes cosine similarity locally. Follows the same transport
    pattern: Ollama native path with keep_alive lifecycle.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        keep_alive: str = "0",
        timeout: float = 30.0,
    ) -> None:
        if not provider:
            raise ValueError("KE_RERANKER_PROVIDER must be set")
        if not model:
            raise ValueError("KE_RERANKER_MODEL must be set")

        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/") or "http://localhost:11434"
        self.keep_alive = keep_alive
        self._timeout = timeout
        self._warm_keep_alive: str | None = None
        self._ollama_root = (
            self.base_url[: -len("/v1")].rstrip("/")
            if self.base_url.endswith("/v1")
            else self.base_url
        )
        self._client = httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> RerankerClient | None:
        """Construct from environment, or return None when provider is 'none'."""
        provider = os.environ.get("KE_RERANKER_PROVIDER", "none")
        if provider == "none":
            return None
        return cls(
            provider=provider,
            model=os.environ.get("KE_RERANKER_MODEL", ""),
            base_url=os.environ.get("KE_RERANKER_API_BASE_URL", ""),
            keep_alive=os.environ.get("KE_RERANKER_KEEP_ALIVE", "0"),
        )

    @property
    def _effective_keep_alive(self) -> str | int:
        return self._warm_keep_alive if self._warm_keep_alive is not None else self.keep_alive

    def rerank(
        self,
        query: str,
        documents: list[dict],
        *,
        top_n: int = 10,
    ) -> list[dict]:
        """Rerank documents by query-document cosine similarity.

        Returns top_n documents with rerank_score added. Fails loudly
        on transport or model errors — no silent degradation.
        """
        if not documents:
            return []

        scores = self._compute_scores(query, documents)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score

        reranked = sorted(documents, key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_n]

    def _compute_scores(self, query: str, documents: list[dict]) -> list[float]:
        """Embed query + documents, compute cosine similarity."""
        texts = [doc.get("statement", "") for doc in documents]

        # Embed query
        query_vec = self._embed_single(query)

        # Embed documents in batch
        doc_vecs = self._embed_batch(texts)

        return [_cosine_similarity(query_vec, dv) for dv in doc_vecs]

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text via Ollama /api/embed."""
        resp = self._client.post(
            f"{self._ollama_root}/api/embed",
            json={
                "model": self.model,
                "input": text,
                "keep_alive": self._effective_keep_alive,
            },
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if not embeddings:
            raise RuntimeError(f"reranker returned empty embeddings for: {text[:80]!r}")
        return embeddings[0]

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts via Ollama /api/embed."""
        resp = self._client.post(
            f"{self._ollama_root}/api/embed",
            json={
                "model": self.model,
                "input": texts,
                "keep_alive": self._effective_keep_alive,
            },
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"reranker returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        return embeddings

    def close(self) -> None:
        self._client.close()
