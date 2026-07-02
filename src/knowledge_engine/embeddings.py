"""Lightweight embedding client for query-time vector generation.

Uses the OpenAI-compatible embedding API. Configuration via environment:
    KE_EMBEDDING_PROVIDER, KE_EMBEDDING_MODEL, KE_EMBEDDING_API_KEY,
    KE_EMBEDDING_API_BASE_URL, KE_EMBEDDING_DIMENSIONS.

Validation is lazy: the client only fails at construction time, not at server
startup.  The server creates it only when all four required vars are present.
Vector search tools return a clear error when no client is configured so that
non-vector tools (ingest, get_entity, get_claim, search_by_domain) still work.

The HTTP layer uses async httpx with tenacity retry so transient 429/503
responses from rate-limited endpoints don't fail the call immediately.
"""

from __future__ import annotations

import asyncio
import os

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class EmbeddingClient:
    """Generates text embeddings via an OpenAI-compatible API.

    Raises ValueError at construction if any required env var is missing.
    Use EmbeddingClient.from_env() to get None when vars are absent (server
    startup path).
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        dimensions: int,
    ) -> None:
        if not provider:
            raise ValueError("KE_EMBEDDING_PROVIDER must be set")
        if not model:
            raise ValueError("KE_EMBEDDING_MODEL must be set")
        if not api_key:
            raise ValueError("KE_EMBEDDING_API_KEY must be set")
        if dimensions <= 0:
            raise ValueError("KE_EMBEDDING_DIMENSIONS must be a positive integer")

        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") or "https://api.openai.com/v1"
        self.dimensions = dimensions
        self._client = httpx.AsyncClient(timeout=30.0)

    @classmethod
    def from_env(cls) -> "EmbeddingClient | None":
        """Construct from environment variables, or return None if any are absent.

        Call this at server startup.  None means vector search tools are
        unavailable; all other tools still work.
        """
        key = os.environ.get("KE_EMBEDDING_API_KEY", "")
        if not key:
            return None
        dims_raw = os.environ.get("KE_EMBEDDING_DIMENSIONS", "0")
        try:
            dims = int(dims_raw)
        except ValueError:
            return None
        return cls(
            provider=os.environ.get("KE_EMBEDDING_PROVIDER", ""),
            model=os.environ.get("KE_EMBEDDING_MODEL", ""),
            api_key=key,
            base_url=os.environ.get("KE_EMBEDDING_API_BASE_URL", ""),
            dimensions=dims,
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts, with retry on transient errors."""
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: dict = {"input": texts, "model": self.model}
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous wrapper — runs the async embed in a new event loop.

        Only for call sites that cannot be made async (e.g. engine.py sync methods).
        """
        return asyncio.run(self.embed(text))

    async def close(self) -> None:
        await self._client.aclose()
