"""Lightweight embedding client for query-time and ingestion vector generation.

Supports two transports:
  * OpenAI-compatible `/embeddings` (provider != "ollama").
  * Ollama native `/api/embed` (provider == "ollama"), which honours the
    `keep_alive` parameter so the local model is loaded on demand and released
    when idle. The OpenAI-compat `/v1/embeddings` route ignores `keep_alive`,
    so it cannot free the model promptly — hence the native path for Ollama.

Configuration via environment:
    KE_EMBEDDING_PROVIDER, KE_EMBEDDING_MODEL, KE_EMBEDDING_API_KEY,
    KE_EMBEDDING_API_BASE_URL, KE_EMBEDDING_DIMENSIONS, KE_EMBEDDING_KEEP_ALIVE.

Validation is lazy: the client only fails at construction time, not at server
startup.  The server creates it only when all four required vars are present.
Vector search tools return a clear error when no client is configured so that
non-vector tools (ingest, get_entity, get_claim, search_by_domain) still work.

The HTTP layer uses async httpx with tenacity retry so transient 429/503
responses from rate-limited endpoints don't fail the call immediately.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import RetrievalCache

import httpx
from tenacity import (
    Retrying,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


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
        keep_alive: str = "0",
        num_ctx: int = 1024,
        batch_size: int = 64,
    ) -> None:
        if not provider:
            raise ValueError("KE_EMBEDDING_PROVIDER must be set")
        if not model:
            raise ValueError("KE_EMBEDDING_MODEL must be set")
        # Ollama runs locally and needs no key; every other provider requires one.
        if provider != "ollama" and not api_key:
            raise ValueError("KE_EMBEDDING_API_KEY must be set")
        if dimensions <= 0:
            raise ValueError("KE_EMBEDDING_DIMENSIONS must be a positive integer")
        if num_ctx <= 0:
            raise ValueError("KE_EMBEDDING_NUM_CTX must be a positive integer")
        if batch_size <= 0:
            raise ValueError("KE_EMBEDDING_BATCH_SIZE must be a positive integer")

        self.provider = provider
        self.model = model
        self.api_key = api_key or "ollama"
        self.base_url = base_url.rstrip("/") or "https://api.openai.com/v1"
        self.dimensions = dimensions
        # keep_alive controls how long Ollama holds the model in RAM after a call.
        # "0" = unload immediately (tight housekeeping); e.g. "5m" = stay warm.
        self.keep_alive = keep_alive
        # Ollama context window per embed call; smaller = less RAM. Inputs longer
        # than this are truncated by Ollama, so callers batch/chunk to stay within it.
        self.num_ctx = num_ctx
        # Max texts per embed request; embed_texts() splits larger lists.
        self.batch_size = batch_size
        self._warm_keep_alive: str | None = None
        # Ollama's native embed API is at the server root (/api/embed), not /v1.
        self._ollama_root = (
            self.base_url[: -len("/v1")].rstrip("/")
            if self.base_url.endswith("/v1")
            else self.base_url
        )
        self._client = httpx.AsyncClient(timeout=60.0)
        # Optional cache for query-time embeddings (not used during ingestion).
        self._cache: "RetrievalCache | None" = None

    def set_cache(self, cache: "RetrievalCache") -> None:
        """Wire a cache for query-time embedding deduplication."""
        self._cache = cache

    @classmethod
    def from_env(cls) -> "EmbeddingClient | None":
        """Construct from environment variables, or return None if unconfigured.

        Call this at server startup. None means vector search tools are
        unavailable; all other tools still work. For provider `ollama` no API key
        is required (it runs locally).
        """
        provider = os.environ.get("KE_EMBEDDING_PROVIDER", "")
        key = os.environ.get("KE_EMBEDDING_API_KEY", "")
        if provider != "ollama" and not key:
            return None
        dims_raw = os.environ.get("KE_EMBEDDING_DIMENSIONS", "0")
        try:
            dims = int(dims_raw)
        except ValueError:
            return None
        return cls(
            provider=provider,
            model=os.environ.get("KE_EMBEDDING_MODEL", ""),
            api_key=key,
            base_url=os.environ.get("KE_EMBEDDING_API_BASE_URL", ""),
            dimensions=dims,
            keep_alive=os.environ.get("KE_EMBEDDING_KEEP_ALIVE", "0"),
            num_ctx=int(os.environ.get("KE_EMBEDDING_NUM_CTX", "1024")),
            batch_size=int(os.environ.get("KE_EMBEDDING_BATCH_SIZE", "64")),
        )

    @property
    def _effective_keep_alive(self) -> str | int:
        """keep_alive to send: the warm-session value if inside `warm()`, else default."""
        return self._warm_keep_alive if self._warm_keep_alive is not None else self.keep_alive

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
        if self.provider == "ollama":
            resp = await self._client.post(
                f"{self._ollama_root}/api/embed",
                json={
                    "model": self.model,
                    "input": texts,
                    "keep_alive": self._effective_keep_alive,
                    "options": {"num_ctx": self.num_ctx},
                },
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]

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

    # -- synchronous transport (used by engine query + ingestion) --------------
    # A short-lived httpx.Client per call avoids reusing an async connection pool
    # across the fresh event loops that repeated asyncio.run() would create.

    def _post_sync(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        for attempt in Retrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            stop=stop_after_attempt(4),
            reraise=True,
        ):
            with attempt:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
        raise RuntimeError("unreachable")  # pragma: no cover

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch embed.

        Ollama loads the model on demand and releases it per `keep_alive`
        (default "0" = unload immediately after the call).
        """
        if self.provider == "ollama":
            data = self._post_sync(
                f"{self._ollama_root}/api/embed",
                {
                    "model": self.model,
                    "input": texts,
                    "keep_alive": self._effective_keep_alive,
                    "options": {"num_ctx": self.num_ctx},
                },
            )
            return data["embeddings"]

        payload: dict = {"input": texts, "model": self.model}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        data = self._post_sync(
            f"{self.base_url}/embeddings",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        ordered = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in ordered]

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous single embed. For call sites that cannot be async.

        Uses cache when available (query-path). Ingestion-path calls use
        embed_texts() which bypasses the cache.
        """
        if self._cache is not None:
            from .cache import RetrievalCache
            cache_key = RetrievalCache.make_key("embed", self.model, text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        result = self.embed_batch_sync([text])[0]
        if self._cache is not None:
            from .cache import RetrievalCache
            cache_key = RetrievalCache.make_key("embed", self.model, text)
            self._cache.set(cache_key, result)
        return result

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, split into batches of at most ``batch_size``.

        Batching (a list per request) is the effective throughput win: a single
        local Ollama model serves one request at a time, so firing parallel
        requests at it would mostly contend rather than speed up. Wrap a call in
        ``warm()`` to load the model once for the whole run and release it after.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self.embed_batch_sync(texts[start : start + self.batch_size]))
        return vectors

    def unload_sync(self) -> None:
        """Release the model from Ollama RAM immediately (keep_alive=0).

        No-op for non-Ollama providers. Best-effort: transport errors are
        swallowed so housekeeping never breaks a caller's main flow.
        """
        if self.provider != "ollama":
            return
        try:
            self._post_sync(
                f"{self._ollama_root}/api/embed",
                {"model": self.model, "input": ["x"], "keep_alive": 0},
            )
        except httpx.HTTPError:
            pass

    @contextmanager
    def warm(self, keep_alive: str = "5m"):
        """Scope a run (e.g. ingestion/chunking) that makes several embed calls.

        Inside the block the model stays resident (`keep_alive` window) so it
        loads once instead of per call; on exit it is unloaded so nothing lingers
        in RAM while idle. Usage:

            with client.warm():
                for chunk in chunks:
                    client.embed_sync(chunk)
            # model released here
        """
        prev = self._warm_keep_alive
        self._warm_keep_alive = keep_alive
        try:
            yield self
        finally:
            self._warm_keep_alive = prev
            self.unload_sync()

    async def close(self) -> None:
        await self._client.aclose()
