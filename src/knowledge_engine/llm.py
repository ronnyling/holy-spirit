"""OpenAI-compatible chat client for the MiMo LLM gateway.

Configuration via environment:
    KE_MIMO_API_BASE_URL, KE_MIMO_API_KEY, KE_MIMO_MODEL.

This is the UNBOUNDED layer's transport. It only turns text into text; all
downstream interpretation (parsing extracted claims, deciding epistemic status,
observing slots) stays in deterministic code. The client fails loudly on
non-transient errors and retries transient 429/5xx responses.

Use MiMoClient.from_env() to get None when KE_MIMO_API_KEY is absent, so the
engine keeps working in manual-claims mode without an LLM.
"""

from __future__ import annotations

import os
import threading
import time

import httpx
from tenacity import (
    Retrying,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# MiMo v2.5 reasoning model: output limited to ~16K tokens.
# Reasoning_content consumes most of the budget; be conservative.
_MIMO_MAX_OUTPUT_TOKENS = 16_000


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, LLMEmptyResponseError):
        # Transient reasoning-model glitch (empty content) — worth another attempt.
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class LLMEmptyResponseError(RuntimeError):
    """The model returned no usable content. Retryable — NOT a silent empty result."""


class LLMTruncatedError(RuntimeError):
    """The completion hit max_tokens (finish_reason=length). Not retryable at the
    same budget — raised loudly with an actionable message instead of returning
    a half-formed answer."""


def _content_or_raise(data: dict, max_tokens: int) -> str:
    """Return non-empty assistant content, or raise loudly.

    MiMo v2.5 is a reasoning model: it emits `reasoning_content` plus the answer
    in `content`, and `content` can come back empty (or truncated when reasoning
    eats the token budget). Returning that empty string would be a silent
    degradation to "0 claims" — instead we surface it: truncation is a hard error
    with an actionable message; a merely-empty answer is retryable upstream.
    """
    choice = (data.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "")
    if choice.get("finish_reason") == "length":
        raise LLMTruncatedError(
            f"completion truncated at max_tokens={max_tokens} "
            "(reasoning likely consumed the budget); raise max_tokens or shorten the input"
        )
    if not content.strip():
        raise LLMEmptyResponseError("model returned empty content")
    return content


class MiMoClient:
    """Chat completion via an OpenAI-compatible API (MiMo gateway).

    Raises ValueError at construction if required config is missing. Prefer
    MiMoClient.from_env() at startup to get None when the key is absent.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        requests_per_second: float = 10.0,
    ) -> None:
        if not model:
            raise ValueError("KE_MIMO_MODEL must be set")
        if not api_key:
            raise ValueError("KE_MIMO_API_KEY must be set")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") or "https://api.openai.com/v1"
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

        # Rate limiting (thread-safe token bucket)
        self._rate_interval = 1.0 / requests_per_second
        self._rate_lock = threading.Lock()
        self._rate_last = 0.0

    def _rate_limit(self) -> None:
        """Block until the rate limit allows the next request."""
        with self._rate_lock:
            now = time.monotonic()
            wait = self._rate_interval - (now - self._rate_last)
            if wait > 0:
                time.sleep(wait)
            self._rate_last = time.monotonic()

    @classmethod
    def from_env(cls) -> "MiMoClient | None":
        """Construct from environment, or return None if KE_MIMO_API_KEY absent."""
        key = os.environ.get("KE_MIMO_API_KEY", "")
        if not key:
            return None
        return cls(
            model=os.environ.get("KE_MIMO_MODEL", ""),
            api_key=key,
            base_url=os.environ.get("KE_MIMO_API_BASE_URL", ""),
            requests_per_second=float(os.environ.get("KE_MIMO_RATE_LIMIT", "10")),
        )

    def _clamp_max_tokens(self, max_tokens: int) -> int:
        """Clamp max_tokens to the model's actual output limit."""
        return min(max_tokens, _MIMO_MAX_OUTPUT_TOKENS)

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 8_000,
    ) -> str:
        """Return the assistant message content for a system+user prompt."""
        max_tokens = self._clamp_max_tokens(max_tokens)
        self._rate_limit()
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return _content_or_raise(data, max_tokens)

    def complete_sync(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 8_000,
    ) -> str:
        """Synchronous chat completion for sync call sites (engine.ingest is sync).

        Uses a short-lived synchronous httpx.Client per call (mirroring
        EmbeddingClient._post_sync) rather than asyncio.run() over a persistent
        AsyncClient. Reusing an async connection pool across the fresh event
        loops that repeated asyncio.run() creates raises 'Event loop is closed'
        on the 2nd+ call on Windows (proactor loop) — this avoids that entirely.
        """
        max_tokens = self._clamp_max_tokens(max_tokens)
        self._rate_limit()
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in Retrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            stop=stop_after_attempt(4),
            reraise=True,
        ):
            with attempt:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    return _content_or_raise(data, max_tokens)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def close(self) -> None:
        await self._client.aclose()
