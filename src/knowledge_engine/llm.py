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

import asyncio
import os

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


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
    ) -> None:
        if not model:
            raise ValueError("KE_MIMO_MODEL must be set")
        if not api_key:
            raise ValueError("KE_MIMO_API_KEY must be set")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") or "https://api.openai.com/v1"
        self._client = httpx.AsyncClient(timeout=timeout)

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
        )

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
        max_tokens: int = 2000,
    ) -> str:
        """Return the assistant message content for a system+user prompt."""
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
        return data["choices"][0]["message"]["content"]

    def complete_sync(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> str:
        """Synchronous wrapper for sync call sites (engine.ingest is sync)."""
        return asyncio.run(
            self.complete(
                system=system, user=user, temperature=temperature, max_tokens=max_tokens
            )
        )

    async def close(self) -> None:
        await self._client.aclose()
