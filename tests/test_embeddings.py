"""Tests for the embedding client: Ollama-native transport + housekeeping lifecycle.

The lifecycle layer is deterministic (bounded) code: endpoint selection,
keep_alive passing, unload, and the warm() session. These tests mock the HTTP
transport (_post_sync) so they run without a live server; a final end-to-end
test hits Ollama directly and skips when it is not running.
"""

from __future__ import annotations

import httpx
import pytest

from knowledge_engine.embeddings import EmbeddingClient


def make_ollama(**kw) -> EmbeddingClient:
    params = dict(
        provider="ollama",
        model="bge-m3",
        api_key="",
        base_url="http://localhost:11434",
        dimensions=1024,
    )
    params.update(kw)
    return EmbeddingClient(**params)


# -- construction / config -----------------------------------------------------


def test_ollama_needs_no_api_key() -> None:
    client = make_ollama()
    assert client.provider == "ollama"
    assert client.keep_alive == "0"  # tight housekeeping by default


def test_non_ollama_still_requires_key() -> None:
    with pytest.raises(ValueError):
        EmbeddingClient(
            provider="openai",
            model="text-embedding-3-small",
            api_key="",
            base_url="https://api.openai.com/v1",
            dimensions=1536,
        )


def test_from_env_builds_ollama_without_key(monkeypatch) -> None:
    monkeypatch.setenv("KE_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("KE_EMBEDDING_MODEL", "bge-m3")
    monkeypatch.setenv("KE_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("KE_EMBEDDING_API_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("KE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("KE_EMBEDDING_KEEP_ALIVE", raising=False)

    client = EmbeddingClient.from_env()

    assert client is not None
    assert client.provider == "ollama"
    assert client.keep_alive == "0"


def test_from_env_returns_none_for_keyless_non_ollama(monkeypatch) -> None:
    monkeypatch.setenv("KE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("KE_EMBEDDING_API_KEY", raising=False)
    assert EmbeddingClient.from_env() is None


def test_native_root_strips_v1_suffix() -> None:
    client = make_ollama(base_url="http://localhost:11434/v1")
    assert client._ollama_root == "http://localhost:11434"


# -- transport routing + keep_alive --------------------------------------------


def test_embed_uses_native_endpoint_with_default_keep_alive(monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_post(self, url, payload, headers=None):
        calls.append((url, payload, headers))
        return {"embeddings": [[0.1] * 1024 for _ in payload["input"]]}

    monkeypatch.setattr(EmbeddingClient, "_post_sync", fake_post)
    client = make_ollama(keep_alive="0")

    out = client.embed_batch_sync(["a", "b"])

    assert len(out) == 2 and len(out[0]) == 1024
    url, payload, headers = calls[0]
    assert url.endswith("/api/embed")
    assert payload["model"] == "bge-m3"
    assert payload["keep_alive"] == "0"  # unload immediately when idle
    assert headers is None  # Ollama needs no auth header


def test_openai_path_targets_embeddings_endpoint_with_auth(monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_post(self, url, payload, headers=None):
        calls.append((url, payload, headers))
        return {"data": [{"index": 0, "embedding": [0.2, 0.2, 0.2]}]}

    monkeypatch.setattr(EmbeddingClient, "_post_sync", fake_post)
    client = EmbeddingClient(
        provider="openai",
        model="text-embedding-3-small",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        dimensions=3,
    )

    out = client.embed_sync("hello")

    url, payload, headers = calls[0]
    assert url.endswith("/embeddings") and not url.endswith("/api/embed")
    assert headers["Authorization"] == "Bearer sk-test"
    assert out == [0.2, 0.2, 0.2]


# -- housekeeping lifecycle ----------------------------------------------------


def test_warm_session_keeps_model_then_unloads_on_exit(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(self, url, payload, headers=None):
        calls.append(payload)
        return {"embeddings": [[0.0] * 1024 for _ in payload["input"]]}

    monkeypatch.setattr(EmbeddingClient, "_post_sync", fake_post)
    client = make_ollama(keep_alive="0")

    with client.warm("5m"):
        client.embed_sync("x")
        client.embed_sync("y")

    # Interior embeds kept the model resident for the run.
    assert calls[0]["keep_alive"] == "5m"
    assert calls[1]["keep_alive"] == "5m"
    # Exiting the session unloaded it (keep_alive 0).
    assert calls[-1]["keep_alive"] == 0
    # Default retention restored after the session.
    assert client._effective_keep_alive == "0"


def test_num_ctx_is_sent_in_ollama_options(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_post(self, url, payload, headers=None):
        captured.append(payload)
        return {"embeddings": [[0.0] * 1024 for _ in payload["input"]]}

    monkeypatch.setattr(EmbeddingClient, "_post_sync", fake_post)
    client = make_ollama(num_ctx=1024)

    client.embed_sync("short claim")

    assert captured[0]["options"]["num_ctx"] == 1024


def test_embed_texts_splits_into_batches(monkeypatch) -> None:
    batches: list[int] = []

    def fake_post(self, url, payload, headers=None):
        batches.append(len(payload["input"]))
        return {"embeddings": [[0.0] * 1024 for _ in payload["input"]]}

    monkeypatch.setattr(EmbeddingClient, "_post_sync", fake_post)
    client = make_ollama(batch_size=2)

    out = client.embed_texts(["a", "b", "c", "d", "e"])

    assert len(out) == 5
    assert batches == [2, 2, 1]  # 5 texts, batch_size 2 -> three requests


def test_unload_is_noop_for_non_ollama(monkeypatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(
        EmbeddingClient,
        "_post_sync",
        lambda self, *a, **k: called.append(1) or {},
    )
    client = EmbeddingClient(
        provider="openai",
        model="m",
        api_key="sk",
        base_url="https://api.openai.com/v1",
        dimensions=3,
    )
    client.unload_sync()
    assert called == []


def test_unload_swallows_transport_errors(monkeypatch) -> None:
    def boom(self, *a, **k):
        raise httpx.ConnectError("ollama down")

    monkeypatch.setattr(EmbeddingClient, "_post_sync", boom)
    client = make_ollama()
    # Housekeeping must never raise into the caller's main flow.
    client.unload_sync()


# -- end-to-end (skips when Ollama is not running) -----------------------------


def _ollama_running() -> bool:
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not _ollama_running(), reason="Ollama server not running")
def test_live_ollama_embed_returns_1024_dims() -> None:
    client = make_ollama(keep_alive="0")
    vector = client.embed_sync("suburban cap rate around 7 percent")
    assert len(vector) == 1024
    # keep_alive=0 asked Ollama to release the model right after the call.
    client.unload_sync()
