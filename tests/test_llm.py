"""Regression tests for the MiMo sync transport.

The network is faked (no calls leave the process). These guard the fix for
`RuntimeError: Event loop is closed`, which occurred when complete_sync reused a
persistent httpx.AsyncClient across the fresh event loops that repeated
asyncio.run() creates. complete_sync now opens a short-lived httpx.Client per
call, so calling it many times in a row must simply work.
"""

from __future__ import annotations

import knowledge_engine.llm as llm_mod
from knowledge_engine.llm import (
    LLMEmptyResponseError,
    LLMTruncatedError,
    MiMoClient,
    _content_or_raise,
    _is_retryable,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeClient:
    """Stands in for httpx.Client as a context manager."""

    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).instances += 1

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def post(self, url, json, headers):  # noqa: A002 - mirrors httpx signature
        return _FakeResponse("ok")


def test_complete_sync_repeated_calls_do_not_reuse_a_loop(monkeypatch):
    _FakeClient.instances = 0
    monkeypatch.setattr(llm_mod.httpx, "Client", _FakeClient)
    # asyncio must NOT be used by the sync path; there is no import to reach for.
    assert not hasattr(llm_mod, "asyncio")

    client = MiMoClient(model="m", api_key="k", base_url="http://example/v1")

    # Two back-to-back calls previously raised 'Event loop is closed' on Windows.
    assert client.complete_sync(system="s", user="u") == "ok"
    assert client.complete_sync(system="s", user="u") == "ok"
    assert client.complete_sync(system="s", user="u") == "ok"

    # A fresh short-lived client per call is the whole point of the fix.
    assert _FakeClient.instances == 3


def test_complete_sync_sends_system_and_user_messages(monkeypatch):
    captured: dict = {}

    class _CapturingClient(_FakeClient):
        def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse("done")

    monkeypatch.setattr(llm_mod.httpx, "Client", _CapturingClient)

    client = MiMoClient(model="mimo-v2.5", api_key="secret", base_url="http://example/v1")
    out = client.complete_sync(system="SYS", user="USR")

    assert out == "done"
    assert captured["url"] == "http://example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    roles = [(m["role"], m["content"]) for m in captured["json"]["messages"]]
    assert roles == [("system", "SYS"), ("user", "USR")]
    assert captured["json"]["model"] == "mimo-v2.5"


def _choice(content, finish_reason="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def test_content_or_raise_returns_nonempty_content():
    assert _content_or_raise(_choice("hello"), 4000) == "hello"


def test_content_or_raise_raises_on_empty_content():
    # Reasoning model returned nothing usable -> loud, NOT a silent empty string.
    for empty in ("", "   ", None):
        try:
            _content_or_raise(_choice(empty), 4000)
        except LLMEmptyResponseError:
            continue
        raise AssertionError(f"expected LLMEmptyResponseError for {empty!r}")


def test_content_or_raise_raises_on_truncation():
    # finish_reason=length means the answer was cut off -> actionable hard error.
    try:
        _content_or_raise(_choice("[{\"statement\":", finish_reason="length"), 4000)
    except LLMTruncatedError as exc:
        assert "max_tokens" in str(exc)
    else:
        raise AssertionError("expected LLMTruncatedError")


def test_retry_predicate_retries_empty_but_not_truncation():
    # Empty content is a transient glitch (retry); truncation won't fix itself at
    # the same budget (fail loud).
    assert _is_retryable(LLMEmptyResponseError()) is True
    assert _is_retryable(LLMTruncatedError()) is False
