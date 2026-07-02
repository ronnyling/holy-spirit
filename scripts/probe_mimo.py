"""Ad-hoc probe of the MiMo gateway. Not committed logic — UAT diagnostics only.
Tests (1) chat completions and (2) whether an OpenAI-compatible /embeddings
endpoint exists. Prints capability findings so we know what we can wire.
"""
from __future__ import annotations

import json
import os
import sys

import httpx

BASE = os.environ.get("KE_MIMO_API_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1").rstrip("/")
KEY = os.environ.get("KE_MIMO_API_KEY", "")
MODEL = os.environ.get("KE_MIMO_MODEL", "mimo-v2.5")

if not KEY:
    print("NO KEY", file=sys.stderr)
    raise SystemExit(1)

headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def probe_chat() -> None:
    print("=== CHAT /chat/completions ===")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You reply with strict JSON only."},
            {"role": "user", "content": 'Return {"ok": true} and nothing else.'},
        ],
        "temperature": 0,
        "max_tokens": 50,
    }
    try:
        r = httpx.post(f"{BASE}/chat/completions", headers=headers, json=payload, timeout=60)
        print("status:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            msg = data["choices"][0]["message"]["content"]
            print("content:", repr(msg))
        else:
            print("body:", r.text[:500])
    except Exception as exc:  # noqa: BLE001 - diagnostics
        print("ERROR:", type(exc).__name__, exc)


def probe_embeddings() -> None:
    print("\n=== EMBEDDINGS /embeddings ===")
    for model in ("text-embedding-3-small", MODEL, "mimo-embedding"):
        payload = {"model": model, "input": "cap rate real estate test"}
        try:
            r = httpx.post(f"{BASE}/embeddings", headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()
                dim = len(data["data"][0]["embedding"])
                print(f"model={model!r} status=200 OK dim={dim}")
                return
            print(f"model={model!r} status={r.status_code} body={r.text[:200]}")
        except Exception as exc:  # noqa: BLE001 - diagnostics
            print(f"model={model!r} ERROR {type(exc).__name__}: {exc}")


def probe_models() -> None:
    print("\n=== MODELS /models ===")
    try:
        r = httpx.get(f"{BASE}/models", headers=headers, timeout=30)
        print("status:", r.status_code)
        if r.status_code == 200:
            print(json.dumps(r.json(), indent=2)[:800])
        else:
            print("body:", r.text[:300])
    except Exception as exc:  # noqa: BLE001 - diagnostics
        print("ERROR:", type(exc).__name__, exc)


if __name__ == "__main__":
    print(f"BASE={BASE} MODEL={MODEL}\n")
    probe_models()
    probe_chat()
    probe_embeddings()
