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


# /embeddings and /models endpoints do not exist on MiMo gateway (404/405 responses)


if __name__ == "__main__":
    print(f"BASE={BASE} MODEL={MODEL}\n")
    print("NOTE: MiMo gateway only supports /chat/completions (no /models or /embeddings endpoints)\n")
    probe_chat()
