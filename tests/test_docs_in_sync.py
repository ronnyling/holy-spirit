"""Guardrail: keep the human docs pinned to code (single source of truth).

The facts that keep drifting between README.md, the wiki, and the agent
definition are all *volatile numbers* or *abandoned-plan wording*. This test
makes code the source of truth for those numbers and fails the build when a
doc restates them wrongly or when a known-stale architecture marker reappears.

SSOT policy this enforces:
- Code is canonical for numbers: evidence gates live in ``policy.py``, slot
  promotion thresholds in ``learning.py``. Docs restate them for humans.
- ``wiki/`` is canonical for design prose; ``README.md`` summarises + links.
- Volatile status (e.g. the current passing-test count) lives in ONE place
  (README) and is never hardcoded a second time.

If this test fails, fix the doc to match code (or, for a deliberate change,
update code first — then the doc, then this test's expectations follow the
code automatically).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from knowledge_engine.learning import SlotLearner
from knowledge_engine.policy import get_domain_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DOMAIN_POLICIES = REPO_ROOT / "wiki" / "Domain-Policies.md"

# The agent definition lives in the workspace .github folder, one level above
# the repo. Guard it only when it is present (a standalone clone won't have it).
AGENT_MD = REPO_ROOT.parent / ".github" / "agents" / "knowledge-engine.agent.md"

# (display label used in docs, a domain key that resolves to that policy)
_GATE_ROWS = [
    ("Trading", "trading"),
    ("TCM", "tcm"),
    ("Real estate", "real estate"),
    ("Default", "an-unmapped-domain-hits-the-default-policy"),
]

# Wording from the abandoned document-first / Postgres plan, plus obsolete
# status strings. None of these may appear in the docs any more.
STALE_MARKERS = [
    "document-first",
    "postgres",
    "pgvector",
    "psycopg",
    "relational first",
    "relational-first",
    "neo4j only later",
    "neo4j later",
    "43 passed",
    "returns 401",
]


def _norm(text: str) -> str:
    """Collapse runs of spaces/tabs so markdown table spacing is irrelevant."""
    return re.sub(r"[ \t]+", " ", text)


def _fmt_score(score: float) -> str:
    return f"{score:g}"


@pytest.mark.parametrize("doc", [README, DOMAIN_POLICIES], ids=lambda p: p.name)
def test_evidence_gates_match_policy_code(doc: Path) -> None:
    text = _norm(doc.read_text(encoding="utf-8"))
    for label, domain_key in _GATE_ROWS:
        policy = get_domain_policy(domain_key)
        row = re.compile(
            rf"\|\s*{re.escape(label)}\s*\|\s*"
            rf"{re.escape(_fmt_score(policy.minimum_score))}\s*\|\s*"
            rf"{policy.minimum_sources}\s*\|",
            re.IGNORECASE,
        )
        assert row.search(text), (
            f"{doc.name}: evidence-gate row for '{label}' should be "
            f"{_fmt_score(policy.minimum_score)} / {policy.minimum_sources} "
            f"to match policy.py, but no matching table row was found."
        )


def test_slot_thresholds_match_learning_code() -> None:
    learner = SlotLearner()
    text = README.read_text(encoding="utf-8")
    assert f"Candidate at {learner.candidate_threshold}" in text, (
        "README slot-promotion threshold for Candidate does not match "
        f"SlotLearner.candidate_threshold={learner.candidate_threshold}."
    )
    assert f"Expected at {learner.expected_threshold}" in text, (
        "README slot-promotion threshold for Expected does not match "
        f"SlotLearner.expected_threshold={learner.expected_threshold}."
    )


def _guarded_docs() -> list[Path]:
    docs = sorted((REPO_ROOT / "wiki").glob("*.md"))
    docs.append(README)
    if AGENT_MD.exists():
        docs.append(AGENT_MD)
    return docs


@pytest.mark.parametrize("doc", _guarded_docs(), ids=lambda p: p.name)
def test_no_stale_architecture_markers(doc: Path) -> None:
    lower = doc.read_text(encoding="utf-8").lower()
    hits = sorted(marker for marker in STALE_MARKERS if marker in lower)
    assert not hits, (
        f"{doc.name}: found stale marker(s) {hits}. The docs must reflect the "
        f"graph-first Neo4j architecture, not the abandoned document-first/"
        f"Postgres plan (and must not hardcode an obsolete test count)."
    )
