"""Automated evidence sourcing for Unverified claims.

The EvidenceHunter closes the evidence gap without removing the evidence gate.
Given an Unverified claim it:

1. Generates a **neutral, informational** search query (not confirmation-seeking
   — ask "what is known about X" not "prove X is true").
2. Executes a web search via a pluggable SearchProvider (Tavily default).
3. Extracts evidence fragments from results using the LLM.
4. Evaluates fragments through the existing per-domain EvidenceLedger.
5. On pass  → auto-promotes claim to Confirmed (no human needed).
   On fail  → records the attempt and surfaces what evidence is still needed as
              a natural-language advisory for the user.

Domain-specific credibility ceilings (enforced regardless of LLM score):
  real estate : 0.7  (market data and legal precedent are web-findable)
  tcm         : 0.4  (lineage corroboration required; web alone cannot confirm)
  trading     : 0.3  (empirical bar — no backtest means no confirmation)
  default     : 0.5

The evidence gate is NOT made optional.  This module is an automated evidence
*supplier*; the domain bar still determines whether the found evidence is
sufficient.  The epistemic distinction between Confirmed and Unverified is
fully preserved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .contracts import EvidenceDraft
from .evidence import EvidenceLedger
from .models import Claim, EpistemicStatus
from .policy import get_domain_policy

if TYPE_CHECKING:
    from .graph.neo4j_store import KnowledgeGraphStore
    from .store import KnowledgeStore

# ---------------------------------------------------------------------------
# Domain credibility ceilings for web-sourced evidence
# ---------------------------------------------------------------------------

_WEB_CREDIBILITY_CEILINGS: dict[str, float] = {
    "real estate": 0.7,
    "real_estate": 0.7,
    "tcm": 0.4,
    "traditional chinese medicine": 0.4,
    "trading": 0.3,
    "stock trading": 0.3,
}
_DEFAULT_WEB_CREDIBILITY = 0.5


def _web_credibility_ceiling(domain: str) -> float:
    return _WEB_CREDIBILITY_CEILINGS.get(domain.strip().lower(), _DEFAULT_WEB_CREDIBILITY)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class SearchProvider(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[dict]: ...


class LLMClient(Protocol):
    def complete_sync(self, *, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HuntResult:
    claim_id: str
    domain: str
    promoted: bool = False
    evidence_count: int = 0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    what_is_needed: str | None = None
    attempted_query: str | None = None


# ---------------------------------------------------------------------------
# EvidenceHunter
# ---------------------------------------------------------------------------

class EvidenceHunter:
    """Automated evidence sourcing with domain-aware credibility ceilings.

    Parameters
    ----------
    llm_client      : any client with ``complete_sync(system, user) -> str``
    search_provider : a SearchProvider implementation (Tavily, Brave, etc.)
    ledger          : the shared EvidenceLedger instance
    """

    def __init__(
        self,
        llm_client: LLMClient,
        search_provider: SearchProvider,
        ledger: EvidenceLedger,
    ) -> None:
        self._llm = llm_client
        self._search = search_provider
        self._ledger = ledger

    def hunt(
        self,
        claim: Claim,
        domain: str,
        store: KnowledgeStore | KnowledgeGraphStore,
        *,
        max_results: int = 5,
    ) -> HuntResult:
        """Hunt for evidence for an Unverified claim.

        Returns a HuntResult describing whether the claim was auto-promoted.
        If not, ``what_is_needed`` explains what evidence would satisfy the gate.
        """
        result = HuntResult(claim_id=claim.id or "", domain=domain)

        # 1. Generate a neutral informational search query.
        query = self._generate_query(claim, domain)
        result.attempted_query = query

        # 2. Execute search.
        try:
            search_results = self._search.search(query, max_results=max_results)
        except Exception as exc:
            result.reasons.append(f"web search failed: {exc}")
            result.what_is_needed = self._what_is_needed(domain, claim.statement)
            return result

        if not search_results:
            result.reasons.append("web search returned no results")
            result.what_is_needed = self._what_is_needed(domain, claim.statement)
            return result

        # 3. Extract evidence fragments.
        ceiling = _web_credibility_ceiling(domain)
        drafts = self._extract_evidence(claim, domain, search_results, ceiling)
        result.evidence_count = len(drafts)

        if not drafts:
            result.reasons.append("no relevant evidence extracted from search results")
            result.what_is_needed = self._what_is_needed(domain, claim.statement)
            return result

        # 4. Evaluate against the domain evidence bar.
        evaluation = self._ledger.evaluate(domain, drafts, store, claim)
        result.score = evaluation.score
        result.reasons = list(evaluation.reasons)

        if evaluation.can_confirm:
            # 5. Record evidence and promote the claim.
            self._ledger.record_for_claim(claim, drafts, store)
            claim.epistemic_status = EpistemicStatus.CONFIRMED
            store.set_claim_status(
                claim_id=claim.id or "", status=str(claim.epistemic_status)
            )
            result.promoted = True
        else:
            result.what_is_needed = self._what_is_needed(
                domain, claim.statement, reasons=evaluation.reasons
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_query(self, claim: Claim, domain: str) -> str:
        """Generate a neutral informational query — not confirmation-seeking."""
        system = (
            "You generate a neutral web search query to find authoritative information "
            "about a topic. The query should INFORM about the topic, not confirm a "
            "specific conclusion. Domain: " + domain + ".\n"
            "Rules: return ONLY the search query, no quotes, no explanation, max 15 words."
        )
        user = f"Claim to research: {claim.statement}"
        try:
            return self._llm.complete_sync(system=system, user=user).strip()[:200]
        except Exception:
            return claim.statement[:200]

    def _extract_evidence(
        self,
        claim: Claim,
        domain: str,
        search_results: list[dict],
        credibility_ceiling: float,
    ) -> list[EvidenceDraft]:
        """Score each search result as evidence for the claim."""
        drafts: list[EvidenceDraft] = []
        for res in search_results:
            title = res.get("title", "")
            url = res.get("url") or res.get("href", "")
            content = (res.get("content") or res.get("body", ""))[:2000]
            if not content:
                continue

            system = (
                "You evaluate whether a web source provides evidence relevant to a claim.\n"
                "Domain: " + domain + ".\n"
                "Respond with exactly two lines:\n"
                "SCORE: <float between 0.0 and " + f"{credibility_ceiling:.1f}" + ">\n"
                "RATIONALE: <one sentence>\n"
                "Score 0.0 = irrelevant, "
                + f"{credibility_ceiling:.1f}"
                + " = strong direct evidence (domain ceiling)."
            )
            user = f"Claim: {claim.statement}\n\nSource title: {title}\nContent: {content}"
            try:
                raw = self._llm.complete_sync(system=system, user=user)
                score_line = next(
                    (ln for ln in raw.splitlines() if ln.startswith("SCORE:")), ""
                )
                score_str = score_line.replace("SCORE:", "").strip()
                score = min(float(score_str), credibility_ceiling) if score_str else 0.0
                if score >= 0.2:
                    drafts.append(
                        EvidenceDraft(
                            source_kind="external_doc",
                            source_id=url or title,
                            source_ref=url or None,
                            credibility=score,
                            notes=f"web: {title[:120]}",
                        )
                    )
            except Exception:
                continue
        return drafts

    def _what_is_needed(
        self,
        domain: str,
        statement: str,
        reasons: list[str] | None = None,
    ) -> str:
        """Human-readable description of what evidence is still needed."""
        policy = get_domain_policy(domain)
        base = (
            f"To confirm this claim in the '{domain}' domain the system needs "
            f"at least {policy.minimum_sources} independent source(s) with a "
            f"combined credibility score \u2265 {policy.minimum_score:.1f}."
        )
        domain_key = domain.strip().lower()
        if domain_key in ("trading", "stock trading"):
            base += (
                " Trading claims require empirical evidence (backtests, quantitative "
                "data, or peer-reviewed research) \u2014 web articles alone cannot "
                "satisfy this bar."
            )
        elif domain_key in ("tcm", "traditional chinese medicine"):
            base += (
                " TCM claims require corroboration across at least two independent "
                "lineages or a classical textual citation."
            )
        if reasons:
            base += " Current shortfall: " + "; ".join(reasons[:2]) + "."
        return base


# ---------------------------------------------------------------------------
# Built-in search providers
# ---------------------------------------------------------------------------

class TavilySearchProvider:
    """Tavily AI search (https://docs.tavily.com).

    Install: ``pip install tavily-python``
    API key: set ``KE_TAVILY_API_KEY`` env var or pass ``api_key`` directly.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("KE_TAVILY_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Tavily API key is required. Set KE_TAVILY_API_KEY or pass api_key."
            )

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        from tavily import TavilyClient  # type: ignore[import-untyped]

        client = TavilyClient(api_key=self._api_key)
        response = client.search(query=query, max_results=max_results)
        return response.get("results", [])


def build_evidence_hunter(llm_client: LLMClient, ledger: EvidenceLedger) -> EvidenceHunter:
    """Construct an EvidenceHunter using Tavily as the search provider.

    Raises ``ValueError`` if ``KE_TAVILY_API_KEY`` is not set.
    """
    provider = TavilySearchProvider()
    return EvidenceHunter(llm_client=llm_client, search_provider=provider, ledger=ledger)
