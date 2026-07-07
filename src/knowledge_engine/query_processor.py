"""Query processing for retrieval quality.

Optional LLM-powered enhancements:
1. Domain-aware expansion: adds domain-specific synonyms
2. Query decomposition: splits complex queries into sub-queries

Both are bounded by LLM call budget and can be disabled via config.

Configuration:
    KE_QUERY_EXPAND (true/false, default false)
    KE_QUERY_DECOMPOSE (true/false, default false)
    KE_QUERY_MAX_LLM_CALLS (default 3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


class SupportsComplete(Protocol):
    def complete_sync(self, *, system: str, user: str, max_tokens: int = 8_000) -> str: ...


@dataclass
class ProcessedQuery:
    """Result of query processing: original + optional expansions/decompositions."""

    original: str
    expansions: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    llm_calls_used: int = 0

    @property
    def all_queries(self) -> list[str]:
        """Return all query variants (original + expansions + sub-queries), deduplicated."""
        seen: set[str] = set()
        queries: list[str] = []
        for q in [self.original] + self.expansions + self.sub_queries:
            normalized = q.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                queries.append(q)
        return queries


class QueryProcessor:
    """LLM-powered query expansion and decomposition.

    Both features are optional and independently toggleable. The processor
    enforces a hard call budget to prevent runaway LLM usage.
    """

    def __init__(
        self,
        llm_client: SupportsComplete | None = None,
        *,
        expand: bool = False,
        decompose: bool = False,
        max_calls: int = 3,
    ) -> None:
        self.llm = llm_client
        self.expand = expand and llm_client is not None
        self.decompose = decompose and llm_client is not None
        self.max_calls = max_calls

    @classmethod
    def from_env(cls, llm_client: SupportsComplete | None = None) -> QueryProcessor:
        """Construct from environment variables."""
        return cls(
            llm_client,
            expand=os.environ.get("KE_QUERY_EXPAND", "false").lower() == "true",
            decompose=os.environ.get("KE_QUERY_DECOMPOSE", "false").lower() == "true",
            max_calls=int(os.environ.get("KE_QUERY_MAX_LLM_CALLS", "3")),
        )

    def process(self, query: str, domain: str | None = None) -> ProcessedQuery:
        """Process a query with optional expansion and decomposition.

        Returns a ProcessedQuery with the original query plus any
        expansions and sub-queries generated within the call budget.
        """
        result = ProcessedQuery(original=query)
        calls_remaining = self.max_calls

        if self.expand and calls_remaining > 0:
            expansions = self._expand(query, domain, budget=calls_remaining)
            result.expansions = expansions
            result.llm_calls_used += len(expansions)
            calls_remaining -= len(expansions)

        if self.decompose and calls_remaining > 0:
            sub_queries = self._decompose(query, budget=calls_remaining)
            result.sub_queries = sub_queries
            result.llm_calls_used += len(sub_queries)

        return result

    def _expand(self, query: str, domain: str | None, budget: int) -> list[str]:
        """Generate domain-specific query expansions. Bounded to 1 LLM call."""
        if budget < 1:
            return []

        system = (
            "You are a domain expert. Generate 2-4 alternative phrasings or "
            "synonyms for the following search query. Focus on domain-specific "
            "terminology that might appear in expert transcripts. Return ONLY "
            "the alternative phrasings, one per line, no numbering or bullets."
        )
        user = f"Domain: {domain or 'general'}\nQuery: {query}"

        raw = self.llm.complete_sync(system=system, user=user, max_tokens=200)  # type: ignore[union-attr]
        expansions = [
            line.strip().lstrip("0123456789.-) ")
            for line in raw.strip().splitlines()
            if line.strip() and line.strip().lower() != query.strip().lower()
        ]
        return expansions[:4]  # hard cap

    def _decompose(self, query: str, budget: int) -> list[str]:
        """Decompose a complex query into sub-queries. Bounded to 2 LLM calls."""
        if budget < 1:
            return []

        system = (
            "You are a search query analyst. If the query contains multiple "
            "distinct questions or topics, decompose it into 2-4 focused "
            "sub-queries. If the query is already focused, return it unchanged. "
            "Return ONLY the sub-queries, one per line, no numbering or bullets."
        )
        user = f"Query: {query}"

        raw = self.llm.complete_sync(system=system, user=user, max_tokens=200)  # type: ignore[union-attr]
        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in raw.strip().splitlines()
            if line.strip()
        ]
        # If decomposition returned the original query unchanged, no decomposition needed
        if len(lines) == 1 and lines[0].lower().strip() == query.lower().strip():
            return []
        return lines[:4]  # hard cap
