from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DomainPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    minimum_score: float
    minimum_sources: int
    allow_user_only: bool = False


_DEFAULT_POLICIES: dict[str, DomainPolicy] = {
    "tcm": DomainPolicy(name="tcm", minimum_score=1.2, minimum_sources=2),
    "traditional chinese medicine": DomainPolicy(name="tcm", minimum_score=1.2, minimum_sources=2),
    "real estate": DomainPolicy(name="real_estate", minimum_score=0.8, minimum_sources=1),
    "real_estate": DomainPolicy(name="real_estate", minimum_score=0.8, minimum_sources=1),
    "trading": DomainPolicy(name="trading", minimum_score=1.2, minimum_sources=2),
    "stock trading": DomainPolicy(name="trading", minimum_score=1.2, minimum_sources=2),
}

_DEFAULT_POLICY = DomainPolicy(name="default", minimum_score=0.8, minimum_sources=1)

# Domains registered at runtime (via auto-classification of new transcripts).
_DYNAMIC_POLICIES: dict[str, DomainPolicy] = {}


def get_domain_policy(domain: str) -> DomainPolicy:
    key = domain.strip().lower()
    return _DEFAULT_POLICIES.get(key) or _DYNAMIC_POLICIES.get(key) or _DEFAULT_POLICY


def register_domain(
    name: str,
    minimum_score: float = 0.8,
    minimum_sources: int = 1,
) -> DomainPolicy:
    """Register a new domain with default evidence bars. Safe to call multiple times."""
    policy = DomainPolicy(name=name, minimum_score=minimum_score, minimum_sources=minimum_sources)
    _DYNAMIC_POLICIES[name.strip().lower()] = policy
    return policy


def list_policy_domains() -> list[str]:
    """All registered domain names (static + dynamic, deduplicated canonical forms)."""
    names: set[str] = set()
    for p in _DEFAULT_POLICIES.values():
        names.add(p.name)
    for p in _DYNAMIC_POLICIES.values():
        names.add(p.name)
    return sorted(names)
