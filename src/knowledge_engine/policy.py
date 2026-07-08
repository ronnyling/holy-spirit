from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DomainPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    minimum_score: float
    minimum_sources: int
    allow_user_only: bool = False


_DEFAULT_POLICIES = {
    "tcm": DomainPolicy(name="tcm", minimum_score=1.2, minimum_sources=2),
    "traditional chinese medicine": DomainPolicy(name="tcm", minimum_score=1.2, minimum_sources=2),
    "real estate": DomainPolicy(name="real_estate", minimum_score=0.8, minimum_sources=1),
    "real_estate": DomainPolicy(name="real_estate", minimum_score=0.8, minimum_sources=1),
    "trading": DomainPolicy(name="trading", minimum_score=1.2, minimum_sources=2),
    "stock trading": DomainPolicy(name="trading", minimum_score=1.2, minimum_sources=2),
}

_DEFAULT_POLICY = DomainPolicy(name="default", minimum_score=0.8, minimum_sources=1)
_DYNAMIC_POLICIES = {}


def get_domain_policy(domain):
    key = domain.strip().lower()
    return _DEFAULT_POLICIES.get(key) or _DYNAMIC_POLICIES.get(key) or _DEFAULT_POLICY


def register_domain(name, minimum_score=0.8, minimum_sources=1):
    policy = DomainPolicy(name=name, minimum_score=minimum_score, minimum_sources=minimum_sources)
    _DYNAMIC_POLICIES[name.strip().lower()] = policy
    return policy


def list_policy_domains(store=None):
    names = set(p.name for p in _DEFAULT_POLICIES.values())
    names.update(p.name for p in _DYNAMIC_POLICIES.values())
    if store is not None and hasattr(store, "list_domains"):
        try:
            names.update(store.list_domains())
        except Exception:
            pass
    return sorted(names)
