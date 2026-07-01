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


def get_domain_policy(domain: str) -> DomainPolicy:
    return _DEFAULT_POLICIES.get(domain.strip().lower(), _DEFAULT_POLICY)
