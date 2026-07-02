"""LLM-backed domain classification (drop-folder auto-classify layer).

BOUNDED vs UNBOUNDED split (per the locked architecture):
  * UNBOUNDED (LLM): read a transcript sample and decide which domain it is.
  * BOUNDED (this module): constrain the answer to the FIXED, known domain set
    (`policy.py`) and map anything else to UNKNOWN. Domain selects the per-domain
    evidence gate, so a misclassification would route claims to the wrong
    promotion bar. Therefore we NEVER invent a domain and NEVER default-guess:
    an unrecognized/low-confidence answer returns None (UNKNOWN) so the caller
    can flag the file for human review (human-in-the-loop, no fallbacks).
"""

from __future__ import annotations

from typing import Protocol

from .policy import get_domain_policy

# The canonical, human-readable domain labels the classifier may emit. These map
# to real (non-"default") policies in policy.py. UNKNOWN is intentionally absent.
KNOWN_DOMAINS: tuple[str, ...] = ("trading", "real estate", "tcm")

_SAMPLE_CHARS = 2000

_SYSTEM_PROMPT = (
    "You classify an expert transcript into exactly ONE knowledge domain for a "
    "knowledge base. Allowed domains:\n"
    "- trading: stock/market/technical trading, positions, entries, dividends.\n"
    "- real estate: property investment, rentals, cap rates, valuations.\n"
    "- tcm: traditional chinese medicine, herbs, patterns, diagnosis.\n"
    "Rules:\n"
    "1. Answer with ONLY the exact domain label (trading, real estate, or tcm).\n"
    "2. If the transcript does not clearly and confidently belong to exactly one "
    "of these, answer UNKNOWN.\n"
    "3. No prose, no punctuation, no explanation \u2014 just the label or UNKNOWN."
)

_OPEN_SYSTEM_PROMPT = (
    "You identify the knowledge domain of an expert transcript. "
    "Rules:\n"
    "1. Answer with a SHORT domain label (2-4 words max, all lowercase, "
    "e.g. 'trading', 'real estate', 'tcm', 'nutrition', 'software engineering').\n"
    "2. Choose the most specific domain that clearly applies.\n"
    "3. If the transcript does not belong to a recognisable expert domain, "
    "answer UNKNOWN.\n"
    "4. No prose, no punctuation, no explanation \u2014 just the label or UNKNOWN."
)


class SupportsComplete(Protocol):
    def complete_sync(self, *, system: str, user: str) -> str: ...


class DomainClassifier:
    """Classify a transcript into a known domain, or None (UNKNOWN)."""

    def __init__(self, client: SupportsComplete) -> None:
        self._client = client

    def classify(self, *, transcript_text: str, entity_name: str = "") -> str | None:
        """Return a known domain label, or None when the domain is UNKNOWN.

        Only a leading sample of the transcript is sent \u2014 classification does not
        need the whole document, which keeps this call cheap even for long files.
        """
        sample = transcript_text.strip()[:_SAMPLE_CHARS]
        if not sample:
            return None

        header = f"Topic hint: {entity_name}\n" if entity_name.strip() else ""
        user = f"{header}Transcript sample:\n{sample}\n"
        raw = self._client.complete_sync(system=_SYSTEM_PROMPT, user=user)
        return self._parse(raw)

    def classify_open(self, *, transcript_text: str, entity_name: str = "") -> str | None:
        """Return any domain label the LLM names, or None when UNKNOWN.

        Unlike ``classify()``, this is NOT bounded to the predefined KNOWN_DOMAINS
        list.  Use this as a fallback when the strict classifier returns None so
        novel domains are named, registered with default evidence bars, and ingested
        rather than silently dropped.
        """
        import re

        sample = transcript_text.strip()[:_SAMPLE_CHARS]
        if not sample:
            return None
        header = f"Topic hint: {entity_name}\n" if entity_name.strip() else ""
        user = f"{header}Transcript sample:\n{sample}\n"
        raw = self._client.complete_sync(system=_OPEN_SYSTEM_PROMPT, user=user)
        token = (raw or "").strip().lower()
        if not token or "unknown" in token:
            return None
        # Sanitise: keep only alphanumeric + spaces + hyphens.
        token = re.sub(r"[^a-z0-9 \-]", "", token).strip()
        return token or None

    @staticmethod
    def _parse(raw: str) -> str | None:
        """Map free-form model output to a known domain label, else None. Bounded."""
        token = (raw or "").strip().lower()
        if not token:
            return None

        # Prefer an exact/substring match against the known labels. Check the more
        # specific multi-word label first so "real estate" isn't shadowed.
        for label in ("real estate", "trading", "tcm"):
            if label in token:
                return _validated(label)

        # Common aliases the model might emit.
        aliases = {
            "real_estate": "real estate",
            "property": "real estate",
            "stock trading": "trading",
            "stocks": "trading",
            "traditional chinese medicine": "tcm",
            "chinese medicine": "tcm",
        }
        for alias, label in aliases.items():
            if alias in token:
                return _validated(label)

        return None  # UNKNOWN — never guessed, never defaulted.


def _validated(label: str) -> str | None:
    """Defensive: only return a label that maps to a real (non-default) policy."""
    return label if get_domain_policy(label).name != "default" else None
