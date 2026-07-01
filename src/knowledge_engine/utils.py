from __future__ import annotations

from difflib import SequenceMatcher
from hashlib import sha256
import re

_WORD_RE = re.compile(r"[^a-z0-9]+")
_MODAL_RE = re.compile(
    r"\b(should|always|never|best|effective|works?|can|may|must|guaranteed|profit|profitable|cause|causes|treat|treats)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    normalized = _WORD_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def stable_signature(*parts: str) -> str:
    joined = "||".join(parts)
    return sha256(joined.encode("utf-8")).hexdigest()[:16]


def needs_context(text: str) -> bool:
    return bool(_MODAL_RE.search(text))
