"""Conflict detection with domain-specific keyword opposition.

NO FALLBACKS: All detection methods must either detect or not detect,
never silently fail or return ambiguous results.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher


# Load domain-specific keyword pairs from JSON config
_KEYWORD_PAIRS_PATH = Path(__file__).parent / "keyword_pairs.json"

def _load_keyword_pairs() -> Dict[str, Dict[str, str]]:
    """Load keyword pairs from JSON config file."""
    if not _KEYWORD_PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"Keyword pairs config not found: {_KEYWORD_PAIRS_PATH}. "
            "Cannot detect conflicts without keyword pairs."
        )

    with open(_KEYWORD_PAIRS_PATH, "r") as f:
        data = json.load(f)

    # Validate structure
    if not isinstance(data, dict):
        raise ValueError("Keyword pairs config must be a dictionary")

    for domain, pairs in data.items():
        if not isinstance(pairs, dict):
            raise ValueError(f"Domain '{domain}' must have dictionary of pairs")

    return data


class ConflictDetector:
    """Detect conflicts between claims with domain-specific keyword opposition."""

    def __init__(self, threshold: float = 0.35, domain: str = "general"):
        """Initialize conflict detector.

        Args:
            threshold: Text similarity threshold for conflict detection
            domain: Domain for keyword pairs (trading, tcm, real_estate, general)

        Raises:
            FileNotFoundError: If keyword pairs config not found
            ValueError: If config structure invalid
        """
        self.threshold = threshold
        self.domain = domain
        self._keyword_pairs = _load_keyword_pairs()

        # Validate domain exists
        if domain not in self._keyword_pairs:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Available domains: {list(self._keyword_pairs.keys())}"
            )

    @property
    def opposition_pairs(self) -> Dict[str, str]:
        """Get keyword opposition pairs for current domain."""
        return self._keyword_pairs[self.domain]

    def detect(
        self,
        new_claim: Dict[str, Any],
        existing_claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between new claim and existing claims.

        NO FALLBACKS: Returns empty list only if no conflicts detected.
        """
        if not new_claim.get("statement"):
            raise ValueError("new_claim must have a 'statement' field")

        conflicts = []

        for existing in existing_claims:
            if existing.get("status") not in ["CONFIRMED", "DISPUTED"]:
                continue

            if not existing.get("statement"):
                continue

            # Same-slot conflict
            if new_claim.get("slot_name") == existing.get("slot_name"):
                if self._detect_same_slot_conflict(new_claim, existing):
                    conflicts.append({
                        "claim_id": new_claim["id"],
                        "conflicting_claim_id": existing["id"],
                        "conflict_type": "same_slot",
                        "confidence": 0.9
                    })
                    continue

            # Keyword opposition
            if self._detect_keyword_opposition(
                new_claim["statement"],
                existing["statement"]
            ):
                conflicts.append({
                    "claim_id": new_claim["id"],
                    "conflicting_claim_id": existing["id"],
                    "conflict_type": "keyword_opposition",
                    "confidence": 0.7
                })
                continue

            # Text similarity
            similarity = self._calculate_similarity(
                new_claim["statement"],
                existing["statement"]
            )
            if similarity >= self.threshold:
                conflicts.append({
                    "claim_id": new_claim["id"],
                    "conflicting_claim_id": existing["id"],
                    "conflict_type": "text_similarity",
                    "confidence": similarity
                })

        return conflicts

    def _detect_same_slot_conflict(
        self,
        claim_a: Dict[str, Any],
        claim_b: Dict[str, Any]
    ) -> bool:
        """Detect conflict within same slot."""
        return self._detect_keyword_opposition(
            claim_a["statement"],
            claim_b["statement"]
        )

    def _detect_keyword_opposition(self, text_a: str, text_b: str) -> bool:
        """Detect keyword opposition between texts.

        Checks both directions: if text_a has word X and text_b has opposite(X)
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        # Check if any word in text_a has its opposite in text_b
        for word_a in words_a:
            if word_a in self.opposition_pairs:
                opposite = self.opposition_pairs[word_a]
                if opposite in words_b:
                    return True

        # Also check reverse direction
        for word_b in words_b:
            if word_b in self.opposition_pairs:
                opposite = self.opposition_pairs[word_b]
                if opposite in words_a:
                    return True

        return False

    def _calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate text similarity using SequenceMatcher."""
        return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

    def get_available_domains(self) -> List[str]:
        """Get list of available domains for keyword pairs."""
        return list(self._keyword_pairs.keys())

    def get_domain_pairs(self, domain: str) -> Dict[str, str]:
        """Get keyword pairs for a specific domain.

        Raises:
            ValueError: If domain not found
        """
        if domain not in self._keyword_pairs:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Available domains: {list(self._keyword_pairs.keys())}"
            )
        return self._keyword_pairs[domain]
