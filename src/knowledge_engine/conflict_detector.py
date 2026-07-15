from typing import List, Dict, Any
from difflib import SequenceMatcher

# Keyword opposition pairs
OPPOSITION_PAIRS = {
    "buy": "sell", "increase": "decrease", "rise": "fall",
    "positive": "negative", "bullish": "bearish", "up": "down"
}

class ConflictDetector:
    """Detect conflicts between claims."""
    
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
    
    def detect(
        self,
        new_claim: Dict[str, Any],
        existing_claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between new claim and existing claims."""
        conflicts = []
        
        for existing in existing_claims:
            if existing.get("status") not in ["CONFIRMED", "DISPUTED"]:
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
            if self._detect_keyword_opposition(new_claim["statement"], existing["statement"]):
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
        # Check for keyword opposition
        return self._detect_keyword_opposition(
            claim_a["statement"],
            claim_b["statement"]
        )
    
    def _detect_keyword_opposition(self, text_a: str, text_b: str) -> bool:
        """Detect keyword opposition between texts."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        for word_a in words_a:
            if word_a in OPPOSITION_PAIRS:
                opposite = OPPOSITION_PAIRS[word_a]
                if opposite in words_b:
                    return True
        
        return False
    
    def _calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate text similarity."""
        return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
