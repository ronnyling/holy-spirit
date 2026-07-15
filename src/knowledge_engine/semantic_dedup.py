from typing import List, Dict, Any
from difflib import SequenceMatcher

class SemanticDeduplicator:
    """Find semantic duplicates across claims."""
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
    
    def find_duplicates(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find duplicate claims using text similarity."""
        if len(claims) < 2:
            return []
        
        duplicates = []
        seen = set()
        
        for i, claim_a in enumerate(claims):
            for j, claim_b in enumerate(claims):
                if i >= j:
                    continue
                
                pair_key = tuple(sorted([claim_a["id"], claim_b["id"]]))
                if pair_key in seen:
                    continue
                
                similarity = self._calculate_similarity(
                    claim_a["statement"],
                    claim_b["statement"]
                )
                
                if similarity >= self.threshold:
                    duplicates.append({
                        "claim_ids": [claim_a["id"], claim_b["id"]],
                        "similarity": similarity,
                        "statements": [claim_a["statement"], claim_b["statement"]]
                    })
                    seen.add(pair_key)
        
        return duplicates
    
    def _calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate text similarity using SequenceMatcher."""
        return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
