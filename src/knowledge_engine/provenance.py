from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ProvenanceChain:
    """Complete trace from claim to source."""
    claim_id: str
    evidence_ids: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    transcript_ids: List[str] = field(default_factory=list)
    
    # Metadata at each level
    claim_metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_metadata: List[Dict[str, Any]] = field(default_factory=list)
    document_metadata: List[Dict[str, Any]] = field(default_factory=list)
    transcript_metadata: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_id": self.claim_id,
            "evidence_ids": self.evidence_ids,
            "document_ids": self.document_ids,
            "transcript_ids": self.transcript_ids,
            "claim_metadata": self.claim_metadata,
            "evidence_metadata": self.evidence_metadata,
            "document_metadata": self.document_metadata,
            "transcript_metadata": self.transcript_metadata
        }
    
    @classmethod
    def from_store_data(cls, claim_id: str, store_data: Dict[str, Any]) -> "ProvenanceChain":
        """Create ProvenanceChain from store data."""
        claim = store_data.get("claim", {})
        evidence = store_data.get("evidence", [])
        documents = store_data.get("documents", [])
        transcripts = store_data.get("transcripts", [])
        
        return cls(
            claim_id=claim_id,
            evidence_ids=[e.get("id", "") for e in evidence],
            document_ids=[d.get("id", "") for d in documents],
            transcript_ids=[t.get("id", "") for t in transcripts],
            claim_metadata=claim,
            evidence_metadata=evidence,
            document_metadata=documents,
            transcript_metadata=transcripts
        )
