from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class EvidenceDraft:
    """Extracted from transcript by hybrid pipeline."""
    statement: str
    source_reference: str
    
    # LLM-assessed factors
    source_quality: str  # "academic", "anecdotal", "commercial", "unknown"
    source_quality_score: float  # 0.0-1.0
    conditions: List[str] = field(default_factory=list)
    measurement_method: str = "unknown"
    methodology_score: float = 0.5  # 0.0-1.0
    confidence_indicator: str = "medium"  # "high", "medium", "low", "uncertain"
    confidence_score: float = 0.5  # 0.0-1.0
    
    # Code-extracted factors
    has_quantification: bool = False
    has_time_period: bool = False
    has_sample_size: bool = False
    has_primary_source: bool = False
    
    # Relationships
    contradicts: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    
    # Provenance
    document_id: str = ""
    transcript_id: str = ""
    extraction_method: str = "llm_hybrid"
    
    def calculate_confidence(self) -> float:
        """Combine LLM assessment with code-based rules."""
        llm_score = (
            self.source_quality_score * 0.4 +
            self.methodology_score * 0.3 +
            self.confidence_score * 0.3
        )
        
        code_score = 0.0
        if self.source_reference:
            code_score += 0.2
        if self.conditions:
            code_score += 0.2
        if self.has_quantification:
            code_score += 0.1
        if self.has_time_period:
            code_score += 0.1
        if self.has_sample_size:
            code_score += 0.1
        if self.has_primary_source:
            code_score += 0.1
        
        return llm_score * 0.7 + code_score * 0.3
    
    @classmethod
    def from_llm_output(
        cls,
        llm_output: Dict[str, Any],
        document_id: str = "",
        transcript_id: str = ""
    ) -> "EvidenceDraft":
        """Create EvidenceDraft from LLM extraction output."""
        indicator_map = {"high": 0.85, "medium": 0.5, "low": 0.2, "uncertain": 0.3}
        quality_map = {"academic": 0.9, "commercial": 0.5, "anecdotal": 0.3, "unknown": 0.2}
        method_map = {"randomized controlled trial": 0.95, "observational": 0.6, "case study": 0.4, "unknown": 0.3}
        
        return cls(
            statement=llm_output.get("statement", ""),
            source_reference=llm_output.get("source_reference", ""),
            source_quality=llm_output.get("source_quality", "unknown"),
            source_quality_score=quality_map.get(llm_output.get("source_quality", "unknown"), 0.2),
            conditions=llm_output.get("conditions", []),
            measurement_method=llm_output.get("measurement_method", "unknown"),
            methodology_score=method_map.get(llm_output.get("measurement_method", "unknown"), 0.3),
            confidence_indicator=llm_output.get("confidence_indicator", "medium"),
            confidence_score=indicator_map.get(llm_output.get("confidence_indicator", "medium"), 0.5),
            has_quantification=llm_output.get("has_quantification", False),
            has_time_period=llm_output.get("has_time_period", False),
            has_sample_size=llm_output.get("has_sample_size", False),
            has_primary_source=llm_output.get("has_primary_source", False),
            contradicts=llm_output.get("contradicts", []),
            supports=llm_output.get("supports", []),
            document_id=document_id,
            transcript_id=transcript_id,
            extraction_method="llm_hybrid"
        )
