# Context-Aware Ingestion Design Spec

## [S1] Problem

The knowledge engine extracts **claims** from transcripts but loses **epistemic scaffolding** — the chain of "how do we know this?" that makes knowledge reliable.

### Current State
```
Transcript → LLM extracts ClaimDraft(statement, slot_name) → Claim(evidence=[])
```

**What's lost:**
- Source quality ("TCM studies" vs "someone's blog")
- Conditions ("in joint pain patients", "with 40mg dosage")
- Measurement methodology ("randomized controlled trial" vs "observational")
- Confidence indicators (explicit uncertainty)
- Provenance chains (can't trace back to original context)

### Impact
- All claims start UNVERIFIED with `evidence=[]`
- System stores assertions but not the reasoning behind them
- No way to audit "why does the system believe X?"
- Evidence gates can never be satisfied from transcript content alone

## [S2] Solution Overview

**Hybrid architecture** combining code logic (deterministic, fast) with LLM (contextual, flexible):

```
┌─────────────────────────────────────────────────────────┐
│                    HYBRID EXTRACTION                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  CODE LAYER  │    │  LLM LAYER   │    │  MERGE     │ │
│  │              │    │              │    │            │ │
│  │ • Doc struct │    │ • Context    │    │ • Validate │ │
│  │ • Metadata   │───▶│ • Evidence   │───▶│ • Dedup    │ │
│  │ • Provenance │    │ • Conditions │    │ • Link     │ │
│  │ • Slot match │    │ • Quality    │    │            │ │
│  └──────────────┘    └──────────────┘    └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## [S3] Responsibility Matrix

### Code Layer Responsibilities

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Document Structure Parser** | Identify sections, paragraphs, citations, references | Regex + heuristics for markdown/text structure |
| **Provenance Chain Builder** | Link evidence→document→transcript with metadata | Graph traversal, foreign key relationships |
| **Slot Matcher** | Match evidence to claims via slot_name | Exact match on structured slot names |
| **Metadata Extractor** | Extract dates, URLs, email, numbers, percentages | Regex patterns, NER for entities |
| **Dedup Engine** | Exact match deduplication across evidence | Hash-based, normalized text comparison |
| **Confidence Scorer (Rules)** | Score evidence based on code-extractable factors | Weighted formula: source_ref, conditions, quantification |

### LLM Layer Responsibilities

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Context Extractor** | Understand what the transcript is actually saying | Structured prompt with few-shot examples |
| **Source Quality Assessor** | Judge if source is academic, anecdotal, commercial, etc. | Classification prompt with examples |
| **Condition Extractor** | Identify when/where/under what circumstances claims apply | Extraction prompt for conditions |
| **Methodology Analyzer** | Determine how measurements were obtained | Classification: RCT, observational, case study, etc. |
| **Confidence Assessor** | Evaluate how confident the system should be | Assessment prompt with uncertainty indicators |
| **Semantic Contradiction Detector** | Find contradictions across different slots | Comparison prompt with existing claims |

### Merge Layer Responsibilities

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Validation Gate** | Check LLM output against code-extracted structure | Schema validation, required fields |
| **Confidence Merger** | Combine LLM assessment with code-based scoring | Weighted average: LLM 70%, Code 30% |
| **Evidence Linker** | Match evidence to claims via slot + context | Slot match + semantic similarity |
| **Dedup Merger** | Deduplicate evidence from multiple sources | Hash + semantic similarity threshold |
| **Provenance Validator** | Ensure provenance chains are complete | Check all required links exist |

## [S4] Data Models

### Enhanced Evidence Model

```python
@dataclass
class EvidenceDraft:
    """Extracted from transcript by hybrid pipeline."""
    statement: str                    # "40% reduction in inflammation"
    source_reference: str             # "paragraph 3, sentence 2"
    
    # LLM-assessed factors
    source_quality: str               # "academic", "anecdotal", "commercial", "unknown"
    source_quality_score: float       # 0.0-1.0, LLM confidence in quality assessment
    conditions: List[str]             # ["joint pain patients", "12-week period"]
    measurement_method: str           # "randomized controlled trial", "observational"
    methodology_score: float          # 0.0-1.0, LLM confidence in methodology
    confidence_indicator: str         # "high", "medium", "low", "uncertain"
    confidence_score: float           # 0.0-1.0, normalized from indicator
    
    # Code-extracted factors
    has_quantification: bool          # Contains numbers/percentages
    has_time_period: bool             # Specifies duration
    has_sample_size: bool             # Mentions n= or participants
    has_primary_source: bool          # References original study vs二手
    
    # Relationships
    contradicts: List[str]            # ["claim_id_123"] if explicit contradiction found
    supports: List[str]               # ["claim_id_456"] if supports existing claim
    
    # Provenance
    document_id: str                  # Link to source document
    transcript_id: str                # Link to original transcript
    extraction_method: str            # "llm_hybrid", "code_only", "manual"
```

### Enhanced Claim Model

```python
@dataclass
class Claim:
    """Claim with evidence chain."""
    id: str
    statement: str
    slot_name: str
    entity_id: str
    domain: str
    epistemic_status: EpistemicStatus  # UNVERIFIED, DISPUTED, CONFIRMED
    
    # Evidence chain
    evidence: List[Evidence]           # Linked evidence objects
    evidence_score: float              # Sum of credibility scores
    evidence_sources: int              # Count of unique sources
    
    # Provenance
    source_transcript_id: str          # Original transcript
    source_document_id: str            # Source document
    created_at: datetime
    updated_at: datetime
    version: int                       # Incremented on status changes
    
    # Confidence
    confidence_score: float            # Overall confidence (0.0-1.0)
    confidence_breakdown: Dict[str, float]  # Factor contributions
```

### Provenance Chain Model

```python
@dataclass
class ProvenanceChain:
    """Complete trace from claim to source."""
    claim_id: str
    evidence_ids: List[str]
    document_ids: List[str]
    transcript_ids: List[str]
    
    # Metadata at each level
    claim_metadata: Dict[str, Any]
    evidence_metadata: List[Dict[str, Any]]
    document_metadata: List[Dict[str, Any]]
    transcript_metadata: List[Dict[str, Any]]
```

## [S5] Extraction Pipeline (Enhanced)

### Current Pipeline (8 stages)
```
1. Dedup Check → 2. Classify → 3. Harbour → 4. Extract Claims → 
5. Process Claims → 6. Gap Detection → 7. Embed → 8. Conflict Detection
```

### Enhanced Pipeline (10 stages)
```
1. Dedup Check → 2. Classify → 3. Harbour → 4. Extract Claims (Enhanced) → 
5. Extract Evidence (NEW) → 6. Process Claims + Evidence → 
7. Gap Detection → 8. Embed → 9. Conflict Detection (Enhanced) → 10. Mark Complete
```

### Stage 4: Extract Claims (Enhanced)

**Current:** LLM extracts `ClaimDraft(statement, slot_name)`

**Enhanced:** LLM extracts `ClaimDraft(statement, slot_name, evidence_drafts[])`

```
Prompt template:
"""
Extract claims AND supporting evidence from this transcript.

For each claim, also extract:
- Evidence statements that support it
- Source quality (academic/anecdotal/commercial/unknown)
- Conditions under which the claim applies
- How the measurement was obtained
- Confidence level

Return JSON:
{
  "claims": [
    {
      "statement": "turmeric reduces inflammation",
      "slot_name": "treatment_outcome",
      "evidence": [
        {
          "statement": "40% reduction in inflammation",
          "source_quality": "academic",
          "conditions": ["joint pain patients", "12-week period"],
          "measurement_method": "randomized controlled trial",
          "confidence_indicator": "medium"
        }
      ]
    }
  ]
}
"""
```

### Stage 5: Extract Evidence (NEW)

**Purpose:** Parallel evidence extraction for richer context

```python
def extract_evidence(transcript: str, claims: List[ClaimDraft]) -> List[EvidenceDraft]:
    """Extract evidence context for each claim."""
    
    # Code layer: Extract structural metadata
    doc_structure = parse_document_structure(transcript)
    metadata = extract_metadata(transcript)  # dates, URLs, numbers
    
    # LLM layer: Extract contextual evidence
    evidence_drafts = []
    for claim in claims:
        llm_evidence = llm_extract_evidence(transcript, claim)
        evidence_drafts.extend(llm_evidence)
    
    # Merge layer: Combine and validate
    merged_evidence = merge_evidence(evidence_drafts, doc_structure, metadata)
    
    return merged_evidence
```

### Stage 6: Process Claims + Evidence

**Current:** Process claims, observe slots

**Enhanced:** Process claims AND evidence, link evidence to claims

```python
def process_claims_with_evidence(claims: List[ClaimDraft], evidence: List[EvidenceDraft]):
    """Process claims and link evidence."""
    
    for claim in claims:
        # Store claim
        store.add_claim(claim)
        
        # Find evidence for this claim
        claim_evidence = [e for e in evidence if matches_claim(e, claim)]
        
        # Link evidence to claim
        for ev in claim_evidence:
            store.add_evidence(ev)
            store.link_evidence_to_claim(ev.id, claim.id)
        
        # Calculate evidence score
        claim.evidence_score = sum(e.credibility for e in claim_evidence)
        claim.evidence_sources = len(set(e.source_id for e in claim_evidence))
        
        # Check if claim can be auto-confirmed
        if can_confirm_claim(claim, claim_evidence):
            claim.epistemic_status = EpistemicStatus.CONFIRMED
```

## [S6] Confidence Scoring System

### Evidence Confidence Formula

```python
def calculate_evidence_confidence(evidence: EvidenceDraft) -> float:
    """Combine LLM assessment with code-based rules."""
    
    # LLM-assessed factors (0-1 each)
    llm_score = (
        evidence.source_quality_score * 0.4 +    # "academic" > "anecdotal"
        evidence.methodology_score * 0.3 +       # RCT > observational
        evidence.confidence_score * 0.3          # LLM's assessment
    )
    
    # Code-based factors
    code_score = 0.0
    if evidence.has_source_reference:
        code_score += 0.2  # Can trace back
    if evidence.conditions:
        code_score += 0.2  # Specifies when
    if evidence.has_quantification:
        code_score += 0.1  # Numbers, not just claims
    if evidence.has_time_period:
        code_score += 0.1  # Specifies duration
    if evidence.has_sample_size:
        code_score += 0.1  # Mentions participants
    if evidence.has_primary_source:
        code_score += 0.1  # Original vs二手
    
    # Weighted combination (LLM higher for context understanding)
    return llm_score * 0.7 + code_score * 0.3
```

### Claim Confidence Formula

```python
def calculate_claim_confidence(claim: Claim, evidence: List[Evidence]) -> float:
    """Overall claim confidence based on evidence chain."""
    
    if not evidence:
        return 0.0
    
    # Individual evidence confidence scores
    evidence_scores = [e.confidence_score for e in evidence]
    
    # Source diversity bonus
    unique_sources = len(set(e.source_id for e in evidence))
    source_diversity = min(unique_sources / 3.0, 1.0)  # Cap at 3 sources
    
    # Condition coverage bonus
    has_conditions = any(e.conditions for e in evidence)
    condition_bonus = 0.1 if has_conditions else 0.0
    
    # Methodology bonus
    methodology_scores = [e.methodology_score for e in evidence if e.methodology_score]
    avg_methodology = sum(methodology_scores) / len(methodology_scores) if methodology_scores else 0.5
    
    # Weighted combination
    base_score = sum(evidence_scores) / len(evidence_scores)
    final_score = (
        base_score * 0.5 +
        source_diversity * 0.2 +
        avg_methodology * 0.2 +
        condition_bonus * 0.1
    )
    
    return min(final_score, 1.0)
```

## [S7] Provenance Chain Implementation

### Storage Structure

```
knowledge_engine/
├── transcripts/
│   ├── {transcript_id}.txt          # Raw text
│   └── {transcript_id}.meta.json    # Metadata, chunks
├── documents/
│   ├── {document_id}.md             # Rendered document
│   └── {document_id}.meta.json      # Provenance links
├── evidence/
│   ├── {evidence_id}.json           # Evidence object
│   └── {evidence_id}.meta.json      # Links to claim, document
└── provenance/
    └── {claim_id}.json              # Full provenance chain
```

### Provenance Query API

```python
class KnowledgeEngine:
    def trace_claim(self, claim_id: str) -> ProvenanceChain:
        """Get complete provenance chain for a claim."""
        
        claim = store.get_claim(claim_id)
        evidence = store.get_evidence_for_claim(claim_id)
        documents = [store.get_document(e.document_id) for e in evidence]
        transcripts = [store.get_transcript(d.transcript_id) for d in documents]
        
        return ProvenanceChain(
            claim_id=claim_id,
            evidence_ids=[e.id for e in evidence],
            document_ids=[d.id for d in documents],
            transcript_ids=[t.id for t in transcripts],
            claim_metadata=claim.to_dict(),
            evidence_metadata=[e.to_dict() for e in evidence],
            document_metadata=[d.to_dict() for d in documents],
            transcript_metadata=[t.to_dict() for t in transcripts]
        )
    
    def get_evidence_for_context(
        self, 
        claim_id: str, 
        conditions: Optional[List[str]] = None
    ) -> List[Evidence]:
        """Get evidence matching specific conditions."""
        
        evidence = store.get_evidence_for_claim(claim_id)
        
        if conditions:
            evidence = [
                e for e in evidence
                if any(c in e.conditions for c in conditions)
            ]
        
        return sorted(evidence, key=lambda e: e.confidence_score, reverse=True)
```

## [S8] Conflict Detection Enhancement

### Current: Slot-Limited
Only detects conflicts between claims with the same `slot_name`.

### Enhanced: Cross-Slot Semantic Detection

```python
def detect_conflicts_enhanced(
    new_claim: Claim,
    existing_claims: List[Claim],
    embedding_client: EmbeddingClient
) -> List[Conflict]:
    """Detect conflicts across slots using semantic similarity."""
    
    conflicts = []
    
    # Get embedding for new claim
    new_embedding = embedding_client.embed(new_claim.statement)
    
    for existing_claim in existing_claims:
        # Skip if same claim
        if existing_claim.id == new_claim.id:
            continue
        
        # Skip if not active status
        if existing_claim.epistemic_status not in [
            EpistemicStatus.CONFIRMED, 
            EpistemicStatus.DISPUTED
        ]:
            continue
        
        # Same-slot conflict (existing logic)
        if new_claim.slot_name == existing_claim.slot_name:
            if detect_same_slot_conflict(new_claim, existing_claim):
                conflicts.append(Conflict(
                    claim_id=new_claim.id,
                    conflicting_claim_id=existing_claim.id,
                    conflict_type="same_slot",
                    confidence=0.9
                ))
                continue
        
        # Cross-slot semantic conflict (NEW)
        existing_embedding = embedding_client.embed(existing_claim.statement)
        similarity = cosine_similarity(new_embedding, existing_embedding)
        
        # Check for semantic opposition
        if similarity > 0.7:  # Similar topic
            if detect_semantic_opposition(new_claim, existing_claim):
                conflicts.append(Conflict(
                    claim_id=new_claim.id,
                    conflicting_claim_id=existing_claim.id,
                    conflict_type="cross_slot_semantic",
                    confidence=similarity * 0.8  # Discount for cross-slot
                ))
    
    return conflicts
```

## [S9] Implementation Phases

### Phase 1: Enhanced Extraction (Week 1-2)
- [ ] Create `EvidenceDraft` data model
- [ ] Modify `ClaimExtractor` prompt to extract evidence
- [ ] Add evidence storage to `store`
- [ ] Link evidence to claims
- [ ] Basic confidence scoring (LLM only)

### Phase 2: Code Layer (Week 3-4)
- [ ] Implement `DocumentStructureParser`
- [ ] Implement `MetadataExtractor`
- [ ] Implement `ProvenanceChainBuilder`
- [ ] Add code-based confidence factors
- [ ] Merge LLM + code confidence scores

### Phase 3: Provenance Chains (Week 5-6)
- [ ] Implement `trace_claim()` API
- [ ] Implement `get_evidence_for_context()` API
- [ ] Add provenance storage structure
- [ ] Add provenance validation

### Phase 4: Enhanced Conflict Detection (Week 7-8)
- [ ] Implement cross-slot semantic detection
- [ ] Add embedding-based conflict scoring
- [ ] Integrate with existing conflict resolution
- [ ] Add conflict confidence scoring

## [S10] Testing Strategy

### Unit Tests
- Evidence extraction accuracy
- Confidence scoring formula
- Provenance chain completeness
- Conflict detection precision/recall

### Integration Tests
- End-to-end ingestion with evidence extraction
- Provenance query responses
- Conflict detection across slots
- Auto-confirmation with evidence

### Manual Tests
- Trace claim back to source document
- Verify evidence context is preserved
- Check confidence scores are reasonable
- Validate conflict detection catches contradictions

## [S11] Success Criteria

- [ ] Evidence extracted from transcripts (not just claims)
- [ ] Provenance chains complete and queryable
- [ ] Confidence scores reflect source quality
- [ ] Cross-slot conflicts detected
- [ ] Claims can be traced to original context
- [ ] Auto-confirmation works with transcript evidence
- [ ] All existing tests pass
- [ ] New tests for enhanced features pass

## [S12] Future Considerations

### White Paper Integration
This design should be documented as part of the research white paper on:
- Evidence-gated knowledge systems
- Hybrid extraction architectures
- Provenance-aware knowledge graphs
- Context-aware conflict detection

### Evolution Path
- **v1.0**: Current pipeline (claims only)
- **v2.0**: Enhanced extraction (claims + evidence) ← This design
- **v3.0**: Real-time learning (continuous ingestion)
- **v4.0**: Multi-modal evidence (images, videos, audio)
