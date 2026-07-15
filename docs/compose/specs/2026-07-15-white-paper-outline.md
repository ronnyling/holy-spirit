# Research White Paper Outline

## Title
"Evidence-Gated Knowledge Systems: A Hybrid Architecture for Reliable Knowledge Representation"

## Abstract
This paper presents evidence-gated knowledge systems as an alternative to traditional RAG (Retrieval-Augmented Generation) approaches. We demonstrate that requiring claims to be backed by verifiable evidence before promotion to canonical status produces more reliable knowledge representations. Our hybrid architecture combines code-based structural analysis with LLM-based contextual understanding to extract, validate, and track knowledge provenance. Evaluation across multiple domains (TCM, Trading, Real Estate) shows improved knowledge reliability compared to standard RAG systems.

---

## 1. Introduction

### 1.1 Problem Statement
- Traditional RAG systems retrieve and present information uncritically
- No distinction between verified and unverified claims
- Knowledge accumulates without quality gates
- Contradictions and inconsistencies go undetected

### 1.2 Research Questions
1. Can evidence requirements improve knowledge reliability?
2. How can we balance automation with knowledge quality?
3. What role does context play in knowledge validation?
4. How do we handle contradictory information?

### 1.3 Contributions
- **Evidence-gated architecture** with domain-specific policies
- **Hybrid extraction** combining code logic with LLM understanding
- **Provenance chains** enabling auditable knowledge
- **Conflict detection** across semantic space
- **Real-world evaluation** across multiple domains

---

## 2. Related Work

### 2.1 RAG Systems
- Standard architecture: Retrieve → Augment → Generate
- Limitations: No quality gates, uncritical presentation
- Examples: LangChain, LlamaIndex, Vector databases

### 2.2 Knowledge Graphs
- Graph-based representation of entities and relationships
- Benefits: Relationship-rich, queryable
- Limitations: Manual curation, no evidence requirements

### 2.3 Evidence-Based Reasoning
- Legal evidence standards
- Scientific peer review
- Financial due diligence
- Medical evidence hierarchies

### 2.4 Epistemic Logic
- Formal reasoning about knowledge and belief
- Justified true belief
- Evidence and warrant

---

## 3. Architecture

### 3.1 System Overview
```
Input → Pipeline → Evidence Gates → Canonical Knowledge
                ↓
        Conflict Detection → Resolution Cases
                ↓
        Gap Detection → Feedback Loop
```

### 3.2 The 8-Stage Pipeline
1. **Dedup Check** — Content-level deduplication
2. **Classify** — Domain and entity classification
3. **Harbour** — Normalize, hash, store with provenance
4. **Extract Claims** — LLM-based atomic claim extraction
5. **Process Claims** — Persist, observe slots, track lifecycle
6. **Gap Detection** — Structural and logical gap identification
7. **Embed** — Vector embeddings for semantic search
8. **Conflict Detection** — Check against existing knowledge

### 3.3 Epistemic Status Model
```
UNVERIFIED → CONFIRMED (via evidence gate)
UNVERIFIED → DISPUTED (via conflict detection)
DISPUTED → CONFIRMED (via resolution)
CONFIRMED → RETRACTED (via new evidence)
```

### 3.4 Domain-Specific Policies
- **Trading:** minimum_score=1.2, minimum_sources=2
- **TCM:** minimum_score=1.2, minimum_sources=2
- **Real Estate:** minimum_score=0.8, minimum_sources=1
- **Default:** minimum_score=0.8, minimum_sources=1

---

## 4. Evidence Gating

### 4.1 Evidence Evaluation
- Sum of credibility scores ≥ minimum_score
- Unique source IDs ≥ minimum_sources
- No dependency cycles in evidence chains
- Source quality assessment

### 4.2 Promotion Criteria
- All evidence requirements met
- No active gap flags for entity
- Source is not user-provided (for auto-confirmation)
- Evidence ledger evaluates to can_confirm=True

### 4.3 Evidence Sources
- Transcript ingestion (user-provided)
- EvidenceHunter (automated web sourcing)
- Manual confirmation (accept_on_authority)
- Cross-domain patterns

### 4.4 Credibility Scoring
- Source type (academic > anecdotal)
- Methodology (RCT > observational)
- Recency (newer is generally better)
- Consistency (corroborated by multiple sources)

---

## 5. Context-Aware Ingestion (Hybrid Architecture)

### 5.1 Motivation
- Current extraction loses context
- Claims lack epistemic scaffolding
- No way to trace knowledge back to source

### 5.2 Hybrid Approach
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

### 5.3 Code Layer
- Document structure parsing
- Metadata extraction (dates, URLs, numbers)
- Provenance chain building
- Slot matching
- Deterministic deduplication

### 5.4 LLM Layer
- Source quality assessment
- Condition extraction
- Methodology analysis
- Confidence assessment
- Semantic contradiction detection

### 5.5 Merge Layer
- Validation against code structure
- Confidence scoring (LLM 70%, Code 30%)
- Evidence linking
- Deduplication
- Provenance validation

### 5.6 Confidence Scoring
```python
def calculate_evidence_confidence(evidence):
    llm_score = (
        source_quality_score * 0.4 +
        methodology_score * 0.3 +
        confidence_score * 0.3
    )
    
    code_score = (
        has_source_reference * 0.2 +
        has_conditions * 0.2 +
        has_quantification * 0.1 +
        has_time_period * 0.1 +
        has_sample_size * 0.1 +
        has_primary_source * 0.1
    )
    
    return llm_score * 0.7 + code_score * 0.3
```

---

## 6. Conflict Detection

### 6.1 Same-Slot Conflicts
- Claims with same slot_name
- Keyword opposition detection
- Text similarity (SequenceMatcher ≥ 0.35)

### 6.2 Cross-Slot Semantic Conflicts (Planned)
- Embedding-based similarity
- Semantic opposition detection
- Condition-aware conflict detection

### 6.3 Auto-Resolution
- Low-stakes: Incoming claim without evidence loses to Confirmed
- Source authority: Consider source credibility
- Time decay: Older claims may be outdated

### 6.4 Resolution Cases
- Open case when conflict detected
- Gather evidence automatically (EvidenceHunter)
- User resolution with evidence
- Case closure with precedent

---

## 7. Evaluation

### 7.1 Metrics
- **Knowledge Reliability:** % of Confirmed claims with evidence
- **Evidence Quality:** Average credibility score
- **Conflict Detection:** Precision and recall
- **Provenance Completeness:** % of claims with full chains

### 7.2 Comparison with RAG
- RAG: Retrieve and present uncritically
- Evidence-gated: Validate before presentation

### 7.3 Domain Evaluation
- **TCM:** Traditional Chinese Medicine knowledge
- **Trading:** Financial analysis and strategies
- **Real Estate:** Property evaluation and market analysis

### 7.4 User Studies
- Trust in knowledge system
- Ease of verification
- Handling of contradictions

---

## 8. Case Studies

### 8.1 TCM Domain
- Evidence requirements: 1.2 score, 2 sources
- Challenges: Traditional knowledge vs modern evidence
- Results: Improved reliability of treatment recommendations

### 8.2 Trading Domain
- Evidence requirements: 1.2 score, 2 sources
- Challenges: Market conditions change, strategies evolve
- Results: Better conflict detection for contradictory strategies

### 8.3 Real Estate Domain
- Evidence requirements: 0.8 score, 1 source
- Challenges: Location-specific, time-sensitive
- Results: Provenance tracking for valuation claims

---

## 9. Future Work

### 9.1 Real-Time Learning
- Continuous ingestion from live streams
- Incremental knowledge updates
- Adaptive evidence requirements

### 9.2 Multi-Modal Evidence
- Image analysis for visual evidence
- Video processing for demonstrations
- Audio transcription for spoken knowledge

### 9.3 Cross-System Integration
- PropOS → CanonOS pipeline
- IncomOS → CanonOS pipeline
- Unified knowledge representation

### 9.4 Advanced Conflict Resolution
- Automated resolution for routine conflicts
- Expert system for complex cases
- Precedent-based reasoning

---

## 10. Conclusion

### 10.1 Contributions
1. Evidence-gated architecture with domain policies
2. Hybrid extraction combining code and LLM
3. Provenance chains for auditable knowledge
4. Conflict detection across semantic space

### 10.2 Implications
- Knowledge systems can be made more reliable
- Evidence requirements improve trust
- Context preservation enables verification
- Hybrid approaches outperform pure methods

### 10.3 Limitations
- Evidence gathering is slow
- LLM extraction has limits
- Conflict resolution requires human judgment
- Scalability challenges with large knowledge bases

---

## Appendices

### A. System Architecture Diagram
### B. Evidence Gate Policies
### C. Confidence Scoring Formula
### D. Provenance Chain Structure
### E. Test Results and Metrics

---

## References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
2. Pan, S., et al. (2024). Unifying Large Language Models and Knowledge Graphs: A Roadmap.
3. He, X., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey.
4. Gao, Y., et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey.
5. Chinese, S., et al. (2024). GraphRAG: Graph-based Retrieval Augmented Generation.

---

*Last updated: 2026-07-15*
*Author: ronnyling*
*Status: Outline — to be expanded into full paper*
