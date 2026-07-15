# Knowledge Engine Evolution Journey

> This document tracks the evolution of the knowledge_engine from its initial conception to its current state and future direction. It serves as the foundation for the research white paper on evidence-gated knowledge systems.

## Executive Summary

The knowledge_engine is a **evidence-gated knowledge system** that builds reliable representations through a structured pipeline of extraction, validation, and promotion. Unlike traditional RAG systems that retrieve and present information uncritically, knowledge_engine evaluates evidence quality and promotes claims only when they pass domain-specific evidence gates.

**Core Thesis:** Evidence-gated knowledge systems produce more reliable representations than RAG by requiring claims to be backed by verifiable evidence before promotion to canonical status.

---

## Phase 1: Foundation (2026-01 to 2026-03)

### Initial Concept
- **Goal:** Build a knowledge system that doesn't just retrieve information, but evaluates it
- **Inspiration:** Legal evidence standards, scientific peer review, financial due diligence
- **Key Insight:** Knowledge should be earned through evidence, not just accumulated

### Architecture Decisions
1. **Graph-based storage** (Neo4j) for relationship-rich knowledge
2. **Vector embeddings** for semantic search
3. **Epistemic status model** (Unverified → Confirmed/Disputed)
4. **Domain-specific policies** for evidence requirements

### Early Challenges
- How to handle contradictory information?
- How to prevent information overload?
- How to maintain knowledge quality as volume grows?

---

## Phase 2: Core Pipeline (2026-03 to 2026-05)

### The 8-Stage Pipeline
Developed the core ingestion pipeline that processes transcripts into validated knowledge:

```
1. Dedup Check → 2. Classify → 3. Harbour → 4. Extract Claims → 
5. Process Claims → 6. Gap Detection → 7. Embed → 8. Conflict Detection
```

### Key Innovations

#### Harbour System
- **Concept:** "Port of entry" for knowledge
- **Function:** Normalize, hash, store, create provenance
- **Benefit:** Idempotent processing, crash recovery, audit trail

#### Claim Extraction
- **Approach:** LLM-based extraction of atomic factual claims
- **Innovation:** Claims start as UNVERIFIED — no fabrication
- **Benefit:** Prevents false confidence in unvalidated information

#### Evidence Gating
- **Concept:** Claims require evidence to be promoted
- **Implementation:** Domain-specific policies (Trading: 1.2/2, TCM: 1.2/2)
- **Benefit:** Knowledge quality scales with evidence quality

#### Conflict Detection
- **Approach:** Check new claims against existing Confirmed claims
- **Innovation:** Auto-resolution for low-stakes conflicts
- **Benefit:** Prevents unverified claims from displacing confirmed knowledge

### Lessons Learned
1. **Deduplication is harder than expected** — need both exact and semantic matching
2. **LLM extraction has limits** — can't always understand context
3. **Evidence accumulation is slow** — need multiple sources for confidence
4. **Conflict resolution requires human judgment** — can't fully automate

---

## Phase 3: Intelligence Layer (2026-05 to 2026-07)

### Intent Classification
- **Problem:** User queries have different intents (evidence, dispute, exploration)
- **Solution:** LLM-based intent classification using Ollama embeddings
- **Result:** 6 intent categories, multi-label support, zero-cost classification

### Logical Gap Detection
- **Problem:** System doesn't know what it doesn't know
- **Solution:** 4 detection methods:
  - Circular reasoning (DFS cycle detection)
  - Cherry-picking (credibility spread analysis)
  - Over-generalization (universal claims vs evidence count)
  - Unstated assumptions (LLM-based analysis)
- **Result:** Proactive identification of knowledge weaknesses

### EvidenceHunter
- **Problem:** Conflicts need evidence to resolve, but gathering evidence is manual
- **Solution:** Automated web evidence sourcing via Tavily
- **Innovation:** Domain credibility ceilings (trading=0.3, tcm=0.4, real_estate=0.7)
- **Result:** Semi-automated evidence gathering for conflict resolution

### Experience Synthesis
- **Problem:** Users need synthesized knowledge, not just raw claims
- **Solution:** `explore_experience()` method with epistemic weighting
- **Innovation:** LLM judgment with credibility weighting (Confirmed=1.0, Disputed=0.8, Unverified=0.6)
- **Result:** Rich, context-aware knowledge synthesis

---

## Phase 4: Integration Layer (2026-07)

### Android Client (6dfov)
- **Goal:** Mobile access to knowledge engine
- **Architecture:** Jetpack Compose + CameraX + ML Kit OCR
- **Innovation:** Vision data ingestion — camera feed → OCR → knowledge engine
- **Result:** Real-time knowledge building from visual input

### HTTP Transport
- **Problem:** Android can't use stdio MCP transport
- **Solution:** HTTP/JSON API for Android connectivity
- **Innovation:** Auto-detection of network mode (hotspot/WiFi/direct)
- **Result:** Seamless mobile integration

### Hardware Capability Detection
- **Problem:** Different devices have different capabilities
- **Solution:** Device mode system (Phone/Phone AR/XR)
- **Innovation:** Capability-aware feature availability
- **Result:** Adaptive experience based on device capabilities

---

## Phase 5: Context-Aware Ingestion (2026-07, In Progress)

### Problem Statement
The system extracts claims but loses epistemic scaffolding — the chain of "how do we know this?" that makes knowledge reliable.

### Solution: Hybrid Architecture
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

### Key Innovations (Planned)
1. **Richer extraction:** Claims + evidence + context + conditions
2. **Provenance chains:** Trace knowledge back to source
3. **Confidence scoring:** Combine LLM assessment with code rules
4. **Cross-slot conflict detection:** Find contradictions across different aspects

### Expected Impact
- Claims will have evidence from the same transcript
- Knowledge will be auditable ("why does the system believe X?")
- Evidence gates can be satisfied from transcript content
- Confidence scores will reflect source quality

---

## Architecture Evolution

### v1.0: Simple Knowledge Graph
```
Claims → Neo4j → Query → Response
```
- Basic claim storage
- Manual evidence management
- No conflict detection

### v2.0: Evidence-Gated Pipeline
```
Transcript → Pipeline → Claims → Evidence Gates → Canonical Knowledge
```
- 8-stage pipeline
- Domain-specific policies
- Conflict detection

### v3.0: Context-Aware System (Current)
```
Transcript → Hybrid Extraction → Claims + Evidence → Provenance Chains → Auditable Knowledge
```
- Richer extraction
- Provenance tracking
- Confidence scoring

### v4.0: Real-Time Learning (Future)
```
Stream → Continuous Ingestion → Live Updates → Adaptive Knowledge
```
- Real-time processing
- Incremental updates
- Adaptive evidence requirements

---

## Technical Debt & Lessons Learned

### What Worked Well
1. **Graph-based storage** — Neo4j relationships are natural for knowledge
2. **Epistemic status model** — Clear progression from unverified to confirmed
3. **Domain-specific policies** — Different domains need different evidence bars
4. **Harbour system** — Idempotent processing prevents rework

### What Needs Improvement
1. **Deduplication** — Still mostly exact-match, need semantic dedup
2. **Evidence extraction** — Currently manual, need automated extraction
3. **Conflict resolution** — Too manual, need better auto-resolution
4. **Context preservation** — Losing provenance information

### Key Learnings
1. **Knowledge quality requires evidence** — Can't just accumulate claims
2. **Context matters** — Same claim can be true under different conditions
3. **Source quality varies** — Need to assess and weight accordingly
4. **Conflicts are opportunities** — Contradictions reveal knowledge gaps

---

## Research White Paper Outline

### Title
"Evidence-Gated Knowledge Systems: A Hybrid Architecture for Reliable Knowledge Representation"

### Sections
1. **Introduction** — Problem statement, research questions
2. **Related Work** — RAG systems, knowledge graphs, evidence-based reasoning
3. **Architecture** — The 8-stage pipeline, epistemic status model
4. **Evidence Gating** — Domain policies, promotion criteria
5. **Context-Aware Ingestion** — Hybrid extraction, provenance chains
6. **Conflict Detection** — Same-slot and cross-slot detection
7. **Evaluation** — Reliability metrics, comparison with RAG
8. **Case Studies** — TCM, Trading, Real Estate domains
9. **Future Work** — Real-time learning, multi-modal evidence
10. **Conclusion** — Contributions, implications

### Key Claims (to be validated)
1. Evidence-gated systems produce more reliable knowledge than RAG
2. Hybrid extraction (code + LLM) outperforms pure LLM approaches
3. Provenance chains enable auditable knowledge systems
4. Domain-specific policies improve knowledge quality

---

## Metrics & Evaluation

### Current Metrics
- **Claim count:** ~211-220 claims across domains
- **Test coverage:** ~95% (211+ tests passing)
- **Evidence gates:** Domain-specific (Trading: 1.2/2, TCM: 1.2/2)
- **Conflict detection:** Same-slot only (cross-slot planned)

### Planned Metrics
- **Evidence extraction rate:** % of claims with extracted evidence
- **Provenance completeness:** % of claims with full provenance chains
- **Confidence accuracy:** Correlation between confidence scores and actual reliability
- **Conflict detection precision/recall:** % of true conflicts detected

---

## Timeline

| Phase | Period | Focus | Status |
|-------|--------|-------|--------|
| Phase 1 | 2026-01 to 2026-03 | Foundation | ✅ Complete |
| Phase 2 | 2026-03 to 2026-05 | Core Pipeline | ✅ Complete |
| Phase 3 | 2026-05 to 2026-07 | Intelligence Layer | ✅ Complete |
| Phase 4 | 2026-07 | Integration Layer | ✅ Complete |
| Phase 5 | 2026-07 (ongoing) | Context-Aware Ingestion | 🔄 In Progress |
| Phase 6 | 2026-Q3 | Real-Time Learning | 📋 Planned |
| Phase 7 | 2026-Q4 | Multi-Modal Evidence | 📋 Planned |

---

## References

### Internal Documentation
- `docs/compose/specs/2026-07-15-context-aware-ingestion-design.md` — Enhanced extraction design
- `docs/compose/specs/2026-07-10-vision-integration-design.md` — 6dfov vision integration
- `wiki/MCP-API.md` — MCP tool documentation
- `README.md` — Project overview

### External Research
- Evidence-based reasoning in AI systems
- Knowledge graph quality metrics
- Hybrid extraction architectures
- Provenance tracking in knowledge systems

---

*Last updated: 2026-07-15*
*Author: ronnyling*
*Status: Living document — update as system evolves*
