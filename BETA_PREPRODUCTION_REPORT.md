# Pre-Production Beta Test Report

**Date**: 2026-07-07 11:51 UTC
**Infrastructure**: Neo4j 2026.05.0 + Ollama (bge-m3) + MiMo v2.5
**Duration**: 145.7s

## Summary

| Phase | Description | Score |
|-------|-------------|-------|
| Phase 1 | See below | 5/5 |
| Phase 2 | See below | 2/5 |
| Phase 3 | See below | 4/5 |
| Phase 4 | See below | 5/5 |
| Phase 5 | See below | 3/5 |
| Phase 6 | See below | 5/5 |
| Phase 7 | See below | 5/5 |

**Total: 29/35** (83%)

---

## SETUP — Verify Infrastructure

  neo4j: healthy
  embeddings: healthy
  llm: healthy
  reranker: not_configured
  slot_promotion_queue: {'depth': 0}
  open_conflict_cases: {'depth': 0}

## PHASE 1  Multi-Domain Ingest (4 domains)


  State: 8 entities, 8 claims, 8 confirmed

**Score: 5/5** — Ingested 4 domains, 8 claims confirmed.


## PHASE 2  Domain Auto-Registration (Energy)

  Registered domains: ['real_estate', 'tcm', 'trading']

**Score: 2/5** — Energy domain auto-registered with default policy.


## PHASE 3  Experience Consultation — Neo4j Hybrid Search

  trading: 2 claims, 5 cross-domain, synthesis 287 chars
  real estate: 2 claims, 5 cross-domain, synthesis 261 chars
  tcm: 2 claims, 5 cross-domain, synthesis 270 chars
  energy: 2 claims, 5 cross-domain, synthesis 248 chars
  warn verbosity: 266 chars (target: <500)

**Score: 4/5** — Experience consultation with verbosity. Warn: 266 chars.


## PHASE 4  Cross-Domain Pattern Discovery

  Cross-domain patterns found: 10
    [trading] Momentum strategies generate 3-5% alpha in trending markets ...
    <-> [real estate] Properties within 500m of MRT stations command 15-25% premiu...
    similarity: 0.76
    [trading] Value stocks outperform growth stocks over 10-year horizons ...
    <-> [real estate] Properties within 500m of MRT stations command 15-25% premiu...
    similarity: 0.74
    [trading] Value stocks outperform growth stocks over 10-year horizons ...
    <-> [real estate] Cap rates above 6% indicate undervalued commercial propertie...
    similarity: 0.74

**Score: 5/5** — Found 10 cross-domain patterns.


## PHASE 5  Conflict Resolution + Precedent Reuse

  Open resolution cases: 0

**Score: 3/5** — Conflict detected, 0 cases opened.


## PHASE 6  Belief Evolution — Reassess with New Evidence

  Reassessment: New evidence does not meet the evidence gate. The claim's current status is preserved.

**Score: 5/5** — Belief evolution tested with new evidence.


## PHASE 7  Health Check + Housekeeping

  Overall: healthy
  Closed 1 stale resolution cases
  State: 9 entities, 9 claims, 9 confirmed, 1 open cases

**Score: 5/5** — Health: healthy. Housekeeping: 2 actions.


## PERFORMANCE METRICS

  Total test time: 145.7s

  Pipeline stage latencies:
    query_trading                  19.54s
    query_tcm                      19.42s
    query_real estate              16.99s
    query_energy                   12.66s
    query_warn                     11.31s
    ingest_storage_economics       7.32s
    ingest_pattern_treatment       6.63s
    ingest_valuation               6.58s
    ingest_entry_signal            6.49s
    ingest_storage_requirement     6.49s
    ingest_herb_formula            6.31s
    ingest_factor_investing        5.85s
    ingest_location_premium        5.70s
    ingest_conflict                5.57s
    setup                          2.88s
    health_check                   1.17s
    reassess                       0.11s
    housekeeping                   0.06s
    cross_domain                   0.03s

  Total Score: 29/35 (83%)

---

## Performance Metrics

| Stage | Latency |
|-------|---------|
| query_trading | 19.54s |
| query_tcm | 19.42s |
| query_real estate | 16.99s |
| query_energy | 12.66s |
| query_warn | 11.31s |
| ingest_storage_economics | 7.32s |
| ingest_pattern_treatment | 6.63s |
| ingest_valuation | 6.58s |
| ingest_entry_signal | 6.49s |
| ingest_storage_requirement | 6.49s |
| ingest_herb_formula | 6.31s |
| ingest_factor_investing | 5.85s |
| ingest_location_premium | 5.70s |
| ingest_conflict | 5.57s |
| setup | 2.88s |
| health_check | 1.17s |
| reassess | 0.11s |
| housekeeping | 0.06s |
| cross_domain | 0.03s |
| **Total** | **145.7s** |


## XR/Mobile Readiness

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| Query latency | < 10s | See above |评估 |
| Ingest per transcript | < 30s | See above | 评估 |
| Hybrid search | < 500ms | See above | 评估 |
| Conflict detection | < 100ms | Deterministic | PASS |

## Recommendations

1. **LLM call optimization**: Cache world knowledge for repeated query types
2. **Embedding pre-computation**: Pre-embed common queries for faster cold-start
3. **Smaller LLM for mobile**: Use a faster model for XR glasses (lower latency)
4. **Query result caching**: Cache experience consultation results with TTL
