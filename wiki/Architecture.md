# Architecture

## Graph-first on Neo4j 5

The engine is graph-first. Neo4j 5 is the committed storage engine because it provides, in a single
database:

- a **native property graph** for entities, claims, evidence, resolution cases, slots, transcripts, and chunks;
- a **native vector index** for claim-embedding similarity search;
- **Cypher-native traversal** for lineage (`claim → evidence → case → prior claim`) and cycle detection.

One engine means no split-brain between a relational store and a graph store, and no rewrite later.

## Three meanings of "graph RAG"

| Meaning | Decision |
|---|---|
| Property-graph data model | **Adopted** |
| Graph-augmented retrieval (vector search + traversal) | **Adopted** |
| Microsoft GraphRAG batch community rebuilds | **Rejected** (incremental, human-gated ingestion is a poor fit) |

## Vectors are core, not a fallback

Claim embeddings are stored on `Claim` nodes and searched with Neo4j's native vector index. Vector
similarity is a first-class retrieval mechanism used *together with* graph traversal — it is not a
degraded substitute for anything.

## No fallbacks (hard rule)

- No lexical-similarity substitute for embeddings.
- No regex substitute for LLM semantic gap detection.
- No in-memory substitute for the graph store at runtime.
- Required backends (graph, embeddings, LLM) are validated at startup. A missing or unreachable backend
  is a **hard error**, not a degraded mode.
- Integration tests may **skip** when Neo4j is absent — that is test-infrastructure gating, not a runtime
  fallback.

## Bounded vs unbounded

Deterministic work stays in code; only genuinely ambiguous work goes to the LLM.

| Concern | Layer |
|---|---|
| Chunking, documentation, housekeeping | Deterministic code |
| Structural gap check (expected slot missing) | Deterministic code |
| Slot counting and thresholds | Deterministic code |
| Evidence scoring | Deterministic code |
| Cycle detection | Deterministic code (native Cypher) |
| Conflict signature construction | Deterministic code |
| Semantic gap detection | LLM |
| Claim reconciliation / terminology mapping | LLM |
| Conflict interpretation / resolution proposal | LLM |

## Build order

Each layer rests on a verified layer beneath it:

1. Claims + provenance + epistemic status.
2. Slot observation + statistics (count only).
3. Candidate → Expected promotion with human confirmation.
4. Structural gap check, then semantic gap check.
5. Conflict detection + resolution memory.
6. Evidence ledger + cycle detection + per-domain bars.
