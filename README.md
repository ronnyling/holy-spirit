# Continuous R&D Knowledge Engine

An incremental, evidence-gated knowledge system for transcript-driven R&D work.

The intention is to build a living AI wiki. You drop in transcripts and notes from TCM, real estate,
and stock trading; the system turns them into refined canonical knowledge; the AI then uses that
knowledge as working experience when it helps with later R&D tasks. The goal is not to flatten
disagreement into a single anonymous answer. The goal is to preserve provenance, dissent, evidence
chains, and prior resolution history so the agent gets better at asking the right questions over time.

## Intention

The chat history established four product goals:

- Treat every transcript as evidence, not truth.
- Detect missing data early so the user can clarify on the spot.
- Resolve conflicts collaboratively with the user, then store the decision as reusable memory.
- Let future transcripts reuse that memory so the agent already knows what went wrong before.

## What We Locked In

- **Graph-first** knowledge model on **Neo4j 5** — a native property graph *and* a native vector index in a single engine.
- Vectors are a **core capability, not a fallback**: claim embeddings live on graph nodes and are searched with Neo4j's native vector index alongside graph traversal.
- Every transcript is **harboured, chunked, auto-documented, and housekept** on ingest (deterministic, offline, no LLM).
- **No fallbacks anywhere.** If the graph, embedding provider, or LLM is unconfigured or unreachable, the engine fails loudly. We resolve problems; we do not silently degrade.
- MCP is the access layer for future agents and tools.
- Human confirmation for schema promotion and canonicalization.
- User input is a source, not an oracle.

## Graph Stance

We commit to graph-first, with the graph and its vectors as the core — not a deferred support layer.

There are three distinct meanings of "graph RAG"; we are explicit about which we adopt:

1. **Property-graph data model** — *adopted.* Entities, claims, evidence, cases, slots, transcripts, and chunks are nodes; lineage and conflict memory are edges.
2. **Graph-augmented retrieval** — *adopted.* Native vector search finds candidate claims, then graph traversal expands `claim -> evidence -> case -> prior claim` lineage.
3. **Microsoft GraphRAG batch rebuilds** — *rejected.* Batch community-summary regeneration is a poor fit for incremental, human-gated ingestion.

Why Neo4j 5: one engine gives us the property graph, native vector index, and Cypher-native cycle detection, so there is no split-brain between a relational store and a graph store, and no rewrite later.

### No Fallbacks (hard rule)

- No lexical-similarity substitute for embeddings, no regex substitute for LLM semantic gap detection, no in-memory substitute for the graph store at runtime.
- Required backends are validated at startup; a missing or unreachable backend is an error, not a degraded mode.
- Integration tests may *skip* when Neo4j is absent — that is test-infrastructure gating, not a runtime fallback.

## Data Model (Locked)

These are the core entities. They map directly onto graph nodes; lineage and conflict memory are edges.

| Entity | Purpose | Key Fields |
|---|---|---|
| `Entity` | Canonical concept; different experts' terms map to the same entity | `id`, `canonical_name`, `aliases`, `description`, `version` |
| `Claim` | A single assertion about an entity | `id`, `entity_id`, `statement`, `slot_name`, `epistemic_status`, `provenance[]`, `embedding`, `version`, `tags[]` |
| `Provenance` | Where a claim came from | `source_kind` (`internal_wiki` \| `external_doc` \| `user`), `source_id`, `source_ref`, `captured_at`, `notes` |
| `Evidence` | Support attached to a claim | `id`, `claim_id`, `source_kind`, `source_id`, `credibility` (0-1), `linked_claim_ids[]`, `notes` |
| `ResolutionCase` | The memory of how a conflict was resolved | `id`, `conflict_signature`, `conflicting_claim_ids[]`, `evidence_ids[]`, `research_notes`, `decision`, `rationale`, `version`, `reopened_from_case_id`, `is_open` |
| `Slot` | An emergent template field learned from repetition | `id`, `entity_id`, `name`, `description`, `lifecycle`, `observed_count`, `candidate_count`, `expected_count`, `last_observed_at`, `retired_at`, `version` |
| `Transcript` | A harboured source document | `transcript_id`, `domain`, `entity_name`, `source_kind`, `source_id`, `sha256`, `char_count`, `chunk_count`, `raw_path`, `document_path`, `ingested_at` |
| `Chunk` | A deterministic slice of a transcript | `index`, `text`, `char_start`, `char_end`, `token_estimate` |

### Graph shape

```
(:Claim)-[:ABOUT]->(:Entity)
(:Evidence)-[:SUPPORTS]->(:Claim)
(:Claim)-[:SUPPORTS]->(:Claim)          // internal evidence (Confirmed only, cycle-checked)
(:Claim)-[:CONFLICTS_WITH]->(:Claim)
(:ResolutionCase)-[:RESOLVES]->(:Claim)
(:Claim)-[:DERIVED_FROM]->(:Chunk)
(:Chunk)-[:PART_OF]->(:Transcript)
(:Transcript)-[:ABOUT]->(:Entity)
(:Entity)-[:ALIAS_OF]->(:Entity)
(:Slot)-[:OF]->(:Entity)
```

## Transcript Harbouring, Chunking & Housekeeping

When a transcript is ingested it is first *harboured* — this happens before any claim processing and is fully deterministic and offline (no LLM, no fallbacks):

1. **Normalize** line endings and collapse blank runs.
2. **Hash** the normalized content (SHA-256). Identical content is never harboured twice — re-ingesting the same transcript is idempotent housekeeping, not a duplicate.
3. **Chunk** with a structure-aware windower (`TranscriptChunker`): paragraph/sentence aware, bounded size with overlap, oversized sentences hard-split. Offsets are recorded against normalized text.
4. **Document** each transcript as human-readable markdown (metadata + chunk map) under `documents/`.
5. **Register** the raw text, the generated document, and metadata in a filesystem-backed manifest.

This layer is engine-agnostic and independently unit-tested, so it stands on its own beneath the graph store.

## Bounded vs Unbounded Split (Locked)

Per the workspace `agent_protocol`, deterministic work stays in code and only genuinely ambiguous work
goes to the LLM. This boundary must not be crossed casually.

| Concern | Layer | Why |
|---|---|---|
| Transcript chunking + documentation + housekeeping | Deterministic code | Structure-aware slicing and hashing |
| Structural gap check (expected slot missing) | Deterministic code | It is a schema-presence check |
| Slot counting and threshold detection | Deterministic code | Pure statistics |
| Evidence scoring | Deterministic code | Arithmetic over credibility |
| Cycle detection | Deterministic code (native Cypher) | Graph reachability over `SUPPORTS` edges |
| Conflict signature construction | Deterministic code | Normalized string key |
| Semantic gap detection | LLM | Requires meaning, not presence |
| Claim reconciliation and terminology mapping | LLM | Same concept, different words |
| Conflict interpretation and resolution proposal | LLM | Requires judgment |

## Knowledge Rules

### Epistemic Status

Every claim carries one of:

`Confirmed | Unverified | Unknown | Unverifiable | Disputed | Retracted`

- `Unknown` is valid and may be permanent.
- `Unverified` means a claim was observed but not yet proven.
- `Disputed` means it conflicts with current canonical knowledge.
- `Retracted` remains in memory but is not treated as active truth.

### Slot Lifecycle

Slots are learned from repetition across sources and move through:

`Observed -> Candidate -> Expected -> Retired`

- The engine observes slots automatically.
- Promotion is never autonomous.
- Candidate / Expected transitions require human confirmation.
- The goal is for the 101st transcript to trigger the right question, not the 1st.

**Cold start and learning safeguards (locked):**

- During cold start (roughly the first 20-30 transcripts) the engine observes only. It must not ask
  confidently, because it has no statistical basis yet.
- Confidence must be earned and stated, never assumed.
- Frequency is not importance. A rare-but-critical slot (for example, a fatal contraindication) still
  needs human signal; counting alone will under-weight it.
- The three failure modes we explicitly guard against are cold start, drift / self-reinforcement
  (the engine inventing a slot then demanding it), and alert fatigue.
- Current promotion thresholds in code: Candidate at 3 observations, Expected at 5. These are
  deliberately low for the prototype and are expected to rise for real cold-start scale.

### Evidence Rules

- Nothing becomes canonical without evidence.
- Evidence can come from the wiki itself or from user / external sources.
- Internal wiki evidence is valid only if the supporting claim is already `Confirmed`.
- Cycles are blocked: a claim cannot justify itself directly or transitively.
- Evidence strength is domain-specific:
	- Trading: empirical, backtest/data first.
	- Real estate: conditional, jurisdiction- and cycle-aware.
	- TCM: corroboration or classical citation, often ending at attributed belief.
- Every canonical claim keeps a traceable evidence chain so you can always ask "why is this canonical?"

**Per-domain evidence gates (as implemented in `policy.py`):**

A claim is only confirmed when it clears both the score and the source-count bar for its domain. Score is
the sum of evidence credibility (0-1 each). Source kind still governs trust and cycle rules.

| Domain | Minimum score | Minimum distinct sources | Rationale |
|---|---|---|---|
| Trading | 1.2 | 2 | Highest bar; claims should be corroborated and ideally empirical |
| TCM | 1.2 | 2 | No single authority; require multiple lineages or a classical citation |
| Real estate | 0.8 | 1 | Conditional/precedent evidence, context-tagged |
| Default | 0.8 | 1 | Fallback for unclassified domains |

- User-sourced claims never auto-confirm on ingestion; they enter as `Unverified` and must be promoted
  with evidence.
- Internal wiki evidence contributes only if the supporting claim is already `Confirmed` and adding it
  creates no dependency cycle.

### Gap Rules

- Structural gaps are deterministic: expected slot missing.
- Semantic gaps are heuristic / LLM-backed: unclear context, missing rationale, missing conditions.
- Gap checking runs before conflict checking.
- The engine should ask for missing data immediately so you can clarify on the spot.

### Conflict Rules

- Conflicts are tracked per aspect or slot, not just per entity.
- Same entity + same slot + contradictory statement -> open a resolution case.
- Resolution cases are versioned, reopenable, and reusable as memory.
- Future similar conflicts should retrieve prior cases first.

## Processing Pipeline

1. Ingest transcript.
2. Harbour it: normalize, hash, chunk, auto-document, register (idempotent housekeeping).
3. Extract claims.
4. Observe slots and learn frequencies.
5. Run structural and semantic gap checks (gap check runs **before** conflict check).
6. If gaps exist, flag them immediately.
7. Run conflict detection against canonical knowledge.
8. Open or reuse a resolution case if needed.
9. Research with the user to gather evidence.
10. Promote to canonical only when evidence gates are satisfied.
11. Keep the evidence chain and resolution history.

## Domain Behavior

| Domain | What Good Looks Like | Typical Conflict Style |
|---|---|---|
| TCM | Preserve school / lineage differences and annotate dissent | Terminology mismatch, lineages, non-falsifiable beliefs |
| Real estate | Conditional rules keyed to market, jurisdiction, and cycle | Context-dependent strategy differences |
| Trading | Evidence-weighted, regime-tagged, backtest-aware claims | Direct contradictions, falsifiable strategy claims |

**Consensus with preserved dissent (locked principle):**

- The system produces consensus with preserved dissent. It never forces a single winner where the
  domain does not support one.
- Source attribution and disagreement are always surfaced. Synthesized knowledge is never presented as
  anonymous fact, because these domains carry real health and financial liability.
- Resolution policy is pluggable per domain but runs through one shared pipeline.

## Build Order (Locked Sequencing)

Each layer must rest on a verified layer beneath it. This order is intentional and should not be skipped.

1. Claims + provenance + epistemic status (substrate).
2. Slot observation + statistics (count only, no asking).
3. Candidate -> Expected promotion with human confirmation.
4. Structural gap check (deterministic), then semantic gap check (LLM).
5. Conflict detection + resolution memory.
6. Evidence ledger + cycle detection + per-domain bars.

## Current Implementation vs Planned

This section is deliberately explicit so no one has to guess what is real today versus what is designed
but not yet built.

**Implemented and locally verified:**

- Entity, claim, evidence, slot, and resolution-case models (layers 1-3, 5, 6 of the build order)
- Transcript harbouring: deterministic chunking (`chunking.py`), auto-documentation (`documentation.py`), and idempotent housekeeping registry (`registry.py`), wired into `ingest_transcript`
- Orchestration of the full pipeline in `engine.py`
- Slot observation and threshold-based promotion suggestions, with human-gated confirmation
- Structural gap detection that runs before conflict detection
- Same-aspect (entity + slot) conflict detection
- Resolution-case creation and reopening from similar prior cases
- Per-domain evidence gates and evidence scoring
- Graph schema (`graph/schema.py`): Cypher constraints + native vector index DDL + cycle-probe query (unit-tested)
- `graph/neo4j_store.py`: real Neo4j-driver store — node/edge upserts, native vector search (eventually-consistent, with `await_indexes`), and Cypher-native cycle detection. **Verified against Neo4j Community 2026.05.0** — full suite `python -m pytest -q` reports **33 passed** (30 unit + 3 integration). See [wiki/Setup-Neo4j.md](wiki/Setup-Neo4j.md).
- MCP server exposure for agent access

**Still to build (no fallbacks — these block full production):**

- API-based embedding provider (httpx + tenacity) writing embeddings onto claim nodes
- LLM-backed semantic gap detection, claim reconciliation, and conflict interpretation (MiMo 2.5)
- Switching the engine's default store from the in-process substrate to `KnowledgeGraphStore` (store itself is now verified against Neo4j)
- Versioned markdown canonical store
- Cold-start-scale thresholds and drift / alert-fatigue safeguards tuned for real volume

## MCP Interface

The project exposes an MCP server with these tools:

- `ingest_transcript`
- `confirm_slot`
- `promote_claim`
- `resolve_case`
- `state_snapshot`

It also exposes a `knowledge://state` resource with a JSON snapshot of the current engine state.

## Project Layout

- `src/knowledge_engine/` - runtime package (engine, models, chunking, documentation, registry, gaps, conflicts, evidence, resolution, policy)
- `src/knowledge_engine/graph/` - Neo4j graph layer (`schema.py` pure DDL, `neo4j_store.py` driver store)
- `tests/` - unit tests, graph schema tests, gated Neo4j integration tests, and beta scenarios
- `docker-compose.yml` - Neo4j 5 (graph + native vector index)
- `.env.example` - required backend configuration (no fallbacks)
- `wiki/` - GitHub wiki pages (architecture, data model, pipeline, setup)
- `server.py` - local MCP bootstrap entrypoint
- `.vscode/mcp.json` - VS Code MCP wiring
- `pyproject.toml` - Python packaging and test config

## Quick Start

```powershell
cd "c:\Users\r.a.ling\OneDrive - Avanade\Documents\work\Native AI\knowledge_engine"

# Unit tests (no external services needed)
python -m pytest

# Stand up the graph engine, then run the gated integration tests.
# Docker requires hardware virtualization; if that is blocked, use the
# Neo4j Community Server (no Docker) — see wiki/Setup-Neo4j.md (Option A).
docker compose up -d
$env:KE_NEO4J_URI = "bolt://localhost:7687"
$env:KE_NEO4J_USER = "neo4j"
$env:KE_NEO4J_PASSWORD = "knowledge-engine"
$env:KE_NEO4J_DATABASE = "neo4j"
python -m pytest tests/test_graph_neo4j.py

# MCP server
python server.py
```

Copy `.env.example` to `.env` and fill in the Neo4j, embedding, and LLM settings. Required backends must be configured — the engine does not run in a degraded mode.

## Beta Test Plan

The beta suite should exercise the full workflow:

- TCM transcript ingestion with slot learning and missing-data prompts
- Real estate transcript ingestion with jurisdiction / cycle clarification
- Trading transcript ingestion with regime and evidence gating
- Conflict creation, resolution, and precedent reuse
- Internal evidence cycle prevention
- User clarification captured as evidence, not automatic truth

Two runnable, self-verifying persona walkthroughs live at the repository root and are documented in
[wiki/Usage-Scenarios.md](wiki/Usage-Scenarios.md):

- `beta_xr_copilot_run.py` — XR smart-glasses research copilot (human-in-the-loop evidence gating).
- `beta_agent_loop_run.py` — autonomous agent loop that scrapes the web and learns from verbatim text only.

They are not `test_`-prefixed, so pytest does not collect them; run them directly with `PYTHONPATH=src`.

## Wiki

Full documentation lives in [`wiki/`](wiki/) and mirrors the GitHub project wiki:

- [Home](wiki/Home.md) — overview and reading order
- [Architecture](wiki/Architecture.md) — graph-first design and no-fallback rule
- [Data Model](wiki/Data-Model.md) — nodes, edges, and epistemic status
- [Pipeline](wiki/Pipeline.md) — ingestion → harbour → gaps → conflicts → promotion
- [Ingestion & Housekeeping](wiki/Ingestion-and-Housekeeping.md) — chunking, documentation, idempotency
- [Setup: Neo4j](wiki/Setup-Neo4j.md) — running the graph engine locally
- [MCP API](wiki/MCP-API.md) — tools and resources
- [Domain Policies](wiki/Domain-Policies.md) — per-domain evidence gates
- [Usage Scenarios](wiki/Usage-Scenarios.md) — end-to-end personas (XR copilot, autonomous scraping agent)

## Implementation Notes

- User input is treated as evidence, not as automatic canonical truth.
- Unknown is a valid permanent state.
- The design is graph-first on Neo4j 5, with vectors as a core capability and no runtime fallbacks.
- This repository is meant to become the AI's working memory for later R&D tasks.
