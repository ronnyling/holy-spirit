# Continuous R&D Knowledge Engine

> A living AI wiki that turns multi-expert transcripts (TCM, real estate, stock trading) into
> **evidence-gated canonical knowledge** — preserving provenance, dissent, and resolution history
> instead of flattening disagreement into one anonymous answer.

Drop in transcripts and notes; the engine extracts claims, flags missing data early, detects
conflicts, and only promotes knowledge to canonical once it clears a per-domain evidence gate. An
agent can then use that knowledge as working experience for later R&D tasks.

## Highlights

- **Evidence, not truth** — every claim carries provenance and epistemic status; nothing is canonical without evidence.
- **Gaps before conflicts** — missing data is flagged for clarification before conflict checks run.
- **Consensus with preserved dissent** — never forces a single winner where the domain doesn't support one.
- **Graph-first on Neo4j 5** — one engine for the property graph *and* native vector search; no split-brain, no runtime fallbacks.
- **Bounded vs unbounded discipline** — deterministic work stays in code; only genuinely semantic work goes to the LLM.
- **User is a source, not an oracle** — user input enters as `Unverified` and must earn confirmation.

## Contents

- [Quick Start](#quick-start)
- [Status: implemented vs planned](#current-implementation-vs-planned)
- [Usage scenarios](wiki/Usage-Scenarios.md)
- [Wiki](#wiki) — full architecture, data model, pipeline, and setup docs
- Deep design: [Architecture](wiki/Architecture.md) · [Data Model](wiki/Data-Model.md) · [Pipeline](wiki/Pipeline.md) · [Domain Policies](wiki/Domain-Policies.md)

## Quick Start

```powershell
# From the repo root
python -m pytest                     # unit tests, no external services needed

# Optional: stand up the graph engine, then run the gated integration tests.
# Docker needs hardware virtualization; if blocked, use Neo4j Community Server
# (no Docker) — see wiki/Setup-Neo4j.md (Option A).
docker compose up -d
$env:KE_NEO4J_URI = "bolt://localhost:7687"
$env:KE_NEO4J_USER = "neo4j"
$env:KE_NEO4J_PASSWORD = "knowledge-engine"
$env:KE_NEO4J_DATABASE = "neo4j"
python -m pytest tests/test_graph_neo4j.py

python server.py                     # MCP server (for VS Code Copilot)
```

Copy `.env.example` to `.env` and fill in the Neo4j, embedding, and LLM settings. Required backends
must be configured — the engine does not run in a degraded mode.

### Try it now (CLI / UAT)

The fastest way to use the system end-to-end today is the CLI. It runs the full
ingest pipeline on a local file-backed store and lets you query without
embeddings. **New here?** Follow the from-scratch setup (install Python, install
Ollama + pull `bge-m3`, configure `.env`, first ingest) in
[docs/UAT-Usage-Guide.md §0](docs/UAT-Usage-Guide.md#0-from-scratch-first-time-setup).
Full walkthrough: [docs/UAT-Usage-Guide.md](docs/UAT-Usage-Guide.md).

```powershell
# 1. Put transcripts in a folder (one .txt per transcript; see the guide for
#    optional .meta.json / .claims.json sidecars).
# 2. Ingest a file or a whole folder:
python scripts/ke.py ingest data/uat_transcripts --domain "real estate"

# 3. Ask research questions:
python scripts/ke.py ask --entity "Cap Rate Rules"
python scripts/ke.py ask --domain "real estate"
python scripts/ke.py ask --query "cap rate suburban"     # keyword search
python scripts/ke.py snapshot
```

> **Storage status:** the CLI uses the in-memory store persisted to
> `data/uat_state.json`. The engine's ingest pipeline is not yet wired to Neo4j
> (the Neo4j store currently implements only the query/vector side). See
> [Known limitations](docs/UAT-Usage-Guide.md#known-limitations).


## Product goals

- Treat every transcript as evidence, not truth.
- Detect missing data early so the user can clarify on the spot.
- Resolve conflicts collaboratively, then store the decision as reusable memory.
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
2. **Hash** the normalized content (SHA-256). Identical content is never harboured twice — re-ingesting the same transcript is idempotent housekeeping, not a duplicate, and it short-circuits the whole pipeline (no re-chunk, re-extract, or re-embed).
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
2. Harbour it: normalize, hash, chunk, auto-document, register (idempotent housekeeping). Identical content under any file name **short-circuits the rest of the pipeline** — no re-extraction, no re-embedding.
3. Extract claims.
4. Observe slots and learn frequencies.
5. Run structural and semantic gap checks (gap check runs **before** conflict check).
6. If gaps exist, flag them immediately.
7. Run conflict detection against canonical knowledge.
8. Open or reuse a resolution case if needed.
9. Embed new claims (on-demand model, warmed then released; batched and `num_ctx`-bounded).
10. Research with the user to gather evidence.
11. Promote to canonical only when evidence gates are satisfied.
12. Keep the evidence chain and resolution history.

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
- Transcript harbouring: deterministic chunking (`chunking.py`), auto-documentation (`documentation.py`), and idempotent housekeeping registry (`registry.py`), wired into `ingest_transcript`. Duplicate content (same normalized text under any file name) **short-circuits the whole pipeline** — chunking, extraction, and embedding are all skipped.
- Claim embedding on ingest: when an embedding provider is configured, new claims are embedded (1024-dim) inside an on-demand warmed session that releases the model afterwards; duplicate transcripts never re-embed.
- Orchestration of the full pipeline in `engine.py`
- Slot observation and threshold-based promotion suggestions, with human-gated confirmation
- Structural gap detection that runs before conflict detection
- Same-aspect (entity + slot) conflict detection
- Resolution-case creation and reopening from similar prior cases
- Per-domain evidence gates and evidence scoring
- Graph schema (`graph/schema.py`): Cypher constraints + native vector index DDL + cycle-probe query (unit-tested)
- `graph/neo4j_store.py`: real Neo4j-driver store — node/edge upserts, native vector search (eventually-consistent, with `await_indexes`), and Cypher-native cycle detection. **Verified against Neo4j Community 2026.05.0** — full suite `python -m pytest -q` reports **59 passed** (57 without the two live Ollama embedding e2e tests, which run only when Ollama is serving; Neo4j integration tests run when a database is reachable, otherwise skipped). See [wiki/Setup-Neo4j.md](wiki/Setup-Neo4j.md).
- `embeddings.py`: embedding transport with two backends — local **Ollama-native** (`/api/embed`, default `bge-m3`, 1024-dim) and OpenAI-compatible (`/v1/embeddings`) — on async httpx + tenacity retry (exponential backoff on 429/503). Embedding is treated as **housekeeping**: the model is loaded on demand for a batch (`warm()`, `keep_alive=5m`) and released immediately afterwards (`unload_sync()`, `keep_alive=0`), so an idle engine holds no model in memory. Input is **batched** (`batch_size`, default 64) and **context-bounded** (`num_ctx`, default 1024) per request. Optional at server startup — when embedding env vars are absent, vector search tools return a clear error; all other tools still work.
- `llm.py` + `extraction.py`: MiMo (OpenAI-compatible) chat client and LLM-backed claim extraction. **Verified live against the MiMo gateway** (`https://api.xiaomimimo.com/v1`, `mimo-v2.5`, `tp-` token-plan key — chat returns 200). When `KE_MIMO_API_KEY` is set and a transcript arrives with no hand-authored claims, claims are extracted from the raw text. Extraction is the **unbounded** layer; parsing/validation is deterministic and **never fabricates evidence**, so extracted claims enter as `Unverified`. Optional and non-breaking (unit-tested with a stubbed client).
- `scripts/ke.py`: command-line interface for ingest + query on the file-backed store — the recommended UAT surface. See [docs/UAT-Usage-Guide.md](docs/UAT-Usage-Guide.md).
- `store.py` JSON persistence (`save`/`load`) so an ingested corpus survives across CLI runs.
- Query tools in `engine.py` + `neo4j_store.py`: vector search over claims and entities, get full entity/claim details, search by domain.
- MCP server with 10 tools (FastMCP, stdio transport) for VS Code Copilot integration.

**Still to build (no fallbacks — these block full production):**

- **Neo4j write-path adapter (top priority).** The engine's ingest pipeline currently targets the in-memory `KnowledgeStore`. The `KnowledgeGraphStore` implements only the query/vector side with a different interface, so the MCP server's *ingestion* tools do not yet persist to Neo4j. The engine needs `KnowledgeGraphStore` to implement the same interface it expects (`upsert_entity(name)→Entity`, `add_claim→Claim`, `observe_slot`, `get_slot`, `confirm_slot`, `list_canonical_claims`, `get_expected_slots`, resolution-case methods, and dict-like accessors).
- LLM-backed semantic gap detection, claim reconciliation, and conflict interpretation (MiMo 2.5). The MiMo chat client is live (see above); these three specific LLM steps are not yet built on top of it.
- Versioned markdown canonical store
- Cold-start-scale thresholds and drift / alert-fatigue safeguards tuned for real volume
- HTTP/SSE transport for the MCP server (currently stdio only — works for Copilot, not for autonomous pipeline agents)

## MCP Interface

The project exposes an MCP server (FastMCP, stdio transport) with these tools:

**Ingestion & curation:**
- `ingest_transcript` — harbour a transcript and run the full pipeline (gap check, conflict detection, evidence gating)
- `confirm_slot` — human-in-the-loop slot lifecycle promotion (Observed → Candidate → Expected)
- `promote_claim` — evidence-gated claim promotion (Unverified → Confirmed when domain gates pass)
- `resolve_case` — record a resolution decision on a conflict case (becomes reusable memory)
- `state_snapshot` — JSON snapshot of engine state (entity/claim/slot/case counts)

**Query & retrieval (require Neo4j + embedding provider):**
- `search_claims` — vector search over claims with optional domain/status filters
- `search_entities` — semantic search: find entities whose claims match the query
- `get_entity` — full entity details (by ID or name) with all claims, slots, and evidence
- `get_claim` — claim details with provenance and evidence chain
- `search_by_domain` — all confirmed knowledge in a domain (entities, claims, slots, cases)

It also exposes a `knowledge://state` resource with a JSON snapshot of the current engine state.

**Current limitations:**
- **Ingestion tools are not yet wired to Neo4j.** The engine pipeline targets the in-memory store; a `KnowledgeGraphStore` write-adapter is required before `ingest_transcript` (and the other curation tools) persist to Neo4j via the server. For working ingest + query today, use the CLI (`scripts/ke.py`, file-backed store) — see [docs/UAT-Usage-Guide.md](docs/UAT-Usage-Guide.md).
- stdio transport only works when VS Code Copilot is active. Autonomous agent pipelines (e.g., `income_research_os` funnel, `native ai auction investment` scraper) cannot use this yet — they need either direct Python import (`pip install -e ../knowledge_engine`) or HTTP/SSE transport (planned).

## Project Layout

- `src/knowledge_engine/` - runtime package (engine, models, chunking, documentation, registry, gaps, conflicts, evidence, resolution, policy, embeddings)
- `src/knowledge_engine/graph/` - Neo4j graph layer (`schema.py` pure DDL, `neo4j_store.py` driver store with vector search)
- `src/knowledge_engine/embeddings.py` - embedding client: Ollama-native (`/api/embed`) + OpenAI-compatible transport, on-demand model lifecycle (`warm()`/`unload_sync()`), batching + `num_ctx` (async httpx + tenacity retry)
- `src/knowledge_engine/llm.py` - MiMo (OpenAI-compatible) chat client (async httpx + tenacity retry)
- `src/knowledge_engine/extraction.py` - LLM-backed claim extraction (unbounded LLM + deterministic parsing)
- `src/knowledge_engine/bootstrap.py` - shared engine bootstrap + `.env` loader (used by server and CLI)
- `scripts/ke.py` - command-line interface (ingest + ask) for UAT on the file-backed store
- `docs/UAT-Usage-Guide.md` - step-by-step guide: setup, transcript folder layout, ingest, query, limitations
- `tests/` - unit tests, graph schema tests, gated Neo4j integration tests, extraction + persistence tests, and beta scenarios
- `beta_*.py` - runnable end-to-end persona walkthroughs (XR copilot, autonomous scraping agent)
- `docker-compose.yml` - Neo4j 5 (graph + native vector index)
- `.env.example` - required backend configuration (no fallbacks)
- `wiki/` - GitHub wiki pages (architecture, data model, pipeline, setup) — embedded as a Git submodule
- `server.py` - local MCP bootstrap entrypoint (PYTHONPATH wiring)
- `.vscode/mcp.json` - VS Code MCP wiring (stdio transport)
- `pyproject.toml` - Python packaging and test config

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

## Author

Maintained by **ronnyckling** ([ronnyckling@gmail.com](mailto:ronnyckling@gmail.com)).
