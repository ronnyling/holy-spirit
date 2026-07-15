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
- **Logical gap detection** — circular reasoning, cherry-picking, over-generalization, and unstated assumptions detected automatically.
- **Consensus with preserved dissent** — never forces a single winner where the domain doesn't support one.
- **Graph-first on Neo4j 5** — one engine for the property graph *and* native vector search; no split-brain, no runtime fallbacks.
- **Bounded vs unbounded discipline** — deterministic work stays in code; only genuinely semantic work goes to the LLM.
- **User is a source, not an oracle** — user input enters as `Unverified` and must earn confirmation.
- **Slot promotion queue** — threshold crossings during ingest write to a persistent queue; the user reviews and confirms at their own pace, never blocking ingestion.
- **Experience synthesis** — `explore_experience` combines LLM world knowledge with system-accumulated claims to produce a grounded, opinionated view: world baseline → experience adds/corrects → discerned position.
- **Intent-aware input** — classifies user intent (evidence, dispute, correction, exploration, learning) via Ollama embeddings for zero-cost routing.
- **Auto-start services** — Neo4j and Ollama auto-start on engine initialization with port conflict resolution.

## Contents

- [Quick Start](#quick-start)
- [Status: implemented vs planned](#current-implementation-vs-planned)
- [Usage guide](wiki/Usage-Guide.md) — setup, CLI walkthrough, Streamlit UI, and curation
- [Wiki](#wiki) — full architecture, data model, pipeline, and setup docs
- Deep design: [Architecture](wiki/Architecture.md) · [Data Model](wiki/Data-Model.md) · [Pipeline](wiki/Pipeline.md) · [Domain Policies](wiki/Domain-Policies.md)

## Quick Start

### Prerequisites

1. **Python 3.11+** (verified with 3.14)
2. **Neo4j 5.13+** — Community Server (no Docker required). See [wiki/Setup-Neo4j.md](wiki/Setup-Neo4j.md).
3. **Ollama** — for local embeddings. Install from <https://ollama.com/download>, then `ollama pull bge-m3`.
4. **MiMo API key** — for LLM claim extraction. Set `KE_MIMO_API_KEY` in `.env`.

### Setup

```powershell
cd knowledge_engine
pip install -e .
pip install ".[ui]"
Copy-Item .env.example .env    # then edit with your settings

# Services auto-start on first run (Neo4j + Ollama)
# Or manually start if needed:
Set-Item Env:JAVA_HOME "C:\Program Files\Android\Android Studio\jbr"
cd path\to\neo4j-community-2026.05.0
bin\neo4j-admin.bat dbms set-initial-password knowledge-engine
bin\neo4j.bat console

# Start Ollama (another terminal)
ollama serve
```

### Run

```powershell
python -m pytest                  # unit tests (no Neo4j needed)
streamlit run app.py             # UI at http://localhost:8501
python server.py                 # MCP server
```

### Try it now (local UI)

The Streamlit app has three tabs:
- **Ingest** — paste a transcript or upload `.txt` files. Domain and entity name are auto-classified from content.
- **Chat** — intent-driven unified chat: conversational messages get a plain reply; domain questions trigger full synthesis (`[WORLD VIEW]` → `[EXPERIENCE]` → `[DISCERNED POSITION]`) grounded in evidence-gated claims.
- **Knowledge Base** — live snapshot, pending slot promotions queue, open-conflict resolution panel, domain browser (Confirmed + Unverified + Disputed), entity lookup, and cross-domain pattern finder.

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

> **Storage:** when `KE_NEO4J_URI` is set and Neo4j is reachable the CLI uses it as the
> **primary store**. See [wiki/Usage-Guide.md](wiki/Usage-Guide.md) for the full setup.


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
| Logical gap detection (circular reasoning, cherry-picking, over-generalization) | Deterministic code | Graph analysis and heuristics over existing claims/evidence |
| Intent classification | Deterministic code (Ollama embeddings) | Embedding similarity matching, zero-cost |
| Slot counting and threshold detection | Deterministic code | Pure statistics |
| Evidence scoring | Deterministic code | Arithmetic over credibility |
| Cycle detection | Deterministic code (native Cypher) | Graph reachability over `SUPPORTS` edges |
| Conflict signature construction | Deterministic code | Normalized string key |
| Unstated-assumption detection | LLM | Requires understanding of implicit reasoning gaps |
| Semantic gap detection | LLM | Requires meaning, not presence |
| Claim reconciliation and terminology mapping | LLM | Same concept, different words |
| Conflict interpretation and resolution proposal | LLM | Requires judgment |
| Experience synthesis | LLM | Integrates world knowledge with accumulated claims |

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

The gate is evaluated by `EvidenceLedger.evaluate()` in `evidence.py`: sum of credibility values across all evidence drafts must meet `minimum_score`, AND the count of unique source_ids must meet `minimum_sources`.

**Important distinction: Evidence gates vs Retrieval**

- **Evidence gates (promotion)**: USE score-based thresholds. Claims must pass the domain-specific gate to move from `Unverified` → `Confirmed`. This is the quality assurance mechanism.
- **Retrieval/query time**: Does NOT use scores. `explore_experience()` sends all hits (up to 30) to the LLM for relevance judgment, ranked by epistemic weight (Confirmed=1.0, Disputed=0.8, Unverified=0.6). The LLM decides relevance, not a score threshold.

- User-sourced claims never auto-confirm on ingestion; they enter as `Unverified` and must be promoted
  with evidence.
- Internal wiki evidence contributes only if the supporting claim is already `Confirmed` and adding it
  creates no dependency cycle.

### Gap Rules

- Structural gaps are deterministic: expected slot missing.
- Logical gaps are deterministic: circular reasoning, cherry-picking, over-generalization (with optional LLM for unstated assumptions).
- Semantic gaps are heuristic / LLM-backed: unclear context, missing rationale, missing conditions.
- Gap checking runs before conflict checking.
- The engine should ask for missing data immediately so you can clarify on the spot.
- Intent classification uses Ollama embeddings (zero-cost) to route user inputs appropriately.

### Conflict Rules

- Conflicts are tracked per aspect or slot, not just per entity.
- Same entity + same slot + contradictory statement -> open a resolution case.
- Resolution cases are versioned, reopenable, and reusable as memory.
- Future similar conflicts should retrieve prior cases first.

## Processing Pipeline

The core pipeline lives in `engine.py` → `KnowledgeEngine.ingest_transcript()`. It has **8 weighted stages**, tracked by `ProgressTracker`:

| Stage | Name | Weight | What Actually Happens |
|-------|------|--------|----------------------|
| 0 | `dedup_check` | 1 | Content-hash check (SHA-256). Identical content **short-circuits the entire pipeline** — no re-chunk, re-extract, or re-embed. |
| 1 | `classify` | 1 | If domain or entity_name is blank, uses `DomainClassifier.classify()` (trading/real estate/tcm) + `classify_open()` (novel domains). Auto-fills domain and entity_name. |
| 2 | `harbour` | 1 | `TranscriptRegistry.harbour()` writes raw text to `transcripts/`, generates markdown document in `documents/`, indexes in `manifest.json`. |
| 3 | `extract` | 5 (heaviest) | `ClaimExtractor.extract()` splits transcript into chunks, sends each chunk to LLM in parallel (`ThreadPoolExecutor`), parses JSON arrays into `ClaimDraft` objects. Auto-tunes parallelism: <50 chunks=2 workers, <500=4, else=8. |
| 4 | `process_claims` | 2 | For each ClaimDraft: builds `Claim` model (UNVERIFIED), stores it, records evidence if any, observes slots via `SlotLearner.observe()`. If slot crosses threshold, queues promotion suggestion. |
| 5 | `gap_check` | 1 | Two sub-checks: (a) **Structural gaps** — Expected slots not observed; (b) **Logical gaps** — LLM-powered detection of circular reasoning, cherry-picking, over-generalization, unstated assumptions. Gap check runs **before** conflict check. |
| 6 | `embed` | 2 | `EmbeddingClient.embed_texts()` batch-embeds all new claims (1024-dim), stores vectors via `store.set_claim_embedding()`. |
| 7 | `conflict_check` | 2 | `ConflictDetector.detect()` compares each new claim against existing Confirmed and Disputed claims. Two conflict paths: keyword opposition (prefix matching against `_OPPOSITION_PAIRS` like "buy/sell", "bullish/bearish") OR text similarity >= 0.35. |

### Post-Pipeline Outcomes

After the 8-stage pipeline completes, claims follow one of these paths:

- **No conflict + evidence passes gate** → Auto-confirm to `Confirmed` (engine.py line 429-433)
- **Conflict detected** → Set to `Disputed`, open `ResolutionCase`, gather conflict evidence, generate "heckle prompt" for user
- **User source** → Stays `Unverified` (user claims never auto-confirm)
- **Gap flagged** → Stays `Unverified`, flagged for human clarification
- **Slot threshold crossed** → Queued for human confirmation via `confirm_slot()`

### Manual Promotion Paths

- `promote_claim()` — User provides evidence drafts → evaluate gate → Confirmed
- `accept_on_authority()` — Batch promote with authority source (credibility 0.9), requires `accepted_by` name
- `EvidenceHunter.hunt()` — Auto web search → extract evidence → evaluate gate → auto-promote if passes

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
- Logical gap detection (`LogicalGapDetector`): circular reasoning, cherry-picking, over-generalization, and unstated-assumption checks — runs as part of the gap-check phase alongside structural and semantic gaps
- Intent classification (`IntentClassifier`): embedding-based intent detection via Ollama (zero-cost) — classifies user input as evidence, dispute, correction, exploration, learning, or chat
- Service management (`ServiceManager`): auto-start Neo4j and Ollama on engine initialization with port conflict resolution
- Same-aspect (entity + slot) conflict detection
- Resolution-case creation and reopening from similar prior cases
- Per-domain evidence gates and evidence scoring
- Graph schema (`graph/schema.py`): Cypher constraints + native vector index DDL + cycle-probe query (unit-tested)
- `graph/neo4j_store.py`: real Neo4j-driver store — full read/write interface (node/edge upserts for all entity types, native vector search, Cypher-native cycle detection, domain listing, cross-domain pattern discovery). **Verified against Neo4j Community 2026.05.0** — full suite `python -m pytest -q` reports **198 passed** (Neo4j integration tests run when a database is reachable, otherwise skipped). See [wiki/Setup-Neo4j.md](wiki/Setup-Neo4j.md).
- `embeddings.py`: embedding transport with two backends — local **Ollama-native** (`/api/embed`, default `bge-m3`, 1024-dim) and OpenAI-compatible (`/v1/embeddings`) — on async httpx + tenacity retry (exponential backoff on 429/503). Embedding is treated as **housekeeping**: the model is loaded on demand for a batch (`warm()`, `keep_alive=5m`) and released immediately afterwards (`unload_sync()`, `keep_alive=0`), so an idle engine holds no model in memory. Input is **batched** (`batch_size`, default 64) and **context-bounded** (`num_ctx`, default 1024) per request. Optional at server startup — when embedding env vars are absent, vector search tools return a clear error; all other tools still work.
- `llm.py` + `extraction.py`: MiMo (OpenAI-compatible) chat client and LLM-backed claim extraction. **Verified live against the MiMo gateway** (`https://api.xiaomimimo.com/v1`, `mimo-v2.5`, `tp-` token-plan key — chat returns 200). When `KE_MIMO_API_KEY` is set and a transcript arrives with no hand-authored claims, claims are extracted from the raw text. Extraction is the **unbounded** layer; parsing/validation is deterministic and **never fabricates evidence**, so extracted claims enter as `Unverified`. Optional and non-breaking (unit-tested with a stubbed client).
- `scripts/ke.py`: command-line interface for ingest + query. When `KE_NEO4J_URI` is set, Neo4j is the primary store and all ingest writes go directly to the graph; otherwise falls back to the file-backed JSON store. See [docs/UAT-Usage-Guide.md](docs/UAT-Usage-Guide.md).
- `store.py` JSON persistence (`save`/`load`) for the fallback store path.
- Query tools in `engine.py` + `neo4j_store.py`: vector search over claims and entities, get full entity/claim details, search by domain, list domains, cross-domain pattern discovery.
- Parallel transcript preparation in `ke.py ingest` — LLM extraction is parallelized across workers (default 5, capped by file count) using `ThreadPoolExecutor`.
- Auto domain classification with open-domain fallback: `classify()` tries known domains first; if no match, `classify_open()` asks the LLM to name a free-form domain (2–4 words) and auto-registers it in the policy registry via `register_domain()`.
- Dynamic domain policy registry (`policy.py`): new domains discovered at ingest time are registered automatically with sensible defaults; `list_policy_domains()` enumerates all known domains (static + dynamic).
- `evidence_hunter.py`: automated evidence sourcing for `Unverified` claims — generates a neutral search query, executes web search (Tavily), extracts evidence via LLM, evaluates against the per-domain bar, and auto-promotes if the gate is satisfied. Domain credibility ceilings: real estate 0.7, TCM 0.4, trading 0.3. Requires `pip install tavily-python` + `KE_TAVILY_API_KEY`.
- MCP server with 13 tools (FastMCP, stdio transport) for VS Code Copilot integration.
- `app.py` Streamlit local UI — three-tab interactive interface: **Ingest** (paste text or multi-file `.txt` upload with batch progress, session history table, auto-clearing form), **Chat** (RAG chat grounded in retrieved knowledge-base claims; sources expander open by default; domain filter built from actual ingested tags; retrieved-claim count shown before the LLM answer), **Knowledge Base** (live counts, open-conflict resolution panel, domain browser, entity lookup, cross-domain pattern finder). Requires `pip install ".[ui]"` and Neo4j + MiMo to be running. Domains are normalised to their canonical policy name on ingest so the domain filter and claim tags stay consistent.
- `engine.list_open_cases()` — returns all open `ResolutionCase` objects enriched with the full statement text of each conflicting claim, ready to display in a UI or pipe to an agent.

**Still to build (no fallbacks — these block full production):**

- LLM-backed semantic gap detection, claim reconciliation, and conflict interpretation (MiMo 2.5). The MiMo chat client is live; these three specific LLM steps are not yet built on top of it.
- `ke hunt <claim_id>` CLI subcommand and `--hunt-evidence` ingest flag to wire `EvidenceHunter` into the pipeline automatically.
- Versioned markdown canonical store.
- Cold-start-scale thresholds and drift / alert-fatigue safeguards tuned for real volume.
- HTTP/SSE transport for the MCP server (currently stdio only — works for Copilot, not for autonomous pipeline agents).
- Tests for `evidence_hunter.py` (requires mocked LLM + `SearchProvider`).

### Context-Aware Ingestion (Implemented 2026-07-15)

**Enhanced extraction with evidence and provenance:**

- `TranscriptEvidenceDraft` — evidence extracted from transcripts with source quality, conditions, methodology, and confidence scoring
- `ProvenanceChain` — complete trace from claim to source documents and original transcripts
- `SemanticDeduplicator` — finds duplicate claims across transcripts using text similarity
- `ConflictDetector` — domain-specific keyword opposition (trading, TCM, real estate) with bidirectional detection
- `CodeExtractors` — document structure parsing and metadata extraction
- LLM evidence extraction with validation — extracts evidence from transcripts to support claims

**No-fallbacks policy enforced throughout:**
- `EvidenceExtractionError` raised instead of silent degradation
- `KeyError` raised for nonexistent claims (no None returns)
- `ValueError` raised for invalid domains/configurations
- All errors include clear messages explaining what failed and why

**New modules:**
- `src/knowledge_engine/transcript_evidence.py` — TranscriptEvidenceDraft model
- `src/knowledge_engine/provenance.py` — ProvenanceChain model
- `src/knowledge_engine/extraction_prompt.py` — Enhanced extraction prompts
- `src/knowledge_engine/code_extractors.py` — Document structure and metadata extractors
- `src/knowledge_engine/semantic_dedup.py` — Semantic deduplication
- `src/knowledge_engine/conflict_detector.py` — Enhanced conflict detection
- `src/knowledge_engine/claim_extractor_evidence.py` — Evidence extraction integration
- `src/knowledge_engine/keyword_pairs.json` — Domain-specific keyword opposition pairs

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

**Domain discovery & cross-domain patterns (require Neo4j):**
- `list_domains` — lists all ingested domains (from claim tags) alongside all policy-registered domains
- `explore_domain` — full epistemic state of a domain: confirmed, unverified, and disputed claims with summary counts
- `find_cross_domain_patterns` — surface claim pairs from different domains with semantic similarity above a threshold

It also exposes a `knowledge://state` resource with a JSON snapshot of the current engine state.

**Current limitations:**
- stdio transport only works when VS Code Copilot is active. Autonomous agent pipelines (e.g., `income_research_os` funnel, `native ai auction investment` scraper) cannot use this yet — they need either direct Python import (`pip install -e ../knowledge_engine`) or HTTP/SSE transport (planned).
- Domain discovery tools (`list_domains`, `explore_domain`, `find_cross_domain_patterns`) require Neo4j as the primary store; they return an error if the engine is on the JSON fallback path.

## Project Layout

- `src/knowledge_engine/` - runtime package (engine, models, chunking, documentation, registry, gaps, conflicts, evidence, resolution, policy, classification, embeddings)
- `src/knowledge_engine/intent_classifier.py` - embedding-based intent classification via Ollama (zero-cost)
- `src/knowledge_engine/logical_gaps.py` - logical gap detection (circular reasoning, cherry-picking, over-generalization, unstated assumptions)
- `src/knowledge_engine/service_manager.py` - service lifecycle management (Neo4j, Ollama auto-start)
- `src/knowledge_engine/evidence_hunter.py` - automated web-evidence sourcing for Unverified claims (pluggable `SearchProvider`, Tavily default, domain credibility ceilings)
- `src/knowledge_engine/graph/` - Neo4j graph layer (`schema.py` pure DDL, `neo4j_store.py` full read/write driver store with vector search, domain listing, cross-domain patterns, and open-case listing)
- `app.py` - Streamlit local UI (ingest with paste/upload modes + session history, RAG chat, KB browser + conflict resolution panel); install with `pip install ".[ui]"`
- `src/knowledge_engine/embeddings.py` - embedding client: Ollama-native (`/api/embed`) + OpenAI-compatible transport, on-demand model lifecycle (`warm()`/`unload_sync()`), batching + `num_ctx` (async httpx + tenacity retry)
- `src/knowledge_engine/llm.py` - MiMo (OpenAI-compatible) chat client (async httpx + tenacity retry)
- `src/knowledge_engine/llm_config.py` - LLM configuration router with task-based priority routing (MiMo for HIGH, Ollama for LOW)
- `src/knowledge_engine/extraction.py` - LLM-backed claim extraction (unbounded LLM + deterministic parsing)
- `src/knowledge_engine/bootstrap.py` - shared engine bootstrap + `.env` loader with auto-start (used by server and CLI)
- `src/knowledge_engine/transcript_evidence.py` - TranscriptEvidenceDraft model for evidence extracted from transcripts
- `src/knowledge_engine/provenance.py` - ProvenanceChain model for knowledge tracing
- `src/knowledge_engine/extraction_prompt.py` - Enhanced extraction prompts for claims + evidence
- `src/knowledge_engine/code_extractors.py` - Document structure and metadata extractors
- `src/knowledge_engine/semantic_dedup.py` - Semantic deduplication for claims across transcripts
- `src/knowledge_engine/conflict_detector.py` - Enhanced conflict detection with domain-specific keyword opposition
- `src/knowledge_engine/claim_extractor_evidence.py` - Evidence extraction integration with LLM
- `src/knowledge_engine/keyword_pairs.json` - Domain-specific keyword opposition pairs (trading, TCM, real estate)
- `scripts/ke.py` - command-line interface (ingest + ask) for UAT on the file-backed store
- `docs/UAT-Usage-Guide.md` - step-by-step guide: setup, transcript folder layout, ingest, query, limitations
- `tests/` - unit tests, graph schema tests, gated Neo4j integration tests, extraction + persistence tests, and beta scenarios
- `tests/e2e_test.py` - end-to-end tests simulating user workflows (main.py, MCP, Streamlit, APK)
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
