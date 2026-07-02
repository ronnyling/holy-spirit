# UAT Usage Guide — Continuous R&D Knowledge Engine

A step-by-step guide to ingest transcripts and ask research questions during
UAT. Written to be honest about what works today and what does not.

> **TL;DR** — Use the CLI (`scripts/ke.py`). Put transcript `.txt` files in a
> folder, run `ingest`, then `ask`. Today the pipeline runs on a local
> file-backed store (`data/uat_state.json`), not Neo4j (see
> [Known limitations](#known-limitations)).

---

## 0. From scratch (first-time setup)

If you have never run this before, do these five steps once. Everything runs
locally; no cloud account is required for the CLI path.

**1. Install Python 3.14** and confirm it:

```powershell
python --version        # expect 3.14.x
```

**2. Get the code and install it (editable) with its dependencies:**

```powershell
cd knowledge_engine
python -m pip install -e .
```

**3. Install Ollama and pull the embedding model** (this is the local
embeddings provider — no API key needed). Download Ollama from
<https://ollama.com/download>, then:

```powershell
ollama pull bge-m3        # 1024-dim embedding model (~1.2 GB)
ollama list              # should show bge-m3:latest
```

Ollama serves on `http://localhost:11434` by default. You do **not** need to
keep a model loaded — the engine loads `bge-m3` only while embedding and
releases it right after (`ollama ps` will be empty when idle).

**4. Create your `.env`** from the template and leave the embedding block on
its local defaults:

```powershell
Copy-Item .env.example .env
```

The embedding lines should read (these are the defaults):

```dotenv
KE_EMBEDDING_PROVIDER=ollama
KE_EMBEDDING_MODEL=bge-m3
KE_EMBEDDING_DIMENSIONS=1024
KE_EMBEDDING_API_BASE_URL=http://localhost:11434
KE_EMBEDDING_API_KEY=            # blank for Ollama
KE_EMBEDDING_KEEP_ALIVE=0        # release the model when idle
KE_EMBEDDING_NUM_CTX=1024        # input context per embed call
KE_EMBEDDING_BATCH_SIZE=64       # texts per request
```

**5. Verify the install** by running the tests (no external services needed):

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

You are now ready. Continue to [Where to put your transcripts](#2-where-to-put-your-transcripts),
or jump straight to a first run:

```powershell
# Ingest the bundled sample folder, then ask about it
python scripts/ke.py ingest data/uat_transcripts --domain "real estate"
python scripts/ke.py ask --entity "Cap Rate Rules"
```

---

## 1. Prerequisites

| Requirement | Status | Notes |
|---|---|---|
| Python 3.14 | required | `python --version` |
| Neo4j running | optional for CLI | only needed for the MCP server / vector search path |
| `.env` configured | required | copied from `.env.example`, secrets filled |
| MiMo LLM key | optional | enables auto-extraction from raw text |
| Ollama + `bge-m3` | recommended | local embeddings provider (no key); claims are embedded at ingest |

Your `.env` is already created and git-ignored. Key lines:

```dotenv
KE_NEO4J_PASSWORD=knowledge-engine
KE_MIMO_API_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1
KE_MIMO_API_KEY=sk-...            # currently returns 401 — see limitations
KE_MIMO_MODEL=mimo-v2.5
KE_EMBEDDING_PROVIDER=ollama      # local embeddings, no key needed
KE_EMBEDDING_MODEL=bge-m3
KE_EMBEDDING_API_BASE_URL=http://localhost:11434
```

---

## 2. Where to put your transcripts

Create a folder and drop one `.txt` file per transcript:

```
data/uat_transcripts/
  cap_rate.txt              # raw transcript text (REQUIRED)
  cap_rate.meta.json        # OPTIONAL metadata
  cap_rate.claims.json      # OPTIONAL hand-authored claims
```

### `*.txt` — the raw transcript
Plain text. Any length. This is the source material.

### `*.meta.json` — metadata (optional)
If omitted, the CLI infers `domain` from the folder name and `entity_name` from
the file name.

```json
{
  "domain": "real estate",
  "entity_name": "Cap Rate Rules",
  "source_kind": "external_doc",
  "source_id": "uat-cap-rate-interview",
  "source_ref": "data/uat_transcripts/cap_rate.txt"
}
```

- `domain` — one of your domains, e.g. `real estate`, `trading`, `tcm`.
- `source_kind` — `external_doc` (a document/expert), `user` (your own input),
  or `internal_wiki`. **`user` claims can never auto-confirm** — by design.

### `*.claims.json` — claims (optional)
This is how the system knows *what was asserted*. See
[the extraction decision](#3-how-claims-get-created) below.

```json
[
  {
    "statement": "A 6% cap rate suits a stable suburban asset in a balanced market.",
    "slot_name": "cap_rate",
    "observed_slots": ["cap_rate"],
    "evidence": [
      { "source_kind": "external_doc", "source_id": "uat-cap-rate-interview", "credibility": 0.9 }
    ]
  }
]
```

- **No evidence → the claim stays `Unverified`.** Evidence is required for a
  claim to become `Confirmed` (canonical). This is the evidence gate.

---

## 3. How claims get created

The ingest pipeline turns *claims* (not raw text) into knowledge. There are two
ways to produce claims:

1. **Hand-authored** — provide a `*.claims.json` sidecar. Always works.
2. **LLM auto-extraction** — if a **valid** `KE_MIMO_API_KEY` is set and there
   is **no** `*.claims.json`, the engine sends the raw text to MiMo and extracts
   claims automatically. Extracted claims carry **no evidence**, so they enter
   as `Unverified` and must earn promotion.

> The MiMo key currently returns **401 Invalid API Key**, so auto-extraction is
> wired but not usable until a valid key is provided. Until then, use
> `*.claims.json`.

---

## 4. Ingest

```powershell
cd knowledge_engine

# Ingest a whole folder
python scripts/ke.py ingest data/uat_transcripts --domain "real estate"

# Or a single file
python scripts/ke.py ingest data/uat_transcripts/cap_rate.txt --entity "Cap Rate Rules"
```

Example output:

```
=== cap_rate.txt → entity ef95ffd7... ===
  domain          : real estate
  claims ingested : 2
  confirmed       : 2
  unverified      : 0
  disputed        : 0
  gaps flagged    : 0
  conflicts       : 0
  note            : recorded 1 evidence item(s) for claim d86ec4ed...
state saved → data/uat_state.json
```

What the pipeline did automatically (deterministic):
1. Created/reused the entity.
2. Recorded each claim with provenance.
3. Observed slots (counts toward emergent schema).
4. **Gap check** (structural + semantic) — before conflict check.
5. **Conflict check** vs existing confirmed claims → opens a ResolutionCase.
6. Recorded evidence and applied the **evidence gate** (`Confirmed` only if
   external source + credible evidence + no gaps; otherwise `Unverified`;
   `Disputed` if it conflicts a canonical claim).
7. Embedded any new claims (local Ollama `bge-m3`, 1024-dim) — the model is
   loaded only for the batch and released afterwards. A re-ingest of identical
   content short-circuits the pipeline and skips this entirely.

The full corpus is saved to `data/uat_state.json` so later `ask` runs see it.

---

## 5. Ask research questions

```powershell
# Everything known about one topic
python scripts/ke.py ask --entity "Cap Rate Rules"

# All claims in a domain
python scripts/ke.py ask --domain "real estate"

# Keyword search across all claims (no embeddings required)
python scripts/ke.py ask --query "cap rate suburban"

# One claim with its evidence chain
python scripts/ke.py ask --claim <claim_id>

# Counts
python scripts/ke.py snapshot
```

Every answer shows each claim's **epistemic status** (`Confirmed`,
`Unverified`, `Disputed`, `Unknown`, ...) so you always see *how settled* a
piece of knowledge is — never anonymous synthesized "fact".

---

## 6. Maintenance / curation

These are the human-in-the-loop gates. Today they run through the Python API or
the MCP server tools (a CLI wrapper can be added on request):

- **Confirm a slot** — promote an emergent field once it recurs enough
  (`Observed → Candidate → Expected`). Requires a human approver.
- **Promote a claim** — move `Unverified → Confirmed` by supplying evidence that
  passes the per-domain gate.
- **Resolve a conflict** — record a decision + rationale on a ResolutionCase;
  it becomes reusable precedent and is versioned/reopenable.

Per-domain evidence bars:
- **Trading** — empirical (backtest/data), highest bar.
- **Real estate** — conditional/precedent, context-tagged.
- **TCM** — corroboration across lineages or classical citation; tops out at
  "attributed belief". Dissent is preserved, never forced into one winner.

---

## 7. Two personas

- **Chat/XR copilot** — a human asks questions; the engine judges evidence and
  answers with status + provenance. See `beta_xr_copilot_run.py`.
- **Autonomous agent** — an agent ingests scraped sources with no fabrication
  and cross-links claims. See `beta_agent_loop_run.py`.

---

## Known limitations

| Limitation | Impact | To fix |
|---|---|---|
| **Neo4j ingest not wired** | The engine's ingest pipeline targets the in-memory store; the Neo4j store implements only the query/vector side with a different interface. The MCP server's *ingestion* tools therefore don't work against Neo4j yet. | Build a `KnowledgeGraphStore` adapter that implements the same interface the engine expects (`upsert_entity(name)→Entity`, `add_claim→Claim`, `observe_slot`, `confirm_slot`, `list_canonical_claims`, `get_expected_slots`, ...). Approved-layer work. |
| **Rate limits** | MiMo gateway has rate limits; rapid repeated requests will hit 429. | Space out ingests or upgrade plan. |
| **Vector search needs Neo4j** | Claims are now embedded at ingest (local Ollama `bge-m3`, 1024-dim), but the file-backed store has no vector consumer, so `search_claims`/`search_entities` still need the Neo4j vector index. CLI `--query` uses keyword matching. | Wire the Neo4j write-path (above); the vectors are already produced and attached to claims. |
| **CLI curation commands** | Slot/claim/case promotion is via API/MCP, not yet the CLI. | Add `confirm-slot` / `promote-claim` / `resolve-case` subcommands on request. |

The CLI path (`scripts/ke.py`) is fully functional today for ingest + query on
the file-backed store and is the recommended UAT surface.
