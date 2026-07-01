# Usage Scenarios

How the engine is used *in practice*, as two end-to-end personas. Both are runnable, self-verifying
beta scripts at the repository root; neither modifies engine code, and both exercise the real pipeline.

Each scenario is explicit about the **bounded vs unbounded split** (per `spec/agent_protocol.md`):
deterministic work runs in code today; the genuinely semantic/natural-language parts are marked
`[stand-in]` and are the job of the future LLM layer (README → *Still to build*).

| Scenario | Script | Human in the loop? | Knowledge source |
|---|---|---|---|
| 1. XR smart-glasses research copilot | [`beta_xr_copilot_run.py`](../beta_xr_copilot_run.py) | Yes — user supplies evidence | User capture + user-provided documents |
| 2. Autonomous agent research loop | [`beta_agent_loop_run.py`](../beta_agent_loop_run.py) | No — fully autonomous | Web scrape (verbatim only) |

Run either from the package root:

```powershell
cd "c:\Users\r.a.ling\OneDrive - Avanade\Documents\work\Native AI\knowledge_engine"
$env:PYTHONPATH = "src"
python beta_xr_copilot_run.py
python beta_agent_loop_run.py
```

Both print `RESULT: PASSED - all checks green.` They are **not** `test_`-prefixed, so pytest does not
collect them; the 33-test suite is unaffected.

---

## Scenario 1 — XR smart-glasses research copilot (human-in-the-loop)

A researcher wearing XR smart glasses does property fieldwork. The copilot is a thin deterministic
wrapper over the engine. It shows how the engine **refuses to accept unsupported or out-of-context
input** and only canonicalizes knowledge once credible evidence clears the domain gate.

### Flow

1. **Capture** — glasses stream a voice note plus OCR from a signboard. The copilot POSTs it (the real
   XR client contract is the MCP `ingest_transcript` tool). Because it is `source_kind="user"` with no
   evidence, the claim enters **Unverified** and the engine **raises a semantic evidence-gap flag** —
   the copilot immediately asks what the claim depends on.
2. **Weak evidence rejected** — the user offers hearsay (low credibility). The **real evidence gate**
   rejects it: the domain score bar is not met. The claim stays Unverified.
3. **Wrong-context evidence rejected** — the user cites an existing but **disputed** wiki claim as
   backing. Internal-wiki evidence is only valid when the supporting claim is already **Confirmed**, so
   the gate rejects it as an unconfirmed source.
4. **Credible evidence accepted** — the user supplies a valuation firm's study (`external_doc`,
   credibility 0.9). The gate passes and the claim is **promoted to Confirmed canonical**.
5. **Query & chat** — the user asks a question. The copilot deterministically **retrieves** the best
   matching confirmed claim, **links** the strongest contrasting confirmed claim (theme-token overlap),
   and **names the root cause** by reading the stored resolution-case decision.

### What is bounded vs unbounded

- **Bounded (real code today):** the evidence gate (score, source count, internal-source confirmation),
  the gap flag, claim promotion, retrieval, linking, and the root-cause lookup from resolution memory.
- **Unbounded (`[stand-in]` — future LLM):** natural-language phrasing of the answer and the *semantic*
  judgement of whether a piece of evidence is contextually on-point.

### Maintenance notes

- Human-in-the-loop is mandatory: user captures never auto-trust — they remain Unverified until credible
  evidence clears the domain gate.
- The XR client contract is the existing MCP tools (`ingest_transcript`, `promote_claim`); no engine
  change is needed to add hardware.

---

## Scenario 2 — Autonomous agent research loop (web-scraped, no smart-guessing)

A plain Python loop spawns an AI worker per task. Each worker uses the LLM wiki to **evaluate** its
task, scrapes the web for real information, and learns **only from scraped text** — it never fabricates
a claim.

### Flow (per spawned worker)

1. **Evaluate against the wiki** — the worker queries confirmed knowledge for the task's keywords. Zero
   matches ⇒ **research gap** ⇒ it must go scrape.
2. **Scrape** — a real `httpx` GET of the source page, with a **verified cached scrape** as the offline
   fallback (corporate networks often block egress). A short distinctive fragment is used to re-verify
   freshness against the live page.
3. **Extract (no smart-guessing)** — claims are built **strictly from verbatim scraped sentences**; the
   code asserts every claim statement is a scraped sentence, so fabrication is structurally impossible.
   Each claim carries its **source URL** as `external_doc` evidence.
4. **Ingest → confirm** — the engine evidence-gates the scraped facts and confirms them as canonical.
5. **Cross-link** — the newly learned facts are linked back to existing wiki knowledge, surfacing a
   **root-cause corroboration** (an independent source converging on a prior finding).

### Housekeeping demonstrated

- **Idempotent harbour** — re-running the identical scrape is deduped by SHA-256 content hash; the
  registry reports `created=False` and no duplicate transcript is stored. (Harbour dedup is at the
  *transcript* level — call the registry directly for a housekeeping check so it does not create a
  duplicate claim.)
- **Freshness** — each loop re-verifies its cached scrape against the live source URL.
- **Provenance** — every claim keeps its source URL; the agent cannot confirm anything it did not scrape.

### What is bounded vs unbounded

- **Bounded (real code today):** the HTTP fetch, verbatim sentence extraction, evidence gating, dedup,
  and cross-linking.
- **Unbounded (`[stand-in]` — future LLM):** selecting *which* scraped sentences are the salient claims
  and any paraphrase/synthesis — which must still cite the scraped text.

### Maintenance notes

- No-fabrication is enforced in code, not by convention: `extract_claims` only echoes scraped text.
- Staleness policy (future): re-scrape and re-verify on a schedule; a source edit that drops the tracked
  fragment flags the cached claim for human review.

---

## How these map to the locked design

- Both honour **user/agent input is a source, not an oracle** — nothing is canonical without evidence.
- Both keep the **bounded/unbounded split** intact; no LLM behaviour is faked in deterministic code.
- Scenario 1 exercises the **evidence gate, gap flagging, and resolution memory**;
  Scenario 2 exercises **harbouring/housekeeping, provenance, and cross-linking**.
- The `[stand-in]` lines are the precise seams where the planned LLM layer (semantic gap detection,
  reconciliation, natural-language answering) will plug in without disturbing the bounded core.
