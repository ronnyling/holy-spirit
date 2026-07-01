# MCP API

The engine is exposed to agents through an MCP server (`server.py`).

## Tools

| Tool | Purpose |
|---|---|
| `ingest_transcript` | Harbour a transcript and run the full pipeline; returns entity, harbour metadata, gaps, conflicts, and claim outcomes. |
| `confirm_slot` | Human-in-the-loop promotion of a slot (`Observed → Candidate → Expected`). Requires `confirmed_by`. |
| `promote_claim` | Promote a claim's epistemic status when evidence gates are satisfied. |
| `resolve_case` | Record a decision + rationale on a `ResolutionCase` (versioned, reopenable). |
| `state_snapshot` | Return a JSON snapshot of current engine state. |

## Resource

- `knowledge://state` — a JSON snapshot of the current knowledge state (entities, claims, slots, cases).

## Ingest outcome

`ingest_transcript` returns a `TranscriptOutcome`:

| Field | Meaning |
|---|---|
| `entity_id` | The entity the transcript is about. |
| `transcript_id` | Stable harbour id (`{domain}-{entity}-{hash12}`), or `null` if no registry is wired. |
| `transcript_created` | `False` when identical content was already harboured (idempotent housekeeping). |
| `chunk_count` | Number of deterministic chunks produced. |
| `claim_ids` / `confirmed_claim_ids` | Claims created and those that cleared their evidence gate. |
| gaps / conflicts | Flagged gaps and detected conflicts for human follow-up. |

## Human-in-the-loop gates

Schema promotion (`confirm_slot`), conflict resolution (`resolve_case`), and canonical promotion
(`promote_claim`) all require explicit human action. The agent surfaces gaps and conflicts; it does not
resolve them autonomously, and user input never becomes canonical without evidence.
