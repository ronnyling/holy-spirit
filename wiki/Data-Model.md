# Data Model

## Nodes

| Node | Purpose | Key fields |
|---|---|---|
| `Entity` | Canonical concept; different experts' terms map to one entity | `id`, `canonical_name`, `aliases`, `description`, `version` |
| `Claim` | A single assertion about an entity | `id`, `entity_id`, `statement`, `slot_name`, `epistemic_status`, `provenance[]`, `embedding`, `version`, `tags[]` |
| `Evidence` | Support attached to a claim | `id`, `claim_id`, `source_kind`, `source_id`, `credibility` (0–1), `linked_claim_ids[]`, `notes` |
| `ResolutionCase` | Memory of how a conflict was resolved | `id`, `conflict_signature`, `conflicting_claim_ids[]`, `evidence_ids[]`, `decision`, `rationale`, `version`, `reopened_from_case_id`, `is_open` |
| `Slot` | Emergent template field learned from repetition | `id`, `entity_id`, `name`, `lifecycle`, `observed_count`, `candidate_count`, `expected_count`, `version` |
| `Transcript` | A harboured source document | `transcript_id`, `domain`, `entity_name`, `source_kind`, `source_id`, `sha256`, `char_count`, `chunk_count`, `raw_path`, `document_path`, `ingested_at` |
| `Chunk` | A deterministic slice of a transcript | `index`, `text`, `char_start`, `char_end`, `token_estimate` |

Provenance is embedded on each claim (`source_kind` ∈ `internal_wiki | external_doc | user`, `source_id`,
`source_ref`, `captured_at`, `notes`).

## Edges

```
(:Claim)-[:ABOUT]->(:Entity)
(:Evidence)-[:SUPPORTS]->(:Claim)
(:Claim)-[:SUPPORTS]->(:Claim)          // internal evidence: Confirmed only, cycle-checked
(:Claim)-[:CONFLICTS_WITH]->(:Claim)
(:ResolutionCase)-[:RESOLVES]->(:Claim)
(:Claim)-[:DERIVED_FROM]->(:Chunk)
(:Chunk)-[:PART_OF]->(:Transcript)
(:Transcript)-[:ABOUT]->(:Entity)
(:Entity)-[:ALIAS_OF]->(:Entity)
(:Slot)-[:OF]->(:Entity)
```

## Epistemic status

Every claim carries exactly one:

`Confirmed | Unverified | Unknown | Unverifiable | Disputed | Retracted`

- `Unknown` is first-class and may be **permanent** — an open question, not an error.
- `Unverified` = observed but not proven. User-sourced claims enter here and never auto-confirm.
- `Disputed` = conflicts with current canonical knowledge.
- `Retracted` = kept in memory but not treated as active truth.

## Slot lifecycle

`Observed → Candidate → Expected → Retired`

- The engine **observes** automatically (counting only).
- Promotion is **never autonomous**; Candidate/Expected transitions require human confirmation.
- Current prototype thresholds: Candidate at 3 observations, Expected at 5 (deliberately low; expected to
  rise for real cold-start scale).
- Guard against cold start, drift/self-reinforcement, frequency ≠ importance, and alert fatigue.

## Vector index

`Claim.embedding` is indexed by Neo4j's native vector index (`claim_embedding_index`). The index
dimension must match the embedding provider's output dimension (see `.env.example`).
