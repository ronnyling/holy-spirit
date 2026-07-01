# Pipeline

The ingestion pipeline order is fixed. **Gap check runs before conflict check.**

```
1. Ingest transcript
2. Harbour: normalize → hash → chunk → auto-document → register   (deterministic, idempotent)
3. Extract claims
4. Observe slots + learn frequencies
5. Structural gap check, then semantic gap check
6. Flag gaps immediately (if any)
7. Conflict detection vs canonical knowledge
8. Open or reuse a ResolutionCase (if conflict)
9. Research with the user to gather evidence
10. Promote to canonical only when evidence gates pass
11. Keep the evidence chain + resolution history
```

## Gap check (before conflicts)

- **Structural gaps** are deterministic: an `Expected` slot is missing from the entity.
- **Semantic gaps** are LLM-backed: unclear context, missing rationale, missing conditions.
- Gaps are surfaced at intake so the user can clarify on the spot.

## Conflict detection

- Tracked per aspect/slot, not just per entity.
- Same entity + same slot + contradictory statement → open a `ResolutionCase`.
- Cases are versioned, reopenable, and reusable — future similar conflicts retrieve prior cases first.

## Evidence-gated promotion

- Nothing reaches `Confirmed`/canonical without evidence (internal wiki OR external).
- Internal evidence counts only if the supporting claim is already `Confirmed` **and** adding it creates
  no dependency cycle. Cycle detection is native Cypher over `SUPPORTS` edges.
- A claim clears its domain's **score** and **source-count** bars before confirmation (see
  [Domain Policies](Domain-Policies.md)).
- User-sourced claims never auto-confirm; they enter as `Unverified` and must be promoted with evidence.

## Consensus with preserved dissent

The system produces consensus with preserved dissent — it never forces a single winner where the domain
does not support one. Source attribution and disagreement are always surfaced; synthesized knowledge is
never presented as anonymous fact (these domains carry real health and financial liability).
