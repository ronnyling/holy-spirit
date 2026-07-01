# Continuous R&D Knowledge Engine — Wiki

A living AI wiki: drop in multi-expert transcripts (TCM, real estate, stock trading); the engine turns
them into evidence-gated canonical knowledge that an agent can use as working experience — without
flattening disagreement into a single anonymous answer.

## Reading order

1. [Architecture](Architecture.md) — the graph-first design and the hard no-fallback rule.
2. [Data Model](Data-Model.md) — nodes, edges, epistemic status, and slot lifecycle.
3. [Pipeline](Pipeline.md) — how a transcript becomes canonical knowledge.
4. [Ingestion & Housekeeping](Ingestion-and-Housekeeping.md) — harbouring, chunking, documentation, idempotency.
5. [Setup: Neo4j](Setup-Neo4j.md) — run the graph engine locally.
6. [MCP API](MCP-API.md) — the agent-facing interface.
7. [Domain Policies](Domain-Policies.md) — per-domain evidence gates and preserved dissent.
8. [Usage Scenarios](Usage-Scenarios.md) — end-to-end personas: XR research copilot and autonomous scraping agent.

## First principles

- Every transcript is **evidence, not truth**.
- **Gaps are flagged before conflicts** are checked.
- Conflicts produce **consensus with preserved dissent**, never a forced winner.
- Nothing becomes canonical **without evidence**, and no claim can justify itself.
- The **user is a source, not an oracle** — user input enters as `Unverified`.
- **No fallbacks**: a missing backend is an error, not a degraded mode.
