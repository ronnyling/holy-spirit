# Domain Policies

Resolution runs through **one shared pipeline**, but the evidence bar and the way dissent is handled are
pluggable per domain. A claim is confirmed only when it clears both the **score** bar (sum of evidence
credibility, each 0–1) and the **minimum distinct sources** bar for its domain.

## Evidence gates (as implemented in `policy.py`)

| Domain | Min score | Min distinct sources | Rationale |
|---|---|---|---|
| Trading | 1.2 | 2 | Highest bar; corroborated and ideally empirical (backtest/data). |
| TCM | 1.2 | 2 | No single authority; require multiple lineages or a classical citation. |
| Real estate | 0.8 | 1 | Conditional/precedent evidence, context-tagged. |
| Default | 0.8 | 1 | For unclassified domains. |

- User-sourced claims never auto-confirm on ingestion — they enter as `Unverified`.
- Internal wiki evidence contributes only if the supporting claim is already `Confirmed` and adds no
  dependency cycle.

## Evidence sufficiency by domain

- **Trading** → empirical, backtest/data first. Regime-tagged.
- **Real estate** → conditional rules keyed to market / jurisdiction / cycle. Context-tagged.
- **TCM** → corroboration across multiple lineages or a classical citation; tops out at "attributed belief".

## Preserved dissent by domain

| Domain | What good looks like | Typical conflict style |
|---|---|---|
| TCM | Preserve school/lineage differences and annotate dissent | Terminology mismatch, lineages, non-falsifiable beliefs |
| Real estate | Conditional rules keyed to market, jurisdiction, cycle | Context-dependent strategy differences |
| Trading | Evidence-weighted, regime-tagged, backtest-aware claims | Direct contradictions, falsifiable strategy claims |

The system produces **consensus with preserved dissent** — it never forces a single winner where the
domain does not support one, and it always surfaces source attribution and disagreement.
