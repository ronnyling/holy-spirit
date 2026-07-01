"""End-to-end beta run for the Continuous R&D Knowledge Engine.

Scenario: two competing real-estate philosophies are fed to the engine as expert
transcripts, then a set of corroborating, location-specific transcripts add nuance.
The engine is expected to:

  1. Confirm the first philosophy as canonical (evidence-gated).
  2. Detect that the second philosophy conflicts on the same aspect (slot) and open a
     collaborative resolution case instead of silently overwriting.
  3. Learn a `location_context` slot from repeated observation and suggest promotion.
  4. Confirm location-specific nuance claims as canonical knowledge.
  5. Reuse a prior resolution case as precedent when a similar conflict recurs.
  6. Flag a structural gap when a transcript omits a now-`Expected` slot.
  7. Prevent an internal-evidence dependency cycle.

The synthesized canonical knowledge at the end reads like:

    Rental buy-and-hold is the superior strategy in tier-1 city cores and dense
    grade-A office districts with white-collar tenants (KLCC, PFCC), along the mature
    LRT Kelana Jaya line, and near major universities (Sunway City) where high yield
    meets easy disposal -- but NOT in Cyberjaya near MMU, where yield is acceptable yet
    the asset is hard to dispose of, even back to investors.

Run:  python beta_full_run.py
"""

from __future__ import annotations

import sys

from knowledge_engine import (
    ClaimDraft,
    EpistemicStatus,
    EvidenceDraft,
    GapKind,
    KnowledgeEngine,
    SlotLifecycle,
    TranscriptInput,
)

ENTITY = "Malaysia Residential Property Strategy"

_failures: list[str] = []


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}")
    if not condition:
        _failures.append(label)


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ingest(engine: KnowledgeEngine, *, speaker: str, source_id: str, source_kind: str,
           statement: str, slot_name: str, observed_slots: list[str],
           evidence: list[EvidenceDraft] | None = None, credibility: float = 0.5):
    """Ingest one claim as a transcript and print a compact outcome line."""
    outcome = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name=ENTITY,
            transcript_text=f"[{speaker}] {statement}",
            source_kind=source_kind,  # type: ignore[arg-type]
            source_id=source_id,
            evidence_credibility=credibility,
            claim_drafts=[
                ClaimDraft(
                    statement=statement,
                    slot_name=slot_name,
                    observed_slots=observed_slots,
                    evidence=evidence or [],
                )
            ],
        )
    )
    tag = (
        "CONFIRMED" if outcome.confirmed_claim_ids
        else "DISPUTED" if outcome.disputed_claim_ids
        else "UNVERIFIED"
    )
    print(f"  - {speaker:<22} slot={slot_name:<24} -> {tag}")
    return outcome


def doc(source_id: str, credibility: float = 0.9) -> list[EvidenceDraft]:
    return [EvidenceDraft(source_kind="external_doc", source_id=source_id, credibility=credibility)]


def main() -> int:
    engine = KnowledgeEngine()

    # ------------------------------------------------------------------ #
    # PHASE 1 - Two competing philosophies collide on the same aspect.
    # ------------------------------------------------------------------ #
    rule("PHASE 1  Competing philosophies -> conflict + resolution case")

    phil_rental = ingest(
        engine,
        speaker="Expert A (rental)",
        source_id="doc-rental-thesis",
        source_kind="external_doc",
        statement="Buy-and-hold rental property is the best residential strategy.",
        slot_name="strategy",
        observed_slots=["strategy"],
        evidence=doc("doc-rental-thesis"),
    )
    check("Philosophy A (rental) confirmed as canonical", bool(phil_rental.confirmed_claim_ids))
    rental_claim_id = phil_rental.confirmed_claim_ids[0] if phil_rental.confirmed_claim_ids else ""

    phil_flip = ingest(
        engine,
        speaker="Expert B (flipping)",
        source_id="doc-flip-thesis",
        source_kind="external_doc",
        statement="Flipping property is the best residential strategy.",
        slot_name="strategy",
        observed_slots=["strategy"],
        evidence=doc("doc-flip-thesis"),
    )
    check("Philosophy B (flipping) detected as conflicting", bool(phil_flip.disputed_claim_ids))
    check("Resolution case opened for the strategy conflict", bool(phil_flip.open_case_ids))
    strategy_case_id = phil_flip.open_case_ids[0] if phil_flip.open_case_ids else ""
    if phil_flip.conflict_summaries:
        print(f"    -> case {strategy_case_id}: {phil_flip.conflict_summaries[0].message}")

    # ------------------------------------------------------------------ #
    # PHASE 2 - Location-specific nuance accrues as canonical knowledge,
    #           and a `location_context` slot is learned from repetition.
    # ------------------------------------------------------------------ #
    rule("PHASE 2  Location nuance -> canonical claims + slot learning")

    location_claims = [
        (
            "Expert C (KLCC/PFCC)",
            "doc-office-core",
            "Rental buy-and-hold excels in KLCC and PFCC grade-A office cores where "
            "white-collar tenant demand keeps occupancy and rents high.",
            "location_office_core",
        ),
        (
            "Expert D (LRT KJ line)",
            "doc-lrt-kj",
            "Rental buy-and-hold performs strongly along the mature LRT Kelana Jaya "
            "line because connectivity sustains tenant demand.",
            "location_lrt_kelana_jaya",
        ),
        (
            "Expert E (Sunway City)",
            "doc-sunway-uni",
            "Rental near Sunway City universities delivers high yield and easy "
            "disposal thanks to constant student demand.",
            "location_university_sunway",
        ),
        (
            "Expert F (Cyberjaya/MMU)",
            "doc-cyberjaya-mmu",
            "Rental near MMU Cyberjaya has acceptable yield but poor disposal "
            "liquidity and is hard to exit, even back to investors.",
            "location_cyberjaya_mmu",
        ),
    ]

    for speaker, source_id, statement, slot in location_claims:
        outcome = ingest(
            engine,
            speaker=speaker,
            source_id=source_id,
            source_kind="external_doc",
            statement=statement,
            slot_name=slot,
            observed_slots=["location_context", slot],
            evidence=doc(source_id),
        )
        check(f"{speaker} nuance confirmed as canonical", bool(outcome.confirmed_claim_ids))

    # A fifth corroboration pushes `location_context` past the Expected threshold (>=5).
    fifth = ingest(
        engine,
        speaker="Expert G (yield survey)",
        source_id="doc-yield-survey",
        source_kind="external_doc",
        statement="Across these submarkets, location context drives rental outcomes "
                  "more than the buy-versus-flip choice alone.",
        slot_name="location_context_summary",
        observed_slots=["location_context"],
        evidence=doc("doc-yield-survey"),
    )
    suggestions = fifth.slot_suggestions
    check("location_context slot crossed a learned threshold", bool(suggestions))
    if suggestions:
        s = suggestions[-1]
        print(f"    -> suggest {s.current_lifecycle} -> {s.suggested_lifecycle} "
              f"({s.observed_count} observations)")

    promoted_slot = engine.confirm_slot(
        entity_name=ENTITY,
        slot_name="location_context",
        confirmed_by="beta-analyst",
        target=SlotLifecycle.EXPECTED,
    )
    check("location_context promoted to Expected by human confirmation",
          promoted_slot["lifecycle"] == SlotLifecycle.EXPECTED.value)

    # ------------------------------------------------------------------ #
    # PHASE 3 - Human resolves the philosophy conflict into nuanced canon.
    # ------------------------------------------------------------------ #
    rule("PHASE 3  Collaborative resolution of the philosophy conflict")

    decision = (
        "Rental buy-and-hold is the canonical superior strategy in tier-1 city cores, "
        "dense grade-A office districts with white-collar tenants (KLCC, PFCC), along the "
        "mature LRT Kelana Jaya line, and near major universities (Sunway City) where high "
        "yield meets easy disposal. Flipping is situational, not categorically best. "
        "Exception: Cyberjaya/MMU carries disposal-liquidity risk despite acceptable yield."
    )
    resolved = engine.resolve_case(
        strategy_case_id,
        decision=decision,
        rationale="Location-context evidence outweighs the location-agnostic 'flipping is best' thesis.",
    )
    check("Strategy conflict case resolved and closed", resolved["is_open"] is False)
    print(f"    -> decision recorded (v{resolved['version']}): {decision[:72]}...")

    # ------------------------------------------------------------------ #
    # PHASE 4 - A similar conflict recurs -> precedent is reused.
    # ------------------------------------------------------------------ #
    rule("PHASE 4  Recurring conflict reuses the prior case as precedent")

    phil_wholesale = ingest(
        engine,
        speaker="Expert H (wholesaling)",
        source_id="doc-wholesale-thesis",
        source_kind="external_doc",
        statement="Wholesaling property is the best residential strategy.",
        slot_name="strategy",
        observed_slots=["strategy"],
        evidence=doc("doc-wholesale-thesis"),
    )
    check("Recurring strategy conflict detected", bool(phil_wholesale.open_case_ids))
    if phil_wholesale.open_case_ids:
        reopened = engine.store.get_resolution_case(phil_wholesale.open_case_ids[0])
        check("New case links back to the original as precedent",
              reopened.reopened_from_case_id == strategy_case_id)
        print(f"    -> reopened_from_case_id = {reopened.reopened_from_case_id}")

    # ------------------------------------------------------------------ #
    # PHASE 5 - Omitting a now-Expected slot triggers a structural gap.
    # ------------------------------------------------------------------ #
    rule("PHASE 5  Missing Expected slot -> proactive structural gap flag")

    gap_outcome = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name=ENTITY,
            transcript_text="[Expert I] A note about financing that forgets to state the location.",
            source_kind="external_doc",
            source_id="doc-financing-note",
            claim_drafts=[
                ClaimDraft(
                    statement="Fixed-rate financing improves buy-and-hold cash flow stability.",
                    slot_name="financing",
                    observed_slots=["financing"],
                    evidence=doc("doc-financing-note"),
                )
            ],
        )
    )
    structural = [g for g in gap_outcome.gap_flags if g.kind == GapKind.STRUCTURAL]
    check("Structural gap raised for the missing location_context",
          any(g.slot_name == "location_context" for g in structural))
    for g in structural:
        print(f"    -> GAP ({g.slot_name}): {g.question}")

    # ------------------------------------------------------------------ #
    # PHASE 6 - Internal evidence promotion + dependency-cycle prevention.
    # ------------------------------------------------------------------ #
    rule("PHASE 6  Internal evidence ledger + cycle prevention")

    base = ingest(
        engine,
        speaker="Expert J (cashflow)",
        source_id="doc-cashflow",
        source_kind="external_doc",
        statement="Positive net rental cash flow is the primary buy-and-hold return driver.",
        slot_name="cashflow_principle",
        observed_slots=["cashflow_principle", "location_context"],
        evidence=doc("doc-cashflow"),
    )
    base_id = base.confirmed_claim_ids[0] if base.confirmed_claim_ids else ""

    dependent = ingest(
        engine,
        speaker="Practitioner (unverified)",
        source_id="session-appreciation",
        source_kind="user",
        statement="Appreciation compounds on top of a stable rental cash-flow base.",
        slot_name="appreciation_principle",
        observed_slots=["appreciation_principle", "location_context"],
    )
    dependent_id = dependent.claim_ids[0] if dependent.claim_ids else ""
    check("User-sourced claim stays Unverified until evidenced",
          dependent_id in dependent.unverified_claim_ids)

    promoted = engine.promote_claim(
        dependent_id,
        [EvidenceDraft(source_kind="internal_wiki", source_id=base_id, credibility=1.0)],
    )
    check("Dependent claim promoted via internal_wiki evidence",
          promoted.epistemic_status == EpistemicStatus.CONFIRMED)

    cycle_prevented = False
    try:
        engine.evidence_ledger.attach_internal_support(
            source_claim_id=dependent_id,
            target_claim_id=base_id,
            store=engine.store,
        )
    except ValueError as exc:
        cycle_prevented = True
        print(f"    -> cycle correctly rejected: {exc}")
    check("Internal-evidence dependency cycle prevented", cycle_prevented)

    # ------------------------------------------------------------------ #
    # SYNTHESIS - The canonical knowledge the engine now holds.
    # ------------------------------------------------------------------ #
    rule("SYNTHESIS  Canonical knowledge for the entity")

    entity_id = phil_rental.entity_id
    canonical = engine.store.list_canonical_claims(entity_id)
    print(f"  Entity: {ENTITY}")
    print(f"  Confirmed canonical claims: {len(canonical)}\n")
    for claim in canonical:
        print(f"   [{claim.slot_name or 'general'}]")
        print(f"      {claim.statement}")

    print("\n  Resolved philosophy decision:")
    resolved_case = engine.store.get_resolution_case(strategy_case_id)
    print(f"      {resolved_case.decision}")

    print("\n  State snapshot:")
    for key, value in engine.state_snapshot().items():
        print(f"      {key:<18}: {value}")

    # ------------------------------------------------------------------ #
    rule("RESULT")
    if _failures:
        print(f"  BETA RUN FAILED - {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  BETA RUN PASSED - all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
