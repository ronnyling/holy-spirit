"""Beta run - Persona 1: XR smart-glasses research copilot (human-in-the-loop).

This models a real researcher wearing XR smart glasses while doing property
research. The copilot is a thin *deterministic* wrapper around the knowledge
engine. It exercises the real pipeline end to end:

    capture (voice + visual OCR)  ->  engine flags missing evidence (semantic gap)
    ->  user offers WEAK evidence  ->  evidence gate rejects (not impactful)
    ->  user offers WRONG-CONTEXT evidence  ->  gate rejects (unconfirmed source)
    ->  user offers CREDIBLE evidence  ->  claim confirmed as canonical
    ->  user queries/chats  ->  copilot answers, links related knowledge,
                                 and names the underlying root cause.

BOUNDED vs UNBOUNDED (per spec/agent_protocol.md 1.3):
  * Evidence sufficiency / contextual validity  -> BOUNDED (the real evidence
    gate: score, source count, and internal-source confirmation). Done in code.
  * Conflict / linking / root-cause tracing      -> BOUNDED (graph traversal over
    confirmed claims + the resolution-case decision). Done in code.
  * Natural-language phrasing of the answer, and TRUE semantic judgement of
    "is this evidence contextually on-point"     -> UNBOUNDED. The current build
    has no LLM layer (README: "still to build"), so the copilot's wording is a
    RULE-BASED STAND-IN, marked [stand-in] wherever the future LLM would speak.

No engine code is modified. XR capture is simulated (no hardware) but the data
path is the real one an XR client would POST to the MCP `ingest_transcript` tool.

Run:  python beta_xr_copilot_run.py
"""

from __future__ import annotations

import sys

from knowledge_engine import (
    ClaimDraft,
    EpistemicStatus,
    EvidenceDraft,
    KnowledgeEngine,
    TranscriptInput,
)
from knowledge_engine.utils import normalize_text

ENTITY = "Malaysia Residential Property Strategy"
_failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"      [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        _failures.append(label)


def step(n: int, title: str) -> None:
    print(f"\n--- STEP {n}: {title} " + "-" * max(0, 60 - len(title)))


def glasses(msg: str) -> None:
    print(f"  (XR capture) {msg}")


def copilot(msg: str, stand_in: bool = False) -> None:
    tag = "[stand-in] " if stand_in else ""
    print(f"  >> COPILOT: {tag}{msg}")


def user(msg: str) -> None:
    print(f"  << USER   : {msg}")


def doc(source_id: str, ref: str, cred: float = 0.9) -> EvidenceDraft:
    return EvidenceDraft(source_kind="external_doc", source_id=source_id, source_ref=ref, credibility=cred)


def find_claim_id(engine: KnowledgeEngine, entity_id: str, slot: str) -> str:
    for claim in engine.store.claims.values():
        if claim.entity_id == entity_id and (claim.slot_name or "") == slot:
            return claim.id or ""
    return ""


# --------------------------------------------------------------------------- #
# Deterministic copilot behaviours (bounded).
# --------------------------------------------------------------------------- #
class ResearchCopilot:
    def __init__(self, engine: KnowledgeEngine, entity_id: str, root_cause_decision: str) -> None:
        self.engine = engine
        self.entity_id = entity_id
        self.root_cause_decision = root_cause_decision

    def judge_evidence(self, claim_id: str, evidence: list[EvidenceDraft]) -> tuple[bool, list[str]]:
        """Wraps the REAL evidence gate. Returns (accepted, reasons_if_rejected)."""
        claim = self.engine.store.load_claim(claim_id)
        domain = claim.tags[0] if claim.tags else "default"
        evaluation = self.engine.evidence_ledger.evaluate(domain, evidence, self.engine.store, claim)
        return evaluation.can_confirm, evaluation.reasons

    def answer(self, query: str) -> None:
        """Deterministic retrieve -> link -> root-cause over confirmed claims."""
        tokens = set(normalize_text(query).split())
        canonical = self.engine.store.list_canonical_claims(self.entity_id)

        def score(text: str) -> int:
            return len(tokens & set(normalize_text(text).split()))

        ranked = sorted(canonical, key=lambda c: score(c.statement), reverse=True)
        primary = [c for c in ranked if score(c.statement) > 0]
        if not primary:
            copilot("I have no confirmed knowledge matching that yet.", stand_in=True)
            return

        top = primary[0]
        copilot(f"Direct answer -> {top.statement}", stand_in=True)

        # Link: the confirmed claim with the strongest shared-theme overlap (contrast).
        theme = {"disposal", "liquidity", "exit", "resale", "demand", "student", "university"}
        linked = sorted(
            (c for c in canonical
             if c.id != top.id and (theme & set(normalize_text(c.statement).split()))),
            key=lambda c: len(theme & set(normalize_text(c.statement).split())),
            reverse=True,
        )
        if linked:
            copilot(f"Related & linked -> {linked[0].statement}", stand_in=True)

        # Root cause: the resolution-memory decision that explains the difference.
        copilot(f"Root cause -> {self.root_cause_decision}", stand_in=True)
        return


def seed_wiki(engine: KnowledgeEngine) -> tuple[str, str]:
    """Seed a small confirmed knowledge base and one resolved philosophy conflict."""
    def ingest(statement: str, slot: str, source_id: str, kind: str = "external_doc",
               evidence: list[EvidenceDraft] | None = None):
        return engine.ingest_transcript(
            TranscriptInput(
                domain="real estate",
                entity_name=ENTITY,
                transcript_text=statement,
                source_kind=kind,  # type: ignore[arg-type]
                source_id=source_id,
                claim_drafts=[ClaimDraft(statement=statement, slot_name=slot,
                                         observed_slots=[slot], evidence=evidence or [])],
            )
        )

    ingest("Buy-and-hold rental property is the best residential strategy.",
           "strategy", "seed-rental", evidence=[doc("seed-rental", "rental thesis")])
    flip = ingest("Flipping property is the best residential strategy.",
                  "strategy", "seed-flip", evidence=[doc("seed-flip", "flip thesis")])
    ingest("Rental buy-and-hold excels in KLCC and PFCC grade-A office cores with white-collar tenant demand.",
           "location_office_core", "seed-office", evidence=[doc("seed-office", "office core report")])
    ingest("Rental near Sunway City universities delivers high yield and easy disposal from constant student demand.",
           "location_university_sunway", "seed-sunway", evidence=[doc("seed-sunway", "campus rental study")])
    ingest("Rental near MMU Cyberjaya has acceptable yield but poor disposal liquidity and is hard to exit.",
           "location_cyberjaya_mmu", "seed-cyberjaya", evidence=[doc("seed-cyberjaya", "Cyberjaya study")])

    entity_id = flip.entity_id
    flip_claim_id = flip.disputed_claim_ids[0] if flip.disputed_claim_ids else ""
    case_id = flip.open_case_ids[0] if flip.open_case_ids else ""

    decision = (
        "Disposal liquidity - not headline yield - decides rental success. Rental wins in "
        "tier-1 cores and university-adjacent submarkets where exit is easy (KLCC/PFCC, Sunway); "
        "Cyberjaya/MMU is the exception: acceptable yield but a thin resale/investor market."
    )
    engine.resolve_case(case_id, decision=decision, rationale="Location-context liquidity outweighs the location-agnostic thesis.")
    return entity_id, flip_claim_id


def main() -> int:
    print("=" * 70)
    print("PERSONA 1  XR smart-glasses research copilot (human-in-the-loop)")
    print("=" * 70)

    engine = KnowledgeEngine()
    entity_id, flip_claim_id = seed_wiki(engine)
    decision = next(c.decision for c in engine.store.resolution_cases.values() if c.decision)
    copilot_agent = ResearchCopilot(engine, entity_id, decision)
    print(f"  Seeded wiki: {len(engine.store.list_canonical_claims(entity_id))} confirmed canonical claims.")

    # --------------------------------------------------------------------- #
    step(1, "XR capture -> engine flags a missing-evidence gap")
    glasses("voice: 'note - rental units near MMU Cyberjaya can almost never be flipped quickly to another investor'")
    glasses("vision OCR: signboard 'FOR SALE - 2 years on market' outside an MMU-area condo")
    capture = engine.ingest_transcript(
        TranscriptInput(
            domain="real estate",
            entity_name=ENTITY,
            transcript_text="Field note near MMU Cyberjaya with a long-standing resale signboard.",
            source_kind="user",
            source_id="xr-session-001",
            claim_drafts=[
                ClaimDraft(
                    statement="Rental units near MMU Cyberjaya can almost never be flipped quickly to another investor.",
                    slot_name="exit_liquidity_cyberjaya",
                    observed_slots=["exit_liquidity_cyberjaya"],
                    evidence=[],  # observation only - no evidence yet
                )
            ],
        )
    )
    new_claim_id = capture.claim_ids[0]
    check("Claim captured but left Unverified", new_claim_id in capture.unverified_claim_ids)
    check("Engine raised a semantic evidence-gap flag", bool(capture.gap_flags))
    if capture.gap_flags:
        copilot(f"I logged that observation, but it needs support. {capture.gap_flags[0].question}")

    # --------------------------------------------------------------------- #
    step(2, "User offers WEAK evidence -> gate rejects as not impactful")
    user("here's a source: 'a guy at the mamak said units there never sell' (hearsay)")
    weak = [EvidenceDraft(source_kind="user", source_id="mamak-hearsay", credibility=0.3)]
    accepted, reasons = copilot_agent.judge_evidence(new_claim_id, weak)
    check("Weak evidence rejected by the gate", not accepted)
    copilot(f"That isn't impactful enough to confirm it. Reason: {reasons[0]} "
            f"Can you get a stronger, verifiable source?")

    # --------------------------------------------------------------------- #
    step(3, "User offers WRONG-CONTEXT evidence -> gate rejects as unconfirmed")
    user("ok, cite the 'flipping is best' claim already in the wiki as backing")
    wrong_ctx = [EvidenceDraft(source_kind="internal_wiki", source_id=flip_claim_id, credibility=1.0)]
    accepted, reasons = copilot_agent.judge_evidence(new_claim_id, wrong_ctx)
    check("Out-of-context internal reference rejected", not accepted)
    copilot(f"That reference isn't valid context - it points to a claim that is disputed, "
            f"not confirmed. Reason: {reasons[0]}")

    # --------------------------------------------------------------------- #
    step(4, "User offers CREDIBLE evidence -> claim confirmed as canonical")
    user("here's a valuation firm's Cyberjaya resale-liquidity study (2026)")
    good = [doc("knight-frank-cyberjaya-2026", "Cyberjaya resale liquidity study 2026", cred=0.9)]
    accepted, reasons = copilot_agent.judge_evidence(new_claim_id, good)
    check("Credible evidence accepted by the gate", accepted)
    promoted = engine.promote_claim(new_claim_id, good)
    check("Claim promoted to Confirmed canonical", promoted.epistemic_status == EpistemicStatus.CONFIRMED)
    copilot("Confirmed. That's now canonical knowledge in the wiki.")

    # --------------------------------------------------------------------- #
    step(5, "User queries the copilot -> answer + link + root cause")
    q = "Should I buy a condo near MMU Cyberjaya to rent it out?"
    user(q)
    copilot_agent.answer(q)

    # --------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("MAINTENANCE / OPERATIONAL NOTES (persona 1)")
    print("=" * 70)
    snap = engine.state_snapshot()
    print(f"  - State after session: {snap}")
    print("  - Human-in-the-loop is mandatory: user-sourced captures stay Unverified")
    print("    until credible evidence clears the domain gate (no silent auto-trust).")
    print("  - XR client contract = MCP `ingest_transcript` / `promote_claim` (unchanged).")
    print("  - FUTURE LLM layer replaces the [stand-in] lines: semantic evidence-relevance")
    print("    judgement and natural-language answering (bounded gate + graph stay in code).")

    print("\n" + "=" * 70)
    if _failures:
        print(f"RESULT: FAILED - {len(_failures)} check(s):")
        for f in _failures:
            print(f"   - {f}")
        return 1
    print("RESULT: PASSED - all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
