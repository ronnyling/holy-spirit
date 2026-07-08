from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from .cache import RetrievalCache

from .conflicts import ConflictDetector, ConflictMatch
from .contracts import ClaimDraft, ConflictSummary, EvidenceDraft, GapFlag, SlotSuggestion, TranscriptInput, TranscriptOutcome
from .embeddings import EmbeddingClient
from .evidence import EvidenceLedger
from .extraction import ClaimExtractor, SupportsComplete
from .gaps import GapDetector
from .graph.neo4j_store import KnowledgeGraphStore
from .learning import SlotLearner
from .models import Claim, EpistemicStatus, Provenance, SlotLifecycle
from .policy import get_domain_policy
from .query_processor import ProcessedQuery, QueryProcessor
from .registry import HarbourResult, TranscriptRegistry
from .reranker import RerankerClient
from .resolution import ResolutionMemory
from .store import KnowledgeStore
from .utils import normalize_text


# Maximum claims to pass to LLM for relevance judgment. The LLM decides what
# matters — no hardcoded similarity threshold. This cap bounds token usage.
_MAX_EXPERIENCE_CLAIMS: int = 30
_VERBOSITY_CLAIM_LIMITS = {"warn": 5, "info": 15, "debug": 30}

# Epistemic status weights for ranking. Confirmed claims rank highest because
# they have passed the evidence gate. Applied as a multiplier on retrieval scores.
_EPISTEMIC_WEIGHTS: dict[str, float] = {
    "Confirmed": 1.0,
    "Disputed": 0.8,
    "Unverified": 0.6,
    "Unknown": 0.4,
    "Unverifiable": 0.3,
    "Retracted": 0.1,
}

_EXPLORE_PROMPTS = {
    "warn": (
        "You are a domain expert giving a brief consultation. "
        "Answer in 1-2 sentences. Focus on the single most important finding "
        "and any critical unknowns. If experts disagree, state that briefly. "
        "Be direct and actionable."
    ),
    "info": (
        "You are a domain expert providing a consultation. You have access to "
        "general knowledge and accumulated experience from expert sources. "
        "Produce ONE integrated response. When claims conflict, present both "
        "sides and note which has stronger evidence. Be practical and direct."
    ),
    "debug": (
        "You are a domain expert providing a detailed analysis. Cover: "
        "1) What is generally known. 2) What the system has learned (reference "
        "claim statuses). 3) Where experts disagree. 4) Cross-domain connections. "
        "5) Confidence level. Be thorough but organized."
    ),
}

# Progress callback signature: (pct: float 0.0-1.0, stage: str, detail: str|None)
ProgressCallback = "Callable[[float, str, str | None], None]"


class ProgressTracker:
    """Dynamic progress tracker that maps pipeline stages to a continuous 0-100% bar.

    Stages have relative weights — the bar scales automatically when stages are
    added, removed, or reordered. Sub-progress within a stage (e.g. chunk
    extraction) is supported via `sub()`.
    """

    def __init__(
        self,
        stages: list[tuple[str, int]],
        callback: ProgressCallback | None = None,
    ) -> None:
        self._stages = stages
        self._weights = [w for _, w in stages]
        self._total_weight = sum(self._weights)
        self._prefix_sums = []
        running = 0
        for w in self._weights:
            self._prefix_sums.append(running)
            running += w
        self._cb = callback or (lambda *a: None)
        self._current_stage = 0

    def _pct(self, stage_idx: int, sub: float = 0.0) -> float:
        """Map (stage, sub_progress) to 0.0-1.0."""
        before = self._prefix_sums[stage_idx]
        current = self._weights[stage_idx]
        return (before + current * max(0.0, min(1.0, sub))) / self._total_weight

    def start(self, stage_idx: int, detail: str = "") -> None:
        """Mark the beginning of a stage."""
        self._current_stage = stage_idx
        name = self._stages[stage_idx][0]
        self._cb(self._pct(stage_idx, 0.0), name, detail)

    def sub(self, stage_idx: int, completed: int, total: int, detail: str = "") -> None:
        """Update sub-progress within a stage (e.g. chunk N of M)."""
        sub = completed / total if total > 0 else 0.0
        name = self._stages[stage_idx][0]
        self._cb(self._pct(stage_idx, sub), name, detail)

    def done(self, stage_idx: int, detail: str = "") -> None:
        """Mark a stage as complete (100% within that stage)."""
        self._cb(self._pct(stage_idx, 1.0), self._stages[stage_idx][0], detail)


class KnowledgeEngine:
    def __init__(
        self,
        *,
        registry: TranscriptRegistry | None = None,
        store: KnowledgeStore | KnowledgeGraphStore | None = None,
        embedding_client: EmbeddingClient | None = None,
        extractor: ClaimExtractor | None = None,
        llm_client: SupportsComplete | None = None,
        reranker: RerankerClient | None = None,
        query_processor: QueryProcessor | None = None,
        cache: "RetrievalCache | None" = None,
    ) -> None:
        self.store = store or KnowledgeStore()
        self.embedding_client = embedding_client
        self.extractor = extractor
        self.llm_client = llm_client
        self.reranker = reranker
        self.query_processor = query_processor
        self.cache = cache
        self.slot_learner = SlotLearner()
        self.gap_detector = GapDetector(llm_client=llm_client)
        self.conflict_detector = ConflictDetector()
        self.resolution_memory = ResolutionMemory(self.store)
        self.evidence_ledger = EvidenceLedger()
        # When a registry is wired (production / e2e path), every ingested transcript
        # is harboured, documented, and housekept before any claim processing.
        self.registry = registry

    # Pipeline stages with relative weights — weights determine how much of
    # the 0–100% bar each stage occupies. Add/remove/reorder freely; the
    # ProgressTracker handles scaling automatically.
    PIPELINE_STAGES: list[tuple[str, int]] = [
        ("dedup_check",    1),
        ("classify",       1),
        ("harbour",        1),
        ("extract",        5),  # heaviest — LLM chunk extraction
        ("process_claims", 2),
        ("gap_check",      1),
        ("embed",          2),
        ("conflict_check", 2),
    ]

    def ingest_transcript(
        self,
        transcript: TranscriptInput,
        progress_callback: "Callable[[float, str, str | None], None] | None" = None,
    ) -> TranscriptOutcome:
        _cb = progress_callback or (lambda *a: None)
        _pt = ProgressTracker(self.PIPELINE_STAGES, _cb)

        # Fast dedup pre-check
        _pt.start(0, "Checking for duplicate content…")
        if self.registry is not None:
            _normalized = self.registry.chunker.normalize(transcript.transcript_text)
            _digest = TranscriptRegistry.content_hash(_normalized)
            _existing = self.registry.find_by_hash(_digest)
            if _existing is not None and _existing.processing_status == "complete":
                _pt.done(0, "Duplicate — skipping")
                entity = self.store.upsert_entity(canonical_name=_existing.entity_name or "unknown")
                return TranscriptOutcome(
                    entity_id=entity.id or "",
                    transcript_id=_existing.transcript_id,
                    transcript_created=False,
                    chunk_count=_existing.chunk_count,
                    canonical_claim_ids=[
                        claim.id or ""
                        for claim in self.store.list_canonical_claims(entity.id or "")
                    ],
                    notes=[
                        f"duplicate content (hash match) — already ingested as "
                        f"{_existing.transcript_id}; skipped classification + extraction."
                ],
            )
        _pt.done(0)

        # Auto-classify domain and entity_name when the caller left them blank.
        _pt.start(1, "Classifying domain and entity…")
        if not transcript.domain or not transcript.entity_name:
            effective_domain = transcript.domain
            effective_entity_name = transcript.entity_name
            if self.llm_client is not None:
                try:
                    from .classification import DomainClassifier
                    classifier = DomainClassifier(self.llm_client)
                    if not effective_domain:
                        effective_domain = (
                            classifier.classify(transcript_text=transcript.transcript_text)
                            or classifier.classify_open(transcript_text=transcript.transcript_text)
                            or "unknown"
                        )
                    if not effective_entity_name:
                        effective_entity_name = (
                            classifier.classify_entity_name(transcript_text=transcript.transcript_text)
                            or "unknown"
                        )
                except Exception:
                    effective_domain = effective_domain or "unknown"
                    effective_entity_name = effective_entity_name or "unknown"
            transcript = transcript.model_copy(update={
                "domain": effective_domain or "unknown",
                "entity_name": effective_entity_name or "unknown",
            })
        _pt.done(1, f"Domain: {transcript.domain}, Entity: {transcript.entity_name}")

        # Harbour
        _pt.start(2, "Harbouring transcript…")
        harbour: HarbourResult | None = None
        if self.registry is not None:
            harbour = self.registry.harbour(
                text=transcript.transcript_text,
                domain=transcript.domain,
                entity_name=transcript.entity_name,
                source_kind=transcript.source_kind,
                source_id=transcript.source_id,
                source_ref=transcript.source_ref,
            )
        _pt.done(2)

        entity = self.store.upsert_entity(canonical_name=transcript.entity_name or "unknown")

        # If already fully processed, check if entity actually has claims.
        # If not, the previous run was incomplete — re-process.
        if harbour is not None and harbour.already_complete:
            existing_claims = self.store.list_claims_for_entity(entity.id or "")
            if not existing_claims:
                harbour.already_complete = False
                # Fall through to claim processing
            else:
                _pt.done(3, "Already complete — skipping")
                return TranscriptOutcome(
                    entity_id=entity.id or "",
                    transcript_id=harbour.record.transcript_id,
                    transcript_created=False,
                    chunk_count=harbour.record.chunk_count,
                    canonical_claim_ids=[
                        claim.id or ""
                        for claim in self.store.list_canonical_claims(entity.id or "")
                    ],
                    notes=[
                        f"duplicate content already ingested as {harbour.record.transcript_id}; "
                        "skipped re-processing"
                    ],
                )

        # LLM auto-extraction
        _pt.start(3, "Extracting claims…")
        _t_extract = time.monotonic()
        claim_drafts = transcript.claim_drafts
        extraction_notes: list[str] = []
        if not claim_drafts and self.extractor is not None:
            _chunks = self.extractor._chunker.chunk(transcript.transcript_text)
            _chunk_count = len(_chunks)
            _pt.sub(3, 0, _chunk_count, f"Chunked into {_chunk_count} segments")
            claim_drafts = self.extractor.extract(
                domain=transcript.domain,
                entity_name=transcript.entity_name,
                transcript_text=transcript.transcript_text,
            )
            extraction_notes.append(
                f"LLM extracted {len(claim_drafts)} claim(s) (Unverified until evidence)"
            )
        _elapsed_extract = time.monotonic() - _t_extract
        _pt.done(3, f"Extracted {len(claim_drafts)} claim(s) in {_elapsed_extract:.1f}s")

        claim_ids: list[str] = []
        new_claims: list[Claim] = []
        confirmed_claim_ids: list[str] = []
        unverified_claim_ids: list[str] = []
        disputed_claim_ids: list[str] = []
        slot_suggestions: list[SlotSuggestion] = []
        gap_flags: list[GapFlag] = []
        conflict_summaries: list[ConflictSummary] = []
        open_case_ids: list[str] = []
        notes: list[str] = list(extraction_notes)
        observed_slot_names: set[str] = set()

        # Process claims + observe slots
        _pt.start(4, "Processing claims and observing slots…")
        for i, draft in enumerate(claim_drafts):
            _pt.sub(4, i + 1, len(claim_drafts), f"Claim {i+1}/{len(claim_drafts)}")
            claim = self._build_claim(transcript, entity.id or "", draft)
            claim = self.store.add_claim(claim)
            claim_ids.append(claim.id or "")
            new_claims.append(claim)

            if draft.evidence:
                try:
                    self.evidence_ledger.record_for_claim(claim, draft.evidence, self.store)
                except ValueError:
                    pass

            for slot_name in draft.observed_slots:
                observed_slot_names.add(slot_name)
                observation = self.slot_learner.observe(self.store, entity.id or "", slot_name)
                if observation.suggested_lifecycle is not None:
                    slot_suggestions.append(
                        SlotSuggestion(
                            slot_name=observation.slot.name,
                            current_lifecycle=observation.slot.lifecycle.value,
                            suggested_lifecycle=observation.suggested_lifecycle.value,
                            observed_count=observation.slot.observed_count,
                            reason=observation.reason or "slot crossed a learned threshold",
                        )
                    )
                    try:
                        self.store.queue_slot_promotion(
                            entity_id=entity.id or "",
                            entity_name=entity.canonical_name,
                            slot_name=observation.slot.name,
                            current_lifecycle=observation.slot.lifecycle.value,
                            suggested_lifecycle=observation.suggested_lifecycle.value,
                            observed_count=observation.slot.observed_count,
                            reason=observation.reason or "slot crossed a learned threshold",
                        )
                    except Exception:
                        pass
        _pt.done(4)

        # Gap detection
        _pt.start(5, "Running gap detection…")
        gap_flags.extend(
            self.gap_detector.structural_gaps(
                entity.id or "",
                observed_slot_names,
                self.store.get_expected_slots(entity.id or ""),
            )
        )
        # Logical gap detection
        if self.llm_client is not None:
            logical_gap_flags = self.gap_detector.logical_gaps(
                claims=new_claims,
                evidence=[],
            )
            gap_flags.extend(logical_gap_flags)
        _pt.done(5)

        # Embedding
        _pt.start(6, "Embedding claims…")
        _t_embed = time.monotonic()
        if self.embedding_client is not None and new_claims:
            _pt.sub(6, 0, len(new_claims), f"Warming model, embedding {len(new_claims)} claims…")
            with self.embedding_client.warm():
                vectors = self.embedding_client.embed_texts(
                    [claim.statement for claim in new_claims]
                )
            for claim, vector in zip(new_claims, vectors):
                claim.embedding = vector
                self.store.set_claim_embedding(claim_id=claim.id or "", embedding=vector)
            notes.append(
                f"embedded {len(new_claims)} claim(s) at "
                f"{len(vectors[0]) if vectors else 0}-dim"
            )
        _elapsed_embed = time.monotonic() - _t_embed
        _pt.done(6, f"Embedded in {_elapsed_embed:.1f}s")

        # Conflict detection — check against Confirmed AND Disputed claims.
        _pt.start(7, "Checking conflicts…")
        canonical_claims = self.store.list_active_claims(entity.id or "")
        for i, (claim, draft) in enumerate(zip(new_claims, claim_drafts)):
            _pt.sub(7, i + 1, len(new_claims), f"Claim {i+1}/{len(new_claims)}")
            matches = self.conflict_detector.detect(canonical_claims, claim)
            if matches:
                # Auto-resolve: if the incoming claim has no evidence and the
                # existing claim is Confirmed, auto-close in favor of Confirmed.
                # The incoming claim stays Unverified and gets a heckle prompt.
                has_evidence = bool(draft.evidence)
                existing_is_confirmed = any(
                    m.existing_claim_ids and
                    self.store.load_claim(m.existing_claim_ids[0]).epistemic_status == EpistemicStatus.CONFIRMED
                    for m in matches if m.existing_claim_ids
                )
                if not has_evidence and existing_is_confirmed:
                    # Low-stakes auto-resolve: incoming claim has no evidence,
                    # existing claim is confirmed. Auto-close the case.
                    claim.epistemic_status = EpistemicStatus.UNVERIFIED
                    self.store.set_claim_status(claim_id=claim.id or "", status=str(claim.epistemic_status))
                    unverified_claim_ids.append(claim.id or "")
                    for match in matches:
                        # Auto-resolve the case
                        case = self.resolution_memory.open_case(match)
                        self.resolution_memory.resolve_case(
                            case.id or "",
                            decision=f"Auto-resolved: existing Confirmed claim prevails. "
                                     f"Incoming claim lacks evidence.",
                            rationale="Incoming claim has no evidence; existing claim is Confirmed.",
                        )
                        notes.append(
                            f"Auto-resolved conflict: {match.message} "
                            f"(incoming claim lacks evidence, existing claim Confirmed)"
                        )
                else:
                    # Standard conflict: both sides have evidence or existing is Disputed
                    claim.epistemic_status = EpistemicStatus.DISPUTED
                    self.store.set_claim_status(claim_id=claim.id or "", status=str(claim.epistemic_status))
                    disputed_claim_ids.append(claim.id or "")
                    for match in matches:
                        case = self.resolution_memory.open_case(match)
                        open_case_ids.append(case.id or "")
                        conflict_summaries.append(
                            ConflictSummary(
                                conflict_signature=match.conflict_signature,
                                incoming_claim_id=match.incoming_claim_id,
                                existing_claim_ids=match.existing_claim_ids,
                                case_id=case.id,
                                message=match.message,
                            )
                        )
                continue

            can_confirm = self._can_confirm_claim(transcript.domain, transcript.source_kind, claim, draft, gap_flags)
            if can_confirm:
                claim.epistemic_status = EpistemicStatus.CONFIRMED
                self.store.set_claim_status(claim_id=claim.id or "", status=str(claim.epistemic_status))
                confirmed_claim_ids.append(claim.id or "")
            else:
                claim.epistemic_status = EpistemicStatus.UNVERIFIED
                self.store.set_claim_status(claim_id=claim.id or "", status=str(claim.epistemic_status))
                unverified_claim_ids.append(claim.id or "")
        _pt.done(7, f"Complete — {len(claim_ids)} claims, {len(gap_flags)} gaps, {len(conflict_summaries)} conflicts")

        # Proactive conflict evidence gathering.
        # When conflicts are detected, search for evidence on both sides and
        # generate a context-aware prompt for the user.
        conflict_prompt: str | None = None
        gathered_evidence: list[dict] | None = None
        if conflict_summaries and self.llm_client is not None:
            conflict_prompt, gathered_evidence = self._gather_conflict_evidence(
                entity_name=transcript.entity_name or "unknown",
                domain=transcript.domain or "unknown",
                conflict_summaries=conflict_summaries,
                existing_claims=canonical_claims,
                incoming_claims=new_claims,
            )

        # Mark transcript as fully processed so retries skip it
        if harbour is not None:
            try:
                self.registry.mark_complete(harbour.record.transcript_id)
            except Exception:
                pass

        return TranscriptOutcome(
            entity_id=entity.id or "",
            transcript_id=harbour.record.transcript_id if harbour else None,
            transcript_created=harbour.created if harbour else None,
            chunk_count=harbour.record.chunk_count if harbour else None,
            claim_ids=claim_ids,
            confirmed_claim_ids=confirmed_claim_ids,
            unverified_claim_ids=unverified_claim_ids,
            disputed_claim_ids=disputed_claim_ids,
            slot_suggestions=slot_suggestions,
            gap_flags=gap_flags,
            conflict_summaries=conflict_summaries,
            open_case_ids=open_case_ids,
            canonical_claim_ids=[claim.id or "" for claim in self.store.list_canonical_claims(entity.id or "")],
            notes=notes,
            conflict_prompt=conflict_prompt,
            gathered_evidence=gathered_evidence,
        )

    def confirm_slot(self, entity_name: str, slot_name: str, confirmed_by: str, target: SlotLifecycle) -> dict[str, Any]:
        entity = self.store.upsert_entity(canonical_name=entity_name)
        slot = self.store.confirm_slot(entity.id or "", slot_name, target=target, confirmed_by=confirmed_by)
        return {
            "slot_id": slot.id,
            "entity_id": entity.id,
            "name": slot.name,
            "lifecycle": slot.lifecycle.value,
            "observed_count": slot.observed_count,
            "candidate_count": slot.candidate_count,
            "expected_count": slot.expected_count,
        }

    def accept_on_authority(
        self,
        claim_ids: list[str],
        accepted_by: str,
        rationale: str = "Accepted on domain authority — well-established fact.",
    ) -> list[dict[str, Any]]:
        """Batch-promote Unverified claims to Confirmed without full evidence chains.

        For well-known facts that don't require rigorous evidence gating.
        Each claim is confirmed with a single authority-source evidence record.
        The user takes responsibility for the accuracy.
        """
        if not accepted_by.strip():
            raise ValueError("accepted_by is required — the system requires accountability")

        results = []
        for claim_id in claim_ids:
            claim = self.store.load_claim(claim_id)
            if claim.epistemic_status != EpistemicStatus.UNVERIFIED:
                results.append({
                    "claim_id": claim_id,
                    "status": "skipped",
                    "reason": f"claim is {claim.epistemic_status.value}, not Unverified",
                })
                continue

            # Create authority evidence
            authority_evidence = EvidenceDraft(
                source_kind="user",
                source_id=f"authority:{accepted_by}",
                credibility=0.9,
                notes=rationale,
            )
            domain = self._claim_domain(claim)
            evaluation = self.evidence_ledger.evaluate(domain, [authority_evidence], self.store, claim)

            if evaluation.can_confirm:
                self.evidence_ledger.record_for_claim(claim, [authority_evidence], self.store)
                claim.epistemic_status = EpistemicStatus.CONFIRMED
                self.store.set_claim_status(claim_id=claim.id or "", status=str(claim.epistemic_status))
                results.append({
                    "claim_id": claim_id,
                    "status": "confirmed",
                    "accepted_by": accepted_by,
                })
            else:
                results.append({
                    "claim_id": claim_id,
                    "status": "rejected",
                    "reasons": evaluation.reasons,
                })

        return results

    def promote_claim(self, claim_id: str, evidence: list[EvidenceDraft]) -> Claim:
        claim = self.store.load_claim(claim_id)
        evaluation = self.evidence_ledger.evaluate(self._claim_domain(claim), evidence, self.store, claim)
        if not evaluation.can_confirm:
            raise ValueError("claim does not meet the evidence gate: " + "; ".join(evaluation.reasons))
        self.evidence_ledger.record_for_claim(claim, evidence, self.store)
        claim.epistemic_status = EpistemicStatus.CONFIRMED
        self.store.set_claim_status(claim_id=claim.id or "", status=str(claim.epistemic_status))
        return claim

    def resolve_case(self, case_id: str, decision: str, rationale: str) -> dict[str, Any]:
        case = self.resolution_memory.resolve_case(case_id, decision=decision, rationale=rationale)
        return {
            "case_id": case.id,
            "decision": case.decision,
            "rationale": case.rationale,
            "version": case.version,
            "is_open": case.is_open,
            "reopened_from_case_id": case.reopened_from_case_id,
        }

    def reassess_claim(self, claim_id: str, new_evidence: list[EvidenceDraft]) -> dict[str, Any]:
        """Re-evaluate a Confirmed claim when new contradictory evidence arrives.

        Does NOT auto-demote. Instead:
        1. Opens a resolution case linking the old claim to the new evidence
        2. Flags the claim for reassessment
        3. Returns the reassessment status for the user to review

        The system learns by accumulating evidence, not by flipping statuses.
        """
        claim = self.store.load_claim(claim_id)
        domain = self._claim_domain(claim)

        # Evaluate new evidence against domain gate
        evaluation = self.evidence_ledger.evaluate(domain, new_evidence, self.store, claim)

        # Build a conflict match for the resolution case
        from .conflicts import ConflictMatch
        match = ConflictMatch(
            conflict_signature=f"{claim.entity_id}:{claim.slot_name or 'general'}:"
                              f"{normalize_text(claim.statement)}:reassessment",
            incoming_claim_id=claim_id,
            existing_claim_ids=[claim_id],
            message=f"Reassessment requested for claim: {claim.statement[:80]}...",
        )
        case = self.resolution_memory.open_case(match)

        # Record the new evidence
        self.evidence_ledger.record_for_claim(claim, new_evidence, self.store)

        return {
            "claim_id": claim_id,
            "current_status": str(claim.epistemic_status),
            "new_evidence_score": evaluation.score,
            "new_evidence_sources": evaluation.source_count,
            "gate_passed": evaluation.can_confirm,
            "case_id": case.id,
            "reasons": evaluation.reasons,
            "recommendation": (
                "New evidence is compelling. Consider updating the resolution case."
                if evaluation.can_confirm
                else "New evidence does not meet the evidence gate. "
                     "The claim's current status is preserved."
            ),
        }

    def list_open_cases(self) -> list[dict[str, Any]]:
        """Return open resolution cases enriched with conflicting claim statements."""
        try:
            cases = self.store.list_open_resolution_cases()
        except Exception:
            return []
        result = []
        for case in cases:
            claims_detail = []
            for cid in case.conflicting_claim_ids:
                try:
                    row = self.get_claim(cid)
                    stmt = row.get("claim", {}).get("statement", cid) if not row.get("error") else cid
                    status = row.get("claim", {}).get("epistemic_status", "?") if not row.get("error") else "?"
                except Exception:
                    stmt, status = cid, "?"
                claims_detail.append({"id": cid, "statement": stmt, "epistemic_status": status})
            result.append({
                "case_id": case.id,
                "conflict_signature": case.conflict_signature,
                "research_notes": case.research_notes,
                "version": case.version,
                "claims": claims_detail,
            })
        return result

    def state_snapshot(self) -> dict[str, Any]:
        return self.store.snapshot()

    def health_check(self) -> dict[str, Any]:
        """Check health of all system components. Returns status of each backend."""
        checks = {}

        # Neo4j
        try:
            if hasattr(self.store, 'verify'):
                self.store.verify()
            checks["neo4j"] = {"status": "healthy"}
        except Exception as e:
            checks["neo4j"] = {"status": "unhealthy", "error": str(e)}

        # Embeddings
        if self.embedding_client is not None:
            try:
                # Quick embed test
                self.embedding_client.embed_sync("health check")
                checks["embeddings"] = {"status": "healthy"}
            except Exception as e:
                checks["embeddings"] = {"status": "unhealthy", "error": str(e)}
        else:
            checks["embeddings"] = {"status": "not_configured"}

        # LLM — use higher max_tokens because MiMo reasoning model consumes
        # budget on reasoning_content before producing the answer.
        if self.llm_client is not None:
            try:
                self.llm_client.complete_sync(
                    system="Reply with one word: healthy",
                    user="health check",
                    max_tokens=200,
                )
                checks["llm"] = {"status": "healthy"}
            except Exception as e:
                checks["llm"] = {"status": "unhealthy", "error": str(e)}
        else:
            checks["llm"] = {"status": "not_configured"}

        # Reranker
        if self.reranker is not None:
            checks["reranker"] = {"status": "configured", "model": self.reranker.model}
        else:
            checks["reranker"] = {"status": "not_configured"}

        # Queue depths
        try:
            pending = self.list_pending_promotions()
            checks["slot_promotion_queue"] = {"depth": len(pending)}
        except Exception:
            checks["slot_promotion_queue"] = {"depth": "unknown"}

        try:
            open_cases = self.list_open_cases()
            checks["open_conflict_cases"] = {"depth": len(open_cases)}
        except Exception:
            checks["open_conflict_cases"] = {"depth": "unknown"}

        # Overall status
        all_healthy = all(
            c.get("status") in ("healthy", "not_configured", "configured")
            for c in checks.values()
            if isinstance(c, dict) and "status" in c
        )
        checks["overall"] = "healthy" if all_healthy else "degraded"

        return checks

    def list_pending_promotions(self) -> list[dict]:
        """Return all pending slot promotion candidates from the queue."""
        try:
            return self.store.list_pending_slot_promotions()
        except Exception:
            return []

    def housekeeping(self) -> dict[str, Any]:
        """Periodic maintenance: close stale cases, report queue depths, validate indexes.

        Run this daily or weekly via a cron job or scheduled task.
        Returns a summary of actions taken.
        """
        actions = []

        # 1. Close stale resolution cases (no activity for 30+ days)
        try:
            open_cases = self.list_open_cases()
            stale_threshold_days = 30
            from datetime import datetime, timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=stale_threshold_days)
            closed_count = 0
            for case_info in open_cases:
                case_id = case_info.get("case_id")
                if not case_id:
                    continue
                try:
                    case = self.store.get_resolution_case(case_id)
                    # Check if case is old (resolution_case doesn't have created_at,
                    # but we can check version — version 1 means never resolved)
                    if case.version == 1:
                        self.resolve_case(
                            case_id,
                            decision="Auto-closed: no resolution provided within 30 days.",
                            rationale="Stale case auto-closed by housekeeping.",
                        )
                        closed_count += 1
                except Exception:
                    pass
            if closed_count:
                actions.append(f"Closed {closed_count} stale resolution cases")
        except Exception:
            pass

        # 2. Report queue depths
        try:
            pending = self.list_pending_promotions()
            if pending:
                actions.append(f"Slot promotion queue: {len(pending)} pending items")
        except Exception:
            pass

        # 3. State snapshot
        try:
            snap = self.state_snapshot()
            actions.append(
                f"State: {snap.get('entities', 0)} entities, "
                f"{snap.get('claims', 0)} claims, "
                f"{snap.get('confirmed_claims', 0)} confirmed, "
                f"{snap.get('open_cases', 0)} open cases"
            )
        except Exception:
            pass

        return {"actions": actions, "timestamp": datetime.now(timezone.utc).isoformat()}

    def explore_experience(self, query: str, domain: str | None = None, verbosity: str = "warn") -> dict[str, Any]:
        """Synthesize world knowledge discerned through accumulated system experience.

        Step 1: Ask the LLM what the world generally knows (parametric knowledge).
        Step 2: Retrieve the system's experience claims (all epistemic statuses).
        Step 3: Ask the LLM to discern — where experience agrees, adds, or corrects
                the world view. The synthesis is strictly grounded in the provided
                claims; the LLM must not invent experience.
        """
        if self.llm_client is None:
            return {"error": "explore_experience requires an LLM client (KE_MIMO_API_KEY)"}

        # Step 1: world knowledge + query expansion in parallel (saves ~3s).
        with ThreadPoolExecutor(max_workers=2) as pool:
            world_future = pool.submit(
                self.llm_client.complete_sync,
                system="You are a knowledgeable expert. Answer based on general world knowledge. Be concise.",
                user=query,
            )
            expansion_future = pool.submit(
                self.query_processor.process, query, domain
            ) if self.query_processor else None

        world_knowledge = world_future.result()
        processed = expansion_future.result() if expansion_future else None
        search_queries = processed.all_queries if processed else [query]

        # Step 2: system experience via hybrid search (all epistemic statuses).
        # Search with all query variants and merge results.
        # Batch embeddings for efficiency — single API call instead of N calls.
        experience_claims: list[dict] = []
        if self._require_graph_store("explore_experience") and self._require_embeddings("explore_experience"):
            all_hits: list[dict] = []
            embeddings = self.embedding_client.embed_texts(search_queries)  # type: ignore[union-attr]
            for sq, embedding in zip(search_queries, embeddings):
                hits = self.store.hybrid_search_claims(
                    embedding=embedding,
                    query_text=sq,
                    domain=domain,
                    k=15,
                )
                all_hits.extend(hits)

            # Deduplicate by claim_id, keeping the best score.
            seen: dict[str, dict] = {}
            for h in all_hits:
                cid = h.get("claim_id") or h.get("id")
                if not cid:
                    continue
                score = h.get("rrf_score") or h.get("similarity") or h.get("score") or 0.0
                if cid not in seen or score > (seen[cid].get("rrf_score") or seen[cid].get("similarity") or seen[cid].get("score") or 0.0):
                    seen[cid] = h
            raw_hits = list(seen.values())

            # No score-based filtering — the LLM judges relevance.
            # Sort by epistemic weight so Confirmed claims are presented first.
            raw_hits.sort(
                key=lambda h: _EPISTEMIC_WEIGHTS.get(
                    h.get("epistemic_status") or h.get("status", "Unknown"), 0.5
                ),
                reverse=True,
            )
            claim_limit = _VERBOSITY_CLAIM_LIMITS.get(verbosity, _MAX_EXPERIENCE_CLAIMS)
            experience_claims = raw_hits[:claim_limit]
        # Step 2b: graph traversal — expand from matched claims to related context.
        graph_context: list[dict] = []
        if experience_claims:
            matched_ids = [h["id"] for h in experience_claims if h.get("id")]
            if matched_ids:
                graph_context = self.store.get_graph_context(
                    matched_claim_ids=matched_ids,
                    limit=10,
                )
        # Step 2c: cross-domain pattern discovery for anti-gaslighting.
        # Only search when a specific domain is provided — searching all domains
        # is expensive and produces noisy results.
        cross_domain_patterns: list[dict] = []
        if domain and hasattr(self.store, 'find_cross_domain_patterns'):
            try:
                cross_domain_patterns = self.store.find_cross_domain_patterns(
                    domains=[domain],
                    min_similarity=0.5,
                    limit=5,
                )
            except Exception:
                pass
        confirmed_count = sum(1 for h in experience_claims if h.get("epistemic_status") == "Confirmed" or h.get("status") == "Confirmed")
        unverified_count = sum(1 for h in experience_claims if h.get("epistemic_status") == "Unverified" or h.get("status") == "Unverified")
        disputed_count = sum(1 for h in experience_claims if h.get("epistemic_status") == "Disputed" or h.get("status") == "Disputed")

        # Optional reranking for precision improvement.
        # Fails loudly if configured but broken — no silent degradation.
        if self.reranker is not None and len(experience_claims) > 5:
            experience_claims = self.reranker.rerank(
                query, experience_claims, top_n=15,
            )

        if not experience_claims and not graph_context and not cross_domain_patterns:
            return {
                "query": query,
                "domain": domain,
                "world_knowledge": world_knowledge,
                "experience_claims": [],
                "synthesis": world_knowledge,
                "confirmed_count": 0,
                "unverified_count": 0,
                "disputed_count": 0,
                "experience_available": False,
                "note": "No system experience found yet \u2014 returning world knowledge only.",
            }

        direct_block = "\n".join(
            f"[{h.get('epistemic_status') or h.get('status', 'Unknown')}] {h.get('statement', '')}"
            for h in experience_claims
        )
        graph_block = ""
        if graph_context:
            graph_block = "\n\nGraph-connected context (same entities / shared slots):\n" + "\n".join(
                f"[{h.get('epistemic_status', 'Unknown')}] ({h.get('context_type', 'graph')}) "
                f"{h.get('statement', '')}"
                for h in graph_context
            )
        cross_domain_block = ""
        if cross_domain_patterns:
            cross_domain_block = "\n\nCross-domain patterns (similar concepts across different fields):\n" + "\n".join(
                f"  - [{p.get('claim_a', {}).get('domains', ['?'])[0]}] "
                f"{p.get('claim_a', {}).get('statement', '')[:80]}...\n"
                f"    <-> [{p.get('claim_b', {}).get('domains', ['?'])[0]}] "
                f"{p.get('claim_b', {}).get('statement', '')[:80]}...\n"
                f"    (similarity: {p.get('similarity', 0):.2f})"
                for p in cross_domain_patterns
            )

        synthesis = self.llm_client.complete_sync(
            system=_EXPLORE_PROMPTS.get(verbosity, _EXPLORE_PROMPTS["info"]),
            user=(
                f"Question: {query}\n\n"
                f"World knowledge:\n{world_knowledge}\n\n"
                f"Direct experience ({len(experience_claims)} claims):\n{direct_block}"
                f"{graph_block}"
                f"{cross_domain_block}"
            ),
        )

        all_context = [
            {**h, "context_type": "direct", "epistemic_status": h.get("epistemic_status") or h.get("status", "Unknown")}
            for h in experience_claims
        ] + graph_context

        return {
            "query": query,
            "domain": domain,
            "world_knowledge": world_knowledge,
            "experience_claims": all_context,
            "synthesis": synthesis,
            "confirmed_count": confirmed_count,
            "unverified_count": unverified_count,
            "disputed_count": disputed_count,
            "cross_domain_patterns": cross_domain_patterns,
            "experience_available": True,
        }

    # -- query tools (hybrid search via Neo4j) ---------------------------------

    def search_claims(
        self,
        query: str,
        domain: str | None = None,
        epistemic_status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Hybrid search over claims (vector + full-text, RRF-merged).

        Combines semantic similarity with lexical matching. Requires both
        Neo4j graph store and embedding provider — fails loudly if either
        is missing.
        """
        if not self._require_graph_store("search_claims"):
            return {"error": "search_claims requires Neo4j graph store", "claims": []}
        if not self._require_embeddings("search_claims"):
            return {"error": "search_claims requires embedding provider", "claims": []}

        embedding = self.embedding_client.embed_sync(query)  # type: ignore[union-attr]
        results = self.store.hybrid_search_claims(
            embedding=embedding,
            query_text=query,
            domain=domain,
            epistemic_status=epistemic_status,
            k=limit,
        )

        # Apply epistemic weighting: Confirmed claims rank first.
        for claim in results:
            status = claim.get("epistemic_status") or claim.get("status", "Unknown")
            weight = _EPISTEMIC_WEIGHTS.get(status, 0.5)
            base_score = claim.get("rrf_score") or claim.get("similarity") or claim.get("score") or 0.0
            claim["weighted_score"] = round(base_score * weight, 6)
        results.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)

        return {"query": query, "domain": domain, "count": len(results), "claims": results}

    def search_entities(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Vector search: find claims like query, return their entities."""
        if not self._require_graph_store("search_entities"):
            return {"error": "search_entities requires Neo4j graph store", "entities": []}
        if not self._require_embeddings("search_entities"):
            return {"error": "search_entities requires embedding provider", "entities": []}

        embedding = self.embedding_client.embed_sync(query)
        results = self.store.vector_search_entities(
            embedding=embedding,
            domain=domain,
            k=limit,
        )
        return {"query": query, "domain": domain, "count": len(results), "entities": results}

    def get_entity(
        self,
        entity_id: str | None = None,
        entity_name: str | None = None,
    ) -> dict[str, Any]:
        """Full entity details with claims, slots, evidence."""
        if not self._require_graph_store("get_entity"):
            return {"error": "get_entity requires Neo4j graph store"}

        if entity_id:
            result = self.store.get_entity_with_details(entity_id)
        elif entity_name:
            result = self.store.get_entity_by_name(entity_name)
        else:
            return {"error": "either entity_id or entity_name is required"}

        if result is None:
            return {"error": f"entity not found: {entity_id or entity_name}"}
        return result

    def get_claim(self, claim_id: str) -> dict[str, Any]:
        """Claim details with provenance and evidence chain."""
        if not self._require_graph_store("get_claim"):
            return {"error": "get_claim requires Neo4j graph store"}

        result = self.store.get_claim_detail(claim_id)
        if result is None:
            return {"error": f"claim not found: {claim_id}"}
        return result

    def search_by_domain(self, domain: str, limit: int = 50) -> dict[str, Any]:
        """All confirmed knowledge in a domain."""
        if not self._require_graph_store("search_by_domain"):
            return {"error": "search_by_domain requires Neo4j graph store", "domain": domain}

        return self.store.search_by_domain(domain=domain, limit=limit)

    def _require_graph_store(self, tool_name: str) -> bool:
        """Check that the store is a Neo4j graph store."""
        return isinstance(self.store, KnowledgeGraphStore)

    def _require_embeddings(self, tool_name: str) -> bool:
        """Check that an embedding client is configured."""
        return self.embedding_client is not None

    def _build_claim(self, transcript: TranscriptInput, entity_id: str, draft: ClaimDraft) -> Claim:
        return Claim(
            id=uuid4().hex,
            entity_id=entity_id,
            statement=draft.statement,
            slot_name=draft.slot_name,
            epistemic_status=EpistemicStatus.UNVERIFIED,
            provenance=[
                Provenance(
                    source_kind=transcript.source_kind,
                    source_id=transcript.source_id,
                    source_ref=transcript.source_ref,
                    notes=transcript.notes,
                )
            ],
            tags=[transcript.domain],
        )

    def _claims_for_entity(self, entity_id: str) -> list[Claim]:
        return self.store.list_claims_for_entity(entity_id)

    def _can_confirm_claim(
        self,
        domain: str,
        source_kind: str,
        claim: Claim,
        draft: ClaimDraft,
        gap_flags: list[GapFlag],
    ) -> bool:
        if source_kind == "user":
            return False
        if any(flag.entity_id == claim.entity_id for flag in gap_flags):
            return False
        if not draft.evidence:
            return False
        evaluation = self.evidence_ledger.evaluate(domain, draft.evidence, self.store, claim)
        return evaluation.can_confirm

    def _recorded_evidence_notes(self, claim: Claim, evidence: list[Any]) -> list[str]:
        if not evidence:
            return []
        return [f"recorded {len(evidence)} evidence item(s) for claim {claim.id}"]

    def _claim_domain(self, claim: Claim) -> str:
        if claim.tags:
            return claim.tags[0]
        return "default"

    def _gather_conflict_evidence(
        self,
        entity_name: str,
        domain: str,
        conflict_summaries: list[ConflictSummary],
        existing_claims: list[Claim],
        incoming_claims: list[Claim],
    ) -> tuple[str | None, list[dict] | None]:
        """When conflicts are detected, generate a context-aware prompt and
        gather evidence from the graph. The system heckles the user until
        they provide compelling evidence or acknowledge uncertainty."""
        if not self.llm_client:
            return None, None

        # Build context from the conflict
        conflict_lines = []
        for cs in conflict_summaries:
            existing_stmts = []
            for eid in cs.existing_claim_ids:
                for c in existing_claims:
                    if c.id == eid:
                        existing_stmts.append(
                            f"  [{c.epistemic_status}] {c.statement}"
                        )
            incoming_stmt = ""
            for c in incoming_claims:
                if c.id == cs.incoming_claim_id:
                    incoming_stmt = c.statement
                    break
            conflict_lines.append(
                f"Slot '{cs.conflict_signature.split(':')[1] if ':' in cs.conflict_signature else 'unknown'}':\n"
                + "\n".join(existing_stmts) + f"\n  [Incoming] {incoming_stmt}"
            )

        # Search graph for cross-domain patterns on the disputed topic
        cross_patterns = []
        try:
            if hasattr(self.store, 'find_cross_domain_patterns'):
                cross_patterns = self.store.find_cross_domain_patterns(
                    domains=[domain] if domain else None,
                    min_similarity=0.5,
                    limit=5,
                )
        except Exception:
            pass

        # Generate conflict prompt via LLM
        system = (
            "You are a research assistant analyzing a knowledge conflict. "
            "Two experts disagree on the same topic. Your job is to:\n"
            "1. Present both positions fairly\n"
            "2. Note which has stronger evidence\n"
            "3. Identify what evidence would resolve the conflict\n"
            "4. Ask the user to provide compelling evidence or acknowledge uncertainty\n\n"
            "Be direct, specific, and actionable. This is a heckle — push for resolution."
        )
        user = (
            f"Entity: {entity_name}\n"
            f"Domain: {domain}\n\n"
            f"Conflicts detected:\n" + "\n\n".join(conflict_lines)
        )
        if cross_patterns:
            user += "\n\nCross-domain patterns found:\n"
            for p in cross_patterns[:3]:
                user += f"  - {p.get('claim_a', {}).get('statement', '')[:80]} <-> {p.get('claim_b', {}).get('statement', '')[:80]}\n"

        prompt = self.llm_client.complete_sync(system=system, user=user)

        # Collect evidence from the graph (existing confirmed claims on the topic)
        gathered = []
        for c in existing_claims:
            if c.epistemic_status == EpistemicStatus.CONFIRMED:
                gathered.append({
                    "statement": c.statement,
                    "status": str(c.epistemic_status),
                    "slot": c.slot_name,
                    "source": "internal_knowledge",
                })

        return prompt, gathered if gathered else None
