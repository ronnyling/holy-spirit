"""Comprehensive Stress Test — Logical Gap Detection System.

Tests performance, functionality, and efficiency of the LogicalGapDetector
under various scenarios: large datasets, edge cases, concurrent operations,
and real-world patterns.

Run: pytest tests/test_stress_logical_gaps.py -v
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from knowledge_engine.contracts import GapFlag, GapKind
from knowledge_engine.logical_gaps import LogicalGapDetector
from knowledge_engine.models import Claim, Evidence, EpistemicStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def detector():
    """Default detector without LLM."""
    return LogicalGapDetector()


@pytest.fixture
def llm_detector():
    """Detector with mocked LLM client."""
    mock_llm = Mock()
    mock_llm.complete_sync.return_value = '{"assumptions": ["test assumption"]}'
    return LogicalGapDetector(llm_client=mock_llm)


# ---------------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------------
class TestPerformance:
    """Measure detection speed under various loads."""

    def test_single_claim_throughput(self, detector):
        """Baseline: single claim with evidence, 100 iterations."""
        claim = Claim(id="c1", entity_id="e1", statement="Test claim",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [Evidence(id="e1", claim_id="c1", source_kind="external_doc",
                           source_id="doc1", credibility=0.7)]

        start = time.monotonic()
        for _ in range(100):
            detector.detect([claim], evidence)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"100 iterations took {elapsed:.2f}s (>1s threshold)"

    def test_100_claims_circular_chain(self, detector):
        """100 claims connected in a chain."""
        claims = []
        evidence = []
        for i in range(100):
            claims.append(Claim(id=f"c{i}", entity_id="e1",
                              statement=f"Claim {i}",
                              epistemic_status=EpistemicStatus.CONFIRMED))
            if i > 0:
                evidence.append(Evidence(id=f"e{i}", claim_id=f"c{i}",
                                       source_kind="internal_wiki",
                                       source_id=f"c{i-1}", credibility=0.8,
                                       linked_claim_ids=[f"c{i-1}"]))

        start = time.monotonic()
        gaps = detector.detect(claims, evidence)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, f"100 claims took {elapsed:.2f}s (>0.5s threshold)"

    def test_1000_claims_stress(self, detector):
        """1000 claims — stress test."""
        claims = []
        evidence = []
        for i in range(1000):
            claims.append(Claim(id=f"c{i}", entity_id="e1",
                              statement=f"Claim {i}",
                              epistemic_status=EpistemicStatus.CONFIRMED))
            if i > 0:
                evidence.append(Evidence(id=f"e{i}", claim_id=f"c{i}",
                                       source_kind="internal_wiki",
                                       source_id=f"c{i-1}", credibility=0.8,
                                       linked_claim_ids=[f"c{i-1}"]))

        start = time.monotonic()
        gaps = detector.detect(claims, evidence)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"1000 claims took {elapsed:.2f}s (>2s threshold)"

    def test_cherry_picking_heavy_evidence(self, detector):
        """Cherry-picking detection with 50 evidence items."""
        claim = Claim(id="c1", entity_id="e1", statement="Strategy works",
                     epistemic_status=EpistemicStatus.CONFIRMED)

        evidence = [
            Evidence(id=f"e{i}", claim_id="c1", source_kind="external_doc",
                    source_id=f"doc{i}", credibility=0.9 if i % 2 == 0 else 0.1)
            for i in range(50)
        ]

        start = time.monotonic()
        for _ in range(100):
            detector.detect([claim], evidence)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Cherry-picking stress took {elapsed:.2f}s"

    def test_over_generalization_many_claims(self, detector):
        """Over-generalization with 200 universal claims."""
        claims = [
            Claim(id=f"c{i}", entity_id="e1",
                 statement=f"All stocks follow pattern {i}",
                 epistemic_status=EpistemicStatus.CONFIRMED)
            for i in range(200)
        ]

        evidence = [
            Evidence(id=f"e{i}", claim_id=f"c{i}", source_kind="external_doc",
                    source_id=f"doc{i}", credibility=0.7)
            for i in range(200)
        ]

        start = time.monotonic()
        gaps = detector.detect(claims, evidence)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Over-generalization stress took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Functionality Tests
# ---------------------------------------------------------------------------
class TestCircularReasoning:
    """Verify circular reasoning detection."""

    def test_three_node_cycle(self, detector):
        """Detect A -> B -> C -> A cycle."""
        claims = [
            Claim(id="a", entity_id="e1", statement="A",
                 epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="b", entity_id="e1", statement="B",
                 epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="c", entity_id="e1", statement="C",
                 epistemic_status=EpistemicStatus.CONFIRMED),
        ]
        evidence = [
            Evidence(id="e1", claim_id="b", source_kind="internal_wiki",
                    source_id="a", credibility=0.8, linked_claim_ids=["a"]),
            Evidence(id="e2", claim_id="c", source_kind="internal_wiki",
                    source_id="b", credibility=0.8, linked_claim_ids=["b"]),
            Evidence(id="e3", claim_id="a", source_kind="internal_wiki",
                    source_id="c", credibility=0.8, linked_claim_ids=["c"]),
        ]

        gaps = detector.detect(claims, evidence)
        circular = [g for g in gaps if "circular" in g.rationale.lower()]

        assert len(circular) == 1
        assert circular[0].severity == "high"
        assert circular[0].kind == GapKind.LOGICAL

    def test_two_node_cycle(self, detector):
        """Detect A -> B -> A cycle."""
        claims = [
            Claim(id="a", entity_id="e1", statement="A",
                 epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="b", entity_id="e1", statement="B",
                 epistemic_status=EpistemicStatus.CONFIRMED),
        ]
        evidence = [
            Evidence(id="e1", claim_id="b", source_kind="internal_wiki",
                    source_id="a", credibility=0.8, linked_claim_ids=["a"]),
            Evidence(id="e2", claim_id="a", source_kind="internal_wiki",
                    source_id="b", credibility=0.8, linked_claim_ids=["b"]),
        ]

        gaps = detector.detect(claims, evidence)
        circular = [g for g in gaps if "circular" in g.rationale.lower()]

        assert len(circular) == 1

    def test_self_referencing_evidence(self, detector):
        """Detect A -> A self-reference."""
        claim = Claim(id="a", entity_id="e1", statement="Self-referencing",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id="e1", claim_id="a", source_kind="user",
                    source_id="user1", credibility=0.5, linked_claim_ids=["a"]),
        ]

        gaps = detector.detect([claim], evidence)
        circular = [g for g in gaps if "circular" in g.rationale.lower()]

        assert len(circular) == 1

    def test_no_cycle_no_flag(self, detector):
        """No cycle = no circular flag."""
        claims = [
            Claim(id="a", entity_id="e1", statement="A",
                 epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="b", entity_id="e1", statement="B",
                 epistemic_status=EpistemicStatus.CONFIRMED),
        ]
        evidence = [
            Evidence(id="e1", claim_id="b", source_kind="internal_wiki",
                    source_id="a", credibility=0.8, linked_claim_ids=["a"]),
        ]

        gaps = detector.detect(claims, evidence)
        circular = [g for g in gaps if "circular" in g.rationale.lower()]

        assert len(circular) == 0


class TestCherryPicking:
    """Verify cherry-picking detection."""

    def test_high_spread_detected(self, detector):
        """Detect cherry-picked evidence (high spread)."""
        claim = Claim(id="c1", entity_id="e1", statement="Strategy works",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id="e1", claim_id="c1", source_kind="external_doc",
                    source_id="doc1", credibility=0.95),
            Evidence(id="e2", claim_id="c1", source_kind="external_doc",
                    source_id="doc2", credibility=0.1),
        ]

        gaps = detector.detect([claim], evidence)
        cherry = [g for g in gaps if "cherry" in g.rationale.lower()]

        assert len(cherry) == 1
        assert cherry[0].severity == "medium"

    def test_balanced_evidence_no_flag(self, detector):
        """No cherry-picking when evidence is balanced."""
        claim = Claim(id="c1", entity_id="e1", statement="Strategy works",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id="e1", claim_id="c1", source_kind="external_doc",
                    source_id="doc1", credibility=0.6),
            Evidence(id="e2", claim_id="c1", source_kind="external_doc",
                    source_id="doc2", credibility=0.5),
        ]

        gaps = detector.detect([claim], evidence)
        cherry = [g for g in gaps if "cherry" in g.rationale.lower()]

        assert len(cherry) == 0


class TestOverGeneralization:
    """Verify over-generalization detection."""

    def test_universal_claim_detected(self, detector):
        """Detect universal claim with limited evidence."""
        claim = Claim(id="c1", entity_id="e1",
                     statement="All stocks follow momentum",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id="e1", claim_id="c1", source_kind="external_doc",
                    source_id="doc1", credibility=0.7),
        ]

        gaps = detector.detect([claim], evidence)
        overgen = [g for g in gaps if "over-generalization" in g.rationale.lower()]

        assert len(overgen) == 1
        assert overgen[0].severity == "medium"

    def test_sufficient_evidence_no_flag(self, detector):
        """No over-generalization when evidence is sufficient."""
        claim = Claim(id="c1", entity_id="e1",
                     statement="All stocks follow momentum",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id=f"e{i}", claim_id="c1", source_kind="external_doc",
                    source_id=f"doc{i}", credibility=0.7)
            for i in range(10)
        ]

        gaps = detector.detect([claim], evidence)
        overgen = [g for g in gaps if "over-generalization" in g.rationale.lower()]

        assert len(overgen) == 0

    def test_non_universal_no_flag(self, detector):
        """Non-universal claims should not trigger."""
        claim = Claim(id="c1", entity_id="e1",
                     statement="Momentum strategies work in trending markets",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id="e1", claim_id="c1", source_kind="external_doc",
                    source_id="doc1", credibility=0.7),
        ]

        gaps = detector.detect([claim], evidence)
        overgen = [g for g in gaps if "over-generalization" in g.rationale.lower()]

        assert len(overgen) == 0


class TestUnstatedAssumptions:
    """Verify LLM-based assumption detection."""

    def test_assumptions_detected(self, llm_detector):
        """LLM detects unstated assumptions."""
        claim = Claim(id="c1", entity_id="e1",
                     statement="High dividend yield means stock is safe",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id="e1", claim_id="c1", source_kind="user",
                    source_id="user1", credibility=0.5),
        ]

        gaps = llm_detector.detect([claim], evidence)
        assumptions = [g for g in gaps if "assumption" in g.rationale.lower()]

        assert len(assumptions) >= 1
        assert assumptions[0].severity == "high"

    def test_skipped_with_sufficient_evidence(self, llm_detector):
        """Skip assumption detection when claim has >2 evidence items."""
        claim = Claim(id="c1", entity_id="e1",
                     statement="High dividend yield means stock is safe",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id=f"e{i}", claim_id="c1", source_kind="external_doc",
                    source_id=f"doc{i}", credibility=0.7)
            for i in range(5)
        ]

        gaps = llm_detector.detect([claim], evidence)
        assumptions = [g for g in gaps if "assumption" in g.rationale.lower()]

        assert len(assumptions) == 0

    def test_llm_failure_graceful(self):
        """LLM failure should not crash the detector."""
        mock_llm = Mock()
        mock_llm.complete_sync.side_effect = ConnectionError("LLM down")

        detector = LogicalGapDetector(llm_client=mock_llm)
        claim = Claim(id="c1", entity_id="e1", statement="Test",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id="e1", claim_id="c1", source_kind="user",
                    source_id="user1", credibility=0.5),
        ]

        gaps = detector.detect([claim], evidence)
        assert isinstance(gaps, list)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test boundary conditions."""

    def test_empty_inputs(self, detector):
        """No claims = no gaps."""
        gaps = detector.detect([], [])
        assert gaps == []

    def test_claims_without_evidence(self, detector):
        """Claims with no evidence should not crash."""
        claims = [
            Claim(id="c1", entity_id="e1", statement="No evidence claim",
                 epistemic_status=EpistemicStatus.UNVERIFIED),
        ]
        gaps = detector.detect(claims, [])
        assert isinstance(gaps, list)

    def test_evidence_without_links(self, detector):
        """Evidence without linked_claim_ids should not crash."""
        claims = [
            Claim(id="c1", entity_id="e1", statement="Claim",
                 epistemic_status=EpistemicStatus.CONFIRMED),
        ]
        evidence = [
            Evidence(id="e1", claim_id="c1", source_kind="external_doc",
                    source_id="doc1", credibility=0.7, linked_claim_ids=[]),
        ]
        gaps = detector.detect(claims, evidence)
        assert isinstance(gaps, list)

    def test_mixed_gap_types(self, detector):
        """Detect multiple gap types in one call."""
        claims = [
            Claim(id="a", entity_id="e1", statement="A",
                 epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="b", entity_id="e1", statement="B",
                 epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="c", entity_id="e1",
                 statement="All stocks follow momentum",
                 epistemic_status=EpistemicStatus.CONFIRMED),
        ]
        evidence = [
            Evidence(id="e1", claim_id="b", source_kind="internal_wiki",
                    source_id="a", credibility=0.8, linked_claim_ids=["a"]),
            Evidence(id="e2", claim_id="a", source_kind="internal_wiki",
                    source_id="b", credibility=0.8, linked_claim_ids=["b"]),
            Evidence(id="e3", claim_id="c", source_kind="external_doc",
                    source_id="doc1", credibility=0.95),
            Evidence(id="e4", claim_id="c", source_kind="external_doc",
                    source_id="doc2", credibility=0.1),
        ]

        gaps = detector.detect(claims, evidence)
        gap_types = {g.kind for g in gaps}

        assert GapKind.LOGICAL in gap_types


# ---------------------------------------------------------------------------
# Efficiency Tests
# ---------------------------------------------------------------------------
class TestEfficiency:
    """Test memory and algorithmic efficiency."""

    def test_no_duplicate_gaps(self, detector):
        """Same claim should not produce duplicate gaps."""
        claims = [
            Claim(id="a", entity_id="e1", statement="A",
                 epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="b", entity_id="e1", statement="B",
                 epistemic_status=EpistemicStatus.CONFIRMED),
        ]
        evidence = [
            Evidence(id="e1", claim_id="b", source_kind="internal_wiki",
                    source_id="a", credibility=0.8, linked_claim_ids=["a"]),
            Evidence(id="e2", claim_id="a", source_kind="internal_wiki",
                    source_id="b", credibility=0.8, linked_claim_ids=["b"]),
        ]

        gaps = detector.detect(claims, evidence)
        assert len(gaps) == len(set((g.entity_id, g.slot_name, g.rationale) for g in gaps))


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestIntegration:
    """Test integration with GapDetector."""

    def test_gap_detector_integration(self):
        """LogicalGapDetector works through GapDetector."""
        from knowledge_engine.gaps import GapDetector

        detector = GapDetector()
        claim = Claim(id="c1", entity_id="e1",
                     statement="All stocks follow momentum",
                     epistemic_status=EpistemicStatus.CONFIRMED)
        evidence = [
            Evidence(id="e1", claim_id="c1", source_kind="external_doc",
                    source_id="doc1", credibility=0.7),
        ]

        gaps = detector.logical_gaps([claim], evidence)
        assert isinstance(gaps, list)
        assert any(g.kind == GapKind.LOGICAL for g in gaps)


# ---------------------------------------------------------------------------
# End-to-End User Scenario
# ---------------------------------------------------------------------------
class TestEndToEndScenario:
    """Simulate real user workflow over time.

    Scenario: Investment Research Analyst
    - Day 1: Ingests trading transcripts, discovers conflicts
    - Day 2: Adds more evidence, resolves conflicts
    - Day 3: Queries knowledge base, identifies logical gaps
    - Day 4: Reassesses beliefs based on new evidence
    """

    def test_investment_researcher_workflow(self, detector):
        """Full workflow: ingest -> conflict -> resolve -> query -> reassess."""
        # Day 1: Ingest initial claims from multiple experts
        day1_claims = [
            Claim(id="c_momentum", entity_id="strategy_momentum",
                 statement="Momentum strategies generate alpha in trending markets",
                 epistemic_status=EpistemicStatus.CONFIRMED,
                 slot_name="entry_signal"),
            Claim(id="c_value", entity_id="strategy_value",
                 statement="Value stocks outperform over 10-year horizons",
                 epistemic_status=EpistemicStatus.CONFIRMED,
                 slot_name="factor_investing"),
            Claim(id="c_risk", entity_id="risk_management",
                 statement="All portfolios should hold 60% equities",
                 epistemic_status=EpistemicStatus.UNVERIFIED,
                 slot_name="asset_allocation"),
        ]

        day1_evidence = [
            Evidence(id="e1", claim_id="c_momentum", source_kind="external_doc",
                    source_id="backtest_2024", credibility=0.85),
            Evidence(id="e2", claim_id="c_value", source_kind="external_doc",
                    source_id="fama_french", credibility=0.9),
            Evidence(id="e3", claim_id="c_risk", source_kind="user",
                    source_id="user_input", credibility=0.5),
        ]

        gaps_day1 = detector.detect(day1_claims, day1_evidence)

        # Should detect over-generalization on "All portfolios"
        overgen = [g for g in gaps_day1 if "over-generalization" in g.rationale.lower()]
        assert len(overgen) >= 1, "Should flag universal claim with limited evidence"

        # Day 2: Add conflicting evidence
        day2_claims = day1_claims + [
            Claim(id="c_contrarian", entity_id="strategy_contrarian",
                 statement="Momentum fails in mean-reverting markets",
                 epistemic_status=EpistemicStatus.UNVERIFIED,
                 slot_name="entry_signal"),
        ]

        day2_evidence = day1_evidence + [
            Evidence(id="e4", claim_id="c_contrarian", source_kind="external_doc",
                    source_id="regime_study", credibility=0.75),
        ]

        gaps_day2 = detector.detect(day2_claims, day2_evidence)

        # Day 3: Query and identify patterns
        # Simulate query that retrieves relevant claims
        relevant_claims = [c for c in day2_claims if "momentum" in c.statement.lower()]
        relevant_evidence = [e for e in day2_evidence if e.claim_id in {c.id for c in relevant_claims}]

        gaps_day3 = detector.detect(relevant_claims, relevant_evidence)

        # Day 4: Reassess with new evidence
        # Add evidence that challenges confirmed claim
        reassessment_evidence = [
            Evidence(id="e5", claim_id="c_momentum", source_kind="external_doc",
                    source_id="new_study_2025", credibility=0.8),
        ]

        # This would trigger reassessment in real system
        # Here we just verify the detector handles it
        all_gaps = detector.detect(day2_claims, day2_evidence + reassessment_evidence)

        # Verify the workflow produces meaningful results
        assert len(all_gaps) > 0, "Workflow should produce gap detections"
        assert any(g.kind == GapKind.LOGICAL for g in all_gaps)

    def test_domain_expertise_accumulation(self, detector):
        """Simulate accumulating expertise over multiple ingestions."""
        # Simulate 10 rounds of ingestion
        all_claims = []
        all_evidence = []

        for round_num in range(10):
            # Each round adds a claim
            claim = Claim(
                id=f"c_round_{round_num}",
                entity_id="accumulated_knowledge",
                statement=f"Insight from round {round_num}: {'All' if round_num % 3 == 0 else 'Some'} patterns repeat",
                epistemic_status=EpistemicStatus.CONFIRMED,
            )
            all_claims.append(claim)

            evidence = Evidence(
                id=f"e_round_{round_num}",
                claim_id=claim.id,
                source_kind="external_doc",
                source_id=f"doc_round_{round_num}",
                credibility=0.7,
            )
            all_evidence.append(evidence)

        # Detect gaps in accumulated knowledge
        gaps = detector.detect(all_claims, all_evidence)

        # Should detect over-generalization on universal claims
        overgen = [g for g in gaps if "over-generalization" in g.rationale.lower()]
        assert len(overgen) >= 3, "Should detect multiple over-generalized claims"

    def test_cross_domain_pattern_recognition(self, detector):
        """Simulate cross-domain analysis."""
        # Trading domain claims
        trading_claims = [
            Claim(id="t1", entity_id="trading",
                 statement="All trends continue indefinitely",
                 epistemic_status=EpistemicStatus.UNVERIFIED),
            Claim(id="t2", entity_id="trading",
                 statement="Past performance predicts future results",
                 epistemic_status=EpistemicStatus.CONFIRMED),
        ]

        # Real estate domain claims
        re_claims = [
            Claim(id="r1", entity_id="real_estate",
                 statement="All property values increase over time",
                 epistemic_status=EpistemicStatus.UNVERIFIED),
        ]

        # Evidence for trading
        trading_evidence = [
            Evidence(id="te1", claim_id="t1", source_kind="external_doc",
                    source_id="trend_study", credibility=0.6),
            Evidence(id="te2", claim_id="t2", source_kind="external_doc",
                    source_id="historical_data", credibility=0.8),
        ]

        # Evidence for real estate
        re_evidence = [
            Evidence(id="re1", claim_id="r1", source_kind="external_doc",
                    source_id="property_index", credibility=0.7),
        ]

        # Detect gaps across domains
        all_gaps = detector.detect(
            trading_claims + re_claims,
            trading_evidence + re_evidence,
        )

        # Should detect over-generalization in both domains
        overgen = [g for g in all_gaps if "over-generalization" in g.rationale.lower()]
        assert len(overgen) >= 2, "Should detect over-generalization across domains"


# ---------------------------------------------------------------------------
# Service Manager Tests
# ---------------------------------------------------------------------------
class TestServiceManager:
    """Test service lifecycle management."""

    def test_check_port(self):
        """Port check returns boolean."""
        from knowledge_engine.service_manager import ServiceManager

        manager = ServiceManager()
        # Port 19999 should be available
        result = manager.check_port(19999)
        assert isinstance(result, bool)

    def test_service_result_dataclass(self):
        """ServiceResult is properly structured."""
        from knowledge_engine.service_manager import ServiceResult, ServiceStatus

        result = ServiceResult(
            service="test",
            status=ServiceStatus.RUNNING,
            message="Test message",
        )
        assert result.service == "test"
        assert result.status == ServiceStatus.RUNNING

    def test_health_check_structure(self):
        """Health check returns proper structure."""
        from knowledge_engine.service_manager import ServiceManager

        manager = ServiceManager()
        health = manager.health_check()

        assert "neo4j" in health
        assert "ollama" in health
        assert "overall" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
