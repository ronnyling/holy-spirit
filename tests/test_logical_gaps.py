from knowledge_engine.logical_gaps import LogicalGapDetector
from knowledge_engine.models import Claim, Evidence, EpistemicStatus


def test_logical_gap_detector_instantiates():
    detector = LogicalGapDetector()
    assert detector is not None


def test_detects_circular_reasoning():
    detector = LogicalGapDetector()

    # Create a circular chain: A -> B -> C -> A
    claim_a = Claim(id="a", entity_id="e1", statement="A is true", epistemic_status=EpistemicStatus.CONFIRMED)
    claim_b = Claim(id="b", entity_id="e1", statement="B is true", epistemic_status=EpistemicStatus.CONFIRMED)
    claim_c = Claim(id="c", entity_id="e1", statement="C is true", epistemic_status=EpistemicStatus.CONFIRMED)

    evidence_ab = Evidence(id="eab", claim_id="b", source_kind="internal_wiki", source_id="a", credibility=0.8, linked_claim_ids=["a"])
    evidence_bc = Evidence(id="ebc", claim_id="c", source_kind="internal_wiki", source_id="b", credibility=0.8, linked_claim_ids=["b"])
    evidence_ca = Evidence(id="eca", claim_id="a", source_kind="internal_wiki", source_id="c", credibility=0.8, linked_claim_ids=["c"])

    gaps = detector.detect([claim_a, claim_b, claim_c], [evidence_ab, evidence_bc, evidence_ca])

    circular_gaps = [g for g in gaps if "circular" in g.rationale.lower()]
    assert len(circular_gaps) == 1
    assert circular_gaps[0].severity == "high"
