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


def test_detects_cherry_picking():
    detector = LogicalGapDetector()

    # Claim supported only by high-credibility evidence (ignoring low-cred)
    claim = Claim(id="c1", entity_id="e1", statement="Strategy X works", epistemic_status=EpistemicStatus.CONFIRMED)

    # Only linked to high-cred evidence
    evidence_high = Evidence(id="eh", claim_id="c1", source_kind="external_doc", source_id="doc1", credibility=0.9, linked_claim_ids=[])
    evidence_low = Evidence(id="el", claim_id="c1", source_kind="external_doc", source_id="doc2", credibility=0.2, linked_claim_ids=[])

    gaps = detector.detect([claim], [evidence_high, evidence_low])

    cherry_gaps = [g for g in gaps if "cherry" in g.rationale.lower()]
    assert len(cherry_gaps) == 1
    assert cherry_gaps[0].severity == "medium"
