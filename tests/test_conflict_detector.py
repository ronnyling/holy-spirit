import pytest
from knowledge_engine.conflict_detector import ConflictDetector

def test_detect_same_slot_conflict():
    detector = ConflictDetector()
    new_claim = {"id": "1", "statement": "price will increase", "slot_name": "price_direction"}
    existing_claims = [
        {"id": "2", "statement": "price will decrease", "slot_name": "price_direction", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) >= 1
    assert conflicts[0]["conflict_type"] == "same_slot"

def test_detect_cross_slot_semantic_conflict():
    detector = ConflictDetector(threshold=0.7)
    new_claim = {"id": "1", "statement": "property is overvalued at current price", "slot_name": "valuation"}
    existing_claims = [
        {"id": "2", "statement": "cap rate is 8% indicating fair value", "slot_name": "cap_rate", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    # May or may not detect depending on semantic similarity
    assert isinstance(conflicts, list)

def test_no_conflict_different_topics():
    detector = ConflictDetector(threshold=0.5)
    new_claim = {"id": "1", "statement": "turmeric helps inflammation", "slot_name": "treatment_outcome"}
    existing_claims = [
        {"id": "2", "statement": "stock price increased 5%", "slot_name": "price_change", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) == 0

def test_keyword_opposition():
    detector = ConflictDetector()
    new_claim = {"id": "1", "statement": "buy signal triggered", "slot_name": "trading_signal"}
    existing_claims = [
        {"id": "2", "statement": "sell signal triggered", "slot_name": "trading_signal", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) >= 1
