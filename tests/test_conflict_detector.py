"""Tests for conflict detector with domain-specific keyword opposition.

NO FALLBACKS: Tests verify explicit behavior, no silent failures.
"""
import pytest
from knowledge_engine.conflict_detector import ConflictDetector


def test_detect_same_slot_conflict():
    """Same-slot conflict detected with keyword opposition."""
    detector = ConflictDetector(domain="trading")
    new_claim = {"id": "1", "statement": "price will increase", "slot_name": "price_direction"}
    existing_claims = [
        {"id": "2", "statement": "price will decrease", "slot_name": "price_direction", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) >= 1
    assert conflicts[0]["conflict_type"] == "same_slot"


def test_detect_keyword_opposition_trading():
    """Trading domain keyword opposition detected."""
    detector = ConflictDetector(domain="trading")
    new_claim = {"id": "1", "statement": "buy signal triggered", "slot_name": "trading_signal"}
    existing_claims = [
        {"id": "2", "statement": "sell signal triggered", "slot_name": "trading_signal", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) >= 1


def test_detect_keyword_opposition_tcm():
    """TCM domain keyword opposition detected."""
    detector = ConflictDetector(domain="tcm")
    new_claim = {"id": "1", "statement": "patient has excess condition", "slot_name": "pattern"}
    existing_claims = [
        {"id": "2", "statement": "patient has deficiency condition", "slot_name": "pattern", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) >= 1


def test_detect_keyword_opposition_real_estate():
    """Real estate domain keyword opposition detected."""
    detector = ConflictDetector(domain="real_estate")
    new_claim = {"id": "1", "statement": "property values will appreciate", "slot_name": "value_trend"}
    existing_claims = [
        {"id": "2", "statement": "property values will depreciate", "slot_name": "value_trend", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) >= 1


def test_no_conflict_different_topics():
    """No conflict between unrelated claims."""
    detector = ConflictDetector(domain="general", threshold=0.5)
    new_claim = {"id": "1", "statement": "turmeric helps inflammation", "slot_name": "treatment_outcome"}
    existing_claims = [
        {"id": "2", "statement": "stock price increased 5%", "slot_name": "price_change", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) == 0


def test_similar_claims_detected_as_conflict():
    """Similar claims are detected as potential conflicts (may be duplicates)."""
    detector = ConflictDetector(domain="general", threshold=0.9)
    new_claim = {"id": "1", "statement": "turmeric reduces inflammation", "slot_name": "treatment"}
    existing_claims = [
        {"id": "2", "statement": "turmeric reduces inflammation", "slot_name": "treatment", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    # Identical claims have similarity 1.0, which exceeds threshold
    # This is correct behavior - duplicates should be detected
    assert len(conflicts) == 1
    assert conflicts[0]["conflict_type"] == "text_similarity"
    assert conflicts[0]["confidence"] == 1.0


def test_skips_non_active_claims():
    """Non-active claims are skipped in conflict detection."""
    detector = ConflictDetector(domain="trading")
    new_claim = {"id": "1", "statement": "buy now", "slot_name": "signal"}
    existing_claims = [
        {"id": "2", "statement": "sell now", "slot_name": "signal", "status": "UNKNOWN"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) == 0


def test_empty_existing_claims():
    """No conflicts when no existing claims."""
    detector = ConflictDetector(domain="general")
    new_claim = {"id": "1", "statement": "test claim", "slot_name": "test"}
    conflicts = detector.detect(new_claim, [])
    assert len(conflicts) == 0


def test_invalid_domain_raises_error():
    """Must raise error for invalid domain - no silent fallback."""
    with pytest.raises(ValueError) as exc_info:
        ConflictDetector(domain="invalid_domain")
    assert "Unknown domain" in str(exc_info.value)


def test_get_available_domains():
    """Get list of available domains."""
    detector = ConflictDetector(domain="general")
    domains = detector.get_available_domains()
    assert "trading" in domains
    assert "tcm" in domains
    assert "real_estate" in domains
    assert "general" in domains


def test_get_domain_pairs():
    """Get keyword pairs for specific domain."""
    detector = ConflictDetector(domain="general")
    pairs = detector.get_domain_pairs("trading")
    assert pairs["buy"] == "sell"
    assert pairs["bullish"] == "bearish"


def test_get_domain_pairs_invalid_raises_error():
    """Must raise error for invalid domain in get_domain_pairs."""
    detector = ConflictDetector(domain="general")
    with pytest.raises(ValueError) as exc_info:
        detector.get_domain_pairs("invalid_domain")
    assert "Unknown domain" in str(exc_info.value)


def test_bidirectional_opposition():
    """Detect opposition in both directions."""
    detector = ConflictDetector(domain="trading")
    # Direction 1: new has "buy", existing has "sell"
    new_claim = {"id": "1", "statement": "buy signal", "slot_name": "signal"}
    existing_claims = [
        {"id": "2", "statement": "sell signal", "slot_name": "signal", "status": "CONFIRMED"}
    ]
    conflicts = detector.detect(new_claim, existing_claims)
    assert len(conflicts) >= 1

    # Direction 2: new has "sell", existing has "buy"
    new_claim2 = {"id": "3", "statement": "sell signal", "slot_name": "signal"}
    existing_claims2 = [
        {"id": "4", "statement": "buy signal", "slot_name": "signal", "status": "CONFIRMED"}
    ]
    conflicts2 = detector.detect(new_claim2, existing_claims2)
    assert len(conflicts2) >= 1
