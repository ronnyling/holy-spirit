import pytest
from knowledge_engine.semantic_dedup import SemanticDeduplicator

def test_find_duplicates_empty():
    dedup = SemanticDeduplicator(threshold=0.9)
    duplicates = dedup.find_duplicates([])
    assert duplicates == []

def test_find_duplicates_exact_match():
    dedup = SemanticDeduplicator(threshold=0.9)
    claims = [
        {"id": "1", "statement": "turmeric reduces inflammation"},
        {"id": "2", "statement": "turmeric reduces inflammation"}
    ]
    duplicates = dedup.find_duplicates(claims)
    assert len(duplicates) == 1
    assert duplicates[0]["claim_ids"] == ["1", "2"]

def test_find_duplicates_semantic_match():
    dedup = SemanticDeduplicator(threshold=0.5)
    claims = [
        {"id": "1", "statement": "turmeric reduces inflammation by 40%"},
        {"id": "2", "statement": "curcumin decreases inflammation approximately 40 percent"}
    ]
    duplicates = dedup.find_duplicates(claims)
    assert len(duplicates) >= 1

def test_find_duplicates_no_match():
    dedup = SemanticDeduplicator(threshold=0.9)
    claims = [
        {"id": "1", "statement": "turmeric reduces inflammation"},
        {"id": "2", "statement": "exercise improves cardiovascular health"}
    ]
    duplicates = dedup.find_duplicates(claims)
    assert len(duplicates) == 0
