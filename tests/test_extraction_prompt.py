import pytest
from knowledge_engine.extraction_prompt import build_enhanced_extraction_prompt

def test_build_prompt_returns_string():
    prompt = build_enhanced_extraction_prompt("Test transcript text")
    assert isinstance(prompt, str)
    assert "Test transcript text" in prompt

def test_prompt_includes_evidence_fields():
    prompt = build_enhanced_extraction_prompt("Test text")
    assert "source_quality" in prompt
    assert "conditions" in prompt
    assert "measurement_method" in prompt
    assert "confidence_indicator" in prompt

def test_prompt_includes_json_format():
    prompt = build_enhanced_extraction_prompt("Test text")
    assert "JSON" in prompt or "json" in prompt
    assert "claims" in prompt
    assert "evidence" in prompt
