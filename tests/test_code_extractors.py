import pytest
from knowledge_engine.code_extractors import DocumentStructureParser, MetadataExtractor

def test_parse_document_structure():
    parser = DocumentStructureParser()
    text = """# Title

Paragraph 1 with some text.

## Section 1

More content here.

### Subsection

Final paragraph."""
    structure = parser.parse(text)
    assert structure["title"] == "Title"
    assert len(structure["sections"]) >= 2
    assert structure["paragraph_count"] >= 3

def test_extract_metadata():
    extractor = MetadataExtractor()
    text = """
    Study conducted in 2024 with 150 participants.
    Results show 40% improvement. See https://example.com/study
    Contact: researcher@university.edu
    """
    metadata = extractor.extract(text)
    assert "2024" in str(metadata["dates"])
    assert metadata["has_urls"] == True
    assert metadata["has_emails"] == True

def test_extract_quantification():
    extractor = MetadataExtractor()
    text = "The study showed a 40% reduction in symptoms over 12 weeks with n=150 participants."
    quant = extractor.extract_quantification(text)
    assert quant["has_percentage"] == True
    assert quant["has_time_period"] == True
    assert quant["has_sample_size"] == True
