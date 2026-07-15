import re
from typing import Dict, List, Any

class DocumentStructureParser:
    """Parse document structure from text."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse document structure."""
        lines = text.split("\n")
        sections = []
        title = ""
        paragraph_count = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not title:
                title = stripped[2:]
            elif stripped.startswith("## ") or stripped.startswith("### "):
                sections.append(stripped.lstrip("#").strip())
            elif stripped and not stripped.startswith("#"):
                paragraph_count += 1
        
        return {
            "title": title,
            "sections": sections,
            "paragraph_count": paragraph_count
        }

class MetadataExtractor:
    """Extract metadata from text using regex."""
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text."""
        dates = re.findall(r'\b(20\d{2})\b', text)
        urls = re.findall(r'https?://\S+', text)
        emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
        
        return {
            "dates": list(set(dates)),
            "has_urls": len(urls) > 0,
            "urls": urls,
            "has_emails": len(emails) > 0,
            "emails": emails
        }
    
    def extract_quantification(self, text: str) -> Dict[str, bool]:
        """Extract quantification indicators."""
        return {
            "has_percentage": bool(re.search(r'\d+%', text)),
            "has_time_period": bool(re.search(r'\d+\s*(weeks?|months?|years?|days?)', text, re.IGNORECASE)),
            "has_sample_size": bool(re.search(r'n\s*=\s*\d+', text, re.IGNORECASE))
        }
