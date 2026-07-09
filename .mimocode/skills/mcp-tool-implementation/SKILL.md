# MCP Tool Implementation Skill

## Purpose
Standardized workflow for implementing new MCP tools in the knowledge_engine project.

## When to Use
- Adding a new tool to `server.py`
- Creating new detection/processing logic
- Adding new models or contracts
- Wiring new functionality into the engine pipeline

## Prerequisites
- Knowledge engine project structure
- Existing MCP tools as reference (server.py)
- Understanding of the pipeline (engine.py)

## Workflow

### Step 1: Define Models/Contracts
Location: `src/knowledge_engine/contracts.py` or new file

```python
# Add to contracts.py or create new file
class NewModel(BaseModel):
    """Description of the model."""
    field1: str
    field2: int = 0
    field3: list[str] = Field(default_factory=list)
```

### Step 2: Implement Detection/Processing
Location: `src/knowledge_engine/new_feature.py`

```python
"""New feature description."""
from __future__ import annotations
from typing import Any

class NewDetector:
    """Detection logic."""
    
    def detect(self, input_data: list) -> list[dict]:
        """Detect patterns in input data."""
        results = []
        for item in input_data:
            # Detection logic here
            results.append({...})
        return results
```

### Step 3: Wire into Engine
Location: `src/knowledge_engine/engine.py`

```python
# Add import at top
from .new_feature import NewDetector

# Add to KnowledgeEngine.__init__
self.new_detector = NewDetector()

# Add method
def new_method(self, param: str) -> dict[str, Any]:
    """Description of the method."""
    result = self.new_detector.detect(param)
    return result
```

### Step 4: Add MCP Tool
Location: `src/knowledge_engine/server.py`

```python
@mcp_server.tool()
def new_tool(param: str) -> dict[str, Any]:
    """Tool description for MCP clients.
    
    Args:
        param: Parameter description
    
    Returns:
        Description of return value
    """
    return engine.new_method(param)
```

### Step 5: Write Tests
Location: `tests/test_new_feature.py`

```python
"""Tests for new feature."""
import pytest
from knowledge_engine.new_feature import NewDetector

class TestNewDetector:
    def test_detect_basic(self):
        detector = NewDetector()
        result = detector.detect([])
        assert isinstance(result, list)
    
    def test_detect_with_data(self):
        detector = NewDetector()
        # Test with actual data
        assert len(result) > 0
```

### Step 6: Update Documentation
Location: `README.md`, `wiki/Architecture.md`

- Add feature to "Implemented" section
- Update architecture diagrams if needed
- Add API documentation if public

### Step 7: Commit
```bash
git add -A
git commit -m "feat: add [feature name]

- [brief description]
- Files: [list key files]"
```

## File Structure Reference

```
src/knowledge_engine/
├── contracts.py      # Models and data types
├── engine.py         # Main orchestrator
├── server.py         # MCP tools (stdio)
├── http_server.py    # HTTP transport (Android)
├── new_feature.py    # New detection/processing
└── tests/
    └── test_new_feature.py
```

## Common Patterns

### Detection Pattern
```python
class Detector:
    def detect(self, items: list) -> list[dict]:
        results = []
        for item in items:
            if self._matches_criteria(item):
                results.append(self._process(item))
        return results
```

### Integration Pattern
```python
# In engine.py
self.detector = Detector()

def detect_method(self, param: str) -> dict:
    return self.detector.detect(param)
```

### MCP Tool Pattern
```python
@mcp_server.tool()
def tool_name(param: str) -> dict[str, Any]:
    """Description."""
    return engine.method(param)
```

## Checklist

- [ ] Models/contracts defined
- [ ] Detection/processing implemented
- [ ] Wired into engine
- [ ] MCP tool added
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Committed with descriptive message
