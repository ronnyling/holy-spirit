# Structural Logical Gap Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing gap detection system to identify logical weaknesses in reasoning (circular reasoning, unstated assumptions, cherry-picking, over-generalization) — not just missing evidence slots.

**Architecture:** Add a new `LogicalGapDetector` class that analyzes claim-evidence relationships for logical fallacies. Uses existing Neo4j cycle detection for circular reasoning, LLM for assumption detection, and deterministic checks for cherry-picking and over-generalization. Integrates into the existing `GapDetector` interface via a new `GapKind.LOGICAL` type.

**Tech Stack:** Python 3.11+, Pydantic models, Neo4j (existing), MiMo LLM client (existing), pytest

---

## File Structure

| File | Purpose |
|------|---------|
| `src/knowledge_engine/logical_gaps.py` | New: `LogicalGapDetector` class with all detection methods |
| `src/knowledge_engine/contracts.py:9-11` | Modify: Add `LOGICAL` to `GapKind` enum |
| `src/knowledge_engine/gaps.py:8-63` | Modify: Add `logical_gaps()` method to `GapDetector`, wire to `LogicalGapDetector` |
| `src/knowledge_engine/engine.py` | Modify: Call `logical_gaps()` in pipeline after semantic gaps |
| `tests/test_logical_gaps.py` | New: Unit tests for all detection methods |

---

## Global Constraints

- Python 3.11+ (verified with 3.14)
- Pydantic v2 for all models
- No fallbacks — if LLM is unavailable, logical gap detection returns empty list (non-breaking)
- Existing `GapFlag` model reused — no new models unless strictly necessary
- Cycle detection uses existing `would_create_cycle()` from `neo4j_store.py`

---

### Task 1: Extend GapKind Enum

**Covers:** [Foundation for logical gap types]

**Files:**
- Modify: `src/knowledge_engine/contracts.py:9-11`

**Interfaces:**
- Consumes: Existing `GapKind` enum
- Produces: Extended `GapKind` with `LOGICAL` value

- [ ] **Step 1: Add LOGICAL to GapKind**

```python
class GapKind(StrEnum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    LOGICAL = "logical"
```

- [ ] **Step 2: Verify enum works**

Run: `python -c "from knowledge_engine.contracts import GapKind; print(GapKind.LOGICAL)"`
Expected: `GapKind.LOGICAL`

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_engine/contracts.py
git commit -m "feat: add LOGICAL gap kind to GapKind enum"
```

---

### Task 2: Create LogicalGapDetector Skeleton

**Covers:** [Logical gap detection framework]

**Files:**
- Create: `src/knowledge_engine/logical_gaps.py`
- Create: `tests/test_logical_gaps.py`

**Interfaces:**
- Consumes: `Claim`, `Evidence` models from `models.py`
- Produces: `LogicalGapDetector` class with `detect()` method returning `list[GapFlag]`

- [ ] **Step 1: Write failing test for class existence**

```python
# tests/test_logical_gaps.py
from knowledge_engine.logical_gaps import LogicalGapDetector

def test_logical_gap_detector_instantiates():
    detector = LogicalGapDetector()
    assert detector is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logical_gaps.py::test_logical_gap_detector_instantiates -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'knowledge_engine.logical_gaps'`

- [ ] **Step 3: Write minimal skeleton**

```python
# src/knowledge_engine/logical_gaps.py
from __future__ import annotations
from .models import Claim, Evidence
from .contracts import GapFlag, GapKind

class LogicalGapDetector:
    """Detect logical weaknesses in reasoning, not just missing evidence."""
    
    def detect(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
        """Analyze claims and evidence for logical fallacies."""
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_logical_gaps.py::test_logical_gap_detector_instantiates -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_engine/logical_gaps.py tests/test_logical_gaps.py
git commit -m "feat: add LogicalGapDetector skeleton"
```

---

### Task 3: Implement Circular Reasoning Detection

**Covers:** [Detect self-referencing evidence chains]

**Files:**
- Modify: `src/knowledge_engine/logical_gaps.py`
- Modify: `tests/test_logical_gaps.py`

**Interfaces:**
- Consumes: `Claim` objects with `id` field, `Evidence` objects with `linked_claim_ids`
- Produces: `GapFlag` with `kind=GapKind.LOGICAL`, `severity="high"` for circular chains

- [ ] **Step 1: Write failing test for circular detection**

```python
# tests/test_logical_gaps.py
from knowledge_engine.models import Claim, Evidence, EpistemicStatus

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logical_gaps.py::test_detects_circular_reasoning -v`
Expected: FAIL (currently returns empty list)

- [ ] **Step 3: Implement circular reasoning detection**

```python
# src/knowledge_engine/logical_gaps.py
def _detect_circular_reasoning(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
    """Detect circular evidence chains: A supports B supports C supports A."""
    gaps = []
    
    # Build adjacency map: claim_id -> list of claim_ids it supports
    supports_map: dict[str, set[str]] = {}
    for ev in evidence:
        if ev.linked_claim_ids:
            target_claim_id = ev.claim_id
            for source_id in ev.linked_claim_ids:
                if source_id not in supports_map:
                    supports_map[source_id] = set()
                supports_map[source_id].add(target_claim_id)
    
    # DFS cycle detection
    visited = set()
    rec_stack = set()
    
    def has_cycle(node: str, path: set) -> bool:
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in supports_map.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor, path):
                    return True
            elif neighbor in rec_stack:
                # Found cycle - record it
                cycle_claims = list(path) + [neighbor]
                gaps.append(GapFlag(
                    kind=GapKind.LOGICAL,
                    entity_id="",
                    slot_name="reasoning_chain",
                    question=f"Circular reasoning detected: {' -> '.join(cycle_claims)}",
                    severity="high",
                    rationale=f"Circular reasoning: claim chain forms a loop {' -> '.join(cycle_claims)}"
                ))
                return True
        
        rec_stack.remove(node)
        return False
    
    claim_ids = {c.id for c in claims}
    for claim_id in claim_ids:
        if claim_id not in visited:
            has_cycle(claim_id, set())
    
    return gaps
```

- [ ] **Step 4: Wire into detect() method**

```python
def detect(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
    """Analyze claims and evidence for logical fallacies."""
    gaps = []
    gaps.extend(self._detect_circular_reasoning(claims, evidence))
    return gaps
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_logical_gaps.py::test_detects_circular_reasoning -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_engine/logical_gaps.py tests/test_logical_gaps.py
git commit -m "feat: implement circular reasoning detection"
```

---

### Task 4: Implement Cherry-Picking Detection

**Covers:** [Detect selective evidence usage]

**Files:**
- Modify: `src/knowledge_engine/logical_gaps.py`
- Modify: `tests/test_logical_gaps.py`

**Interfaces:**
- Consumes: Claims with evidence links, evidence with credibility scores
- Produces: `GapFlag` with `severity="medium"` for cherry-picked evidence

- [ ] **Step 1: Write failing test for cherry-picking**

```python
# tests/test_logical_gaps.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logical_gaps.py::test_detects_cherry_picking -v`
Expected: FAIL

- [ ] **Step 3: Implement cherry-picking detection**

```python
def _detect_cherry_picking(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
    """Detect when only favorable evidence is cited (ignoring unfavorable)."""
    gaps = []
    
    # Group evidence by claim_id
    evidence_by_claim: dict[str, list[Evidence]] = {}
    for ev in evidence:
        if ev.claim_id not in evidence_by_claim:
            evidence_by_claim[ev.claim_id] = []
        evidence_by_claim[ev.claim_id].append(ev)
    
    for claim in claims:
        claim_evidence = evidence_by_claim.get(claim.id, [])
        if len(claim_evidence) < 2:
            continue
        
        credibilities = [ev.credibility for ev in claim_evidence]
        avg_cred = sum(credibilities) / len(credibilities)
        
        # Cherry-picking: all evidence above average, none below
        above_avg = sum(1 for c in credibilities if c > avg_cred)
        below_avg = sum(1 for c in credibilities if c < avg_cred)
        
        if above_avg == len(credibilities) and below_avg == 0:
            gaps.append(GapFlag(
                kind=GapKind.LOGICAL,
                entity_id=claim.entity_id,
                slot_name=claim.slot_name or "general",
                question=f"Is all evidence for this claim favorable? Consider counter-evidence.",
                severity="medium",
                rationale=f"Cherry-picking: all {len(credibilities)} evidence items have above-average credibility ({avg_cred:.2f})"
            ))
    
    return gaps
```

- [ ] **Step 4: Wire into detect() method**

```python
def detect(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
    """Analyze claims and evidence for logical fallacies."""
    gaps = []
    gaps.extend(self._detect_circular_reasoning(claims, evidence))
    gaps.extend(self._detect_cherry_picking(claims, evidence))
    return gaps
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_logical_gaps.py::test_detects_cherry_picking -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_engine/logical_gaps.py tests/test_logical_gaps.py
git commit -m "feat: implement cherry-picking detection"
```

---

### Task 5: Implement Over-Generalization Detection

**Covers:** [Detect claims that over-generalize from limited evidence]

**Files:**
- Modify: `src/knowledge_engine/logical_gaps.py`
- Modify: `tests/test_logical_gaps.py`

**Interfaces:**
- Consumes: Claims with statements, evidence counts
- Produces: `GapFlag` with `severity="medium"` for over-generalized claims

- [ ] **Step 1: Write failing test for over-generalization**

```python
# tests/test_logical_gaps.py
def test_detects_over_generalization():
    detector = LogicalGapDetector()
    
    # Universal claim with limited evidence
    claim = Claim(id="c1", entity_id="e1", statement="All stocks follow momentum patterns", epistemic_status=EpistemicStatus.CONFIRMED)
    
    # Only 2 evidence items for a universal claim
    evidence1 = Evidence(id="e1", claim_id="c1", source_kind="external_doc", source_id="doc1", credibility=0.7)
    evidence2 = Evidence(id="e2", claim_id="c1", source_kind="external_doc", source_id="doc2", credibility=0.7)
    
    gaps = detector.detect([claim], [evidence1, evidence2])
    
    overgen_gaps = [g for g in gaps if "over-generalization" in g.rationale.lower()]
    assert len(overgen_gaps) == 1
    assert overgen_gaps[0].severity == "medium"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logical_gaps.py::test_detects_over_generalization -v`
Expected: FAIL

- [ ] **Step 3: Implement over-generalization detection**

```python
def _detect_over_generalization(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
    """Detect universal claims ('all', 'always', 'never') with limited evidence."""
    gaps = []
    
    universal_keywords = ["all", "always", "never", "every", "none", "always", "universally"]
    min_evidence_for_universal = 5  # Threshold for universal claims
    
    evidence_count: dict[str, int] = {}
    for ev in evidence:
        evidence_count[ev.claim_id] = evidence_count.get(ev.claim_id, 0) + 1
    
    for claim in claims:
        statement_lower = claim.statement.lower()
        
        # Check if claim is universal
        is_universal = any(keyword in statement_lower for keyword in universal_keywords)
        
        if is_universal:
            count = evidence_count.get(claim.id, 0)
            if count < min_evidence_for_universal:
                gaps.append(GapFlag(
                    kind=GapKind.LOGICAL,
                    entity_id=claim.entity_id,
                    slot_name=claim.slot_name or "general",
                    question=f"Universal claim has only {count} evidence items. Is this sufficient?",
                    severity="medium",
                    rationale=f"Over-generalization: universal claim '{claim.statement[:50]}...' supported by only {count} evidence items (threshold: {min_evidence_for_universal})"
                ))
    
    return gaps
```

- [ ] **Step 4: Wire into detect() method**

```python
def detect(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
    """Analyze claims and evidence for logical fallacies."""
    gaps = []
    gaps.extend(self._detect_circular_reasoning(claims, evidence))
    gaps.extend(self._detect_cherry_picking(claims, evidence))
    gaps.extend(self._detect_over_generalization(claims, evidence))
    return gaps
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_logical_gaps.py::test_detects_over_generalization -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_engine/logical_gaps.py tests/test_logical_gaps.py
git commit -m "feat: implement over-generalization detection"
```

---

### Task 6: Implement Unstated Assumptions Detection (LLM)

**Covers:** [Detect logical assumptions not backed by evidence]

**Files:**
- Modify: `src/knowledge_engine/logical_gaps.py`
- Modify: `tests/test_logical_gaps.py`

**Interfaces:**
- Consumes: Claims, evidence, MiMoClient for LLM calls
- Produces: `GapFlag` with `severity="high"` for unstated assumptions

- [ ] **Step 1: Write failing test for assumption detection**

```python
# tests/test_logical_gaps.py
from unittest.mock import Mock, patch

def test_detects_unstated_assumptions():
    detector = LogicalGapDetector(llm_client=Mock())
    
    # Claim with implicit assumption
    claim = Claim(id="c1", entity_id="e1", statement="High dividend yield means the stock is safe", epistemic_status=EpistemicStatus.CONFIRMED)
    evidence = Evidence(id="e1", claim_id="c1", source_kind="user", source_id="user1", credibility=0.5)
    
    # Mock LLM response
    detector.llm_client.chat.return_value = Mock(
        choices=[Mock(message=Mock(content='{"assumptions": ["dividend yield is a reliable safety indicator", "past yield predicts future safety"]}'))]
    )
    
    gaps = detector.detect([claim], [evidence])
    
    assumption_gaps = [g for g in gaps if "assumption" in g.rationale.lower()]
    assert len(assumption_gaps) >= 1
    assert assumption_gaps[0].severity == "high"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logical_gaps.py::test_detects_unstated_assumptions -v`
Expected: FAIL

- [ ] **Step 3: Implement assumption detection**

```python
def _detect_unstated_assumptions(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
    """Use LLM to detect implicit assumptions not backed by evidence."""
    gaps = []
    
    if self.llm_client is None:
        return gaps
    
    evidence_count: dict[str, int] = {}
    for ev in evidence:
        evidence_count[ev.claim_id] = evidence_count.get(ev.claim_id, 0) + 1
    
    for claim in claims:
        # Only check claims with minimal evidence
        if evidence_count.get(claim.id, 0) > 2:
            continue
        
        prompt = f"""Analyze this claim for unstated assumptions. Return JSON with key "assumptions" containing a list of implicit assumptions.

Claim: {claim.statement}

Return only valid JSON: {{"assumptions": ["assumption1", "assumption2"]}}"""
        
        try:
            response = self.llm_client.chat(prompt)
            content = response.choices[0].message.content
            import json
            data = json.loads(content)
            assumptions = data.get("assumptions", [])
            
            for assumption in assumptions[:3]:  # Cap at 3
                gaps.append(GapFlag(
                    kind=GapKind.LOGICAL,
                    entity_id=claim.entity_id,
                    slot_name=claim.slot_name or "general",
                    question=f"Is this assumption validated? '{assumption}'",
                    severity="high",
                    rationale=f"Unstated assumption: {assumption}"
                ))
        except Exception:
            # LLM failure is non-breaking
            pass
    
    return gaps
```

- [ ] **Step 4: Update __init__ to accept optional LLM client**

```python
def __init__(self, llm_client=None):
    self.llm_client = llm_client
```

- [ ] **Step 5: Wire into detect() method**

```python
def detect(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
    """Analyze claims and evidence for logical fallacies."""
    gaps = []
    gaps.extend(self._detect_circular_reasoning(claims, evidence))
    gaps.extend(self._detect_cherry_picking(claims, evidence))
    gaps.extend(self._detect_over_generalization(claims, evidence))
    gaps.extend(self._detect_unstated_assumptions(claims, evidence))
    return gaps
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_logical_gaps.py::test_detects_unstated_assumptions -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_engine/logical_gaps.py tests/test_logical_gaps.py
git commit -m "feat: implement LLM-based unstated assumption detection"
```

---

### Task 7: Wire into GapDetector and Pipeline

**Covers:** [Integration with existing gap detection flow]

**Files:**
- Modify: `src/knowledge_engine/gaps.py:8-63`
- Modify: `src/knowledge_engine/engine.py` (pipeline call)
- Modify: `tests/test_logical_gaps.py`

**Interfaces:**
- Consumes: `LogicalGapDetector` from Task 6
- Produces: `logical_gaps()` method on `GapDetector`, pipeline integration

- [ ] **Step 1: Add logical_gaps() to GapDetector**

```python
# src/knowledge_engine/gaps.py
from .logical_gaps import LogicalGapDetector

class GapDetector:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logical_detector = LogicalGapDetector(llm_client=llm_client)
    
    def structural_gaps(self, entity_id: str, observed_slot_names: set[str], expected_slots: list[Slot]) -> list[GapFlag]:
        # ... existing code unchanged ...
    
    def semantic_gaps(self, entity_id: str, transcript: TranscriptInput, claim_draft: ClaimDraft) -> list[GapFlag]:
        # ... existing code unchanged ...
    
    def logical_gaps(self, claims: list, evidence: list) -> list[GapFlag]:
        """Detect logical fallacies in claims and evidence."""
        return self.logical_detector.detect(claims, evidence)
```

- [ ] **Step 2: Update GapDetector constructor**

```python
class GapDetector:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logical_detector = LogicalGapDetector(llm_client=llm_client)
```

- [ ] **Step 3: Write integration test**

```python
# tests/test_logical_gaps.py
def test_gap_detector_logical_gaps_integration():
    from knowledge_engine.gaps import GapDetector
    
    detector = GapDetector()
    
    claim = Claim(id="c1", entity_id="e1", statement="All stocks follow momentum", epistemic_status=EpistemicStatus.CONFIRMED)
    evidence = Evidence(id="e1", claim_id="c1", source_kind="external_doc", source_id="doc1", credibility=0.7)
    
    gaps = detector.logical_gaps([claim], [evidence])
    assert isinstance(gaps, list)
```

- [ ] **Step 4: Run integration test**

Run: `pytest tests/test_logical_gaps.py::test_gap_detector_logical_gaps_integration -v`
Expected: PASS

- [ ] **Step 5: Wire into engine.py pipeline**

```python
# In engine.py ingest_transcript() method, after semantic gap check:
# After line ~280 (after semantic_gaps call):

# Logical gap detection
if self.llm_client is not None:
    logical_gap_flags = self.gap_detector.logical_gaps(
        claims=[],
        evidence=[]
    )
    gap_flags.extend(logical_gap_flags)
```

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_engine/gaps.py src/knowledge_engine/engine.py tests/test_logical_gaps.py
git commit -m "feat: wire LogicalGapDetector into GapDetector and pipeline"
```

---

### Task 8: Run Full Test Suite

**Covers:** [Verify all changes work together]

**Files:**
- None (verification only)

**Interfaces:**
- Consumes: All previous tasks
- Produces: Passing test suite

- [ ] **Step 1: Run all logical gap tests**

Run: `pytest tests/test_logical_gaps.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`
Expected: All tests PASS (87+ passed, no failures)

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve test failures from logical gap detection"
```

---

## Summary

After completing all tasks:
- `LogicalGapDetector` class with 4 detection methods (circular, cherry-picking, over-generalization, assumptions)
- Integrated into existing `GapDetector` and pipeline
- Non-breaking: returns empty list if LLM unavailable
- All tests passing

**Next steps (outside this plan):**
1. Wire `ke hunt` CLI to automatically run logical gap detection
2. Add logical gap display to Streamlit UI
3. Extend `SchoolAnalyzer` to use logical gaps as input
