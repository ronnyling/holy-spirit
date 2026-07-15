# Context-Aware Ingestion Testing Plan

## Overview

Comprehensive testing strategy for the Context-Aware Ingestion system, covering:
1. **Component Testing** - Unit tests for each new module
2. **Integration Testing** - Tests for module interactions
3. **E2E Testing** - Full user workflow simulation

## Testing Philosophy

**NO FALLBACKS**: All tests must verify explicit behavior. Silent failures = test failures.

---

## 1. Component Testing

### 1.1 TranscriptEvidenceDraft

| Test Case | Input | Expected Output | Type |
|-----------|-------|-----------------|------|
| Create with all fields | Full evidence draft | Successful creation | Best case |
| Create with minimal fields | Only required fields | Successful creation | Edge case |
| Invalid source quality | source_quality="invalid" | ValueError | Negative case |
| Confidence calculation | Known inputs | Expected confidence score | Best case |
| from_llm_output with missing fields | Partial LLM output | Use defaults | Edge case |
| from_llm_output with invalid data | Non-dict input | TypeError | Negative case |

### 1.2 ProvenanceChain

| Test Case | Input | Expected Output | Type |
|-----------|-------|-----------------|------|
| Create with full data | All fields populated | Successful creation | Best case |
| Create with empty lists | Empty evidence/document/transcript IDs | Successful creation | Edge case |
| to_dict serialization | Any chain | Valid dict | Best case |
| from_store_data with missing claim | Store data without claim | KeyError | Negative case |
| from_store_data with empty data | Empty store data | Empty chain | Edge case |

### 1.3 SemanticDeduplicator

| Test Case | Input | Expected Output | Type |
|-----------|-------|-----------------|------|
| Empty claims list | [] | [] | Edge case |
| Exact duplicates | Two identical claims | One duplicate found | Best case |
| Semantic duplicates | Similar but not identical claims | Duplicates found | Best case |
| No duplicates | Completely different claims | [] | Best case |
| Threshold boundary | Claims at threshold boundary | Edge behavior | Edge case |
| Single claim | One claim only | [] | Edge case |

### 1.4 ConflictDetector

| Test Case | Input | Expected Output | Type |
|-----------|-------|-----------------|------|
| Same-slot conflict | Claims with same slot, opposite keywords | Conflict detected | Best case |
| Keyword opposition | "buy" vs "sell" | Conflict detected | Best case |
| Different topics | Unrelated claims | No conflict | Best case |
| Same-slot, no opposition | Same slot, similar claims | No conflict | Edge case |
| Empty existing claims | No existing claims | [] | Edge case |
| Non-active status | Claims with UNKNOWN status | Skipped | Edge case |

### 1.5 CodeExtractors

| Test Case | Input | Expected Output | Type |
|-----------|-------|-----------------|------|
| Parse markdown structure | Markdown with headers | Correct sections | Best case |
| Parse plain text | No headers | Empty sections | Edge case |
| Extract metadata with dates | Text with years | Dates extracted | Best case |
| Extract metadata with URLs | Text with URLs | URLs extracted | Best case |
| Extract quantification | Text with percentages | has_percentage=True | Best case |
| Empty text | "" | Empty results | Edge case |

---

## 2. Integration Testing

### 2.1 TranscriptEvidence + ClaimDraft

| Test Case | Components | Expected Behavior | Type |
|-----------|------------|-------------------|------|
| Link evidence to claim | TranscriptEvidenceDraft → ClaimDraft | Evidence linked | Best case |
| Multiple evidence per claim | 3 evidence → 1 claim | All linked | Best case |
| Evidence with no claim | TranscriptEvidenceDraft without claim | Not linked | Edge case |

### 2.2 SemanticDedup + ConflictDetector

| Test Case | Components | Expected Behavior | Type |
|-----------|------------|-------------------|------|
| Dedup before conflict detection | Duplicates removed → conflict check | No duplicate conflicts | Best case |
| Conflict in duplicates | Duplicate claims that conflict | Conflict detected | Edge case |

### 2.3 ProvenanceChain + KnowledgeStore

| Test Case | Components | Expected Behavior | Type |
|-----------|------------|-------------------|------|
| Store claim + get provenance | Claim → Store → get_provenance | Chain returned | Best case |
| Claim with evidence | Claim + Evidence → Store | Evidence in chain | Best case |
| Nonexistent claim | Store → get_provenance(bad_id) | KeyError raised | Negative case |

### 2.4 CodeExtractors + TranscriptEvidence

| Test Case | Components | Expected Behavior | Type |
|-----------|------------|-------------------|------|
| Extract + enrich | MetadataExtractor → TranscriptEvidenceDraft | Metadata populated | Best case |
| Structure + evidence | DocumentStructureParser → Evidence | Sections extracted | Best case |

---

## 3. E2E Testing

### 3.1 Scenario: User Starts Knowledge Engine

**User Action**: Run `python main.py`

**Expected Flow**:
1. Neo4j starts successfully
2. Ollama starts successfully
3. HTTP server starts on port 8080
4. Streamlit UI starts on port 8501
5. All services report "[OK]" status

**Test Cases**:
- **Best case**: All services start successfully
- **Edge case**: Neo4j already running (skip start)
- **Negative case**: Neo4j path not found (error message)

### 3.2 Scenario: User Ingests Transcript via MCP

**User Action**: Call `ingest_transcript` tool

**Mock Data**:
```json
{
  "domain": "tcm",
  "entity_name": "turmeric",
  "transcript_text": "TCM studies show turmeric reduces inflammation by 40% in joint pain patients over 12 weeks.",
  "source_kind": "external_doc",
  "source_id": "study_001"
}
```

**Expected Flow**:
1. Transcript ingested successfully
2. Claims extracted with evidence
3. Provenance chain created
4. Domain classified as "tcm"
5. Entity created for "turmeric"

**Test Cases**:
- **Best case**: Full ingestion with claims and evidence
- **Edge case**: Empty transcript (no claims extracted)
- **Negative case**: Invalid domain (error raised)

### 3.3 Scenario: User Queries via Streamlit

**User Action**: Open Streamlit UI, ask question

**Mock Data**:
- Streamlit running on port 8501
- Knowledge base has existing claims

**Expected Flow**:
1. User types question
2. Intent classified
3. Claims retrieved
4. Response generated with provenance

**Test Cases**:
- **Best case**: Question answered with relevant claims
- **Edge case**: No matching claims (empty response)
- **Negative case**: Server not running (error message)

### 3.4 Scenario: User Uses Android APK

**User Action**: Open 6dfov app, take photo

**Mock Data**:
- APK connected to knowledge_engine
- Camera feed with text

**Expected Flow**:
1. Camera opens
2. OCR extracts text
3. Text sent to knowledge_engine
4. Claims created from OCR text
5. Response displayed in chat

**Test Cases**:
- **Best case**: Photo with text → claims created
- **Edge case**: Photo with no text (no claims)
- **Negative case**: Server not connected (error message)

---

## 4. Test Execution

### 4.1 Run Component Tests

```bash
cd knowledge_engine
python -m pytest tests/test_transcript_evidence.py tests/test_provenance.py tests/test_semantic_dedup.py tests/test_conflict_detector.py tests/test_code_extractors.py -v
```

### 4.2 Run Integration Tests

```bash
python -m pytest tests/test_provenance_tracking.py tests/test_claim_extractor_evidence.py -v
```

### 4.3 Run E2E Tests

```bash
# Start services
python main.py &

# Wait for services
sleep 10

# Run E2E test script
python tests/e2e_test.py

# Stop services
kill %1
```

---

## 5. Evaluation Criteria

### 5.1 Test Coverage

- **Component tests**: 100% of new modules
- **Integration tests**: All module interactions
- **E2E tests**: All user workflows

### 5.2 Test Quality

- **No silent failures**: All errors raised explicitly
- **Clear assertions**: Each test verifies one behavior
- **Independent tests**: No test depends on another

### 5.3 Performance

- **Component tests**: < 100ms each
- **Integration tests**: < 1s each
- **E2E tests**: < 30s each

---

## 6. Potential Issues

### 6.1 Known Limitations

1. **Evidence extraction not implemented**: Placeholder only
2. **LLM integration pending**: Requires LLM client
3. **Neo4j integration**: Provenance not persisted to graph

### 6.2 Risk Areas

1. **Performance**: Semantic dedup O(n²) complexity
2. **Scalability**: Large transcripts may timeout
3. **Accuracy**: Keyword opposition limited to predefined pairs

### 6.3 Mitigation

1. **Performance**: Add indexing for large datasets
2. **Scalability**: Implement chunked processing
3. **Accuracy**: Expand keyword pairs, add embedding-based detection
