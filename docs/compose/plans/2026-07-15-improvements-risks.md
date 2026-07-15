# Improvements & Risks Implementation Plan

## Overview

Implementation plan for all identified improvements and risks, with explicit weaknesses for each approach.

**Philosophy**: NO FALLBACKS. Every implementation must either succeed completely or fail loudly.

---

## 1. LLM-Based Evidence Extraction

### Current State
Placeholder only - `enrich_claims_with_evidence()` raises `EvidenceExtractionError`

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/claim_extractor_evidence.py` - Implement LLM extraction
- `src/knowledge_engine/extraction_prompt.py` - Add evidence extraction prompt
- `tests/test_claim_extractor_evidence.py` - Update tests

**Approach:**
```python
def enrich_claims_with_evidence(
    claims: List[ClaimDraft],
    transcript_text: str,
    llm_client: SupportsComplete
) -> List[ClaimDraft]:
    """Extract evidence from transcript for each claim."""
    
    for claim in claims:
        # Build evidence extraction prompt
        prompt = build_evidence_extraction_prompt(claim, transcript_text)
        
        # Call LLM
        raw_response = llm_client.complete_sync(
            system=EVIDENCE_SYSTEM_PROMPT,
            user=prompt,
            max_tokens=4000
        )
        
        # Parse response - NO FALLBACKS
        evidence_drafts = parse_evidence_response(raw_response)
        
        # Validate evidence - NO FALLBACKS
        validate_evidence_drafts(evidence_drafts, claim)
        
        # Link evidence to claim
        claim.evidence = evidence_drafts
    
    return claims
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **LLM hallucination** | Extracted evidence may be fabricated | Validate against transcript text, require source references |
| **Cost** | Each claim requires LLM call | Batch claims, use cheaper models for evidence extraction |
| **Latency** | 2-5s per claim | Parallel extraction, async processing |
| **Inconsistency** | Same transcript may extract different evidence | Temperature=0, deterministic parsing |
| **Context window** | Long transcripts may exceed limits | Chunk transcript, extract evidence per chunk |

---

## 2. Neo4j Provenance Persistence

### Current State
Provenance stored in-memory only, lost on restart

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/graph/neo4j_store.py` - Add provenance queries
- `src/knowledge_engine/provenance.py` - Add Neo4j integration

**Approach:**
```python
class Neo4jProvenanceStore:
    """Persist provenance chains to Neo4j."""
    
    def save_provenance(self, chain: ProvenanceChain) -> None:
        """Save provenance chain to Neo4j."""
        query = """
        MERGE (c:Claim {id: $claim_id})
        SET c.statement = $statement
        WITH c
        UNWIND $evidence_ids as ev_id
        MATCH (e:Evidence {id: ev_id})
        MERGE (c)-[:HAS_EVIDENCE]->(e)
        WITH c
        UNWIND $document_ids as doc_id
        MATCH (d:Document {id: doc_id})
        MERGE (c)-[:SOURCED_FROM]->(d)
        """
        
        self.session.run(query, 
            claim_id=chain.claim_id,
            statement=chain.claim_metadata.get("statement", ""),
            evidence_ids=chain.evidence_ids,
            document_ids=chain.document_ids
        )
    
    def get_provenance(self, claim_id: str) -> ProvenanceChain:
        """Retrieve provenance chain from Neo4j."""
        query = """
        MATCH (c:Claim {id: $claim_id})
        OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(e:Evidence)
        OPTIONAL MATCH (c)-[:SOURCED_FROM]->(d:Document)
        RETURN c, collect(e) as evidence, collect(d) as documents
        """
        
        result = self.session.run(query, claim_id=claim_id)
        record = result.single()
        
        if not record:
            raise KeyError(
                f"Claim '{claim_id}' not found in Neo4j. "
                "No silent fallback allowed."
            )
        
        return self._build_chain(record)
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Neo4j dependency** | System fails if Neo4j unavailable | Graceful degradation with clear error messages |
| **Query performance** | Complex traversals may be slow | Index on claim_id, evidence_id, document_id |
| **Data consistency** | Neo4j and in-memory may diverge | Single source of truth, transaction support |
| **Migration complexity** | Existing data needs migration | One-time migration script |
| **Cost** | Neo4j licensing for production | Use community edition for development |

---

## 3. Embedding-Based Semantic Dedup

### Current State
SequenceMatcher (character-level) - misses paraphrases

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/semantic_dedup.py` - Add embedding-based dedup
- `tests/test_semantic_dedup.py` - Update tests

**Approach:**
```python
class EmbeddingDeduplicator:
    """Semantic dedup using embeddings."""
    
    def __init__(self, embedding_client, threshold: float = 0.92):
        self.embedding_client = embedding_client
        self.threshold = threshold
        self.embeddings_cache: Dict[str, List[float]] = {}
    
    def find_duplicates(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find semantic duplicates using embeddings."""
        
        # Batch embed all claims
        statements = [c["statement"] for c in claims]
        embeddings = self.embedding_client.embed_batch(statements)
        
        # Cache embeddings
        for claim, embedding in zip(claims, embeddings):
            self.embeddings_cache[claim["id"]] = embedding
        
        # Find duplicates using cosine similarity
        duplicates = []
        seen = set()
        
        for i, claim_a in enumerate(claims):
            for j, claim_b in enumerate(claims):
                if i >= j:
                    continue
                
                pair_key = tuple(sorted([claim_a["id"], claim_b["id"]]))
                if pair_key in seen:
                    continue
                
                similarity = cosine_similarity(
                    self.embeddings_cache[claim_a["id"]],
                    self.embeddings_cache[claim_b["id"]]
                )
                
                if similarity >= self.threshold:
                    duplicates.append({
                        "claim_ids": [claim_a["id"], claim_b["id"]],
                        "similarity": similarity,
                        "statements": [claim_a["statement"], claim_b["statement"]]
                    })
                    seen.add(pair_key)
        
        return duplicates
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Embedding quality** | Poor embeddings = poor dedup | Use proven models (bge-m3), validate on domain data |
| **Threshold tuning** | Wrong threshold = over/under dedup | A/B testing, domain-specific thresholds |
| **Cost** | Embedding API calls | Cache embeddings, batch processing |
| **Latency** | Embedding computation | Async processing, pre-compute |
| **Model drift** | Embedding model updates break compatibility | Version embeddings, migration strategy |

---

## 4. Embedding-Based Conflict Detection

### Current State
SequenceMatcher + keyword opposition - misses semantic conflicts

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/conflict_detector.py` - Add embedding-based detection
- `tests/test_conflict_detector.py` - Update tests

**Approach:**
```python
class EmbeddingConflictDetector:
    """Detect conflicts using semantic similarity."""
    
    def __init__(self, embedding_client, similarity_threshold: float = 0.7):
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.opposition_pairs = self._load_opposition_pairs()
    
    def detect(
        self,
        new_claim: Dict[str, Any],
        existing_claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect semantic conflicts using embeddings."""
        
        # Embed new claim
        new_embedding = self.embedding_client.embed(new_claim["statement"])
        
        conflicts = []
        
        for existing in existing_claims:
            if existing.get("status") not in ["CONFIRMED", "DISPUTED"]:
                continue
            
            # Embed existing claim
            existing_embedding = self.embedding_client.embed(existing["statement"])
            
            # Calculate similarity
            similarity = cosine_similarity(new_embedding, existing_embedding)
            
            # Check for semantic opposition
            if similarity >= self.similarity_threshold:
                if self._detect_semantic_opposition(
                    new_claim["statement"],
                    existing["statement"]
                ):
                    conflicts.append({
                        "claim_id": new_claim["id"],
                        "conflicting_claim_id": existing["id"],
                        "conflict_type": "semantic_opposition",
                        "confidence": similarity
                    })
        
        return conflicts
    
    def _detect_semantic_opposition(self, text_a: str, text_b: str) -> bool:
        """Detect semantic opposition using LLM or rules."""
        # Use LLM to detect opposition
        prompt = f"""Are these two statements contradictory?

Statement A: {text_a}
Statement B: {text_b}

Answer ONLY with "yes" or "no"."""
        
        response = self.llm_client.complete_sync(
            system="You detect contradictions between statements.",
            user=prompt,
            max_tokens=10
        )
        
        return response.strip().lower() == "yes"
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **LLM cost** | Each conflict check requires LLM call | Batch checks, cache results |
| **Latency** | LLM inference adds delay | Async processing, pre-compute |
| **False positives** | Similar but not contradictory | Tune thresholds, human review |
| **Context dependency** | May miss conditional contradictions | Extract conditions, check context |
| **Scalability** | O(n) LLM calls per new claim | Index embeddings, approximate nearest neighbor |

---

## 5. Chunked Processing for Large Transcripts

### Current State
Single-pass processing - may timeout on large files

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/extraction.py` - Add chunked processing
- `src/knowledge_engine/chunking.py` - Enhance chunking logic

**Approach:**
```python
class ChunkedProcessor:
    """Process large transcripts in chunks."""
    
    def __init__(self, chunk_size: int = 5000, overlap: int = 500):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process_transcript(
        self,
        transcript: str,
        processor: Callable[[str], Any]
    ) -> List[Any]:
        """Process transcript in chunks."""
        
        chunks = self._create_chunks(transcript)
        results = []
        
        for i, chunk in enumerate(chunks):
            try:
                result = processor(chunk)
                results.append(result)
            except Exception as e:
                # NO FALLBACKS: Log error, don't silently skip
                logger.error(f"Chunk {i} processing failed: {e}")
                raise
        
        return results
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks from text."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Context loss** | Chunks may miss cross-chunk context | Overlap chunks, post-process to merge |
| **Deduplication** | Same claim may appear in multiple chunks | Cross-chunk dedup, confidence scoring |
| **Complexity** | More code to maintain | Clear interfaces, comprehensive tests |
| **Performance** | More processing overhead | Parallel chunk processing |
| **Consistency** | Results may vary by chunk boundary | Deterministic chunking, fixed boundaries |

---

## 6. Expanded Keyword Opposition Pairs

### Current State
Limited predefined pairs (buy/sell, increase/decrease)

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/conflict_detector.py` - Expand keyword pairs
- `knowledge_engine/keyword_pairs.json` - External configuration

**Approach:**
```json
{
  "trading": {
    "buy": "sell",
    "long": "short",
    "bullish": "bearish",
    "call": "put",
    "up": "down",
    "rise": "fall",
    "increase": "decrease",
    "positive": "negative",
    "overvalued": "undervalued",
    "entry": "exit"
  },
  "tcm": {
    "hot": "cold",
    "tonify": "sedate",
    "excess": "deficiency",
    "yang": "yin",
    "exterior": "interior",
    "heat": "cold"
  },
  "real_estate": {
    "buy": "sell",
    "appreciate": "depreciate",
    "rent": "own",
    "residential": "commercial",
    "urban": "suburban"
  }
}
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Domain specificity** | Pairs may not generalize | Domain-specific configurations |
| **Maintenance** | Pairs need updating | Community contributions, LLM-suggested pairs |
| **Context dependence** | "up" may not always oppose "down" | Context-aware matching |
| **Coverage gaps** | Missing important pairs | Regular review, user feedback |
| **Ambiguity** | Some pairs are not true opposites | Human validation, confidence scoring |

---

## 7. Async Processing Pipeline

### Current State
Synchronous processing - blocks on LLM calls

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/engine.py` - Add async methods
- `src/knowledge_engine/extraction.py` - Async extraction
- `src/knowledge_engine/embedding.py` - Async embedding

**Approach:**
```python
class AsyncKnowledgeEngine:
    """Async version of KnowledgeEngine."""
    
    async def ingest_transcript_async(
        self,
        transcript: TranscriptInput
    ) -> TranscriptOutcome:
        """Ingest transcript asynchronously."""
        
        # Parallel extraction
        extraction_tasks = [
            self._extract_claims_async(chunk)
            for chunk in self._chunk_transcript(transcript.transcript_text)
        ]
        claim_lists = await asyncio.gather(*extraction_tasks)
        claims = [claim for sublist in claim_lists for claim in sublist]
        
        # Parallel embedding
        embedding_tasks = [
            self._embed_claim_async(claim)
            for claim in claims
        ]
        await asyncio.gather(*embedding_tasks)
        
        # Parallel conflict detection
        conflict_tasks = [
            self._detect_conflicts_async(claim)
            for claim in claims
        ]
        conflicts = await asyncio.gather(*conflict_tasks)
        
        return self._build_outcome(claims, conflicts)
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Complexity** | Async code harder to debug | Clear error handling, comprehensive logging |
| **Resource contention** | Too many parallel tasks | Rate limiting, connection pooling |
| **Ordering** | Results may arrive out of order | Task tracking, result assembly |
| **Error handling** | Async exceptions harder to catch | Structured error handling, retries |
| **Testing** | Async tests more complex | pytest-asyncio, mock async clients |

---

## 8. Human-in-the-Loop Review

### Current State
Auto-resolution for low-stacks conflicts only

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/engine.py` - Add review queue
- `src/knowledge_engine/models.py` - Add review status
- `app.py` - Add Streamlit review UI

**Approach:**
```python
class ReviewQueue:
    """Queue for human review of conflicts."""
    
    def __init__(self, store: KnowledgeStore):
        self.store = store
    
    def add_for_review(
        self,
        claim_id: str,
        conflict_id: str,
        reason: str
    ) -> ReviewItem:
        """Add claim to review queue."""
        
        item = ReviewItem(
            id=uuid4().hex,
            claim_id=claim_id,
            conflict_id=conflict_id,
            reason=reason,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
        
        self.store.add_review_item(item)
        return item
    
    def approve(self, item_id: str, reviewer: str, decision: str) -> None:
        """Approve or reject review item."""
        
        item = self.store.get_review_item(item_id)
        
        if item.status != "pending":
            raise ValueError(
                f"Review item {item_id} is not pending. "
                f"Current status: {item.status}"
            )
        
        item.status = decision  # "approved" or "rejected"
        item.reviewer = reviewer
        item.reviewed_at = datetime.now(timezone.utc)
        
        # Apply decision
        if decision == "approved":
            self._approve_claim(item.claim_id)
        elif decision == "rejected":
            self._reject_claim(item.claim_id)
        else:
            raise ValueError(f"Invalid decision: {decision}")
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Human bottleneck** | Review queue may grow | Prioritization, batch review |
| **Inconsistency** | Different reviewers may decide differently | Clear guidelines, review history |
| **Latency** | Delays in knowledge promotion | Auto-approve low-risk, escalate high-risk |
| **Scalability** | Doesn't scale with knowledge growth | ML-assisted review, confidence thresholds |
| **Fatigue** | Reviewers may make errors | Rotation, breaks, decision support |

---

## 9. Confidence Decay Over Time

### Current State
Confidence scores static after creation

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/models.py` - Add decay parameters
- `src/knowledge_engine/engine.py` - Add decay calculation

**Approach:**
```python
class ConfidenceDecay:
    """Calculate confidence decay over time."""
    
    def __init__(
        self,
        half_life_days: float = 365,  # 1 year half-life
        min_confidence: float = 0.1
    ):
        self.half_life_days = half_life_days
        self.min_confidence = min_confidence
    
    def calculate_decay(
        self,
        original_confidence: float,
        created_at: datetime,
        current_time: datetime = None
    ) -> float:
        """Calculate confidence with time decay."""
        
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        days_elapsed = (current_time - created_at).days
        decay_factor = 0.5 ** (days_elapsed / self.half_life_days)
        
        decayed_confidence = original_confidence * decay_factor
        
        return max(decayed_confidence, self.min_confidence)
    
    def apply_decay_to_claims(
        self,
        claims: List[Claim]
    ) -> List[Claim]:
        """Apply decay to all claims."""
        
        for claim in claims:
            if claim.confidence_score:
                claim.confidence_score = self.calculate_decay(
                    claim.confidence_score,
                    claim.created_at
                )
        
        return claims
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Arbitrary parameters** | Half-life may not fit all domains | Domain-specific decay rates |
| **Information loss** | Old but valid knowledge loses confidence | Reinforcement mechanism |
| **Complexity** | More parameters to tune | Sensitivity analysis, A/B testing |
| **Performance** | Decay calculation on every query | Cache decayed values, batch updates |
| **User confusion** | Confidence changes without visible reason | Explain decay in UI, show original confidence |

---

## 10. Batch Ingestion

### Current State
Single transcript processing only

### Implementation Plan

**Files to modify:**
- `src/knowledge_engine/engine.py` - Add batch method
- `src/knowledge_engine/extraction.py` - Batch extraction

**Approach:**
```python
class BatchIngestor:
    """Batch ingestion for multiple transcripts."""
    
    def __init__(self, engine: KnowledgeEngine):
        self.engine = engine
    
    def ingest_batch(
        self,
        transcripts: List[TranscriptInput],
        max_concurrent: int = 5
    ) -> List[TranscriptOutcome]:
        """Ingest multiple transcripts in batch."""
        
        outcomes = []
        
        # Process in batches
        for i in range(0, len(transcripts), max_concurrent):
            batch = transcripts[i:i + max_concurrent]
            
            # Parallel processing
            batch_outcomes = []
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = [
                    executor.submit(self.engine.ingest_transcript, t)
                    for t in batch
                ]
                
                for future in as_completed(futures):
                    try:
                        outcome = future.result()
                        batch_outcomes.append(outcome)
                    except Exception as e:
                        # NO FALLBACKS: Log error, don't skip
                        logger.error(f"Batch ingestion failed: {e}")
                        raise
            
            outcomes.extend(batch_outcomes)
        
        return outcomes
```

### Weaknesses

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Resource contention** | Too many concurrent requests | Rate limiting, connection pooling |
| **Error handling** | One failure may affect others | Isolate failures, partial success reporting |
| **Deduplication** | Cross-transcript dedup needed | Pre-batch dedup, post-batch validation |
| **Ordering** | Results may arrive out of order | Batch tracking, result assembly |
| **Monitoring** | Hard to track batch progress | Progress callbacks, logging |

---

## Implementation Priority

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **P0** | LLM Evidence Extraction | High | High |
| **P1** | Neo4j Provenance Persistence | Medium | High |
| **P1** | Expanded Keyword Pairs | Low | Medium |
| **P2** | Embedding-Based Dedup | Medium | Medium |
| **P2** | Embedding-Based Conflict Detection | Medium | Medium |
| **P3** | Chunked Processing | Medium | Medium |
| **P3** | Async Processing | High | Medium |
| **P4** | Human-in-the-Loop Review | High | Low |
| **P4** | Confidence Decay | Low | Low |
| **P5** | Batch Ingestion | Medium | Low |

---

## Risk Mitigation Summary

| Risk | Mitigation Strategy |
|------|---------------------|
| **LLM Hallucination** | Validate against transcript, require source references |
| **Performance Degradation** | Chunked processing, async, caching |
| **Data Integrity** | No-fallbacks policy, explicit error handling |
| **Scalability** | Embedding indexing, batch processing |
| **Model Drift** | Version embeddings, migration strategy |
| **Cost Overrun** | Caching, cheaper models for non-critical tasks |
