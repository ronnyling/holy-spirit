# Ingestion & Housekeeping

Every transcript is *harboured* before any claim processing. Harbouring is fully deterministic, offline,
and has no fallbacks — it never calls an LLM and never guesses.

## Steps

1. **Normalize** — unify line endings, collapse runs of blank lines. All offsets are recorded against
   the normalized text.
2. **Hash** — SHA-256 of the normalized content. This is the idempotency key.
3. **Chunk** — `TranscriptChunker` produces structure-aware, bounded, overlapping chunks.
4. **Document** — render a human-readable markdown document (metadata + chunk map) under `documents/`.
5. **Register** — record raw text, generated document, and metadata in a filesystem-backed manifest.

## Idempotent housekeeping

Re-ingesting a transcript with identical normalized content does **not** create a duplicate. The registry
recognises the content hash and returns the existing record (`created = False`). This keeps the corpus
clean automatically as transcripts are re-dropped or re-processed.

```python
from knowledge_engine import TranscriptRegistry, KnowledgeEngine, TranscriptInput

registry = TranscriptRegistry("./data")
engine = KnowledgeEngine(registry=registry)

outcome = engine.ingest_transcript(TranscriptInput(...))
outcome.transcript_id       # stable id: {domain}-{entity}-{hash12}
outcome.transcript_created  # False if identical content was already harboured
outcome.chunk_count
```

## Chunker configuration

`TranscriptChunker(max_chars=1200, overlap_chars=150, min_chars=200)`:

- Paragraph- and sentence-aware; packs sentences into windows up to `max_chars`.
- Carries an `overlap_chars` prefix (on a word boundary) between adjacent chunks.
- Hard-splits any single sentence that exceeds `max_chars`.
- Invalid configuration raises `ValueError` — there is no silent correction.

Defaults are configurable via `KE_CHUNK_MAX_CHARS`, `KE_CHUNK_OVERLAP_CHARS`, and `KE_CHUNK_MIN_CHARS`.

## On-disk layout

```
<data-dir>/
  transcripts/   raw normalized transcript text
  documents/     generated markdown documentation, one per transcript
  manifest.json  registry of all harboured transcripts
```

## Tests

- `tests/test_chunking.py` — chunk bounds, overlap, hard-split, invalid config, normalization.
- `tests/test_registry.py` — harbouring writes, idempotency, manifest reload, empty-transcript rejection.
- `tests/test_ingest_harbour.py` — end-to-end: the engine harbours, documents, and housekeeps on ingest.
