"""Knowledge Engine CLI — ingest transcripts and ask research questions.

STORAGE & SEMANTIC SEARCH
  The engine funnel runs on the in-memory KnowledgeStore, persisted to a local
  JSON file (--state) so an ingested corpus survives across `ingest` and `ask`.
  When embeddings (Ollama bge-m3) and Neo4j are configured, every new claim is
  embedded at ingest and mirrored into Neo4j's native vector index, so
  `ask --query` performs true semantic search. If either is unavailable the CLI
  degrades loudly to keyword matching; the JSON store stays the source of truth.

USAGE
  # Ingest one file or a whole folder of .txt transcripts
  python scripts/ke.py ingest data/uat_transcripts --domain "real estate"
  python scripts/ke.py ingest data/uat_transcripts/deal.txt --domain trading --entity "Breakout Rules"

  # Ask questions
  python scripts/ke.py ask --entity "Cap Rate Rules"
  python scripts/ke.py ask --domain "real estate"
  python scripts/ke.py ask --query "cap rate suburban"      # semantic search (keyword fallback)
  python scripts/ke.py ask --claim <claim_id>

  # State snapshot
  python scripts/ke.py snapshot

TRANSCRIPT FOLDER CONVENTION
  data/uat_transcripts/
    cap_rate.txt              # raw transcript text (required)
    cap_rate.meta.json        # optional: {domain, entity_name, source_kind, source_id, source_ref}
    cap_rate.claims.json      # optional: [{statement, observed_slots?, slot_name?, evidence?}]

  - If cap_rate.claims.json exists, those hand-authored claims are used.
  - Else, if the LLM (KE_MIMO_API_KEY) is valid, claims are extracted from the
    transcript text automatically.
  - Else, the transcript yields zero claims (nothing to ingest).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import argparse
import contextlib
import itertools
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# Make `src/` importable when run as `python scripts/ke.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knowledge_engine.bootstrap import load_dotenv  # noqa: E402
from knowledge_engine.chunking import TranscriptChunker  # noqa: E402
from knowledge_engine.classification import DomainClassifier  # noqa: E402
from knowledge_engine.contracts import TranscriptInput  # noqa: E402
from knowledge_engine.embeddings import EmbeddingClient  # noqa: E402
from knowledge_engine.engine import KnowledgeEngine  # noqa: E402
from knowledge_engine.extraction import ClaimExtractor  # noqa: E402
from knowledge_engine.graph.neo4j_store import KnowledgeGraphStore  # noqa: E402
from knowledge_engine.llm import MiMoClient  # noqa: E402
from knowledge_engine.policy import get_domain_policy  # noqa: E402
from knowledge_engine.store import KnowledgeStore  # noqa: E402

DEFAULT_STATE = "data/uat_state.json"

# Shared normalizer so the dedup content-hash matches the registry's definition
# (normalize -> SHA-256), keeping a future migration to the filesystem registry
# consistent.
_NORMALIZER = TranscriptChunker()


@dataclass(slots=True)
class _PreparedIngest:
    index: int
    txt_path: Path
    digest: str
    status: str
    transcript: TranscriptInput | None = None
    how: str = ""
    notes: list[str] = field(default_factory=list)
    detail: str = ""


def _build_engine(
    state_path: str,
) -> tuple[KnowledgeEngine, DomainClassifier | None]:
    """Build the engine, preferring Neo4j as the primary store when available.

    When both KE_NEO4J_URI and embedding vars are set and Neo4j is reachable,
    the engine uses KnowledgeGraphStore as its store — all ingest writes go
    directly to the graph (no JSON save, no secondary sync step).  If Neo4j is
    unavailable, falls back to the JSON-backed KnowledgeStore with loud stderr
    output so the degraded path is never silent.

    Check ``isinstance(engine.store, KnowledgeGraphStore)`` at call sites to
    know which path is active.
    """
    load_dotenv()
    mimo = MiMoClient.from_env()
    extractor = ClaimExtractor(mimo) if mimo is not None else None
    classifier = DomainClassifier(mimo) if mimo is not None else None
    if mimo is None:
        print(
            "note: KE_MIMO_API_KEY not set — LLM extraction & auto-classification off; "
            "using claim_drafts + explicit/folder domains only.",
            file=sys.stderr,
        )

    embed = EmbeddingClient.from_env()
    if embed is None:
        print(
            "note: embeddings unconfigured (KE_EMBEDDING_*) — semantic search off, keyword only.",
            file=sys.stderr,
        )

    uri = os.environ.get("KE_NEO4J_URI", "")
    if embed is not None and uri:
        try:
            graph = KnowledgeGraphStore(
                uri=uri,
                user=os.environ.get("KE_NEO4J_USER", ""),
                password=os.environ.get("KE_NEO4J_PASSWORD", ""),
                database=os.environ.get("KE_NEO4J_DATABASE", "neo4j"),
                embedding_dimensions=embed.dimensions,
            )
            graph.verify()
            graph.apply_schema()
            print(
                "note: Neo4j is the primary store — all writes go directly to the graph.",
                file=sys.stderr,
            )
            return KnowledgeEngine(store=graph, embedding_client=embed, extractor=extractor), classifier
        except Exception as exc:
            print(
                f"note: Neo4j unavailable ({type(exc).__name__}: {exc})"
                " — falling back to JSON store.",
                file=sys.stderr,
            )

    # JSON store fallback (keeps existing behaviour for offline / test runs).
    store = KnowledgeStore.load(state_path)
    return KnowledgeEngine(store=store, embedding_client=embed, extractor=extractor), classifier


def _open_vector_backend_for_ask() -> tuple[KnowledgeGraphStore | None, EmbeddingClient | None]:
    """Open a read-only Neo4j+embedding handle for `ask --query` when the engine
    is running on the JSON store fallback.  Not needed when Neo4j is primary.
    """
    load_dotenv()
    embed = EmbeddingClient.from_env()
    if embed is None:
        print(
            "note: embeddings unconfigured — semantic search off, keyword only.",
            file=sys.stderr,
        )
        return None, None
    uri = os.environ.get("KE_NEO4J_URI", "")
    if not uri:
        print("note: KE_NEO4J_URI unset — vector index off, keyword search only.", file=sys.stderr)
        return None, embed
    try:
        graph = KnowledgeGraphStore(
            uri=uri,
            user=os.environ.get("KE_NEO4J_USER", ""),
            password=os.environ.get("KE_NEO4J_PASSWORD", ""),
            database=os.environ.get("KE_NEO4J_DATABASE", "neo4j"),
            embedding_dimensions=embed.dimensions,
        )
        graph.verify()
        graph.apply_schema()
        return graph, embed
    except Exception as exc:
        print(
            f"note: Neo4j unreachable ({type(exc).__name__}: {exc}) — vector index off, keyword only.",
            file=sys.stderr,
        )
        return None, embed


def _default_ingest_workers(total_files: int) -> int:
    configured = os.environ.get("KE_INGEST_WORKERS", "5")
    try:
        workers = int(configured)
    except ValueError:
        workers = 5
    if total_files <= 1:
        return 1
    return max(1, min(workers, total_files))


def _domain_attention_message(how: str, txt_path: Path) -> str:
    if how == "no-classifier":
        return (
            f"{txt_path.name}: no domain and LLM classifier off — set KE_MIMO_API_KEY, "
            "pass --domain, or use a domain subfolder"
        )
    return (
        f"{txt_path.name}: domain UNKNOWN — pass --domain, add a .meta.json domain, "
        "or move into a domain subfolder"
    )


def _prepare_ingest_item(
    index: int,
    txt_path: Path,
    args: argparse.Namespace,
    ingested_hashes: set[str],
    classifier: DomainClassifier | None,
    extractor: ClaimExtractor | None,
) -> _PreparedIngest:
    text = txt_path.read_text(encoding="utf-8")
    digest = _content_hash(text)
    if digest in ingested_hashes:
        return _PreparedIngest(
            index=index,
            txt_path=txt_path,
            digest=digest,
            status="skipped",
            detail=f"already ingested ({digest[:12]})",
        )

    meta = _load_meta(txt_path, args)
    try:
        domain, how = _resolve_domain(
            txt_path, args, meta["_meta_domain"], meta["entity_name"], text, classifier
        )
    except Exception as exc:  # classification LLM error — never guess a domain
        return _PreparedIngest(
            index=index,
            txt_path=txt_path,
            digest=digest,
            status="failed",
            detail=f"domain classification error: {exc}",
        )

    if domain is None:
        return _PreparedIngest(
            index=index,
            txt_path=txt_path,
            digest=digest,
            status="needs_domain",
            detail=_domain_attention_message(how, txt_path),
        )

    try:
        claim_drafts = _load_claims(txt_path)
        notes: list[str] = []
        if not claim_drafts and extractor is not None:
            claim_drafts = extractor.extract(
                domain=domain,
                entity_name=meta["entity_name"],
                transcript_text=text,
            )
            notes.append(
                f"LLM extracted {len(claim_drafts)} claim(s) from transcript text (Unverified until evidence)"
            )

        transcript = TranscriptInput(
            transcript_text=text,
            claim_drafts=claim_drafts,
            domain=domain,
            entity_name=meta["entity_name"],
            source_kind=meta["source_kind"],
            source_id=meta["source_id"],
            source_ref=meta["source_ref"],
        )
    except Exception as exc:
        return _PreparedIngest(
            index=index,
            txt_path=txt_path,
            digest=digest,
            status="failed",
            detail=f"{type(exc).__name__}: {exc}",
        )

    return _PreparedIngest(
        index=index,
        txt_path=txt_path,
        digest=digest,
        status="ready",
        transcript=transcript,
        how=how,
        notes=notes,
    )


def _prepare_ingest_batch(
    files: list[Path],
    args: argparse.Namespace,
    ingested_hashes: set[str],
    classifier: DomainClassifier | None,
    extractor: ClaimExtractor | None,
    workers: int,
) -> list[_PreparedIngest]:
    if workers <= 1 or len(files) <= 1:
        return [
            _prepare_ingest_item(i, txt_path, args, ingested_hashes, classifier, extractor)
            for i, txt_path in enumerate(files, 1)
        ]

    prepared: list[_PreparedIngest | None] = [None] * len(files)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _prepare_ingest_item,
                i,
                txt_path,
                args,
                ingested_hashes,
                classifier,
                extractor,
            ): i
            for i, txt_path in enumerate(files, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            prepared[result.index - 1] = result

    return [item for item in prepared if item is not None]


def _sync_to_graph(
    graph: KnowledgeGraphStore, store: KnowledgeStore, entity_id: str, claim_ids: list[str]
) -> int:
    """Mirror entity and embedded claims into the Neo4j vector index (JSON-store path only).

    Used only when KnowledgeStore (JSON) is the primary store and Neo4j is the
    secondary vector mirror.  When KnowledgeGraphStore is primary this step is
    skipped — the engine writes directly to the graph.
    """
    entity = store.entities.get(entity_id)
    if entity is None:
        return 0
    graph.upsert_entity(entity)
    mirrored = 0
    for cid in claim_ids:
        claim = store.claims.get(cid)
        if claim is None or not claim.embedding:
            continue
        graph.add_claim(claim)
        mirrored += 1
    return mirrored


# -- metadata / claim loading --------------------------------------------------


def _load_meta(txt_path: Path, args: argparse.Namespace) -> dict:
    meta_path = txt_path.with_suffix(".meta.json")
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "entity_name": meta.get("entity_name") or args.entity or txt_path.stem.replace("_", " ").title(),
        "source_kind": meta.get("source_kind") or args.source_kind,
        "source_id": meta.get("source_id") or txt_path.stem,
        "source_ref": meta.get("source_ref") or str(txt_path),
        # Explicit domain from the sidecar only; folder/LLM fallbacks happen in
        # _resolve_domain so an UNKNOWN file can be flagged rather than guessed.
        "_meta_domain": meta.get("domain"),
    }


def _known_domain(candidate: str | None) -> str | None:
    """Return the candidate iff it maps to a real (non-default) domain policy."""
    if not candidate or not candidate.strip():
        return None
    return candidate if get_domain_policy(candidate).name != "default" else None


def _resolve_domain(
    txt_path: Path,
    args: argparse.Namespace,
    meta_domain: str | None,
    entity_name: str,
    text: str,
    classifier: DomainClassifier | None,
) -> tuple[str | None, str]:
    """Hybrid domain resolution. Returns (domain | None, how).

    Precedence: explicit .meta.json domain -> --domain -> known-domain subfolder
    -> LLM classification -> UNKNOWN (None). UNKNOWN is never defaulted; the
    caller flags the file for a human, preserving the per-domain evidence gate.
    """
    if meta_domain and meta_domain.strip():
        return meta_domain, "meta"
    if args.domain and args.domain.strip():
        return args.domain, "cli"
    folder = _known_domain(txt_path.parent.name)
    if folder:
        return folder, "folder"
    if classifier is not None:
        classified = classifier.classify(transcript_text=text, entity_name=entity_name)
        if classified:
            return classified, "llm"
        # Strict classifier returned UNKNOWN — try open classification to handle
        # novel domains rather than dropping the file.
        from knowledge_engine.policy import get_domain_policy, register_domain
        open_domain = classifier.classify_open(transcript_text=text, entity_name=entity_name)
        if open_domain:
            if get_domain_policy(open_domain).name == "default":
                register_domain(open_domain)
                print(
                    f"note: new domain '{open_domain}' registered with default evidence bars.",
                    file=sys.stderr,
                )
            return open_domain, "llm-open"
        return None, "unknown"
    return None, "no-classifier"


# -- dedup ledger (content-hash, recorded only on successful ingest) -----------


def _content_hash(text: str) -> str:
    return sha256(_NORMALIZER.normalize(text).encode("utf-8")).hexdigest()


def _ingested_path(state_path: str) -> Path:
    p = Path(state_path)
    return p.parent / f"{p.stem}.ingested.json"


def _load_ingested(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("ingested", {}) if isinstance(data, dict) else {}


def _save_ingested(path: Path, ingested: dict) -> None:
    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ingested": ingested,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_claims(txt_path: Path) -> list[dict]:
    claims_path = txt_path.with_suffix(".claims.json")
    if not claims_path.is_file():
        return []
    data = json.loads(claims_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{claims_path}: expected a JSON array of claim drafts")
    return data


def _iter_transcripts(target: Path):
    if target.is_file():
        yield target
    elif target.is_dir():
        # Recurse so a single drop-folder can hold per-domain subfolders.
        yield from sorted(target.rglob("*.txt"))
    else:
        raise SystemExit(f"not found: {target}")


# -- commands ------------------------------------------------------------------


@contextlib.contextmanager
def _progress(label: str):
    """Show a live cue while a slow (LLM) step runs, so it never looks frozen.

    On a TTY: animates a spinner + elapsed seconds on stderr, then clears the
    line. When output is redirected/piped: prints plain start/done lines so log
    files stay readable. stderr keeps stdout's result blocks clean.
    """
    start = time.monotonic()
    if not sys.stderr.isatty():
        print(f"{label} ...", file=sys.stderr, flush=True)
        try:
            yield
        finally:
            print(f"{label} done in {time.monotonic() - start:.1f}s", file=sys.stderr, flush=True)
        return

    stop = threading.Event()
    frames = itertools.cycle("|/-\\")

    def _spin() -> None:
        while not stop.wait(0.1):
            sys.stderr.write(f"\r{label} {next(frames)} {time.monotonic() - start:4.1f}s")
            sys.stderr.flush()

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        sys.stderr.write("\r" + " " * (len(label) + 16) + "\r")
        sys.stderr.flush()


def cmd_ingest(args: argparse.Namespace) -> None:
    engine, classifier = _build_engine(args.state)
    is_graph_primary = isinstance(engine.store, KnowledgeGraphStore)

    # When Neo4j is NOT primary, open a separate graph handle for vector mirroring.
    graph_mirror: KnowledgeGraphStore | None = None
    if not is_graph_primary:
        embed = engine.embedding_client
        if embed is not None:
            uri = os.environ.get("KE_NEO4J_URI", "")
            if uri:
                try:
                    graph_mirror = KnowledgeGraphStore(
                        uri=uri,
                        user=os.environ.get("KE_NEO4J_USER", ""),
                        password=os.environ.get("KE_NEO4J_PASSWORD", ""),
                        database=os.environ.get("KE_NEO4J_DATABASE", "neo4j"),
                        embedding_dimensions=embed.dimensions,
                    )
                    graph_mirror.verify()
                    graph_mirror.apply_schema()
                except Exception as exc:
                    print(
                        f"note: Neo4j mirror unavailable ({type(exc).__name__}: {exc})"
                        " — vector index disabled for this run.",
                        file=sys.stderr,
                    )
                    graph_mirror = None
    files = list(_iter_transcripts(Path(args.path)))
    if not files:
        raise SystemExit(f"no .txt transcripts found in {args.path}")

    ingested_path = _ingested_path(args.state)
    ingested = _load_ingested(ingested_path)
    workers = _default_ingest_workers(len(files))
    if workers > 1:
        print(
            f"note: preparing transcripts in parallel (workers={workers})",
            file=sys.stderr,
        )

    prepared = _prepare_ingest_batch(
        files,
        args,
        set(ingested),
        classifier,
        engine.extractor,
        workers,
    )

    total = len(files)
    done = skipped = flagged = failed = 0
    attention: list[str] = []

    for prep in prepared:
        if prep.status == "skipped":
            skipped += 1
            print(
                f"[{prep.index}/{total}] {prep.txt_path.name} — already ingested ({prep.digest[:12]}); skipped",
                file=sys.stderr,
            )
            continue

        if prep.status == "needs_domain":
            flagged += 1
            attention.append(f"  NEEDS DOMAIN  {prep.txt_path.name}: {prep.detail}")
            continue

        if prep.status == "failed":
            failed += 1
            attention.append(f"  FAILED        {prep.txt_path.name}: {prep.detail}")
            continue

        if prep.transcript is None:
            failed += 1
            attention.append(f"  FAILED        {prep.txt_path.name}: transcript preparation returned no payload")
            continue

        try:
            with _progress(
                f"[{prep.index}/{total}] {prep.txt_path.name} — committing transcript ({prep.transcript.domain}, via {prep.how})"
            ):
                outcome = engine.ingest_transcript(prep.transcript)
        except Exception as exc:  # LLM/pipeline failure — surface loudly, don't record
            failed += 1
            attention.append(f"  FAILED        {prep.txt_path.name}: {type(exc).__name__}: {exc}")
            continue

        # Success — record in dedup ledger.
        ingested[prep.digest] = {
            "file": prep.txt_path.name,
            "domain": prep.transcript.domain,
            "domain_source": prep.how,
            "entity_id": outcome.entity_id,
            "claims": len(outcome.claim_ids),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        # JSON store path: save state file after every successful ingest so a
        # crash only loses the in-progress file, not all previous work.
        if not is_graph_primary:
            engine.store.save(args.state)
        _save_ingested(ingested_path, ingested)
        done += 1

        print(f"\n=== {prep.txt_path.name} -> entity {outcome.entity_id} ===")
        print(f"  domain          : {prep.transcript.domain} (via {prep.how})")
        print(f"  claims ingested : {len(outcome.claim_ids)}")
        print(f"  confirmed       : {len(outcome.confirmed_claim_ids)}")
        print(f"  unverified      : {len(outcome.unverified_claim_ids)}")
        print(f"  disputed        : {len(outcome.disputed_claim_ids)}")
        print(f"  gaps flagged    : {len(outcome.gap_flags)}")
        print(f"  conflicts       : {len(outcome.conflict_summaries)}")
        for s in outcome.slot_suggestions:
            print(f"  slot suggestion : {s.slot_name} {s.current_lifecycle}->{s.suggested_lifecycle}")
        for note in prep.notes:
            print(f"  note            : {note}")
        for note in outcome.notes:
            print(f"  note            : {note}")

        if is_graph_primary:
            print(f"  store           : written to Neo4j graph (primary)")
        elif graph_mirror is not None:
            try:
                mirrored = _sync_to_graph(graph_mirror, engine.store, outcome.entity_id, outcome.claim_ids)
                if mirrored:
                    print(f"  vector index    : mirrored {mirrored} claim(s) -> Neo4j")
            except Exception as exc:
                print(
                    f"  WARNING         : Neo4j mirror failed ({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )

    # Always persist ledger; persist JSON store only on the fallback path.
    if not is_graph_primary:
        engine.store.save(args.state)
    _save_ingested(ingested_path, ingested)

    print(
        f"\nsummary: {done} ingested, {skipped} already-ingested, "
        f"{flagged} need a domain, {failed} failed  (of {total} file(s))"
    )
    if attention:
        print("files needing attention:")
        for line in attention:
            print(line)
    if is_graph_primary:
        print("store -> Neo4j graph (primary, writes committed directly)")
    else:
        print(f"state saved -> {args.state}")
    print(f"ledger saved -> {ingested_path}")
    if graph_mirror is not None:
        print("vector index -> Neo4j mirror (semantic search ready via `ask --query`)")
        graph_mirror.close()
    elif is_graph_primary:
        print("vector index -> Neo4j graph (semantic search ready via `ask --query`)")


def _entity_view(store: KnowledgeStore, entity) -> dict:
    claims = [c for c in store.claims.values() if c.entity_id == entity.id]
    slots = [s for s in store.slots.values() if s.entity_id == entity.id]
    return {
        "entity": entity.canonical_name,
        "entity_id": entity.id,
        "claims": [
            {
                "id": c.id,
                "statement": c.statement,
                "status": str(c.epistemic_status),
                "slot": c.slot_name,
                "tags": c.tags,
            }
            for c in claims
        ],
        "slots": [
            {"name": s.name, "lifecycle": str(s.lifecycle), "observed": s.observed_count}
            for s in slots
        ],
    }


def cmd_ask(args: argparse.Namespace) -> None:
    engine, _ = _build_engine(args.state)
    is_graph = isinstance(engine.store, KnowledgeGraphStore)
    # JSON-store fallback for non-semantic queries.
    json_store = engine.store if not is_graph else None

    if args.entity:
        if is_graph:
            result = engine.get_entity(entity_name=args.entity)
            if result and "error" in result:
                raise SystemExit(result["error"])
        else:
            target = args.entity.strip().lower()
            entity = next(
                (e for e in json_store.entities.values() if e.canonical_name.strip().lower() == target),
                None,
            )
            if entity is None:
                raise SystemExit(f"entity not found: {args.entity}")
            result = _entity_view(json_store, entity)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.domain:
        if is_graph:
            result = engine.search_by_domain(args.domain, limit=args.limit)
            if result and "error" in result:
                raise SystemExit(result["error"])
        else:
            d = args.domain.strip().lower()
            hits = [c for c in json_store.claims.values() if any(t.strip().lower() == d for t in c.tags)]
            result = {
                "domain": args.domain,
                "count": len(hits),
                "claims": [
                    {"id": c.id, "statement": c.statement, "status": str(c.epistemic_status)}
                    for c in hits[: args.limit]
                ],
            }
        print(json.dumps(result, indent=2, default=str))
        return

    if args.claim:
        if is_graph:
            result = engine.get_claim(args.claim)
            if result and "error" in result:
                raise SystemExit(result["error"])
        else:
            claim = json_store.claims.get(args.claim)
            if claim is None:
                raise SystemExit(f"claim not found: {args.claim}")
            evidence = [e.model_dump(mode="json") for e in json_store.evidence.values() if e.claim_id == args.claim]
            result = {"claim": claim.model_dump(mode="json"), "evidence": evidence}
        print(json.dumps(result, indent=2, default=str))
        return

    if args.query:
        if is_graph and engine.embedding_client is not None:
            try:
                qvec = engine.embedding_client.embed_sync(args.query)
                hits = engine.store.vector_search_claims(embedding=qvec, k=args.limit)
                result = {
                    "query": args.query,
                    "mode": "semantic vector search (bge-m3 + Neo4j)",
                    "count": len(hits),
                    "matches": [
                        {
                            "id": h["claim_id"],
                            "similarity": round(float(h["similarity"]), 4),
                            "statement": h["statement"],
                            "status": h["epistemic_status"],
                        }
                        for h in hits
                    ],
                }
                print(json.dumps(result, indent=2, default=str))
                return
            except Exception as exc:
                print(f"note: vector search failed ({exc}), falling back to keyword.", file=sys.stderr)
        elif not is_graph:
            # JSON-store path: try the separate Neo4j handle for semantic search.
            graph, embed = _open_vector_backend_for_ask()
            if graph is not None and embed is not None:
                try:
                    qvec = embed.embed_sync(args.query)
                    hits = graph.vector_search_claims(embedding=qvec, k=args.limit)
                    result = {
                        "query": args.query,
                        "mode": "semantic vector search (bge-m3 + Neo4j)",
                        "count": len(hits),
                        "matches": [
                            {
                                "id": h["claim_id"],
                                "similarity": round(float(h["similarity"]), 4),
                                "statement": h["statement"],
                                "status": h["epistemic_status"],
                            }
                            for h in hits
                        ],
                    }
                    print(json.dumps(result, indent=2, default=str))
                    return
                finally:
                    graph.close()

        # Keyword fallback: substring term matching over the JSON store.
        claims_iter = json_store.claims.values() if json_store else []
        terms = [t for t in args.query.lower().split() if t]
        scored = []
        for c in claims_iter:
            score = sum(1 for t in terms if t in c.statement.lower())
            if score:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = {
            "query": args.query,
            "count": len(scored),
            "matches": [
                {"id": c.id, "score": score, "statement": c.statement, "status": str(c.epistemic_status)}
                for score, c in scored[: args.limit]
            ],
        }
        print(json.dumps(result, indent=2, default=str))
        return

    raise SystemExit("ask: provide one of --entity / --domain / --claim / --query")


def cmd_snapshot(args: argparse.Namespace) -> None:
    engine, _ = _build_engine(args.state)
    print(json.dumps(engine.state_snapshot(), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(prog="ke", description="Knowledge Engine CLI (UAT)")
    parser.add_argument("--state", default=DEFAULT_STATE, help=f"state file (default: {DEFAULT_STATE})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a transcript file or folder")
    p_ingest.add_argument("path", help="Path to a .txt file or a folder of .txt files")
    p_ingest.add_argument("--domain", help="Domain (e.g. 'real estate', 'trading', 'tcm')")
    p_ingest.add_argument("--entity", help="Entity/topic name (defaults to filename)")
    p_ingest.add_argument(
        "--source-kind",
        default="external_doc",
        choices=["internal_wiki", "external_doc", "user"],
    )
    p_ingest.add_argument(
        "--workers",
        type=int,
        help="Parallel transcript prep workers (default: 2, capped by file count)",
    )

    p_ask = sub.add_parser("ask", help="Query the knowledge base")
    p_ask.add_argument("--entity", help="Full details for an entity by name")
    p_ask.add_argument("--domain", help="All claims tagged with a domain")
    p_ask.add_argument("--claim", help="Full details for a claim by id")
    p_ask.add_argument("--query", help="Keyword search over claim statements")
    p_ask.add_argument("--limit", type=int, default=25)

    sub.add_parser("snapshot", help="Print engine state counts")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "snapshot":
        cmd_snapshot(args)


if __name__ == "__main__":
    main()
