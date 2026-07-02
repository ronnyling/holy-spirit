"""Knowledge Engine CLI — ingest transcripts and ask research questions.

UAT NOTE ON STORAGE
  The engine's ingest pipeline currently targets the in-memory KnowledgeStore.
  The Neo4j store implements only the query/vector side and speaks a different
  interface, so it cannot yet be fed by the pipeline (see docs/UAT-Usage-Guide).
  Therefore this CLI uses the in-memory store and persists it to a local JSON
  file (--state) so an ingested corpus survives across `ingest` and `ask` runs.
  Queries here read that store directly (no embeddings required).

USAGE
  # Ingest one file or a whole folder of .txt transcripts
  python scripts/ke.py ingest data/uat_transcripts --domain "real estate"
  python scripts/ke.py ingest data/uat_transcripts/deal.txt --domain trading --entity "Breakout Rules"

  # Ask questions
  python scripts/ke.py ask --entity "Cap Rate Rules"
  python scripts/ke.py ask --domain "real estate"
  python scripts/ke.py ask --query "cap rate suburban"      # keyword search over claims
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

import argparse
import json
import sys
from pathlib import Path

# Make `src/` importable when run as `python scripts/ke.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knowledge_engine.bootstrap import load_dotenv  # noqa: E402
from knowledge_engine.contracts import TranscriptInput  # noqa: E402
from knowledge_engine.engine import KnowledgeEngine  # noqa: E402
from knowledge_engine.extraction import ClaimExtractor  # noqa: E402
from knowledge_engine.llm import MiMoClient  # noqa: E402
from knowledge_engine.store import KnowledgeStore  # noqa: E402

DEFAULT_STATE = "data/uat_state.json"


def _build_engine(state_path: str) -> KnowledgeEngine:
    load_dotenv()
    store = KnowledgeStore.load(state_path)
    mimo = MiMoClient.from_env()
    extractor = ClaimExtractor(mimo) if mimo is not None else None
    if extractor is None:
        print(
            "note: KE_MIMO_API_KEY not set — LLM extraction off; using claim_drafts only.",
            file=sys.stderr,
        )
    return KnowledgeEngine(store=store, extractor=extractor)


# -- metadata / claim loading --------------------------------------------------


def _load_meta(txt_path: Path, args: argparse.Namespace) -> dict:
    meta_path = txt_path.with_suffix(".meta.json")
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "domain": meta.get("domain") or args.domain or txt_path.parent.name,
        "entity_name": meta.get("entity_name") or args.entity or txt_path.stem.replace("_", " ").title(),
        "source_kind": meta.get("source_kind") or args.source_kind,
        "source_id": meta.get("source_id") or txt_path.stem,
        "source_ref": meta.get("source_ref") or str(txt_path),
    }


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
        yield from sorted(target.glob("*.txt"))
    else:
        raise SystemExit(f"not found: {target}")


# -- commands ------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    engine = _build_engine(args.state)
    files = list(_iter_transcripts(Path(args.path)))
    if not files:
        raise SystemExit(f"no .txt transcripts found in {args.path}")

    for txt_path in files:
        text = txt_path.read_text(encoding="utf-8")
        meta = _load_meta(txt_path, args)
        transcript = TranscriptInput(
            transcript_text=text, claim_drafts=_load_claims(txt_path), **meta
        )
        outcome = engine.ingest_transcript(transcript)

        print(f"\n=== {txt_path.name} -> entity {outcome.entity_id} ===")
        print(f"  domain          : {meta['domain']}")
        print(f"  claims ingested : {len(outcome.claim_ids)}")
        print(f"  confirmed       : {len(outcome.confirmed_claim_ids)}")
        print(f"  unverified      : {len(outcome.unverified_claim_ids)}")
        print(f"  disputed        : {len(outcome.disputed_claim_ids)}")
        print(f"  gaps flagged    : {len(outcome.gap_flags)}")
        print(f"  conflicts       : {len(outcome.conflict_summaries)}")
        for s in outcome.slot_suggestions:
            print(f"  slot suggestion : {s.slot_name} {s.current_lifecycle}->{s.suggested_lifecycle}")
        for note in outcome.notes:
            print(f"  note            : {note}")

    engine.store.save(args.state)
    print(f"\nstate saved -> {args.state}")


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
    store = KnowledgeStore.load(args.state)

    if args.entity:
        target = args.entity.strip().lower()
        entity = next(
            (e for e in store.entities.values() if e.canonical_name.strip().lower() == target),
            None,
        )
        if entity is None:
            raise SystemExit(f"entity not found: {args.entity}")
        print(json.dumps(_entity_view(store, entity), indent=2, default=str))
        return

    if args.domain:
        d = args.domain.strip().lower()
        hits = [c for c in store.claims.values() if any(t.strip().lower() == d for t in c.tags)]
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
        claim = store.claims.get(args.claim)
        if claim is None:
            raise SystemExit(f"claim not found: {args.claim}")
        evidence = [e.model_dump(mode="json") for e in store.evidence.values() if e.claim_id == args.claim]
        print(json.dumps({"claim": claim.model_dump(mode="json"), "evidence": evidence}, indent=2, default=str))
        return

    if args.query:
        terms = [t for t in args.query.lower().split() if t]
        scored = []
        for c in store.claims.values():
            text = c.statement.lower()
            score = sum(1 for t in terms if t in text)
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
    store = KnowledgeStore.load(args.state)
    print(json.dumps(store.snapshot(), indent=2, sort_keys=True))


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
