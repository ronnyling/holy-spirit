"""Shared engine bootstrap for the MCP server and the CLI.

Centralizes: (1) loading a local .env file, and (2) constructing a
KnowledgeEngine from environment variables (Neo4j required; embeddings and LLM
optional). Keeping this in one place stops the server and CLI from drifting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .cache import RetrievalCache
from .embeddings import EmbeddingClient
from .engine import KnowledgeEngine
from .extraction import ClaimExtractor
from .llm import MiMoClient
from .query_processor import QueryProcessor
from .registry import TranscriptRegistry
from .reranker import RerankerClient
from .service_manager import ServiceManager, ServiceStatus


def load_dotenv(path: str | os.PathLike[str] = ".env") -> bool:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Does not override variables already set in the environment. Returns True if
    a file was found and read. No external dependency, no interpolation.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def build_engine_from_env(
    *,
    dotenv_path: str | os.PathLike[str] = ".env",
    auto_start: bool = True,
) -> KnowledgeEngine:
    """Construct the engine from env vars. Neo4j is required; the rest optional.

    - Neo4j missing        -> SystemExit(1) (no in-memory fallback).
    - Embeddings key absent -> vector search tools disabled (warned).
    - MiMo key absent       -> LLM extraction disabled; manual claim_drafts only.

    Args:
        dotenv_path: Path to .env file
        auto_start: If True, auto-start Neo4j and Ollama if not running
    """
    load_dotenv(dotenv_path)

    # Auto-start services if requested
    if auto_start:
        manager = ServiceManager()
        results = manager.ensure_all_services()
        for name, result in results.items():
            if result.status == ServiceStatus.NOT_CONFIGURED:
                print(f"INFO: {name} not configured — {result.message}", file=sys.stderr)
            elif result.status == ServiceStatus.ERROR:
                print(f"WARNING: {name} failed to start: {result.message}", file=sys.stderr)
            elif result.status == ServiceStatus.RUNNING:
                print(f"  ✓ {name}: {result.message}")

    neo4j_uri = os.environ.get("KE_NEO4J_URI", "")
    neo4j_user = os.environ.get("KE_NEO4J_USER", "")
    neo4j_password = os.environ.get("KE_NEO4J_PASSWORD", "")
    neo4j_database = os.environ.get("KE_NEO4J_DATABASE", "neo4j")
    embedding_dims = int(os.environ.get("KE_EMBEDDING_DIMENSIONS", "0"))

    if not neo4j_uri:
        print("FATAL: KE_NEO4J_URI must be set", file=sys.stderr)
        raise SystemExit(1)
    if not neo4j_user:
        print("FATAL: KE_NEO4J_USER must be set", file=sys.stderr)
        raise SystemExit(1)
    if not neo4j_password:
        print("FATAL: KE_NEO4J_PASSWORD must be set", file=sys.stderr)
        raise SystemExit(1)

    from .graph.neo4j_store import KnowledgeGraphStore

    store = KnowledgeGraphStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        embedding_dimensions=embedding_dims,
    )
    store.verify()
    store.apply_schema()

    embedding_client = EmbeddingClient.from_env()
    if embedding_client is None:
        print(
            "WARNING: KE_EMBEDDING_API_KEY not set — vector search tools disabled.",
            file=sys.stderr,
        )

    # Wire query-path cache for embedding deduplication.
    cache = RetrievalCache.from_env()
    if embedding_client is not None:
        embedding_client.set_cache(cache)

    mimo_client = MiMoClient.from_env()

    # ClaimExtractor auto-tunes parallelism and checkpointing based on file size.
    # No manual configuration needed — the system selects optimal settings automatically.
    extractor = ClaimExtractor(mimo_client) if mimo_client is not None else None
    if extractor is None:
        print(
            "WARNING: KE_MIMO_API_KEY not set — LLM claim extraction disabled "
            "(supply claim_drafts manually).",
            file=sys.stderr,
        )

    registry_path = os.environ.get("KE_REGISTRY_PATH", "./ke_data")
    registry = TranscriptRegistry(root=registry_path)

    reranker = RerankerClient.from_env()
    if reranker is None:
        print(
            "INFO: KE_RERANKER_PROVIDER=none — cross-encoder reranking disabled "
            "(retrieval uses evidence-weighted similarity).",
            file=sys.stderr,
        )

    query_processor = QueryProcessor.from_env(mimo_client)

    return KnowledgeEngine(
        store=store, embedding_client=embedding_client, extractor=extractor,
        llm_client=mimo_client, registry=registry, reranker=reranker,
        query_processor=query_processor, cache=cache,
    )
