"""Shared engine bootstrap for the MCP server and the CLI.

Centralizes: (1) loading a local .env file, and (2) constructing a
KnowledgeEngine from environment variables (Neo4j required; embeddings and LLM
optional). Keeping this in one place stops the server and CLI from drifting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .embeddings import EmbeddingClient
from .engine import KnowledgeEngine
from .extraction import ClaimExtractor
from .llm import MiMoClient


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


def build_engine_from_env(*, dotenv_path: str | os.PathLike[str] = ".env") -> KnowledgeEngine:
    """Construct the engine from env vars. Neo4j is required; the rest optional.

    - Neo4j missing        -> SystemExit(1) (no in-memory fallback).
    - Embeddings key absent -> vector search tools disabled (warned).
    - MiMo key absent       -> LLM extraction disabled; manual claim_drafts only.
    """
    load_dotenv(dotenv_path)

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

    mimo_client = MiMoClient.from_env()
    extractor = ClaimExtractor(mimo_client) if mimo_client is not None else None
    if extractor is None:
        print(
            "WARNING: KE_MIMO_API_KEY not set — LLM claim extraction disabled "
            "(supply claim_drafts manually).",
            file=sys.stderr,
        )

    return KnowledgeEngine(
        store=store, embedding_client=embedding_client, extractor=extractor
    )
