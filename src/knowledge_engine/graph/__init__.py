"""Graph persistence layer (Neo4j 5) for the Continuous R&D Knowledge Engine.

Neo4j 5 is the committed storage engine: a native property graph *and* a native
vector index in a single database. This package holds:

* :mod:`knowledge_engine.graph.schema` — pure, side-effect-free Cypher DDL
  (uniqueness constraints + the native vector index). Unit-testable without a
  running database.
* :mod:`knowledge_engine.graph.neo4j_store` — the real Neo4j-driver store. Its
  behaviour is exercised by integration tests that are skipped unless a Neo4j
  instance is reachable (``KE_NEO4J_URI``); there is **no** in-memory fallback
  at runtime.

Importing this subpackage does not import the ``neo4j`` driver, so the core
engine and its unit tests do not depend on it.
"""

from __future__ import annotations

from . import schema

__all__ = ["schema"]
