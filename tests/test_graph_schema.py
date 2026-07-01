from __future__ import annotations

import pytest

from knowledge_engine.graph import schema


def test_vector_index_cypher_embeds_dimension_and_similarity() -> None:
    ddl = schema.vector_index_cypher(dimensions=1536, similarity="cosine")
    assert "CREATE VECTOR INDEX" in ddl
    assert schema.CLAIM_VECTOR_INDEX in ddl
    assert "`vector.dimensions`: 1536" in ddl
    assert "`vector.similarity_function`: 'cosine'" in ddl
    assert "IF NOT EXISTS" in ddl


def test_vector_index_cypher_rejects_bad_dimension() -> None:
    with pytest.raises(ValueError):
        schema.vector_index_cypher(dimensions=0)


def test_vector_index_cypher_rejects_unknown_similarity() -> None:
    with pytest.raises(ValueError):
        schema.vector_index_cypher(dimensions=8, similarity="manhattan")


def test_schema_statements_include_all_constraints_and_index() -> None:
    statements = schema.schema_statements(dimensions=8)
    # one vector index + every uniqueness constraint
    assert len(statements) == len(schema.UNIQUENESS_CONSTRAINTS) + 1
    assert any("CREATE VECTOR INDEX" in s for s in statements)
    assert all("IF NOT EXISTS" in s for s in statements)


def test_cycle_probe_uses_supports_traversal() -> None:
    cypher = schema.cycle_probe_cypher()
    assert "$source_id" in cypher
    assert "$target_id" in cypher
    assert "SUPPORTS*1.." in cypher
    assert "creates_cycle" in cypher
