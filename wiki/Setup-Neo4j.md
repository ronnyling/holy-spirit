# Setup: Neo4j

Neo4j is the committed storage engine (property graph + native vector index). There is no in-memory
fallback at runtime — the store requires a live database. Verified against Neo4j Community **2026.05.0**
(also works on 5.13+).

The application connects over a **Bolt URI**, so it does not care *how* Neo4j is hosted. Pick whichever
launch option works on your machine — the rest of the setup is identical.

## 1. Start Neo4j

### Option A — Neo4j Community Server (no Docker, no virtualization) — recommended when Docker is unavailable

Use this when hardware virtualization is disabled/blocked (Docker Desktop reports "Virtualization support
not detected"). This is the option the local beta was **verified** against.

> **Java version matters.** Neo4j calendar releases (2026.x) require **Java 21**. Java 17 fails at startup
> with `UnsupportedClassVersionError` (class file version 65.0 vs 61.0). If you only have Java 17, either
> use a Neo4j 5.x zip instead, or point `JAVA_HOME` at any JDK 21. On this machine the JetBrains Runtime
> bundled with Android Studio is a usable JDK 21: `C:\Program Files\Android\Android Studio\jbr`.

```powershell
# 1. Download Neo4j Community (zip) from https://neo4j.com/deployment-center/
#    and unzip somewhere writable. Verified location for this beta:
#    C:\Users\r.a.ling\OneDrive - Avanade\Documents\work\neo4j-community-2026.05.0

# 2. Point Neo4j at a JDK 21 (Set-Item avoids the flaky `$env:` assignment on this shell)
Set-Item Env:JAVA_HOME "C:\Program Files\Android\Android Studio\jbr"

# 3. Set the initial password (run once, before first start)
cd "C:\Users\r.a.ling\OneDrive - Avanade\Documents\work\neo4j-community-2026.05.0"
bin\neo4j-admin.bat dbms set-initial-password knowledge-engine

# 4. Run it in the foreground
bin\neo4j.bat console   # prints "Started." when ready
```

Native vector indexes are available in Community edition from 5.13+, so the free Community Server is
sufficient for this project. The Python driver (`neo4j` 6.x) talks Bolt and does **not** need Java itself —
only the server does, so pytest is unaffected by your Java version.

> **Beta credentials.** Password `knowledge-engine` is committed here on purpose — everything is local
> during beta. Rotate it (and move it to a secret) before any pre-release / shared deployment.

### Option B — Docker (requires hardware virtualization / WSL 2)

```powershell
cd "c:\Users\r.a.ling\OneDrive - Avanade\Documents\work\Native AI\knowledge_engine"
docker compose up -d
```

### Option C — AuraDB Free (cloud, no local install)

Create a free instance at https://neo4j.com/product/auradb/. Use the connection URI it gives you
(`neo4j+s://<id>.databases.neo4j.io`) and its generated password. Requires outbound internet, and your
data leaves the machine — check this is acceptable for your corporate policy first.

- Bolt (local options A/B): `bolt://localhost:7687`
- Browser UI (local options A/B): `http://localhost:7474` (user `neo4j`, password `knowledge-engine`)

## 2. Configure environment

Copy `.env.example` to `.env` and set at least:

```
KE_NEO4J_URI=bolt://localhost:7687
KE_NEO4J_USER=neo4j
KE_NEO4J_PASSWORD=knowledge-engine
KE_NEO4J_DATABASE=neo4j
KE_EMBEDDING_DIMENSIONS=1536
```

The embedding dimension **must** match the vector index dimension.

## 3. Apply the schema

The store applies constraints and the native vector index for you:

```python
from knowledge_engine.graph.neo4j_store import KnowledgeGraphStore

store = KnowledgeGraphStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="knowledge-engine",
    embedding_dimensions=1536,
)
store.verify()        # hard-fails if unreachable
store.apply_schema()  # constraints + claim_embedding_index
```

The DDL itself is pure and unit-tested in `knowledge_engine.graph.schema`.

## 4. Run the integration tests

```powershell
$env:KE_NEO4J_URI = "bolt://localhost:7687"
$env:KE_NEO4J_USER = "neo4j"
$env:KE_NEO4J_PASSWORD = "knowledge-engine"
$env:KE_NEO4J_DATABASE = "neo4j"
python -m pytest tests/test_graph_neo4j.py -v
```

These tests are **skipped** when `KE_NEO4J_URI` is unset — that is test gating, not a runtime fallback.

**Verified on 2026.05.0:** full suite `python -m pytest -q` → **33 passed** (30 unit + 3 integration:
entity/claim roundtrip, cycle rejection, vector similarity search).

### Vector index is eventually consistent

Neo4j vector indexes update asynchronously, so a claim you just wrote is not guaranteed to be searchable
on the very next query. After writing embeddings, callers that need read-your-writes should call
`store.await_indexes()` (wraps `CALL db.awaitIndexes(...)`) and/or poll `find_similar_claims` briefly —
see `test_vector_search_returns_similar_claim` for the pattern.

> **Deprecation note (pre-release follow-up):** `find_similar_claims` uses `CALL db.index.vector.queryNodes`,
> which is deprecated in Neo4j 2026 in favour of the `SEARCH` clause. It still works on 2026.05; migrate
> before pre-release.

## Cycle detection

Internal support edges (`Claim -[:SUPPORTS]-> Claim`) are guarded by a native Cypher reachability probe.
Adding an edge that would close a cycle raises `GraphCycleError`. A claim can never justify itself,
directly or transitively.
