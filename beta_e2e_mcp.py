"""End-to-End Beta Test — Real-World User Scenarios via MCP Standard.

Simulates a user interacting with the knowledge engine through MCP tools:
1. Ingest transcripts from multiple experts
2. Query knowledge base
3. Detect and resolve conflicts
4. Process user feedback (evidence, disputes, corrections)
5. Measure performance at each stage

Run: python beta_e2e_mcp.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

from knowledge_engine import (
    ClaimDraft,
    EpistemicStatus,
    EvidenceDraft,
    KnowledgeEngine,
    TranscriptInput,
)
from knowledge_engine.bootstrap import build_engine_from_env
from knowledge_engine.intent_classifier import IntentClassifier


# ---------------------------------------------------------------------------
# MCP Tool Simulation
# ---------------------------------------------------------------------------
class MCPToolCaller:
    """Simulate MCP tool calls with timing and logging."""

    def __init__(self, engine: KnowledgeEngine):
        self.engine = engine
        self.results: list[dict] = []
        self.metrics: dict[str, float] = {}

    def call(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Execute an MCP tool and record results."""
        start = time.monotonic()
        try:
            result = getattr(self.engine, tool_name)(**kwargs)
            elapsed = time.monotonic() - start
            self.metrics[tool_name] = elapsed
            self.results.append({
                "tool": tool_name,
                "args": kwargs,
                "result": result,
                "latency_ms": round(elapsed * 1000, 2),
                "status": "success",
            })
            return result
        except Exception as e:
            elapsed = time.monotonic() - start
            self.metrics[tool_name] = elapsed
            self.results.append({
                "tool": tool_name,
                "args": kwargs,
                "error": str(e),
                "latency_ms": round(elapsed * 1000, 2),
                "status": "error",
            })
            return {"error": str(e)}

    def summary(self) -> dict:
        """Return test summary."""
        success = sum(1 for r in self.results if r["status"] == "success")
        error = sum(1 for r in self.results if r["status"] == "error")
        total_latency = sum(r["latency_ms"] for r in self.results)
        return {
            "total_calls": len(self.results),
            "success": success,
            "error": error,
            "total_latency_ms": round(total_latency, 2),
            "avg_latency_ms": round(total_latency / len(self.results), 2) if self.results else 0,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------
def test_scenario_1_ingestion(mcp: MCPToolCaller) -> dict:
    """Scenario 1: Ingest transcripts from multiple experts."""
    print("\n" + "=" * 78)
    print("SCENARIO 1: Multi-Expert Transcript Ingestion")
    print("=" * 78)

    transcripts = [
        {
            "domain": "trading",
            "entity_name": "Dr. Momentum",
            "text": "Momentum strategies generate 3-5% alpha in trending markets with ADX above 25.",
            "source_id": "doc-momentum-2024",
        },
        {
            "domain": "trading",
            "entity_name": "Dr. Value",
            "text": "Value stocks outperform growth stocks over 10-year horizons due to mean reversion.",
            "source_id": "doc-value-investing",
        },
        {
            "domain": "real estate",
            "entity_name": "Prof. Cap Rate",
            "text": "Cap rates above 6% indicate undervalued commercial properties in tier-2 cities.",
            "source_id": "doc-cap-rate-analysis",
        },
        {
            "domain": "trading",
            "entity_name": "Contrarian Quant",
            "text": "Momentum strategies fail in mean-reverting markets and lose money 40% of the time.",
            "source_id": "doc-contrarian-momentum",
        },
    ]

    results = []
    for t in transcripts:
        outcome = mcp.call(
            "ingest_transcript",
            transcript=TranscriptInput(
                domain=t["domain"],
                entity_name=t["entity_name"],
                transcript_text=f"[{t['entity_name']}] {t['text']}",
                source_kind="external_doc",
                source_id=t["source_id"],
                claim_drafts=[
                    ClaimDraft(
                        statement=t["text"],
                        slot_name="strategy",
                        observed_slots=["strategy"],
                        evidence=[
                            EvidenceDraft(
                                source_kind="external_doc",
                                source_id=f"{t['source_id']}-evidence",
                                credibility=0.7,
                            )
                        ],
                    )
                ],
            ),
        )
        status = "CONFIRMED" if outcome.confirmed_claim_ids else "UNVERIFIED"
        print(f"  [{status}] {t['entity_name']}: {t['text'][:60]}...")
        results.append({"entity": t["entity_name"], "status": status})

    # Check state
    state = mcp.call("state_snapshot")
    print(f"\n  State: {state.get('entities', 0)} entities, {state.get('claims', 0)} claims")

    return {"transcripts": results, "state": state}


def test_scenario_2_query(mcp: MCPToolCaller) -> dict:
    """Scenario 2: Query knowledge base."""
    print("\n" + "=" * 78)
    print("SCENARIO 2: Knowledge Base Query")
    print("=" * 78)

    queries = [
        "What are the key principles in trading?",
        "How do different schools of thought view momentum?",
        "What conflicts exist in the trading domain?",
    ]

    results = []
    for q in queries:
        result = mcp.call("explore_experience", query=q, domain="trading")
        claims_count = len(result.get("experience_claims", []))
        cross_count = len(result.get("cross_domain_patterns", []))
        print(f"  Query: {q[:50]}...")
        print(f"    Claims: {claims_count}, Cross-domain: {cross_count}")
        results.append({"query": q, "claims": claims_count, "patterns": cross_count})

    return {"queries": results}


def test_scenario_3_conflicts(mcp: MCPToolCaller) -> dict:
    """Scenario 3: Detect and resolve conflicts."""
    print("\n" + "=" * 78)
    print("SCENARIO 3: Conflict Detection & Resolution")
    print("=" * 78)

    # Check for open cases
    open_cases = mcp.call("list_open_cases")
    case_count = len(open_cases) if isinstance(open_cases, list) else 0
    print(f"  Open conflict cases: {case_count}")

    # Resolve first case if exists
    resolved = None
    if case_count > 0 and isinstance(open_cases, list):
        case_id = open_cases[0].get("case_id")
        if case_id:
            resolved = mcp.call(
                "resolve_case",
                case_id=case_id,
                decision="Momentum works in trending regimes, fails in ranging. Regime detection is key.",
                rationale="Both sides are valid under different market conditions.",
            )
            print(f"  Resolved case {case_id}")

    return {"open_cases": case_count, "resolved": resolved is not None}


def test_scenario_4_intent_classification(mcp: MCPToolCaller) -> dict:
    """Scenario 4: Intent classification."""
    print("\n" + "=" * 78)
    print("SCENARIO 4: Intent Classification")
    print("=" * 78)

    test_inputs = [
        ("Hello, how are you?", "chat"),
        ("Here's evidence showing momentum works in trending markets", "evidence"),
        ("I disagree with the claim that momentum always works", "dispute"),
        ("Let me clarify - this only applies in bull markets", "correction"),
        ("What are the key principles in trading?", "exploration"),
        ("Let me share my analysis of the market", "learning"),
    ]

    results = []
    for text, expected in test_inputs:
        result = mcp.call("classify_intent", text=text)
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0)
        match = intent == expected
        status = "✓" if match else "✗"
        print(f"  {status} '{text[:40]}...' -> {intent} (expected: {expected}, conf: {confidence:.2f})")
        results.append({"text": text, "expected": expected, "actual": intent, "match": match})

    accuracy = sum(1 for r in results if r["match"]) / len(results) if results else 0
    print(f"\n  Accuracy: {accuracy:.1%}")

    return {"results": results, "accuracy": accuracy}


def test_scenario_5_logical_gaps(mcp: MCPToolCaller) -> dict:
    """Scenario 5: Logical gap detection."""
    print("\n" + "=" * 78)
    print("SCENARIO 5: Logical Gap Detection")
    print("=" * 78)

    # Ingest a transcript with logical issues
    mcp.call(
        "ingest_transcript",
        transcript=TranscriptInput(
            domain="trading",
            entity_name="Over-Generalizer",
            transcript_text="[Over-Generalizer] All stocks always follow momentum patterns without exception.",
            source_kind="external_doc",
            source_id="doc-overgeneralization",
            claim_drafts=[
                ClaimDraft(
                    statement="All stocks always follow momentum patterns without exception.",
                    slot_name="pattern",
                    observed_slots=["pattern"],
                    evidence=[
                        EvidenceDraft(
                            source_kind="external_doc",
                            source_id="doc-single-study",
                            credibility=0.6,
                        )
                    ],
                )
            ],
        ),
    )

    # Run logical gap detection
    from knowledge_engine.logical_gaps import LogicalGapDetector
    detector = LogicalGapDetector()

    # Get recent claims
    state = mcp.call("state_snapshot")
    print(f"  State: {state.get('claims', 0)} claims")

    # Simulate gap detection on sample claims
    from knowledge_engine.models import Claim, Evidence
    sample_claims = [
        Claim(id="test1", entity_id="test", statement="All stocks always follow momentum",
              epistemic_status=EpistemicStatus.UNVERIFIED),
    ]
    sample_evidence = [
        Evidence(id="e1", claim_id="test1", source_kind="external_doc",
                source_id="doc1", credibility=0.6),
    ]

    gaps = detector.detect(sample_claims, sample_evidence)
    print(f"  Gaps detected: {len(gaps)}")
    for gap in gaps:
        print(f"    - {gap.kind}: {gap.rationale[:60]}...")

    return {"gaps_count": len(gaps), "gaps": [{"kind": g.kind, "rationale": g.rationale} for g in gaps]}


def test_scenario_6_performance(mcp: MCPToolCaller) -> dict:
    """Scenario 6: Performance benchmarks."""
    print("\n" + "=" * 78)
    print("SCENARIO 6: Performance Benchmarks")
    print("=" * 78)

    benchmarks = {}

    # Ingest latency
    start = time.monotonic()
    mcp.call(
        "ingest_transcript",
        transcript=TranscriptInput(
            domain="trading",
            entity_name="Perf Test",
            transcript_text="[Perf Test] Performance test claim for benchmarking.",
            source_kind="external_doc",
            source_id="doc-perf-test",
            claim_drafts=[
                ClaimDraft(
                    statement="Performance test claim for benchmarking.",
                    slot_name="test",
                    observed_slots=["test"],
                )
            ],
        ),
    )
    benchmarks["ingest_ms"] = round((time.monotonic() - start) * 1000, 2)

    # Query latency
    start = time.monotonic()
    mcp.call("explore_experience", query="What is momentum?", domain="trading")
    benchmarks["query_ms"] = round((time.monotonic() - start) * 1000, 2)

    # Intent classification latency
    start = time.monotonic()
    mcp.call("classify_intent", text="Hello, how are you?")
    benchmarks["intent_ms"] = round((time.monotonic() - start) * 1000, 2)

    # State snapshot latency
    start = time.monotonic()
    mcp.call("state_snapshot")
    benchmarks["snapshot_ms"] = round((time.monotonic() - start) * 1000, 2)

    for name, latency in benchmarks.items():
        print(f"  {name}: {latency:.2f}ms")

    return {"benchmarks": benchmarks}


# ---------------------------------------------------------------------------
# Main Test Runner
# ---------------------------------------------------------------------------
def main() -> int:
    """Run full end-to-end beta test."""
    print("=" * 78)
    print("KNOWLEDGE ENGINE — END-TO-END MCP BETA TEST")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 78)

    # Initialize engine
    print("\nInitializing engine...")
    try:
        engine = build_engine_from_env(auto_start=True)
        print("  Engine initialized successfully")
    except (SystemExit, Exception) as e:
        error_msg = str(e)
        if "7687" in error_msg or "neo4j" in error_msg.lower() or "KE_NEO4J" in error_msg:
            print("  WARNING: Neo4j not available. Running limited test suite.")
            print("  To run full test, start Neo4j first:")
            print("    - Install Neo4j 5.13+ from https://neo4j.com/download/")
            print("    - Set KE_NEO4J_PATH in .env")
            print("    - Or run: docker-compose up -d")
            return run_limited_test()
        else:
            print(f"  FATAL: Engine initialization failed: {e}")
            return 1

    # Health check
    health = engine.health_check()
    print(f"  Health: {health.get('overall', 'unknown')}")

    # Create MCP tool caller
    mcp = MCPToolCaller(engine)

    # Run scenarios
    results = {}
    try:
        results["scenario_1"] = test_scenario_1_ingestion(mcp)
        results["scenario_2"] = test_scenario_2_query(mcp)
        results["scenario_3"] = test_scenario_3_conflicts(mcp)
        results["scenario_4"] = test_scenario_4_intent_classification(mcp)
        results["scenario_5"] = test_scenario_5_logical_gaps(mcp)
        results["scenario_6"] = test_scenario_6_performance(mcp)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 78)
    print("TEST SUMMARY")
    print("=" * 78)

    summary = mcp.summary()
    print(f"  Total tool calls: {summary['total_calls']}")
    print(f"  Success: {summary['success']}")
    print(f"  Errors: {summary['error']}")
    print(f"  Total latency: {summary['total_latency_ms']:.2f}ms")
    print(f"  Average latency: {summary['avg_latency_ms']:.2f}ms")

    # Intent accuracy
    if "scenario_4" in results:
        accuracy = results["scenario_4"].get("accuracy", 0)
        print(f"  Intent classification accuracy: {accuracy:.1%}")

    # Logical gaps
    if "scenario_5" in results:
        gaps = results["scenario_5"].get("gaps_count", 0)
        print(f"  Logical gaps detected: {gaps}")

    # Performance
    if "scenario_6" in results:
        benchmarks = results["scenario_6"].get("benchmarks", {})
        print(f"  Ingest latency: {benchmarks.get('ingest_ms', 0):.2f}ms")
        print(f"  Query latency: {benchmarks.get('query_ms', 0):.2f}ms")
        print(f"  Intent latency: {benchmarks.get('intent_ms', 0):.2f}ms")

    # Write results to file
    output_file = "beta_e2e_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")

    # Verdict
    print("\n" + "=" * 78)
    if summary["error"] == 0 and summary["success"] > 0:
        print("VERDICT: PASS — System is ready for 6dfov integration")
    else:
        print("VERDICT: FAIL — Issues detected, review results")
    print("=" * 78)

    return 0 if summary["error"] == 0 else 1


def run_limited_test() -> int:
    """Run limited test suite without Neo4j (intent classification + logical gaps)."""
    print("\n" + "=" * 78)
    print("LIMITED TEST SUITE — No Neo4j Required")
    print("=" * 78)

    from knowledge_engine.intent_classifier import IntentClassifier
    from knowledge_engine.logical_gaps import LogicalGapDetector
    from knowledge_engine.models import Claim, Evidence, EpistemicStatus
    from knowledge_engine.embeddings import EmbeddingClient

    results = {}

    # Test 1: Intent Classification
    print("\n--- Test 1: Intent Classification ---")
    try:
        client = EmbeddingClient.from_env()
        if client:
            classifier = IntentClassifier(client)
            classifier.warm_up()

            test_cases = [
                ("Hello, how are you?", "chat"),
                ("Here's evidence showing momentum works", "evidence"),
                ("I disagree with this claim", "dispute"),
                ("Let me clarify - this only applies in bull markets", "correction"),
                ("What are the key principles in trading?", "exploration"),
                ("Let me share my analysis of the market", "learning"),
            ]

            correct = 0
            for text, expected in test_cases:
                result = classifier.classify(text)
                match = result.intent == expected
                correct += match
                status = "✓" if match else "✗"
                print(f"  {status} '{text[:40]}...' -> {result.intent} (expected: {expected})")

            accuracy = correct / len(test_cases)
            print(f"\n  Accuracy: {accuracy:.1%}")
            results["intent_accuracy"] = accuracy
        else:
            print("  SKIPPED: Embedding client not configured")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 2: Logical Gap Detection
    print("\n--- Test 2: Logical Gap Detection ---")
    try:
        detector = LogicalGapDetector()

        # Test circular reasoning
        claims = [
            Claim(id="a", entity_id="e1", statement="A", epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="b", entity_id="e1", statement="B", epistemic_status=EpistemicStatus.CONFIRMED),
            Claim(id="c", entity_id="e1", statement="C", epistemic_status=EpistemicStatus.CONFIRMED),
        ]
        evidence = [
            Evidence(id="e1", claim_id="b", source_kind="internal_wiki", source_id="a",
                    credibility=0.8, linked_claim_ids=["a"]),
            Evidence(id="e2", claim_id="c", source_kind="internal_wiki", source_id="b",
                    credibility=0.8, linked_claim_ids=["b"]),
            Evidence(id="e3", claim_id="a", source_kind="internal_wiki", source_id="c",
                    credibility=0.8, linked_claim_ids=["c"]),
        ]

        gaps = detector.detect(claims, evidence)
        circular = [g for g in gaps if "circular" in g.rationale.lower()]
        print(f"  Circular reasoning detected: {len(circular) > 0}")
        results["circular_detection"] = len(circular) > 0

        # Test over-generalization
        claims2 = [
            Claim(id="c1", entity_id="e1", statement="All stocks follow momentum",
                 epistemic_status=EpistemicStatus.UNVERIFIED),
        ]
        evidence2 = [
            Evidence(id="e1", claim_id="c1", source_kind="external_doc",
                    source_id="doc1", credibility=0.6),
        ]

        gaps2 = detector.detect(claims2, evidence2)
        overgen = [g for g in gaps2 if "over-generalization" in g.rationale.lower()]
        print(f"  Over-generalization detected: {len(overgen) > 0}")
        results["overgen_detection"] = len(overgen) > 0

    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 78)
    print("LIMITED TEST SUMMARY")
    print("=" * 78)

    all_pass = all(results.values()) if results else False
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all_pass:
        print("\nVERDICT: PASS — Core features working, ready for 6dfov integration")
        print("  Note: Full MCP test requires Neo4j running")
    else:
        print("\nVERDICT: PARTIAL — Some features need attention")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
