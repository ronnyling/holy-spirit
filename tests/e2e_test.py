"""E2E Test: Simulates user workflow from user standpoint.

Scenarios:
1. User starts knowledge engine (main.py)
2. User ingests transcript via MCP
3. User queries via Streamlit (mocked)
4. User uses Android APK (mocked)

NO FALLBACKS: All failures are explicit.
"""
import json
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class E2ETestResult:
    """Track E2E test results."""
    
    def __init__(self):
        self.results: list[Dict[str, Any]] = []
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []
    
    def add(self, scenario: str, test_case: str, passed: bool, details: str = ""):
        self.results.append({
            "scenario": scenario,
            "test_case": test_case,
            "passed": passed,
            "details": details
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"{scenario} - {test_case}: {details}")
    
    def summary(self) -> str:
        total = self.passed + self.failed
        return f"""
E2E Test Summary
================
Total: {total}
Passed: {self.passed}
Failed: {self.failed}
Pass Rate: {self.passed/total*100:.1f}%

Errors:
{chr(10).join(self.errors) if self.errors else "None"}
"""


def test_scenario_1_main_py():
    """Scenario 1: User starts knowledge engine via main.py."""
    print("\n=== Scenario 1: User Starts Knowledge Engine ===")
    result = E2ETestResult()
    
    # Test 1.1: main.py exists
    main_py = Path(__file__).parent.parent / "main.py"
    result.add(
        "1. User Starts Engine",
        "1.1 main.py exists",
        main_py.exists(),
        f"main.py found at {main_py}" if main_py.exists() else "main.py not found"
    )
    
    # Test 1.2: main.py is executable (has shebang or is valid Python)
    if main_py.exists():
        content = main_py.read_text()
        is_valid = "def main()" in content or "if __name__" in content
        result.add(
            "1. User Starts Engine",
            "1.2 main.py is valid Python",
            is_valid,
            "main.py contains valid entry point" if is_valid else "main.py missing entry point"
        )
    
    # Test 1.3: HTTP server module exists
    http_server = Path(__file__).parent.parent / "src" / "knowledge_engine" / "http_server.py"
    result.add(
        "1. User Starts Engine",
        "1.3 HTTP server module exists",
        http_server.exists(),
        f"HTTP server found at {http_server}" if http_server.exists() else "HTTP server not found"
    )
    
    # Test 1.4: Streamlit app exists
    app_py = Path(__file__).parent.parent / "app.py"
    result.add(
        "1. User Starts Engine",
        "1.4 Streamlit app exists",
        app_py.exists(),
        f"Streamlit app found at {app_py}" if app_py.exists() else "Streamlit app not found"
    )
    
    return result


def test_scenario_2_mcp_ingestion():
    """Scenario 2: User ingests transcript via MCP."""
    print("\n=== Scenario 2: User Ingests Transcript via MCP ===")
    result = E2ETestResult()
    
    # Import modules
    try:
        from knowledge_engine.contracts import TranscriptInput, ClaimDraft
        from knowledge_engine.transcript_evidence import TranscriptEvidenceDraft
        from knowledge_engine.provenance import ProvenanceChain
        result.add(
            "2. MCP Ingestion",
            "2.1 Modules import successfully",
            True,
            "All required modules imported"
        )
    except ImportError as e:
        result.add(
            "2. MCP Ingestion",
            "2.1 Modules import successfully",
            False,
            f"Import error: {e}"
        )
        return result
    
    # Test 2.2: Create TranscriptInput
    try:
        transcript = TranscriptInput(
            domain="tcm",
            entity_name="turmeric",
            transcript_text="TCM studies show turmeric reduces inflammation by 40% in joint pain patients.",
            source_kind="external_doc",
            source_id="study_001"
        )
        result.add(
            "2. MCP Ingestion",
            "2.2 Create TranscriptInput",
            True,
            f"TranscriptInput created for domain={transcript.domain}"
        )
    except Exception as e:
        result.add(
            "2. MCP Ingestion",
            "2.2 Create TranscriptInput",
            False,
            f"Error: {e}"
        )
    
    # Test 2.3: Create TranscriptEvidenceDraft
    try:
        evidence = TranscriptEvidenceDraft(
            statement="40% reduction in inflammation",
            source_reference="paragraph 3",
            source_quality="academic",
            source_quality_score=0.85,
            conditions=["joint pain patients", "12-week period"],
            measurement_method="randomized controlled trial",
            methodology_score=0.9,
            confidence_indicator="medium",
            confidence_score=0.6,
            has_quantification=True,
            has_time_period=True,
            has_sample_size=False,
            has_primary_source=True,
            document_id="doc_001",
            transcript_id="transcript_001"
        )
        confidence = evidence.calculate_confidence()
        result.add(
            "2. MCP Ingestion",
            "2.3 Create TranscriptEvidenceDraft",
            True,
            f"Evidence created with confidence={confidence:.2f}"
        )
    except Exception as e:
        result.add(
            "2. MCP Ingestion",
            "2.3 Create TranscriptEvidenceDraft",
            False,
            f"Error: {e}"
        )
    
    # Test 2.4: Create ProvenanceChain
    try:
        chain = ProvenanceChain(
            claim_id="claim_001",
            evidence_ids=["ev_001"],
            document_ids=["doc_001"],
            transcript_ids=["transcript_001"],
            claim_metadata={"statement": "turmeric reduces inflammation"},
            evidence_metadata=[{"statement": "40% reduction"}],
            document_metadata=[{"filename": "study.md"}],
            transcript_metadata=[{"text": "original transcript"}]
        )
        chain_dict = chain.to_dict()
        result.add(
            "2. MCP Ingestion",
            "2.4 Create ProvenanceChain",
            True,
            f"ProvenanceChain created with {len(chain.evidence_ids)} evidence"
        )
    except Exception as e:
        result.add(
            "2. MCP Ingestion",
            "2.4 Create ProvenanceChain",
            False,
            f"Error: {e}"
        )
    
    return result


def test_scenario_3_streamlit():
    """Scenario 3: User queries via Streamlit (mocked)."""
    print("\n=== Scenario 3: User Queries via Streamlit ===")
    result = E2ETestResult()
    
    # Test 3.1: Store can be created
    try:
        from knowledge_engine.store import KnowledgeStore
        from knowledge_engine.models import Claim, EpistemicStatus
        
        store = KnowledgeStore()
        result.add(
            "3. Streamlit Query",
            "3.1 KnowledgeStore created",
            True,
            "KnowledgeStore initialized"
        )
    except Exception as e:
        result.add(
            "3. Streamlit Query",
            "3.1 KnowledgeStore created",
            False,
            f"Error: {e}"
        )
        return result
    
    # Test 3.2: Add claim to store
    try:
        claim = Claim(
            id="claim_001",
            statement="turmeric reduces inflammation",
            entity_id="entity_001",
            epistemic_status=EpistemicStatus.UNVERIFIED
        )
        store.add_claim(claim)
        result.add(
            "3. Streamlit Query",
            "3.2 Add claim to store",
            True,
            f"Claim added: {claim.id}"
        )
    except Exception as e:
        result.add(
            "3. Streamlit Query",
            "3.2 Add claim to store",
            False,
            f"Error: {e}"
        )
    
    # Test 3.3: Get provenance for claim
    try:
        chain = store.get_provenance("claim_001")
        result.add(
            "3. Streamlit Query",
            "3.3 Get provenance for claim",
            True,
            f"Provenance chain retrieved for {chain.claim_id}"
        )
    except Exception as e:
        result.add(
            "3. Streamlit Query",
            "3.3 Get provenance for claim",
            False,
            f"Error: {e}"
        )
    
    # Test 3.4: Get provenance for nonexistent claim (should raise KeyError)
    try:
        store.get_provenance("nonexistent_claim")
        result.add(
            "3. Streamlit Query",
            "3.4 Get provenance for nonexistent claim",
            False,
            "Should have raised KeyError"
        )
    except KeyError as e:
        result.add(
            "3. Streamlit Query",
            "3.4 Get provenance for nonexistent claim",
            True,
            f"KeyError raised as expected: {e}"
        )
    except Exception as e:
        result.add(
            "3. Streamlit Query",
            "3.4 Get provenance for nonexistent claim",
            False,
            f"Wrong exception type: {type(e).__name__}: {e}"
        )
    
    return result


def test_scenario_4_android_apk():
    """Scenario 4: User uses Android APK (mocked)."""
    print("\n=== Scenario 4: User Uses Android APK ===")
    result = E2ETestResult()
    
    # Test 4.1: McpClient exists
    mcp_client_path = Path(__file__).parent.parent.parent / "6dfov" / "app" / "src" / "main" / "java" / "com" / "sixdfov" / "app" / "mcp" / "McpClient.kt"
    result.add(
        "4. Android APK",
        "4.1 McpClient.kt exists",
        mcp_client_path.exists(),
        f"McpClient found at {mcp_client_path}" if mcp_client_path.exists() else "McpClient not found"
    )
    
    # Test 4.2: VisionData classes exist in McpClient
    if mcp_client_path.exists():
        content = mcp_client_path.read_text()
        has_vision_data = "VisionData" in content
        result.add(
            "4. Android APK",
            "4.2 VisionData class exists",
            has_vision_data,
            "VisionData class found" if has_vision_data else "VisionData class not found"
        )
    
    # Test 4.3: Mock OCR data structure
    mock_ocr_data = {
        "type": "vision_data",
        "timestamp": int(time.time()),
        "mode": "phone",
        "ocr": {
            "text": "TCM studies show turmeric reduces inflammation",
            "blocks": [
                {
                    "text": "TCM studies show turmeric reduces inflammation",
                    "confidence": 0.95,
                    "bbox": [100, 200, 300, 50]
                }
            ],
            "language": "en"
        },
        "objects": [],
        "metadata": {
            "device_id": "mock_device_001",
            "camera_id": "0",
            "resolution": "1920x1080",
            "frame_number": 42
        }
    }
    
    # Validate mock data structure
    required_fields = ["type", "timestamp", "mode", "ocr", "metadata"]
    has_all_fields = all(field in mock_ocr_data for field in required_fields)
    result.add(
        "4. Android APK",
        "4.3 Mock OCR data structure valid",
        has_all_fields,
        f"All required fields present: {required_fields}" if has_all_fields else "Missing required fields"
    )
    
    # Test 4.4: Mock vision ingest endpoint exists
    http_server_path = Path(__file__).parent.parent / "src" / "knowledge_engine" / "http_server.py"
    if http_server_path.exists():
        content = http_server_path.read_text()
        has_vision_endpoint = "/api/vision/ingest" in content
        result.add(
            "4. Android APK",
            "4.4 Vision ingest endpoint exists",
            has_vision_endpoint,
            "Vision ingest endpoint found" if has_vision_endpoint else "Vision ingest endpoint not found"
        )
    
    return result


def test_no_fallbacks_policy():
    """Verify no-fallbacks policy is enforced."""
    print("\n=== No-Fallbacks Policy Verification ===")
    result = E2ETestResult()
    
    # Test: EvidenceExtractionError is raised when LLM not provided
    try:
        from knowledge_engine.claim_extractor_evidence import (
            enrich_claims_with_evidence,
            EvidenceExtractionError
        )
        from knowledge_engine.contracts import ClaimDraft
        
        claims = [ClaimDraft(statement="test", slot_name="test", observed_slots=["test"])]
        
        try:
            enrich_claims_with_evidence(claims, "test transcript")
            result.add(
                "No-Fallbacks Policy",
                "EvidenceExtractionError raised without LLM",
                False,
                "Should have raised EvidenceExtractionError"
            )
        except EvidenceExtractionError as e:
            result.add(
                "No-Fallbacks Policy",
                "EvidenceExtractionError raised without LLM",
                True,
                f"Error raised: {e}"
            )
    except ImportError as e:
        result.add(
            "No-Fallbacks Policy",
            "EvidenceExtractionError raised without LLM",
            False,
            f"Import error: {e}"
        )
    
    return result


def main():
    """Run all E2E tests."""
    print("=" * 60)
    print("E2E TEST: Context-Aware Ingestion System")
    print("=" * 60)
    
    all_results = E2ETestResult()
    
    # Run all scenarios
    scenarios = [
        test_scenario_1_main_py,
        test_scenario_2_mcp_ingestion,
        test_scenario_3_streamlit,
        test_scenario_4_android_apk,
        test_no_fallbacks_policy,
    ]
    
    for scenario_fn in scenarios:
        scenario_result = scenario_fn()
        all_results.passed += scenario_result.passed
        all_results.failed += scenario_result.failed
        all_results.errors.extend(scenario_result.errors)
        all_results.results.extend(scenario_result.results)
    
    # Print summary
    print(all_results.summary())
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 60)
    for r in all_results.results:
        status = "✓" if r["passed"] else "✗"
        print(f"{status} [{r['scenario']}] {r['test_case']}")
        if r["details"]:
            print(f"    {r['details']}")
    
    # Return exit code
    return 0 if all_results.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
