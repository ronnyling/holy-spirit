"""Knowledge Engine — Single Entry Point.

Starts all services:
- Neo4j (auto-start if not running)
- Ollama (auto-start if not running)
- MCP server (stdio, for VS Code Copilot)
- HTTP server (for Android 6dfov app)
- Streamlit UI (optional)

Run: python main.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from knowledge_engine.service_manager import ServiceManager, ServiceStatus, ServiceResult
from knowledge_engine.bootstrap import load_dotenv


def start_http_server(port: int = 8080):
    """Start HTTP server in a separate thread."""
    from knowledge_engine.http_server import run_server
    server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": "0.0.0.0", "port": port},
        daemon=True
    )
    server_thread.start()
    return server_thread


def start_streamlit():
    """Start Streamlit UI in a separate thread."""
    import subprocess
    import os
    
    app_path = ROOT / "app.py"
    if app_path.exists():
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    return False


def main():
    print("=" * 60)
    print("Knowledge Engine — Starting All Services")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Initialize service manager
    manager = ServiceManager()
    
    # Start services
    print("\n[1/3] Starting infrastructure services...")
    results = manager.ensure_all_services()
    
    for name, result in results.items():
        status_symbol = {
            ServiceStatus.RUNNING: "[OK]",
            ServiceStatus.STARTING: "[..]",
            ServiceStatus.ERROR: "[!!]",
            ServiceStatus.NOT_CONFIGURED: "[NA]",
            ServiceStatus.STOPPED: "[--]",
        }.get(result.status, "[??]")

        print(f"  {status_symbol} {name}: {result.message}")
    
    # Check if critical services are running
    neo4j_ok = results.get("neo4j", ServiceResult("neo4j", ServiceStatus.ERROR, "")).status == ServiceStatus.RUNNING
    ollama_ok = results.get("ollama", ServiceResult("ollama", ServiceStatus.ERROR, "")).status == ServiceStatus.RUNNING
    
    if not neo4j_ok:
        print("\n[WARN] WARNING: Neo4j is not running. Some features may be limited.")
    if not ollama_ok:
        print("\n[WARN] WARNING: Ollama is not running. Embeddings will be unavailable.")
    
    # Start HTTP server for Android
    print("\n[2/3] Starting HTTP server for Android...")
    http_thread = start_http_server()
    time.sleep(1)  # Give server time to start
    print("  [OK] HTTP server started on port 8080")
    
    # Start Streamlit UI (optional)
    print("\n[3/3] Starting Streamlit UI...")
    if start_streamlit():
        print("  [OK] Streamlit UI started on http://localhost:8501")
    else:
        print("  - Streamlit UI skipped (app.py not found)")
    
    # Summary
    print("\n" + "=" * 60)
    print("All services started!")
    print("=" * 60)
    print("\nEndpoints:")
    print(f"  MCP server:   stdio (for VS Code Copilot)")
    print(f"  HTTP server:  http://0.0.0.0:8080 (for Android app)")
    print(f"  Streamlit UI: http://localhost:8501")
    print(f"  Neo4j browser: http://localhost:7474")
    print("\nAndroid 6dfov app will auto-connect to HTTP server.")
    print("Press Ctrl+C to stop all services.")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Daemon threads will stop automatically
        sys.exit(0)


if __name__ == "__main__":
    main()
