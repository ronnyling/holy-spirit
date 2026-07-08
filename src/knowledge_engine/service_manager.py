"""Service Manager — Auto-start and health check for Neo4j and Ollama.

Manages service lifecycle: detects ports, kills conflicting processes,
starts services, and waits for readiness. Integrates with bootstrap
for seamless engine initialization.

Usage:
    from knowledge_engine.service_manager import ServiceManager

    manager = ServiceManager()
    results = manager.ensure_all_services()
    # Proceed with engine initialization
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service status values."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"


@dataclass
class ServiceResult:
    """Result of a service operation."""
    service: str
    status: ServiceStatus
    message: str
    pid: int | None = None


class ServiceManager:
    """Manages Neo4j and Ollama service lifecycle.

    Features:
    - Port detection and conflict resolution
    - Service startup with readiness waiting
    - Health check integration
    - Configurable paths and ports
    """

    # Default ports
    NEO4J_PORT = 7474  # HTTP
    NEO4J_BOLT_PORT = 7687  # Bolt protocol
    OLLAMA_PORT = 11434

    def __init__(
        self,
        neo4j_path: str | None = None,
        neo4j_uri: str | None = None,
        ollama_path: str | None = None,
    ):
        """Initialize service manager.

        Args:
            neo4j_path: Path to neo4j binary (e.g., "C:\\neo4j\\bin\\neo4j.bat")
            neo4j_uri: Neo4j URI (e.g., "bolt://localhost:7687")
            ollama_path: Path to ollama binary (default: "ollama" in PATH)
        """
        self.neo4j_path = neo4j_path or os.environ.get("KE_NEO4J_PATH", "")
        self.neo4j_uri = neo4j_uri or os.environ.get("KE_NEO4J_URI", "")
        self.ollama_path = ollama_path or os.environ.get("KE_OLLAMA_PATH", "ollama")

        # Extract port from URI if provided
        if self.neo4j_uri:
            try:
                # bolt://localhost:7687 -> 7687
                port_str = self.neo4j_uri.rsplit(":", 1)[-1].rstrip("/")
                self.neo4j_bolt_port = int(port_str)
            except (ValueError, IndexError):
                self.neo4j_bolt_port = self.NEO4J_BOLT_PORT
        else:
            self.neo4j_bolt_port = self.NEO4J_BOLT_PORT

    def check_port(self, port: int) -> bool:
        """Check if a port is in use.

        Args:
            port: Port number to check

        Returns:
            True if port is in use, False if available
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                return s.connect_ex(("localhost", port)) == 0
        except Exception:
            return False

    def get_port_pid(self, port: int) -> int | None:
        """Get PID of process using the port.

        Args:
            port: Port number

        Returns:
            PID if found, None otherwise
        """
        try:
            # Windows: netstat -ano | findstr :PORT
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    # Extract PID (last column)
                    parts = line.split()
                    if parts:
                        return int(parts[-1])
        except Exception as e:
            logger.debug(f"Failed to get PID for port {port}: {e}")
        return None

    def kill_port(self, port: int) -> list[int]:
        """Kill all processes using the specified port.

        Args:
            port: Port number to free

        Returns:
            List of PIDs that were killed
        """
        killed_pids = []
        pid = self.get_port_pid(port)

        if pid:
            try:
                # Windows: taskkill /PID <pid> /F
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    timeout=5,
                )
                killed_pids.append(pid)
                logger.info(f"Killed PID {pid} on port {port}")
            except Exception as e:
                logger.warning(f"Failed to kill PID {pid}: {e}")

        return killed_pids

    def wait_for_port(
        self,
        port: int,
        timeout: float = 30.0,
        interval: float = 0.5,
    ) -> bool:
        """Wait for a port to become available (service ready).

        Args:
            port: Port to wait for
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds

        Returns:
            True if port is ready, False if timeout
        """
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            if self.check_port(port):
                return True
            time.sleep(interval)

        return False

    def start_neo4j(self) -> ServiceResult:
        """Start Neo4j database.

        Returns:
            ServiceResult with status and details
        """
        if not self.neo4j_path:
            return ServiceResult(
                service="neo4j",
                status=ServiceStatus.NOT_CONFIGURED,
                message="KE_NEO4J_PATH not configured",
            )

        neo4j_path = Path(self.neo4j_path)
        if not neo4j.exists():
            return ServiceResult(
                service="neo4j",
                status=ServiceStatus.ERROR,
                message=f"Neo4j not found at {self.neo4j_path}",
            )

        # Check if already running
        if self.check_port(self.neo4j_bolt_port):
            return ServiceResult(
                service="neo4j",
                status=ServiceStatus.RUNNING,
                message=f"Neo4j already running on port {self.neo4j_bolt_port}",
            )

        # Kill any process on the port
        if self.check_port(self.neo4j_bolt_port):
            self.kill_port(self.neo4j_bolt_port)
            time.sleep(1)

        # Start Neo4j
        try:
            subprocess.Popen(
                [str(neo4j_path), "console"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            logger.info("Starting Neo4j...")
        except Exception as e:
            return ServiceResult(
                service="neo4j",
                status=ServiceStatus.ERROR,
                message=f"Failed to start Neo4j: {e}",
            )

        # Wait for readiness
        if self.wait_for_port(self.neo4j_bolt_port, timeout=30):
            return ServiceResult(
                service="neo4j",
                status=ServiceStatus.RUNNING,
                message=f"Neo4j started on port {self.neo4j_bolt_port}",
            )
        else:
            return ServiceResult(
                service="neo4j",
                status=ServiceStatus.ERROR,
                message=f"Neo4j failed to start within 30s on port {self.neo4j_bolt_port}",
            )

    def start_ollama(self) -> ServiceResult:
        """Start Ollama embedding server.

        Returns:
            ServiceResult with status and details
        """
        # Check if already running
        if self.check_port(self.OLLAMA_PORT):
            return ServiceResult(
                service="ollama",
                status=ServiceStatus.RUNNING,
                message=f"Ollama already running on port {self.OLLAMA_PORT}",
            )

        # Check if ollama is available
        try:
            subprocess.run(
                [self.ollama_path, "--version"],
                capture_output=True,
                timeout=5,
            )
        except FileNotFoundError:
            return ServiceResult(
                service="ollama",
                status=ServiceStatus.NOT_CONFIGURED,
                message="Ollama not found in PATH",
            )
        except Exception as e:
            return ServiceResult(
                service="ollama",
                status=ServiceStatus.ERROR,
                message=f"Ollama check failed: {e}",
            )

        # Kill any process on the port
        if self.check_port(self.OLLAMA_PORT):
            self.kill_port(self.OLLAMA_PORT)
            time.sleep(1)

        # Start Ollama
        try:
            subprocess.Popen(
                [self.ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            logger.info("Starting Ollama...")
        except Exception as e:
            return ServiceResult(
                service="ollama",
                status=ServiceStatus.ERROR,
                message=f"Failed to start Ollama: {e}",
            )

        # Wait for readiness
        if self.wait_for_port(self.OLLAMA_PORT, timeout=15):
            return ServiceResult(
                service="ollama",
                status=ServiceStatus.RUNNING,
                message=f"Ollama started on port {self.OLLAMA_PORT}",
            )
        else:
            return ServiceResult(
                service="ollama",
                status=ServiceStatus.ERROR,
                message=f"Ollama failed to start within 15s on port {self.OLLAMA_PORT}",
            )

    def ensure_all_services(self) -> dict[str, ServiceResult]:
        """Ensure all configured services are running.

        Returns:
            Dict mapping service names to their results
        """
        results = {}

        # Neo4j
        results["neo4j"] = self.start_neo4j()

        # Ollama
        results["ollama"] = self.start_ollama()

        # Log summary
        for name, result in results.items():
            status_symbol = {
                ServiceStatus.RUNNING: "✓",
                ServiceStatus.STOPPED: "○",
                ServiceStatus.STARTING: "◌",
                ServiceStatus.ERROR: "✗",
                ServiceStatus.NOT_CONFIGURED: "-",
            }.get(result.status, "?")

            logger.info(f"  {status_symbol} {name}: {result.message}")

        return results

    def health_check(self) -> dict[str, dict]:
        """Check health of all services.

        Returns:
            Dict with health status for each service
        """
        health = {}

        # Neo4j
        neo4j_running = self.check_port(self.neo4j_bolt_port)
        health["neo4j"] = {
            "status": "healthy" if neo4j_running else "unhealthy",
            "port": self.neo4j_bolt_port,
        }

        # Ollama
        ollama_running = self.check_port(self.OLLAMA_PORT)
        health["ollama"] = {
            "status": "healthy" if ollama_running else "unhealthy",
            "port": self.OLLAMA_PORT,
        }

        # Overall
        health["overall"] = "healthy" if all(
            h["status"] == "healthy" for h in health.values() if isinstance(h, dict)
        ) else "degraded"

        return health
