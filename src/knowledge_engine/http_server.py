"""HTTP Transport for Knowledge Engine MCP Server.

Provides HTTP/JSON API for Android clients that cannot use stdio transport.
Run: python -m knowledge_engine.http_server

Endpoints:
    POST /api/tools/call      - Call an MCP tool
    POST /api/tools/list      - List available tools
    GET  /api/state           - Get engine state
    GET  /api/hardware/requirements - Get 6dfov requirements
    POST /api/hardware/register    - Register device capabilities
    GET  /api/hardware/check/{device_id} - Check device readiness
    GET  /api/hardware/devices     - List registered devices
    GET  /api/capabilities/unused  - List unused capabilities
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from .bootstrap import build_engine_from_env
from .hardware import get_6dfov_components, build_device_capabilities

engine = build_engine_from_env()

# Store for unused capabilities
_unused_capabilities: dict[str, list[dict]] = {}


class MCPHandler(BaseHTTPRequestHandler):
    """HTTP handler for MCP tools."""

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        path = self.path.rstrip("/")

        if path == "/api/state":
            self._handle_state()
        elif path == "/api/hardware/requirements":
            self._handle_hardware_requirements()
        elif path.startswith("/api/hardware/check/"):
            device_id = path.split("/")[-1]
            self._handle_hardware_check(device_id)
        elif path == "/api/hardware/devices":
            self._handle_list_devices()
        elif path == "/api/capabilities/unused":
            self._handle_unused_capabilities()
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_POST(self):
        path = self.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return

        if path == "/api/tools/call":
            self._handle_tool_call(data)
        elif path == "/api/tools/list":
            self._handle_tool_list()
        elif path == "/api/hardware/register":
            self._handle_hardware_register(data)
        elif path == "/api/capabilities/log":
            self._handle_log_capabilities(data)
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def _handle_state(self):
        result = engine.state_snapshot()
        self._set_headers(200)
        self.wfile.write(json.dumps(result, indent=2).encode())

    def _handle_hardware_requirements(self):
        components = get_6dfov_components()
        result = {
            "components": [c.to_dict() for c in components],
            "required_count": sum(1 for c in components if c.requirement.value == "required"),
            "recommended_count": sum(1 for c in components if c.requirement.value == "recommended"),
            "optional_count": sum(1 for c in components if c.requirement.value == "optional"),
        }
        self._set_headers(200)
        self.wfile.write(json.dumps(result, indent=2).encode())

    def _handle_hardware_register(self, data: dict):
        try:
            capabilities = build_device_capabilities(
                device_id=data["device_id"],
                device_name=data["device_name"],
                os_version=data["os_version"],
                app_version=data["app_version"],
                component_status=data["component_status"],
            )

            # Store capabilities
            if not hasattr(engine, '_device_capabilities'):
                engine._device_capabilities = {}
            engine._device_capabilities[data["device_id"]] = capabilities

            # Log unused capabilities
            self._log_unused_capabilities(data["device_id"], capabilities, data.get("unused_capabilities", []))

            result = capabilities.to_dict()
            result["registered"] = True

            self._set_headers(200)
            self.wfile.write(json.dumps(result, indent=2).encode())
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_hardware_check(self, device_id: str):
        if not hasattr(engine, '_device_capabilities'):
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "No devices registered"}).encode())
            return

        capabilities = engine._device_capabilities.get(device_id)
        if capabilities is None:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Device {device_id} not found"}).encode())
            return

        result = capabilities.to_dict()

        # Add recommendations
        recommendations = []
        for c in capabilities.components:
            if c.status.value == "unavailable":
                if c.requirement.value == "required":
                    recommendations.append(f"CRITICAL: {c.name} is required for 6dfov")
                elif c.requirement.value == "recommended":
                    recommendations.append(f"RECOMMENDED: {c.name} improves 6dfov experience")

        result["recommendations"] = recommendations
        result["can_run_6dfov"] = capabilities.is_6dfov_ready

        self._set_headers(200)
        self.wfile.write(json.dumps(result, indent=2).encode())

    def _handle_list_devices(self):
        if not hasattr(engine, '_device_capabilities'):
            self._set_headers(200)
            self.wfile.write(json.dumps({"devices": [], "count": 0}).encode())
            return

        devices = []
        for device_id, caps in engine._device_capabilities.items():
            devices.append({
                "device_id": device_id,
                "device_name": caps.device_name,
                "is_ready": caps.is_6dfov_ready,
                "readiness_score": caps.readiness_score,
                "mode": caps.detectedMode.displayName,
            })

        self._set_headers(200)
        self.wfile.write(json.dumps({"devices": devices, "count": len(devices)}).encode())

    def _handle_tool_call(self, data: dict):
        tool_name = data.get("tool")
        args = data.get("arguments", {})

        if not tool_name:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Missing tool name"}).encode())
            return

        try:
            # Route to appropriate engine method
            result = self._call_tool(tool_name, args)
            self._set_headers(200)
            self.wfile.write(json.dumps(result, indent=2).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _call_tool(self, tool_name: str, args: dict) -> Any:
        """Route tool calls to engine methods."""
        if tool_name == "state_snapshot":
            return engine.state_snapshot()
        elif tool_name == "classify_intent":
            return engine.classify_intent(args.get("text", ""))
        elif tool_name == "search_claims":
            return engine.search_claims(
                query=args.get("query", ""),
                domain=args.get("domain"),
                limit=args.get("limit", 10)
            )
        elif tool_name == "explore_experience":
            return engine.explore_experience(
                query=args.get("query", ""),
                domain=args.get("domain")
            )
        elif tool_name == "ingest_transcript":
            from .contracts import TranscriptInput, ClaimDraft
            transcript_data = args.get("transcript", {})
            transcript = TranscriptInput(**transcript_data)
            return engine.ingest_transcript(transcript)
        elif tool_name == "list_domains":
            return self._list_domains()
        elif tool_name == "get_6dfov_requirements":
            components = get_6dfov_components()
            return {
                "components": [c.to_dict() for c in components],
            }
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _list_domains(self) -> dict:
        """List all knowledge domains."""
        from .graph.neo4j_store import KnowledgeGraphStore
        from .policy import list_policy_domains

        if isinstance(engine.store, KnowledgeGraphStore):
            graph_domains = engine.store.list_domains()
        else:
            graph_domains = sorted({
                t for c in engine.store.claims.values() for t in c.tags if t
            })

        policy_domains = list_policy_domains()
        return {
            "ingested_domains": graph_domains,
            "policy_domains": policy_domains,
            "total": len(graph_domains),
        }

    def _handle_tool_list(self):
        tools = [
            {"name": "state_snapshot", "description": "Get engine state"},
            {"name": "classify_intent", "description": "Classify user intent"},
            {"name": "search_claims", "description": "Vector search over claims"},
            {"name": "explore_experience", "description": "Synthesize knowledge"},
            {"name": "ingest_transcript", "description": "Add new transcript"},
            {"name": "list_domains", "description": "List all domains"},
            {"name": "get_6dfov_requirements", "description": "Get 6dfov hardware requirements"},
        ]
        self._set_headers(200)
        self.wfile.write(json.dumps({"tools": tools}, indent=2).encode())

    def _handle_unused_capabilities(self):
        """Return logged unused capabilities from all devices."""
        all_unused = {}
        for device_id, caps_list in _unused_capabilities.items():
            all_unused[device_id] = caps_list

        self._set_headers(200)
        self.wfile.write(json.dumps({
            "unused_capabilities": all_unused,
            "total_devices": len(all_unused),
            "total_items": sum(len(v) for v in all_unused.values())
        }, indent=2).encode())

    def _handle_log_capabilities(self, data: dict):
        """Log unused capabilities from a device."""
        device_id = data.get("device_id", "unknown")
        unused = data.get("unused_capabilities", [])

        if device_id not in _unused_capabilities:
            _unused_capabilities[device_id] = []

        _unused_capabilities[device_id].extend(unused)

        self._set_headers(200)
        self.wfile.write(json.dumps({
            "logged": True,
            "device_id": device_id,
            "count": len(unused)
        }).encode())

    def _log_unused_capabilities(
        self,
        device_id: str,
        capabilities,
        extra_unused: list[dict]
    ):
        """Log capabilities that the system doesn't fully utilize yet."""
        unused = []

        # Check each component
        for component in capabilities.components:
            if component.status.value == "available":
                # Component is available but may not be fully utilized
                if component.name == "lidar":
                    unused.append({
                        "component": component.name,
                        "status": "available_but_not_integrated",
                        "capabilities": component.capabilities,
                        "note": "LiDAR/depth sensor detected but not yet used for spatial mapping"
                    })
                elif component.name == "gps":
                    unused.append({
                        "component": component.name,
                        "status": "available_but_not_integrated",
                        "capabilities": component.capabilities,
                        "note": "GPS detected but location-based features not yet implemented"
                    })

        # Add extra unused from device
        unused.extend(extra_unused)

        if unused:
            if device_id not in _unused_capabilities:
                _unused_capabilities[device_id] = []
            _unused_capabilities[device_id].extend(unused)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the HTTP server.

    Args:
        host: Bind address (0.0.0.0 for all interfaces)
        port: Port number
    """
    server = HTTPServer((host, port), MCPHandler)
    print(f"Knowledge Engine HTTP Server")
    print(f"Listening on http://{host}:{port}")
    print(f"Android can connect to http://<your-ip>:{port}")
    print(f"Endpoints:")
    print(f"  GET  /api/state")
    print(f"  POST /api/tools/call")
    print(f"  GET  /api/hardware/requirements")
    print(f"  POST /api/hardware/register")
    print(f"  GET  /api/hardware/check/<device_id>")
    print(f"  GET  /api/capabilities/unused")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    run_server()
