"""Hardware Capability Models for 6dfov Integration.

Defines required and optional hardware components for 6dfov streaming,
and provides MCP tools to check device capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComponentRequirement(str, Enum):
    """Hardware component requirement level."""
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class ComponentStatus(str, Enum):
    """Hardware component availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class HardwareComponent:
    """A single hardware component specification."""
    name: str
    description: str
    requirement: ComponentRequirement
    status: ComponentStatus = ComponentStatus.UNKNOWN
    capabilities: dict[str, Any] = field(default_factory=dict)
    min_spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "requirement": self.requirement.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "min_spec": self.min_spec,
        }


@dataclass
class DeviceCapabilities:
    """Complete device capability profile for 6dfov."""
    device_id: str
    device_name: str
    os_version: str
    app_version: str
    components: list[HardwareComponent] = field(default_factory=list)

    @property
    def is_6dfov_ready(self) -> bool:
        """Check if all REQUIRED components are available."""
        return all(
            c.status == ComponentStatus.AVAILABLE
            for c in self.components
            if c.requirement == ComponentRequirement.REQUIRED
        )

    @property
    def missing_required(self) -> list[HardwareComponent]:
        """List required components that are not available."""
        return [
            c for c in self.components
            if c.requirement == ComponentRequirement.REQUIRED
            and c.status != ComponentStatus.AVAILABLE
        ]

    @property
    def readiness_score(self) -> float:
        """Score 0-100 based on available components."""
        if not self.components:
            return 0.0

        total_weight = 0.0
        earned_weight = 0.0

        for c in self.components:
            weight = {
                ComponentRequirement.REQUIRED: 3.0,
                ComponentRequirement.RECOMMENDED: 2.0,
                ComponentRequirement.OPTIONAL: 1.0,
            }[c.requirement]

            total_weight += weight
            if c.status == ComponentStatus.AVAILABLE:
                earned_weight += weight

        return round((earned_weight / total_weight) * 100, 1) if total_weight > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "os_version": self.os_version,
            "app_version": self.app_version,
            "is_6dfov_ready": self.is_6dfov_ready,
            "readiness_score": self.readiness_score,
            "missing_required": [c.name for c in self.missing_required],
            "components": [c.to_dict() for c in self.components],
        }


# ---------------------------------------------------------------------------
# 6dfov Component Registry
# ---------------------------------------------------------------------------

SIXDFOV_COMPONENTS = [
    HardwareComponent(
        name="camera",
        description="Camera for visual input and AR overlay",
        requirement=ComponentRequirement.REQUIRED,
        min_spec={
            "resolution": "720p",
            "fps": 30,
            "autofocus": True,
        },
    ),
    HardwareComponent(
        name="gyroscope",
        description="Gyroscope for rotation tracking (pitch, yaw, roll)",
        requirement=ComponentRequirement.REQUIRED,
        min_spec={
            "axes": 3,
            "sampling_rate_hz": 60,
        },
    ),
    HardwareComponent(
        name="accelerometer",
        description="Accelerometer for linear movement tracking",
        requirement=ComponentRequirement.REQUIRED,
        min_spec={
            "axes": 3,
            "sampling_rate_hz": 60,
        },
    ),
    HardwareComponent(
        name="magnetometer",
        description="Magnetometer for compass heading",
        requirement=ComponentRequirement.REQUIRED,
        min_spec={
            "axes": 3,
        },
    ),
    HardwareComponent(
        name="imu",
        description="Inertial Measurement Unit (combined gyro + accel)",
        requirement=ComponentRequirement.RECOMMENDED,
        min_spec={
            "fusion": True,
        },
    ),
    HardwareComponent(
        name="gps",
        description="GPS for location context",
        requirement=ComponentRequirement.OPTIONAL,
        min_spec={
            "accuracy_meters": 10,
        },
    ),
    HardwareComponent(
        name="lidar",
        description="LiDAR or depth sensor for spatial mapping",
        requirement=ComponentRequirement.OPTIONAL,
        min_spec={
            "range_meters": 5,
        },
    ),
    HardwareComponent(
        name="network",
        description="Network connectivity for MCP communication",
        requirement=ComponentRequirement.REQUIRED,
        min_spec={
            "type": "wifi_or_cellular",
        },
    ),
]


def get_6dfov_components() -> list[HardwareComponent]:
    """Return the list of components required for 6dfov."""
    return [HardwareComponent(
        name=c.name,
        description=c.description,
        requirement=c.requirement,
        min_spec=c.min_spec,
    ) for c in SIXDFOV_COMPONENTS]


def check_capability(
    component_name: str,
    available: bool,
    capabilities: dict[str, Any] | None = None,
) -> HardwareComponent:
    """Create a checked component status."""
    for c in SIXDFOV_COMPONENTS:
        if c.name == component_name:
            return HardwareComponent(
                name=c.name,
                description=c.description,
                requirement=c.requirement,
                status=ComponentStatus.AVAILABLE if available else ComponentStatus.UNAVAILABLE,
                capabilities=capabilities or {},
                min_spec=c.min_spec,
            )
    return HardwareComponent(
        name=component_name,
        description="Unknown component",
        requirement=ComponentRequirement.OPTIONAL,
        status=ComponentStatus.UNKNOWN,
    )


def build_device_capabilities(
    device_id: str,
    device_name: str,
    os_version: str,
    app_version: str,
    component_status: dict[str, dict[str, Any]],
) -> DeviceCapabilities:
    """Build a complete device capability profile.

    Args:
        device_id: Unique device identifier
        device_name: Human-readable device name
        os_version: OS version string
        app_version: App version string
        component_status: Dict of component_name -> {available: bool, capabilities: dict}
    """
    components = []
    for c in SIXDFOV_COMPONENTS:
        status_info = component_status.get(c.name, {})
        available = status_info.get("available", False)
        caps = status_info.get("capabilities", {})

        components.append(HardwareComponent(
            name=c.name,
            description=c.description,
            requirement=c.requirement,
            status=ComponentStatus.AVAILABLE if available else ComponentStatus.UNAVAILABLE,
            capabilities=caps,
            min_spec=c.min_spec,
        ))

    return DeviceCapabilities(
        device_id=device_id,
        device_name=device_name,
        os_version=os_version,
        app_version=app_version,
        components=components,
    )
