"""
Wrong-Way Detection — core package.
Public API exports for external use.
"""
from core.config import DetectorConfig, ResolverConfig, SimulatorConfig
from core.interfaces import RoadProvider, MockRoadProvider
from core.detector import WrongWayDetector, WrongWayAlert, VehicleState
from core.osm_resolver import OSMRoadResolver, RoadSegment

__all__ = [
    "DetectorConfig", "ResolverConfig", "SimulatorConfig",
    "RoadProvider", "MockRoadProvider",
    "WrongWayDetector", "WrongWayAlert", "VehicleState",
    "OSMRoadResolver", "RoadSegment",
]
