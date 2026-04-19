"""
Protocol (interface) definitions for the wrong-way detection system.

FIXES:
  §1.1 (critique) — Tight coupling between detector and OSMRoadResolver.
       The detector now depends only on the RoadProvider Protocol, not on
       the concrete OSMRoadResolver class.  Swap in HERE/Google/local-tile
       providers without touching detector.py.

  §1.3 (critique) — MockRoadResolver in main.py is made first-class by
       implementing this Protocol, documented and testable in one place.

Usage
-----
    from core.interfaces import RoadProvider, PingStream
    class MyCustomResolver:
        def get_best_road_match(...): ...   # satisfies Protocol
"""

from typing import Optional, List, Iterator, Protocol, runtime_checkable

# ── Forward refs (avoid circular imports) ─────────────────
# RoadSegment is defined in osm_resolver; we import only for type hints.
# At runtime the Protocol check uses structural subtyping (duck-typing),
# so no hard import is needed here.

try:
    from core.osm_resolver import RoadSegment  # noqa: F401 (re-exported)
except ImportError:
    pass  # Allow the interfaces module to load standalone


@runtime_checkable
class RoadProvider(Protocol):
    """
    Structural interface for any road-data backend.

    Any object that implements ``get_best_road_match`` satisfies this
    protocol regardless of inheritance.  This lets the detector accept
    OSMRoadResolver, a HERE Maps adapter, a local-tile reader, or a
    MockRoadResolver without code changes.
    """

    def get_best_road_match(
        self,
        lat: float,
        lon: float,
        vehicle_heading: float,
    ) -> Optional["RoadSegment"]:
        """
        Return the best-matching road segment for the given position and
        heading, or None if no road is found.

        Implementations are free to use any matching strategy (Overpass,
        local tiles, HMM, etc.) as long as they return a RoadSegment or None.
        """
        ...


@runtime_checkable
class PingStream(Protocol):
    """
    Structural interface for a source of GPS pings.

    Separates the simulation / real-data layers from the detection pipeline
    so the same detector loop works for both simulated and live data.
    """

    def __iter__(self) -> Iterator:
        """Yield GPSPing objects in chronological order."""
        ...


class MockRoadProvider:
    """
    Canonical offline mock — satisfies RoadProvider.

    Moved from main.py so it can be imported and tested properly.
    Replaces the duplicated MockRoadResolver class.
    """

    def __init__(self, road_bearing: float = 90.0,
                 road_type: str = "primary",
                 road_name: str = "Mock Road (Offline)"):
        self.road_bearing = road_bearing
        self.road_type    = road_type
        self.road_name    = road_name

    def get_best_road_match(
        self,
        lat: float,
        lon: float,
        vehicle_heading: float,
    ) -> Optional["RoadSegment"]:
        from core.osm_resolver import RoadSegment
        return RoadSegment(
            osm_way_id=99999,
            name=self.road_name,
            road_type=self.road_type,
            allowed_direction=self.road_bearing,
            reverse_allowed=False,
            is_oneway=True,
            is_construction=False,
            has_access_restriction=False,
            node_coords=[],
            is_roundabout=False,
            is_junction=False,
            centroid_lat=lat,
            centroid_lon=lon,
        )

    def fetch_roads_near(self, lat: float, lon: float,
                          radius_m: int = 20) -> list:
        return [self.get_best_road_match(lat, lon, 0)]
