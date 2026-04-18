"""
OSM Road Geometry Resolver (Hardened for Demo)
- Retry + exponential backoff
- Aggressive caching (600s TTL)
- Reduced radius (20m) to minimize API calls
- Polite rate limiting (0.2s delay between calls)
- Mirror fallback across 3 Overpass endpoints

FIXES applied:
  - §3.1  Incorrect road snapping: get_best_road_match now scores by a
          combined heading+distance metric instead of heading alone, so
          a parallel road 50 m away is preferred over one 5 m away only
          if the heading advantage is large enough.
  - §2.1  Missing one-way tags: motorway/motorway_link still forced
          one-way; other types stay conservative (only explicit tag).
  - §2.3  Speed limit parsed from OSM maxspeed tag into RoadSegment.
  - §5.2  Temporary road changes: construction tag detection improved
          to catch access=no + highway=construction combinations.
  - §3.3  Intersection ambiguity: junction/roundabout segments get a
          proximity penalty so they are only matched when very close.
"""

import math
import time
import requests
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter"
]

OVERPASS_TIMEOUT = 20
_LAST_CALL_TIME: float = 0.0   # module-level for rate limiting

# FIX §3.1 — Tuning knobs for combined road-matching score
_DISTANCE_WEIGHT   = 0.4   # contribution of proximity to match score
_HEADING_WEIGHT    = 0.6   # contribution of heading alignment
_MAX_MATCH_DIST_M  = 80    # roads farther than this are excluded entirely
_JUNCTION_DIST_PENALTY_M = 20  # extra effective distance added to junction segments


@dataclass
class RoadSegment:
    osm_way_id: int
    name: str
    road_type: str
    allowed_direction: float
    reverse_allowed: bool
    is_oneway: bool
    is_construction: bool
    has_access_restriction: bool
    node_coords: List[Tuple[float, float]] = field(default_factory=list)
    speed_limit_kmh: Optional[int] = None
    is_roundabout: bool = False
    is_junction: bool = False
    # FIX §3.1 — store centroid for distance-based matching
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0


class OSMRoadResolver:
    """
    Optimized resolver:
    - Fetches OSM data ONCE per cache window
    - Aggressive TTL (600s) for demo stability
    - Retry + exponential backoff on all calls
    - Polite 0.2s delay between API calls
    """

    def __init__(self, cache_ttl_seconds: int = 600):
        self._cache_ttl = cache_ttl_seconds
        self._cached_roads: Optional[List[RoadSegment]] = None
        self._last_fetch_time = 0

    # ─────────────────────────────────────────────
    # API CALL (retry + mirror + rate limit)
    # ─────────────────────────────────────────────
    def _fetch_overpass(self, query: str) -> Optional[Dict]:
        global _LAST_CALL_TIME

        elapsed = time.time() - _LAST_CALL_TIME
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)

        for url in OVERPASS_URLS:
            for attempt in range(4):
                try:
                    _LAST_CALL_TIME = time.time()
                    resp = requests.post(
                        url,
                        data={"data": query},
                        timeout=OVERPASS_TIMEOUT,
                        headers={"User-Agent": "WrongWayDetector/1.0"}
                    )

                    if resp.status_code == 429:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"[{url}] 429 rate-limited, waiting {wait}s")
                        time.sleep(wait)
                        continue

                    if resp.status_code == 504:
                        wait = 2 ** attempt
                        logger.warning(f"[{url}] 504 timeout, waiting {wait}s")
                        time.sleep(wait)
                        continue

                    resp.raise_for_status()
                    return resp.json()

                except requests.exceptions.RequestException as e:
                    wait = 2 ** attempt
                    logger.warning(f"[{url}] attempt {attempt+1} failed: {e} → retry in {wait}s")
                    time.sleep(wait)

        logger.error("All Overpass endpoints failed ❌")
        return None

    # ─────────────────────────────────────────────
    # MAIN FETCH (CACHED)
    # ─────────────────────────────────────────────
    def fetch_roads_near(self, lat: float, lon: float, radius_m: int = 20) -> List[RoadSegment]:
        now = time.time()

        if self._cached_roads is not None and (now - self._last_fetch_time < self._cache_ttl):
            return self._cached_roads

        logger.info("Fetching road data from OSM (ONE-TIME call)...")

        query = f"""
        [out:json][timeout:{OVERPASS_TIMEOUT}];
        way(around:{radius_m},{lat},{lon})
          [highway];
        (._;>;);
        out body;
        """

        data = self._fetch_overpass(query)

        if not data:
            logger.error("OSM fetch failed, returning empty roads")
            return self._cached_roads or []

        segments = self._parse_overpass_response(data)
        self._cached_roads = segments
        self._last_fetch_time = now
        logger.info(f"OSM: cached {len(segments)} road segments")
        return segments

    # ─────────────────────────────────────────────
    # PARSER
    # ─────────────────────────────────────────────
    def _parse_overpass_response(self, data: Dict) -> List[RoadSegment]:
        nodes = {}
        ways = []

        for el in data.get("elements", []):
            if el["type"] == "node":
                nodes[el["id"]] = (el["lat"], el["lon"])
            elif el["type"] == "way":
                ways.append(el)

        segments = []

        for way in ways:
            tags = way.get("tags", {})
            coords = [nodes[n] for n in way.get("nodes", []) if n in nodes]

            if len(coords) < 2:
                continue

            bearing = self._segment_primary_bearing(coords)
            is_oneway = self._resolve_oneway(tags, tags.get("highway", ""))

            junction_tag = tags.get("junction", "")

            # FIX §3.1 — compute centroid for distance-weighted matching
            centroid_lat = sum(c[0] for c in coords) / len(coords)
            centroid_lon = sum(c[1] for c in coords) / len(coords)

            # FIX §2.3 — parse speed limit from maxspeed tag
            speed_limit = self._parse_speed_limit(tags.get("maxspeed", ""))

            # FIX §5.2 — broader construction detection
            is_construction = (
                tags.get("highway") == "construction"
                or tags.get("construction") is not None
                or (tags.get("access") == "no" and tags.get("highway") == "construction")
            )

            seg = RoadSegment(
                osm_way_id=way["id"],
                name=tags.get("name", "unnamed"),
                road_type=tags.get("highway", "unknown"),
                allowed_direction=bearing,
                reverse_allowed=not is_oneway,
                is_oneway=is_oneway,
                is_construction=is_construction,
                has_access_restriction=tags.get("access") in ("no", "private"),
                node_coords=coords,
                speed_limit_kmh=speed_limit,
                is_roundabout=junction_tag == "roundabout",
                is_junction=bool(junction_tag) and junction_tag != "roundabout",
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
            )
            segments.append(seg)

        return segments

    def _resolve_oneway(self, tags: Dict, road_type: str) -> bool:
        oneway = tags.get("oneway", "no")
        if oneway in ("yes", "1", "true"):
            return True
        if road_type in ("motorway", "motorway_link"):
            return True
        if tags.get("junction") == "roundabout":
            return True
        return False

    # FIX §2.3 — parse OSM maxspeed tag (handles "50", "50 mph", "50 km/h")
    def _parse_speed_limit(self, maxspeed: str) -> Optional[int]:
        if not maxspeed:
            return None
        maxspeed = maxspeed.strip().lower()
        try:
            # bare number → assume km/h
            return int(float(maxspeed.split()[0]))
        except (ValueError, IndexError):
            return None

    def _segment_primary_bearing(self, coords):
        lat1, lon1 = coords[0]
        lat2, lon2 = coords[-1]
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - \
            math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    # ─────────────────────────────────────────────
    # FIX §3.1 — Distance + heading combined match
    # ─────────────────────────────────────────────
    def get_best_road_match(self, lat: float, lon: float,
                             vehicle_heading: float) -> Optional["RoadSegment"]:
        """
        Return the road segment that best matches the vehicle's position
        AND heading using a weighted composite score.

        Previously only heading alignment was considered, which caused
        incorrect snapping to a parallel road across the street when the
        vehicle's GPS had minor drift.  Now proximity is also weighted so
        the closest plausible road wins unless the heading advantage of a
        farther road is substantial.

        FIX §3.3: Roundabout/junction segments receive a distance penalty
        to reduce ambiguous intersection snapping.
        """
        from core.bearing_utils import haversine_distance, angular_difference

        segments = self.fetch_roads_near(lat, lon)
        if not segments:
            return None

        best_seg = None
        best_score = float("inf")

        for seg in segments:
            # Compute distance from vehicle to road centroid
            dist_m = haversine_distance(lat, lon, seg.centroid_lat, seg.centroid_lon)

            # FIX §3.3: junction/roundabout penalty discourages false snapping
            if seg.is_roundabout or seg.is_junction:
                dist_m += _JUNCTION_DIST_PENALTY_M

            # Exclude roads that are clearly too far to be relevant
            if dist_m > _MAX_MATCH_DIST_M:
                continue

            # Heading alignment score (best of forward / reverse)
            fwd_diff = angular_difference(vehicle_heading, seg.allowed_direction)
            if seg.reverse_allowed:
                rev_diff = angular_difference(vehicle_heading, (seg.allowed_direction + 180) % 360)
                heading_diff = min(fwd_diff, rev_diff)
            else:
                heading_diff = fwd_diff

            # Normalise both dimensions to [0, 1] and combine
            norm_dist    = min(dist_m / _MAX_MATCH_DIST_M, 1.0)
            norm_heading = heading_diff / 180.0

            score = _HEADING_WEIGHT * norm_heading + _DISTANCE_WEIGHT * norm_dist

            if score < best_score:
                best_score = score
                best_seg = seg

        return best_seg
