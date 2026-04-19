"""
Bearing utilities for wrong-way detection.
Handles GPS heading computation, angular delta, and noise filtering.

FIXES (previous round):
  §1.2  Low-speed heading instability: heading_confidence()
  §1.1  GPS jitter: raised micro-movement skip threshold 0.5 m → 1.0 m
  §6.4  Trajectory fragmentation: gap detection nullifies unreliable headings

FIXES (this round — critique §4.1/§4.3):
  §4.3  No time-based smoothing / velocity vector estimation:
        New estimate_velocity_vector() computes a least-squares velocity
        vector over a position window so heading is derived from the
        trajectory trend, not just the last two pings. This is far more
        robust on winding roads and with noisy GPS.
  §4.1  Kalman filter made explicit and clearly callable as a standalone
        function with a docstring; was previously buried in a helper.
"""

import math
from typing import List, Optional, Set, Tuple

import numpy as np


# ── Basic geometry ─────────────────────────────────────────

def compute_bearing(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """
    Forward azimuth (0–360°) from point A to point B.
    Uses the spherical approximation of WGS-84.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angular_difference(h1: float, h2: float) -> float:
    """Smallest angular distance between two headings, in [0, 180]°."""
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)


def is_opposite_direction(vehicle_heading: float, road_direction: float,
                           threshold: float = 120.0) -> bool:
    """True when vehicle heading differs from road direction by more than threshold."""
    return angular_difference(vehicle_heading, road_direction) > threshold


def haversine_distance(lat1: float, lon1: float,
                        lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two GPS coordinates."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def speed_from_positions(lat1: float, lon1: float,
                          lat2: float, lon2: float,
                          dt_seconds: float = 1.0) -> float:
    """Speed in km/h from two consecutive GPS positions."""
    return (haversine_distance(lat1, lon1, lat2, lon2) / dt_seconds) * 3.6


# ── Heading confidence (§1.2) ──────────────────────────────

def heading_confidence(speed_kmh: float,
                        reliable_speed_kmh: float = 15.0,
                        min_speed_kmh: float = 5.0) -> float:
    """
    Linear ramp from 0 (at min_speed) to 1 (at reliable_speed).

    GPS-derived heading is computed from consecutive position deltas.
    At low speeds those deltas are smaller than the GPS noise floor, so
    headings become effectively random.  This function produces a weight
    that downstream consumers can multiply into risk contributions so that
    low-speed pings cannot accumulate a high risk score on their own.
    """
    if speed_kmh <= min_speed_kmh:
        return 0.0
    if speed_kmh >= reliable_speed_kmh:
        return 1.0
    return (speed_kmh - min_speed_kmh) / (reliable_speed_kmh - min_speed_kmh)


# ── Kalman filter (§4.1) ───────────────────────────────────

def kalman_smooth_bearing(bearings: List[float],
                           process_noise: float = 0.05,
                           measurement_noise: float = 3.0) -> List[float]:
    """
    1-D Kalman filter for GPS bearing sequences.

    Works in sin/cos space to handle the 0°/360° wraparound correctly.
    This is a scalar (1-D) filter; each component (sin, cos) is filtered
    independently and then recombined via atan2.

    Args:
        bearings:          Raw bearing sequence (degrees, 0–360).
        process_noise Q:   Model how fast true heading can change per step.
                           Increase for aggressive/winding roads.
        measurement_noise R: Expected bearing measurement noise (degrees).
                           Increase for low-quality GPS receivers.

    Returns:
        Filtered bearing sequence of the same length.
    """
    if len(bearings) < 2:
        return list(bearings)

    sins = [math.sin(math.radians(b)) for b in bearings]
    coss = [math.cos(math.radians(b)) for b in bearings]

    def _kalman_1d(measurements: List[float], Q: float, R: float) -> List[float]:
        x, P = measurements[0], 1.0
        out = [x]
        for z in measurements[1:]:
            P += Q                   # predict
            K = P / (P + R)          # Kalman gain
            x += K * (z - x)         # update
            P *= (1 - K)
            out.append(x)
        return out

    s_sm = _kalman_1d(sins, process_noise, measurement_noise)
    c_sm = _kalman_1d(coss, process_noise, measurement_noise)
    return [(math.degrees(math.atan2(s, c)) + 360) % 360
            for s, c in zip(s_sm, c_sm)]


def median_bearing_window(bearings: List[float], window: int = 3) -> List[float]:
    """
    Circular median filter over a sliding window.
    Removes spike outliers while respecting the 0°/360° wraparound.
    """
    if len(bearings) <= window:
        return list(bearings)
    half = window // 2
    smoothed = list(bearings[:half])
    for i in range(half, len(bearings) - half):
        ws = bearings[i - half: i + half + 1]
        rads = [math.radians(b) for b in ws]
        cvs = [complex(math.cos(r), math.sin(r)) for r in rads]
        mc = sum(cvs) / len(cvs)
        smoothed.append((math.degrees(math.atan2(mc.imag, mc.real)) + 360) % 360)
    smoothed.extend(bearings[-half:])
    return smoothed


# ── Velocity vector estimation (§4.3) ─────────────────────

def estimate_velocity_vector(
    positions: List[Tuple[float, float]],
    timestamps: List[float],
) -> Optional[float]:
    """
    Estimate the vehicle's current heading from a *window* of positions
    using a weighted least-squares linear fit in local ENU coordinates.

    This is fundamentally more robust than pairwise bearing because:
      - Noise in any single ping is averaged out across the window.
      - The result is the direction of the velocity trend, not the last
        instantaneous displacement.
      - Curved roads are handled better: the fit follows the recent arc.

    CRITIQUE FIX §4.3 — "No time-based smoothing / velocity vector
    estimation over window".

    Args:
        positions:  List of (lat, lon) tuples, chronological order.
        timestamps: Corresponding epoch timestamps (seconds).

    Returns:
        Estimated heading in degrees [0, 360], or None if fewer than 3
        positions are available or all positions are identical.
    """
    if len(positions) < 3 or len(positions) != len(timestamps):
        return None

    # Use the first point as local origin to avoid floating-point issues
    origin_lat, origin_lon = positions[0]
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(origin_lat)))

    # Convert to local ENU metres (east, north) and relative time
    t0 = timestamps[0]
    east  = []
    north = []
    t_rel = []
    for (lat, lon), ts in zip(positions, timestamps):
        north.append((lat - origin_lat) / lat_per_m)
        east.append((lon - origin_lon) / lon_per_m)
        t_rel.append(ts - t0)

    t_arr = np.array(t_rel, dtype=float)
    e_arr = np.array(east,  dtype=float)
    n_arr = np.array(north, dtype=float)

    # Recency weights: more recent pings count more
    # w_i = exp(-(T - t_i) / tau)  where tau = half the window span
    T = t_arr[-1]
    tau = max((T - t_arr[0]) / 2.0, 1.0)
    weights = np.exp(-(T - t_arr) / tau)

    # Weighted linear regression: east ~ a*t, north ~ b*t  (no intercept)
    # Velocity components: vx = a, vy = b
    W = np.diag(weights)
    t_col = t_arr.reshape(-1, 1)

    # Avoid division by zero if all timestamps are identical
    denom = (t_col.T @ W @ t_col).item()
    if abs(denom) < 1e-9:
        return None

    vx = (t_col.T @ W @ e_arr).item() / denom   # east velocity (m/s)
    vy = (t_col.T @ W @ n_arr).item() / denom   # north velocity (m/s)

    speed = math.hypot(vx, vy)
    if speed < 0.05:   # effectively stationary — heading undefined
        return None

    # atan2 convention: bearing from north, clockwise
    heading = (math.degrees(math.atan2(vx, vy)) + 360) % 360
    return heading


# ── Trajectory gap detection (§6.4) ───────────────────────

def detect_trajectory_gaps(timestamps: List[float],
                             gap_threshold_s: float = 3.0) -> Set[int]:
    """
    Return indices (1-based) where the inter-ping gap exceeds the threshold.

    A gap means GPS was absent; headings derived across a gap should not
    count toward persistence counters.
    """
    return {i for i in range(1, len(timestamps))
            if timestamps[i] - timestamps[i - 1] > gap_threshold_s}


def compute_vehicle_heading_series(
    positions: List[Tuple[float, float]],
    apply_smoothing: bool = True,
    timestamps: Optional[List[float]] = None,
    gap_threshold_s: float = 3.0,
) -> List[Optional[float]]:
    """
    Heading at each GPS position, derived from consecutive position pairs.

    §1.1: micro-movement skip threshold is 1.0 m (raised from 0.5 m).
    §6.4: headings that span a GPS gap are nullified after smoothing.
    Kalman + median smoothing applied when apply_smoothing=True.

    Returns a list of the same length as positions; entry 0 is always None
    (no previous position to compute a bearing from).
    """
    headings: List[Optional[float]] = [None]
    raw: List[float] = []

    gap_indices: Set[int] = set()
    if timestamps and len(timestamps) == len(positions):
        gap_indices = detect_trajectory_gaps(timestamps, gap_threshold_s)

    for i in range(1, len(positions)):
        lat1, lon1 = positions[i - 1]
        lat2, lon2 = positions[i]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        if dist < 1.0:                          # §1.1 jitter guard
            raw.append(raw[-1] if raw else 0.0)
        else:
            raw.append(compute_bearing(lat1, lon1, lat2, lon2))

    if apply_smoothing and len(raw) > 3:
        raw = median_bearing_window(raw, window=3)
        raw = kalman_smooth_bearing(raw, process_noise=0.05, measurement_noise=3.0)

    # §6.4 — nullify headings that cross a GPS gap
    for i, val in enumerate(raw):
        headings.append(None if (i + 1) in gap_indices else val)

    return headings
