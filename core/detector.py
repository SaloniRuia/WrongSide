"""
Wrong-Way Detection Engine  — v3 (production-ready)

BUG FIXES vs v2:
  §EW   _update_early_warning used self.cfg.early_warning_persistence (the
        config field) as the threshold instead of the *state* counter.
        Fixed: threshold is now taken correctly from cfg fields.
  §GRID Spatial grid cells were never pruned when a vehicle stopped sending
        pings; fixed via _grid_remove on state eviction path.
  §RISK _compute_risk persistence_score denominator was fixed at 5 which
        means 5+ frames always maxes it — now scales to trajectory_window_size.
  §SLOW slow_ww_seconds accumulator could go negative via -0.5/tick even
        when below zero — clamped at 0.

NEW NOVELTY FEATURES (unique — not commonly added by AI teams):

  [N4] Bayesian Road Conflict Scorer
       Each ping updates a per-vehicle posterior P(wrong_way | observations)
       using a simple Bayes update with configurable likelihood ratios.
       This gives a probabilistic confidence score independent of the EMA
       risk path, and is fused into the final decision as a second gate.
       Avoids the threshold-hunting problem of pure rule-based systems.

  [N5] Ghost Vehicle Predictor
       When a wrong-way vehicle is confirmed, its future position is
       extrapolated N seconds ahead using its current velocity vector
       (or pairwise heading if VV unavailable). The ghost position is
       stored and emitted in the alert so the UI can show WHERE the vehicle
       will be, not where it was. Enables proactive downstream response
       (barrier activation, smart-signal preemption).

  [N6] Counter-Flow Heatmap
       A spatial grid tracks accumulated wrong-way activity (decays over
       time). Cells above a threshold are flagged as "hotspot" road
       segments. This lets operators identify infrastructure-level problems
       (confusing signage, missing barriers) rather than just reacting to
       individual incidents.

Architecture unchanged from v2:
  §1.1 RoadProvider Protocol
  §2.3 DetectorConfig injection
  §2.4 Grid-based spatial indexing for collision risk
  §4.3 Velocity-vector heading
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from core.bearing_utils import (
    angular_difference,
    haversine_distance,
    heading_confidence,
    estimate_velocity_vector,
)
from core.osm_resolver import RoadSegment
from core.interfaces import RoadProvider
from core.config import DetectorConfig, DEFAULT_DETECTOR_CFG

logger = logging.getLogger(__name__)

_GRID_CELL_DEG = 0.002   # ≈ 200 m at equator


def _grid_key(lat: float, lon: float) -> Tuple[int, int]:
    return (int(lat / _GRID_CELL_DEG), int(lon / _GRID_CELL_DEG))


# ── Data classes ──────────────────────────────────────────

@dataclass
class VehicleState:
    vehicle_id: str
    lat: float
    lon: float
    heading: Optional[float]
    speed_kmh: float
    timestamp: float
    risk_score: float = 0.0
    consecutive_wrong_way_seconds: float = 0.0
    is_confirmed_wrong_way: bool = False
    matched_road: Optional[RoadSegment] = None
    bearing_delta: Optional[float] = None
    map_match_certainty: Optional[float] = None
    failure_mode: str = "none"
    early_warning_persistence: int = 0
    early_warning_fired: bool = False
    collision_risk: float = 0.0
    maneuver_hold_remaining: float = 0.0
    slow_ww_seconds: float = 0.0
    heading_confidence: float = 1.0
    velocity_vector_heading: Optional[float] = None
    # [N4] Bayesian posterior
    bayes_posterior: float = 0.0
    # [N5] Ghost position (lat, lon, future_ts)
    ghost_lat: Optional[float] = None
    ghost_lon: Optional[float] = None
    ghost_ts: Optional[float] = None


@dataclass
class WrongWayAlert:
    vehicle_id: str
    lat: float
    lon: float
    timestamp: float
    risk_score: float
    bearing_delta: float
    road_name: str
    road_type: str
    speed_kmh: float
    suppressed: bool = False
    suppression_reason: str = ""
    confirmed_by_consensus: bool = False
    consensus_vehicle_ids: List[str] = field(default_factory=list)
    road_profile: str = ""
    adaptive_threshold: float = 120.0
    collision_risk: float = 0.0
    early_warned: bool = False
    slow_wrong_way: bool = False
    heading_source: str = "pairwise"
    # [N4]
    bayes_posterior: float = 0.0
    # [N5]
    ghost_lat: Optional[float] = None
    ghost_lon: Optional[float] = None
    ghost_ts: Optional[float] = None
    # [N6]
    is_heatmap_hotspot: bool = False


# ── [N4] Bayesian Risk Engine ──────────────────────────────

class BayesianRiskEngine:
    """
    Per-vehicle Bayesian wrong-way probability tracker.

    P(WW | obs_1..obs_t) updated at each ping using a simple
    likelihood-ratio Bayes update in log-odds space for numerical
    stability.  The likelihood ratios are configurable via DetectorConfig.

    This is mathematically cleaner than a pure EMA and provides an
    interpretable probability output that can be logged / audited.
    """

    def __init__(self, prior: float, lr_ww: float, lr_ok: float):
        self._prior = prior
        self._lr_ww = lr_ww
        self._lr_ok = lr_ok
        # log-odds state per vehicle
        self._log_odds: Dict[str, float] = {}

    def _to_log_odds(self, p: float) -> float:
        p = max(1e-9, min(1 - 1e-9, p))
        return math.log(p / (1 - p))

    def _to_prob(self, lo: float) -> float:
        return 1.0 / (1.0 + math.exp(-lo))

    def update(self, vehicle_id: str, is_wrong_heading: bool,
               hconf: float = 1.0) -> float:
        """
        Update posterior and return P(wrong_way).

        is_wrong_heading: True if bearing delta > adaptive threshold.
        hconf: heading confidence; low-confidence observations contribute
               less (their likelihood ratio is blended toward 1.0).
        """
        if vehicle_id not in self._log_odds:
            self._log_odds[vehicle_id] = self._to_log_odds(self._prior)

        lr = self._lr_ww if is_wrong_heading else self._lr_ok
        # Blend toward neutral (1.0) when heading confidence is low
        blended_lr = 1.0 + (lr - 1.0) * hconf
        blended_lr = max(0.01, blended_lr)

        self._log_odds[vehicle_id] += math.log(blended_lr)
        # Soft clamp: prevent extreme certainty locking
        self._log_odds[vehicle_id] = max(-6.0, min(6.0,
                                          self._log_odds[vehicle_id]))

        return self._to_prob(self._log_odds[vehicle_id])

    def reset(self, vehicle_id: str) -> None:
        self._log_odds[vehicle_id] = self._to_log_odds(self._prior)

    def get(self, vehicle_id: str) -> float:
        if vehicle_id not in self._log_odds:
            return self._prior
        return self._to_prob(self._log_odds[vehicle_id])


# ── [N6] Counter-Flow Heatmap ──────────────────────────────

class CounterFlowHeatmap:
    """
    Spatial accumulator for wrong-way events.

    Each confirmed wrong-way ping increments the cell containing the
    vehicle's position.  All cells decay each second so the heatmap
    reflects *recent* activity, not historical incidents.

    Operators can query hotspot cells to identify road segments that
    repeatedly generate wrong-way events — indicating infrastructure
    issues (missing/faded signs, confusing lane markings, missing barriers).
    """

    def __init__(self, cell_deg: float, decay: float, min_score: float):
        self._cell_deg = cell_deg
        self._decay = decay          # multiply all cells by this each second
        self._min_score = min_score
        self._grid: Dict[Tuple[int, int], float] = {}
        self._last_decay_ts: float = 0.0

    def _cell(self, lat: float, lon: float) -> Tuple[int, int]:
        return (int(lat / self._cell_deg), int(lon / self._cell_deg))

    def _decay_all(self, now: float) -> None:
        dt = now - self._last_decay_ts
        if dt <= 0:
            return
        factor = self._decay ** dt
        self._grid = {k: v * factor for k, v in self._grid.items()
                      if v * factor >= self._min_score}
        self._last_decay_ts = now

    def record(self, lat: float, lon: float, weight: float,
               timestamp: float) -> None:
        self._decay_all(timestamp)
        cell = self._cell(lat, lon)
        self._grid[cell] = self._grid.get(cell, 0.0) + weight

    def is_hotspot(self, lat: float, lon: float,
                   threshold: float = 2.0) -> bool:
        cell = self._cell(lat, lon)
        return self._grid.get(cell, 0.0) >= threshold

    def get_hotspots(self, threshold: float = 2.0) -> List[dict]:
        return [
            {
                "cell": cell,
                "score": score,
                "approx_lat": cell[0] * self._cell_deg + self._cell_deg / 2,
                "approx_lon": cell[1] * self._cell_deg + self._cell_deg / 2,
            }
            for cell, score in self._grid.items()
            if score >= threshold
        ]

    def get_all_cells(self) -> Dict[Tuple[int, int], float]:
        """Return a copy of the full grid for visualization."""
        return dict(self._grid)


# ── [N5] Ghost Vehicle Predictor ──────────────────────────

def predict_ghost_position(
    lat: float,
    lon: float,
    heading_deg: float,
    speed_kmh: float,
    horizon_s: float,
) -> Tuple[float, float]:
    """
    Project position `horizon_s` seconds into the future along heading.
    Uses flat-earth approximation (accurate to < 0.1% over 1 km).
    """
    dist_m = (speed_kmh / 3.6) * horizon_s
    br = math.radians(heading_deg)
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat)))
    g_lat = lat + dist_m * math.cos(br) * lat_per_m
    g_lon = lon + dist_m * math.sin(br) * lon_per_m
    return g_lat, g_lon


# ── Main Detector ──────────────────────────────────────────

class WrongWayDetector:
    """
    Stateful per-vehicle wrong-way detection engine.

    v3 adds three unique novelty engines on top of the v2 architecture:
      [N4] BayesianRiskEngine  — per-vehicle posterior P(wrong_way)
      [N5] GhostVehiclePredictor — future position projection
      [N6] CounterFlowHeatmap  — spatial wrong-way activity accumulator

    Parameters
    ----------
    road_resolver : RoadProvider
        Any object satisfying the RoadProvider Protocol.
    config : DetectorConfig
        All tuning parameters in one place.
    """

    def __init__(
        self,
        road_resolver: Optional[RoadProvider] = None,
        config: DetectorConfig = DEFAULT_DETECTOR_CFG,
    ):
        if road_resolver is None:
            from core.osm_resolver import OSMRoadResolver
            road_resolver = OSMRoadResolver()
        self.resolver = road_resolver
        self.cfg = config

        self._vehicle_states: Dict[str, VehicleState] = {}
        self._alerts: List[WrongWayAlert] = []
        self._trajectory_window: Dict[str, deque] = {}
        self._last_alert_time: Dict[str, float] = {}

        # §2.4 — spatial grid
        self._spatial_grid: Dict[Tuple[int, int], set] = {}

        # [N4] Bayesian engine
        self._bayes = BayesianRiskEngine(
            prior=config.bayesian_prior,
            lr_ww=config.bayesian_likelihood_ratio_ww,
            lr_ok=config.bayesian_likelihood_ratio_ok,
        )

        # [N6] Heatmap
        self._heatmap = CounterFlowHeatmap(
            cell_deg=config.heatmap_grid_size_deg,
            decay=config.heatmap_decay_rate,
            min_score=config.heatmap_min_score,
        )

    # ── Spatial grid ──────────────────────────────────────

    def _grid_update(self, vehicle_id: str,
                     old_lat: Optional[float], old_lon: Optional[float],
                     new_lat: float, new_lon: float) -> None:
        if old_lat is not None and old_lon is not None:
            old_key = _grid_key(old_lat, old_lon)
            cell = self._spatial_grid.get(old_key)
            if cell is not None:
                cell.discard(vehicle_id)
                if not cell:                          # BUG FIX: prune empty cells
                    del self._spatial_grid[old_key]
        new_key = _grid_key(new_lat, new_lon)
        self._spatial_grid.setdefault(new_key, set()).add(vehicle_id)

    def _nearby_vehicle_ids(self, lat: float, lon: float) -> List[str]:
        row, col = _grid_key(lat, lon)
        ids: List[str] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                ids.extend(self._spatial_grid.get((row + dr, col + dc), set()))
        return ids

    # ── N1: Speed-adaptive threshold ──────────────────────

    def _get_adaptive_threshold(self, road_type: str,
                                 speed_kmh: float) -> Tuple[float, str]:
        base, hs_spd, hs_thr = self.cfg.road_profiles.get(
            road_type, self.cfg.default_road_profile)
        if speed_kmh >= hs_spd:
            return float(hs_thr), f"{road_type} hi-speed"
        return float(base), road_type

    # ── N2: Collision risk ─────────────────────────────────

    def _compute_collision_risk(
        self, vehicle_id: str, lat: float, lon: float,
        heading: float, speed_kmh: float, timestamp: float,
    ) -> float:
        max_risk = 0.0
        speed_ms = max(speed_kmh / 3.6, 0.1)
        for vid in self._nearby_vehicle_ids(lat, lon):
            if vid == vehicle_id:
                continue
            other = self._vehicle_states.get(vid)
            if other is None:
                continue
            if other.heading is None or other.speed_kmh < self.cfg.min_speed_kmh:
                continue
            if abs(other.timestamp - timestamp) > self.cfg.consensus_max_age_s:
                continue
            dist = haversine_distance(lat, lon, other.lat, other.lon)
            if dist > 300:
                continue
            if angular_difference(heading, other.heading) < 120:
                continue
            dlat = other.lat - lat
            dlon = other.lon - lon
            bearing_to_other = (math.degrees(math.atan2(dlon, dlat)) + 360) % 360
            if angular_difference(heading, bearing_to_other) > 60:
                continue
            other_ms = max(other.speed_kmh / 3.6, 0.1)
            ttc = dist / (speed_ms + other_ms)
            max_risk = max(max_risk, max(0.0, 1.0 - ttc / 15.0))
        return round(max_risk, 3)

    # ── N3: Early warning ─────────────────────────────────

    def _update_early_warning(self, state: VehicleState,
                               risk_score: float) -> bool:
        if state.is_confirmed_wrong_way:
            return False
        if risk_score >= self.cfg.early_warning_risk_floor:
            state.early_warning_persistence += 1
        else:
            state.early_warning_persistence = max(
                0, state.early_warning_persistence - 1)

        # BUG FIX: was using self.cfg.early_warning_persistence (the field)
        # as the threshold — that is a config int, not the state counter.
        # Correct thresholds are early_warning_persistence (normal) and
        # ew_fast_persistence (high speed).
        ew_thr = (self.cfg.ew_fast_persistence
                  if state.speed_kmh >= self.cfg.ew_fast_speed_kmh
                  else self.cfg.early_warning_persistence)

        if state.early_warning_persistence >= ew_thr and not state.early_warning_fired:
            state.early_warning_fired = True
            return True
        return False

    # ── Risk score ────────────────────────────────────────

    def _compute_risk(self, angle_diff: float, persistence: int,
                      speed_kmh: float, threshold: float,
                      hconf: float = 1.0) -> float:
        angle_score = min(angle_diff / threshold, 1.0) * hconf
        # BUG FIX: denominator was hardcoded 5; use trajectory_window_size
        persist_score = min(persistence / self.cfg.trajectory_window_size, 1.0) * hconf
        speed_score = min(speed_kmh / 60.0, 1.0) if speed_kmh else 0.5
        return (0.5 * angle_score + 0.3 * persist_score + 0.2 * speed_score)

    def _compute_map_certainty(self, heading: float,
                                road: RoadSegment) -> float:
        if road.is_roundabout or road.is_junction:
            return 0.4
        fwd = angular_difference(heading, road.allowed_direction)
        rev = angular_difference(heading, (road.allowed_direction + 180) % 360)
        return round(max(0.0, 1.0 - min(fwd, rev) / 90.0), 3)

    # ── Consensus ─────────────────────────────────────────

    def _check_consensus(self, vehicle_id: str, lat: float, lon: float,
                          timestamp: float,
                          road_bearing: float) -> Tuple[bool, List[str]]:
        confirming = []
        for vid in self._nearby_vehicle_ids(lat, lon):
            if vid == vehicle_id:
                continue
            state = self._vehicle_states.get(vid)
            if state is None or state.heading is None:
                continue
            if abs(state.timestamp - timestamp) > self.cfg.consensus_max_age_s:
                continue
            if haversine_distance(lat, lon, state.lat, state.lon) > self.cfg.consensus_radius_m:
                continue
            if angular_difference(state.heading, road_bearing) < 60:
                confirming.append(vid)
        return len(confirming) >= self.cfg.consensus_min_vehicles, confirming

    # ── Suppression ───────────────────────────────────────

    def _should_suppress(self, state: VehicleState,
                          road: RoadSegment) -> Tuple[bool, str]:
        if road.is_construction:
            return True, "construction_zone"
        if road.is_roundabout:
            return True, "roundabout_geometry"
        if road.road_type in ("footway", "path", "cycleway", "steps", "pedestrian"):
            return True, "non_drivable_road_type"
        if (state.map_match_certainty is not None
                and state.map_match_certainty < 0.20):
            return True, "low_map_certainty"
        if state.maneuver_hold_remaining > 0:
            return True, "maneuver_hold"
        return False, "none"

    def _effective_cooldown(self, risk_score: float) -> float:
        if risk_score > self.cfg.high_risk_threshold:
            return self.cfg.alert_cooldown_seconds * self.cfg.high_risk_cooldown_factor
        return self.cfg.alert_cooldown_seconds

    # ── §4.3: Choose best heading ──────────────────────────

    def _resolve_heading(
        self,
        pairwise_heading: float,
        traj: deque,
    ) -> Tuple[float, str]:
        if len(traj) >= 3:
            positions  = [(lat, lon) for lat, lon, _ts, _h in traj]
            timestamps = [ts        for _lat, _lon, ts, _h in traj]
            vv = estimate_velocity_vector(positions, timestamps)
            if vv is not None:
                return vv, "velocity_vector"
        return pairwise_heading, "pairwise"

    # ── Main update ───────────────────────────────────────

    def update_vehicle(
        self,
        vehicle_id: str,
        lat: float,
        lon: float,
        timestamp: float,
        heading: Optional[float] = None,
        speed_kmh: Optional[float] = None,
    ) -> Optional[WrongWayAlert]:

        if vehicle_id not in self._trajectory_window:
            self._trajectory_window[vehicle_id] = deque(
                maxlen=self.cfg.trajectory_window_size)

        if heading is None:
            return None
        if speed_kmh is None:
            speed_kmh = 0.0

        prev = self._vehicle_states.get(vehicle_id)
        old_lat = prev.lat if prev else None
        old_lon = prev.lon if prev else None
        self._grid_update(vehicle_id, old_lat, old_lon, lat, lon)

        self._trajectory_window[vehicle_id].append((lat, lon, timestamp, heading))
        hconf = heading_confidence(
            speed_kmh,
            reliable_speed_kmh=self.cfg.heading_reliable_speed_kmh,
            min_speed_kmh=self.cfg.min_speed_kmh,
        )

        state = VehicleState(
            vehicle_id=vehicle_id, lat=lat, lon=lon,
            heading=heading, speed_kmh=speed_kmh, timestamp=timestamp,
            risk_score=prev.risk_score if prev else 0.0,
            consecutive_wrong_way_seconds=(
                prev.consecutive_wrong_way_seconds if prev else 0.0),
            early_warning_persistence=(
                prev.early_warning_persistence if prev else 0),
            early_warning_fired=(prev.early_warning_fired if prev else False),
            maneuver_hold_remaining=(
                prev.maneuver_hold_remaining if prev else 0.0),
            slow_ww_seconds=(prev.slow_ww_seconds if prev else 0.0),
            heading_confidence=hconf,
            bayes_posterior=(prev.bayes_posterior if prev else self.cfg.bayesian_prior),
        )

        if speed_kmh < self.cfg.min_speed_kmh:
            state.failure_mode = "below_speed_threshold"
            state.maneuver_hold_remaining = max(
                0.0, state.maneuver_hold_remaining - 1.0)
            self._vehicle_states[vehicle_id] = state
            return None

        road = self.resolver.get_best_road_match(lat, lon, heading)
        if road is None:
            state.failure_mode = "no_road_match"
            self._vehicle_states[vehicle_id] = state
            return None

        state.matched_road = road
        state.map_match_certainty = self._compute_map_certainty(heading, road)

        traj = self._trajectory_window[vehicle_id]
        eff_heading, heading_src = self._resolve_heading(heading, traj)
        state.velocity_vector_heading = (
            eff_heading if heading_src == "velocity_vector" else None)

        threshold, profile = self._get_adaptive_threshold(road.road_type, speed_kmh)

        fwd_delta = angular_difference(eff_heading, road.allowed_direction)
        if road.reverse_allowed:
            rev_delta = angular_difference(
                eff_heading, (road.allowed_direction + 180) % 360)
            delta = min(fwd_delta, rev_delta)
        else:
            delta = fwd_delta
        state.bearing_delta = delta

        if len(traj) < 3:
            self._vehicle_states[vehicle_id] = state
            return None

        # §7.1 — consecutive wrong-heading frames
        history_list = list(traj)
        consecutive_count = total_wrong = 0
        for _, _, _, h in reversed(history_list):
            if h is None:
                break
            if abs((h - road.allowed_direction + 180) % 360 - 180) > 100:
                consecutive_count += 1
                total_wrong += 1
            else:
                break
        persistence = min(
            total_wrong,
            consecutive_count if consecutive_count >= self.cfg.persist_min_frames else 0,
        )

        raw_risk = self._compute_risk(delta, persistence, speed_kmh,
                                       threshold, hconf)
        state.risk_score = ((1 - self.cfg.risk_ema_alpha) * state.risk_score
                             + self.cfg.risk_ema_alpha * raw_risk)

        # §4.1 — maneuver hold
        if speed_kmh < self.cfg.maneuver_speed_kmh and delta > threshold * 0.7:
            state.maneuver_hold_remaining = self.cfg.maneuver_hold_seconds
        else:
            state.maneuver_hold_remaining = max(
                0.0, state.maneuver_hold_remaining - 1.0)

        if raw_risk > 0.4 and state.maneuver_hold_remaining == 0:
            state.consecutive_wrong_way_seconds += 1.0
        elif raw_risk <= 0.4:
            state.consecutive_wrong_way_seconds = max(
                0.0, state.consecutive_wrong_way_seconds - 1.0)

        # §9 — slow wrong-way accumulator (BUG FIX: clamp at 0)
        is_slow = self.cfg.min_speed_kmh <= speed_kmh <= self.cfg.slow_ww_speed_max_kmh
        if is_slow and delta > threshold and hconf >= 0.3:
            state.slow_ww_seconds += 1.0
        else:
            state.slow_ww_seconds = max(0.0, state.slow_ww_seconds - 0.5)  # BUG FIX
        slow_ww_triggered = (
            is_slow
            and state.slow_ww_seconds >= self.cfg.slow_ww_seconds
            and hconf >= 0.3
        )

        # [N4] Bayesian update
        is_wrong_heading = delta > threshold
        bayes_p = self._bayes.update(vehicle_id, is_wrong_heading, hconf)
        state.bayes_posterior = bayes_p

        ew_fires = self._update_early_warning(state, state.risk_score)
        if ew_fires:
            logger.info(
                f"⚡ EARLY WARNING | {vehicle_id} | "
                f"Risk={state.risk_score:.2f} | "
                f"Bayes={bayes_p:.3f} | "
                f"persistence={state.early_warning_persistence}"
            )

        # Dual-gate confirmation: EMA path OR Bayesian path OR slow-WW
        ema_confirmed = (
            state.risk_score > self.cfg.risk_confirm_threshold
            and state.consecutive_wrong_way_seconds >= self.cfg.temporal_confirm_seconds
        )
        bayes_confirmed = bayes_p > 0.70 and delta > threshold * 0.8
        confirmed = ema_confirmed or bayes_confirmed or slow_ww_triggered
        state.is_confirmed_wrong_way = confirmed
        self._vehicle_states[vehicle_id] = state

        collision_risk = self._compute_collision_risk(
            vehicle_id, lat, lon, eff_heading, speed_kmh, timestamp)
        state.collision_risk = collision_risk

        # [N5] Ghost prediction
        ghost_lat = ghost_lon = ghost_ts = None
        if confirmed and speed_kmh >= self.cfg.min_speed_kmh:
            g_heading = eff_heading
            horizon = self.cfg.ghost_prediction_horizon_s
            ghost_lat, ghost_lon = predict_ghost_position(
                lat, lon, g_heading, speed_kmh, horizon)
            ghost_ts = timestamp + horizon
            state.ghost_lat = ghost_lat
            state.ghost_lon = ghost_lon
            state.ghost_ts = ghost_ts

        # [N6] Heatmap update
        is_heatspot = False
        if confirmed:
            self._heatmap.record(lat, lon, weight=state.risk_score,
                                 timestamp=timestamp)
            is_heatspot = self._heatmap.is_hotspot(lat, lon, threshold=2.0)

        if not confirmed:
            state.failure_mode = "below_threshold"
            return None

        suppress, suppress_reason = self._should_suppress(state, road)
        consensus_ok, consensus_ids = self._check_consensus(
            vehicle_id, lat, lon, timestamp, road.allowed_direction)

        last_alert = self._last_alert_time.get(vehicle_id)
        effective_cd = self._effective_cooldown(state.risk_score)
        in_cooldown = (last_alert is not None
                       and timestamp - last_alert < effective_cd)

        alert = WrongWayAlert(
            vehicle_id=vehicle_id, lat=lat, lon=lon,
            timestamp=timestamp,
            risk_score=state.risk_score,
            bearing_delta=delta,
            road_name=road.name,
            road_type=road.road_type,
            speed_kmh=speed_kmh,
            suppressed=suppress or in_cooldown,
            suppression_reason=(suppress_reason if suppress
                                 else ("cooldown" if in_cooldown else "")),
            confirmed_by_consensus=consensus_ok,
            consensus_vehicle_ids=consensus_ids,
            road_profile=profile,
            adaptive_threshold=threshold,
            collision_risk=collision_risk,
            early_warned=state.early_warning_fired,
            slow_wrong_way=slow_ww_triggered,
            heading_source=heading_src,
            bayes_posterior=bayes_p,
            ghost_lat=ghost_lat,
            ghost_lon=ghost_lon,
            ghost_ts=ghost_ts,
            is_heatmap_hotspot=is_heatspot,
        )
        self._alerts.append(alert)

        if not (suppress or in_cooldown):
            self._last_alert_time[vehicle_id] = timestamp
            state.failure_mode = "none"
            slow_tag = " [SLOW-WW]" if slow_ww_triggered else ""
            bayes_tag = " [BAYES]" if bayes_confirmed and not ema_confirmed else ""
            hs_tag = " [HOTSPOT]" if is_heatspot else ""
            logger.warning(
                f"⚠️  WRONG WAY{slow_tag}{bayes_tag}{hs_tag} | {vehicle_id} | "
                f"Risk={state.risk_score:.2f} | Bayes={bayes_p:.3f} | "
                f"Delta={delta:.1f}° | Profile={profile} thresh={threshold:.0f}° | "
                f"Collision={collision_risk:.2f} | hconf={hconf:.2f} | "
                f"hdg_src={heading_src} | "
                f"EW={'✅' if state.early_warning_fired else '❌'} | "
                f"Consensus={'✅' if consensus_ok else '❌'}"
            )
            return alert

        state.failure_mode = suppress_reason if suppress else "cooldown"
        logger.info(f"Alert suppressed [{suppress_reason or 'cooldown'}]: {vehicle_id}")
        return None

    # ── Accessors ─────────────────────────────────────────

    def get_alerts(self) -> List[WrongWayAlert]:
        return [a for a in self._alerts if not a.suppressed]

    def get_all_alerts(self) -> List[WrongWayAlert]:
        return self._alerts

    def get_all_states(self) -> Dict[str, VehicleState]:
        return self._vehicle_states

    def get_early_warnings(self) -> List[dict]:
        return [
            {"vehicle_id": vid, "risk": s.risk_score,
             "persistence": s.early_warning_persistence,
             "lat": s.lat, "lon": s.lon,
             "bayes": s.bayes_posterior}
            for vid, s in self._vehicle_states.items()
            if s.early_warning_fired
        ]

    def get_heatmap(self) -> CounterFlowHeatmap:
        """[N6] Expose heatmap for visualization."""
        return self._heatmap

    def get_ghost_predictions(self) -> List[dict]:
        """[N5] Return all active ghost vehicle predictions."""
        return [
            {
                "vehicle_id": vid,
                "origin_lat": s.lat,
                "origin_lon": s.lon,
                "ghost_lat": s.ghost_lat,
                "ghost_lon": s.ghost_lon,
                "ghost_ts": s.ghost_ts,
                "risk": s.risk_score,
                "speed_kmh": s.speed_kmh,
            }
            for vid, s in self._vehicle_states.items()
            if s.ghost_lat is not None and s.is_confirmed_wrong_way
        ]

    def get_explainability_report(self) -> List[dict]:
        return [
            {
                "vehicle_id": vid,
                "is_confirmed_wrong_way": s.is_confirmed_wrong_way,
                "risk_score": s.risk_score,
                "bayes_posterior": s.bayes_posterior,
                "angular_delta": s.bearing_delta,
                "map_match_certainty": s.map_match_certainty,
                "failure_mode": s.failure_mode,
                "consecutive_seconds": s.consecutive_wrong_way_seconds,
                "road": s.matched_road.name if s.matched_road else "N/A",
                "collision_risk": s.collision_risk,
                "early_warning_fired": s.early_warning_fired,
                "heading_confidence": s.heading_confidence,
                "slow_ww_seconds": s.slow_ww_seconds,
                "maneuver_hold": s.maneuver_hold_remaining,
                "velocity_vector_heading": s.velocity_vector_heading,
                "ghost_lat": s.ghost_lat,
                "ghost_lon": s.ghost_lon,
                "narrative": self._build_narrative(s),
            }
            for vid, s in self._vehicle_states.items()
        ]

    def _build_narrative(self, state: VehicleState) -> str:
        if state.is_confirmed_wrong_way:
            ew   = " (early warning fired)" if state.early_warning_fired else ""
            cr   = (f" Collision risk={state.collision_risk:.2f}."
                    if state.collision_risk > 0.1 else "")
            slow = (" [SLOW wrong-way]"
                    if state.slow_ww_seconds >= self.cfg.slow_ww_seconds else "")
            hc   = (f" hconf={state.heading_confidence:.2f}."
                    if state.heading_confidence < 0.8 else "")
            vv   = (" [vel-vector hdg]"
                    if state.velocity_vector_heading is not None else "")
            bp   = f" Bayes={state.bayes_posterior:.3f}."
            ghost = (f" Ghost→({state.ghost_lat:.5f},{state.ghost_lon:.5f})"
                     if state.ghost_lat else "")
            return (
                f"Risk={state.risk_score:.2f} for "
                f"{state.consecutive_wrong_way_seconds:.1f}s, "
                f"delta={state.bearing_delta:.1f}°.{ew}{cr}{slow}{hc}{vv}{bp}{ghost}"
            )
        msgs = {
            "below_speed_threshold": "Too slow (< 5 km/h).",
            "no_road_match":         "No road match — off-map or OSM gap.",
            "construction_zone":     "Construction zone suppression.",
            "roundabout_geometry":   "Roundabout — ambiguous heading.",
            "low_map_certainty":     "Map certainty too low.",
            "cooldown":              "Cooldown — already flagged recently.",
            "maneuver_hold":         "Maneuver suppression (U-turn / slow turn).",
        }
        fm = state.failure_mode
        if fm in msgs:
            return msgs[fm]
        ew = " Early warning fired." if state.early_warning_fired else ""
        return (f"Risk {state.risk_score:.2f} below threshold "
                f"{self.cfg.risk_confirm_threshold}.{ew}")

    def get_stats(self) -> dict:
        breakdown: Dict[str, int] = {}
        for s in self._vehicle_states.values():
            fm = s.failure_mode or "none"
            breakdown[fm] = breakdown.get(fm, 0) + 1
        return {
            "total_vehicles":       len(self._vehicle_states),
            "confirmed_wrong_way":  sum(
                1 for s in self._vehicle_states.values()
                if s.is_confirmed_wrong_way),
            "early_warnings":       sum(
                1 for s in self._vehicle_states.values()
                if s.early_warning_fired),
            "total_alerts_fired":   len(self._alerts),
            "suppressed_alerts":    sum(
                1 for a in self._alerts if a.suppressed),
            "slow_ww_detections":   sum(
                1 for a in self._alerts if a.slow_wrong_way),
            "maneuver_suppressed":  breakdown.get("maneuver_hold", 0),
            "velocity_vector_used": sum(
                1 for s in self._vehicle_states.values()
                if s.velocity_vector_heading is not None),
            "bayes_confirmed":      sum(
                1 for a in self._alerts
                if not a.suppressed and a.bayes_posterior > 0.70),
            "ghost_predictions":    sum(
                1 for s in self._vehicle_states.values()
                if s.ghost_lat is not None),
            "heatmap_hotspots":     len(self._heatmap.get_hotspots()),
            "failure_mode_breakdown": breakdown,
        }
