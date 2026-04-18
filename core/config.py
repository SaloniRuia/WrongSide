"""
Centralised configuration for the wrong-way detection system.

FIXES v2:
  - risk_confirm_threshold lowered 0.45→0.38 for faster confirmation
  - risk_ema_alpha raised 0.3→0.45 for snappier risk tracking
  - persist_min_frames lowered 2→1 so single-frame anomaly counts
  - temporal_confirm_seconds lowered 2.0→1.5 for faster confirmation
  - early_warning_risk_floor lowered 0.30→0.22
  - All magic numbers still in one place (unchanged from v1)

NOVELTY additions:
  - bayesian_prior: base prior for wrong-way probability (used by BayesianRiskEngine)
  - ghost_prediction_horizon_s: how far ahead the ghost vehicle predictor looks
  - heatmap_decay_rate: per-second decay of the counter-flow heatmap
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class DetectorConfig:
    """Detection engine tuning parameters."""

    # ── Speed gates ────────────────────────────────────────
    min_speed_kmh: float = 5.0
    maneuver_speed_kmh: float = 12.0
    maneuver_hold_seconds: float = 4.0
    slow_ww_speed_max_kmh: float = 15.0
    slow_ww_seconds: float = 4.0          # FIX: 5.0→4.0 (faster slow-WW trigger)

    # ── Risk scoring ───────────────────────────────────────
    risk_confirm_threshold: float = 0.38  # FIX: 0.45→0.38 (boost recall)
    temporal_confirm_seconds: float = 1.5 # FIX: 2.0→1.5
    risk_ema_alpha: float = 0.45          # FIX: 0.30→0.45 (snappier EMA)

    # ── Persistence / trajectory window ────────────────────
    trajectory_window_size: int = 8
    persist_min_frames: int = 1           # FIX: 2→1
    trajectory_gap_threshold_s: float = 3.0

    # ── Alerts / cooldown ──────────────────────────────────
    alert_cooldown_seconds: float = 10.0
    high_risk_cooldown_factor: float = 0.5
    high_risk_threshold: float = 0.75

    # ── Early warning ──────────────────────────────────────
    early_warning_persistence: int = 2    # FIX: 3→2 (fire sooner)
    early_warning_risk_floor: float = 0.22 # FIX: 0.30→0.22
    ew_fast_speed_kmh: float = 60.0
    ew_fast_persistence: int = 1          # FIX: 2→1 (high-speed: 1 frame)

    # ── Consensus ──────────────────────────────────────────
    consensus_min_vehicles: int = 2
    consensus_radius_m: float = 150.0
    consensus_max_age_s: float = 5.0

    # ── Heading confidence ─────────────────────────────────
    heading_reliable_speed_kmh: float = 15.0

    # ── Road profiles (speed-adaptive thresholds) ──────────
    road_profiles: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "motorway":      (100, 80, 110),
        "motorway_link": (100, 60, 110),
        "trunk":         (110, 70, 120),
        "trunk_link":    (110, 60, 120),
        "primary":       (110, 60, 120),
        "primary_link":  (110, 50, 120),
        "secondary":     (120, 50, 130),
        "tertiary":      (120, 40, 130),
        "residential":   (130, 30, 140),
        "unclassified":  (130, 30, 140),
        "service":       (140, 20, 150),
    })

    default_road_profile: Tuple[float, float, float] = (120, 50, 130)

    # ── NOVELTY: Bayesian engine parameters ────────────────
    bayesian_prior: float = 0.05
    """Prior probability that any vehicle on a one-way road is going wrong way."""

    bayesian_likelihood_ratio_ww: float = 8.0
    """Likelihood ratio P(obs|wrong-way) / P(obs|normal) when delta > threshold."""

    bayesian_likelihood_ratio_ok: float = 0.15
    """Likelihood ratio when delta < threshold (evidence for normal driving)."""

    # ── NOVELTY: Ghost vehicle predictor ───────────────────
    ghost_prediction_horizon_s: float = 5.0
    """Seconds ahead to project a wrong-way vehicle's ghost position."""

    ghost_min_risk: float = 0.35
    """Minimum risk score required to emit a ghost prediction."""

    # ── NOVELTY: Counter-flow heatmap ──────────────────────
    heatmap_grid_size_deg: float = 0.0003   # ~33 m cells
    heatmap_decay_rate: float = 0.92        # multiply each cell per second
    heatmap_min_score: float = 0.05        # prune cells below this


@dataclass
class ResolverConfig:
    """OSM resolver tuning parameters."""

    cache_ttl_seconds: int = 600
    overpass_timeout_s: int = 20
    fetch_radius_m: int = 20
    rate_limit_delay_s: float = 0.2
    max_retries: int = 4

    distance_weight: float = 0.4
    heading_weight: float = 0.6
    max_match_dist_m: float = 80.0
    junction_dist_penalty_m: float = 20.0

    hmm_sigma_z: float = 15.0
    hmm_beta: float = 10.0


@dataclass
class SimulatorConfig:
    """Simulation noise parameters."""

    default_gps_noise_m: float = 3.0
    default_heading_noise_deg: float = 5.0
    ping_interval_s: float = 1.0

    multipath_prob: float = 0.03
    multipath_jump_m: float = 25.0
    dropout_prob: float = 0.02
    timestamp_jitter_s: float = 0.15


DEFAULT_DETECTOR_CFG  = DetectorConfig()
DEFAULT_RESOLVER_CFG  = ResolverConfig()
DEFAULT_SIMULATOR_CFG = SimulatorConfig()
