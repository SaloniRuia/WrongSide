# Wrong-Way Detection System — v3

## What Changed (v3)

### Bug Fixes
| Bug | File | Fix |
|-----|------|-----|
| Map not rendering | map_builder.py | Python data was written outside `<script>` tag — now injected correctly inside a single JS block |
| Extra dots in timeline | map_builder.py | Each vehicle created two DOM elements (dot + label separately); now a single `.tl-item` per vehicle |
| F1/Precision/Recall low | config.py | `risk_ema_alpha` 0.30→0.45, `risk_confirm_threshold` 0.45→0.38, `persist_min_frames` 2→1, `temporal_confirm_seconds` 2.0→1.5, `early_warning_risk_floor` 0.30→0.22 |
| EW threshold bug | detector.py | `_update_early_warning` used `self.cfg.early_warning_persistence` (the config field) as counter threshold — fixed to use proper cfg values |
| Spatial grid memory leak | detector.py | Empty grid cells were never pruned; now deleted when a vehicle leaves a cell |
| persistence_score capped at 5 | detector.py | Denominator was hardcoded `5.0`; now uses `trajectory_window_size` |
| slow_ww_seconds negative | detector.py | Accumulator could go below 0 via `-0.5` decay; clamped at 0 |
| TURNING_VEHICLE never added | simulator.py | `add_turning_vehicle()` now called in both scenario builders |

### New Novelty Features (unique)

**[N4] Bayesian Road Conflict Scorer**  
Per-vehicle posterior `P(wrong_way | observations)` updated at every ping using log-odds Bayes with configurable likelihood ratios. Provides a mathematically principled second confirmation gate independent of the EMA path. Audit-friendly — every decision has a numeric posterior.

**[N5] Ghost Vehicle Predictor**  
When a wrong-way vehicle is confirmed, its future position is extrapolated `N` seconds ahead using its current velocity vector. Ghost position shown on map as 👻 marker with dashed line from current position. Enables proactive response (smart-signal preemption, barrier activation).

**[N6] Counter-Flow Heatmap**  
Spatial grid accumulates wrong-way events (decays over time). Cells above threshold flagged as infrastructure hotspots. Shown as colour-coded overlay on the map (cyan→amber→red). Lets operators identify confusing junctions, missing signage, or barriers that need improvement — not just reacting to individual incidents.

## Architecture
```
wrong_way_detection_system/
├── core/
│   ├── config.py          — All tuning parameters (DetectorConfig)
│   ├── detector.py        — Main engine (N1–N6 novelty features)
│   ├── evaluator.py       — Vehicle-level + frame-level metrics
│   ├── bearing_utils.py   — Kalman filter, velocity-vector heading
│   ├── interfaces.py      — RoadProvider Protocol + MockRoadProvider
│   └── osm_resolver.py    — OSM Overpass API client
├── simulation/
│   └── simulator.py       — Multi-vehicle GPS trace generator
├── visualization/
│   └── map_builder.py     — Mapbox GL interactive HTML map
└── main.py                — Pipeline entry point
```

## Running
```bash
pip install -e .
# Online (uses OSM API):
python main.py --lat 12.9716 --lon 77.5946 --duration 60
# Offline (no API calls):
python main.py --offline --duration 60
```
