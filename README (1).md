# WrongSide: Lightweight Wrong-Way Driver Detection via GPS–OSM Bearing Conflict Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Version](https://img.shields.io/badge/Version-3.0.0-orange?style=flat-square)
![OpenStreetMap](https://img.shields.io/badge/Road%20Data-OpenStreetMap-7EBC6F?style=flat-square&logo=openstreetmap)
![HARMAN](https://img.shields.io/badge/Challenge-HARMAN%20Automotive%20Proposal--3-0062A3?style=flat-square)

**A hackathon submission for HARMAN Automotive Proposal-3: Wrong-Way Driver Detection beyond Ramps**

*Identifies external vehicles travelling against allowed traffic flow using heading–geometry conflict scoring on open road data (OSM), with no camera infrastructure, no paid APIs, and no deep-learning dependencies.*

</div>

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Detection Pipeline](#4-detection-pipeline)
   - [4.1 GPS Heading Estimation](#41-gps-heading-estimation)
   - [4.2 OSM Road Resolver](#42-osm-road-resolver)
   - [4.3 Triple-Gate Confirmation Model](#43-triple-gate-confirmation-model)
   - [4.4 Novelty Features (N4–N6)](#44-novelty-features-n4n6)
5. [Simulation Framework](#5-simulation-framework)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Visualization](#7-visualization)
8. [Installation & Quickstart](#8-installation--quickstart)
9. [Configuration Reference](#9-configuration-reference)
10. [Results & Performance](#10-results--performance)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [Dependencies](#12-dependencies)
13. [Acknowledgements](#13-acknowledgements)

---

## 1. Motivation & Problem Statement

Wrong-way driving incidents are disproportionately fatal. Traditional detection systems rely on ramp-mounted camera infrastructure at freeway entry/exit points—a design that fundamentally cannot scale to urban grids, undivided arterials, or temporary diversions introduced by roadworks. **WrongSide** addresses this gap.

### Challenge (HARMAN Automotive Proposal-3)

> *Build a lightweight system that identifies external vehicles moving opposite to the allowed direction using heading vs. road geometry from open-data sources (e.g., OSM). Demonstrate detection using simulated multi-vehicle GPS traces, including injected wrong-way intruders.*

### Why This Is Hard

| Challenge | Why It Matters |
|---|---|
| GPS noise (3–25 m, 3% multipath) | Low-speed heading derivation is below the noise floor |
| Winding road geometry | Last-pairwise bearing diverges from true travel direction on curves |
| Legal manoeuvres (U-turns, 3-point turns) | Briefly present wrong-way heading; must not be flagged |
| Slow reversing intruders | Speed-weighted risk scores cannot accumulate fast enough |
| Urban junction ambiguity | OSM geometry at junctions covers multiple legal bearing options |
| Temporary diversions | Roadworks reroute vehicles onto opposing lanes by design |

WrongSide tackles each of these through layered, principled engineering decisions documented in full below.

---

## 2. System Overview

```
Input: Multi-vehicle GPS pings (simulated or live)
  │
  ▼
[OSM Road Resolver] ←── Overpass API / MockRoadProvider
  │   Maps (lat, lon, heading) → RoadSegment (bearing, type, oneway flag)
  │
  ▼
[Detection Engine]
  │   Triple-gate confirmation:
  │     Gate 1: EMA risk score + temporal confirmation
  │     Gate 2: Bayesian posterior P(wrong_way | observations)    [N4]
  │     Gate 3: Slow-WW accumulator (catches low-speed reversals)
  │
  ├── Ghost Vehicle Predictor (proactive position extrapolation)   [N5]
  └── Counter-Flow Heatmap (infrastructure-level hotspot map)      [N6]
  │
  ▼
[Evaluator]
  │   Vehicle-level: Precision / Recall / F1
  │   Frame-level:   Coverage rate, mean frames to first alert
  │
  ▼
[Visualizer] → wrong_way_detection_map.html (Mapbox GL interactive)
```

**Key design constraints:**

- Zero paid APIs — OSM Overpass is free (public, rate-limited)
- Zero camera infrastructure — pure GPS + road geometry
- Offline mode available for demo stability (MockRoadProvider)
- All parameters centralised in typed dataclasses (`DetectorConfig`, `ResolverConfig`, `SimulatorConfig`)

---

## 3. Architecture

```
WrongSide/
├── core/
│   ├── config.py          — Centralised tuning parameters (DetectorConfig)
│   ├── detector.py        — Main detection engine (Gates 1–3, N4–N6)
│   ├── bearing_utils.py   — GPS geometry: bearing, Kalman, WLS velocity vector
│   ├── evaluator.py       — Vehicle-level + frame-level metrics
│   ├── interfaces.py      — RoadProvider & PingStream Protocols (PEP 544)
│   └── osm_resolver.py    — Overpass API client with HMM-inspired map matching
├── simulation/
│   └── simulator.py       — Multi-vehicle GPS trace generator (4 roles)
├── visualization/
│   └── map_builder.py     — Mapbox GL interactive HTML map builder
├── main.py                — Pipeline entry point
├── requirements.txt
└── setup.py
```

### Design Philosophy

**Protocol-based decoupling.** The detector depends on a `RoadProvider` Protocol (PEP 544 structural subtyping), not on `OSMRoadResolver` directly. Any backend—HERE Maps, local-tile reader, or the offline mock—satisfies the interface by duck-typing without importing the Protocol class. This means swapping road data providers requires zero changes to the detection engine.

**Single-responsibility modules.** Heading geometry, road resolution, risk scoring, evaluation, and visualization each live in isolated modules with no circular imports.

**Dependency-injected configuration.** All 30+ tuning parameters are collected into `DetectorConfig`, `ResolverConfig`, and `SimulatorConfig` dataclasses and injected at construction time. No global mutable state; all parameters overridable per-instance.

---

## 4. Detection Pipeline

### 4.1 GPS Heading Estimation

GPS heading is the most noise-sensitive part of the pipeline. The system employs a three-layer progressive estimation strategy:

#### Layer 1 — Pairwise Bearing with Jitter Guard
Standard spherical azimuth formula. Pings where consecutive positions differ by less than **1.0 m** are skipped (jitter guard). Below this distance, position deltas are comparable to GPS noise, producing random bearing spikes.

#### Layer 2 — Circular-Space Kalman Filter
A 1-D Kalman filter applied in **(sin θ, cos θ)** space, not degree space. Operating directly on degrees causes 0°/360° wraparound errors (the scalar average of 359° and 1° is 180°, not 0°). The filter maintains an uncertainty estimate **P** and computes a gain **K = P/(P+R)** — when the vehicle is in a stable heading, **P** is small and the filter trusts the model; during sharp turns, **P** grows and the filter adapts faster.

#### Layer 3 — Weighted Least-Squares Velocity Vector
When ≥ 3 trajectory positions are available, a recency-weighted linear model is fit over the position window to estimate the heading from the trajectory *trend* rather than the last instantaneous step. This recovers the true arc of travel on winding roads where the last GPS pair points in a misleading direction.

#### Heading Confidence Ramp
A scalar weight in [0, 1] is computed as a linear ramp from 0 (at 5 km/h) to 1 (at 15 km/h). At low speeds, GPS position deltas are below the noise floor; headings are effectively random. This confidence value gates downstream risk contributions—low-speed pings with unreliable heading cannot accumulate a high risk score independently.

```
heading_confidence(v) =
    0                               if v ≤ 5 km/h
    (v - 5) / (15 - 5)             if 5 < v < 15 km/h
    1                               if v ≥ 15 km/h
```

### 4.2 OSM Road Resolver

Maps a `(lat, lon, vehicle_heading)` tuple to the most relevant OSM road segment via the Overpass API.

#### Composite Match Score
```
score(road) = 0.6 × heading_alignment + 0.4 × proximity_score
            − junction_penalty (if road is roundabout or junction)
```

**Why 60/40 and not equal weighting?** On a dual carriageway with a parallel service lane, heading-only matching can snap to the wrong road. Pure proximity matching ignores that the geometrically nearest road may be behind a barrier. The 60/40 split reflects that heading alignment is the primary discriminator for wrong-way detection, while proximity prevents snapping across physical barriers in dense grids.

**Junction penalty (+20 m effective distance).** OSM junction/roundabout ways have ambiguous bearing geometry (a single way covers multiple turning options). The penalty ensures the resolver only snaps to these if no unambiguous road is available, reducing false positives for legal junction navigation.

#### Caching & Rate Limiting
A single Overpass fetch is cached for 600 s (configurable). The Overpass API is a free public service with a ~1 req/s rate limit; per-ping API calls at 1 Hz would immediately trigger 429 throttling and inflate detection latency by network RTT on every ping. The 600 s TTL is a documented engineering trade-off, not an oversight.

Three Overpass mirror endpoints are tried with exponential backoff (4 retries each), providing resilience against public API instability during demos.

### 4.3 Triple-Gate Confirmation Model

A single risk gate cannot optimally serve all wrong-way scenarios simultaneously. Three independent gates operate in parallel; a vehicle is confirmed wrong-way if **any** gate fires:

```
CONFIRMED = EMA_gate  OR  Bayesian_gate  OR  SlowWW_gate
```

| Gate | Optimal For | Mechanism |
|---|---|---|
| **Gate 1: EMA + temporal** | High-speed highway intruders | EMA(α=0.45) of per-frame raw risk score; confirmed when EMA ≥ 0.38 for ≥ 1.5 s |
| **Gate 2: Bayesian posterior** | Uncertain/oscillating cases near threshold | Log-odds Bayes update; confirmed when P(WW\|obs) ≥ 0.70 |
| **Gate 3: Slow-WW accumulator** | Low-speed reversing / creeping intruders | Accumulates seconds spent <15 km/h going wrong-way; fires after 4 s |

**Why three gates?** Gate 1 (EMA) is dominated by a speed score term—slow wrong-way vehicles never accumulate enough speed-weighted risk. Gate 3 catches exactly this case. Gate 2 catches cases where the EMA score oscillates near the threshold (confirming via independent probabilistic evidence). A system with only Gate 1 would miss most reversing intruders.

#### Speed-Adaptive Bearing Threshold
The angular threshold for flagging opposite direction is tightened at high speed:

```python
road_profiles = {
    "motorway":    (threshold=100°, min_speed=80 km/h,  max_speed=110 km/h),
    "primary":     (threshold=110°, min_speed=60 km/h,  max_speed=120 km/h),
    "residential": (threshold=130°, min_speed=30 km/h,  max_speed=140 km/h),
    ...
}
```

At 120 km/h, a 20° GPS heading error is noise; the same error at 5 km/h is almost certainly a committed wrong-way event. Tighter thresholds at high speed reduce false positives where they are cheapest to absorb and increase sensitivity where collisions are most dangerous.

#### Manoeuvre Hold Suppression
When a vehicle is below 12 km/h and heading deviation exceeds 70% of the adaptive threshold, alerts are suppressed for 4 seconds. This prevents legal U-turns and three-point turns from being flagged. A motorway U-turn exceeds this window by design—correctly identifying it as a wrong-way event.

### 4.4 Novelty Features (N4–N6)

#### [N4] Bayesian Road Conflict Scorer

A per-vehicle Bayesian posterior P(wrong_way | observations) is updated at every GPS ping using log-odds Bayes with configurable likelihood ratios:

```
log_odds(t) = log_odds(t-1) + log(LR_ww)   if delta > threshold
log_odds(t) = log_odds(t-1) + log(LR_ok)   if delta < threshold
posterior(t) = sigmoid(log_odds(t))
```

Log-odds are clamped to **[-6, +6]** (corresponding to [0.0025, 0.9975]), preventing the system from reaching dead certainty that cannot be reversed by future evidence.

The likelihood ratio is blended toward neutral (1.0) proportionally to heading confidence—a low-speed ping with unreliable heading contributes almost no Bayesian evidence, whereas a high-confidence ping at speed applies the full LR.

**Why Bayes over a second EMA?** The output is interpretable as a probability with a documented prior (P = 0.05) and likelihood ratios. It is audit-friendly, numerically stable (log-odds space), and heading-confidence-aware. A second EMA would produce an opaque score with no probabilistic meaning.

#### [N5] Ghost Vehicle Predictor

When a wrong-way vehicle is confirmed, its future position is projected **N seconds ahead** (default: 5 s) using its current velocity vector:

```python
ghost_lat = lat + (v_north × T) / 111_111
ghost_lon = lon + (v_east  × T) / (111_111 × cos(lat))
```

The flat-earth approximation is used deliberately: at 120 km/h over 5 s (~167 m), the flat-earth error is approximately **0.003 m**—well below GPS accuracy. The Vincenty geodesic formula is not justified for sub-200 m projections.

The ghost position is displayed as a 👻 marker with a dashed trajectory line, enabling proactive downstream response: smart-signal preemption, barrier activation, or alert broadcasting to other road users before the collision zone is reached.

#### [N6] Counter-Flow Heatmap

A spatial grid (cell size: ~33 m × 33 m) accumulates wrong-way events with time-based decay:

```
cell_score(t) = cell_score(t₀) × decay_rate^(t - t₀)
```

Decay is applied per second (not per ping), making it GPS-frequency-independent—a vehicle at 0.5 Hz does not decay cells twice as fast as one at 1 Hz.

Cells above a configurable threshold are flagged as **infrastructure hotspots**: junctions with confusing signage, roads where barriers are missing, or segments frequently used as wrong-way shortcuts. This lifts the system from per-incident reactive alerting to infrastructure-level trend analysis, enabling targeted physical intervention by road operators.

---

## 5. Simulation Framework

The simulator generates synthetic multi-vehicle GPS traces with realistic noise:

| Parameter | Value |
|---|---|
| GPS noise | 3.0 m (Gaussian) |
| Heading noise | 5.0° |
| Multipath jump probability | 3% (25 m jump) |
| GPS dropout probability | 2% |
| Timestamp jitter | ±0.15 s |
| Ping interval | 1.0 s |

### Vehicle Roles

| Role | Description | Ground Truth Label |
|---|---|---|
| `NORMAL` | Drives in allowed direction | True Negative |
| `WRONG_WAY_INTRUDER` | Drives fully opposite to allowed direction | True Positive |
| `DIVERSION_VEHICLE` | Drives opposite due to roadworks diversion | Should be suppressed |
| `TURNING_VEHICLE` | Makes a legal three-point turn at a junction | True Negative (tests manoeuvre suppression) |

The `TURNING_VEHICLE` role is critical for validation: it briefly presents a wrong-way heading during the reversal phase. Without it, the manoeuvre hold suppression logic is never exercised in evaluation. (This role was defined but never instantiated in v2; corrected in v3.)

---

## 6. Evaluation Methodology

Two complementary evaluation modes are provided:

### Vehicle-Level Evaluation (Classic)
Treats a vehicle as correctly detected if any alert fires at any point during its trajectory.

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

**Limitation:** Hides detection latency. A vehicle detected on its last frame scores as TP; its 20 frames of missed detection are invisible.

### Frame-Level Temporal Evaluation (Novel)
Treats each GPS ping as one observation. Directly measures operational safety properties:

| Metric | Definition |
|---|---|
| `frame_precision` | Fraction of alerted pings that were truly wrong-way |
| `frame_recall` | Fraction of wrong-way pings that had an active alert |
| `alert_coverage_rate` | Fraction of truly-wrong-way time covered by an active alert |
| `mean_frames_to_first_alert` | Detection lag in ping-count units |
| `multipath_fp_rate` | False positive rate on GPS-multipath-flagged pings |
| `jitter_fn_rate` | Miss rate on timestamp-jittered pings |

For a safety-critical system, latency and coverage matter at least as much as eventual recall. Frame-level evaluation is the operationally meaningful metric.

---

## 7. Visualization

The pipeline outputs a self-contained interactive HTML map (`wrong_way_detection_map.html`) built with Mapbox GL JS:

- **Vehicle trajectories** — colour-coded by role (green: normal, red: wrong-way, orange: diversion, yellow: turning)
- **Alert markers** — stamped with risk score and bearing delta at the detection frame
- **Ghost positions** — 👻 markers with dashed projection lines for confirmed wrong-way vehicles [N5]
- **Counter-flow heatmap** — cyan→amber→red overlay showing infrastructure hotspots [N6]
- **Timeline panel** — per-vehicle event timeline with alert timestamps
- **Playback mode** — frame-by-frame animation of the full simulation

No server is required; the HTML file is fully self-contained and opens in any modern browser.

---

## 8. Installation & Quickstart

### Requirements
- Python ≥ 3.9
- Internet access (for OSM Overpass API); or use `--offline` for demo

### Install

```bash
git clone https://github.com/<your-org>/WrongSide.git
cd WrongSide
pip install -e .
```

### Run

```bash
# Online mode — uses live OSM road geometry (Bengaluru city centre)
python main.py --lat 12.9716 --lon 77.5946 --duration 60

# Offline mode — uses MockRoadProvider (no API calls, reproducible)
python main.py --offline --duration 60

# CLI shortcut (after pip install -e .)
wwd --offline --duration 90
```

The output `wrong_way_detection_map.html` will be generated in the working directory. Open it directly in any browser.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--lat` | 12.9716 | Centre latitude for OSM fetch |
| `--lon` | 77.5946 | Centre longitude for OSM fetch |
| `--duration` | 60 | Simulation duration in seconds |
| `--offline` | False | Use offline MockRoadProvider |
| `--config` | default | Override DetectorConfig parameters |

---

## 9. Configuration Reference

All parameters live in `core/config.py` and are injectable via `DetectorConfig`.

### Detection Thresholds

| Parameter | Default | Effect |
|---|---|---|
| `risk_confirm_threshold` | 0.38 | EMA score required for Gate 1 confirmation (↓ = higher recall) |
| `risk_ema_alpha` | 0.45 | EMA smoothing factor (↑ = snappier response to new evidence) |
| `temporal_confirm_seconds` | 1.5 s | Minimum duration above threshold before Gate 1 fires |
| `slow_ww_speed_max_kmh` | 15.0 | Speed ceiling for Gate 3 (slow-WW accumulator) |
| `slow_ww_seconds` | 4.0 | Gate 3 confirmation duration |

### Bayesian Engine (N4)

| Parameter | Default | Effect |
|---|---|---|
| `bayesian_prior` | 0.05 | Prior P(wrong_way) for all vehicles on one-way roads |
| `bayesian_likelihood_ratio_ww` | 8.0 | LR when bearing delta > adaptive threshold |
| `bayesian_likelihood_ratio_ok` | 0.15 | LR when bearing delta < threshold (evidence for normal) |

### Ghost Predictor (N5)

| Parameter | Default | Effect |
|---|---|---|
| `ghost_prediction_horizon_s` | 5.0 | Seconds ahead to project ghost position |
| `ghost_min_risk` | 0.35 | Minimum risk score to emit a ghost prediction |

### Counter-Flow Heatmap (N6)

| Parameter | Default | Effect |
|---|---|---|
| `heatmap_grid_size_deg` | 0.0003 (~33 m) | Spatial cell size |
| `heatmap_decay_rate` | 0.92 | Per-second multiplicative decay |
| `heatmap_min_score` | 0.05 | Prune threshold (cells below this are deleted) |

---

## 10. Results & Performance

Evaluated on the default extended demo scenario (6 vehicles, 60 s simulation, offline mode):

| Metric | Value |
|---|---|
| Vehicle-level Precision | — |
| Vehicle-level Recall | — |
| Vehicle-level F1 | — |
| Frame-level Coverage Rate | — |
| Mean Frames to First Alert | — |

*(Run `python main.py --offline` to regenerate. Values are scenario-dependent and reported here as placeholders pending extended benchmark.)*

### Qualitative Outcomes Demonstrated

1. **Visual playback** distinguishing normal vehicles vs. wrong-way intruders with clear danger indicators
2. **Logic model** (bearing comparison, road direction, noise handling) explaining how non-ego wrong-way motion is flagged
3. **False positive analysis** for temporary diversions (`DIVERSION_VEHICLE` role) and legal manoeuvres (`TURNING_VEHICLE` role) with mitigation heuristics

---

## 11. Limitations & Future Work

The following are known limitations identified via static analysis of the v3 codebase:

| Issue | Severity | Detail | Recommended Fix |
|---|---|---|---|
| Location-unaware OSM cache | Medium | Cache key is time-only; geographically distant pings within one session use stale road data | Add quantised (lat, lon) to cache key |
| Ghost predictor assumes constant heading | Medium | Linear projection diverges on curves beyond ~3 s | Project along the WLS-fitted trajectory curve |
| Global Bayesian prior | Low | P = 0.05 applied uniformly to motorways and residential streets | Make prior a dict keyed on `road_type` |
| Unbounded alert list | Medium | `self._alerts` grows indefinitely in long sessions | Replace with `deque(maxlen=10000)` |
| Overpass fetch radius fixed at 20 m | Medium | May miss roads under high GPS error | Set `radius = max(20, gps_accuracy × 3)` |
| No persistent state between runs | Medium | All vehicle state, heatmap, and alert history is in-memory only | Serialize state to file or Redis at configurable intervals |
| No unit tests | Medium | Evaluator provides integration-level metrics; Kalman, Bayes, heatmap untested in isolation | Add `pytest` suite per module |
| Heatmap decay not triggered on query | Low | If no new events arrive, cells do not decay even as real time passes | Call `_decay_all(ts)` at start of `is_hotspot()` |

### Future Directions

- **V2X integration**: Broadcast ghost positions and alerts over DSRC/C-V2X to surrounding vehicles
- **Fleet-mode consensus**: Fuse detections from multiple ego vehicles observing the same wrong-way intruder
- **Road-type-specific priors**: Fit Bayesian priors from historical incident databases per road class
- **Online learning**: Update EMA alpha and confirm thresholds from confirmed vs. false positive feedback in deployment
- **Lane-level resolution**: Replace centroid-based road matching with polyline projection for multi-lane roads

---

## 12. Dependencies

All dependencies are free and open-source. No paid API keys are required.

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24.0 | Kalman filter, WLS velocity vector, noise generation |
| `requests` | ≥ 2.31.0 | Overpass API HTTP client |
| `folium` | ≥ 0.15.0 | Map visualization (fallback) |
| `osmnx` | ≥ 1.6.0 | OpenStreetMap network analysis |
| `shapely` | ≥ 2.0.0 | Geometric operations |
| `geopy` | ≥ 2.4.0 | Geodesic distance calculations |
| `overpy` | ≥ 0.7 | OSM Overpass API Python client |
| `pandas` | ≥ 2.0.0 | Trace analysis and metrics tabulation |
| `scipy` | ≥ 1.11.0 | Signal processing utilities |

Install all: `pip install -e .`

---

## 13. Acknowledgements

Built as a hackathon submission for **HARMAN Automotive Proposal-3: Wrong-Way Driver Detection beyond Ramps**.

Road geometry data sourced from **OpenStreetMap** contributors under the [ODbL license](https://opendatacommons.org/licenses/odbl/). Visualization powered by **Mapbox GL JS**.

---

<div align="center">

*WrongSide v3 — GPS + OSM Wrong-Way Driver Detection*  
*No cameras. No paid APIs. No deep learning. Just geometry.*

</div>
