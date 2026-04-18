"""
Wrong-Way Driver Detection — Main Pipeline  v3

Changes vs v2:
  - Ghost predictions and heatmap passed to build_visualization
  - Novelty features N4/N5/N6 logged in summary
  - sys.path fallback retained as before
"""

import argparse
import logging
import sys
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wwd.main")

_pkg_root = os.path.dirname(os.path.abspath(__file__))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from simulation.simulator import build_extended_demo_scenario, build_harman_demo_scenario
from core.osm_resolver import OSMRoadResolver
from core.detector import WrongWayDetector
from core.evaluator import evaluate_detection, evaluate_detection_temporal
from core.interfaces import MockRoadProvider
from core.config import DetectorConfig, DEFAULT_DETECTOR_CFG
from visualization.map_builder import build_visualization


def detect_dominant_bearing(segments) -> float:
    drivable = ("motorway", "trunk", "primary", "secondary",
                "tertiary", "residential", "unclassified",
                "motorway_link", "trunk_link", "primary_link")
    candidates = [s for s in segments if s.road_type in drivable and s.is_oneway]
    if not candidates:
        candidates = [s for s in segments if s.road_type in drivable]
    if not candidates:
        logger.warning("No drivable roads in OSM — using default bearing 90°")
        return 90.0
    priority = {t: i for i, t in enumerate(drivable)}
    best = min(candidates, key=lambda s: priority.get(s.road_type, 99))
    logger.info(f"      Dominant road: '{best.name}' ({best.road_type}) "
                f"bearing={best.allowed_direction:.1f}°")
    return best.allowed_direction


def create_resolver(offline_mode: bool, metadata: dict,
                    center_lat: float, center_lon: float,
                    max_retries: int = 3):
    if offline_mode:
        logger.info("      OFFLINE MODE — using MockRoadProvider ✅")
        return (MockRoadProvider(road_bearing=metadata["road_bearing"]),
                metadata["road_bearing"])

    logger.info("      Connecting to OSM Overpass API…")
    for attempt in range(max_retries):
        try:
            resolver = OSMRoadResolver(cache_ttl_seconds=600)
            logger.info(f"      Testing API (attempt {attempt + 1})…")
            segments = resolver.fetch_roads_near(center_lat, center_lon)
            real_bearing = detect_dominant_bearing(segments)
            logger.info(f"      API OK ✅  road bearing = {real_bearing:.1f}°")
            return resolver, real_bearing
        except Exception as exc:
            wait = (attempt + 1) * 3
            logger.warning(f"      Attempt {attempt + 1} failed: {exc}")
            if attempt < max_retries - 1:
                logger.info(f"      Retrying in {wait}s…")
                time.sleep(wait)
            else:
                logger.warning("      All retries exhausted — OFFLINE fallback ⚠️")
                return (MockRoadProvider(road_bearing=metadata["road_bearing"]),
                        metadata["road_bearing"])


def run_pipeline(
    center_lat: float = 12.9716,
    center_lon: float = 77.5946,
    duration_s: float = 60.0,
    offline_mode: bool = False,
    output_dir: str = ".",
    detector_config: DetectorConfig = DEFAULT_DETECTOR_CFG,
) -> dict:
    logger.info("=" * 60)
    logger.info("Wrong-Way Detection System  v3")
    logger.info("=" * 60)

    # 1 — Build scenario
    logger.info("[1/6] Building simulation")
    sim, metadata = build_extended_demo_scenario(center_lat, center_lon)

    # 2 — Resolver
    logger.info("[2/6] Initialising resolver")
    resolver, real_bearing = create_resolver(
        offline_mode, metadata, center_lat, center_lon)

    if not offline_mode and abs(real_bearing - metadata["road_bearing"]) > 5.0:
        logger.info(f"      Realigning simulation: "
                    f"{metadata['road_bearing']}° → {real_bearing:.1f}°")
        sim, metadata = build_extended_demo_scenario(
            center_lat, center_lon, road_bearing=real_bearing)

    # 3 — Generate traces
    logger.info("[3/6] Generating GPS traces")
    pings = sim.generate_traces(duration_s=duration_s, ping_interval_s=1.0)
    artifact_pings = sum(1 for p in pings
                         if getattr(p, "is_multipath", False)
                         or getattr(p, "is_timestamp_jittered", False))
    logger.info(f"      {len(pings)} pings generated | "
                f"{artifact_pings} artifact pings "
                f"(road bearing={metadata['road_bearing']:.1f}°)")

    # 4 — Detection
    logger.info("[4/6] Running detection")
    detector = WrongWayDetector(road_resolver=resolver, config=detector_config)
    t0 = time.time()
    for i, ping in enumerate(pings):
        detector.update_vehicle(
            vehicle_id=ping.vehicle_id,
            lat=ping.lat, lon=ping.lon,
            timestamp=ping.timestamp,
            heading=ping.heading,
            speed_kmh=ping.speed_kmh,
        )
        if not offline_mode and i % 5 == 0:
            time.sleep(0.2)
    logger.info(f"      Done in {time.time() - t0:.2f}s")

    # 5 — Evaluation
    logger.info("[5/6] Evaluating")
    eval_result = evaluate_detection(
        pings=pings,
        alerts=detector.get_alerts(),
        wrong_way_vehicle_ids=metadata["wrong_way_vehicles"],
        diversion_vehicle_ids=metadata["diversion_vehicles"],
    )
    temporal_result = evaluate_detection_temporal(
        pings=pings,
        alerts=detector.get_all_alerts(),
        wrong_way_vehicle_ids=metadata["wrong_way_vehicles"],
    )

    logger.info(f"      [Vehicle] P={eval_result.precision:.2f} "
                f"R={eval_result.recall:.2f} F1={eval_result.f1:.2f}")
    logger.info(f"      [Frame]   P={temporal_result.frame_precision:.2f} "
                f"R={temporal_result.frame_recall:.2f} "
                f"F1={temporal_result.frame_f1:.2f} | "
                f"coverage={temporal_result.alert_coverage_rate:.0%} | "
                f"lag={temporal_result.mean_frames_to_first_alert:.1f} frames")

    # Explainability
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPLAINABILITY REPORT (per vehicle)")
    logger.info("=" * 60)
    for entry in detector.get_explainability_report():
        status = "WRONG-WAY" if entry["is_confirmed_wrong_way"] else "normal  "
        delta = entry.get("angular_delta")
        delta_str = f"{delta:.1f}°" if delta is not None else "N/A"
        certainty = entry.get("map_match_certainty")
        cert_str = f"{certainty:.2f}" if certainty is not None else "N/A"
        vv = "VV" if entry.get("velocity_vector_heading") is not None else "pw"
        bayes_str = f"{entry.get('bayes_posterior', 0):.3f}"
        logger.info(
            f"  [{status}] {entry['vehicle_id']:15s} | "
            f"risk={entry['risk_score']:.2f} | "
            f"bayes={bayes_str} | "
            f"delta={delta_str:>7} | "
            f"cert={cert_str} | "
            f"hconf={entry.get('heading_confidence', 1.0):.2f} | "
            f"hdg={vv} | "
            f"fail={entry['failure_mode']:20s} | "
            f"{entry['narrative']}"
        )

    stats = detector.get_stats()
    early_warnings = detector.get_early_warnings()
    ghost_preds = detector.get_ghost_predictions()
    heatmap = detector.get_heatmap()

    logger.info("")
    logger.info("=" * 60)
    logger.info("NOVELTY FEATURES")
    logger.info("=" * 60)
    logger.info(f"  [N1] Speed-adaptive road profiles : active")
    logger.info(f"  [N2] Collision risk (spatial grid): "
                f"{sum(1 for a in detector.get_alerts() if a.collision_risk > 0.1)}")
    logger.info(f"  [N3] Early warnings fired         : {stats['early_warnings']}")
    logger.info(f"  [N4] Bayesian confirmations       : {stats.get('bayes_confirmed',0)}")
    logger.info(f"       Prior={detector_config.bayesian_prior:.2f}  "
                f"LR_ww={detector_config.bayesian_likelihood_ratio_ww}  "
                f"LR_ok={detector_config.bayesian_likelihood_ratio_ok}")
    logger.info(f"  [N5] Ghost vehicle predictions    : {stats.get('ghost_predictions',0)}")
    for gp in ghost_preds:
        logger.info(f"       👻 {gp['vehicle_id']} → "
                    f"({gp['ghost_lat']:.5f},{gp['ghost_lon']:.5f}) "
                    f"@ T+{gp['ghost_ts']:.1f}s")
    logger.info(f"  [N6] Counter-flow hotspot cells   : {stats.get('heatmap_hotspots',0)}")
    for hs in heatmap.get_hotspots():
        logger.info(f"       🌡️  cell=({hs['approx_lat']:.4f},{hs['approx_lon']:.4f}) "
                    f"score={hs['score']:.2f}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("ARCHITECTURAL & ALGORITHMIC FIXES")
    logger.info("=" * 60)
    logger.info(f"  [§arch]  RoadProvider Protocol    : MockRoadProvider / OSMRoadResolver")
    logger.info(f"  [§arch]  DetectorConfig injected  : risk_threshold={detector_config.risk_confirm_threshold}")
    logger.info(f"  [§4.3]   Velocity-vector heading  : {stats.get('velocity_vector_used', 0)} vehicle(s)")
    logger.info(f"  [§2.4]   Spatial grid collision   : O(1) neighbour lookup active")
    logger.info(f"  [§5.2]   GPS artifacts in sim     : multipath + dropout + jitter")
    logger.info(f"  [§6.2]   Frame-level evaluation   : coverage={temporal_result.alert_coverage_rate:.0%}")
    logger.info(f"  [§4.1]   Maneuver suppression     : {stats.get('maneuver_suppressed', 0)} vehicle(s)")
    logger.info(f"  [§9]     Slow wrong-way detection : {stats.get('slow_ww_detections', 0)} alert(s)")
    logger.info(f"  [§11.3]  Severity-scaled cooldown : active")
    logger.info(f"  [BUG]    EW threshold fix          : correctly uses cfg int fields")
    logger.info(f"  [BUG]    Spatial grid pruning      : empty cells removed on update")
    logger.info(f"  [BUG]    persist_min_frames fix    : uses trajectory_window_size denominator")
    logger.info(f"  [BUG]    slow_ww clamp fix         : accumulator floored at 0")
    logger.info(f"  [BUG]    TURNING_VEHICLE added     : now properly instantiated in sim")
    logger.info("=" * 60)

    # 6 — Visualization
    output_map = os.path.join(output_dir, "wrong_way_detection_map.html")
    logger.info(f"[6/6] Generating map → {output_map}")
    build_visualization(
        pings=pings,
        alerts=detector.get_all_alerts(),
        vehicle_states=detector.get_all_states(),
        center_lat=center_lat,
        center_lon=center_lon,
        output_path=output_map,
        early_warnings=early_warnings,
        heatmap=heatmap,
        ghost_predictions=ghost_preds,
    )

    logger.info("✅ COMPLETE")
    return {
        "eval": eval_result,
        "temporal_eval": temporal_result,
        "alerts": detector.get_alerts(),
        "map": output_map,
    }


def main_cli():
    parser = argparse.ArgumentParser(description="Wrong-Way Driver Detection v3")
    parser.add_argument("--lat",      type=float, default=12.9716)
    parser.add_argument("--lon",      type=float, default=77.5946)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--offline",  action="store_true")
    parser.add_argument("--output",   type=str,   default=".")
    parser.add_argument("--risk-threshold", type=float,
                        default=DEFAULT_DETECTOR_CFG.risk_confirm_threshold)
    parser.add_argument("--min-speed", type=float,
                        default=DEFAULT_DETECTOR_CFG.min_speed_kmh)
    args = parser.parse_args()

    cfg = DetectorConfig(
        risk_confirm_threshold=args.risk_threshold,
        min_speed_kmh=args.min_speed,
    )

    run_pipeline(
        center_lat=args.lat,
        center_lon=args.lon,
        duration_s=args.duration,
        offline_mode=args.offline,
        output_dir=args.output,
        detector_config=cfg,
    )


if __name__ == "__main__":
    main_cli()
