"""
Detection Evaluation & False Positive Analysis

ORIGINAL: Precision / Recall / F1 at vehicle level.

NEW (critique §6.2) — Frame-level temporal evaluation added:
  evaluate_detection_temporal() measures:
    • Per-ping accuracy (not just per-vehicle)
    • Early vs late detection timing
    • How many wrong-way frames were missed before first alert
    • Repeated-alert coverage (how long the vehicle stays alerted)
    • Multipath / jitter impact on detection quality (if artifact flags present)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Set
from collections import defaultdict

from core.detector import WrongWayAlert
from simulation.simulator import GPSPing

logger = logging.getLogger(__name__)


# ── Vehicle-level result (original) ───────────────────────

@dataclass
class EvalResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    mean_detection_latency_s: float
    suppression_rate: float


# ── Frame-level result (NEW §6.2) ─────────────────────────

@dataclass
class TemporalEvalResult:
    """
    Frame-level evaluation — each GPS ping is treated as one observation.

    This captures:
      - How many individual wrong-way pings were correctly alerted
        (even if the vehicle was eventually caught, early pings may be missed).
      - Alert coverage: what fraction of wrong-way *time* had an active alert.
      - Mean frames missed before first alert (detection lag in ping-count).
      - Multipath / jitter impact: separate precision on artifact-free pings.
    """
    # Per-frame TP/FP/FN/TN
    frame_tp: int
    frame_fp: int
    frame_fn: int
    frame_tn: int
    frame_precision: float
    frame_recall: float
    frame_f1: float

    # Timing
    mean_frames_to_first_alert: float
    """Average number of wrong-way pings seen before the first alert fires."""

    alert_coverage_rate: float
    """Fraction of truly-wrong-way pings that occurred while an alert was active."""

    # Artifact impact
    multipath_fp_rate: float
    """False positive rate on pings flagged as multipath (0 if none present)."""

    jitter_fn_rate: float
    """Miss rate on pings flagged as timestamp-jittered (0 if none present)."""


# ── Original vehicle-level evaluation ─────────────────────

def evaluate_detection(
    pings: List[GPSPing],
    alerts: List[WrongWayAlert],
    wrong_way_vehicle_ids: List[str],
    diversion_vehicle_ids: List[str],
    sim_start_time: float = 0.0,
) -> EvalResult:
    """
    Compare detector alerts against ground truth at the vehicle level.
    Only non-suppressed alerts count.
    """
    alerted_vehicles: Set[str] = {a.vehicle_id for a in alerts if not a.suppressed}

    tp = len([v for v in wrong_way_vehicle_ids if v in alerted_vehicles])
    fn = len([v for v in wrong_way_vehicle_ids if v not in alerted_vehicles])
    fp = len([v for v in alerted_vehicles if v not in wrong_way_vehicle_ids])

    all_vehicles = {p.vehicle_id for p in pings}
    non_ww = all_vehicles - set(wrong_way_vehicle_ids)
    tn = len([v for v in non_ww if v not in alerted_vehicles])

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    # Detection latency
    first_wrong_pings: Dict[str, float] = {}
    for p in pings:
        if p.is_truly_wrong_way and p.vehicle_id not in first_wrong_pings:
            first_wrong_pings[p.vehicle_id] = p.timestamp
    latencies = [
        max(0.0, a.timestamp - first_wrong_pings[a.vehicle_id])
        for a in alerts
        if not a.suppressed and a.vehicle_id in first_wrong_pings
    ]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    diversion_suppressed = len(
        [v for v in diversion_vehicle_ids if v not in alerted_vehicles])
    suppression_rate = (diversion_suppressed / len(diversion_vehicle_ids)
                        if diversion_vehicle_ids else 1.0)

    result = EvalResult(
        true_positives=tp, false_positives=fp,
        false_negatives=fn, true_negatives=tn,
        precision=precision, recall=recall, f1=f1,
        mean_detection_latency_s=mean_latency,
        suppression_rate=suppression_rate,
    )
    _log_vehicle_report(result, alerted_vehicles, wrong_way_vehicle_ids)
    return result


# ── NEW §6.2: Frame-level temporal evaluation ─────────────

def evaluate_detection_temporal(
    pings: List[GPSPing],
    alerts: List[WrongWayAlert],
    wrong_way_vehicle_ids: List[str],
) -> TemporalEvalResult:
    """
    Evaluate detection quality at the individual ping (frame) level.

    For each GPS ping we determine:
      - Ground truth: is this ping truly wrong-way?
      - Prediction: was an alert active for this vehicle at this timestamp?

    An alert is considered 'active' from its timestamp until the next alert
    for the same vehicle or until ALERT_ACTIVE_WINDOW_S seconds later,
    whichever comes first.  This models a real system that holds an alert
    state between events.

    CRITIQUE FIX §6.2: "Vehicle-level evaluation ignores temporal correctness,
    early vs late detection, and repeated alerts."
    """
    ALERT_ACTIVE_WINDOW_S = 10.0   # alert stays active for this many seconds

    # Build a per-vehicle list of (alert_start, alert_end) windows
    non_suppressed = [a for a in alerts if not a.suppressed]
    alert_windows: Dict[str, List[tuple]] = defaultdict(list)
    for a in sorted(non_suppressed, key=lambda x: x.timestamp):
        vid = a.vehicle_id
        windows = alert_windows[vid]
        if windows and a.timestamp <= windows[-1][1]:
            # Extend existing window
            windows[-1] = (windows[-1][0], a.timestamp + ALERT_ACTIVE_WINDOW_S)
        else:
            windows.append((a.timestamp, a.timestamp + ALERT_ACTIVE_WINDOW_S))

    def _is_alerted(vehicle_id: str, timestamp: float) -> bool:
        for start, end in alert_windows.get(vehicle_id, []):
            if start <= timestamp <= end:
                return True
        return False

    ww_ids = set(wrong_way_vehicle_ids)

    # Per-frame counters
    frame_tp = frame_fp = frame_fn = frame_tn = 0

    # Timing: frames before first alert per vehicle
    first_alert_t: Dict[str, float] = {
        a.vehicle_id: a.timestamp
        for a in sorted(non_suppressed, key=lambda x: x.timestamp)
        if a.vehicle_id not in {b.vehicle_id for b in non_suppressed
                                 if b.timestamp < a.timestamp}
    }
    # Simpler rebuild: earliest alert per vehicle
    first_alert_t = {}
    for a in sorted(non_suppressed, key=lambda x: x.timestamp):
        if a.vehicle_id not in first_alert_t:
            first_alert_t[a.vehicle_id] = a.timestamp

    frames_before_alert: Dict[str, int] = defaultdict(int)   # vehicle → count
    first_ww_t: Dict[str, float] = {}

    # Artifact impact
    multipath_fp = multipath_total_non_ww = 0
    jitter_fn = jitter_total_ww = 0

    # Coverage: wrong-way pings while alert was active
    ww_ping_count = ww_alerted_count = 0

    for p in pings:
        is_ww    = p.is_truly_wrong_way
        alerted  = _is_alerted(p.vehicle_id, p.timestamp)

        if is_ww:
            ww_ping_count += 1
            if alerted:
                ww_alerted_count += 1

        # Frames-before-first-alert: count wrong-way pings before alert fires
        if is_ww and p.vehicle_id in ww_ids:
            if p.vehicle_id not in first_ww_t:
                first_ww_t[p.vehicle_id] = p.timestamp
            fat = first_alert_t.get(p.vehicle_id)
            if fat is None or p.timestamp < fat:
                frames_before_alert[p.vehicle_id] += 1

        # Frame-level confusion matrix
        if is_ww and alerted:
            frame_tp += 1
        elif is_ww and not alerted:
            frame_fn += 1
        elif not is_ww and alerted:
            frame_fp += 1
        else:
            frame_tn += 1

        # Artifact impact
        mp = getattr(p, "is_multipath", False)
        jt = getattr(p, "is_timestamp_jittered", False)
        if mp and not is_ww:
            multipath_total_non_ww += 1
            if alerted:
                multipath_fp += 1
        if jt and is_ww:
            jitter_total_ww += 1
            if not alerted:
                jitter_fn += 1

    fp_prec  = frame_tp / (frame_tp + frame_fp + 1e-9)
    fp_rec   = frame_tp / (frame_tp + frame_fn + 1e-9)
    fp_f1    = 2 * fp_prec * fp_rec / (fp_prec + fp_rec + 1e-9)

    mean_frames = (
        sum(frames_before_alert.values()) / len(frames_before_alert)
        if frames_before_alert else 0.0
    )
    coverage = ww_alerted_count / (ww_ping_count + 1e-9)
    mp_fp_rate = multipath_fp / (multipath_total_non_ww + 1e-9)
    jt_fn_rate = jitter_fn   / (jitter_total_ww   + 1e-9)

    result = TemporalEvalResult(
        frame_tp=frame_tp, frame_fp=frame_fp,
        frame_fn=frame_fn, frame_tn=frame_tn,
        frame_precision=fp_prec,
        frame_recall=fp_rec,
        frame_f1=fp_f1,
        mean_frames_to_first_alert=mean_frames,
        alert_coverage_rate=coverage,
        multipath_fp_rate=mp_fp_rate,
        jitter_fn_rate=jt_fn_rate,
    )
    _log_temporal_report(result)
    return result


# ── Logging helpers ────────────────────────────────────────

def _log_vehicle_report(result: EvalResult,
                         alerted: Set[str],
                         ground_truth: List[str]):
    logger.info("=" * 60)
    logger.info("VEHICLE-LEVEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"  True Positives       : {result.true_positives}")
    logger.info(f"  False Positives      : {result.false_positives}")
    logger.info(f"  False Negatives      : {result.false_negatives}")
    logger.info(f"  True Negatives       : {result.true_negatives}")
    logger.info(f"  Precision            : {result.precision:.3f}")
    logger.info(f"  Recall               : {result.recall:.3f}")
    logger.info(f"  F1 Score             : {result.f1:.3f}")
    logger.info(f"  Mean Detect Latency  : {result.mean_detection_latency_s:.2f}s")
    logger.info(f"  Diversion Suppression: {result.suppression_rate:.1%}")
    logger.info("=" * 60)


def _log_temporal_report(result: TemporalEvalResult):
    logger.info("=" * 60)
    logger.info("FRAME-LEVEL TEMPORAL EVALUATION  (NEW §6.2)")
    logger.info("=" * 60)
    logger.info(f"  Frame TP             : {result.frame_tp}")
    logger.info(f"  Frame FP             : {result.frame_fp}")
    logger.info(f"  Frame FN             : {result.frame_fn}")
    logger.info(f"  Frame TN             : {result.frame_tn}")
    logger.info(f"  Frame Precision      : {result.frame_precision:.3f}")
    logger.info(f"  Frame Recall         : {result.frame_recall:.3f}")
    logger.info(f"  Frame F1             : {result.frame_f1:.3f}")
    logger.info(f"  Mean frames before 1st alert : {result.mean_frames_to_first_alert:.1f}")
    logger.info(f"  Alert coverage rate  : {result.alert_coverage_rate:.1%}")
    logger.info(f"  Multipath FP rate    : {result.multipath_fp_rate:.1%}")
    logger.info(f"  Timestamp jitter FN  : {result.jitter_fn_rate:.1%}")
    logger.info("=" * 60)
