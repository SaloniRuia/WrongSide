# visualization/map_builder.py
"""
Interactive Map Visualization — Mapbox GL Dark Theme  v3

BUG FIXES vs v2:
  MAP NOT RENDERING:
    py_data now injected inside proper <script> block (was raw text after </script>)

  EXTRA DOTS IN TIMELINE:
    single .tl-item compound element per vehicle

BUG FIXES vs v3-draft:
  [F1] ALERTS POPUP CRASH: `const a` was never declared in alerts-layer click handler
       Fix: added `const a = e.features[0].properties;` at top of handler

  [F2] WRONG BADGES (all showed NORMAL): buildUI used `st.ww` from STATES
       but STATES may be empty at page-load for non-alerted vehicles.
       Fix: derive badge from `role` field on the ping, not from STATES.

  [F3] CONS_IDS PARSE ERROR: Mapbox serialises array properties as JSON strings;
       the try/catch swallowed the error silently leaving consensus blank.
       Fix: always string-coerce then JSON.parse with a safe fallback.

  [F4] SUPPRESSED CARD OVERFLOW: 44 rows flooded the HUD.
       Fix: cap max-height on sup-list and the sup card itself.

NEW in v3:
  [N4] Bayes posterior shown in alert and vehicle popups
  [N5] Ghost vehicle arrows — dashed white lines + 👻 markers
  [N6] Counter-flow heatmap overlay — colour-coded grid cells
"""

import json
import os
import logging
from typing import List, Dict, Optional
api_key = os.getenv("MAPBOX_API_KEY")

from core.detector import WrongWayAlert, VehicleState, CounterFlowHeatmap
from simulation.simulator import GPSPing, VehicleRole

logger = logging.getLogger(__name__)

MAPBOX_TOKEN = api_key

ROLE_JS_KEY = {
    VehicleRole.NORMAL:             "normal",
    VehicleRole.WRONG_WAY_INTRUDER: "wrong_way_intruder",
    VehicleRole.DIVERSION_VEHICLE:  "diversion_vehicle",
    VehicleRole.TURNING_VEHICLE:    "turning_vehicle",
}


def build_visualization(
    pings: List[GPSPing],
    alerts: List[WrongWayAlert],
    vehicle_states: Dict[str, VehicleState],
    center_lat: float,
    center_lon: float,
    output_path: str = "wrong_way_detection_map.html",
    early_warnings: Optional[List[dict]] = None,
    heatmap: Optional[CounterFlowHeatmap] = None,
    ghost_predictions: Optional[List[dict]] = None,
) -> str:

    pings_json = json.dumps([
        {
            "id":   p.vehicle_id,
            "lat":  round(p.lat, 7),
            "lon":  round(p.lon, 7),
            "ts":   round(p.timestamp, 2),
            "hdg":  round(p.heading, 1),
            "spd":  round(p.speed_kmh, 1),
            "role": ROLE_JS_KEY.get(p.role, "normal"),
            "ww":   bool(p.is_truly_wrong_way),
        }
        for p in pings
    ])

    alerts_json = json.dumps([
        {
            "vid":      a.vehicle_id,
            "lat":      round(a.lat, 7),
            "lon":      round(a.lon, 7),
            "ts":       round(a.timestamp, 2),
            "risk":     round(a.risk_score, 3),
            "delta":    round(a.bearing_delta, 1),
            "road":     a.road_name,
            "type":     a.road_type,
            "spd":      round(a.speed_kmh, 1),
            "cr":       round(a.collision_risk, 3),
            "ew":       bool(a.early_warned),
            "sup":      bool(a.suppressed),
            "sup_why":  getattr(a, "suppression_reason", ""),
            "cons":     bool(a.confirmed_by_consensus),
            # [F3] serialise as JSON string so Mapbox round-trip is safe
            "cons_ids": json.dumps(list(a.consensus_vehicle_ids) if a.consensus_vehicle_ids else []),
            "profile":  getattr(a, "road_profile", ""),
            "thresh":   round(getattr(a, "adaptive_threshold", 110.0), 1),
            "slow_ww":  bool(getattr(a, "slow_wrong_way", False)),
            "hdg_src":  getattr(a, "heading_source", "pairwise"),
            "bayes":    round(getattr(a, "bayes_posterior", 0.0), 3),
            "ghost_lat": round(a.ghost_lat, 6) if getattr(a, "ghost_lat", None) is not None else None,
            "ghost_lon": round(a.ghost_lon, 6) if getattr(a, "ghost_lon", None) is not None else None,
            "hotspot":  bool(getattr(a, "is_heatmap_hotspot", False)),
        }
        for a in alerts
    ])

    states_json = json.dumps({
        vid: {
            "risk":     round(s.risk_score, 3),
            "ww":       bool(s.is_confirmed_wrong_way),
            "ew_fired": bool(s.early_warning_fired),
            "cr":       round(getattr(s, "collision_risk", 0.0), 3),
            "lat":      round(s.lat, 7),
            "lon":      round(s.lon, 7),
            "spd":      round(s.speed_kmh, 1),
            "hdg":      round(s.heading, 1) if s.heading is not None else 0.0,
            "fail":     getattr(s, "failure_mode", "none"),
            "delta":    round(s.bearing_delta, 1) if s.bearing_delta is not None else 0.0,
            "cert":     round(s.map_match_certainty, 3) if s.map_match_certainty is not None else 0.0,
            "hconf":    round(getattr(s, "heading_confidence", 1.0), 3),
            "maneuver": round(getattr(s, "maneuver_hold_remaining", 0.0), 1),
            "vv_hdg":   round(s.velocity_vector_heading, 1) if getattr(s, "velocity_vector_heading", None) is not None else None,
            "bayes":    round(getattr(s, "bayes_posterior", 0.0), 3),
            "ghost_lat": round(s.ghost_lat, 6) if getattr(s, "ghost_lat", None) is not None else None,
            "ghost_lon": round(s.ghost_lon, 6) if getattr(s, "ghost_lon", None) is not None else None,
        }
        for vid, s in vehicle_states.items()
    })

    # [N6] Heatmap cells
    heatmap_cells = []
    if heatmap is not None:
        cell_deg = heatmap._cell_deg
        for (row, col), score in heatmap.get_all_cells().items():
            heatmap_cells.append({
                "lat": row * cell_deg + cell_deg / 2,
                "lon": col * cell_deg + cell_deg / 2,
                "score": round(score, 3),
                "size": cell_deg,
            })
    heatmap_json = json.dumps(heatmap_cells)

    # [N5] Ghost predictions
    ghost_json = json.dumps(ghost_predictions or [])

    total_v    = len({p.vehicle_id for p in pings})
    wrong_v    = len({p.vehicle_id for p in pings if p.role == VehicleRole.WRONG_WAY_INTRUDER})
    n_alerts   = len([a for a in alerts if not a.suppressed])
    n_sup      = len([a for a in alerts if a.suppressed])
    ew_count   = sum(1 for s in vehicle_states.values() if s.early_warning_fired)
    cr_count   = sum(1 for a in alerts if not a.suppressed and a.collision_risk > 0.1)
    cons_count = sum(1 for a in alerts if not a.suppressed and a.confirmed_by_consensus)
    n_pings    = len(pings)
    n_hotspot  = sum(1 for a in alerts if not a.suppressed and getattr(a, "is_heatmap_hotspot", False))

    ts_list = [p.timestamp for p in pings]
    ts_min  = round(min(ts_list), 2) if ts_list else 0.0
    ts_max  = round(max(ts_list), 2) if ts_list else 60.0

    center_js   = json.dumps([round(center_lon, 6), round(center_lat, 6)])
    sup_display = "block" if n_sup > 0 else "none"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Wrong-Way Detection System v3</title>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet"/>
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0a0e1a;--panel:#111827;--panel2:#1a2035;--border:#1e2d45;
  --accent:#3b82f6;--red:#ef4444;--amber:#f59e0b;--green:#22c55e;
  --purple:#a855f7;--cyan:#06b6d4;
  --text:#e2e8f0;--muted:#64748b;
  --font:'Space Grotesk',sans-serif;--mono:'JetBrains Mono',monospace;
}}
html,body{{width:100%;height:100%;background:var(--bg);color:var(--text);font-family:var(--font);overflow:hidden}}
#map{{position:absolute;inset:0}}
#hud{{position:fixed;top:16px;left:16px;z-index:200;display:flex;flex-direction:column;gap:10px;pointer-events:none;max-height:calc(100vh - 120px);overflow-y:auto}}
#hud::-webkit-scrollbar{{width:3px}}#hud::-webkit-scrollbar-thumb{{background:var(--border)}}
.card{{background:rgba(17,24,39,.94);border:1px solid var(--border);border-radius:12px;backdrop-filter:blur(14px);pointer-events:all;overflow:hidden}}
.card-header{{padding:10px 14px 8px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}}
.card-header .title{{font-size:13px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}}
.card-body{{padding:10px 14px}}
#clock-card .card-body{{text-align:center;padding:8px 20px 10px}}
#clock-time{{font-family:var(--mono);font-size:28px;font-weight:600;color:var(--accent);letter-spacing:.06em;line-height:1}}
#clock-date{{font-size:11px;color:var(--muted);margin-top:3px;letter-spacing:.04em}}
#clock-sim{{font-size:11px;color:var(--amber);margin-top:2px;font-family:var(--mono)}}
.stats-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
.stat{{background:var(--panel2);border-radius:8px;padding:7px 10px}}
.stat .val{{font-size:20px;font-weight:700;font-family:var(--mono);line-height:1}}
.stat .lbl{{font-size:10px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.05em}}
.stat.red .val{{color:var(--red)}}.stat.amber .val{{color:var(--amber)}}
.stat.green .val{{color:var(--green)}}.stat.blue .val{{color:var(--accent)}}
.stat.purple .val{{color:var(--purple)}}.stat.cyan .val{{color:var(--cyan)}}
.layer-row{{display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);font-size:12px}}
.layer-row:last-child{{border-bottom:none}}
.toggle{{width:36px;height:20px;border-radius:10px;background:var(--border);border:none;cursor:pointer;position:relative;transition:background .2s;flex-shrink:0}}
.toggle.on{{background:var(--accent)}}
.toggle::after{{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:8px;background:#fff;transition:left .2s}}
.toggle.on::after{{left:18px}}
#detail-panel{{position:fixed;top:16px;right:16px;z-index:200;width:280px;display:flex;flex-direction:column;gap:10px;max-height:calc(100vh - 120px);overflow-y:auto}}
#detail-panel::-webkit-scrollbar{{width:3px}}#detail-panel::-webkit-scrollbar-thumb{{background:var(--border)}}
.legend-row{{display:flex;align-items:center;gap:8px;font-size:12px;padding:3px 0}}
.dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.line-sample{{width:24px;height:3px;border-radius:2px;flex-shrink:0}}
#vehicle-list{{max-height:240px;overflow-y:auto}}
#vehicle-list::-webkit-scrollbar{{width:4px}}#vehicle-list::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
.veh-row{{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:8px;cursor:pointer;transition:background .15s;font-size:12px}}
.veh-row:hover{{background:var(--panel2)}}.veh-row.selected{{background:var(--panel2);outline:1px solid var(--accent)}}
.veh-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.veh-name{{flex:1;font-family:var(--mono);font-size:11px}}
.veh-badge{{font-size:9px;padding:2px 6px;border-radius:4px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}}
.badge-ww{{background:rgba(239,68,68,.2);color:var(--red)}}
.badge-ok{{background:rgba(34,197,94,.15);color:var(--green)}}
.badge-div{{background:rgba(245,158,11,.15);color:var(--amber)}}
.badge-turn{{background:rgba(168,85,247,.15);color:var(--purple)}}
.veh-risk{{font-family:var(--mono);font-size:10px;color:var(--muted);min-width:32px;text-align:right}}
/* [F4] cap suppressed card height */
#sup-card{{max-height:180px}}
#sup-list{{max-height:120px;overflow-y:auto;font-size:11px}}
#sup-list::-webkit-scrollbar{{width:3px}}#sup-list::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
.sup-row{{padding:4px 6px;border-radius:6px;margin-bottom:3px;background:var(--panel2);display:flex;justify-content:space-between;gap:6px}}
.sup-vid{{font-family:var(--mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100px}}
.sup-why{{color:var(--muted);font-size:10px;font-family:var(--mono);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:130px}}
#timeline{{position:fixed;bottom:0;left:0;right:0;z-index:300;background:rgba(10,14,26,.97);border-top:1px solid var(--border);backdrop-filter:blur(14px);padding:10px 20px 12px;user-select:none}}
.tl-top{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
#play-btn{{width:34px;height:34px;border-radius:50%;border:none;background:var(--accent);color:#fff;font-size:15px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s;flex-shrink:0;line-height:1}}
#play-btn:hover{{background:#2563eb}}
#rewind-btn{{width:28px;height:28px;border-radius:50%;border:1px solid var(--border);background:var(--panel2);color:var(--text);font-size:12px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0}}
#speed-btn{{background:var(--panel2);border:1px solid var(--border);color:var(--text);font-family:var(--mono);font-size:11px;padding:4px 10px;border-radius:6px;cursor:pointer;flex-shrink:0}}
#ts-label{{font-family:var(--mono);font-size:12px;color:var(--accent);min-width:80px;flex-shrink:0}}
#ts-end-label{{font-family:var(--mono);font-size:10px;color:var(--muted);flex-shrink:0}}
#tl-wrap{{flex:1;display:flex;flex-direction:column;gap:6px}}
#tl-bar{{position:relative;height:8px;background:var(--border);border-radius:4px;cursor:pointer}}
#tl-fill{{height:100%;background:linear-gradient(90deg,var(--accent),#60a5fa);border-radius:4px;pointer-events:none}}
#tl-handle{{position:absolute;top:50%;width:16px;height:16px;background:#fff;border:3px solid var(--accent);border-radius:50%;transform:translate(-50%,-50%);cursor:grab;z-index:2;transition:transform .1s}}
#tl-handle:hover{{transform:translate(-50%,-50%) scale(1.25)}}#tl-handle:active{{cursor:grabbing}}
.tl-tick{{position:absolute;top:0;width:2px;height:100%;background:var(--red);opacity:.55;border-radius:1px;pointer-events:none}}
.tl-vehicles{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding-top:2px}}
.tl-item{{display:flex;align-items:center;gap:4px;cursor:pointer}}
.tl-dot{{width:8px;height:8px;border-radius:50%;display:inline-block;transition:transform .15s,box-shadow .15s;flex-shrink:0}}
.tl-dot.active{{transform:scale(2.2);box-shadow:0 0 8px currentColor}}
.tl-label{{font-size:9px;color:var(--muted)}}
.mapboxgl-popup-content{{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:0!important;box-shadow:0 8px 32px rgba(0,0,0,.6)!important;color:var(--text)!important;font-family:var(--font)!important;min-width:230px}}
.mapboxgl-popup-close-button{{color:var(--muted)!important;font-size:18px!important;right:8px!important;top:6px!important;background:none!important;line-height:1}}
.popup-head{{padding:11px 14px 8px;border-bottom:1px solid var(--border);font-weight:700;font-size:13px;display:flex;align-items:center;gap:6px}}
.popup-body{{padding:9px 14px 12px}}
.popup-row{{display:flex;justify-content:space-between;align-items:center;font-size:11px;padding:2px 0}}
.popup-key{{color:var(--muted)}}.popup-val{{font-family:var(--mono);font-weight:600;text-align:right;max-width:160px;word-break:break-all}}
.popup-divider{{border:none;border-top:1px solid var(--border);margin:5px 0}}
.badge-pill{{font-size:10px;padding:2px 7px;border-radius:10px;font-weight:600;display:inline-block}}
#kb-hint{{position:fixed;bottom:82px;right:16px;background:rgba(17,24,39,.85);border:1px solid var(--border);border-radius:8px;padding:5px 10px;font-size:10px;color:var(--muted);pointer-events:none;animation:fadeout 3s forwards 4s;z-index:400}}
@keyframes fadeout{{to{{opacity:0}}}}
#mode-toggle{{
  position:fixed;top:16px;right:308px;z-index:300;
  display:flex;gap:3px;padding:4px;
  background:rgba(17,24,39,.96);border:1px solid var(--border);
  border-radius:10px;backdrop-filter:blur(14px);pointer-events:all;
}}
.mode-btn{{
  padding:5px 14px;border-radius:7px;border:none;cursor:pointer;
  font-family:var(--font);font-size:11px;font-weight:600;
  letter-spacing:.03em;transition:background .15s,color .15s;
  background:transparent;color:var(--muted);line-height:1.5;
}}
.mode-btn.active{{background:var(--accent);color:#fff}}
.mode-btn:disabled{{opacity:.35;cursor:default}}
#vc-panel{{
  display:none;position:fixed;top:16px;right:16px;z-index:200;
  width:280px;flex-direction:column;gap:10px;
  max-height:calc(100vh - 120px);overflow-y:auto;
}}
#vc-panel::-webkit-scrollbar{{width:3px}}
#vc-panel::-webkit-scrollbar-thumb{{background:var(--border)}}
.vc-row{{display:flex;justify-content:space-between;align-items:center;
  padding:4px 0;font-size:11px;border-bottom:1px solid var(--border)}}
.vc-row:last-child{{border-bottom:none}}
.vc-key{{color:var(--muted)}}
.vc-val{{font-family:var(--mono);font-weight:600}}
.thr-row{{padding:4px 7px;border-radius:6px;margin-bottom:3px;
  background:var(--panel2);display:flex;justify-content:space-between;font-size:11px}}
</style>
</head>
<body>
<div id="map"></div>

<div id="hud">
  <div class="card" id="clock-card">
    <div class="card-header"><span>🕐</span><span class="title">Live Clock</span></div>
    <div class="card-body">
      <div id="clock-time">--:--:--</div>
      <div id="clock-date">Initialising…</div>
      <div id="clock-sim">SIM T+0.0 s</div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><span>📊</span><span class="title">System Stats</span></div>
    <div class="card-body">
      <div class="stats-grid">
        <div class="stat blue"><div class="val">{total_v}</div><div class="lbl">Vehicles</div></div>
        <div class="stat red"><div class="val">{wrong_v}</div><div class="lbl">Wrong-Way</div></div>
        <div class="stat red"><div class="val">{n_alerts}</div><div class="lbl">Alerts</div></div>
        <div class="stat amber"><div class="val">{ew_count}</div><div class="lbl">Early Warn</div></div>
        <div class="stat amber"><div class="val">{cr_count}</div><div class="lbl">Coll. Risk</div></div>
        <div class="stat purple"><div class="val">{cons_count}</div><div class="lbl">Consensus</div></div>
        <div class="stat cyan"><div class="val">{n_hotspot}</div><div class="lbl">Hotspots</div></div>
        <div class="stat green"><div class="val">{n_pings}</div><div class="lbl">Pings</div></div>
        <div class="stat"><div class="val" style="color:var(--muted)">{n_sup}</div><div class="lbl">Suppressed</div></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><span>🗂️</span><span class="title">Layers</span></div>
    <div class="card-body" style="padding:8px 14px">
      <div class="layer-row"><span>Vehicle Traces</span><button class="toggle on" id="tog-traces" onclick="toggleLayer('traces',this)"></button></div>
      <div class="layer-row"><span>🚗 Live Dots</span><button class="toggle on" id="tog-live" onclick="toggleLayer('live',this)"></button></div>
      <div class="layer-row"><span>🚨 Alerts</span><button class="toggle on" id="tog-alerts" onclick="toggleLayer('alerts',this)"></button></div>
      <div class="layer-row"><span>⚠️ Danger Zones</span><button class="toggle on" id="tog-danger" onclick="toggleLayer('danger',this)"></button></div>
      <div class="layer-row"><span>🟠 Collision Halos</span><button class="toggle on" id="tog-collision" onclick="toggleLayer('collision',this)"></button></div>
      <div class="layer-row"><span>⚡ Early Warnings</span><button class="toggle on" id="tog-ew" onclick="toggleLayer('ew',this)"></button></div>
      <div class="layer-row"><span>👻 Ghost Predictions</span><button class="toggle on" id="tog-ghost" onclick="toggleLayer('ghost',this)"></button></div>
      <div class="layer-row"><span>🌡️ Counter-Flow Heatmap</span><button class="toggle on" id="tog-heatmap" onclick="toggleLayer('heatmap',this)"></button></div>
      <div class="layer-row"><span>🔕 Suppressed Alerts</span><button class="toggle" id="tog-sup" onclick="toggleLayer('sup',this)"></button></div>
      <div class="layer-row"><span>🛰 Satellite View</span><button class="toggle" id="tog-sat" onclick="toggleSatellite(this)"></button></div>
    </div>
  </div>

  <!-- [F4] suppressed card capped with id for max-height CSS -->
  <div class="card" id="sup-card" style="display:{sup_display}">
    <div class="card-header"><span>🔕</span><span class="title">Suppressed ({n_sup})</span></div>
    <div class="card-body" style="padding:8px 10px"><div id="sup-list"></div></div>
  </div>
</div>

<div id="detail-panel">
  <div class="card">
    <div class="card-header"><span>🗺</span><span class="title">Legend</span></div>
    <div class="card-body">
      <div class="legend-row"><div class="line-sample" style="background:#3b82f6"></div>Normal vehicle</div>
      <div class="legend-row"><div class="line-sample" style="background:#ef4444"></div>Wrong-way (dashed)</div>
      <div class="legend-row"><div class="line-sample" style="background:#f59e0b"></div>Diversion vehicle</div>
      <div class="legend-row"><div class="line-sample" style="background:#a855f7"></div>Turning vehicle</div>
      <div style="margin:6px 0 4px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em">Markers</div>
      <div class="legend-row"><div class="dot" style="background:#ef4444"></div>Confirmed alert</div>
      <div class="legend-row"><div class="dot" style="background:#64748b"></div>Suppressed alert</div>
      <div class="legend-row"><div class="dot" style="background:#f97316;opacity:.6"></div>Collision halo</div>
      <div class="legend-row"><span style="font-size:12px;width:10px">⚡</span>Early warning</div>
      <div class="legend-row"><span style="font-size:12px;width:10px">👻</span>Ghost prediction [N5]</div>
      <div class="legend-row"><div class="dot" style="background:#06b6d4;opacity:.5"></div>Heatmap hotspot [N6]</div>
      <div style="margin:6px 0 4px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em">Live dot colour</div>
      <div class="legend-row"><div class="dot" style="background:#ef4444"></div>WW confirmed</div>
      <div class="legend-row"><div class="dot" style="background:#f59e0b"></div>High risk ≥ 0.5</div>
      <div class="legend-row"><div class="dot" style="background:#3b82f6"></div>Normal / low risk</div>
      <div class="legend-row"><div class="dot" style="background:#4b5563;border:1px solid #6b7280"></div>Low map certainty</div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-size:10px;color:var(--muted)">
        Cert &lt;0.2 → suppressed &nbsp;·&nbsp; Red ticks = alerts on scrub bar
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><span>🚗</span><span class="title">Vehicles</span></div>
    <div class="card-body" style="padding:6px 10px"><div id="vehicle-list"></div></div>
  </div>
</div>

<div id="timeline">
  <div class="tl-top">
    <button id="rewind-btn" title="Rewind (Home)" onclick="rewind()">⏮</button>
    <button id="play-btn" title="Play/Pause (Space)">▶</button>
    <button id="speed-btn" title="Cycle speed (+)">1×</button>
    <div id="tl-wrap">
      <div id="tl-bar">
        <div id="tl-fill" style="width:0%"></div>
        <div id="tl-handle" style="left:0%"></div>
      </div>
      <div class="tl-vehicles" id="tl-dots"></div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px">
      <div id="ts-label">T+0.0 s</div>
      <div id="ts-end-label">/ {ts_max:.1f} s</div>
    </div>
  </div>
</div>

<div id="mode-toggle">
  <button class="mode-btn active" id="btn-global"  onclick="vcGlobal()">🌐 Global</button>
  <button class="mode-btn"        id="btn-vehicle" onclick="vcVehicle()" disabled>🚗 Vehicle</button>
</div>

<div id="vc-panel">
  <div class="card">
    <div class="card-header" style="justify-content:space-between">
      <span>🎯</span>
      <span class="title" id="vc-name" style="flex:1;margin-left:6px">—</span>
      <button onclick="vcGlobal()" style="background:none;border:1px solid var(--border);border-radius:6px;color:var(--muted);font-size:10px;padding:2px 8px;cursor:pointer">← Back</button>
    </div>
    <div class="card-body">
      <div class="vc-row"><span class="vc-key">Role</span>        <span class="vc-val" id="vc-role">—</span></div>
      <div class="vc-row"><span class="vc-key">Speed</span>       <span class="vc-val" id="vc-spd">—</span></div>
      <div class="vc-row"><span class="vc-key">Heading</span>     <span class="vc-val" id="vc-hdg">—</span></div>
      <div class="vc-row"><span class="vc-key">Risk</span>        <span class="vc-val" id="vc-risk">—</span></div>
      <div class="vc-row"><span class="vc-key">Bayes P(WW)</span><span class="vc-val" id="vc-bayes">—</span></div>
      <div class="vc-row"><span class="vc-key">Status</span>      <span class="vc-val" id="vc-status">—</span></div>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><span>⚠️</span><span class="title">Nearby Threats</span></div>
    <div class="card-body" style="padding:8px 10px"><div id="vc-threats"></div></div>
  </div>
</div>

<div id="kb-hint">SPACE play/pause &nbsp;·&nbsp; ←/→ scrub 1 s &nbsp;·&nbsp; + speed &nbsp;·&nbsp; Home rewind &nbsp;·&nbsp; click any marker</div>

<script>
// ── DATA ──────────────────────────────────────────────────
const PINGS         = {pings_json};
const ALERTS        = {alerts_json};
const STATES        = {states_json};
const HEATMAP_CELLS = {heatmap_json};
const GHOST_PREDS   = {ghost_json};
const CENTER        = {center_js};
const TS_MIN        = {ts_min};
const TS_MAX        = {ts_max};
const MAPBOX_TOKEN  = '{MAPBOX_TOKEN}';

// ── ROLE COLOURS ──────────────────────────────────────────
const ROLE_COLOR = {{
  normal:             '#3b82f6',
  wrong_way_intruder: '#ef4444',
  diversion_vehicle:  '#f59e0b',
  turning_vehicle:    '#a855f7',
}};
const FAIL_LABELS = {{
  none: null,
  below_speed_threshold: 'Below speed threshold',
  no_road_match:         'No road match',
  below_threshold:       'Risk below threshold',
  maneuver_hold:         'Maneuver hold active',
}};

// ── CLOCK ─────────────────────────────────────────────────
(function tickClock() {{
  const now = new Date(), pad = n => String(n).padStart(2,'0');
  document.getElementById('clock-time').textContent =
    pad(now.getHours())+':'+pad(now.getMinutes())+':'+pad(now.getSeconds());
  const D=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const M=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  document.getElementById('clock-date').textContent =
    D[now.getDay()]+', '+now.getDate()+' '+M[now.getMonth()]+' '+now.getFullYear();
  setTimeout(tickClock, 1000);
}})();

// ── GROUP PINGS ───────────────────────────────────────────
const vehiclePings = {{}};
PINGS.forEach(p => {{ (vehiclePings[p.id] = vehiclePings[p.id]||[]).push(p); }});
const vehicleIds = Object.keys(vehiclePings);

// ── BUILD VEHICLE LIST + TIMELINE DOTS ───────────────────
// [F2] Badge derived from role field, not from STATES (STATES may be empty at load)
(function buildUI() {{
  const list = document.getElementById('vehicle-list');
  const dotsContainer = document.getElementById('tl-dots');

  vehicleIds.forEach(vid => {{
    const firstPing = vehiclePings[vid][0] || {{}};
    const role  = firstPing.role || 'normal';
    const color = ROLE_COLOR[role] || '#6b7280';
    const st    = STATES[vid] || {{}};

    // [F2] Role-based badge — correct even when STATES.ww hasn't been set yet
    const isWW   = role === 'wrong_way_intruder';
    const isDiv  = role === 'diversion_vehicle';
    const isTurn = role === 'turning_vehicle';
    const badge  = isWW ? 'badge-ww' : isDiv ? 'badge-div' : isTurn ? 'badge-turn' : 'badge-ok';
    const label  = isWW ? 'WRONG WAY' : isDiv ? 'DIVERSION' : isTurn ? 'TURNING' : 'NORMAL';

    const row = document.createElement('div');
    row.className = 'veh-row';
    row.id = 'vr-' + vid;
    row.innerHTML = `<div class="veh-dot" style="background:${{color}}"></div>
      <span class="veh-name">${{vid}}</span>
      <span class="veh-badge ${{badge}}">${{label}}</span>
      <span class="veh-risk">${{(st.risk||0).toFixed(2)}}</span>`;
    row.onclick = () => vcSelect(vid);
    list.appendChild(row);

    // Single compound .tl-item per vehicle (no duplication)
    const item = document.createElement('div');
    item.className = 'tl-item';
    item.title = vid;
    item.onclick = () => flyToVehicle(vid);
    const dotEl = document.createElement('span');
    dotEl.className = 'tl-dot';
    dotEl.id = 'tld-' + vid;
    dotEl.style.background = color;
    const lblEl = document.createElement('span');
    lblEl.className = 'tl-label';
    lblEl.textContent = vid.slice(0, 8);
    item.appendChild(dotEl);
    item.appendChild(lblEl);
    dotsContainer.appendChild(item);
  }});

  // [F4] Suppressed list — truncated display
  ALERTS.filter(a => a.sup).forEach(a => {{
    const row = document.createElement('div');
    row.className = 'sup-row';
    row.innerHTML = `<span class="sup-vid">${{a.vid}}</span><span class="sup-why">${{a.sup_why||'—'}}</span>`;
    document.getElementById('sup-list').appendChild(row);
  }});

  // Alert tick marks on scrub bar
  const bar  = document.getElementById('tl-bar');
  const span = (TS_MAX - TS_MIN) || 1;
  ALERTS.filter(a => !a.sup).forEach(a => {{
    const t = document.createElement('div');
    t.className = 'tl-tick';
    t.style.left = ((a.ts - TS_MIN) / span * 100) + '%';
    t.title = a.vid + ' alert T+' + a.ts.toFixed(1) + 's';
    bar.appendChild(t);
  }});
}})();

// ── INTERPOLATION ─────────────────────────────────────────
function interp(vid, ts) {{
  const arr = vehiclePings[vid];
  if (!arr || !arr.length) return null;
  if (ts <= arr[0].ts) return arr[0];
  if (ts >= arr[arr.length-1].ts) return arr[arr.length-1];
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) {{ const m = (lo+hi)>>1; arr[m].ts < ts ? lo = m+1 : hi = m; }}
  const a = arr[lo-1], b = arr[lo], t = (ts-a.ts)/(b.ts-a.ts+1e-9);
  const dd = ((b.hdg-a.hdg+540)%360)-180;
  return {{ ...b, lat:a.lat+(b.lat-a.lat)*t, lon:a.lon+(b.lon-a.lon)*t,
            hdg:a.hdg+dd*t, spd:a.spd+(b.spd-a.spd)*t }};
}}

// ── PLAYBACK STATE ────────────────────────────────────────
let currentTs = TS_MIN, playing = false, speed = 1, speedIdx = 0;
let rafId = null, lastWall = null, mapReady = false;
const speeds = [0.5, 1, 2, 5, 10];

function setTs(ts) {{
  currentTs = Math.max(TS_MIN, Math.min(TS_MAX, ts));
  const frac = (TS_MAX > TS_MIN) ? (currentTs - TS_MIN) / (TS_MAX - TS_MIN) : 0;
  document.getElementById('tl-fill').style.width  = (frac * 100) + '%';
  document.getElementById('tl-handle').style.left = (frac * 100) + '%';
  document.getElementById('ts-label').textContent  = 'T+' + currentTs.toFixed(1) + ' s';
  document.getElementById('clock-sim').textContent = 'SIM T+' + currentTs.toFixed(1) + ' s';
  if (mapReady) renderFrame(currentTs);
}}

// ── DUAL MODE STATE ───────────────────────────────────────
let vcMode = 'global', vcFocus = null, vcLastPos = null;

function renderFrame(ts) {{
  let fLon = null, fLat = null;
  if (vcMode === 'vehicle' && vcFocus) {{
    const fp = interp(vcFocus, ts);
    if (fp) {{ fLon = fp.lon; fLat = fp.lat; }}
    else    {{ vcFocus = null; vcGlobal(); }}
  }}
  const NEAR = 0.004;

  const feats = [];
  vehicleIds.forEach(vid => {{
    const p  = interp(vid, ts); if (!p) return;
    const st = STATES[vid] || {{}};
    let color;
    if (p.ww)                  color = '#ef4444';
    else if ((st.risk||0)>=0.5) color = '#f59e0b';
    else if ((st.cert||1)<0.25) color = '#4b5563';
    else                        color = ROLE_COLOR[p.role] || '#6b7280';

    let _op = (st.cert||1) < 0.25 ? 0.65 : 0.95;
    let _rBonus = 0;
    if (vcMode === 'vehicle' && fLon !== null) {{
      if (vid === vcFocus)                          {{ _op = 1.0; _rBonus = 5; }}
      else if (p.ww || (st.risk||0) >= 0.5)        {{ _op = 0.9; }}
      else {{
        const near = Math.abs(p.lon-fLon)<NEAR && Math.abs(p.lat-fLat)<NEAR;
        _op = near ? 0.65 : 0.18;
      }}
    }}

    feats.push({{ type:'Feature',
      properties: {{ ...p, color, cert:st.cert||0, maneuver:st.maneuver||0,
                    fail:st.fail||'none', vv_hdg:st.vv_hdg, risk:st.risk||0,
                    hconf:st.hconf||1, bayes:st.bayes||0, _op, _rBonus }},
      geometry: {{ type:'Point', coordinates:[p.lon, p.lat] }} }});

    const dot = document.getElementById('tld-' + vid);
    if (dot) dot.classList.toggle('active', p.ww || (st.risk||0) >= 0.5);
  }});

  const src = map.getSource('live');
  if (src) src.setData({{ type:'FeatureCollection', features:feats }});

  if (vcMode === 'vehicle' && vcFocus) {{
    const p = interp(vcFocus, ts);
    if (p) {{
      const moved = !vcLastPos
        || Math.abs(p.lon-vcLastPos[0]) > 0.00005
        || Math.abs(p.lat-vcLastPos[1]) > 0.00005;
      if (moved) {{
        vcLastPos = [p.lon, p.lat];
        map.easeTo({{ center:[p.lon,p.lat], zoom:17.5, pitch:55, bearing:p.hdg-10, duration:400 }});
      }}
      vcUpdatePanel(ts, p, feats);
    }}
  }}
}}

function rafLoop(wall) {{
  if (!playing) {{ rafId = null; return; }}
  const dt = lastWall ? (wall - lastWall) / 1000 : 0;
  lastWall = wall;
  const next = currentTs + dt * speed;
  if (next >= TS_MAX) {{
    setTs(TS_MIN); playing = false;
    document.getElementById('play-btn').textContent = '▶';
    rafId = null; return;
  }}
  setTs(next);
  rafId = requestAnimationFrame(rafLoop);
}}

window.togglePlay = function() {{
  playing = !playing;
  document.getElementById('play-btn').textContent = playing ? '⏸' : '▶';
  if (playing) {{
    if (currentTs >= TS_MAX) setTs(TS_MIN);
    lastWall = null;
    rafId = requestAnimationFrame(rafLoop);
  }} else {{
    if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
  }}
}};
window.rewind = function() {{
  if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
  playing = false;
  document.getElementById('play-btn').textContent = '▶';
  setTs(TS_MIN);
}};
window.cycleSpeed = function() {{
  speedIdx = (speedIdx + 1) % speeds.length;
  speed = speeds[speedIdx];
  document.getElementById('speed-btn').textContent = speed + '×';
}};

document.getElementById('play-btn').addEventListener('click', window.togglePlay);
document.getElementById('speed-btn').addEventListener('click', window.cycleSpeed);

// ── SCRUB BAR ─────────────────────────────────────────────
(function setupScrub() {{
  const bar = document.getElementById('tl-bar');
  function getFrac(cx) {{
    const r = bar.getBoundingClientRect();
    return Math.max(0, Math.min(1, (cx - r.left) / r.width));
  }}
  let dragging = false;
  bar.addEventListener('mousedown', e => {{ dragging = true; e.preventDefault(); setTs(TS_MIN + getFrac(e.clientX)*(TS_MAX-TS_MIN)); }});
  window.addEventListener('mousemove', e => {{ if (!dragging) return; setTs(TS_MIN + getFrac(e.clientX)*(TS_MAX-TS_MIN)); }});
  window.addEventListener('mouseup', () => {{ dragging = false; }});
  bar.addEventListener('touchstart', e => {{ dragging = true; setTs(TS_MIN + getFrac(e.touches[0].clientX)*(TS_MAX-TS_MIN)); }}, {{passive:true}});
  window.addEventListener('touchmove', e => {{ if (!dragging) return; setTs(TS_MIN + getFrac(e.touches[0].clientX)*(TS_MAX-TS_MIN)); }}, {{passive:true}});
  window.addEventListener('touchend', () => {{ dragging = false; }});
}})();

// ── KEYBOARD ──────────────────────────────────────────────
document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space')      {{ e.preventDefault(); window.togglePlay(); }}
  if (e.code === 'ArrowRight') setTs(currentTs + 1);
  if (e.code === 'ArrowLeft')  setTs(currentTs - 1);
  if (e.key  === '+' || e.key === '=') window.cycleSpeed();
  if (e.code === 'Home') window.rewind();
}});

// ── MAP ───────────────────────────────────────────────────
mapboxgl.accessToken = MAPBOX_TOKEN;
const map = new mapboxgl.Map({{
  container: 'map',
  style: 'mapbox://styles/mapbox/dark-v11',
  center: CENTER, zoom: 16, pitch: 30, bearing: -10, antialias: true,
}});
map.addControl(new mapboxgl.NavigationControl(), 'bottom-left');
map.addControl(new mapboxgl.ScaleControl({{ unit:'metric' }}), 'bottom-left');

function addAllLayers() {{
  // ── Traces ──────────────────────────────────────────────
  if (!map.getSource('traces')) {{
    const traceFeats = vehicleIds.map(vid => {{
      const vp = vehiclePings[vid], role = vp[0].role;
      return {{ type:'Feature',
        properties: {{ vid, role, color:ROLE_COLOR[role]||'#6b7280', isWW:role==='wrong_way_intruder' }},
        geometry: {{ type:'LineString', coordinates: vp.map(p => [p.lon, p.lat]) }} }};
    }});
    map.addSource('traces', {{ type:'geojson', data:{{ type:'FeatureCollection', features:traceFeats }} }});
    map.addLayer({{ id:'traces-layer', type:'line', source:'traces',
      paint: {{ 'line-color':['get','color'], 'line-width':['case',['get','isWW'],3.5,2], 'line-opacity':0.65 }} }});
    map.addLayer({{ id:'traces-ww-dash', type:'line', source:'traces',
      filter: ['==',['get','isWW'],true],
      paint: {{ 'line-color':'#ef4444', 'line-width':3, 'line-dasharray':[4,3], 'line-opacity':0.9 }} }});
  }}

  // ── [N6] Counter-flow heatmap ────────────────────────────
  if (!map.getSource('heatmap') && HEATMAP_CELLS.length > 0) {{
    const hf = HEATMAP_CELLS.map(c => {{
      const half = c.size / 2;
      return {{ type:'Feature', properties:{{ score:c.score }},
        geometry: {{ type:'Polygon', coordinates:[[
          [c.lon-half,c.lat-half],[c.lon+half,c.lat-half],
          [c.lon+half,c.lat+half],[c.lon-half,c.lat+half],
          [c.lon-half,c.lat-half]
        ]] }} }};
    }});
    map.addSource('heatmap', {{ type:'geojson', data:{{ type:'FeatureCollection', features:hf }} }});
    map.addLayer({{ id:'heatmap-layer', type:'fill', source:'heatmap', paint: {{
      'fill-color': ['interpolate',['linear'],['get','score'],
        0,'rgba(6,182,212,0)', 1,'rgba(6,182,212,0.2)',
        3,'rgba(245,158,11,0.35)', 6,'rgba(239,68,68,0.5)'],
      'fill-outline-color': 'rgba(6,182,212,0.3)' }} }});
  }}

  // ── Alerts ───────────────────────────────────────────────
  if (!map.getSource('alerts')) {{
    const af = ALERTS.filter(a => !a.sup).map(a => ({{
      type:'Feature', properties:{{...a}},
      geometry:{{ type:'Point', coordinates:[a.lon, a.lat] }}
    }}));
    map.addSource('alerts', {{ type:'geojson', data:{{ type:'FeatureCollection', features:af }} }});
    map.addLayer({{ id:'danger-layer', type:'circle', source:'alerts', paint: {{
      'circle-radius': {{ stops:[[14,50],[18,150]] }},
      'circle-color':'#ef4444','circle-opacity':0.06,
      'circle-stroke-width':1,'circle-stroke-color':'#ef4444','circle-stroke-opacity':0.2 }} }});
    map.addLayer({{ id:'alerts-layer', type:'circle', source:'alerts', paint: {{
      'circle-radius': ['case',['get','hotspot'],10,8],
      'circle-color':  ['case',['get','hotspot'],'#f97316','#ef4444'],
      'circle-stroke-width':2,'circle-stroke-color':'#fff','circle-opacity':0.95 }} }});
  }}

  // ── Collision halos ──────────────────────────────────────
  if (!map.getSource('collision')) {{
    const cf = ALERTS.filter(a => !a.sup && a.cr > 0.05).map(a => ({{
      type:'Feature', properties:{{...a}},
      geometry:{{ type:'Point', coordinates:[a.lon, a.lat] }}
    }}));
    map.addSource('collision', {{ type:'geojson', data:{{ type:'FeatureCollection', features:cf }} }});
    map.addLayer({{ id:'collision-layer', type:'circle', source:'collision', paint: {{
      'circle-radius':  ['+', 28, ['*',['get','cr'],90]],
      'circle-color':   '#f97316',
      'circle-opacity': ['+', 0.07, ['*',['get','cr'],0.22]] }} }});
  }}

  // ── [N5] Ghost vehicle arrows ────────────────────────────
  if (!map.getSource('ghost')) {{
    const gLines = ALERTS.filter(a => !a.sup && a.ghost_lat != null).map(a => ({{
      type:'Feature', properties:{{ vid:a.vid }},
      geometry:{{ type:'LineString', coordinates:[[a.lon,a.lat],[a.ghost_lon,a.ghost_lat]] }}
    }}));
    const gMarkers = ALERTS.filter(a => !a.sup && a.ghost_lat != null).map(a => ({{
      type:'Feature', properties:{{ vid:a.vid }},
      geometry:{{ type:'Point', coordinates:[a.ghost_lon, a.ghost_lat] }}
    }}));
    map.addSource('ghost',         {{ type:'geojson', data:{{ type:'FeatureCollection', features:gLines }} }});
    map.addSource('ghost-markers', {{ type:'geojson', data:{{ type:'FeatureCollection', features:gMarkers }} }});
    map.addLayer({{ id:'ghost-lines', type:'line', source:'ghost',
      paint:{{ 'line-color':'rgba(255,255,255,0.5)', 'line-width':1.5, 'line-dasharray':[3,3] }} }});
    map.addLayer({{ id:'ghost-layer', type:'symbol', source:'ghost-markers',
      layout:{{ 'text-field':'👻', 'text-size':18, 'text-allow-overlap':true }},
      paint:{{ 'text-color':'#fff', 'text-halo-color':'rgba(0,0,0,.5)', 'text-halo-width':1 }} }});
  }}

  // ── Suppressed ───────────────────────────────────────────
  if (!map.getSource('sup')) {{
    const sf = ALERTS.filter(a => a.sup).map(a => ({{
      type:'Feature', properties:{{...a}},
      geometry:{{ type:'Point', coordinates:[a.lon, a.lat] }}
    }}));
    map.addSource('sup', {{ type:'geojson', data:{{ type:'FeatureCollection', features:sf }} }});
    map.addLayer({{ id:'sup-layer', type:'circle', source:'sup',
      layout:{{ visibility:'none' }},
      paint:{{ 'circle-radius':6, 'circle-color':'#64748b',
               'circle-stroke-width':1, 'circle-stroke-color':'#94a3b8', 'circle-opacity':0.7 }} }});
  }}

  // ── Early warnings ───────────────────────────────────────
  if (!map.getSource('ew')) {{
    const ef = Object.entries(STATES).filter(([,s]) => s.ew_fired).map(([vid,s]) => ({{
      type:'Feature', properties:{{ vid }},
      geometry:{{ type:'Point', coordinates:[s.lon, s.lat] }}
    }}));
    map.addSource('ew', {{ type:'geojson', data:{{ type:'FeatureCollection', features:ef }} }});
    map.addLayer({{ id:'ew-layer', type:'symbol', source:'ew',
      layout:{{ 'text-field':'⚡', 'text-size':22, 'text-allow-overlap':true, 'text-offset':[0,-1] }},
      paint:{{ 'text-color':'#f59e0b', 'text-halo-color':'rgba(0,0,0,.6)', 'text-halo-width':2 }} }});
  }}

  // ── Live dots ────────────────────────────────────────────
  if (!map.getSource('live')) {{
    map.addSource('live', {{ type:'geojson', data:{{ type:'FeatureCollection', features:[] }} }});
    map.addLayer({{ id:'live-layer', type:'circle', source:'live', paint: {{
      'circle-radius':       ['case',['get','ww'],8,6],
      'circle-color':        ['get','color'],
      'circle-stroke-width': ['case',['<',['get','cert'],0.25],1,1.5],
      'circle-stroke-color': ['case',['<',['get','cert'],0.25],'#6b7280','#fff'],
      'circle-opacity':      ['case',['<',['get','cert'],0.25],0.65,0.95] }} }});
  }}

  // ── POPUPS ────────────────────────────────────────────────
  const popup = new mapboxgl.Popup({{ closeButton:true, closeOnClick:false, maxWidth:'320px' }});
  const riskColor  = r => r>0.7?'#ef4444':r>0.5?'#f59e0b':r>0.3?'#60a5fa':'#94a3b8';
  const crColor    = c => c>0.6?'#ef4444':c>0.3?'#f97316':'#f59e0b';
  const bayesColor = b => b>0.8?'#ef4444':b>0.5?'#f59e0b':'#94a3b8';

  // [F1] FIX: `const a` was missing — browser crashed silently on every alert click
  map.on('click', 'alerts-layer', e => {{
    const a = e.features[0].properties;  // ← [F1] THE FIX

    // [F3] FIX: Mapbox serialises array props as JSON strings — always safe-parse
    let consIds = [];
    try {{ consIds = JSON.parse(a.cons_ids || '[]'); }} catch(_) {{}}

    popup.setLngLat(e.features[0].geometry.coordinates).setHTML(`
      <div class="popup-head" style="color:#ef4444">🚨 Wrong-Way Alert${{a.hotspot?' 🌡️':''}}</div>
      <div class="popup-body">
        ${{a.profile?`<div style="margin-bottom:6px"><span class="badge-pill" style="background:#1e3a8a;color:#93c5fd">📐 ${{a.profile}} · thresh ${{parseFloat(a.thresh||110).toFixed(0)}}°</span></div>`:''}}
        ${{a.slow_ww?`<div style="margin-bottom:6px"><span class="badge-pill" style="background:#451a03;color:#fbbf24">🐢 Slow Wrong-Way</span></div>`:''}}
        ${{a.hotspot?`<div style="margin-bottom:6px"><span class="badge-pill" style="background:#164e63;color:#06b6d4">🌡️ Heatmap Hotspot</span></div>`:''}}
        <div class="popup-row"><span class="popup-key">Vehicle</span><span class="popup-val">${{a.vid}}</span></div>
        <div class="popup-row"><span class="popup-key">Road</span><span class="popup-val">${{a.road}} (${{a.type}})</span></div>
        <div class="popup-row"><span class="popup-key">Risk score</span><span class="popup-val" style="color:${{riskColor(parseFloat(a.risk))}}">${{parseFloat(a.risk).toFixed(3)}}</span></div>
        <div class="popup-row"><span class="popup-key">Bayes P(WW)</span><span class="popup-val" style="color:${{bayesColor(parseFloat(a.bayes||0))}}">${{parseFloat(a.bayes||0).toFixed(3)}}</span></div>
        <div class="popup-row"><span class="popup-key">Bearing Δ</span><span class="popup-val">${{parseFloat(a.delta).toFixed(1)}}°</span></div>
        <div class="popup-row"><span class="popup-key">Speed</span><span class="popup-val">${{parseFloat(a.spd).toFixed(1)}} km/h</span></div>
        <div class="popup-row"><span class="popup-key">Heading source</span><span class="popup-val" style="color:#a78bfa">${{a.hdg_src||'pairwise'}}</span></div>
        ${{parseFloat(a.cr)>0.05?`<div class="popup-row"><span class="popup-key">Collision risk</span><span class="popup-val" style="color:${{crColor(parseFloat(a.cr))}}">${{parseFloat(a.cr).toFixed(3)}} 💥</span></div>`:''}}
        ${{a.ew?`<div class="popup-row"><span class="popup-key">Early warned</span><span class="popup-val" style="color:#f59e0b">⚡ Yes</span></div>`:''}}
        ${{a.ghost_lat!=null?`<div class="popup-row"><span class="popup-key">Ghost pos</span><span class="popup-val" style="color:#94a3b8">👻 ${{parseFloat(a.ghost_lat).toFixed(4)}}, ${{parseFloat(a.ghost_lon).toFixed(4)}}</span></div>`:''}}
        <hr class="popup-divider"/>
        <div class="popup-row"><span class="popup-key">Consensus</span><span class="popup-val">${{a.cons?'✅ '+consIds.join(', '):'❌ none'}}</span></div>
        <div class="popup-row"><span class="popup-key">Time</span><span class="popup-val">T+${{parseFloat(a.ts).toFixed(1)}} s</span></div>
      </div>`).addTo(map);
  }});

  map.on('click', 'live-layer', e => {{
    const p  = e.features[0].properties;
    vcSelect(p.id);
    const st       = STATES[p.id] || {{}};
    const certPct  = ((st.cert||0)*100).toFixed(0) + '%';
    const certColor= (st.cert||0)<0.25?'#ef4444':(st.cert||0)<0.5?'#f59e0b':'#22c55e';
    const failLabel= FAIL_LABELS[st.fail] || st.fail || null;
    popup.setLngLat(e.features[0].geometry.coordinates).setHTML(`
      <div class="popup-head">
        ${{p.ww?'⚠️':'🚗'}} ${{p.id}}
        ${{p.role==='wrong_way_intruder'?'<span class="badge-pill" style="background:rgba(239,68,68,.2);color:#ef4444;margin-left:4px;font-size:9px">WRONG WAY</span>':''}}
      </div>
      <div class="popup-body">
        <div class="popup-row"><span class="popup-key">Role</span><span class="popup-val">${{(p.role||'').replace(/_/g,' ')}}</span></div>
        <div class="popup-row"><span class="popup-key">Speed</span><span class="popup-val">${{parseFloat(p.spd).toFixed(1)}} km/h</span></div>
        <div class="popup-row"><span class="popup-key">Heading</span><span class="popup-val">${{parseFloat(p.hdg).toFixed(1)}}°${{st.vv_hdg!=null?' <span style="color:#a78bfa;font-size:10px">(vv:'+st.vv_hdg.toFixed(0)+'°)</span>':''}}</span></div>
        <div class="popup-row"><span class="popup-key">Risk score</span><span class="popup-val" style="color:${{riskColor(st.risk||0)}}">${{(st.risk||0).toFixed(3)}}</span></div>
        <div class="popup-row"><span class="popup-key">Bayes P(WW)</span><span class="popup-val" style="color:${{bayesColor(st.bayes||0)}}">${{(st.bayes||0).toFixed(3)}}</span></div>
        <div class="popup-row"><span class="popup-key">Map certainty</span><span class="popup-val" style="color:${{certColor}}">${{certPct}}</span></div>
        <div class="popup-row"><span class="popup-key">Hdg confidence</span><span class="popup-val">${{((st.hconf||1)*100).toFixed(0)}}%</span></div>
        ${{(st.maneuver||0)>0?`<div class="popup-row"><span class="popup-key">Maneuver hold</span><span class="popup-val" style="color:#f59e0b">${{st.maneuver.toFixed(1)}} s left</span></div>`:''}}
        ${{failLabel?`<hr class="popup-divider"/><div class="popup-row"><span class="popup-key">Failure mode</span><span class="popup-val" style="color:#f59e0b">${{failLabel}}</span></div>`:''}}
        <div class="popup-row"><span class="popup-key">Time</span><span class="popup-val">T+${{parseFloat(p.ts).toFixed(1)}} s</span></div>
      </div>`).addTo(map);
  }});

  map.on('click', 'sup-layer', e => {{
    const a = e.features[0].properties;
    popup.setLngLat(e.features[0].geometry.coordinates).setHTML(`
      <div class="popup-head" style="color:#94a3b8">🔕 Suppressed Alert</div>
      <div class="popup-body">
        <div class="popup-row"><span class="popup-key">Vehicle</span><span class="popup-val">${{a.vid}}</span></div>
        <div class="popup-row"><span class="popup-key">Reason</span><span class="popup-val" style="color:#f59e0b">${{a.sup_why||'unknown'}}</span></div>
        <div class="popup-row"><span class="popup-key">Risk</span><span class="popup-val">${{parseFloat(a.risk).toFixed(3)}}</span></div>
        <div class="popup-row"><span class="popup-key">Time</span><span class="popup-val">T+${{parseFloat(a.ts).toFixed(1)}} s</span></div>
      </div>`).addTo(map);
  }});

  ['alerts-layer','live-layer','sup-layer','ew-layer','ghost-layer'].forEach(id => {{
    map.on('mouseenter', id, () => {{ map.getCanvas().style.cursor = 'pointer'; }});
    map.on('mouseleave', id, () => {{ map.getCanvas().style.cursor = ''; }});
  }});

  mapReady = true;
  renderFrame(currentTs);
}}

map.on('load', addAllLayers);

// ── LAYER TOGGLES ─────────────────────────────────────────
const LAYER_MAP = {{
  traces:    ['traces-layer','traces-ww-dash'],
  live:      ['live-layer'],
  alerts:    ['alerts-layer'],
  danger:    ['danger-layer'],
  collision: ['collision-layer'],
  ew:        ['ew-layer'],
  sup:       ['sup-layer'],
  ghost:     ['ghost-layer','ghost-lines'],
  heatmap:   ['heatmap-layer'],
}};
window.toggleLayer = function(key, btn) {{
  btn.classList.toggle('on');
  const v = btn.classList.contains('on') ? 'visible' : 'none';
  (LAYER_MAP[key]||[]).forEach(id => {{ if (map.getLayer(id)) map.setLayoutProperty(id,'visibility',v); }});
}};

let isSat = false;
window.toggleSatellite = function(btn) {{
  btn.classList.toggle('on'); isSat = !isSat;
  map.setStyle(isSat ? 'mapbox://styles/mapbox/satellite-streets-v12' : 'mapbox://styles/mapbox/dark-v11');
  map.once('styledata', () => {{ mapReady = false; setTimeout(addAllLayers, 200); }});
}};

window.flyToVehicle = function(vid) {{
  const s = STATES[vid]; if (!s) return;
  map.flyTo({{ center:[s.lon,s.lat], zoom:17.5, speed:0.9, pitch:50 }});
}};

// ── DUAL MODE ─────────────────────────────────────────────
window.vcSelect = function(vid) {{
  vcFocus = vid; vcLastPos = null;
  const vBtn = document.getElementById('btn-vehicle');
  if (vBtn) vBtn.disabled = false;
  document.querySelectorAll('.veh-row').forEach(r => r.classList.remove('selected'));
  const row = document.getElementById('vr-' + vid);
  if (row) {{ row.classList.add('selected'); row.scrollIntoView({{block:'nearest'}}); }}
  vcVehicle();
}};

window.vcVehicle = function() {{
  if (!vcFocus) return;
  vcMode = 'vehicle';
  document.getElementById('btn-global').classList.remove('active');
  document.getElementById('btn-vehicle').classList.add('active');
  document.getElementById('detail-panel').style.display = 'none';
  document.getElementById('vc-panel').style.display = 'flex';
  vcLastPos = null;
  if (mapReady) renderFrame(currentTs);
}};

window.vcGlobal = function() {{
  vcMode = 'global'; vcFocus = null; vcLastPos = null;
  document.getElementById('btn-global').classList.add('active');
  document.getElementById('btn-vehicle').classList.remove('active');
  document.getElementById('btn-vehicle').disabled = true;
  document.getElementById('detail-panel').style.display = '';
  document.getElementById('vc-panel').style.display = 'none';
  document.querySelectorAll('.veh-row').forEach(r => r.classList.remove('selected'));
  if (mapReady) renderFrame(currentTs);
}};

function vcUpdatePanel(ts, p, feats) {{
  const st  = STATES[vcFocus] || {{}};
  const el  = id => document.getElementById(id);
  el('vc-name').textContent  = vcFocus;
  el('vc-role').textContent  = (p.role||'unknown').replace(/_/g,' ');
  el('vc-spd').textContent   = p.spd.toFixed(1) + ' km/h';
  el('vc-hdg').textContent   = p.hdg.toFixed(1) + '°';
  const risk = st.risk || 0;
  const rEl  = el('vc-risk');
  rEl.textContent = risk.toFixed(3);
  rEl.style.color = risk>0.7?'#ef4444':risk>0.5?'#f59e0b':'#94a3b8';
  const bEl = el('vc-bayes');
  bEl.textContent = (st.bayes||0).toFixed(3);
  bEl.style.color = (st.bayes||0)>0.5?'#ef4444':'#94a3b8';
  const sEl = el('vc-status');
  if      (p.ww)       {{ sEl.textContent='⚠️ WRONG WAY'; sEl.style.color='#ef4444'; }}
  else if (risk >= 0.5){{ sEl.textContent='⚡ HIGH RISK';  sEl.style.color='#f59e0b'; }}
  else                  {{ sEl.textContent='✅ Normal';     sEl.style.color='#22c55e'; }}

  const NEAR = 0.004;
  const threats = feats.filter(f => {{
    if (f.properties.id === vcFocus) return false;
    const dx = Math.abs(f.geometry.coordinates[0] - p.lon);
    const dy = Math.abs(f.geometry.coordinates[1] - p.lat);
    const s2 = STATES[f.properties.id] || {{}};
    return dx<NEAR && dy<NEAR && (f.properties.ww || (s2.risk||0)>=0.5);
  }});
  const tEl = el('vc-threats');
  if (!threats.length) {{
    tEl.innerHTML = '<span style="color:var(--muted);font-size:11px">None nearby</span>';
  }} else {{
    tEl.innerHTML = threats.map(f => {{
      const s2  = STATES[f.properties.id] || {{}};
      const col = f.properties.ww ? '#ef4444' : '#f59e0b';
      return `<div class="thr-row">
        <span style="font-family:var(--mono);color:${{col}}">${{f.properties.id}}</span>
        <span style="color:var(--muted)">risk ${{(s2.risk||0).toFixed(2)}}</span>
      </div>`;
    }}).join('');
  }}
}}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Map v3 saved → {output_path}")
    return output_path
