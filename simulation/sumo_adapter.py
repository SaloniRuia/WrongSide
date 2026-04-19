"""
SUMO TraCI Adapter — Wrong-Way Detection System v3
Windows-compatible drop-in replacement for build_extended_demo_scenario().

HOW IT FITS:
  - Produces the same (simulator_obj, metadata) tuple that main.py expects
  - Your detector.py, evaluator.py, map_builder.py are UNTOUCHED
  - Only ONE line changes in main.py (see bottom of this file)

WINDOWS SETUP:
  1. Download SUMO installer from https://sumo.dlr.de/docs/Downloads.php
  2. Install to default path  (C:\\Program Files (x86)\\Eclipse\\Sumo)
  3. pip install traci sumolib
  4. Run this file once to auto-download the Chennai road network:
       python simulation/sumo_adapter.py --download-map

USAGE (from main.py):
  from simulation.sumo_adapter import build_sumo_scenario
  sim, metadata = build_sumo_scenario(duration_s=duration_s)
"""

import os
import sys
import math
import time
import random
import logging
import tempfile
import subprocess
import platform
from typing import List, Tuple, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Windows SUMO binary detection ─────────────────────────

def _find_sumo_binary(gui: bool = False) -> str:
    """
    Finds sumo or sumo-gui on Windows, Mac, or Linux.
    Checks the system PATH first, then common Windows install locations.
    """
    binary_name = "sumo-gui" if gui else "sumo"

    # 1. Try PATH first (works if user added SUMO to environment variables)
    import shutil
    found = shutil.which(binary_name)
    if found:
        return found

    # 2. Windows default install locations
    if platform.system() == "Windows":
        win_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo\bin",
            r"C:\Program Files\Eclipse\Sumo\bin",
            r"C:\Sumo\bin",
            r"C:\sumo\bin",
        ]
        for folder in win_paths:
            candidate = os.path.join(folder, binary_name + ".exe")
            if os.path.isfile(candidate):
                logger.info(f"Found SUMO at: {candidate}")
                return candidate

    raise FileNotFoundError(
        f"\n\nCould not find '{binary_name}' on your system.\n"
        f"Windows fix:\n"
        f"  1. Download from https://sumo.dlr.de/docs/Downloads.php\n"
        f"  2. Install it (default path is fine)\n"
        f"  3. Add  C:\\Program Files (x86)\\Eclipse\\Sumo\\bin  to your PATH\n"
        f"     (Search 'Environment Variables' in Start → System Variables → Path → Edit → New)\n"
        f"  4. Restart your terminal and try again."
    )


def _find_netconvert() -> str:
    """Finds netconvert (ships with SUMO) — needed to build the road network."""
    import shutil
    found = shutil.which("netconvert")
    if found:
        return found

    if platform.system() == "Windows":
        win_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo\bin\netconvert.exe",
            r"C:\Program Files\Eclipse\Sumo\bin\netconvert.exe",
            r"C:\Sumo\bin\netconvert.exe",
        ]
        for p in win_paths:
            if os.path.isfile(p):
                return p

    raise FileNotFoundError(
        "netconvert not found. It ships with SUMO — check your SUMO installation."
    )


# ── Road network download ──────────────────────────────────

# Chennai area — Anna Salai stretch (good one-way road mix)
CHENNAI_BBOX = {
    "name":    "Chennai_AnnaSalai",
    "min_lon": 80.2400,
    "min_lat": 13.0300,
    "max_lon": 80.2700,
    "max_lat": 13.0600,
}

# Bangalore fallback (matches your existing default lat/lon)
BANGALORE_BBOX = {
    "name":    "Bangalore_MG_Road",
    "min_lon": 77.5900,
    "min_lat": 12.9700,
    "max_lon": 77.6100,
    "max_lat": 12.9800,
}


def download_road_network(
    bbox: dict = None,
    output_dir: str = None,
    force: bool = False,
) -> str:
    """
    Downloads OSM data and converts it to a SUMO .net.xml file.
    Returns the path to the .net.xml file.

    Uses the same Overpass API your osm_resolver.py already calls —
    no new dependencies needed for the download step.
    """
    if bbox is None:
        bbox = CHENNAI_BBOX  # Chennai Anna Salai is the default

    if output_dir is None:
        # Store next to this file so it persists between runs
        output_dir = os.path.join(os.path.dirname(__file__), "sumo_maps")

    os.makedirs(output_dir, exist_ok=True)

    net_file = os.path.join(output_dir, f"{bbox['name']}.net.xml")
    osm_file = os.path.join(output_dir, f"{bbox['name']}.osm")

    if os.path.isfile(net_file) and not force:
        logger.info(f"Road network already exists: {net_file}  (use force=True to re-download)")
        return net_file

    # Step 1 — Download OSM data via Overpass
    logger.info(f"Downloading OSM data for {bbox['name']}...")
    _download_osm_bbox(bbox, osm_file)

    # Step 2 — Convert OSM → SUMO network
    logger.info("Converting OSM → SUMO network (netconvert)...")
    netconvert = _find_netconvert()
    cmd = [
        netconvert,
        "--osm-files",          osm_file,
        "--output-file",        net_file,
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--no-warnings",
        "--log",                os.path.join(output_dir, "netconvert.log"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"netconvert failed:\n{result.stderr}\n"
            f"Check {output_dir}/netconvert.log for details."
        )

    logger.info(f"Road network ready: {net_file}")
    return net_file


def _download_osm_bbox(bbox: dict, output_path: str):
    """
    Downloads raw OSM XML via Overpass API using POST + User-Agent header.
    GET requests without a User-Agent return HTTP 406 Not Acceptable.
    """
    import urllib.request
    import urllib.parse

    query = (
        f"[out:xml][timeout:60];"
        f"(way[highway]({bbox['min_lat']},{bbox['min_lon']},"
        f"{bbox['max_lat']},{bbox['max_lon']});"
        f">;);"
        f"out body;"
    )
    url = "https://overpass-api.de/api/interpreter"

    # POST with a real User-Agent — Overpass rejects Python urllib's default string with 406
    post_data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=post_data,
        headers={
            "User-Agent": "WrongWayDetectionSystem/3.0 (research project)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    logger.info(f"  Fetching from Overpass API: {url}")
    logger.info(f"  Area: {bbox['name']}")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
        with open(output_path, "wb") as f:
            f.write(data)
        size_kb = len(data) // 1024
        logger.info(f"  Downloaded {size_kb} KB → {output_path}")
    except Exception as exc:
        raise ConnectionError(
            f"Overpass API download failed: {exc}\n"
            f"Check your internet connection and try again."
        )


# ── Temp config writer (Windows-safe path) ────────────────

def _write_gui_settings(tmp_dir: str) -> str:
    """
    Writes a sumo-gui view-settings file that enables:
      - OSM tile background (real satellite/map tiles)
      - Coloured vehicles with labels
      - Wrong-way vehicles highlighted in red
    """
    settings_content = """<viewsettings>
    <scheme name="real_roads">
        <background backgroundColor="255,255,255"/>
        <vehicles>
            <vehicleColorer scheme="by speed"/>
            <vehicle minSize="5" exaggeration="2" showBlinker="true" showMinGap="true"/>
            <vehicleName show="true" size="50"/>
        </vehicles>
        <junctions>
            <junction show="true"/>
        </junctions>
        <edges>
            <edge showStreetName="true"/>
        </edges>
    </scheme>
    <delay value="100"/>
    <viewport zoom="1000"/>
    <!-- OSM tile background — requires internet on first open -->
    <decals>
        <decal filename="https://tile.openstreetmap.org"
               centerX="0" centerY="0"
               width="1" height="1"
               tileZoom="17"
               layer="-1"/>
    </decals>
</viewsettings>"""
    path = os.path.join(tmp_dir, "wwd_gui_settings.xml")
    with open(path, "w") as f:
        f.write(settings_content)
    return path


def _write_sumo_config(net_file: str, duration_s: float,
                        route_file: str = None,
                        gui_mode: bool = False) -> str:
    """
    Writes a minimal SUMO .sumocfg file.
    Uses tempfile.gettempdir() so it works on Windows, Mac, and Linux.
    When gui_mode=True, injects view-settings with OSM tile background.
    """
    tmp_dir = tempfile.gettempdir()
    route_line = (
        f'  <route-files value="{route_file}"/>'
        if route_file else ""
    )
    gui_line = ""
    if gui_mode:
        gui_cfg = _write_gui_settings(tmp_dir)
        gui_line = f'    <gui-settings-file value="{gui_cfg}"/>'

    cfg_content = f"""<configuration>
  <input>
    <net-file value="{net_file}"/>
    {route_line}
    {gui_line}
  </input>
  <time>
    <begin value="0"/>
    <end value="{int(duration_s)}"/>
  </time>
  <processing>
    <ignore-route-errors value="true"/>
    <collision.action value="none"/>
  </processing>
  <gui_only>
    <start value="true"/>
    <tracker-interval value="0.5"/>
  </gui_only>
</configuration>"""

    cfg_path = os.path.join(tmp_dir, "wwd_sumo.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(cfg_content)
    return cfg_path


# ── Bearing helper ─────────────────────────────────────────

def _bearing_from_xy(x1, y1, x2, y2) -> float:
    """SUMO uses Cartesian metres. Convert movement vector to compass bearing."""
    dx, dy = x2 - x1, y2 - y1
    angle = math.degrees(math.atan2(dx, dy))
    return (angle + 360) % 360


def _add_gps_noise(coord_deg: float, noise_m: float) -> float:
    noise_deg = noise_m / 111_320.0
    return coord_deg + random.gauss(0, noise_deg)


# ── Fake simulator wrapper ─────────────────────────────────
# main.py calls  sim.generate_traces(duration_s, ping_interval_s)
# We pre-collect all pings during the SUMO run, then return them
# from generate_traces() so the interface is identical.

class SumoSimulatorWrapper:
    """
    Drop-in replacement for MultiVehicleSimulator.
    Stores pre-collected pings and returns them on generate_traces().
    """

    def __init__(self, pings, wrong_way_ids: List[str]):
        self._pings = pings
        self._wrong_way_ids = set(wrong_way_ids)

    def generate_traces(self, duration_s: float = 60.0,
                         ping_interval_s: float = 1.0):
        logger.info(f"[SUMO] Returning {len(self._pings)} pre-collected pings")
        return self._pings

    def get_ground_truth(self, pings):
        truth = {}
        for p in pings:
            truth.setdefault(p.vehicle_id, []).append(p.is_truly_wrong_way)
        return truth


# ── Main scenario builder ──────────────────────────────────

def build_sumo_scenario(
    net_file: str = None,
    duration_s: float = 60.0,
    normal_vehicle_count: int = 6,
    wrong_way_inject_at_s: float = 10.0,
    gui_mode: bool = False,
    gps_noise_m: float = 3.0,
) -> Tuple["SumoSimulatorWrapper", Dict]:
    """
    Runs a SUMO simulation on a real OSM road network.
    Returns (SumoSimulatorWrapper, metadata) — same shape as
    build_extended_demo_scenario() so main.py needs only ONE line changed.

    Parameters
    ----------
    net_file   : path to .net.xml — auto-downloaded if None
    duration_s : simulation length in seconds
    gui_mode   : True opens sumo-gui (great for demos/debug)
    gps_noise_m: metres of Gaussian GPS noise applied to each ping
    """

    # ── Lazy imports so missing traci doesn't crash the whole project ──
    try:
        import traci
        import sumolib
    except ImportError:
        raise ImportError(
            "traci / sumolib not installed.\n"
            "Run:  pip install traci sumolib"
        )

    # ── Download map if needed ──────────────────────────────
    if net_file is None:
        net_file = download_road_network()

    if not os.path.isfile(net_file):
        raise FileNotFoundError(
            f"SUMO network file not found: {net_file}\n"
            f"Run once with:  python simulation/sumo_adapter.py --download-map"
        )

    logger.info(f"[SUMO] Loading network: {net_file}")
    net = sumolib.net.readNet(net_file, withInternal=False)

    # ── Write config ────────────────────────────────────────
    cfg_path = _write_sumo_config(net_file, duration_s, gui_mode=gui_mode)

    # ── Start SUMO ──────────────────────────────────────────
    sumo_binary = _find_sumo_binary(gui=gui_mode)
    logger.info(f"[SUMO] Starting {'sumo-gui' if gui_mode else 'sumo'}: {sumo_binary}")

    sumo_cmd = [
        sumo_binary,
        "-c", cfg_path,
        "--no-step-log",
        "--no-warnings",
        "--step-length", "1.0",
    ]

    traci.start(sumo_cmd)
    logger.info("[SUMO] TraCI connected ✅")

    # ── Find a good one-way edge for wrong-way injection ────
    target_edge, road_bearing = _find_best_oneway_edge(net)
    logger.info(f"[SUMO] Target edge: {target_edge}  bearing={road_bearing:.1f}°")

    # ── Derive map centre from edge midpoint ────────────────
    edge_obj  = net.getEdge(target_edge)
    mid_x     = (edge_obj.getFromNode().getCoord()[0] +
                 edge_obj.getToNode().getCoord()[0]) / 2
    mid_y     = (edge_obj.getFromNode().getCoord()[1] +
                 edge_obj.getToNode().getCoord()[1]) / 2
    center_lon, center_lat = net.convertXY2LonLat(mid_x, mid_y)

    # ── Spawn vehicles ──────────────────────────────────────
    normal_ids    = _spawn_normal_fleet(net, target_edge, normal_vehicle_count)
    ww_vehicle_id = "SUMO_INTRUDER_001"
    ww_injected   = False

    # ── Simulation loop ─────────────────────────────────────
    from simulation.simulator import GPSPing, VehicleRole

    all_pings: List[GPSPing] = []

    logger.info(f"[SUMO] Running simulation for {duration_s}s...")
    while traci.simulation.getTime() < duration_s:
        traci.simulationStep()
        t = float(traci.simulation.getTime())

        # Inject wrong-way vehicle once
        if not ww_injected and t >= wrong_way_inject_at_s:
            _inject_wrong_way_vehicle(ww_vehicle_id, target_edge, net, traci)
            ww_injected = True
            logger.info(f"[SUMO] Wrong-way vehicle injected at t={t:.0f}s")

        active_ids = traci.vehicle.getIDList()

        for vid in active_ids:
            try:
                x, y     = traci.vehicle.getPosition(vid)
                lon, lat  = net.convertXY2LonLat(x, y)
                speed_ms  = traci.vehicle.getSpeed(vid)
                sumo_angle = traci.vehicle.getAngle(vid)  # 0=North, clockwise

                is_ww = (vid == ww_vehicle_id and t >= wrong_way_inject_at_s)

                ping = GPSPing(
                    vehicle_id        = vid,
                    lat               = _add_gps_noise(lat, gps_noise_m),
                    lon               = _add_gps_noise(lon, gps_noise_m),
                    timestamp         = t,
                    heading           = (sumo_angle + random.gauss(0, 5.0)) % 360,
                    speed_kmh         = speed_ms * 3.6 + random.gauss(0, 1.5),
                    role              = (VehicleRole.WRONG_WAY_INTRUDER
                                        if is_ww else VehicleRole.NORMAL),
                    is_truly_wrong_way = is_ww,
                )
                all_pings.append(ping)

            except traci.TraCIException:
                # Vehicle may have left the network — skip quietly
                continue

    traci.close()
    logger.info(f"[SUMO] Simulation done. {len(all_pings)} pings collected.")

    # ── Sort pings by time (same order as your simulator) ───
    all_pings.sort(key=lambda p: (p.timestamp, p.vehicle_id))

    metadata = {
        "center_lat":         center_lat,
        "center_lon":         center_lon,
        "road_bearing":       road_bearing,
        "wrong_way_vehicles": [ww_vehicle_id],
        "diversion_vehicles": [],
        "turning_vehicles":   [],
        "normal_vehicle_ids": normal_ids,
        "scenario":           f"SUMO real-road simulation — {os.path.basename(net_file)}",
        "net_file":           net_file,
        "sumo_pings":         len(all_pings),
    }

    return SumoSimulatorWrapper(all_pings, [ww_vehicle_id]), metadata


# ── SUMO helpers ───────────────────────────────────────────

def _find_best_oneway_edge(net) -> Tuple[str, float]:
    """
    Picks the best one-way edge for wrong-way injection:
    - Prefers primary / secondary highway types
    - Long enough for vehicles to travel on (> 100 m)
    - Not an internal junction edge
    """
    PREFERRED = {"primary", "secondary", "trunk", "motorway",
                 "motorway_link", "primary_link"}

    candidates = []
    for edge in net.getEdges():
        if edge.getFunction() == "internal":
            continue
        length = edge.getLength()
        if length < 100:
            continue
        edge_type = edge.getType() or ""
        is_preferred = any(p in edge_type for p in PREFERRED)
        candidates.append((edge, is_preferred, length))

    if not candidates:
        raise RuntimeError(
            "No suitable edges found in the SUMO network. "
            "Try downloading a different area."
        )

    # Prefer longer primary roads
    candidates.sort(key=lambda x: (not x[1], -x[2]))
    best_edge = candidates[0][0]

    fx, fy = best_edge.getFromNode().getCoord()
    tx, ty = best_edge.getToNode().getCoord()
    bearing = _bearing_from_xy(fx, fy, tx, ty)

    return best_edge.getID(), bearing


def _spawn_normal_fleet(net, target_edge_id: str,
                         count: int) -> List[str]:
    """Add normal vehicles driving in the correct direction."""
    import traci

    all_edges = [e for e in net.getEdges()
                 if e.getFunction() != "internal" and e.getLength() > 50]

    spawned = []
    for i in range(count):
        vid   = f"normal_{i:03d}"
        edge  = all_edges[i % len(all_edges)]
        route = f"route_normal_{i}"
        try:
            traci.route.add(route, [edge.getID()])
            traci.vehicle.add(
                vid, route,
                depart=str(i * 3),       # stagger departures
                departSpeed="max",
            )
            spawned.append(vid)
        except traci.TraCIException as e:
            logger.debug(f"Could not spawn {vid}: {e}")

    logger.info(f"[SUMO] Spawned {len(spawned)} normal vehicles")
    return spawned


def _inject_wrong_way_vehicle(vid: str, edge_id: str, net, traci):
    """
    Injects a vehicle at the END of a one-way edge facing backward —
    so it drives head-on into correct-direction traffic.

    FIX: Use keepRoute=0 (snap to nearest lane) instead of keepRoute=2
    (off-network teleport). keepRoute=2 causes SUMO to silently drop the
    vehicle if the XY coordinate doesn't land exactly on a lane.
    """
    edge_obj = net.getEdge(edge_id)
    route_id = "route_ww"

    try:
        traci.route.add(route_id, [edge_id])
    except traci.TraCIException:
        pass  # Route may already exist from a previous run

    try:
        traci.vehicle.add(
            vid, route_id,
            depart="now",
            departSpeed="15",          # ~54 km/h
            departPos="last",          # Start at end of edge → drives backward
        )

        # Get the end-node coordinates
        end_x, end_y = edge_obj.getToNode().getCoord()

        # Compute bearing opposite to road direction
        fx, fy = edge_obj.getFromNode().getCoord()
        road_angle = math.degrees(math.atan2(end_x - fx, end_y - fy)) % 360
        wrong_way_angle = (road_angle + 180) % 360

        # keepRoute=0: snap vehicle to the nearest lane on the network.
        # This is far more robust than keepRoute=2 (off-network float).
        traci.vehicle.moveToXY(
            vid, edge_id, 0,
            end_x, end_y,
            angle=wrong_way_angle,
            keepRoute=0,
        )
        traci.vehicle.setSpeed(vid, 15.0)   # 54 km/h wrong-way
        logger.info(f"[SUMO] Injected wrong-way vehicle '{vid}' on edge '{edge_id}' "
                    f"heading={wrong_way_angle:.1f}°")

    except traci.TraCIException as e:
        logger.warning(f"[SUMO] Wrong-way injection error: {e}")


# ── CLI: download map + quick test ────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SUMO adapter for Wrong-Way Detection System"
    )
    parser.add_argument("--download-map", action="store_true",
                        help="Download Chennai road network and exit")
    parser.add_argument("--city",  choices=["bangalore", "chennai"],
                        default="bangalore",
                        help="Which city to download (default: bangalore)")
    parser.add_argument("--gui",   action="store_true",
                        help="Open sumo-gui instead of headless sumo")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Quick test simulation length in seconds")
    args = parser.parse_args()

    bbox = CHENNAI_BBOX if args.city == "chennai" else BANGALORE_BBOX

    if args.download_map:
        net = download_road_network(bbox=bbox, force=False)
        print(f"\nNetwork ready: {net}")
        print("You can now run main.py with --sumo flag.")
        sys.exit(0)

    # Quick smoke test
    print(f"\nRunning quick {args.duration}s test (gui={args.gui})...\n")
    sim, meta = build_sumo_scenario(
        duration_s=args.duration,
        gui_mode=args.gui,
    )
    pings = sim.generate_traces()
    print(f"\nResult: {len(pings)} pings")
    print(f"Center: {meta['center_lat']:.4f}, {meta['center_lon']:.4f}")
    print(f"Road bearing: {meta['road_bearing']:.1f}°")
    print(f"Wrong-way vehicles: {meta['wrong_way_vehicles']}")
    ww_pings = [p for p in pings if p.is_truly_wrong_way]
    print(f"Wrong-way pings: {len(ww_pings)}")
    print("\nSUMO adapter working correctly ✅")
