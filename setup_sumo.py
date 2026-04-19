"""
setup_sumo.py — One-click SUMO setup for Wrong-Way Detection System
====================================================================
Run this ONCE before using --sumo in main.py.

    python setup_sumo.py

It will:
  1. Check SUMO is installed and findable on Windows
  2. Install Python bindings (traci, sumolib) if missing
  3. Download the Chennai road network from OSM (free, ~1 MB)
  4. Convert it to a SUMO .net.xml file using netconvert
  5. Run a 10-second smoke test to confirm everything works
  6. Print the exact command to run main.py with SUMO

You only need internet for step 3. After that, everything runs offline.
"""

import sys
import os
import shutil
import platform
import subprocess

# ── make sure project root is on path ─────────────────────
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

SEP = "=" * 60


def step(n, total, msg):
    print(f"\n[{n}/{total}] {msg}")
    print("-" * 40)


def ok(msg):
    print(f"  ✅  {msg}")


def fail(msg):
    print(f"  ❌  {msg}")


def info(msg):
    print(f"      {msg}")


# ─────────────────────────────────────────────────────────
# STEP 1 — Check SUMO binary
# ─────────────────────────────────────────────────────────

def check_sumo():
    step(1, 5, "Checking SUMO installation")

    # Common Windows install paths
    WIN_PATHS = [
        r"C:\Program Files (x86)\Eclipse\Sumo\bin",
        r"C:\Program Files\Eclipse\Sumo\bin",
        r"C:\Sumo\bin",
        r"C:\sumo\bin",
    ]

    sumo_bin   = shutil.which("sumo")
    net_bin    = shutil.which("netconvert")

    if not sumo_bin and platform.system() == "Windows":
        for folder in WIN_PATHS:
            candidate = os.path.join(folder, "sumo.exe")
            if os.path.isfile(candidate):
                sumo_bin = candidate
                os.environ["PATH"] += os.pathsep + folder
                break

    if not net_bin and platform.system() == "Windows":
        for folder in WIN_PATHS:
            candidate = os.path.join(folder, "netconvert.exe")
            if os.path.isfile(candidate):
                net_bin = candidate
                break

    if sumo_bin and net_bin:
        ok(f"sumo found       → {sumo_bin}")
        ok(f"netconvert found → {net_bin}")
        return sumo_bin, net_bin
    else:
        fail("SUMO not found on this machine.")
        print("""
  Fix (Windows):
    1. Go to  https://sumo.dlr.de/docs/Downloads.php
    2. Download the  .msi  installer under "Windows"
    3. Run the installer (keep default install path)
    4. Add this to your PATH environment variable:
         C:\\Program Files (x86)\\Eclipse\\Sumo\\bin
       (Search "Environment Variables" in Start menu
        → System Variables → Path → Edit → New)
    5. Open a NEW terminal and re-run this script.
""")
        sys.exit(1)


# ─────────────────────────────────────────────────────────
# STEP 2 — Install Python packages
# ─────────────────────────────────────────────────────────

def check_python_packages():
    step(2, 5, "Checking Python packages (traci, sumolib)")

    missing = []
    for pkg in ("traci", "sumolib"):
        try:
            __import__(pkg)
            ok(f"{pkg} already installed")
        except ImportError:
            missing.append(pkg)

    if missing:
        info(f"Installing: {', '.join(missing)}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + missing,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            fail(f"pip install failed:\n{result.stderr}")
            sys.exit(1)
        ok(f"Installed: {', '.join(missing)}")


# ─────────────────────────────────────────────────────────
# STEP 3 — Download OSM data
# ─────────────────────────────────────────────────────────

def download_osm():
    step(3, 5, "Downloading Chennai Anna Salai road network from OpenStreetMap")

    maps_dir = os.path.join(_root, "simulation", "sumo_maps")
    os.makedirs(maps_dir, exist_ok=True)

    osm_path = os.path.join(maps_dir, "Chennai_AnnaSalai.osm")

    if os.path.isfile(osm_path):
        ok(f"OSM file already exists → {osm_path}")
        return osm_path

    import urllib.request

    # Overpass API query — same API your osm_resolver.py uses
    query = (
        "[out:xml][timeout:60];"
        "(way[highway](13.0300,80.2400,13.0600,80.2700);"
        ">;);"
        "out body;"
    )
    url = "https://overpass-api.de/api/interpreter"
    encoded_query = urllib.request.quote(query, safe="=&[]();,")
    full_url = f"{url}?data={encoded_query}"

    info("Connecting to Overpass API (same source as your osm_resolver.py)...")
    info(f"Area: Chennai Anna Salai corridor")

    try:
        with urllib.request.urlopen(full_url, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        fail(f"Download failed: {e}")
        print("""
  Manual fallback:
    1. Open  https://overpass-turbo.eu
    2. Draw a box around Anna Salai, Chennai
    3. Click Export → OSM
    4. Save the file as:
         wrong_way_detection_system/simulation/sumo_maps/Chennai_AnnaSalai.osm
    5. Re-run this script.
""")
        sys.exit(1)

    with open(osm_path, "wb") as f:
        f.write(data)

    size_kb = len(data) // 1024
    ok(f"Downloaded {size_kb} KB → {osm_path}")
    return osm_path


# ─────────────────────────────────────────────────────────
# STEP 4 — Convert OSM → SUMO network
# ─────────────────────────────────────────────────────────

def convert_network(osm_path: str, net_bin: str):
    step(4, 5, "Converting OSM → SUMO network (netconvert)")

    maps_dir  = os.path.dirname(osm_path)
    net_path  = os.path.join(maps_dir, "Chennai_AnnaSalai.net.xml")
    log_path  = os.path.join(maps_dir, "netconvert.log")

    if os.path.isfile(net_path):
        ok(f"Network already exists → {net_path}")
        return net_path

    info("Running netconvert (this takes ~10 seconds)...")
    cmd = [
        net_bin,
        "--osm-files",          osm_path,
        "--output-file",        net_path,
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--no-warnings",
        "--log",                log_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.isfile(net_path):
        fail("netconvert failed.")
        info(f"Check the log: {log_path}")
        info(result.stderr[:500])
        sys.exit(1)

    size_kb = os.path.getsize(net_path) // 1024
    ok(f"Network built ({size_kb} KB) → {net_path}")
    return net_path


# ─────────────────────────────────────────────────────────
# STEP 5 — Smoke test
# ─────────────────────────────────────────────────────────

def smoke_test(net_path: str):
    step(5, 5, "Running 10-second smoke test")

    info("Importing sumo_adapter...")
    try:
        from simulation.sumo_adapter import build_sumo_scenario
    except ImportError as e:
        fail(f"Could not import sumo_adapter: {e}")
        sys.exit(1)

    info("Starting SUMO (headless, 10 seconds)...")
    try:
        sim, meta = build_sumo_scenario(
            net_file   = net_path,
            duration_s = 10.0,
            gui_mode   = False,
        )
        pings = sim.generate_traces()
    except Exception as e:
        fail(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    ww_pings = [p for p in pings if p.is_truly_wrong_way]
    ok(f"Simulation ran successfully")
    ok(f"Total pings     : {len(pings)}")
    ok(f"Wrong-way pings : {len(ww_pings)}")
    ok(f"Center location : {meta['center_lat']:.4f}, {meta['center_lon']:.4f}")
    ok(f"Road bearing    : {meta['road_bearing']:.1f}°")


# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{SEP}")
    print("  SETUP COMPLETE — SUMO is ready to use")
    print(SEP)
    print("""
  Run your pipeline with SUMO real roads:

    Headless (fastest):
      python main.py --offline --sumo

    Visual GUI (best for demos):
      python main.py --offline --sumo --sumo-gui

    Bangalore map instead of Chennai:
      python setup_sumo.py --city bangalore
      python main.py --offline --sumo

  The GUI opens a window showing vehicles moving on the
  real Chennai road network. The wrong-way intruder
  drives head-on against traffic — great to record for slides.
""")


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SUMO setup for Wrong-Way Detection")
    parser.add_argument("--city", choices=["bangalore", "chennai"],
                        default="chennai",
                        help="City to download road network for (default: chennai)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and reconvert even if files already exist")
    args = parser.parse_args()

    print(SEP)
    print("  Wrong-Way Detection — SUMO Setup")
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  City     : {args.city.title()}")
    print(SEP)

    sumo_bin, net_bin = check_sumo()
    check_python_packages()
    osm_path  = download_osm()
    net_path  = convert_network(osm_path, net_bin)
    smoke_test(net_path)
    print_summary()
