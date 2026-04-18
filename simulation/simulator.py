"""
Multi-Vehicle GPS Trace Simulator  — v3

BUG FIX vs v2:
  - build_extended_demo_scenario never added a TURNING_VEHICLE; the role
    existed in the enum and visualiser but was never instantiated.
    Fixed: add_turning_vehicle() added and called in both scenarios.

NEW in v3:
  - add_turning_vehicle() helper so a vehicle that makes a legal U-turn
    at a junction is included as a true negative (should NOT be flagged).
  - VehicleRole.TURNING_VEHICLE properly simulated: drives correct
    direction, then at a configurable time reverses for a short window
    mimicking a three-point turn, then resumes normal direction.
"""

import math
import random
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

from core.config import SimulatorConfig, DEFAULT_SIMULATOR_CFG

logger = logging.getLogger(__name__)


class VehicleRole(Enum):
    NORMAL             = "normal"
    WRONG_WAY_INTRUDER = "wrong_way_intruder"
    DIVERSION_VEHICLE  = "diversion_vehicle"
    TURNING_VEHICLE    = "turning_vehicle"


@dataclass
class GPSPing:
    """A single GPS position report from a vehicle."""
    vehicle_id: str
    lat: float
    lon: float
    timestamp: float
    heading: Optional[float] = None
    speed_kmh: Optional[float] = None
    role: VehicleRole = VehicleRole.NORMAL
    is_truly_wrong_way: bool = False
    is_multipath: bool = False
    is_timestamp_jittered: bool = False


@dataclass
class SimVehicle:
    """Simulated vehicle with trajectory parameters."""
    vehicle_id: str
    role: VehicleRole
    start_lat: float
    start_lon: float
    direction_bearing: float
    speed_kmh: float
    start_time: float = 0.0
    gps_noise_m: float = 3.0
    heading_noise_deg: float = 5.0
    inject_wrong_way_at_t: Optional[float] = None
    wrong_way_duration_s: float = 30.0
    # Turning vehicle: reverses briefly then resumes
    turn_at_t: Optional[float] = None
    turn_duration_s: float = 6.0


class RoadNetwork:
    """Simplified road network for simulation geometry."""

    def __init__(self, center_lat: float, center_lon: float):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.lat_per_m  = 1.0 / 111_320.0
        self.lon_per_m  = 1.0 / (111_320.0 * math.cos(math.radians(center_lat)))

    def offset(self, lat, lon, north_m, east_m):
        return (lat + north_m * self.lat_per_m,
                lon + east_m * self.lon_per_m)

    def move_along_bearing(self, lat, lon, bearing_deg, dist_m):
        br = math.radians(bearing_deg)
        return self.offset(lat, lon,
                           dist_m * math.cos(br),
                           dist_m * math.sin(br))


class MultiVehicleSimulator:
    """
    Simulates a fleet of vehicles on a road network.
    """

    def __init__(self, center_lat: float, center_lon: float,
                 seed: int = 42,
                 sim_config: SimulatorConfig = DEFAULT_SIMULATOR_CFG):
        random.seed(seed)
        np.random.seed(seed)
        self.network    = RoadNetwork(center_lat, center_lon)
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.cfg        = sim_config
        self._vehicles: List[SimVehicle] = []

    def add_normal_fleet(self, count: int, bearing: float,
                         speed_range: Tuple[float, float] = (40.0, 60.0),
                         road_offset_m: float = 0.0):
        for i in range(count):
            start_offset_m = random.uniform(-300, -50)
            lat, lon = self.network.move_along_bearing(
                self.center_lat, self.center_lon, bearing, start_offset_m)
            perp = (bearing + 90) % 360
            lat, lon = self.network.move_along_bearing(
                lat, lon, perp, random.uniform(-1.5, 1.5))
            self._vehicles.append(SimVehicle(
                vehicle_id=f"normal_{i:03d}",
                role=VehicleRole.NORMAL,
                start_lat=lat, start_lon=lon,
                direction_bearing=bearing + random.uniform(-3, 3),
                speed_kmh=random.uniform(*speed_range),
                gps_noise_m=random.uniform(2.0, 5.0),
            ))

    def add_wrong_way_intruder(self, vehicle_id: str,
                                base_bearing: float,
                                start_offset_m: float = 0.0,
                                speed_kmh: float = 45.0,
                                inject_at_t: float = 10.0,
                                duration_s: float = 30.0):
        lat, lon = self.network.move_along_bearing(
            self.center_lat, self.center_lon, base_bearing, start_offset_m)
        self._vehicles.append(SimVehicle(
            vehicle_id=vehicle_id,
            role=VehicleRole.WRONG_WAY_INTRUDER,
            start_lat=lat, start_lon=lon,
            direction_bearing=base_bearing,
            speed_kmh=speed_kmh,
            inject_wrong_way_at_t=inject_at_t,
            wrong_way_duration_s=duration_s,
            gps_noise_m=3.5,
        ))

    def add_diversion_vehicle(self, vehicle_id: str, near_bearing: float):
        lat, lon = self.network.move_along_bearing(
            self.center_lat, self.center_lon,
            near_bearing, random.uniform(-50, 50))
        self._vehicles.append(SimVehicle(
            vehicle_id=vehicle_id,
            role=VehicleRole.DIVERSION_VEHICLE,
            start_lat=lat, start_lon=lon,
            direction_bearing=(near_bearing + 90) % 360,
            speed_kmh=15.0,
            gps_noise_m=4.0,
        ))

    def add_turning_vehicle(self, vehicle_id: str, bearing: float,
                             start_offset_m: float = -100.0,
                             speed_kmh: float = 20.0,
                             turn_at_t: float = 15.0,
                             turn_duration_s: float = 6.0):
        """
        BUG FIX: This vehicle was never instantiated in previous versions.
        A turning vehicle drives normally, briefly reverses during a
        3-point turn (should trigger maneuver suppression, NOT a WW alert),
        then resumes normal direction.
        """
        lat, lon = self.network.move_along_bearing(
            self.center_lat, self.center_lon, bearing, start_offset_m)
        self._vehicles.append(SimVehicle(
            vehicle_id=vehicle_id,
            role=VehicleRole.TURNING_VEHICLE,
            start_lat=lat, start_lon=lon,
            direction_bearing=bearing,
            speed_kmh=speed_kmh,
            turn_at_t=turn_at_t,
            turn_duration_s=turn_duration_s,
            gps_noise_m=4.0,
        ))

    def generate_traces(self, duration_s: float = 60.0,
                         ping_interval_s: float = 1.0) -> List[GPSPing]:
        all_pings: List[GPSPing] = []
        t = 0.0
        while t <= duration_s:
            for v in self._vehicles:
                ping = self._generate_ping(v, t)
                if ping is not None:
                    ping = self._apply_artifacts(ping)
                    if ping is not None:
                        all_pings.append(ping)
            t += ping_interval_s

        all_pings.sort(key=lambda p: (p.timestamp, p.vehicle_id))
        logger.info(
            f"Generated {len(all_pings)} pings for {len(self._vehicles)} "
            f"vehicles over {duration_s}s"
        )
        return all_pings

    def _generate_ping(self, v: SimVehicle, t: float) -> Optional[GPSPing]:
        is_wrong_way = False
        effective_bearing = v.direction_bearing
        effective_speed = v.speed_kmh

        # Wrong-way intruder
        if (v.role == VehicleRole.WRONG_WAY_INTRUDER
                and v.inject_wrong_way_at_t is not None):
            if v.inject_wrong_way_at_t <= t <= (v.inject_wrong_way_at_t
                                                  + v.wrong_way_duration_s):
                effective_bearing = (v.direction_bearing + 180) % 360
                is_wrong_way = True

        # Turning vehicle: slow reverse during turn window (not wrong-way)
        if v.role == VehicleRole.TURNING_VEHICLE and v.turn_at_t is not None:
            turn_end = v.turn_at_t + v.turn_duration_s
            if v.turn_at_t <= t <= turn_end:
                # Briefly moves in reverse at low speed — maneuver suppression
                effective_bearing = (v.direction_bearing + 170) % 360
                effective_speed = 8.0

        dist_m = (v.speed_kmh / 3.6) * t
        lat, lon = self.network.move_along_bearing(
            v.start_lat, v.start_lon, effective_bearing, dist_m)

        lat += np.random.normal(0, v.gps_noise_m * self.network.lat_per_m)
        lon += np.random.normal(0, v.gps_noise_m * self.network.lon_per_m)

        heading = (effective_bearing
                   + np.random.normal(0, v.heading_noise_deg)) % 360

        return GPSPing(
            vehicle_id=v.vehicle_id,
            lat=lat, lon=lon,
            timestamp=t,
            heading=heading,
            speed_kmh=effective_speed + np.random.normal(0, 2.0),
            role=v.role,
            is_truly_wrong_way=is_wrong_way,
        )

    def _apply_artifacts(self, ping: GPSPing) -> Optional[GPSPing]:
        if random.random() < self.cfg.dropout_prob:
            return None

        if random.random() < self.cfg.multipath_prob:
            jump_m = self.cfg.multipath_jump_m * random.uniform(0.5, 2.0)
            direction = random.uniform(0, 360)
            br = math.radians(direction)
            lat_per_m = 1.0 / 111_320.0
            lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(ping.lat)))
            ping.lat += jump_m * math.cos(br) * lat_per_m
            ping.lon += jump_m * math.sin(br) * lon_per_m
            ping.is_multipath = True

        if self.cfg.timestamp_jitter_s > 0:
            jitter = np.random.normal(0, self.cfg.timestamp_jitter_s)
            ping.timestamp += jitter
            if abs(jitter) > self.cfg.timestamp_jitter_s * 0.5:
                ping.is_timestamp_jittered = True

        return ping

    def get_ground_truth(self, pings: List[GPSPing]) -> Dict[str, List[bool]]:
        truth: Dict[str, List[bool]] = {}
        for p in pings:
            truth.setdefault(p.vehicle_id, []).append(p.is_truly_wrong_way)
        return truth


def build_harman_demo_scenario(
    center_lat: float = 12.9716,
    center_lon: float = 77.5946,
    road_bearing: float = 90.0,
    sim_config: SimulatorConfig = DEFAULT_SIMULATOR_CFG,
) -> Tuple[MultiVehicleSimulator, dict]:
    sim = MultiVehicleSimulator(center_lat, center_lon, sim_config=sim_config)
    RB = road_bearing

    sim.add_normal_fleet(count=8, bearing=RB, speed_range=(35.0, 55.0))
    sim.add_wrong_way_intruder("INTRUDER_001", RB,
                                start_offset_m=200, speed_kmh=42.0,
                                inject_at_t=8.0,  duration_s=35.0)
    sim.add_wrong_way_intruder("INTRUDER_002", RB,
                                start_offset_m=350, speed_kmh=68.0,
                                inject_at_t=25.0, duration_s=20.0)
    sim.add_wrong_way_intruder("UTURN_003", RB,
                                start_offset_m=120, speed_kmh=25.0,
                                inject_at_t=15.0, duration_s=25.0)
    sim.add_diversion_vehicle("DIVERSION_001", RB)
    # BUG FIX: turning vehicle was never added
    sim.add_turning_vehicle("TURNING_001", RB,
                             start_offset_m=-80, speed_kmh=18.0,
                             turn_at_t=20.0, turn_duration_s=7.0)

    return sim, {
        "center_lat": center_lat, "center_lon": center_lon,
        "road_bearing": RB,
        "wrong_way_vehicles": ["INTRUDER_001", "INTRUDER_002", "UTURN_003"],
        "diversion_vehicles": ["DIVERSION_001"],
        "turning_vehicles":   ["TURNING_001"],
        "scenario": "Bangalore urban one-way road — dual intruder + U-turn + diversion + turning",
    }


def build_extended_demo_scenario(
    center_lat: float = 12.9716,
    center_lon: float = 77.5946,
    road_bearing: float = 90.0,
    sim_config: SimulatorConfig = DEFAULT_SIMULATOR_CFG,
) -> Tuple[MultiVehicleSimulator, dict]:
    sim, metadata = build_harman_demo_scenario(
        center_lat, center_lon, road_bearing, sim_config)

    sim.add_wrong_way_intruder(
        "SLOW_WW_004", road_bearing,
        start_offset_m=80,  speed_kmh=8.0,
        inject_at_t=20.0,   duration_s=30.0)

    sim.add_wrong_way_intruder(
        "SPARSE_005", road_bearing,
        start_offset_m=250, speed_kmh=40.0,
        inject_at_t=12.0,   duration_s=25.0)

    metadata["wrong_way_vehicles"] += ["SLOW_WW_004", "SPARSE_005"]
    metadata["scenario"] = (
        "Bangalore urban — dual intruder + U-turn + diversion "
        "+ slow wrong-way + sparse-GPS wrong-way + turning vehicle"
    )
    return sim, metadata
