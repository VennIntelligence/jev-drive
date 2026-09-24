"""TFv6 rules x interface factorial and Bench2Drive by hazard family
(todos/2026-09-25-tfv6-rules-interface/README.md).

  build   route XML for the runs: the 209 Bench2Drive routes that do not crash the server (original ids, each
          tagged p5_suppress=0 so b2d_hooks logs the scenario's hazard actors), plus, for the 45 hazard routes of
          P5 v0, the x- world (hazard hidden) and the weather-only null world as P5 pair variants (seed 0).
          With B2D_RESEED_AFTER_BUILD=1 the Bench2Drive route itself is the x+ world of its pair.
"""
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import data_dir, get_logger
from .p5_pairs import B2D_XML, CRASHERS, _variant

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "tfv6-rules-interface"

# P5 v0's hazard families (Light is R layer, decision 35 / P5 v1), 5 routes each, none of them a crasher.
PAIR_FAMILIES = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian",
                 "ParkingCrossingPedestrian", "OppositeVehicleRunningRedLight", "HardBreakRoute", "StaticCutIn",
                 "ParkingCutIn", "HighwayCutIn")

# Bench2Drive's own multi-ability taxonomy, verbatim from tools/ability_benchmark.py (Bench2Drive 0.0.4); a
# scenario can count for several abilities.
B2D_ABILITY = {
    "Overtaking": ["Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
                   "HazardAtSideLaneTwoWays", "HazardAtSideLane", "ParkedObstacleTwoWays", "ParkedObstacle",
                   "VehicleOpensDoorTwoWays"],
    "Merging": ["CrossingBicycleFlow", "EnterActorFlow", "HighwayExit", "InterurbanActorFlow", "HighwayCutIn",
                "InterurbanAdvancedActorFlow", "MergerIntoSlowTrafficV2", "MergerIntoSlowTraffic",
                "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
                "NonSignalizedJunctionLeftTurnEnterFlow", "ParkingExit", "SequentialLaneChange",
                "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow"],
    "Emergency_Brake": ["BlockedIntersection", "DynamicObjectCrossing", "HardBreakRoute",
                        "OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight", "ParkingCutIn",
                        "PedestrianCrossing", "ParkingCrossingPedestrian", "StaticCutIn", "VehicleTurningRoute",
                        "VehicleTurningRoutePedestrian", "ControlLoss"],
    "Give_Way": ["InvadingTurn", "YieldToEmergencyVehicle"],
    "Traffic_Signs": ["BlockedIntersection", "OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight",
                      "PedestrianCrossing", "VehicleTurningRoute", "VehicleTurningRoutePedestrian", "EnterActorFlow",
                      "CrossingBicycleFlow", "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
                      "NonSignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionLeftTurn",
                      "SignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow", "T_Junction",
                      "VanillaNonSignalizedTurn", "VanillaSignalizedTurnEncounterGreenLight",
                      "VanillaSignalizedTurnEncounterRedLight", "VanillaNonSignalizedTurnEncounterStopsign"],
}

# Our partition (pre-registered): every one of the 44 scenario types in exactly one hazard family. The first
# five are the "sudden hazard" (E layer) families; the rest need planning, negotiation or routine control.
FAMILY = {
    "vru_emerging": ["DynamicObjectCrossing", "ParkingCrossingPedestrian"],          # occluded VRU steps out
    "vru_crossing": ["PedestrianCrossing", "VehicleTurningRoutePedestrian", "VehicleTurningRoute",
                     "CrossingBicycleFlow"],                                          # VRUs crossing at junctions
    "cut_in": ["StaticCutIn", "ParkingCutIn", "HighwayCutIn"],
    "lead_hard_brake": ["HardBreakRoute"],
    "junction_violator": ["OppositeVehicleRunningRedLight", "OppositeVehicleTakingPriority",
                          "BlockedIntersection"],                                     # someone else breaks priority
    "unprotected_turn": ["NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
                         "NonSignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionLeftTurn",
                         "SignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionRightTurn"],
    "merge_lane_change": ["EnterActorFlow", "InterurbanActorFlow", "InterurbanAdvancedActorFlow", "HighwayExit",
                          "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2", "SequentialLaneChange",
                          "ParkingExit"],
    "obstacle_bypass": ["Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
                        "HazardAtSideLane", "HazardAtSideLaneTwoWays", "ParkedObstacle", "ParkedObstacleTwoWays",
                        "VehicleOpensDoorTwoWays", "InvadingTurn"],
    "emergency_vehicle": ["YieldToEmergencyVehicle"],
    "routine_control": ["T_Junction", "VanillaNonSignalizedTurn", "VanillaNonSignalizedTurnEncounterStopsign",
                        "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
                        "ControlLoss"],
}
SUDDEN = ("vru_emerging", "vru_crossing", "cut_in", "lead_hard_brake", "junction_violator")
SCENARIO_FAMILY = {s: f for f, ss in FAMILY.items() for s in ss}


def route_table(src: Path | None = None):
    """route id -> (town, scenario type, family) for all 220 Bench2Drive routes."""
    import pandas as pd
    root = ET.parse(src or data_dir() / B2D_XML).getroot()
    rows = []
    for r in root.findall("route"):
        (s,) = r.iter("scenario")
        t = s.get("type")
        rows.append({"route_id": r.get("id"), "town": r.get("town"), "scenario": t, "family": SCENARIO_FAMILY[t],
                     "sudden": SCENARIO_FAMILY[t] in SUDDEN, "crasher": r.get("id") in CRASHERS,
                     "pair": t in PAIR_FAMILIES})
    return pd.DataFrame(rows)


def build(src: Path | None = None) -> Path:
    root = ET.parse(src or data_dir() / B2D_XML).getroot()
    out = ET.Element("routes")
    n_route = n_pair = 0
    for r in root.findall("route"):
        if r.get("id") in CRASHERS:
            continue
        (s,) = r.iter("scenario")
        b = copy.deepcopy(r)
        b.set("p5_suppress", "0")                 # log hazard actors (hidden.json); hides nothing
        out.append(b)
        n_route += 1
        if s.get("type") in PAIR_FAMILIES:
            out.append(_variant(r, "minus", 0))
            out.append(_variant(r, "null", 0))
            n_pair += 1
    xml = data_dir() / "runs" / "tfv6_rules" / "routes.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(out)
    ET.ElementTree(out).write(xml)
    RESULTS.mkdir(parents=True, exist_ok=True)
    route_table(src).to_csv(RESULTS / "routes.csv", index=False)
    log.info("%d Bench2Drive routes + %d pairs (x-, null) -> %s", n_route, n_pair, xml)
    return xml


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    a = p.parse_args()
    if a.cmd == "build":
        build()


if __name__ == "__main__":
    main()
