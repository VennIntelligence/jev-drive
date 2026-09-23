"""P5 v0: counterfactual scenario pairs in CARLA and an open-loop exam on them
(todos/2026-09-24-p5-carla-pairs-v0.md).

  build     write the variant route XML (x+ as is, x- with the scenario removed (trigger point moved 10 km off the route) or the light swapped, a
            weather-only null) and the case table; each variant is its own numeric route id
            base_id * 100 + world * 10 + tm_seed, world 1 = plus, 2 = minus, 3 = null

The recorder is scripts/p5_pair_agent.py, run through scripts/b2d_run.py in envs/scout-tfv6.
"""
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "p5-carla-pairs"
B2D_XML = "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
RED, GREEN = "VanillaSignalizedTurnEncounterRedLight", "VanillaSignalizedTurnEncounterGreenLight"
FAMILIES = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian",
            "ParkingCrossingPedestrian", "OppositeVehicleRunningRedLight", "HardBreakRoute", "StaticCutIn",
            "ParkingCutIn", "HighwayCutIn", "Light")
WORLDS = {"plus": 1, "minus": 2, "null": 3}
SEEDS = (0, 1, 2)
# docs/carla.md: every one of these crashed the server three times in the 220-route round
CRASHERS = {"3048", "11715", "11755", "23687", "23708", "3785", "3800", "23670", "23695", "24041", "24071"}
NIGHT = dict(cloudiness=30.0, fog_density=2.0, precipitation=0.0, precipitation_deposits=0.0,
             sun_altitude_angle=-60.0, sun_azimuth_angle=-1.0, wetness=0.0, wind_intensity=10.0)
NOON = dict(cloudiness=5.0, fog_density=2.0, precipitation=0.0, precipitation_deposits=0.0,
            sun_altitude_angle=60.0, sun_azimuth_angle=-1.0, wetness=0.0, wind_intensity=10.0)


def runs_dir(*parts) -> Path:
    p = data_dir() / "runs" / "p5_pairs" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def variant_id(base: str, world: str, seed: int) -> str:
    return str(int(base) * 100 + WORLDS[world] * 10 + seed)


def parse_id(rid: str) -> tuple[str, str, int]:
    k = int(rid)
    return str(k // 100), {v: w for w, v in WORLDS.items()}[(k // 10) % 10], k % 10


def _family(stype: str) -> str:
    return "Light" if stype in (RED, GREEN) else stype


def _variant(route: ET.Element, world: str, seed: int) -> ET.Element:
    """One variant of a single-scenario route: the XML is the only thing that differs between worlds."""
    r = copy.deepcopy(route)
    r.set("id", variant_id(route.get("id"), world, seed))
    sc = r.find("scenarios")
    (s,) = sc.findall("scenario")
    stype = s.get("type")
    if stype in (RED, GREEN):                    # plus = red light, minus = green light, whatever the original was
        want = RED if world in ("plus", "null") else GREEN
        s.set("type", want)
        s.set("name", s.get("name").replace(stype, want))
    elif world == "minus":
        # Bench2Drive's evaluator names the route after its first scenario config, so the element has to stay; a
        # trigger point 10 km away makes RouteScenario._filter_scenarios drop it ("too far from the route").
        tp = s.find("trigger_point")
        tp.set("x", str(float(tp.get("x")) + 10000.0))
        s.set("name", s.get("name") + "_removed")
    if world == "null":
        ws = r.find("weathers")
        day = float(ws.find("weather").get("sun_altitude_angle")) > 0
        for w in ws.findall("weather"):
            for k, v in (NIGHT if day else NOON).items():
                w.set(k, str(v))
    return r


def build(src: Path | None = None) -> pd.DataFrame:
    """Case table (one row per base route x seed) and the variant XML with every world of every case."""
    root = ET.parse(src or data_dir() / B2D_XML).getroot()
    out, rows = ET.Element("routes"), []
    for route in root.findall("route"):
        (s,) = route.iter("scenario")
        fam = _family(s.get("type"))
        rid = route.get("id")
        if fam not in FAMILIES or rid in CRASHERS:
            continue
        for seed in SEEDS:
            for world in WORLDS:
                if world == "null" and seed != 0:
                    continue
                out.append(_variant(route, world, seed))
            rows.append({"base_id": rid, "town": route.get("town"), "family": fam, "orig_type": s.get("type"),
                         "seed": seed, "plus": variant_id(rid, "plus", seed), "minus": variant_id(rid, "minus", seed),
                         "null": variant_id(rid, "null", 0) if seed == 0 else ""})
    cases = pd.DataFrame(rows)
    xml = runs_dir() / "pairs.xml"
    ET.indent(out)
    ET.ElementTree(out).write(xml)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases.to_csv(RESULTS / "cases.csv", index=False)
    log.info("%d cases (%d base routes) -> %s; per family %s", len(cases), cases.base_id.nunique(), xml,
             cases.groupby("family").size().to_dict())
    return cases


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    a = p.parse_args()
    if a.cmd == "build":
        build()


if __name__ == "__main__":
    main()
