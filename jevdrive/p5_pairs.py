"""P5 v0: counterfactual scenario pairs in CARLA and an open-loop exam on them
(todos/2026-09-24-p5-carla-pairs-v0.md).

  build     write the variant route XML (x+ as is, x- with the hazard actors suppressed (HardBreakRoute: the
            scenario dropped; Light: red swapped for green), a weather-only null) and the case table; each variant is its own numeric route id
            base_id * 100 + world * 10 + tm_seed, world 1 = plus, 2 = minus, 3 = null

The recorder is scripts/p5_pair_agent.py, run through scripts/b2d_run.py in envs/scout-tfv6.
"""
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
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
    elif world == "minus" and stype == "HardBreakRoute":
        # The factor is the background's hard brake itself, so x- drops the scenario. The evaluator names the
        # route after its first scenario config, so the element stays, with a trigger point 10 km away that
        # RouteScenario._filter_scenarios drops ("too far from the route").
        tp = s.find("trigger_point")
        tp.set("x", str(float(tp.get("x")) + 10000.0))
        s.set("name", s.get("name") + "_removed")
    else:
        # Every other scenario also commands the background (clear the junction, leave space), so deleting it
        # changes the background too; x- keeps it and b2d_hooks.track_hazards hides its hazard actors (and
        # names them in every world, so x+ and x- can be matched).
        r.set("p5_suppress", "1" if world == "minus" else "0")
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


# ---------------------------------------------------------------- one recorded world

TICK = 0.05
CAM_TICKS = 4
DIV_M, DIV_DEG = 0.01, 0.1                 # ego divergence: position >= 1 cm or heading >= 0.1 deg
ACTOR_DIFF_M = 0.1                         # a matched actor "differs" between worlds beyond this
VIS_M = 60.0


def attempt(gen: Path, rid: str) -> Path | None:
    """The finished attempt of a route variant (the one that wrote p5_summary.json and the most pose lines)."""
    best, n = None, -1
    for a in sorted((gen / "attempts" / rid).glob("*")):
        f = a / "pose.jsonl"
        if not (a / "p5_summary.json").exists() or not f.exists():
            continue
        k = sum(1 for _ in open(f))
        if k > n:
            best, n = a, k
    return best


def load_world(adir: Path) -> dict:
    """Everything a pair needs from one attempt, keyed by the tick index k = round(t / 0.05). Frame numbers are
    the server's and differ between runs; actor ids differ by a constant offset (the hero's id) when the same
    actors were spawned in the same order, which is the case for the worlds of one pair."""
    import json
    pose = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame")
    pose["k"] = (pose.t / TICK).round().astype(int)
    f2k = dict(zip(pose.frame, pose.k))
    frames = pd.read_json(adir / "frames.jsonl", lines=True)
    frames["k"] = (frames.t / TICK).round().astype(int)
    tf = pd.read_json(adir / "tfv6.jsonl", lines=True) if (adir / "tfv6.jsonl").stat().st_size else pd.DataFrame()
    if len(tf):
        tf["k"] = tf.frame.map(f2k)
    a = np.load(adir / "actors.npz")
    meta = json.loads((adir / "meta.json").read_text())
    hz = json.loads((adir / "hidden.json").read_text()) if (adir / "hidden.json").exists() else []
    lights = json.loads((adir / "lights.json").read_text())
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    return {"dir": adir, "pose": pose.set_index("k"), "frames": frames.set_index("k"), "tfv6": tf.set_index("k") if len(tf) else tf,
            "act": {"k": np.vectorize(f2k.get)(a["frame"]) if len(a["frame"]) else a["frame"], "id": a["id"],
                    "xyz": a["xyz"], "yaw": a["yaw"], "v": a["v"]},
            "meta": meta, "hero": meta.get("hero_id"), "hazards": [h["id"] for h in hz], "hazard_types": [h["type_id"] for h in hz],
            "light_loc": {l["id"]: tuple(np.round(l["loc"], 1)) for l in lights}, "kinds": kinds}


def ego_divergence(A: dict, B: dict) -> tuple[int, int]:
    """(t_div as a tick index, last common tick): the first tick the two egos differ by >= 1 cm or >= 0.1 deg."""
    m = A["pose"][["x", "y", "yaw"]].join(B["pose"][["x", "y", "yaw"]], rsuffix="_b", how="inner")
    d = np.hypot(m.x - m.x_b, m.y - m.y_b)
    dy = np.abs((m.yaw - m.yaw_b + 180) % 360 - 180)
    bad = np.flatnonzero((d >= DIV_M) | (dy >= DIV_DEG))
    last = int(m.index.max())
    return (int(m.index[bad[0]]) if len(bad) else last + 1), last


def _id_offset(A: dict, B: dict) -> int:
    return int(B["hero"] - A["hero"]) if A["hero"] is not None and B["hero"] is not None else 0


def factor_visibility(A: dict, B: dict, family: str) -> dict:
    """Per camera tick of A (x+): is a factor element visible, and how many non-factor actors differ in view.

    Factor elements: the hazard actors B hides (named in A's own hidden.json); for HardBreakRoute the background
    vehicles whose position differs between the worlds; for Light the traffic lights whose state differs. The
    rest of the actors that differ are the purity diagnostic."""
    off = _id_offset(A, B)
    hz = set(A["hazards"]) or {i - off for i in B["hazards"]}
    ka, kb = A["act"], B["act"]
    posb = {(int(k), int(i)): xyz for k, i, xyz in zip(kb["k"], kb["id"], kb["xyz"])}
    out = {}
    for k, row in A["frames"].iterrows():
        vis = {v["id"]: v for v in row.vis}
        sel = ka["k"] == k
        differ = set()
        for i, xyz in zip(ka["id"][sel], ka["xyz"][sel]):
            q = posb.get((k, int(i) + off))
            if q is None or np.hypot(*(xyz[:2] - q[:2])) > ACTOR_DIFF_M:
                differ.add(int(i))
        if family == "HardBreakRoute":
            factor = {i for i in differ if A["kinds"].get(str(i), ["", ""])[1] == "background"}
        else:
            factor = hz
        light_vis = False
        if family == "Light" and k in B["frames"].index:
            lb = {B["light_loc"].get(int(i)): s for i, s in B["frames"].loc[k].lights.items()}
            for lv in row.lvis:
                loc = A["light_loc"].get(lv["id"])
                sa, sb = row.lights.get(str(lv["id"])), lb.get(loc)
                if lv["vis"] and sa is not None and sb is not None and sa != sb:
                    light_vis = True
        fv = any(vis[i]["vis"] for i in factor if i in vis) or light_vis
        in_frustum = any(i in vis for i in factor) or light_vis
        impure = sum(1 for i in differ - factor if i in vis and vis[i]["vis"])
        out[int(k)] = {"factor_visible": bool(fv), "factor_in_frustum": bool(in_frustum), "impure_visible": impure,
                       "trig": bool(row.trig[0])}
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    a = p.parse_args()
    if a.cmd == "build":
        build()


if __name__ == "__main__":
    main()
