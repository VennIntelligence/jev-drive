"""P6 v0: the behaviour-mode exam (todos/2026-09-26-night-queue-2.md N1). Counterfactual worlds on the Bench2Drive
obstacle-bypass scenarios, driven by PDM-Lite, recorded by scripts/p5_pair_agent.py (P5 v1's recorder) and paired
with P5's machinery (jevdrive/p5_pairs.py: load_world, ego_divergence, world_rows).

  build     variant XML + case table: per base route (bench2drive220 only) and TM seed
                x10  obstacle, no oncoming flow        x00  obstacle hidden and its PDM-Lite registration deleted
                x11  obstacle, oncoming flow (2W)      x01  no obstacle, oncoming flow (2W)
              and on seed 0: null (x10 under swapped weather), shoulder (placement null: obstacle moved onto the
              shoulder, registration deleted; 1W and 2W) and mirror (x10 with an unbroken oncoming flow; 2W).
              Variant id = base_id * 100 + world code * 10 + seed. The hooks are scripts/b2d_hooks.py p6_world.
  ids       the variant ids still to drive (comma list, for scripts/p6_gen.sh)
"""
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from . import p5_pairs as P
from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "night2" / "N1"
SOURCE = "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
CLASS = {**{s: "1W" for s in ("Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane")},
         **{s: "2W" for s in ("AccidentTwoWays", "ConstructionObstacleTwoWays", "ParkedObstacleTwoWays",
                              "HazardAtSideLaneTwoWays", "VehicleOpensDoorTwoWays")},
         "InvadingTurn": "IT", "YieldToEmergencyVehicle": "EV"}
WORLDS = {"x10": 1, "x00": 2, "null": 3, "x11": 4, "x01": 5, "shoulder": 6, "mirror": 7}
CODE = {v: k for k, v in WORLDS.items()}
SEEDS = (0, 1, 2)
# world -> (p6_obstacle, p6_oncoming for 2W)
SPEC = {"x10": ("on", "off"), "x00": ("hide", "off"), "null": ("on", "off"), "x11": ("on", "on"),
        "x01": ("hide", "on"), "shoulder": ("shoulder", "off"), "mirror": ("on", "dense")}


def root(*parts) -> Path:
    p = data_dir() / "runs" / "p6" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def variant_id(base: str, world: str, seed: int) -> str:
    return str(int(base) * 100 + WORLDS[world] * 10 + seed)


def parse_id(rid: str) -> tuple[str, str, int]:
    k = int(rid)
    return str(k // 100), CODE[(k // 10) % 10], k % 10


def worlds_of(cls: str, seed: int) -> list[str]:
    w = ["x10", "x00"] + (["x11", "x01"] if cls == "2W" else [])
    if seed == 0:
        w += ["null"] + (["shoulder"] if cls in ("1W", "2W") else []) + (["mirror"] if cls == "2W" else [])
    return w


def _variant(route: ET.Element, cls: str, world: str, seed: int) -> ET.Element:
    r = copy.deepcopy(route)
    r.set("id", variant_id(route.get("id"), world, seed))
    obstacle, oncoming = SPEC[world]
    r.set("p6_obstacle", obstacle)
    if cls == "2W":
        r.set("p6_oncoming", oncoming)
    if world == "null":
        ws = r.find("weathers")
        day = float(ws.find("weather").get("sun_altitude_angle")) > 0
        for w in ws.findall("weather"):
            for k, v in (P.NIGHT if day else P.NOON).items():
                w.set(k, str(v))
    return r


def build(src: Path | None = None) -> pd.DataFrame:
    out, rows = ET.Element("routes"), []
    for route in ET.parse(src or data_dir() / SOURCE).getroot().findall("route"):
        sc = list(route.iter("scenario"))
        if len(sc) != 1 or sc[0].get("type") not in CLASS or route.get("id") in P.CRASHERS:
            continue
        stype, rid = sc[0].get("type"), route.get("id")
        cls = CLASS[stype]
        for seed in SEEDS:
            ws = worlds_of(cls, seed)
            for w in ws:
                out.append(_variant(route, cls, w, seed))
            rows.append({"base_id": rid, "town": route.get("town"), "scenario": stype, "cls": cls, "seed": seed,
                         **{w: variant_id(rid, w, seed) if w in ws else "" for w in WORLDS}})
    cases = pd.DataFrame(rows)
    ET.indent(out)
    ET.ElementTree(out).write(root() / "pairs.xml")
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases.to_csv(RESULTS / "cases.csv", index=False)
    n = {w: int((cases[w] != "").sum()) for w in WORLDS}
    log.info("%d base routes, %d cases, %d worlds %s; per class %s", cases.base_id.nunique(), len(cases),
             sum(n.values()), n, cases.drop_duplicates("base_id").groupby("cls").size().to_dict())
    return cases


def cases() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "cases.csv", dtype=str, keep_default_na=False).astype({"seed": int})


def variants(c: pd.DataFrame) -> list[str]:
    """Every world, grouped by base route (consecutive runs share a town); x10 / x00 first, so a partial run pairs."""
    out = []
    for _, g in c.sort_values(["town", "base_id", "seed"]).groupby(["town", "base_id"], sort=False):
        for _, r in g.iterrows():
            out += [r[w] for w in WORLDS if r[w]]
    return out


def ids(only: str = "", out: str = ""):
    g = Path(out) if out else root("gen")
    want = set(only.split(",")) if only else None
    todo = [v for v in variants(cases()) if not (g / "done" / (v + ".json")).exists()
            and (want is None or v in want)]
    print(",".join(todo))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "ids"])
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="", help="generation dir (default runs/p6/gen)")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        ids(a.only, a.out)


if __name__ == "__main__":
    main()
