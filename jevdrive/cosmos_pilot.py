"""Cosmos-Transfer2.5 pilot: CARLA counterfactual pairs re-rendered photoreal (todos/2026-09-28-cosmos-pilot.md).

  select     10 P5 v1 BehaviorAgent pedestrian pairs (small towns, seed 0) and a 93-tick window each, ending at the
             last tick the two egos share -> research/results/cosmos/pairs.csv, runs/cosmos/agent.json
  controls   per re-rendered world: determinism check against the P5 v1 attempt, then the Cosmos inputs (rgb / edge /
             seg / depth mp4) and the hazard ground truth (mask per frame) -> runs/cosmos/clips/<pair>/<member>/
  specs      Cosmos inference specs (jsonl) for a control variant and a set of pairs; both members of a pair get the
             same prompt, seed and settings
The GPU work lives in scripts/cosmos_gen.sh (CARLA re-render), scripts/cosmos_infer.py (Cosmos, envs/cosmos-transfer),
scripts/cosmos_openpilot.py (openpilot, envs/openpilot) and jevdrive/cosmos_eval.py (checks 1-4, figures).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "cosmos"
P5 = "processed/carla_p5v1_ba"
GEN_V1 = "runs/p5v1/gen-ba"
T = 93                     # Cosmos chunk (the distilled model takes exactly 93 frames); 4.65 s at CARLA's 20 Hz
FPS = 20
PED = ("PedestrianCrossing", "DynamicObjectCrossing", "ParkingCrossingPedestrian", "VehicleTurningRoutePedestrian")
SMALL = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town07", "Town10HD", "Town11")
# the pilot's base routes: every small-town pedestrian route of P5 v1, one per town and family first
BASES = ("24211", "24224", "24294", "24206", "27515", "27529", "24519", "25863", "24252", "27297")
PILOT1 = "24211"


def root(*parts) -> Path:
    p = data_dir() / "runs" / "cosmos" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def pairs() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "pairs.csv", dtype={"base_id": str, "plus": str, "minus": str})


def select():
    p = pd.read_csv(data_dir() / P5 / "pairs.csv", dtype={"base_id": str, "plus": str, "minus": str})
    o = pd.read_parquet(data_dir() / P5 / "obs.parquet")
    p = p[p.base_id.isin(BASES) & (p.seed == 0) & (p.reason == "ok")].copy()
    assert len(p) == len(BASES) and p.family.isin(PED).all() and p.town.isin(SMALL).all(), p
    p["k1"] = p.t_div.astype(int)                  # the egos differ from t_div on: stop one tick before
    p["k0"] = p.k1 - T
    assert (p.k0 >= 8).all(), p[["base_id", "k0"]]
    rows = []
    for _, r in p.iterrows():
        g = o[(o.base_id == r.base_id) & (o.seed == 0) & (o.k >= r.k0) & (o.k < r.k1)]
        rows.append({"obs_frames": len(g), "px_max": int(g.factor_px.max()) if len(g) else 0,
                     "vis_frac": float((g.factor_px >= 100).mean()) if len(g) else 0.0,
                     "impure_max": int(g.impure_visible.max()) if len(g) else 0, "v0_mean": float(g.v0.mean()) if len(g) else np.nan})
    p = pd.concat([p.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    p["pair"] = p.base_id + "-s0"
    cols = ["pair", "base_id", "seed", "town", "family", "plus", "minus", "t_vis", "t_div", "k0", "k1", "obs_frames",
            "px_max", "vis_frac", "impure_max", "v0_mean"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    p[cols].to_csv(RESULTS / "pairs.csv", index=False)
    tf = json.loads((data_dir() / "runs/p5_pairs/agent_config.json").read_text())["tfv6_model_dir"]
    win = {r[w]: [int(r.k0), int(r.k1)] for _, r in p.iterrows() for w in ("plus", "minus")}
    (root() / "agent.json").write_text(json.dumps({"tfv6_model_dir": tf, "save_threads": 3, "cosmos_windows": win}, indent=1))
    log.info("\n%s", p[cols].to_markdown(index=False))
    return p


def world_ids(which: str = "all") -> str:
    p = pairs()
    if which == "pilot1":
        p = p[p.base_id == PILOT1]
    return ",".join(r[w] for _, r in p.iterrows() for w in ("plus", "minus"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("select", "ids"))
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    if a.step == "select":
        select()
    elif a.step == "ids":
        print(world_ids(a.which))


if __name__ == "__main__":
    main()
