"""Night queue 3, lane C: the P6 v0 judge shared by Q1 (examinee readouts) and Q2 (bypass elicitation)
(todos/2026-09-26-night-queue-3.md, general rule 7 and the [C] entries under Q1 / Q2, written before any number).

  frames   the exam's window frames per case -> processed/<set>/nq3_exam_frames.parquet (frame_name, world, reading,
           priority) and nq3_cases.parquet (one row per case with its windows); every examinee is read on these
  judge    predictions {examinee: (n, 20, 2) ego-frame futures, rear axle, 0.25 s grid} aligned to the set's index ->
           per-frame Delta_lat / Delta_v, tau_lat / tau_lon, bypass flips, stop substitutions, route-bootstrap CIs,
           the gate (CI low > weather-null out-of-sample false flips + 10 pp, and selectivity on the placement null),
           negotiation (x11 vs x10), the mirror's borrow-the-oncoming-lane rate, examinee world modes vs the expert's

Readings (rule 7):
  bypass   x10 vs x00 at the same tick, frames t_vis .. t_div_lat + 2 s of the bypass pairs with t_div >= t_vis
  wnull    wnull vs x10 on the seed-0 case's bypass window (the noise pair: tau and false flips)
  shoulder shoulder vs x00 on the seed-0 window, the 40 placement-null worlds whose expert mode is keep
  neg      x11 vs x01 (2W), frames t_vis .. t_div_lat(x11, x01) + 2 s (to the end of the recording if they never split)
  mirror   mirror vs x01 (2W, seed 0) on the seed-0 bypass window
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
N1 = REPO / "research" / "results" / "night2" / "N1"
MAIN = ("Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane", "AccidentTwoWays",
        "ConstructionObstacleTwoWays", "ParkedObstacleTwoWays", "HazardAtSideLaneTwoWays")
SEPARATE = ("VehicleOpensDoorTwoWays", "InvadingTurn", "YieldToEmergencyVehicle")
TWO_WAY = ("AccidentTwoWays", "ConstructionObstacleTwoWays", "ParkedObstacleTwoWays", "HazardAtSideLaneTwoWays",
           "VehicleOpensDoorTwoWays")
CAM = 4                       # ticks per 5 Hz camera frame
POST = 40                     # 2 s after t_div_lat, in ticks
LAT_DIV = 0.3                 # m, lateral divergence (N1 [A] 09:58)
I3S, I2S = 11, 7              # future index of 3 s and 2 s on the 0.25 s grid
N_BOOT = 10_000
MARGIN = 0.10


def proc(set_: str, *p) -> Path:
    return data_dir() / "processed" / set_ / Path(*p)


# ---------------------------------------------------------------- windows and frames

def _lat(g: Path, rid: str, cache: dict):
    from . import p6
    x = p6._load(g, rid, cache) if rid else None
    return None if x is None else x[1]


def _div_lat(la, lb) -> int | None:
    m = la[["d"]].join(lb[["d"]], rsuffix="_b", how="inner")
    i = np.flatnonzero(np.abs(m.d - m.d_b) >= LAT_DIV)
    return int(m.index[i[0]]) if len(i) else None


def cases_table(g: Path | None = None, workers: int = 24) -> pd.DataFrame:
    """One row per (base_id, seed) with the expert's x10 mode / side, t_vis and the windows of every reading."""
    from joblib import Parallel, delayed
    from . import p6
    g = g or p6.root("gen")
    c = p6.cases()
    w = pd.read_csv(N1 / "worlds.csv", dtype={"base_id": str, "rid": str})
    pr = pd.read_csv(N1 / "pairs_bypass.csv", dtype={"base_id": str})
    mode = w.pivot_table(index=["base_id", "seed"], columns="world", values="mode", aggfunc="first")
    c = c.merge(pr[["base_id", "seed", "t_vis", "t_div", "t_div_lat", "reason"]], on=["base_id", "seed"], how="left")
    c = c.join(mode.add_prefix("mode_"), on=["base_id", "seed"])

    def neg(r):
        if not (r.x11 and r.x01):
            return None
        cache = {}
        a, b = _lat(g, r.x11, cache), _lat(g, r.x01, cache)
        return None if a is None or b is None else _div_lat(a, b)
    c["t_div_lat_neg"] = Parallel(workers)(delayed(neg)(r) for _, r in c.iterrows())
    m10 = c.mode_x10.fillna("")
    c["side"] = np.where(m10.str.endswith("_L"), 1, np.where(m10.str.endswith("_R"), -1, 0))
    c["bypass_pair"] = m10.str.contains("bypass") & (c.reason == "ok") & c.t_div_lat.notna()
    c["main"] = c.scenario.isin(MAIN)
    return c


def exam_frames(set_: str = "carla_p6") -> pd.DataFrame:
    """Window frames of every reading. priority 0 = bypass (x10, x00) + wnull + shoulder on seed 0; 1 = the same on
    seeds 1-2; 2 = negotiation and mirror (x11 / x01 / mirror)."""
    t = pd.read_parquet(proc(set_, "index.parquet"), columns=["frame_name", "base_id", "seed", "world", "k"])
    c = cases_table()
    c.to_parquet(proc(set_, "nq3_cases.parquet"), index=False)
    seed0 = c[c.seed == 0].set_index("base_id")
    key = t.set_index(["base_id", "seed", "world"]).sort_index()
    rows = []

    def add(base, seed, world, lo, hi, reading, pri):
        try:
            f = key.loc[(base, seed, world)]
        except KeyError:
            return
        f = f[(f.k >= lo) & (f.k <= hi)]
        rows.append(pd.DataFrame({"frame_name": f.frame_name.to_numpy(), "k": f.k.to_numpy(), "base_id": base,
                                  "seed": seed, "world": world, "reading": reading, "priority": pri}))
    for _, r in c.iterrows():
        if r.bypass_pair:
            lo, hi = r.t_vis, r.t_div_lat + POST
            pri = 0 if r.seed == 0 else 1
            add(r.base_id, r.seed, "x10", lo, hi, "bypass", pri)
            add(r.base_id, r.seed, "x00", lo, hi, "bypass", pri)
            if r.seed == 0:
                add(r.base_id, 0, "wnull", lo, hi, "wnull", 0)
                if r.mode_shoulder == "keep":
                    add(r.base_id, 0, "shoulder", lo, hi, "shoulder", 0)
                if r.scenario in TWO_WAY:
                    add(r.base_id, 0, "mirror", lo, hi, "mirror", 2)
                    add(r.base_id, 0, "x01", lo, hi, "mirror", 2)
        if r.scenario in TWO_WAY and pd.notna(r.t_vis) and r.x11:
            hi = r.t_div_lat_neg + POST if pd.notna(r.t_div_lat_neg) else 10 ** 9
            add(r.base_id, r.seed, "x11", r.t_vis, hi, "neg", 2)
            add(r.base_id, r.seed, "x01", r.t_vis, hi, "neg", 2)
    fr = pd.concat(rows, ignore_index=True)
    # x11 only while its reference x01 is recorded (x01 stops 8 s after the ego passes the hidden obstacle)
    ref = fr[(fr.reading == "neg") & (fr.world == "x01")].groupby(["base_id", "seed"]).k.max()
    last = fr.set_index(["base_id", "seed"]).index.map(ref.to_dict()).to_numpy(dtype=float)
    fr = fr[~((fr.reading == "neg") & (fr.world == "x11") & ~(fr.k.to_numpy() <= np.nan_to_num(last, nan=-1)))]
    fr = fr.reset_index(drop=True)
    fr.to_parquet(proc(set_, "nq3_exam_frames.parquet"), index=False)
    u = fr.drop_duplicates("frame_name")
    log.info("%d reading rows, %d unique frames; by reading %s; unique by priority %s", len(fr), len(u),
             fr.groupby("reading").size().to_dict(), u.groupby("priority").size().to_dict())
    return fr
