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


# ---------------------------------------------------------------- judge (rule 7)

READ_WORLDS = {"bypass": ("x10", "x00"), "wnull": ("wnull", "x10"), "shoulder": ("shoulder", "x00"),
               "neg": ("x11", "x01"), "mirror": ("mirror", "x01")}


def pairs(set_: str = "carla_p6") -> pd.DataFrame:
    """One row per (reading, case, tick): the two frames (row positions in the set's index) the reading compares."""
    t = pd.read_parquet(proc(set_, "index.parquet"), columns=["frame_name"])
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    fr = pd.read_parquet(proc(set_, "nq3_exam_frames.parquet"))
    c = pd.read_parquet(proc(set_, "nq3_cases.parquet"))
    out = []
    for rd, (wa, wb) in READ_WORLDS.items():
        f = fr[fr.reading == rd]
        a = f[f.world == wa][["base_id", "seed", "k", "frame_name"]]
        b = f[f.world == wb][["base_id", "seed", "k", "frame_name"]]
        m = a.merge(b, on=["base_id", "seed", "k"], suffixes=("_a", "_b"))
        out.append(m.assign(reading=rd))
    p = pd.concat(out, ignore_index=True)
    p["ia"], p["ib"] = pos[p.frame_name_a].to_numpy(), pos[p.frame_name_b].to_numpy()
    p = p.merge(c[["base_id", "seed", "scenario", "cls", "side", "main", "mode_x10", "mode_x11", "t_vis"]],
                on=["base_id", "seed"], how="left")
    return p


def lat_v(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Lateral position at 3 s (2 s if the examinee's horizon ends before 3 s) and speed at 2 s."""
    from .p5_pairs import v2
    y3 = pred[:, I3S, 1]
    has = ~np.isnan(pred[:, 0, 0])
    if has.any() and np.isnan(y3[has]).all():
        return pred[:, I2S, 1], v2(pred), "y at 2 s (horizon < 3 s)"
    return y3, v2(pred), "y at 3 s"


def boot(num: np.ndarray, groups: np.ndarray, b: int = N_BOOT, seed: int = 0):
    """Mean of num and its route-bootstrap percentile CI (whole routes resampled)."""
    from .p5_exam import boot_ratio
    if not len(num):
        return np.nan, np.nan, np.nan
    return boot_ratio(num.astype(float), np.ones(len(num)), groups, b, seed)


def _moved(x, tau):
    return (np.abs(x) > tau) & (np.abs(x) > 0)


def score(p: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Per-pair Delta_lat / Delta_v for one examinee (NaN where either frame lacks a prediction)."""
    y, v, how = lat_v(pred)
    return p.assign(dlat=y[p.ia] - y[p.ib], dv=v[p.ia] - v[p.ib], y_b=y[p.ib], how=how)


def taus(s: pd.DataFrame) -> tuple[float, float]:
    w = s[(s.reading == "wnull") & s.dlat.notna()]
    return float(np.quantile(np.abs(w.dlat), 0.95)), float(np.quantile(np.abs(w.dv), 0.95))


def oos_ff(s: pd.DataFrame) -> float:
    """p5_exam's out-of-sample null false flip: tau from one half of the weather-null routes, any-direction false
    flips on the other half, both ways, averaged."""
    w = s[(s.reading == "wnull") & s.dlat.notna()]
    nb = np.array(sorted(w.base_id.unique()))
    perm = np.random.default_rng(0).permutation(nb)
    h = set(perm[: len(nb) // 2])
    out = []
    for a_in in (True, False):
        a, b = w[w.base_id.isin(h) == a_in], w[w.base_id.isin(h) != a_in]
        if len(a) and len(b):
            out.append(float(_moved(b.dlat, np.quantile(np.abs(a.dlat), 0.95)).mean()))
    return float(np.mean(out)) if out else np.nan


def flips(s: pd.DataFrame, tl: float, tv: float) -> pd.DataFrame:
    s = s[s.dlat.notna()].copy()
    s["flip"] = _moved(s.dlat, tl) & (np.sign(s.dlat) == s.side)
    s["stop_sub"] = (s.dv < -tv) & ~s.flip
    s["moved"] = _moved(s.dlat, tl)
    return s


def negotiation(s: pd.DataFrame, tl: float) -> dict:
    """Per 2W case: first tick |Delta_lat| > tau in x10 (vs x00) and in x11 (vs x01); 'later' = x11 at least 1 s
    after x10, or x11 never within a window that runs >= 1 s past x10's crossing."""
    rows = []
    for (b, sd), g in s[s.cls == "2W"].groupby(["base_id", "seed"]):
        g10, g11 = g[g.reading == "bypass"], g[g.reading == "neg"]
        if not len(g10) or not len(g11):
            continue
        k10 = g10.k[g10.moved]
        if not len(k10):
            continue
        k10 = int(k10.min())
        k11 = g11.k[g11.moved]
        if len(k11):
            later = int(k11.min()) - k10 >= 20
        elif g11.k.max() >= k10 + 20:
            later = True
        else:
            continue
        mx11 = str(g.mode_x11.iloc[0])
        rows.append({"base_id": b, "seed": sd, "scenario": g.scenario.iloc[0], "later": later,
                     "expert_wait": mx11.startswith("wait") or mx11 == "stop"})
    r = pd.DataFrame(rows)
    if not len(r):
        return {"neg_cases": 0}
    return {"neg_cases": len(r), "neg_later_rate": float(r.later.mean()),
            "neg_agree_expert": float((r.later == r.expert_wait).mean()), "neg_expert_wait": float(r.expert_wait.mean())}


def judge_one(name: str, p: pd.DataFrame, pred: np.ndarray, scopes=True) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Rule 7 for one examinee: the summary row, the per-class rows and the scored pairs."""
    s = score(p, pred)
    tl, tv = taus(s)
    s = flips(s, tl, tv)
    ff = oos_ff(s)
    by = s[(s.reading == "bypass") & s.main]
    fr, lo, hi = boot(by.flip.to_numpy(), by.base_id.to_numpy())
    sr, slo, shi = boot(by.stop_sub.to_numpy(), by.base_id.to_numpy())
    wn = s[s.reading == "wnull"]
    ref = float(((np.sign(wn.dlat) == wn.side) & wn.moved).mean()) if len(wn) else np.nan
    sh = s[(s.reading == "shoulder") & s.main]
    shf = float(sh.flip.mean()) if len(sh) else np.nan
    x00 = s[(s.reading == "bypass") & s.main]
    x00_abs = float((_moved(x00.y_b, tl) & (np.sign(x00.y_b) == x00.side)).mean()) if len(x00) else np.nan
    gate_a = bool(lo > ff + MARGIN)
    sel = bool(shf <= ref + MARGIN) if not np.isnan(shf) else False
    mi = s[s.reading == "mirror"]
    row = {"examinee": name, "readout": s.how.iloc[0] if len(s) else "", "n_frames": len(by),
           "routes": by.base_id.nunique(), "tau_lat": tl, "tau_lon": tv, "bypass_flip": fr, "lo": lo, "hi": hi,
           "null_ff_oos": ff, "gate_a": gate_a, "shoulder_flip": shf, "n_shoulder": len(sh), "shoulder_ref": ref,
           "x00_abs_side_rate": x00_abs, "selective": sel, "has_bypass": gate_a and sel,
           "verdict": ("has bypass" if gate_a and sel else "reacts to 'something', not bypass" if gate_a else "no bypass"),
           "stop_sub": sr, "stop_lo": slo, "stop_hi": shi,
           "mirror_borrow": float(((mi.dlat > tl)).mean()) if len(mi) else np.nan, "n_mirror": len(mi),
           **negotiation(s, tl)}
    row["mirror_flag"] = bool(row["mirror_borrow"] > 0.5) if len(mi) else False
    per = []
    if scopes:
        for sc, g in s[s.reading == "bypass"].groupby("scenario"):
            f_, l_, h_ = boot(g.flip.to_numpy(), g.base_id.to_numpy(), 2000)
            per.append({"examinee": name, "scenario": sc, "main": bool(g.main.iloc[0]), "n_frames": len(g),
                        "bypass_flip": f_, "lo": l_, "hi": h_, "stop_sub": float(g.stop_sub.mean()),
                        "wrong_side": float((g.moved & ~g.flip & (np.sign(g.dlat) != 0)).mean())})
    return row, pd.DataFrame(per), s


def paired_diff(sa: pd.DataFrame, sb: pd.DataFrame, col: str = "flip") -> tuple[float, float, float]:
    """Flip-rate difference a - b on the same bypass frames (main classes), one route bootstrap."""
    key = ["base_id", "seed", "k"]
    a = sa[(sa.reading == "bypass") & sa.main][key + [col]]
    b = sb[(sb.reading == "bypass") & sb.main][key + [col]]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    d = m[f"{col}_a"].astype(float).to_numpy() - m[f"{col}_b"].astype(float).to_numpy()
    return boot(d, m.base_id.to_numpy()) if len(m) else (np.nan, np.nan, np.nan)


# ---------------------------------------------------------------- examinee world modes (descriptive)

BYPASS_21 = ("nudge_return", "nudge_hold", "lane_change")


def world_modes(s: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Examinee world mode per (case, world) from its own 5 s futures on the reading's window frames ([C] entry):
    section 2.1 frame modes (p6.mode_21, ego frame); bypass_<side> if >= 2 consecutive bypass-type frames (side = sign
    of the lateral peak), wait_then_bypass if a stop frame precedes them, else stop if >= 2 consecutive stop frames,
    else keep. Worlds: x10 / x00 (bypass window), x11 (negotiation window), mirror and shoulder (their windows)."""
    from . import p6
    rows = []
    specs = (("bypass", "a", "x10"), ("bypass", "b", "x00"), ("neg", "a", "x11"), ("mirror", "a", "mirror"),
             ("shoulder", "a", "shoulder"))
    for rd, side_ab, world in specs:
        g = s[s.reading == rd]
        if not len(g):
            continue
        idx = g[f"i{side_ab}"].to_numpy()
        ok = ~np.isnan(pred[idx, :, 0]).any(1)
        m = np.full(len(g), "", object)
        m[ok] = p6.mode_21(pred[idx[ok]])
        yp = np.full(len(g), 0.0)
        pk = np.abs(pred[idx[ok], :, 1]).argmax(1)
        yp[ok] = pred[idx[ok], pk, 1]
        g = g.assign(m=m, yp=yp).sort_values(["base_id", "seed", "k"])
        for (b, sd), h in g.groupby(["base_id", "seed"]):
            mm = h.m.to_numpy()
            byp = np.isin(mm, BYPASS_21)
            st = mm == "stop"
            run2 = lambda x: np.flatnonzero(x[:-1] & x[1:])  # noqa: E731
            rb, rs = run2(byp), run2(st)
            if len(rb):
                side = "L" if h.yp.to_numpy()[rb[0]] > 0 else "R"
                mode = ("wait_then_bypass_" if st[: rb[0]].any() else "bypass_") + side
            else:
                mode = "stop" if len(rs) else "keep"
            rows.append({"base_id": b, "seed": sd, "world": world, "scenario": h.scenario.iloc[0], "mode": mode})
    return pd.DataFrame(rows)


def mode_agreement(wm: pd.DataFrame) -> pd.DataFrame:
    """Examinee vs expert world mode (N1 worlds.csv), wait_then_bypass_[LR] collapsed, per world type."""
    w = pd.read_csv(N1 / "worlds.csv", dtype={"base_id": str})[["base_id", "seed", "world", "mode"]]
    m = wm.merge(w, on=["base_id", "seed", "world"], suffixes=("", "_expert"))
    col = lambda x: x.str.replace("wait_then_bypass_[LR]", "wait_then_bypass", regex=True)  # noqa: E731
    m["agree"] = col(m["mode"]) == col(m["mode_expert"])
    return m
