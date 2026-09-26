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
  stats     per-world lateral offset from the route, world-level behaviour mode, the two smoke checks, t_div vs t_vis,
            negotiation (x11 - x10) and the gates -> research/results/night2/N1/*.csv (runs on whatever is done)
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


# ---------------------------------------------------------------- per world: lateral offset and behaviour mode

LAT_ON, LAT_HOLD = 0.5, 10          # lateral start: |d| >= 0.5 m held for 10 ticks (0.5 s)
BYPASS_M = 1.0                      # section 2.1's 1 m nudge threshold
STOP_V, STOP_HOLD = 0.5, 20         # stopped: < 0.5 m/s for 20 ticks (1 s) ...
MOVING_V = 3.0                      # ... after having driven faster than 3 m/s (standing at the spawn is not a stop)
ONC_DEG, ONC_M = 135.0, 50.0        # oncoming: heading opposite by > 135 deg, within 50 m
SMOKE_BYPASS, SMOKE_KEEP = 1.0, 0.3
F3 = 60                             # 3 s in ticks


def lateral(W: dict) -> pd.DataFrame:
    """Per tick k: signed distance d of the ego from the (unshifted) route polyline, left positive, and the ego heading
    relative to the route tangent (right-handed, deg); nearest segment searched near the previous match (routes
    revisit places)."""
    route = pd.read_json(W["dir"] / "route.json")
    R = np.c_[route.x.to_numpy(), -route.y.to_numpy()]
    a, b = R[:-1], R[1:]
    seg = b - a
    L2 = np.maximum((seg ** 2).sum(1), 1e-9)
    pose = W["pose"]
    X = np.c_[pose.x.to_numpy(), -pose.y.to_numpy()]
    d, rel, prog = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X), int)
    j0 = None
    for i, x in enumerate(X):
        lo, hi = (0, len(seg)) if j0 is None else (max(0, j0 - 20), min(len(seg), j0 + 60))
        u = np.clip(((x - a[lo:hi]) * seg[lo:hi]).sum(1) / L2[lo:hi], 0, 1)
        q = a[lo:hi] + u[:, None] * seg[lo:hi]
        dist = np.hypot(*(x - q).T)
        j = lo + int(np.argmin(dist))
        j0 = j
        cr = seg[j, 0] * (x[1] - a[j, 1]) - seg[j, 1] * (x[0] - a[j, 0])
        d[i] = np.sign(cr) * dist[j - lo]
        tang = np.degrees(np.arctan2(seg[j, 1], seg[j, 0]))
        rel[i] = (-pose.yaw.iloc[i] - tang + 180) % 360 - 180
        prog[i] = j
    v = np.hypot(pose.vx.to_numpy(), pose.vy.to_numpy())
    return pd.DataFrame({"d": d, "psi": rel, "v": v, "seg": prog}, index=pose.index)


def _runs(mask: np.ndarray, n: int) -> np.ndarray:
    """Start indices of runs of True at least n long."""
    m = np.r_[False, mask, False].astype(int)
    st, en = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    return st[(en - st) >= n]


def world_mode(lat: pd.DataFrame, k0: int) -> dict:
    """World-level mode from tick k0 on (todo N1 [A] 09:58): bypass = max|d| >= 1 m, side by its sign; wait = a stop
    (< 0.5 m/s for 1 s, after having driven > 3 m/s) before the lateral start (|d| >= 0.5 m for 0.5 s)."""
    w = lat[lat.index >= k0]
    if not len(w):
        return {"mode": "no_window"}
    d, v, ks = w.d.to_numpy(), w.v.to_numpy(), w.index.to_numpy()
    st = _runs(np.abs(d) >= LAT_ON, LAT_HOLD)
    t_lat = int(ks[st[0]]) if len(st) else None
    ip = int(np.argmax(np.abs(d)))
    bypass = abs(d[ip]) >= BYPASS_M
    moved = np.maximum.accumulate(lat.v.to_numpy() > MOVING_V)[lat.index >= k0]
    sr = [int(ks[i]) for i in _runs((v < STOP_V) & moved, STOP_HOLD)]
    stop_before = [k for k in sr if t_lat is None or k < t_lat]
    side = "L" if d[ip] > 0 else "R"
    mode = (("wait_then_bypass_" if stop_before else "bypass_") + side) if bypass else ("stop" if sr else "keep")
    return {"mode": mode, "t_lat": t_lat, "max_abs_d": round(float(abs(d[ip])), 3), "d_at_max": round(float(d[ip]), 3),
            "t_first_stop": sr[0] if sr else None, "v_min": round(float(v.min()), 3)}


def frame_mode_21(lat: pd.DataFrame, k: int) -> str | None:
    """Section 2.1's rules on the 5 s future of frame k in route (Frenet) coordinates: offset change and heading
    relative to the route tangent instead of ego-frame y and yaw (CARLA has the map, so curves drop out)."""
    fk = k + 5 * np.arange(0, 21)
    if fk[-1] not in lat.index:
        return None
    w = lat.loc[fk]
    dd = w.d.to_numpy() - w.d.iloc[0]
    h = w.psi.to_numpy() - w.psi.iloc[0]
    h = np.convolve(h, np.ones(3) / 3, "same")[1:-1]
    h_end, h_max = abs(h[-1]), np.abs(h).max()
    peak, y_end = np.abs(dd).max(), abs(dd[-1])
    v0, v_end = w.v.iloc[0], w.v.iloc[-1]
    if h_end > 25:
        return "turn"
    if peak >= 1 and y_end < 0.5 * peak:
        return "nudge_return"
    if peak >= 1 and 1 <= y_end < 2.5 and h_end < 5 and h_max > 2 * h_end:
        return "nudge_hold"
    if y_end >= 2.5 and h_end < 10 and h_max > 2 * h_end:
        return "lane_change"
    if peak >= 1:
        return "curve_or_other"
    if v_end < 0.5 or (v0 > 3 and v_end < 0.3 * v0):
        return "stop"
    return "keep"


def _oncoming(W: dict, ks: range) -> int:
    """Distinct oncoming vehicles (heading opposite to the ego's by > 135 deg) within 50 m at any tick in ks."""
    act, pose = W["act"], W["pose"]
    sel = np.isin(act["k"], list(ks))
    if not sel.any():
        return 0
    k, ids, xyz, yaw = act["k"][sel], act["id"][sel], act["xyz"][sel], act["yaw"][sel]
    kinds = W["kinds"]
    veh = np.array([str(kinds.get(str(i), [""])[0]).startswith("vehicle.") for i in ids])
    p = pose.reindex(k)
    dist = np.hypot(xyz[:, 0] - p.x.to_numpy(), xyz[:, 1] - p.y.to_numpy())
    dyaw = np.abs((yaw - p.yaw.to_numpy() + 180) % 360 - 180)
    return int(len(set(ids[veh & (dist <= ONC_M) & (dyaw > ONC_DEG)].tolist())))


def _load(g: Path, rid: str, cache: dict):
    if rid not in cache:
        a = P.attempt(g, rid) if rid else None
        if a is None:
            cache[rid] = None
        else:
            W = P.load_world(a)
            W["summary"] = json.loads((a / "p5_summary.json").read_text())
            W["p6"] = json.loads((a / "p6_world.json").read_text()) if (a / "p6_world.json").exists() else {}
            cache[rid] = (W, lateral(W))
    return cache[rid]


def _case(g: Path, c: pd.Series) -> tuple[list, dict, list]:
    """One case: world rows, the (x10, x00) pair row with the smoke-1 frames, and the frame-level 2.1 modes."""
    cache, worlds, frames = {}, [], []
    X10, X00 = _load(g, c.x10, cache), _load(g, c.x00, cache)
    pair = {k: c[k] for k in ("base_id", "scenario", "cls", "seed")}
    t_vis = t_trig = None
    if X10 is not None:
        (A, la) = X10
        t_trig = next((int(k) for k, r in A["frames"].iterrows() if r.trig[0]), None)
    if X10 is not None and X00 is not None:
        (A, la), (B, lb) = X10, X00
        fv = P.factor_visibility(A, B, c.scenario)
        t_vis = next((k for k in sorted(fv) if fv[k]["factor_visible"]), None)
        t_div, last = P.ego_divergence(A, B)
        m = la[["d"]].join(lb[["d"]], rsuffix="_0", how="inner")
        dl = np.flatnonzero(np.abs(m.d - m.d_0) >= 0.3)
        cams = [k for k in sorted(fv) if k + F3 in la.index and k + F3 in lb.index]
        win = [k for k in cams if abs(la.d[k + F3]) >= SMOKE_BYPASS]
        ok = [k for k in win if abs(lb.d[k + F3]) < SMOKE_KEEP]
        pair.update(t_trig=t_trig, t_vis=t_vis, t_div=t_div, t_last=last,
                    t_div_lat=int(m.index[dl[0]]) if len(dl) else None,
                    smoke1_frames=len(win), smoke1_keep=len(ok),
                    max_d_x00_in_window=round(float(np.abs(lb.d.reindex([k + F3 for k in win])).max()), 3) if win else None,
                    reason="never_visible" if t_vis is None else "ok" if t_div >= t_vis else "early")
        for k in sorted(fv):
            if t_vis is not None and k >= t_vis:
                frames.append({**{q: c[q] for q in ("base_id", "scenario", "cls", "seed")}, "k": k,
                               "mode_x10": frame_mode_21(la, k), "mode_x00": frame_mode_21(lb, k)})
    k0 = t_vis if t_vis is not None else t_trig if t_trig is not None else 0
    onc_win = None
    X11 = _load(g, c.x11, cache) if c.x11 else None
    if X11 is not None:
        ml = world_mode(X11[1], k0)
        onc_win = (range(max(0, ml["t_lat"] - 200), ml["t_lat"] + F3) if ml.get("t_lat") is not None else
                   range((t_trig or 0) + 100, (t_trig or 0) + 500))
    for w in WORLDS:
        rid = c[w]
        if not rid:
            continue
        X = _load(g, rid, cache)
        row = {**{q: c[q] for q in ("base_id", "scenario", "cls", "seed")}, "world": w, "rid": rid}
        if X is None:
            worlds.append({**row, "mode": "missing"})
            continue
        W, lat = X
        row.update(world_mode(lat, k0), k0=k0, stop=W["summary"].get("stop"), ticks=W["summary"].get("ticks"),
                   registry_dropped=",".join(W["p6"].get("registry_dropped", [])),
                   registry_kept=",".join(W["p6"].get("registry_kept", [])))
        if c.cls == "2W" and onc_win is not None:
            row["oncoming_in_window"] = _oncoming(W, onc_win)
        worlds.append(row)
    return worlds, pair, frames


def stats(g: Path | None = None, out: Path | None = None, workers: int = 16):
    """Expert statistics on whatever worlds are done in g (default runs/p6/gen)."""
    from joblib import Parallel, delayed
    g, out = g or root("gen"), out or RESULTS
    out.mkdir(parents=True, exist_ok=True)
    c = cases()
    done = {p.stem for p in (g / "done").glob("*.json")}
    c = c[c.apply(lambda r: any(r[w] in done for w in WORLDS if r[w]), axis=1)]
    res = Parallel(workers)(delayed(_case)(g, r) for _, r in c.iterrows())
    worlds = pd.DataFrame([w for r in res for w in r[0]])
    pairs = pd.DataFrame([r[1] for r in res])
    frames = pd.DataFrame([f for r in res for f in r[2]])
    worlds.to_csv(out / "worlds.csv", index=False)
    pairs.to_csv(out / "pairs_bypass.csv", index=False)
    frames.to_csv(out / "frame_modes.csv", index=False)
    return worlds, pairs, frames


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "ids", "stats"])
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="", help="generation dir (default runs/p6/gen)")
    ap.add_argument("--results", default="", help="stats output dir (default research/results/night2/N1)")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "ids":
        ids(a.only, a.out)
    else:
        stats(Path(a.out) if a.out else None, Path(a.results) if a.results else None)


if __name__ == "__main__":
    main()
