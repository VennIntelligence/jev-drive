"""P6 v0: the behaviour-mode exam (todos/2026-09-26-night-queue-2.md N1). Counterfactual worlds on the Bench2Drive
obstacle-bypass scenarios, driven by PDM-Lite, recorded by scripts/p5_pair_agent.py (P5 v1's recorder) and paired
with P5's machinery (jevdrive/p5_pairs.py: load_world, ego_divergence, world_rows).

  build     variant XML + case table: per base route (bench2drive220 only) and TM seed
                x10  obstacle, no oncoming flow        x00  obstacle hidden and its PDM-Lite registration deleted
                x11  obstacle, oncoming flow (2W)      x01  no obstacle, oncoming flow (2W)
              and on seed 0: wnull (weather null: x10 under swapped weather), shoulder (placement null: obstacle moved onto the
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
WORLDS = {"x10": 1, "x00": 2, "wnull": 3, "x11": 4, "x01": 5, "shoulder": 6, "mirror": 7}
CODE = {v: k for k, v in WORLDS.items()}
SEEDS = (0, 1, 2)
# world -> (p6_obstacle, p6_oncoming for 2W)
SPEC = {"x10": ("on", "off"), "x00": ("hide", "off"), "wnull": ("on", "off"), "x11": ("on", "on"),
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
        w += ["wnull"] + (["shoulder"] if cls in ("1W", "2W") else []) + (["mirror"] if cls == "2W" else [])
    return w


def _variant(route: ET.Element, cls: str, world: str, seed: int) -> ET.Element:
    r = copy.deepcopy(route)
    r.set("id", variant_id(route.get("id"), world, seed))
    obstacle, oncoming = SPEC[world]
    r.set("p6_obstacle", obstacle)
    if cls == "2W":
        r.set("p6_oncoming", oncoming)
    if world == "wnull":
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
        # the attempt b2d_run recorded as done (a stopped or crashed attempt may also have written a summary)
        f = g / "done" / (rid + ".json") if rid else None
        a = g / "attempts" / rid / str(json.loads(f.read_text())["attempt"]) if f is not None and f.exists() else None
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
    neg = {}
    if c.cls == "2W":
        wr = {r["world"]: r for r in worlds}
        x10, x11 = wr.get("x10", {}), wr.get("x11", {})
        neg = {q: c[q] for q in ("base_id", "scenario", "seed")}
        neg.update(mode_x10=x10.get("mode"), mode_x11=x11.get("mode"), t_lat_x10=x10.get("t_lat"),
                   t_lat_x11=x11.get("t_lat"))
        if X10 is not None and X11 is not None and x10.get("t_lat") is not None:
            k = int(x10["t_lat"])
            l10, l11 = X10[1], X11[1]
            if k in l11.index:
                neg.update(v_x10_at_tlat=round(float(l10.v[k]), 3), v_x11_at_tlat=round(float(l11.v[k]), 3))
            if x11.get("t_lat") is not None:
                neg["lat_delay_s"] = round((int(x11["t_lat"]) - k) * P.TICK, 2)
    return worlds, pair, frames, neg


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
    neg = pd.DataFrame([r[3] for r in res if r[3]])
    neg.to_csv(out / "negotiation.csv", index=False)
    worlds.to_csv(out / "worlds.csv", index=False)
    pairs.to_csv(out / "pairs_bypass.csv", index=False)
    frames.to_csv(out / "frame_modes.csv", index=False)
    return worlds, pairs, frames


MODES = ("keep", "stop", "bypass_L", "bypass_R", "wait_then_bypass_L", "wait_then_bypass_R")


def _modes(w: pd.DataFrame, by: str) -> pd.DataFrame:
    t = w.pivot_table(index=by, columns="mode", values="rid", aggfunc="count", fill_value=0)
    t = t.reindex(columns=[m for m in MODES if m in t.columns] + [m for m in t.columns if m not in MODES], fill_value=0)
    t.insert(0, "n", t.sum(1))
    byp = t[[m for m in t.columns if str(m).startswith(("bypass", "wait_then"))]].sum(1)
    t["bypass_share"] = (byp / t.n).round(3)
    return t


def report(out: Path | None = None) -> str:
    """Expert statistics tables (todo N1's delivery) from stats()' CSVs -> expert_stats.md."""
    out = out or RESULTS
    w = pd.read_csv(out / "worlds.csv", dtype={"base_id": str, "rid": str})
    w = w[w["mode"] != "missing"]
    pr = pd.read_csv(out / "pairs_bypass.csv", dtype={"base_id": str})
    ng = pd.read_csv(out / "negotiation.csv", dtype={"base_id": str}) if (out / "negotiation.csv").stat().st_size > 1 else pd.DataFrame()
    fm = pd.read_csv(out / "frame_modes.csv", dtype={"base_id": str}) if (out / "frame_modes.csv").stat().st_size > 1 else pd.DataFrame()
    md = []
    t1 = _modes(w[w.world == "x10"], "scenario")
    t1["usable (>= 0.70)"] = np.where(t1.bypass_share >= 0.7, "yes", "no")
    md += ["## x10: expert mode per scenario", t1.to_markdown()]
    md += ["## mode per world and class", _modes(w, ["cls", "world"]).to_markdown()]
    if len(pr):
        pr["div_minus_vis_s"] = (pr.t_div - pr.t_vis) * P.TICK
        pr["divlat_minus_vis_s"] = (pr.t_div_lat - pr.t_vis) * P.TICK
        t2 = pr.groupby("scenario").agg(pairs=("reason", "size"), ok=("reason", lambda r: int((r == "ok").sum())),
                                        early=("reason", lambda r: int((r == "early").sum())),
                                        never_visible=("reason", lambda r: int((r == "never_visible").sum())),
                                        med_div_minus_vis_s=("div_minus_vis_s", "median"),
                                        med_divlat_minus_vis_s=("divlat_minus_vis_s", "median")).round(2)
        md += ["## t_div vs t_vis (x10 vs x00)", t2.to_markdown()]
        sm = pr[pr.smoke1_frames > 0]
        md += [f"smoke 1: x00 |d(k+3 s)| < {SMOKE_KEEP} m on {int(sm.smoke1_keep.sum())} / {int(sm.smoke1_frames.sum())} "
               f"x10-bypass frames ({sm.smoke1_keep.sum() / max(sm.smoke1_frames.sum(), 1):.3f}; gate >= 0.95), "
               f"{len(sm)} pairs with such frames of {len(pr)}"]
    if len(ng):
        ng["wait_x11"] = ng.mode_x11.astype(str).str.startswith("wait") | (ng.mode_x11 == "stop")
        ng["wait_x10"] = ng.mode_x10.astype(str).str.startswith("wait") | (ng.mode_x10 == "stop")
        ng["dv"] = ng.get("v_x11_at_tlat", np.nan) - ng.get("v_x10_at_tlat", np.nan)
        t3 = ng.groupby("scenario").agg(cases=("wait_x11", "size"), wait_share_x11=("wait_x11", "mean"),
                                        wait_share_x10=("wait_x10", "mean"), med_lat_delay_s=("lat_delay_s", "median"),
                                        med_dv_x11_minus_x10=("dv", "median")).round(3)
        tot = ng[["wait_x11", "wait_x10"]].mean().round(3)
        md += ["## negotiation: x11 - x10 (2W)", t3.to_markdown(),
               f"pooled wait share x11 {tot.wait_x11} (gate >= 0.50), x10 {tot.wait_x10}"]
    sh, mi = w[w.world == "shoulder"], w[w.world == "mirror"]
    if len(sh):
        md += [f"placement null keep: {int((sh['mode'] == 'keep').sum())} / {len(sh)} = {(sh['mode'] == 'keep').mean():.3f} (gate >= 0.90)",
               _modes(sh, "scenario").to_markdown()]
    if len(mi):
        md += [f"mirror stop: {int((mi['mode'] == 'stop').sum())} / {len(mi)} = {(mi['mode'] == 'stop').mean():.3f} (gate >= 0.80)",
               _modes(mi, "scenario").to_markdown()]
    wn = w[w.world == "wnull"].merge(w[w.world == "x10"][["base_id", "seed", "mode"]], on=["base_id", "seed"],
                                     suffixes=("", "_x10"))
    if len(wn):
        md += [f"weather null: same world mode as x10 in {int((wn['mode'] == wn.mode_x10).sum())} / {len(wn)}"]
    if "oncoming_in_window" in w:
        o = w[w.cls == "2W"].groupby("world").oncoming_in_window.agg(["size", lambda x: int((x >= 1).sum()), "mean"])
        o.columns = ["worlds", "with >= 1 oncoming", "mean oncoming"]
        md += ["## smoke 2: oncoming vehicles within 50 m in the lane-change window (2W)", o.round(2).to_markdown()]
    if len(fm):
        for col in ("mode_x10", "mode_x00"):
            md += [f"## frame-level section 2.1 modes ({col}, frames from t_vis on)",
                   fm.pivot_table(index="cls", columns=col, values="k", aggfunc="count", fill_value=0).to_markdown()]
    text = "\n\n".join(md) + "\n"
    (out / "expert_stats.md").write_text(text)
    return text


# ---------------------------------------------------------------- frame index (N2 probes, openpilot streams)

def _world_index(g: Path, rid: str, town: str, meta: dict):
    f = g / "done" / (rid + ".json")
    if not f.exists():
        return None
    a = g / "attempts" / rid / str(json.loads(f.read_text())["attempt"])
    r = P.world_rows(a, rid, town)
    if r is None:
        return None
    t, past, fut = r
    base, world, seed = parse_id(rid)
    return t.assign(base_id=base, world=world, seed=seed, source="p6", role="obs", **meta), past, fut


def index(g: Path | None = None, set_name: str = "carla_p6", workers: int = 24) -> pd.DataFrame:
    """Every 5 Hz camera frame with a 5 s future of every finished world -> processed/<set>/index.parquet, past.npy,
    future.npy (the P5 index layout, source "p6"), for night2_n2 labels / probes and scripts/p5_openpilot.py."""
    from joblib import Parallel, delayed
    g = g or root("gen")
    c = cases()
    jobs = [(r[w], r.town, {"scenario": r.scenario, "cls": r.cls}) for _, r in c.iterrows() for w in WORLDS if r[w]]
    res = [x for x in Parallel(workers)(delayed(_world_index)(g, *j) for j in jobs) if x is not None]
    t = pd.concat([x[0] for x in res], ignore_index=True)
    d = data_dir() / "processed" / set_name
    d.mkdir(parents=True, exist_ok=True)
    t.to_parquet(d / "index.parquet", index=False)
    np.save(d / "past.npy", np.concatenate([x[1] for x in res]))
    np.save(d / "future.npy", np.concatenate([x[2] for x in res]))
    log.info("%s: %d frames from %d worlds; by class/world %s", set_name, len(t), len(res),
             t.groupby(["cls", "world"]).size().to_dict())
    return t


# ---------------------------------------------------------------- CPU check (i): bypass anchors in cls_late's vocabulary

def bypass_shape(F: np.ndarray) -> np.ndarray:
    """(n, 20, 2) ego-frame futures at 0.25 s: |y(3 s)| >= 1 m and back within |y| <= 0.5 m somewhere in (3 s, 5 s]."""
    y = np.abs(F[:, :, 1])
    return (y[:, 11] >= 1.0) & (y[:, 12:].min(1) <= 0.5)


def mode_21(F: np.ndarray) -> np.ndarray:
    """Section 2.1 on ego-frame futures (n, 20, 2), speeds from the 0.25 s steps (v0 = first step)."""
    P0 = np.concatenate([np.zeros_like(F[:, :1]), F], 1)
    st = np.diff(P0, axis=1)
    hd = np.degrees(np.arctan2(st[..., 1], st[..., 0]))
    hd = np.stack([np.convolve(h, np.ones(3) / 3, "same") for h in hd])[:, 1:-1]
    sp = np.linalg.norm(st, axis=-1) / 0.25
    y = F[:, :, 1]
    peak, y_end = np.abs(y).max(1), np.abs(y[:, -1])
    h_end, h_max = np.abs(hd[:, -1]), np.abs(hd).max(1)
    v0, v_end = sp[:, 0], sp[:, -1]
    out = np.full(len(F), "keep", object)
    out[(v_end < 0.5) | ((v0 > 3) & (v_end < 0.3 * v0))] = "stop"
    out[peak >= 1] = "curve_or_other"
    out[(y_end >= 2.5) & (h_end < 10) & (h_max > 2 * h_end)] = "lane_change"
    out[(peak >= 1) & (y_end >= 1) & (y_end < 2.5) & (h_end < 5) & (h_max > 2 * h_end)] = "nudge_hold"
    out[(peak >= 1) & (y_end < 0.5 * peak)] = "nudge_return"
    out[h_end > 25] = "turn"
    return out


def vocab_check(out: Path | None = None, p6_futures: np.ndarray | None = None) -> pd.DataFrame:
    import torch
    from . import traj, waymo_heads as H
    out = out or RESULTS
    fut = H.train_futures()
    F = torch.as_tensor(fut.reshape(len(fut), -1), device=H.DEV)
    voc = traj.kmeans(F, 1024, seed=0).reshape(-1, 20, 2)
    voc = voc.cpu().numpy() if hasattr(voc, "cpu") else np.asarray(voc)
    np.save(out / "vocab_k1024_seed0.npy", voc.astype(np.float32))
    ab, am = bypass_shape(voc), mode_21(voc)
    rows = [{"set": "anchors", "n": len(voc), "bypass_shape": int(ab.sum()),
             **{f"mode_{m}": int((am == m).sum()) for m in ("keep", "stop", "nudge_return", "nudge_hold",
                                                            "lane_change", "curve_or_other", "turn")}}]
    V = torch.as_tensor(voc.reshape(len(voc), -1), device=H.DEV)
    sets = {"wod_train": fut}
    if p6_futures is not None:
        sets["p6_x10_expert"] = p6_futures
    for name, T in sets.items():
        b = bypass_shape(T)
        Tb = torch.as_tensor(T[b].reshape(int(b.sum()), -1), device=H.DEV)
        ade, idx = [], []
        for i in range(0, len(Tb), 8192):
            d = torch.cdist(Tb[i:i + 8192].view(-1, 20, 2).permute(1, 0, 2), V.view(-1, 20, 2).permute(1, 0, 2)).mean(0)
            m = d.min(1)
            ade.append(m.values.cpu().numpy())
            idx.append(m.indices.cpu().numpy())
        ade, idx = np.concatenate(ade) if ade else np.zeros(0), np.concatenate(idx) if idx else np.zeros(0, int)
        rows.append({"set": name, "n": len(T), "bypass_shape": int(b.sum()),
                     "nearest_anchor_bypass_share": round(float(ab[idx].mean()), 4) if len(idx) else None,
                     "min_ade_median_m": round(float(np.median(ade)), 3) if len(ade) else None,
                     "nearest_anchor_modes": json.dumps(pd.Series(am[idx]).value_counts().to_dict()) if len(idx) else ""})
    df = pd.DataFrame(rows)
    df.to_csv(out / "vocab_bypass.csv", index=False)
    log.info("\n%s", df.to_string())
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "ids", "stats", "vocab", "report", "index"])
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="", help="generation dir (default runs/p6/gen)")
    ap.add_argument("--results", default="", help="stats output dir (default research/results/night2/N1)")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "ids":
        ids(a.only, a.out)
    elif a.cmd == "vocab":
        vocab_check()
    elif a.cmd == "index":
        index(Path(a.out) if a.out else None)
    elif a.cmd == "report":
        print(report(Path(a.results) if a.results else None))
    else:
        stats(Path(a.out) if a.out else None, Path(a.results) if a.results else None)
        print(report(Path(a.results) if a.results else None))


if __name__ == "__main__":
    main()
