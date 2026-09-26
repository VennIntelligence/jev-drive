"""Night queue 4, G (ghost test and perturbation collapse), plus the closed-loop tables of K and X
(todos/2026-09-26-night-queue-4.md, sections G / K / X and the [F] entries, written before any G / K / X number).

  build    route table, variant XML and the registered windows ($DATA_DIR/runs/nq4/gk/):
             routes.csv      the G routes: Bench2Drive 220 routes whose trigger spawns the hazard actor, i.e. P5's
                             families without HardBreakRoute / Light (8 x 5) and P6's 8 usable obstacle classes (8 x 5)
             g_routes.xml    variants per base route b, id = 10 b + code: orig 0 (visibility camera on), ghost 1,
                             shift 2 (+15 m for odd b, -15 m for even b), swap 3 (only where a same-class swap exists)
             windows.csv     trigger window [s_trig - 30, s_trig + 10] m and <= 2 control windows (40 m, >= 80 m from the
                             trigger along the route and in the plane, same shape class, the nearest to the trigger)
  report   every table: runs/nq4/gk/results/{g,k,x}/ (the chain calls it after every step)

Readouts (G, registered): ghost reaction per (run, window); pre-visibility reaction per orig run with the visibility
camera; scenario pass per run (no collision in the scenario zone [s_trig - 30, s_trig + 80] and the car reached its end).
"""
from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
B2D_XML = "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
P5_CLASSES = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian", "ParkingCrossingPedestrian",
              "OppositeVehicleRunningRedLight", "StaticCutIn", "ParkingCutIn", "HighwayCutIn")
P6_CLASSES = ("Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane", "AccidentTwoWays",
              "ConstructionObstacleTwoWays", "ParkedObstacleTwoWays", "HazardAtSideLaneTwoWays")
SWAP_TYPE = {"Accident": "ConstructionObstacle", "ConstructionObstacle": "ParkedObstacle", "ParkedObstacle": "Accident",
             "AccidentTwoWays": "ConstructionObstacleTwoWays", "ConstructionObstacleTwoWays": "ParkedObstacleTwoWays",
             "ParkedObstacleTwoWays": "AccidentTwoWays", "VehicleTurningRoutePedestrian": "VehicleTurningRoute"}
SWAP_VAN = ("StaticCutIn", "ParkingCutIn", "HighwayCutIn")
VARIANTS = {"orig": 0, "ghost": 1, "shift": 2, "swap": 3}
SHIFT_M = 15.0
TRIG_WIN = (-30.0, 10.0)
CTRL_LEN, CTRL_GAP, CTRL_MAX, CTRL_START, CTRL_END_MARGIN, CTRL_STEP = 40.0, 80.0, 2, 50.0, 20.0, 5.0
TURN_DEG = 20.0                   # a 40 m window whose heading departs > 20 deg from its start heading is "turn"
DECEL_MPS, LAT_M = 3.0, 1.0       # ghost reaction
LEAD_M, LANE_HALF_M = 30.0, 1.75  # "no traffic actor in the ego lane within 30 m ahead" (else the window is following)
ZONE = (-30.0, 80.0)              # scenario pass zone
ZONE_LAT_M = 10.0
PX_ACTOR = 20
N_BOOT = 10_000


def root(*p) -> Path:
    d = data_dir() / "runs" / "nq4" / "gk" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def vid(base, variant: str) -> str:
    return str(int(base) * 10 + VARIANTS[variant])


# ---------------------------------------------------------------- routes, variants, windows

def route_table() -> pd.DataFrame:
    from .nq3_cl_report import GROUPS
    from .tfv6_rules import route_table as rt
    t = rt()
    t = t[t.scenario.isin(P5_CLASSES + P6_CLASSES)].copy()
    t["base"] = t.route_id
    t["set"] = np.where(t.scenario.isin(P6_CLASSES), "p6_obstacle", "p5_hazard")
    t["obstacle"] = t.scenario.isin(P6_CLASSES)
    t["group"] = t.family.map({f: g for g, fs in GROUPS.items() for f in fs})
    t["shift_m"] = np.where(t.base.astype(int) % 2 == 1, SHIFT_M, -SHIFT_M)
    t["swap"] = [SWAP_TYPE.get(s, "van" if s in SWAP_VAN else "") for s in t.scenario]
    return t.drop(columns=["route_id"]).reset_index(drop=True)


def build_xml(t: pd.DataFrame, out: Path) -> pd.DataFrame:
    src = ET.parse(data_dir() / B2D_XML).getroot()
    official = {r.get("id") for r in src.iter("route")}
    by_id = {r.get("id"): r for r in src.iter("route")}
    xml, rows = ET.Element("routes"), []
    for b, sc, shift, swap in t[["base", "scenario", "shift_m", "swap"]].itertuples(index=False):
        for v in VARIANTS:
            if v == "swap" and not swap:
                continue
            r = copy.deepcopy(by_id[b])
            i = vid(b, v)
            assert i not in official, i
            r.set("id", i)
            r.set("nq4_world", v)
            r.set("nq4_base", b)
            if v == "orig":
                r.set("nq4_vis", "1")
            elif v == "shift":
                r.set("nq4_shift_m", "%.1f" % shift)
            elif v == "swap":
                (s,) = r.iter("scenario")
                if swap == "van":
                    r.set("nq4_swap", "van")
                else:
                    s.set("type", swap)
                    s.set("name", s.get("name").replace(sc, swap))
            xml.append(r)
            rows.append({"id": i, "base": b, "variant": v, "scenario": sc, "swap_to": swap if v == "swap" else ""})
    ET.indent(xml)
    ET.ElementTree(xml).write(out)
    return pd.DataFrame(rows)


def dense_route(base: str) -> np.ndarray | None:
    """(n, 2) dense route of a base route as the evaluator interpolates it, from any recording of that route (P6 / P5
    recorder route.json, the zero-shot agent's route.json, or our own trace)."""
    D = data_dir() / "runs"
    for pat in (f"p6/gen/attempts/{int(base) * 100 + 10}/*/route.json", f"p5_pairs/gen/attempts/{int(base) * 100 + 10}/*/route.json",
                f"p5_pairs/gen/attempts/{int(base) * 100 + 11}/*/route.json"):
        for f in sorted(D.glob(pat)):
            j = json.loads(f.read_text())
            return np.array([[p["x"], p["y"]] for p in j], float)
    for f in sorted((D / "nq4" / "gk" / "arms").glob(f"*/*/s*/attempts/{vid(base, 'orig')}/*/nq4_meta.json")):
        xy = json.loads(f.read_text()).get("route_xy")
        if xy:
            return np.asarray(xy, float)
    return None


def arc(P: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def project(P: np.ndarray, s: np.ndarray, Q: np.ndarray, hint: np.ndarray | None = None):
    """Arc position and signed lateral offset (left of the route positive, CARLA world) of points Q on polyline P."""
    from scipy.spatial import cKDTree
    Q = np.atleast_2d(Q)
    _, i = cKDTree(P).query(Q)
    i0 = np.clip(i - 1, 0, len(P) - 2)
    best_s, best_l, best_d = np.zeros(len(Q)), np.zeros(len(Q)), np.full(len(Q), np.inf)
    for j in (i0, np.clip(i, 0, len(P) - 2)):
        a, b = P[j], P[j + 1]
        d = b - a
        L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        u = np.clip(((Q - a) * d).sum(1) / L ** 2, 0, 1)
        foot = a + u[:, None] * d
        dist = np.hypot(*(Q - foot).T)
        left = np.stack([d[:, 1], -d[:, 0]], 1) / L[:, None]       # CARLA is left-handed: (t_y, -t_x) points left
        lat = ((Q - foot) * left).sum(1)
        m = dist < best_d
        best_s[m], best_l[m], best_d[m] = s[j][m] + u[m] * L[m], lat[m], dist[m]
    return best_s, best_l


def _heading(P: np.ndarray) -> np.ndarray:
    d = np.diff(P, axis=0)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.r_[h, h[-1:]]


def shape(P, s, a, b) -> str:
    h = _heading(P)
    m = (s >= a) & (s <= b)
    if not m.any():
        return "straight"
    return "turn" if np.degrees(np.abs(h[m] - h[m][0]).max()) > TURN_DEG else "straight"


def windows(t: pd.DataFrame) -> pd.DataFrame:
    src = {r.get("id"): r for r in ET.parse(data_dir() / B2D_XML).getroot().iter("route")}
    rows = []
    for b in t.base:
        P = dense_route(b)
        (sc,) = src[b].iter("scenario")
        tp = sc.find("trigger_point")
        if P is None:
            rows.append({"base": b, "kind": "missing_route"})
            continue
        s = arc(P)
        L = float(s[-1])
        st = float(project(P, s, [[float(tp.get("x")), float(tp.get("y"))]])[0][0])
        a, z = max(0.0, st + TRIG_WIN[0]), min(L, st + TRIG_WIN[1])
        shp = shape(P, s, a, z)
        rows.append({"base": b, "kind": "trigger", "k": 0, "s0": a, "s1": z, "shape": shp, "s_trig": st, "route_len": L})
        cands = []
        T = np.array([float(tp.get("x")), float(tp.get("y"))])
        for c in np.arange(CTRL_START, L - CTRL_END_MARGIN - CTRL_LEN + 1e-6, CTRL_STEP):
            c1 = c + CTRL_LEN
            if not (c >= st + CTRL_GAP or c1 <= st - CTRL_GAP):
                continue
            m = (s >= c) & (s <= c1)
            if np.hypot(*(P[m] - T).T).min() < CTRL_GAP or shape(P, s, c, c1) != shp:
                continue
            cands.append((abs(c + CTRL_LEN / 2 - st), c, c1))
        chosen = []
        for _, c, c1 in sorted(cands):
            if len(chosen) < CTRL_MAX and all(c1 <= x0 or c >= x1 for x0, x1 in chosen):
                chosen.append((c, c1))
        for k, (c, c1) in enumerate(sorted(chosen), 1):
            rows.append({"base": b, "kind": "control", "k": k, "s0": c, "s1": c1, "shape": shp, "s_trig": st, "route_len": L})
    return pd.DataFrame(rows)


def build():
    out = root()
    t = route_table()
    t.to_csv(out / "routes.csv", index=False)
    v = build_xml(t, out / "g_routes.xml")
    v.to_csv(out / "variants.csv", index=False)
    w = windows(t)
    w.to_csv(out / "windows.csv", index=False)
    res = root("results", "g")
    w.to_csv(res / "windows.csv", index=False)
    t.to_csv(res / "routes.csv", index=False)
    n_ctrl = w[w.kind == "control"].groupby("base").size().reindex(t.base).fillna(0).astype(int)
    log.info("%d routes (%s), %d variants (%s); control windows per route %s; missing dense route %d", len(t),
             t.set.value_counts().to_dict(), len(v), v.variant.value_counts().to_dict(), n_ctrl.value_counts().to_dict(),
             int((w.kind == "missing_route").sum()))
    return t, v, w


# ---------------------------------------------------------------- one run

def record(adir: Path) -> dict | None:
    try:
        recs = json.loads((adir / "results.json").read_text())["_checkpoint"]["records"]
    except (OSError, ValueError, KeyError):
        return None
    return recs[0] if recs else None


def finished_attempt(out: Path, rid: str) -> Path | None:
    f = out / "done" / f"{rid}.json"
    if not f.exists():
        return None
    return out / "attempts" / rid / str(json.loads(f.read_text())["attempt"])


def ego_track(adir: Path):
    """(t, xy, v, actors or None, meta or None) from our trace; else the zero-shot agent's ticks.jsonl truth (rear axle),
    the expert's log, or nothing."""
    f = adir / "nq4_trace.npz"
    if f.exists():
        z = np.load(f)
        meta = json.loads((adir / "nq4_meta.json").read_text()) if (adir / "nq4_meta.json").exists() else {}
        e = z["ego"]
        return {"k": e[:, 0].astype(int), "t": e[:, 1], "xy": e[:, 2:4], "v": e[:, 6], "act": z["act"], "vis": z["vis"],
                "vis_ticks": z["vis_ticks"], "meta": meta}
    f = adir / "ticks.jsonl"
    if f.exists() and f.stat().st_size:
        d = pd.read_json(f, lines=True)
        if "truth" in d:
            xy = np.array(d.truth.tolist())[:, :2]
            return {"k": np.arange(len(d)), "t": d.t.to_numpy(), "xy": xy, "v": d.v.to_numpy(), "act": None, "vis": None,
                    "vis_ticks": None, "meta": {}}
    return None


def _collisions(rec: dict):
    out = []
    for k in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle"):
        for msg in rec.get("infractions", {}).get(k, []):
            m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", msg)
            i = re.search(r"id=(\d+)", msg)
            out.append((k, float(m.group(1)) if m else np.nan, float(m.group(2)) if m else np.nan, int(i.group(1)) if i else -1))
    return out


def scenario_pass(rec: dict, P, s, s_trig: float, scen_ids=()) -> dict:
    L = float(rec.get("meta", {}).get("route_length") or s[-1])
    reached = float(rec["scores"]["score_route"]) / 100.0 * L
    end = min(s_trig + ZONE[1], L - 1.0)
    hit = 0
    for _, x, y, i in _collisions(rec):
        if i in scen_ids:
            hit += 1
            continue
        if np.isfinite(x):
            cs, cl = project(P, s, [[x, y]])
            if ZONE[0] <= cs[0] - s_trig <= ZONE[1] and abs(cl[0]) <= ZONE_LAT_M:
                hit += 1
    return {"reached_m": reached, "zone_end_m": end, "zone_collisions": hit, "passed": bool(reached >= end and hit == 0)}


def ghost_reaction(tr: dict, P, s, s0: float, s1: float) -> dict:
    se, le = project(P, s, tr["xy"])
    if se.max() < s1:
        return {"status": "not_reached"}
    i0 = int(np.argmax(se >= s0))
    i1 = i0 + int(np.argmax(se[i0:] >= s1))
    sl = slice(i0, max(i1, i0 + 1))
    v = tr["v"]
    decel = float(v[i0] - v[sl].min())
    lat = float(np.abs(le[sl]).max())
    lead = False
    if tr["act"] is not None and len(tr["act"]):
        A = tr["act"]
        ks = set(tr["k"][sl].tolist())
        m = np.isin(A[:, 0].astype(int), list(ks)) & (A[:, 6] < 2)
        if m.any():
            sa, la = project(P, s, A[m][:, 2:4].astype(float))
            ek = dict(zip(tr["k"][sl], se[sl]))
            ds = sa - np.array([ek[int(k)] for k in A[m][:, 0]])
            lead = bool(((ds > 0) & (ds <= LEAD_M) & (np.abs(la) < LANE_HALF_M)).any())
    return {"status": "following" if lead else "ok", "decel": decel, "lat": lat,
            "reaction": bool(decel >= DECEL_MPS or lat >= LAT_M), "v_entry": float(v[i0]),
            "v_mean": float(v[sl].mean())}


def pre_visible(tr: dict, P, s) -> dict:
    """Readout 2: a sustained deceleration (a < -1 m/s^2 for >= 0.5 s) that starts 1-5 s before the hazard is first
    visible (>= PX_ACTOR pixels of a scenario actor in the visibility camera), not while following."""
    if tr["vis"] is None or not len(tr["vis_ticks"]):
        return {"status": "no_camera"}
    vis = tr["vis"]
    seen = vis[vis[:, 2] >= PX_ACTOR]
    if not len(seen):
        return {"status": "never_visible"}
    kv = int(seen[:, 0].min())
    k, v = tr["k"], tr["v"]
    a = np.full(len(v), np.nan)
    a[5:-5] = (v[10:] - v[:-10]) / 0.5
    idx = {int(x): j for j, x in enumerate(k)}
    if kv not in idx:
        return {"status": "no_ego_at_visible"}
    jv = idx[kv]
    onset = None
    for j in range(max(0, jv - 100), max(0, jv - 20) + 1):
        if j + 10 <= len(a) and np.all(a[j:j + 10] < -1.0):
            onset = j
            break
    if onset is not None and tr["act"] is not None:
        se, _ = project(P, s, tr["xy"][onset:onset + 1])
        A = tr["act"]
        m = (A[:, 0].astype(int) == int(k[onset])) & (A[:, 6] < 2)
        if m.any():
            sa, la = project(P, s, A[m][:, 2:4].astype(float))
            if (((sa - se[0]) > 0) & ((sa - se[0]) <= LEAD_M) & (np.abs(la) < LANE_HALF_M)).any():
                return {"status": "following", "k_visible": kv}
    return {"status": "ok", "k_visible": kv, "pre_reaction": onset is not None,
            "lead_s": None if onset is None else float((jv - onset) * 0.05)}


# ---------------------------------------------------------------- collect

def reuse_dirs() -> dict:
    """(candidate, seed) -> night queue 3 run dirs whose orig routes (official ids) are reused, same examinee, route, seed."""
    B = data_dir() / "runs" / "nq3" / "b"
    m = {("pdm", 0): [B / "cl1_expert"]}
    for s in (0, 1, 2):
        m[("cinque", s)] = [B / "arms" / "cl2" / f"s{s}"]
        for a in ("tfv6", "bridgedrive", "simlingo", "blue"):
            m[(a, s)] = [B / "arms" / a / f"s{s}"]
    return m


def collect() -> pd.DataFrame:
    t = route_table().set_index("base")
    w = pd.read_csv(root() / "windows.csv", dtype={"base": str})
    var = pd.read_csv(root() / "variants.csv", dtype={"id": str, "base": str}).set_index("id")
    geo = {}
    rows = []

    def geom(b):
        if b not in geo:
            P = dense_route(b)
            geo[b] = (P, arc(P)) if P is not None else (None, None)
        return geo[b]

    def one(cand, variant, seed, b, rid, adir, source):
        r = {"cand": cand, "variant": variant, "seed": seed, "base": b, "id": rid, "source": source}
        rec = record(adir) if adir is not None else None
        P, s = geom(b)
        if rec is None or P is None:
            r["status"] = "missing"
            rows.append(r)
            return
        r.update(DS=float(rec["scores"]["score_composed"]), RC=float(rec["scores"]["score_route"]), rec_status=rec["status"],
                 duration=rec.get("meta", {}).get("duration_game"))
        ww = w[w.base == b]
        trig = ww[ww.kind == "trigger"]
        if not len(trig):
            r["status"] = "no_window"
            rows.append(r)
            return
        st = float(trig.s_trig.iloc[0])
        tr = ego_track(adir)
        meta = tr["meta"] if tr else {}
        scen_ids = {i for x in meta.get("built", []) for i in x["ids"]}
        if variant == "shift" and meta.get("shifted"):
            st_v = st + float(meta["shifted"][0]["shift_m"])
            r["built"] = ",".join(x["type"] for x in meta.get("built", []))
        else:
            st_v = st
        r.update(scenario_pass(rec, P, s, st_v, scen_ids), status="ok", traced=tr is not None and tr["act"] is not None)
        if tr is not None:
            for _, x in ww.iterrows():
                g = ghost_reaction(tr, P, s, x.s0, x.s1)
                rows.append({**{k: r[k] for k in ("cand", "variant", "seed", "base", "id", "source")},
                             "window": f"{x.kind}{int(x.k)}", "wkind": x.kind, **{f"w_{k}": v for k, v in g.items()}})
            if variant == "orig":
                r.update({f"pv_{k}": v for k, v in pre_visible(tr, P, s).items()})
        rows.append(r)

    for d in sorted(root("arms").glob("*/*/s*")):
        cand, variant, seed = d.parent.parent.name, d.parent.name, int(d.name[1:])
        if variant not in VARIANTS:
            continue
        for rid in json.loads((d / "requested.json").read_text()) if (d / "requested.json").exists() else []:
            b = var.loc[rid, "base"] if rid in var.index else rid
            one(cand, variant, seed, b, rid, finished_attempt(d, rid), "nq4")
    have = {(r["cand"], r["seed"], r["base"]) for r in rows if r.get("variant") == "orig" and r.get("status") == "ok"
            and "window" not in r}
    for (cand, seed), dirs in reuse_dirs().items():
        for d in dirs:
            for b in t.index:
                if (cand, seed, b) in have or not (d / "done" / f"{b}.json").exists():
                    continue
                one(cand, "orig", seed, b, b, finished_attempt(d, b), "nq3")
                have.add((cand, seed, b))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.join(t[["scenario", "set", "obstacle", "group", "family"]], on="base")
    return df


# ---------------------------------------------------------------- tables

def boot(x, n=N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan, 0
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(x)


def g_tables(df: pd.DataFrame, out: Path):
    W = df[df.get("window").notna()] if "window" in df else pd.DataFrame()
    R = df[df.get("window").isna()] if "window" in df else df
    md = ["# G tables (generated by jevdrive.nq4_g report; registered readouts in the todo)"]
    if len(W):
        ok = W[W.w_status == "ok"].copy()
        ok["reaction"] = ok.w_reaction.astype(float)
        cov = W.groupby(["cand", "variant", "wkind"]).w_status.value_counts().unstack(fill_value=0).reset_index()
        cov.to_csv(out / "ghost_coverage.csv", index=False)
        per_route = ok.groupby(["cand", "variant", "wkind", "base"]).reaction.mean().reset_index()   # seeds (and windows) averaged
        rates = []
        pdm = per_route[(per_route.cand == "pdm") & (per_route.variant == "ghost") & (per_route.wkind == "trigger")].reaction.mean()
        for cand, g in per_route[per_route.variant == "ghost"].groupby("cand"):
            tg = g[g.wkind == "trigger"].reaction
            cg = g[g.wkind == "control"].reaction
            m, lo, hi, n = boot(tg.to_numpy())
            ctrl = float(cg.mean()) if len(cg) else np.nan
            ref = np.nanmax([ctrl, pdm]) if np.isfinite([ctrl, pdm]).any() else np.nan
            rates.append({"cand": cand, "ghost_rate": m, "lo": lo, "hi": hi, "routes": n, "control_rate": ctrl,
                          "pdm_ghost_rate": pdm, "position_memory": bool(np.isfinite(lo) and np.isfinite(ref) and lo > ref + 0.10)})
        rt = pd.DataFrame(rates)
        rt.to_csv(out / "ghost_rate.csv", index=False)
        md += ["## Readout 1: ghost reaction (ghost worlds; route bootstrap, seeds averaged)", rt.to_markdown(index=False, floatfmt=".3f"),
               "### coverage (windows by status)", cov.to_markdown(index=False)]
        cr = ok[(ok.variant == "orig") & (ok.wkind == "control")].groupby(["cand", "base"]).w_v_mean.mean().groupby("cand").mean()
        md += ["### orig: mean speed in the control windows (m/s)", cr.to_frame("cruise_mps").to_markdown(floatfmt=".2f")]
    if "pv_status" in R:
        pv = R[(R.variant == "orig") & (R.pv_status == "ok")]
        if len(pv):
            tab = pv.groupby("cand").agg(episodes=("pv_pre_reaction", "size"), pre_visible=("pv_pre_reaction", "mean")).reset_index()
            tab.to_csv(out / "pre_visible.csv", index=False)
            md += ["## Readout 2: reaction >= 1 s before first visibility (orig, traced runs)", tab.to_markdown(index=False, floatfmt=".3f")]
    ok = R[R.status == "ok"]
    if len(ok):
        sr = ok.groupby(["cand", "variant", "set"]).agg(runs=("passed", "size"), pass_rate=("passed", "mean"), DS=("DS", "mean")).reset_index()
        sr.to_csv(out / "pass_rates.csv", index=False)
        md += ["## Scenario pass by candidate / variant / route set", sr.to_markdown(index=False, floatfmt=".3f")]
        col = []
        for cand, g in ok.groupby("cand"):
            o = g[g.variant == "orig"].set_index(["base", "seed"]).passed
            for v in ("shift", "swap"):
                p = g[g.variant == v].set_index(["base", "seed"]).passed
                j = pd.concat([o.rename("o"), p.rename("p")], axis=1, join="inner")
                if not len(j):
                    continue
                d = (j.o.astype(float) - j.p.astype(float)).groupby(level=0).mean()
                m, lo, hi, n = boot(d.to_numpy())
                col.append({"cand": cand, "variant": v, "drop": m, "lo": lo, "hi": hi, "routes": n,
                            "seeds": int(j.index.get_level_values(1).nunique()), "collapse": bool(lo > 0.10) if n else False})
        if col:
            ct = pd.DataFrame(col)
            ct.to_csv(out / "collapse.csv", index=False)
            md += ["## Readout 3: collapse = orig pass - variant pass (paired, route bootstrap; 1 seed: direction only)",
                   ct.to_markdown(index=False, floatfmt=".3f")]
        o = ok[(ok.variant == "orig") & (ok.group == "sudden")]
        pb = o[o.cand == "blue"].groupby("base").passed.mean()
        ps = o[o.cand == "simlingo"].groupby("base").passed.mean()
        j = pd.concat([pb.rename("b"), ps.rename("s")], axis=1, join="inner").dropna()
        if len(j):
            m, lo, hi, n = boot((j.b - j.s).to_numpy())
            md += ["## Readout 5: BLUE - SimLingo, orig sudden-hazard routes, pass rate (seeds averaged, route bootstrap)",
                   f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}], {n} routes"]
        dur = ok[(ok.variant == "orig") & (ok.rec_status.isin(["Completed", "Perfect"]))].groupby("cand").duration.mean()
        md += ["### orig: mean route completion time of completed routes (s, game time)", dur.to_frame("duration_s").to_markdown(floatfmt=".1f")]
    (out / "g.md").write_text("\n\n".join(md) + "\n")


def kx_tables(out_k: Path, out_x: Path):
    """K (P7 ladder on the 220, unseen and seen) and X (obstacle routes) from the chain's arm dirs."""
    from .nq3_cl_report import collect as cl_collect
    base = root("arms_k")
    df = cl_collect(base) if any(base.glob("*/s*")) else pd.DataFrame()
    if len(df):
        df.to_csv(out_k / "per_route.csv", index=False)
        agg = df.groupby(["arm", "seed"]).agg(routes=("route", "size"), finished=("finished", "sum"), DS=("DS", "mean"),
                                               SR=("success", "mean")).reset_index()
        agg.to_csv(out_k / "arms.csv", index=False)
        md = ["# K closed-loop tables", agg.to_markdown(index=False, floatfmt=".3f")]
        pairs = [("k1_unseen", "k0_unseen"), ("k2_unseen", "k1_unseen"), ("k3_unseen", "k2_unseen"), ("k3_unseen", "k0_unseen"),
                 ("k0_seen", "k0_unseen"), ("k3_seen", "k3_unseen")]
        for a, b in pairs:
            A = df[df.arm == a].groupby("route").DS.mean()
            B = df[df.arm == b].groupby("route").DS.mean()
            j = pd.concat([A.rename("a"), B.rename("b")], axis=1, join="inner").dropna()
            if len(j):
                m, lo, hi, n = boot((j.a - j.b).to_numpy())
                md.append(f"- {a} - {b}: DS {m:+.2f} [{lo:+.2f}, {hi:+.2f}], {n} routes")
        (out_k / "k.md").write_text("\n\n".join(md) + "\n")
    g = collect_cached()
    if len(g):
        R = g[g.get("window").isna() & (g.status == "ok") & (g.variant == "orig") & g.obstacle.astype(bool)] if "window" in g else g
        x = R[R.cand.isin(["x", "q2", "pdm"])]
        if len(x):
            tab = x.groupby(["cand", "seed"]).agg(routes=("passed", "size"), pass_rate=("passed", "mean"), DS=("DS", "mean")).reset_index()
            tab.to_csv(out_x / "x.csv", index=False)
            md = ["# X tables (obstacle routes, orig)", tab.to_markdown(index=False, floatfmt=".3f")]
            A = x[x.cand == "x"].groupby("base").passed.mean()
            B = x[x.cand == "q2"].groupby("base").passed.mean()
            j = pd.concat([A.rename("a"), B.rename("b")], axis=1, join="inner").dropna()
            if len(j):
                m, lo, hi, n = boot((j.a - j.b).to_numpy())
                md.append(f"- X - Q2 head (same cross-fitted fold): pass {m:+.3f} [{lo:+.3f}, {hi:+.3f}], {n} routes "
                          f"(reading: CI low > -0.10 -> geometry suffices)")
            (out_x / "x.md").write_text("\n\n".join(md) + "\n")


_CACHE = {}


def collect_cached() -> pd.DataFrame:
    if "g" not in _CACHE:
        _CACHE["g"] = collect()
    return _CACHE["g"]


def report():
    df = collect_cached()
    out = root("results", "g")
    if len(df):
        df.to_csv(out / "per_run.csv", index=False)
        g_tables(df, out)
    kx_tables(root("results", "k"), root("results", "x"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "report"))
    a = ap.parse_args()
    build() if a.cmd == "build" else report()


if __name__ == "__main__":
    main()
