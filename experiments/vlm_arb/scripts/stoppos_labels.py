"""Frame tables and labels for the stop-position probe (plan: experiments/vlm_arb/plans/2026-10-03-stoppos-probe.md).

  python stoppos_labels.py validate     map-derived labels vs the logged CARLA truth of the closed-loop junctions
  python stoppos_labels.py build        frames_cl.parquet, frames_p4.parquet, routes.parquet, streams.json under
                                        $DATA_DIR/processed/vlm_arb_stoppos (project venv, CPU only)

Distances are metres along the route from the FRONT BUMPER, positive ahead. Labels are targets only, never model input.
  CL  from the agent logs: d_junc = gt.junc_dist - REAR_TO_BUMPER, d_stop = gt.tl_dist, d_sign = gt.stop_dist
  P4  from route.json + pose.jsonl + the OpenDRIVE file of the town (stoppos_xodr.py)
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stoppos_xodr as X  # noqa: E402

REAR = X.REAR_TO_BUMPER                 # rear axle -> front bumper (CL: arcs are rear-axle arcs)
CENTRE_TO_BUMPER = 2.4508               # P4: the logged pose is the actor origin (box centre)
LINE_OFFSET = 3.0                       # painted stop line minus CARLA stop waypoint, measured on 8 CL junctions (validate)
LINE_TOWNS = ("Town12", "Town13", "Town15")
ARMS = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/vlm_arb/arms"
P4 = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/p4_carla/gen/attempts"
OUT = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "processed/vlm_arb_stoppos"
BENCH = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "third_party/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml"
TURN = {"LEFT", "RIGHT", "STRAIGHT"}
CL_TURN = {1, 2, 3}


def jl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def run_start(arcs, cmd, codes):
    """Arc of the first point of the first run of a turn / straight command (None if the route has none)."""
    for i, c in enumerate(cmd):
        if c in codes:
            return float(arcs[i])
    return None


# ------------------------------------------------------------------------------------------------ CL
def cl_attempt(d):
    d = Path(d)
    fr = d / "vlm_frames"
    if not fr.is_dir():
        return None
    unit, route, att = d.parts[-4], d.parts[-2], d.parts[-1]
    recs = jl(d / "vlm_decisions.jsonl")
    S = [r for r in recs if r.get("k") == "s"]
    A = [r for r in recs if r.get("k") == "a"]
    if not A or not S:
        return None
    st = np.array([r["t"] for r in S])
    try:
        rt = json.load(open(d / "route.json"))
    except (OSError, ValueError):
        return None
    xy = np.asarray(rt["xy"], float)
    arcs = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    C = run_start(arcs, rt["cmd"], CL_TURN)
    rows, miss = [], 0
    for k, a in enumerate(sorted(A, key=lambda r: r["t_q"])):
        name = "%08.2f" % a["t_q"]
        road, wide = fr / f"{name}_road.jpg", fr / f"{name}_wide.jpg"
        if not (road.exists() and wide.exists()):
            miss += 1
            continue
        i = int(np.abs(st - a["t_q"]).argmin())
        if abs(st[i] - a["t_q"]) > 0.03:
            miss += 1
            continue
        s, g = S[i], a["gt"]
        jd = s["junc_dist"]
        rows.append(dict(t=a["t_q"], road=str(road), wide=str(wide), v=s["v"], ego_s=s["ego_s"], jid=s.get("jid", -1),
                         junc_dist=jd, tl=g.get("tl"), tl_id=g.get("tl_id"), tl_dist=g.get("tl_dist"), stop_dist=g.get("stop_dist"),
                         C=C))
    if not rows:
        return None
    return dict(unit=unit, route=route, att=att, rows=rows, miss=miss)


def cl_route_tables(atts):
    """Per route: junction entrance arcs (by id), light stop-line arcs and stop-sign arcs, from all attempts (rear-axle arcs)."""
    ent, lights, signs = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for a in atts:
        for r in a["rows"]:
            if r["junc_dist"] is not None and 1.0 < r["junc_dist"] < 100 and r["jid"] is not None and r["jid"] >= 0:
                ent[a["route"]][r["jid"]].append(r["ego_s"] + r["junc_dist"])
            if r["tl_dist"] is not None and r["tl_id"] is not None:
                lights[a["route"]][(a["unit"], a["att"], r["tl_id"])].append(r["ego_s"] + REAR + r["tl_dist"])
            if r["stop_dist"] is not None:
                signs[a["route"]].append((a["unit"], a["att"], r["ego_s"] + REAR + r["stop_dist"]))
    out = {}
    for route in {a["route"] for a in atts}:
        E = {j: float(np.median(v)) for j, v in ent[route].items()}
        arcs = sorted(float(np.median(v)) for v in lights[route].values() if len(v) >= 3)
        grp = []
        for x in arcs:
            if grp and x - grp[-1][-1] <= 4.0:
                grp[-1].append(x)
            else:
                grp.append([x])
        L = [float(np.median(g)) for g in grp]
        sg = []
        for k in sorted({(u, a) for u, a, _ in signs[route]}):
            sg.append(float(np.median([x for u, a, x in signs[route] if (u, a) == k])))
        out[route] = dict(E=E, L=L, sign=sorted(sg))
    return out


def cl_frames(atts, tabs):
    rows = []
    for a in atts:
        tb = tabs[a["route"]]
        for k, r in enumerate(a["rows"]):
            jd = r["junc_dist"]
            d_j = jd - REAR if jd is not None and jd < 999 else np.nan
            e = tb["E"].get(r["jid"]) if r["jid"] is not None and r["jid"] >= 0 else None
            light = sign = np.nan
            if e is not None:
                light = float(any(e - 15.0 <= x <= e + 1.0 for x in tb["L"]))
                sg = [x for x in tb["sign"] if e - 20.0 <= x <= e + 3.0]
                sign = float(len(sg) > 0 and not light)
            bumper = r["ego_s"] + REAR
            rows.append(dict(src="cl", route=a["route"], unit=a["unit"], att=a["att"], key=f"cl/{a['unit']}/{a['route']}/{a['att']}", k=k,
                             t=r["t"], v=r["v"], dist=bumper - REAR - a["rows"][0]["ego_s"], tdrive=r["t"] - a["rows"][0]["t"],
                             d_junc=d_j, d_stop=np.nan if r["tl_dist"] is None else r["tl_dist"],
                             d_sign=np.nan if r["stop_dist"] is None else r["stop_dist"],
                             d_cmd=np.nan if r["C"] is None else r["C"] - bumper, tl=np.nan if r["tl"] is None else r["tl"],
                             has_light=light, has_sign=sign, stop_ok=True, town=""))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ P4
def p4_attempt(d):
    d = Path(d)
    try:
        meta = json.load(open(d / "meta.json"))
        rt = json.load(open(d / "route.json"))
    except (OSError, ValueError):
        return None
    town = meta["town"].split("/")[-1]
    xy = np.array([[p["x"], p["y"]] for p in rt])
    cmd = [p["option_name"] for p in rt]
    L = X.label_route(town, xy)
    arcs = L["arc"]
    C = run_start(arcs, cmd, TURN)
    pose = pd.DataFrame(jl(d / "pose.jsonl")).drop_duplicates("frame").set_index("frame")
    frames = sorted(jl(d / "frames.jsonl"), key=lambda r: r["frame"])
    if not len(pose) or not frames:
        return None
    # heading-consistent projection of the pose on the route polyline
    P = xy
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-3]
    P, cum = P[keep], arcs[keep]
    ryaw = np.degrees(np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0])))
    ryaw = np.r_[ryaw, ryaw[-1:]]
    A, B = P[:-1], P[1:]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)

    def proj(x, y, yaw):
        q = np.array([x, y])
        t = np.clip(((q - A) * AB).sum(1) / L2, 0, 1)
        dist = np.hypot(*(q - (A + t[:, None] * AB)).T)
        ok = np.abs((ryaw[:-1] - yaw + 180) % 360 - 180) < 60
        if ok.any():
            dist = np.where(ok, dist, 1e9)
        j = int(dist.argmin())
        return cum[j] + t[j] * np.sqrt(L2[j])
    t0, f0 = float(pose.t.iloc[0]), int(pose.index[0])
    s0 = None
    S = L["S"]
    S_hat = None if (S is None or not L["has_light"] or town not in LINE_TOWNS) else S - LINE_OFFSET
    rows = []
    for k, fr in enumerate(frames):
        f = fr["frame"]
        base = dict(src="p4", route=d.parts[-2], unit="p4", att=d.parts[-1], key=f"p4/{d.parts[-2]}/{d.parts[-1]}", k=k, town=town,
                    files=[str(d / fr["files"][c]) for c in ("front", "front_left", "front_right")])
        if f not in pose.index:
            rows.append(dict(base, t=t0 + 0.05 * (f - f0), v=np.nan))
            continue
        e = pose.loc[f]
        sc = proj(e.x, e.y, e.yaw)
        if s0 is None:
            s0 = sc
        bump = sc + CENTRE_TO_BUMPER
        has_l, has_s = L["has_light"], L["has_sign"]
        stop_ok = (not has_l) or S_hat is not None
        rows.append(dict(base, t=float(e.t), v=float(np.hypot(e.vx, e.vy)), dist=sc - s0, tdrive=float(e.t) - t0,
                         d_junc=np.nan if L["E"] is None else L["E"] - bump,
                         d_stop=np.nan if S_hat is None else S_hat - bump, d_sign=np.nan,
                         d_cmd=np.nan if C is None else C - bump, tl=np.nan,
                         has_light=(float(has_l) if L["E"] is not None else np.nan),
                         has_sign=(float(has_s and not has_l) if L["E"] is not None else np.nan), stop_ok=stop_ok))
    route = dict(src="p4", route=d.parts[-2], att=d.parts[-1], town=town, E=L["E"], S=S, S_hat=S_hat, has_light=L["has_light"],
                 has_sign=L["has_sign"], C=C, n_frames=len(frames))
    return dict(rows=rows, route=route)


# ------------------------------------------------------------------------------------------------ build / validate
def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for t in X.XODR:
        X.parse(t)
    with Pool(16) as pool:
        cl = [a for a in pool.map(cl_attempt, sorted(glob.glob(f"{ARMS}/*/attempts/*/*")), chunksize=4) if a]
        p4 = [a for a in pool.map(p4_attempt, sorted(glob.glob(f"{P4}/*/*")), chunksize=2) if a]
    tabs = cl_route_tables(cl)
    fcl, fp4 = cl_frames(cl, tabs), pd.DataFrame([r for a in p4 for r in a["rows"]])
    fcl["road"] = [r["road"] for a in cl for r in a["rows"]]
    fcl["wide"] = [r["wide"] for a in cl for r in a["rows"]]
    town = {}
    xml = BENCH.read_text()
    for r in fcl.route.unique():
        m = re.search(r'<route id="%s" road_id="\d+" town="(\w+)"' % r, xml)
        town[r] = m.group(1) if m else ""
    fcl["town"] = fcl.route.map(town)
    for f in (fcl, fp4):
        f["kind"] = np.where(f.has_light == 1, "light", np.where(f.has_sign == 1, "sign", "none"))
    # route level kind: any approach frame of the route with a junction
    rt = []
    for name, f in (("cl", fcl), ("p4", fp4)):
        for route, g in f.groupby("route"):
            j = g[g.has_light.notna()]
            kind = "nojunc" if j.empty else ("light" if (j.has_light == 1).any() else "sign" if (j.has_sign == 1).any() else "none")
            rt.append(dict(src=name, route=route, id=f"{name}-{route}", town=g.town.iloc[0], kind=kind, n_att=g.att.nunique(), n_frames=len(g)))
    rt = pd.DataFrame(rt)
    fcl.to_parquet(OUT / "frames_cl.parquet")
    fp4.to_parquet(OUT / "frames_p4.parquet")
    rt.to_parquet(OUT / "routes.parquet")
    pd.DataFrame([a["route"] for a in p4]).to_parquet(OUT / "p4_route_labels.parquet")
    json.dump({r: dict(E={str(k): v for k, v in t["E"].items()}, L=t["L"], sign=t["sign"]) for r, t in tabs.items()}, open(OUT / "cl_route_tables.json", "w"))
    streams = []
    for key, g in fcl.groupby("key", sort=True):
        g = g.sort_values("k")
        streams.append(dict(key=key, src="cl", n=len(g), files=[[r, w] for r, w in zip(g.road, g.wide)], hold=10))
    for key, g in fp4.groupby("key", sort=True):
        g = g.sort_values("k")
        streams.append(dict(key=key, src="p4", n=len(g), files=[list(x) for x in g.files], hold=4))
    json.dump(streams, open(OUT / "streams.json", "w"))
    print("CL attempts", len(cl), "frames", len(fcl), "routes", fcl.route.nunique())
    print("P4 attempts", len(p4), "frames", len(fp4), "routes", fp4.route.nunique())
    print(rt.groupby(["src", "kind"]).size())


def validate():
    """Map-derived entrance / stop line vs the logged truth of the closed-loop junctions (route_junction_error.csv)."""
    import csv
    xml = BENCH.read_text()
    rows = list(csv.DictReader(open(Path(__file__).resolve().parents[1] / "results/route_junction_error.csv")))
    out = []
    for r in rows:
        rid = r["route"]
        town = re.search(r'<route id="%s" road_id="\d+" town="(\w+)"' % rid, xml).group(1)
        f = sorted(glob.glob(f"{ARMS}/*/attempts/{rid}/*/route.json"))[0]
        xy = np.array(json.load(open(f))["xy"])
        L = X.label_route(town, xy)
        E, S = float(r["E"]), float(r["S"]) if r["S"] else None
        out.append(dict(route=rid, town=town, E_log=E, E_map=L["E"], S_log=S, line_map=L["S"], has_light_map=L["has_light"],
                        has_sign_map=L["has_sign"], light_log=S is not None))
    t = pd.DataFrame(out)
    t["dE"] = t.E_log - t.E_map
    t["line_minus_S"] = t.line_map - t.S_log
    print(t.to_string())
    print("entrance |E_log - E_map|: mean %.2f, within 0.8 m %d/%d, within 2.5 m %d/%d" % (
        t.dE.abs().mean(), (t.dE.abs() <= 0.8).sum(), len(t), (t.dE.abs() <= 2.5).sum(), len(t)))
    m = t.line_minus_S.dropna()
    print("painted line minus CARLA stop waypoint: n %d mean %.2f sd %.2f" % (len(m), m.mean(), m.std()))
    print("light presence correct: %d/%d" % ((t.has_light_map == t.light_log).sum(), len(t)))
    t.to_csv(Path(__file__).resolve().parents[1] / "results/stoppos_label_validation.csv", index=False)


if __name__ == "__main__":
    {"validate": validate, "build": build}[sys.argv[1]]()
