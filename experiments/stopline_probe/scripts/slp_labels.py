#!/usr/bin/env python3
"""STOPLINE-PROBE step 2 (anywhere with numpy / pandas): red / green behaviour labels of the PAI_DUMP frames from slp_facts.py's pickle.

  slp_labels.py --facts facts.pkl --out DIR        -> DIR/events.csv (one row per (scene, wait line) event), DIR/frames.csv (labelled frames)

The PAI scenes carry no light state (map.traffic_light_status is None in all of them; the usdz holds no light field), so the label is the
human's behaviour at a map wait line (RULES): red (a stop at a signal line), stop_sign (a stop at a line with a mapped stop sign), green (passing a signal / unsigned line). A frame is a dump decision of any rollout of the scene (policy or log-following), labelled by
its time on the log clock, kept when the ego is on the logged path before the line.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RULES = dict(
    kinds=("STOP", "UNKNOWN"),   # wait-line types used; YIELD lines are dropped
    line_lat=4.0,                # line centre within this lateral distance of the logged path ...
    line_angle=45.0,             # ... the line at >= 45 deg to the path heading ...
    line_cross=0.5,              # ... and its end points on both sides of the path (0.5 m margin): the line the human's lane meets
    stop_v=0.5, stop_min_s=1.0,  # a human stop: log speed < 0.5 m/s for >= 1 s ...
    stop_ahead=(-3.0, 12.0),     # ... with the line 3 m behind to 12 m ahead of the stop point (rf_tax rule)
    lead_m=15.0, lead_lat=2.0,   # no vehicle on the logged path 0-15 m ahead of the stop point during its first 2 s -> red; else queue (dropped)
    sign_m=20.0,                 # a stop sign (R1_STOP) within 20 m of the line centre -> stop-sign line: a stop there is `stop_sign`, not red
    go_min_v=2.0,                # green: the human passes the line, min log speed over the 15 m before it >= 2 m/s
    win_s=5.0,                   # frames from 5 s before the event: red = [stop start - 5 s, stop end - 1 s], green = [crossing - 5 s, crossing]
    red_tail_s=1.0,
    ego_lat=4.0, ego_d=(0.0, 60.0),  # ego rear axle on the logged path (|lateral| <= 4 m), 0-60 m before the line
)
CLS = ("red", "stop_sign", "green")
VEH = ("automobile", "heavy_truck", "trailer", "bus", "other_vehicle", "train_or_tram_car")


def proj(P, X):
    """Polyline P (n, 2), points X (m, 2) -> station, signed lateral (left +) of the nearest point on P."""
    d = np.diff(P, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    ok = L > 1e-6
    P0, d, L = P[:-1][ok], d[ok], L[ok]
    s0 = np.r_[0, np.cumsum(L)][:-1]
    r = X[:, None] - P0[None]
    u = np.clip((r * d[None]).sum(-1) / L ** 2, 0, 1)
    q = P0[None] + u[..., None] * d[None]
    e = np.hypot(*(X[:, None] - q).transpose(2, 0, 1))
    j = e.argmin(1)
    i = np.arange(len(X))
    lat = np.sign(d[j, 0] * r[i, j, 1] - d[j, 1] * r[i, j, 0]) * e[i, j]
    return s0[j] + u[i, j] * L[j], lat


def events(sc):
    """One scene -> list of event dicts (red / green / dropped with the reason)."""
    T, P = sc["T"].astype(np.float64), sc["P"]
    ext = P[-1] + 15.0 * (P[-1] - P[-2]) / max(np.hypot(*(P[-1] - P[-2])), 1e-6)
    Px = np.vstack([P, ext])
    S, _ = proj(Px, P)
    S = np.maximum.accumulate(S)
    k = max(1, int(round(0.5e6 / np.median(np.diff(T)))))
    v = np.gradient(S, T * 1e-6)
    v = np.convolve(v, np.ones(k) / k, "same")
    stops, st, i = [], v < RULES["stop_v"], 0
    while i < len(st):
        if st[i]:
            j = i
            while j < len(st) and st[j]:
                j += 1
            if (T[j - 1] - T[i]) * 1e-6 >= RULES["stop_min_s"]:
                stops.append((i, j - 1))
            i = j
        else:
            i += 1
    stopsigns = np.array([g["pos"] for g in sc["signs"] if "R1_STOP" in g["type"]]).reshape(-1, 2)
    out, seen = [], []
    for w in sorted(sc["wait"], key=lambda w: w["id"]):
        if w["kind"] not in RULES["kinds"]:
            continue
        c = w["pts"].mean(0)
        sl, lat = proj(Px, c[None])
        sl, lat = float(sl[0]), float(lat[0])
        se, le = proj(Px, w["pts"][[0, -1]])
        hp = np.interp(sl, S, np.unwrap(np.arctan2(*np.gradient(P, axis=0).T[::-1])))
        q = w["pts"][-1] - w["pts"][0]
        perp = abs(np.sin(np.arctan2(q[1], q[0]) - hp)) >= np.sin(np.radians(RULES["line_angle"]))
        crosses = le.min() <= RULES["line_cross"] and le.max() >= -RULES["line_cross"]      # endpoints straddle the path (with margin)
        if abs(lat) > RULES["line_lat"] or sl < 5.0 or not (perp and crosses):
            continue
        if any(abs(sl - x) < 3.0 for x in seen):             # one event per crossing (duplicate lines of one lane)
            continue
        seen.append(sl)
        ev = dict(scene=sc["name"], line=w["id"], kind=w["kind"], s_line=sl, t_log0=T[0], t_log1=T[-1])
        ssign = bool(len(stopsigns) and np.hypot(*(stopsigns - c).T).min() < RULES["sign_m"])
        st_hit = [(a, b) for a, b in stops if RULES["stop_ahead"][0] <= sl - S[a] <= RULES["stop_ahead"][1]]
        if st_hit:
            a, b = st_hit[0]
            lead = False
            for t in np.linspace(T[a], min(T[b], T[a] + 2e6), 5):
                for o in sc["objs"].values():
                    if o["cls"] not in VEH or t < o["T"][0] or t > o["T"][-1]:
                        continue
                    p = np.array([np.interp(t, o["T"], o["P"][:, 0]), np.interp(t, o["T"], o["P"][:, 1])])
                    so, lo = proj(Px, p[None])
                    if 0 < so[0] - S[a] <= RULES["lead_m"] and abs(lo[0]) < RULES["lead_lat"]:
                        lead = True
            ev.update(t_stop=T[a], t_go=T[b], wait_s=(T[b] - T[a]) * 1e-6, s_stop=float(S[a]), ends_stopped=bool(b >= len(T) - 2))
            if lead:
                out.append({**ev, "label": "drop", "why": "lead"})
            else:
                out.append({**ev, "label": "stop_sign" if ssign else "red", "why": "", "w0": T[a] - RULES["win_s"] * 1e6,
                            "w1": T[b] - RULES["red_tail_s"] * 1e6})
            continue
        if ssign:
            out.append({**ev, "label": "drop", "why": "stop_sign_no_stop"})
            continue
        if S[-1] < sl + 1.0:
            out.append({**ev, "label": "drop", "why": "not_reached"})
            continue
        m = (S > sl - 15.0) & (S < sl)
        tc = float(np.interp(sl, S, T))
        if m.any() and v[m].min() >= RULES["go_min_v"]:
            out.append({**ev, "label": "green", "why": "", "t_cross": tc, "v_cross": float(np.interp(tc, T, v)),
                        "w0": tc - RULES["win_s"] * 1e6, "w1": tc})
        else:
            out.append({**ev, "label": "drop", "why": "slow_no_stop"})
    return out, Px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = pickle.load(open(a.facts, "rb"))
    E, paths = [], {}
    for name, sc in F["scenes"].items():
        ev, Px = events({**sc, "name": name})
        E += ev
        paths[name] = Px
    E = pd.DataFrame(E)
    rows = []
    for d in F["dumps"]:
        es = E[(E.scene == d["scene"]) & E.label.isin(CLS)]
        if es.empty:
            continue
        s, lat = proj(paths[d["scene"]], d["pose"][None, :2])
        for _, e in es.iterrows():
            dd = e.s_line - s[0]
            if e.w0 <= d["t0"] <= e.w1 and abs(lat[0]) <= RULES["ego_lat"] and RULES["ego_d"][0] < dd <= RULES["ego_d"][1]:
                rows.append(dict(file=d["file"], scene=d["scene"], run=d["run"], log_follow=d["log_follow"], t0=d["t0"],
                                 cls=e.label, y=int(e.label != "green"), kind=e.kind, line=e.line, d_line=dd, v=d["v"], lat=lat[0],
                                 dt_event=(d["t0"] - (e.t_cross if e.label == "green" else e.t_stop)) * 1e-6))
    fr = pd.DataFrame(rows, columns=["file", "scene", "run", "log_follow", "t0", "cls", "y", "kind", "line", "d_line", "v", "lat", "dt_event"])
    fr = fr.drop_duplicates("file")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    E.to_csv(out / "events.csv", index=False)
    fr.to_csv(out / "frames.csv", index=False)
    print(E.groupby(["label", "why"]).size().to_string())
    print(E[E.label.isin(CLS)].groupby(["label", "kind"]).agg(n=("scene", "size"), scenes=("scene", "nunique")).to_string())
    print(fr.groupby(["log_follow", "cls"]).agg(n=("file", "size"), scenes=("scene", "nunique"), d=("d_line", "median"), v=("v", "median")).to_string())


if __name__ == "__main__":
    main()
