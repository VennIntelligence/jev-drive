#!/usr/bin/env python3
"""Score the Bench2Drive controller acceptance (todos/2026-09-25-closed-loop-infra-acceptance/b2d-controllers.md, rules
frozen before the runs). Reads the run dir of scripts/infra_ctl_accept.sh and writes per-route and per-arm tables.

    python scripts/infra_ctl_score.py $DATA_DIR/runs/infra-accept/b2d-ctl --out <dir>

Tracking is against expert a's rear-axle path (simulator truth), over ticks at v >= 1 m/s from the first tick to the
arm's own first collision or the end of the expert log, whichever is first:
  e_lat   distance of the arm's rear axle to the expert path (polyline)
  e_lon   arc position of that projection minus the expert's arc position at the same elapsed time (negative = behind)
  e_head  arm heading minus the expert heading at the projection
lat_ratio as in the Alpamayo diagnosis 7.3 (scripts/zeroshot_b2d_alp_speed.analyse windows). Pass: A1 e_lat median
<= 0.30 m and p95 <= 1.00 m; A2 |e_lon| median <= 2.0 m and p95 <= 8.0 m; A3 lat_ratio >= 0.7; A4 mean DS >= expert a
mean DS - 5 (or the a/b gap if larger); A5 no route stuck (blocked / route timeout) that expert a completed.
NumPy + standard library.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_b2d_alp_speed import analyse, collisions  # noqa: E402

ARMS = ["z2", "z5", "f2", "f5", "p1", "p2"]
STUCK = ("blocked", "timeout", "TickRuntime")


def last_attempt(run, rid):
    done = run / "done" / (rid + ".json")
    if not done.exists():
        return None
    return run / "attempts" / rid / str(json.loads(done.read_text())["attempt"])


def record(att):
    return json.loads((att / "results.json").read_text())["_checkpoint"]["records"][0]


def expert_track(att, rear):
    rows = [json.loads(line) for line in open(att / "expert.jsonl")]
    t = np.array([r["t"] for r in rows])
    yaw = np.unwrap(np.radians([r["yaw"] for r in rows]))
    xy = np.c_[[r["x"] for r in rows], [r["y"] for r in rows]] + rear * np.c_[np.cos(yaw), np.sin(yaw)]
    return t - t[0], xy, yaw, np.array([r["v"] for r in rows])


def expert_row(att, rear):
    rec = record(att)
    t, xy, yaw, v = expert_track(att, rear)
    cols = collisions(rec, t, v, np.c_[xy, yaw])
    return {"status": rec["status"], "DS": rec["scores"]["score_composed"], "RC": rec["scores"]["score_route"],
            "n_collisions": len(cols), "duration_game": rec["meta"].get("duration_game")}


def project(path_xy, s_path, pts, window=(-40, 400)):
    """Sequential projection of pts (N,2) onto the polyline: (arc position, signed distance left-positive, segment)."""
    seg = np.diff(path_xy, axis=0)
    L2 = np.maximum((seg ** 2).sum(1), 1e-9)
    out, i0 = [], 0
    for p in pts:
        a, b = max(i0 + window[0], 0), min(i0 + window[1], len(seg))
        d = p - path_xy[a:b]
        u = np.clip((d * seg[a:b]).sum(1) / L2[a:b], 0, 1)
        foot = path_xy[a:b] + u[:, None] * seg[a:b]
        dist = np.hypot(*(p - foot).T)
        k = int(np.argmin(dist))
        j = a + k
        cross = seg[j, 0] * (p[1] - path_xy[j, 1]) - seg[j, 1] * (p[0] - path_xy[j, 0])
        out.append((s_path[j] + u[k] * math.sqrt(L2[j]), dist[k] * np.sign(cross), j))
        i0 = j
    return np.array(out)


def tracking(att, expert, rear_offset):
    te, xe, yawe, _ = expert
    s_e = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xe, axis=0).T))]
    ticks = [json.loads(line) for line in open(att / "ticks.jsonl")]
    t = np.array([x["t"] for x in ticks])
    t = t - t[0]
    v = np.array([x["v"] for x in ticks])
    truth = np.array([x["truth"] for x in ticks])
    rec = record(att)
    cols = collisions(rec, t, v, truth)
    t_end = min(cols[0][0] if cols else math.inf, te[-1])
    keep = (t <= t_end) & (v >= 1.0)
    if keep.sum() < 5:
        return None
    pr = project(xe, s_e, truth[keep, :2])
    s_ref = np.interp(t[keep], te, s_e)
    head = np.degrees(np.angle(np.exp(1j * (truth[keep, 2] - yawe[pr[:, 2].astype(int)]))))
    return {"e_lat": np.abs(pr[:, 1]), "e_lon": pr[:, 0] - s_ref, "e_head": np.abs(head), "n": int(keep.sum())}


def stuck_cause(att, row):
    """Descriptive only (added after the runs, not a criterion): why a stuck route stopped for good.
    gap_stop  the car stands still while the plan's first point (0.25 s) is >= 1 m ahead and the rest of the plan is
              (nearly) stationary: a controller without position feedback (Zoo PID reads speed from waypoint spacing)
              stops short of a stop point ahead; the replay plan then waits for a car that never arrives
    pinned    the car's last motion ended within 3 s after a collision (collision then standstill)
    other     anything else"""
    ticks = [json.loads(line) for line in open(att / "ticks.jsonl")]
    plans = [json.loads(line) for line in open(att / "plans.jsonl")]
    v = np.array([x["v"] for x in ticks])
    t = np.array([x["t"] for x in ticks])
    moving = np.flatnonzero(v > 0.5)
    t_last = t[moving[-1]] if len(moving) else t[0]
    tc = row["t_col"]                       # sim time, as t
    if tc is not None and tc <= t_last + 0.5 and t_last - tc <= 3.0:
        return "pinned"
    late = [p for p in plans if p["t"] > t_last + 2.0]
    if late:
        path = np.array(late[-1]["path"])
        if path[0, 0] >= 1.0 and abs(path[-1, 0] - path[1, 0]) < 0.5:
            return "gap_stop"
    return "other"


def q(x, p):
    return round(float(np.percentile(x, p)), 3) if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rear-offset", type=float, default=None, help="default: controller_config.json")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rear = a.rear_offset
    if rear is None:
        cfg = json.loads((Path(__file__).resolve().parents[1] /
                          "todos/2026-09-22-b2d-controller/results/controller_config.json").read_text())
        rear = float(cfg["rear_axle_offset_m"])
    routes = sorted(p.stem for p in (run / "expert-logs").glob("*.jsonl"))
    exp = {}
    for tag in ("a", "b"):
        for rid in routes:
            att = last_attempt(run / ("expert-" + tag), rid)
            if att is not None and (att / "results.json").exists():
                exp[tag, rid] = expert_row(att, rear)
    tracks = {rid: expert_track(last_attempt(run / "expert-a", rid), rear) for rid in routes}
    rows, summary = [], {}
    for tag in ("a", "b"):
        rr = [exp[tag, r] for r in routes if (tag, r) in exp]
        summary["E-" + tag] = {"n_routes": len(rr), "DS": round(float(np.mean([r["DS"] for r in rr])), 2) if rr else None,
                               "RC": round(float(np.mean([r["RC"] for r in rr])), 2) if rr else None,
                               "completed": sum(r["status"] == "Completed" for r in rr),
                               "collision_routes": sum(r["n_collisions"] > 0 for r in rr)}
        for rid in routes:
            if (tag, rid) in exp:
                rows.append(dict(arm="E-" + tag, route=rid, **exp[tag, rid]))
    ab = [abs(exp["a", r]["DS"] - exp["b", r]["DS"]) for r in routes if ("a", r) in exp and ("b", r) in exp]
    gap = abs(summary["E-a"]["DS"] - summary["E-b"]["DS"]) if summary["E-b"]["DS"] is not None else 0.0
    tol = max(5.0, gap)
    for arm in ARMS:
        adir = run / ("arm-" + arm)
        if not adir.exists():
            continue
        E = {k: [] for k in ("e_lat", "e_lon", "e_head")}
        wins_all, arows = [], []
        for rid in routes:
            att = last_attempt(adir, rid)
            if att is None or not (att / "ticks.jsonl").exists():
                arows.append({"arm": arm, "route": rid, "status": "missing"})
                continue
            row, wins = analyse(att, arm, {})
            wins_all += wins
            tr = tracking(att, tracks[rid], rear)
            ea = exp.get(("a", rid), {})
            r = {"arm": arm, "route": rid, "status": row["status"], "DS": row["DS"], "RC": row["RC"],
                 "n_collisions": row["n_collisions"], "t_col": row["t_col"], "col_type": row["col_type"],
                 "pinned_s": row["pinned_s"], "expert_status": ea.get("status"), "expert_DS": ea.get("DS"),
                 "extra_collision": int(row["n_collisions"] > 0 and ea.get("n_collisions", 0) == 0),
                 "stuck": int(any(s in row["status"] for s in STUCK)),
                 "stuck_vs_expert": int(any(s in row["status"] for s in STUCK) and ea.get("status") == "Completed")}
            if r["stuck"]:
                r["stuck_cause"] = stuck_cause(att, row)
            if tr is not None:
                for k in E:
                    E[k].append(tr[k])
                r.update(n_track=tr["n"], e_lat_med=q(tr["e_lat"], 50), e_lat_p95=q(tr["e_lat"], 95),
                         e_lon_med=q(tr["e_lon"], 50), abs_e_lon_med=q(np.abs(tr["e_lon"]), 50),
                         abs_e_lon_p95=q(np.abs(tr["e_lon"]), 95), e_head_med=q(tr["e_head"], 50))
            arows.append(r)
        rows += arows
        E = {k: np.concatenate(v) if v else np.array([]) for k, v in E.items()}
        L = [w for w in wins_all if w["pre_col"] and w["v0"] >= 2.0 and abs(w["y2"]) >= 1.0]
        lat = float(np.median([w["y2_drove"] / w["y2"] for w in L])) if L else None
        ok = [r for r in arows if r["status"] != "missing"]
        ds = float(np.mean([r["DS"] for r in ok])) if ok else None
        m = {"n_routes": len(ok), "missing": len(arows) - len(ok),
             "DS": round(ds, 2) if ds is not None else None,
             "RC": round(float(np.mean([r["RC"] for r in ok])), 2) if ok else None,
             "completed": sum(r["status"] == "Completed" for r in ok),
             "stuck": sum(r["stuck"] for r in ok), "stuck_vs_expert": sum(r["stuck_vs_expert"] for r in ok),
             "extra_collision_routes": sum(r["extra_collision"] for r in ok),
             "stuck_causes": {c: sum(r.get("stuck_cause") == c for r in ok) for c in ("gap_stop", "pinned", "other")},
             "e_lat_med": q(E["e_lat"], 50), "e_lat_p95": q(E["e_lat"], 95),
             "abs_e_lon_med": q(np.abs(E["e_lon"]), 50), "abs_e_lon_p95": q(np.abs(E["e_lon"]), 95),
             "e_lon_med": q(E["e_lon"], 50), "e_head_med": q(E["e_head"], 50), "n_track_ticks": int(len(E["e_lat"])),
             "lat_ratio": None if lat is None else round(lat, 3), "lat_n": len(L)}
        m["A1"] = m["e_lat_med"] is not None and m["e_lat_med"] <= 0.30 and m["e_lat_p95"] <= 1.00
        m["A2"] = m["abs_e_lon_med"] is not None and m["abs_e_lon_med"] <= 2.0 and m["abs_e_lon_p95"] <= 8.0
        m["A3"] = m["lat_ratio"] is not None and m["lat_ratio"] >= 0.7
        m["A4"] = ds is not None and ds >= summary["E-a"]["DS"] - tol
        m["A5"] = m["stuck_vs_expert"] == 0
        m["pass"] = all(m[k] for k in ("A1", "A2", "A3", "A4", "A5"))
        summary[arm] = m
    summary["_meta"] = {"routes": routes, "A4_tolerance": tol, "expert_ab_route_abs_ds_mean":
                        round(float(np.mean(ab)), 2) if ab else None, "rear_axle_offset_m": rear}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "per_route.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "_meta"}, indent=1))


if __name__ == "__main__":
    main()
