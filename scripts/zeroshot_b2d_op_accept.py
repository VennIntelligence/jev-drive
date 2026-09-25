#!/usr/bin/env python3
"""Adapter acceptance of the openpilot + TCP-partner Bench2Drive agent (docs/zeroshot-adapters.md; criteria frozen in
todos/2026-09-24-zeroshot-exam/openpilot-migration.md, section D3, before the run). One arm = one run dir of
scripts/zeroshot_b2d_op.sh accept; the last attempt of every route counts.

    python3 zeroshot_b2d_op_accept.py f1=<run dir> f1f2b=<run dir> --zoo <Bench2DriveZoo checkout> --out accept.json

Gating items (checklist numbers in brackets):
  A1 [infra]   every route finished (harness status), plans logged
  A2 [2]       reference point: model driving, v < 0.3 m/s, plan in force moves < 1 m in 3 s -> throttle share <= 5 %
  A3 [3]       sensor timing: road and wide of every plan from one frame; plan frames exactly plan_every apart (>= 99 %)
  A4 [4]       warm-up: the model's control has zero weight before its first post-warm-up plan
  A5 [5]       standstill start: the ego moves (v > 0.5 m/s) within 20 s of the route start on every route
  A6 [6]       lane keeping: no outside-lane or route-deviation infraction under the model or a blend
  A7 [7]       junction turns: every reached LEFT / RIGHT route turn passed; at least one of each reached
  A8 [8]       red light / stop sign: none run while the control is being blended between drivers (a handover
               artifact); under the model or the partner: reported as that driver's behaviour
  A9 [9]       resume: no standstill >= 45 s after the first move
  A10 [1]      plan frame / heading, model driving at v > 2 m/s: median |lateral error @2 s| <= 1.0 m; lateral sign
               agrees with the truth on >= 80 % of plans with |truth y@2s| > 0.5 m; yaw sign agrees on >= 80 % of plans
               with |truth dyaw@2s| > 0.05 rad (vacuous below 5 plans)
  A11 [10]     controller identity: scripts/b2d_zoo_pid.py equals the Zoo uniad/vad team_code/pid_controller.py
Standard library + NumPy.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_b2d_junctions import LEFT, RIGHT, attempt_summary, events  # noqa: E402

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
ZOO_PID_COMMIT = "498c1f7"


def jl(p):
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def last_attempts(run):
    out = []
    for rdir in sorted((Path(run) / "attempts").iterdir()):
        atts = sorted((a for a in rdir.iterdir() if a.name.isdigit()), key=lambda a: int(a.name))
        if atts:
            out.append(atts[-1])
    return out


def route_checks(att):
    ticks, plans = jl(att / "ticks.jsonl"), jl(att / "plans.jsonl")
    cfg = json.loads((att / "agent_summary.json").read_text()).get("config") or {} if (att / "agent_summary.json").exists() else {}
    every = int(cfg.get("plan_every", 4))
    t = np.array([x["t"] for x in ticks])
    v = np.array([x["v"] for x in ticks])
    w = np.array([x.get("w_model", 1.0) for x in ticks])
    thr = np.array([x["throttle"] for x in ticks])
    r = {"route": att.parent.name}
    # A2: plan in force at each tick
    pt = np.array([p["t"] for p in plans])
    x3 = np.array([np.interp(3.0, np.arange(1, 21) * 0.25, np.asarray(p["path"])[:, 0]) for p in plans])
    warm = np.array([bool(p.get("warmup")) for p in plans])
    k = np.searchsorted(pt, t, side="right") - 1
    sel = (k >= 0) & (w >= 1.0) & (v < 0.3)
    sel[sel] &= (x3[k[sel]] < 1.0) & ~warm[k[sel]]
    r["a2_n"], r["a2_throttle_ticks"] = int(sel.sum()), int((thr[sel] > 0).sum())
    # A3
    same = [len(set(p["cam_frames"].values())) == 1 for p in plans]
    fr = np.array([p["frame"] for p in plans])
    r["a3_same_frame"], r["a3_spacing_ok"] = float(np.mean(same)), float(np.mean(np.diff(fr) == every)) if len(fr) > 1 else 1.0
    # A4
    first_live = pt[~warm][0] if (~warm).any() else np.inf
    r["a4_weight_before_live"] = float(w[t < first_live].max()) if (t < first_live).any() else 0.0
    r["a4_warm_accepted"] = int(sum(bool(p.get("warmup")) and bool(p.get("accepted")) for p in plans))
    # A5, A9
    mv = np.flatnonzero(v > 0.5)
    r["a5_start_s"] = round(float(t[mv[0]] - t[0]), 1) if len(mv) else None
    longest = 0.0
    if len(mv):
        st = (v < 0.1) & (np.arange(len(v)) > mv[0])
        e = np.flatnonzero(np.diff(np.r_[0, st.astype(int), 0]))
        longest = max([(b - a) * 0.05 for a, b in zip(e[::2], e[1::2])] or [0.0])
    r["a9_longest_stop_s"] = round(longest, 1)
    # A6, A8 from the per-driver infraction attribution
    s = attempt_summary(att)
    by = json.loads(s["infractions_by_driver"])
    r["infractions_by_driver"] = by
    r["a6_lane"] = sum(k in ("outside_route_lanes", "route_dev") for who in ("model", "blend") for k in by.get(who, []))
    r["a8_blend"] = sum(k in ("red_light", "stop_infraction") for k in by.get("blend", []))
    r["a8_model"] = sum(k in ("red_light", "stop_infraction") for k in by.get("model", []))
    r["a8_partner"] = sum(k in ("red_light", "stop_infraction") for k in by.get("partner", []))
    r.update(ds=s["ds"], rc=s["rc"], dist_share_model=s["dist_share_model"], time_share_model=s["time_share_model"])
    # A7
    tr = np.array([x["truth"][:2] for x in ticks if "truth" in x])
    turns = []
    if (att / "route.json").exists() and len(tr):
        for c, a, b in events(json.loads((att / "route.json").read_text())):
            if c not in (LEFT, RIGHT):
                continue
            da = np.linalg.norm(tr - a, axis=1)
            if da.min() > 5.0:
                continue
            passed = bool(np.linalg.norm(tr[int(np.argmin(da)):] - b, axis=1).min() <= 3.0)
            turns.append(("left" if c == LEFT else "right", passed))
    r["a7_turns"] = turns
    # A10: plan (rear frame, x forward, y left) vs the truth rear-axle track 2 s later, while the model drives
    tt = np.array([x["t"] for x in ticks if "truth" in x])
    tru = np.array([x["truth"] for x in ticks if "truth" in x])
    lat, sgn, ysg = [], [], []
    if len(tt):
        yaw = np.unwrap(tru[:, 2])
        for p in plans:
            t0 = p["t"]
            if p.get("warmup") or p["speed"] <= 2.0 or t0 + 2.0 > tt[-1] or t0 < tt[0]:
                continue
            win = (t >= t0) & (t <= t0 + 2.0)
            if not win.any() or w[win].min() < 1.0:
                continue
            x0, y0, h0 = (np.interp(t0, tt, c) for c in (tru[:, 0], tru[:, 1], yaw))
            x2, y2, h2 = (np.interp(t0 + 2.0, tt, c) for c in (tru[:, 0], tru[:, 1], yaw))
            dx, dy = x2 - x0, y2 - y0
            gy = dx * np.sin(h0) - dy * np.cos(h0)                     # left of the t0 heading
            py = float(np.interp(2.0, np.arange(1, 21) * 0.25, np.asarray(p["path"])[:, 1]))
            lat.append(abs(py - gy))
            if abs(gy) > 0.5:
                sgn.append(np.sign(py) == np.sign(gy))
            if p.get("plan_yaw") is not None and abs(h2 - h0) > 0.05:  # both right-positive
                ysg.append(np.sign(np.interp(2.0, T_IDXS, np.asarray(p["plan_yaw"]))) == np.sign(h2 - h0))
    r["a10"] = dict(n=len(lat), lat_med=round(float(np.median(lat)), 3) if lat else None,
                    sign_n=len(sgn), sign=round(float(np.mean(sgn)), 3) if sgn else None,
                    yaw_n=len(ysg), yaw=round(float(np.mean(ysg)), 3) if ysg else None)
    return r


def arm(run):
    rows, infra = [], []
    for att in last_attempts(run):
        rr = json.loads((att / "route_result.json").read_text()) if (att / "route_result.json").exists() else {}
        if rr.get("status") != "finished" or not (att / "plans.jsonl").exists() or not (att / "results.json").exists():
            infra.append("%s: %s" % (att.parent.name, rr.get("status")))
            continue
        rows.append(route_checks(att))
    a2n, a2t = sum(r["a2_n"] for r in rows), sum(r["a2_throttle_ticks"] for r in rows)
    turns = [x for r in rows for x in r["a7_turns"]]
    lat = [r["a10"] for r in rows]
    n10 = sum(x["n"] for x in lat)
    wmean = lambda key, nkey: (sum((x[key] or 0) * x[nkey] for x in lat) / max(sum(x[nkey] for x in lat), 1))  # noqa: E731
    lat_med = float(np.median([x["lat_med"] for x in lat if x["lat_med"] is not None])) if n10 else None
    g = {
        "A1": not infra and len(rows) > 0,
        "A2": a2n == 0 or a2t / a2n <= 0.05,
        "A3": all(r["a3_same_frame"] == 1.0 and r["a3_spacing_ok"] >= 0.99 for r in rows),
        "A4": all(r["a4_weight_before_live"] == 0.0 and r["a4_warm_accepted"] == 0 for r in rows),
        "A5": all(r["a5_start_s"] is not None and r["a5_start_s"] <= 20.0 for r in rows),
        "A6": sum(r["a6_lane"] for r in rows) == 0,
        "A7": all(p for _, p in turns) and {"left", "right"} <= {d for d, _ in turns},
        "A8": sum(r["a8_blend"] for r in rows) == 0,
        "A9": all(r["a9_longest_stop_s"] < 45.0 for r in rows),
        "A10": n10 < 5 or (lat_med <= 1.0 and wmean("sign", "sign_n") >= 0.8 and wmean("yaw", "yaw_n") >= 0.8),
    }
    return {"gates": g, "pass": all(g.values()), "infra": infra, "a2": [a2t, a2n],
            "turns": turns, "a10": dict(n=n10, lat_med=lat_med, sign=round(wmean("sign", "sign_n"), 3),
                                        yaw=round(wmean("yaw", "yaw_n"), 3)),
            "red_stop_by_driver": {d: sum(r["a8_" + d] for r in rows) for d in ("model", "partner", "blend")},
            "routes": rows}


def controller_identity(zoo):
    ours = hashlib.sha256(Path(__file__).with_name("b2d_zoo_pid.py").read_bytes()).hexdigest()
    try:
        ship = subprocess.run(["git", "-C", zoo, "show", ZOO_PID_COMMIT + ":team_code/pid_controller.py"],
                              capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        return {"ok": None, "note": "shipped file unavailable: %s" % exc}
    return {"ok": hashlib.sha256(ship).hexdigest() == ours, "ours": ours[:12], "shipped": hashlib.sha256(ship).hexdigest()[:12]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=run dir")
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"A11_controller_identity": controller_identity(a.zoo)}
    for spec in a.arms:
        name, _, path = spec.partition("=")
        res[name] = arm(path)
        res[name]["pass"] = res[name]["pass"] and res["A11_controller_identity"]["ok"] is not False
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    for name in (s.partition("=")[0] for s in a.arms):
        print(name, "PASS" if res[name]["pass"] else "FAIL", json.dumps(res[name]["gates"]), "turns", res[name]["turns"],
              "a10", res[name]["a10"], "a2", res[name]["a2"])
    print("controller identity", res["A11_controller_identity"])
    sys.exit(0 if any(res[s.partition("=")[0]]["pass"] for s in a.arms) else 1)


if __name__ == "__main__":
    main()
