#!/usr/bin/env python3
"""Turning school P2, branch ceiling (experiments/route_follow/plans/2026-10-13-p2-branch-prereg.md): the ego follows the LOG up to a
handover decision k and the policy drives from k on (experiments/onpol_pai/lib/onp_log_driver.py, ONP_SWITCH), k at the junction entry
-40 / -20 / -10 / 0 m along the log. Does the policy complete a turn when it is handed the right entry state at the right speed?

  plan  (Tokyo, inside alpasim-base:0.89.0)  rf_branch.py plan --root <a PAI read with these scenes> --scenes <ids file> --out <dir>
        Per scene, from the logged rig path: the turn (largest heading change within 30 m of arc, >= 60 deg; turning-school-checks.md 1),
        the entry (first point of the first 30 m window that reaches 60 deg departing 5 deg from the window start), the exit (first point
        within 5 deg of the largest heading change from the entry inside entry + 40 m), the target (exit + 10 m), and per arm the decision
        k = round(10 * t_log(entry - d)) - OFF (t_log from the first pose of the log; OFF the decision-0 offset measured in the smoke,
        --off, default 0). k < 17 hands over at the end of the dev preset's force-GT (k_eff 17 = the plain policy run). Writes
        <dir>/plan.json, <dir>/scenes.txt and <dir>/sw_d<d>.txt (ONP_SWITCH files).
  read  (Tokyo, same image)  rf_branch.py read --plan <plan.json> --arm d40=<onp_pass2 out root> ... [--arm k17=<pai_eval root>] --out <json>
        Per rollout (runs/*/sim of each root): score, zero type and time (rf_tax.classify's rule: the earliest flag that zeroed it), the
        ego's station along the log (windowed projection, monotone), the route (AlpaSim RouteGenerator MAP at the ego's 0.5 s poses, as
        rf_tax.extract), and
          done     at some step before its zero event the ego's station is at or past the target and it is within ROUTE_TOL m of the route
          d_exit   the ego's distance to the log when it first reaches the exit station before its zero event (None: never)
          s_short  target station minus the furthest station reached (> 0: did not get there)
          ho_off   the ego's offset from the log at the handover decision (drive.jsonl `log_now` of record k_eff)
  table (Mac or Tokyo, plain python3)  rf_branch.py table --read <json> [--md <md>]   per-scene table (4 branches), ceiling, completion by arm.

Turning school P3 (plans/2026-10-14-p3-branch-prereg.md): arms `<handover>@<branch>` (rf_branch.sh: ONP_BRANCH, the plan transform for
--win decisions from k_eff). `read` then also writes the state at the window end (decision k_eff + --win, default 30): w_off / w_lat (offset
from the log, lateral signed, drive.jsonl `log_now`), w_v (fed speed), w_droute (distance to the route at that time).
  cmp   rf_branch.py cmp <root A> <root B>   per scene: served plans of every decision and controller tracks of two runs (S1 gate);
        with --branch also the raw and transformed plan at k_eff and the decisions with the transform on (S2 gate)
  p3    rf_branch.py p3 --read <json> [--md <md>]   per handover a scene x transform table, the ceiling over all cells, secondaries.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

ARMS, WIN, THR, DEP, EXIT_WIN, EXIT_TOL, TGT, ROUTE_TOL, FGT_K, FAR = (40, 20, 10, 0), 30, 60.0, 5.0, 40, 5.0, 10.0, 2.5, 17, 10.0


def arc_profile(P, Y, T):
    """Logged path -> 1 m grid: station s, heading (deg, unwrapped), time from the first pose (s)."""
    import numpy as np
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    d, Y, T = d[keep], Y[keep], T[keep]
    s = np.arange(0, d[-1], 1.0)
    return s, np.degrees(np.interp(s, d, np.unwrap(Y))), (np.interp(s, d, T) - T[0]) * 1e-6


def turn_points(s, y, t):
    import numpy as np
    m = t <= 20.2                                            # the rollout window (turning-school-checks.md 1)
    yy, n = y[m], int(m.sum())
    best = (0.0, 0)
    for i in range(n):
        seg = yy[i:i + WIN + 1] - yy[i]
        j = int(np.argmax(np.abs(seg)))
        if abs(seg[j]) > abs(best[0]):
            best = (float(seg[j]), i)
    if abs(best[0]) < THR:
        return None
    for i0 in range(n):
        seg = yy[i0:i0 + WIN + 1] - yy[i0]
        if np.abs(seg).max() >= THR:
            ent = i0 + int(np.argmax(np.abs(seg) > DEP))
            break
    seg = y[ent:ent + EXIT_WIN + 1] - y[ent]
    ex = ent + int(np.argmax(np.abs(seg) >= np.abs(seg).max() - EXIT_TOL))
    return dict(ang=best[0], side="L" if best[0] > 0 else "R", ent=float(s[ent]), exit=float(s[ex]), target=float(s[ex] + TGT),
                t_ent=float(t[ent]), t_target=float(np.interp(s[ex] + TGT, s, t)) if s[ex] + TGT <= s[-1] else None, log_len=float(s[-1]))


def plan(a):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rf_tax as X
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes, P = [x.split()[0] for x in open(a.scenes) if x.strip() and not x.startswith("#")], {}
    cfg = {}                                                 # scene -> the generated config of the run that rolled it (its loader knows it)
    for f in glob.glob(os.path.join(a.root, "runs", "*", "sim", "aggregate", "results-summary.json")):
        for r in json.load(open(f))["rollouts"]:
            cfg.setdefault(r["clipgt_id"], f.replace("aggregate/results-summary.json", "generated-user-config-0.yaml"))
    for sc in scenes:
        tr = X.loader(cfg[sc]).get_data_source(sc).rig.trajectory
        s, y, t = arc_profile(np.asarray(tr.positions, float)[:, :2], np.asarray(tr.yaws, float), np.asarray(tr.timestamps_us, float))
        tp = turn_points(s, y, t)
        if tp is None:
            raise SystemExit(f"{sc}: no >= {THR} deg turn in the window")
        tp["k"] = {f"d{d}": max(int(round(10 * float(np.interp(max(tp['ent'] - d, 0.0), s, t)))) - a.off, 0) for d in ARMS}
        tp["k_eff"] = {k: max(v, FGT_K) for k, v in tp["k"].items()}
        tp["v_log"] = {f"d{d}": float(np.interp(max(tp["ent"] - d, 0.0), s[1:], np.diff(s) / np.maximum(np.diff(t), 1e-3))) for d in ARMS}
        P[sc] = tp
        print(sc[7:15], tp["side"], round(tp["ang"]), "entry", tp["ent"], "exit", tp["exit"], "t_target", tp["t_target"], tp["k_eff"], flush=True)
    json.dump(dict(off=a.off, scenes=P), open(out / "plan.json", "w"), indent=1)
    (out / "scenes.txt").write_text("".join(f"{sc}\n" for sc in P))
    for d in ARMS:
        (out / f"sw_d{d}.txt").write_text("".join(f"{sc} {v['k'][f'd{d}']}\n" for sc, v in P.items()))
    print(len(P), "scenes ->", out)


def station(L, xy, s_hint=0.0):
    """Monotone windowed projection of the ego path xy (n, 2) on the log polyline L (m, 2) -> station (n,), distance (n,); the station
    only advances while the ego is within FAR m of the log."""
    import numpy as np
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(L, axis=0).T))]
    st, dist, cur = np.zeros(len(xy)), np.zeros(len(xy)), s_hint
    for i, p in enumerate(xy):
        m = np.where((S >= cur - 5) & (S <= cur + 30))[0]
        m = np.arange(max(m.min() - 1, 0), min(m.max() + 2, len(L))) if len(m) else np.arange(len(L))
        a, b = L[m[:-1]], L[m[1:]]
        ab = b - a
        l2 = np.maximum((ab ** 2).sum(1), 1e-12)
        u = np.clip(((p - a) * ab).sum(1) / l2, 0, 1)
        q = a + u[:, None] * ab
        dd = np.hypot(*(p - q).T)
        j = int(dd.argmin())
        dist[i] = dd[j]
        if dd[j] <= FAR:                                     # off the log the station is held: no creep along the window
            cur = max(cur, S[m[j]] + u[j] * np.sqrt(l2[j]))
        st[i] = cur
    return st, dist


def handover(root, win=30):
    """{scene: handover facts} from the driver record at the effective switch decision: offset from the log, log time, fed speed; and
    the same at the window end k_eff + win (w_*; w_now = its time for the route distance)."""
    out = {}
    for f in glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")):
        for line in open(f):
            if '"switch"' not in line or '"kind": "drive"' not in line:
                continue
            r = json.loads(line)
            sw = r.get("switch")
            if sw is not None and r["k"] == max(sw, FGT_K) and "log_now" in r:
                out.setdefault(r["scene"], {}).update(ho_off=round(math.hypot(*r["log_now"][:2]), 2), ho_log_t=r.get("log_t"),
                                       ho_v=round(10.0 * math.hypot(*r["ego"][4:6]), 2) if "ego" in r else None)
            if sw is not None and r["k"] == max(sw, FGT_K) + win and "log_now" in r:
                out.setdefault(r["scene"], {}).update(w_off=round(math.hypot(*r["log_now"][:2]), 2), w_lat=round(r["log_now"][1], 2),
                                                      w_v=round(10.0 * math.hypot(*r["ego"][4:6]), 2) if "ego" in r else None, w_now=r["now"])
    return out


def read(a):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rf_tax as X
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    PL = json.load(open(a.plan))["scenes"]
    rows = []
    for spec in a.arm:
        arm, root = spec.split("=", 1)
        ho = handover(root, a.win)
        for sim in sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))):
            f = os.path.join(sim, "aggregate", "results-summary.json")
            if not os.path.exists(f):
                continue
            cfg = os.path.join(sim, "generated-user-config-0.yaml")
            for r in json.load(open(f))["rollouts"]:
                sc = r["clipgt_id"]
                if sc not in PL:
                    continue
                p, m, score = PL[sc], r.get("metrics") or {}, float(r.get("score") or 0)
                C, (ser, _) = X.ctrl(sim, r["rollout_id"]), X.series(sim, sc, r["rollout_id"])
                row = dict(arm=arm, scene=sc, side=p["side"], score=score, k=p["k_eff"].get(re.split("[+@]", arm)[0]), rid=r["rollout_id"], **ho.get(sc, {}))
                if C is None or ser is None:
                    rows.append(dict(row, done=None, note="no controller trace / metrics"))
                    continue
                tf = {"collision": np.nanmin([X.first_pos(ser, "collision_front"), X.first_pos(ser, "collision_lateral")]),
                      "offroad": X.first_pos(ser, "offroad"), "corridor": X.first_pos(ser, "left_corridor_laterally")}
                flag = {"collision": m.get("collision_at_fault"), "offroad": m.get("offroad"), "corridor": m.get("left_corridor_laterally")}
                c = {k: v for k, v in tf.items() if float(flag[k] or 0) > 0 and np.isfinite(v)} if score == 0 else {}
                zt = (min(c, key=c.get) if c else "other") if score == 0 else ""
                t_zero = c[zt] if c else np.inf
                ds = X.loader(cfg).get_data_source(sc)
                tr = ds.rig.trajectory
                Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)
                L = X.dedupe(Pg[:, :2])
                xy = np.c_[C["x"], C["y"]]
                st, dl = station(L, xy)
                g = RouteGenerator.create(Pg, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
                try:
                    for i in range(min(4, len(C["t"]) - 1), len(C["t"]), 5):
                        z = float(np.interp(C["t"][i], Tg, Pg[:, 2]))
                        g.generate_route(int(C["t"][i]), Pose(np.array([C["x"][i], C["y"][i], z]),
                                                              np.array([0.0, 0.0, np.sin(C["yaw"][i] / 2), np.cos(C["yaw"][i] / 2)])))
                    R, rerr = X.dedupe(np.asarray(g.route_polyline_in_local.points, float)[:, :2]), 0
                except Exception:  # noqa: BLE001
                    R, rerr = L, 1
                dr = X.poly_proj(R, xy)[0]
                if row.get("w_now") is not None:
                    iw = int(np.argmin(np.abs(np.asarray(C["t"], float) - row["w_now"])))
                    row.update(w_droute=round(float(dr[iw]), 2), w_zeroed=bool(C["t"][iw] >= t_zero))
                ok = C["t"] < t_zero
                hit = np.where((st >= p["target"]) & ok & (dr <= ROUTE_TOL))[0]
                ix = np.where((st >= p["exit"]) & ok)[0]
                row.update(zero=zt, t_zero=(t_zero - C["t"][0]) * 1e-6 if np.isfinite(t_zero) else None, route_err=rerr,
                           done=bool(len(hit)), d_route_tgt=float(dr[hit[0]]) if len(hit) else None,
                           t_tgt=float((C["t"][hit[0]] - C["t"][0]) * 1e-6) if len(hit) else None,
                           d_exit=float(dl[ix[0]]) if len(ix) else None, s_short=float(p["target"] - st[C["t"] < t_zero].max()),
                           end=("zero" if np.isfinite(t_zero) else "rear" if float(m.get("collision_rear") or 0) > 0 else "end"))
                rows.append(row)
                print(arm, sc[7:15], row["score"], row["zero"], row["done"], row["d_exit"], flush=True)
    json.dump(dict(plan=a.plan, rows=rows), open(a.out, "w"), indent=1, default=float)
    print(len(rows), "rollouts ->", a.out)


def table(a):
    D = json.load(open(a.read))
    by = {}
    for r in D["rows"]:
        by.setdefault(r["scene"], {})[r["arm"]] = r
    arms = [f"d{d}" for d in ARMS]
    extra = sorted({r["arm"] for r in D["rows"]} - set(arms))
    def f(r):
        if r is None:
            return "-"
        v = "**done**" if r.get("done") else (r.get("zero") or r.get("end", "?"))
        return v + (f" (k {r['k']}, d_exit {r['d_exit']:.1f})" if r.get("d_exit") is not None else f" (k {r.get('k')}, short {r.get('s_short', 0):.0f} m)")
    L = ["| scene | side | " + " | ".join(arms + extra) + " | any |", "|:--|:--|" + "--:|" * (len(arms) + len(extra) + 1)]
    ceil = 0
    for sc, a_ in sorted(by.items(), key=lambda x: (x[1][next(iter(x[1]))]["side"], x[0])):
        anyd = any(a_.get(x, {}).get("done") for x in arms)
        ceil += anyd
        L.append(f"| {sc[7:15]} | {next(iter(a_.values()))['side']} | " + " | ".join(f(a_.get(x)) for x in arms + extra) + f" | {'yes' if anyd else 'no'} |")
    n = len(by)
    L += ["", f"ceiling (>= 1 completing branch of {', '.join(arms)}): {ceil} / {n} = {100 * ceil / max(n, 1):.0f} %", ""]
    for x in arms + extra:
        rr = [r[x] for r in by.values() if x in r]
        L.append(f"- {x}: completed {sum(bool(r.get('done')) for r in rr)} / {len(rr)}, zeros {sum(r['score'] == 0 for r in rr)}, "
                 f"mean score {sum(r['score'] for r in rr) / max(len(rr), 1):.3f}")
    s = "\n".join(L)
    print(s)
    if a.md:
        Path(a.md).write_text(s + "\n")


def drive_recs(root):
    """{scene: {k: drive record}} of every run under root."""
    R = {}
    for f in glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")):
        for line in open(f):
            if '"kind": "drive"' in line:
                r = json.loads(line)
                R.setdefault(r["scene"], {}).setdefault(r["k"], r)
    return R


def tracks(root):
    """{scene: (score, (n, 2) controller x, y)}."""
    import csv
    import numpy as np
    out = {}
    for f in glob.glob(os.path.join(root, "runs", "*", "sim", "aggregate", "results-summary.json")):
        for r in json.load(open(f))["rollouts"]:
            c = list(csv.DictReader(open(f.replace("aggregate/results-summary.json", f"controller/alpasim_controller_{r['rollout_id']}.csv"))))
            out[r["clipgt_id"]] = (r["score"], np.array([[float(x["x"]), float(x["y"])] for x in c]))
    return out


def cmp(a):
    """S1: the two runs' served plans (`poses_branch` where present, else `poses`) at every decision and their tracks, per scene.
    S2 (--branch): raw vs transformed plan at k_eff and the decisions with the transform on."""
    import numpy as np
    A, B, TA, TB = drive_recs(a.a), drive_recs(a.b), tracks(a.a), tracks(a.b)
    srv = lambda r: np.array(r.get("poses_branch", r["poses"]))
    for sc in sorted(set(A) & set(B)):
        ks = sorted(k for k in set(A[sc]) & set(B[sc]) if "poses" in A[sc][k] and "poses" in B[sc][k])
        d = np.array([np.abs(srv(A[sc][k]) - srv(B[sc][k])).max() for k in ks])
        (sa, ta), (sb, tb) = TA.get(sc, (None, np.zeros((0, 2)))), TB.get(sc, (None, np.zeros((0, 2))))
        n = min(len(ta), len(tb))
        e = np.hypot(*(ta[:n] - tb[:n]).T) if n else np.array([np.nan])
        first = next((k for k, x in zip(ks, d) if x > 0), None)
        print(f"{sc[7:15]}: decisions {len(ks)}, plan max|diff| {d.max():.6f}, first differing k {first}; track rows {len(ta)}/{len(tb)}, "
              f"pos diff median {np.median(e):.4f} max {np.max(e):.3f}; score {sa} / {sb}")
        if a.branch:
            on = [k for k in sorted(A[sc]) if (A[sc][k].get("branch") or {}).get("on")]
            if on:
                r = A[sc][on[0]]
                print(f"  branch {r['branch']}: on at k {on[0]}..{on[-1]} ({len(on)} decisions), poses_branch on all of them: "
                      f"{all('poses_branch' in A[sc][k] for k in on)}, off records with poses_branch: "
                      f"{sum('poses_branch' in x for k, x in A[sc].items() if k not in on)}")
                print("  raw   ", np.round(r["poses"], 2).tolist())
                print("  served", np.round(r["poses_branch"], 2).tolist())
                k1 = on[-1] + 1
                if k1 in A[sc] and k1 in B[sc]:
                    print(f"  k {k1} (window over): raw plan == served plan: {'poses_branch' not in A[sc][k1]}")


TRANSFORMS = ["g1s1", "g1.25s1", "g1.5s1", "g2s1", "g1s0.8", "g1s0.6", "g1.5s0.8", "g1.5s0.6"]


def p3(a):
    """P3 tables: per handover a scene x transform outcome table; ceiling over all cells; per completing cell what it needed; window-end state."""
    D = json.load(open(a.read))
    by = {(r["scene"], *r["arm"].split("@")): r for r in D["rows"]}
    hos = [h for h in ("d10", "d0") if any(k[1] == h for k in by)]
    tfs = [t for t in TRANSFORMS if any(k[2] == t for k in by if len(k) > 2)]
    scenes = sorted({k[0] for k in by}, key=lambda sc: (by[next(k for k in by if k[0] == sc)]["side"], sc))
    ab = {"collision": "coll", "offroad": "offr", "corridor": "corr", "other": "oth", "": ""}
    def f(r):
        if r is None:
            return "-"
        if r.get("done"):
            return "**done**"
        return ab.get(r.get("zero") or "", r.get("zero")) or r.get("end", "?")
    L, done_sc = [], {}
    for h in hos:
        L += [f"### handover {h}", "", "| scene | side | " + " | ".join(tfs) + " |", "|:--|:--|" + "--:|" * len(tfs)]
        for sc in scenes:
            rr = [by.get((sc, h, t)) for t in tfs]
            side = next(r["side"] for r in rr if r)
            L.append(f"| {sc[7:15]} | {side} | " + " | ".join(f(r) for r in rr) + " |")
            for t, r in zip(tfs, rr):
                if r and r.get("done"):
                    done_sc.setdefault(sc, []).append(f"{h}@{t}")
        L.append("")
        for t in tfs:
            rr = [by[(sc, h, t)] for sc in scenes if (sc, h, t) in by]
            L.append(f"- {h}@{t}: completed {sum(bool(r.get('done')) for r in rr)} / {len(rr)}, zeros {sum(r['score'] == 0 for r in rr)}, "
                     f"mean score {sum(r['score'] for r in rr) / max(len(rr), 1):.3f}")
        L.append("")
    n = len(scenes)
    idc = {sc for sc, c in done_sc.items() if any(x.endswith("@g1s1") for x in c)}
    L += [f"ceiling (>= 1 completing cell over {len(hos)} handovers x {len(tfs)} transforms): {len(done_sc)} / {n}; identity only: {len(idc)} / {n}", ""]
    need = lambda c: "identity" if c == "g1s1" else "both" if not c.startswith("g1s") and not c.endswith("s1") else "curvature" if c.endswith("s1") else "speed"
    L += ["| scene | completing cells | needed |", "|:--|:--|:--|"]
    for sc in scenes:
        c = done_sc.get(sc, [])
        L.append(f"| {sc[7:15]} | {', '.join(c) or '-'} | {(lambda u: 'none (identity completes)' if 'identity' in u else ', '.join(sorted(u)) or '-')({need(x.split('@')[1]) for x in c})} |")
    L += ["", "window end (k_eff + win): offset from the log / lateral (+ = ego right of the log), distance to the route, fed speed; mean over",
          "completing vs failing cells of each transform", "", "| cell | n done | w_off done | w_droute done | w_v done | w_off fail | w_droute fail | w_v fail |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    import statistics as st
    m = lambda v: f"{st.mean(v):.2f}" if v else "-"
    for h in hos:
        for t in tfs:
            rr = [by[(sc, h, t)] for sc in scenes if (sc, h, t) in by]
            dd, ff = [r for r in rr if r.get("done")], [r for r in rr if not r.get("done")]
            g = lambda R_, k: [r[k] for r in R_ if r.get(k) is not None]
            L.append(f"| {h}@{t} | {len(dd)} | {m(g(dd, 'w_off'))} | {m(g(dd, 'w_droute'))} | {m(g(dd, 'w_v'))} | {m(g(ff, 'w_off'))} | "
                     f"{m(g(ff, 'w_droute'))} | {m(g(ff, 'w_v'))} |")
    s = "\n".join(L)
    print(s)
    if a.md:
        Path(a.md).write_text(s + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--root", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--off", type=int, default=0)
    p = sp.add_parser("read")
    p.add_argument("--plan", required=True)
    p.add_argument("--arm", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--win", type=int, default=30)
    p = sp.add_parser("table")
    p.add_argument("--read", required=True)
    p.add_argument("--md")
    p = sp.add_parser("cmp")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--branch", action="store_true")
    p = sp.add_parser("p3")
    p.add_argument("--read", required=True)
    p.add_argument("--md")
    a = ap.parse_args()
    {"plan": plan, "read": read, "table": table, "cmp": cmp, "p3": p3}[a.cmd](a)
