#!/usr/bin/env python3
"""PAI-HOLD-DIAG: would the nuPlan serve-side static hold (lib/static_hold.py, JEV_HOLD=leash3 / stop3; decisions 350 / 351) fix the
PAI ACC-GAP zeros (decision 330)? CPU only; stored rollouts are read, never written. Note: results/pai_hold_diag.md.

  $P pai_hold_diag.py extract --runs D [D ...] --rows ROWS.csv.gz --out F.csv.gz [--workers N]
        per rollout of the run dirs (<D>/sim): the human's rig track (the scorer's GT) and the ego's controller trace ->
        the handoff state as the hold sees it (speed at the 1.6 s call and at the call 0.1 s before it, i.e. static_hold.step's inputs
        on the PAI clock serve_fix.ACC_T_HAND), the leash3 / stop3 latch, the human's displacement / end state / stop time, and at the
        corridor trip (first metric step with d >= 4 m) the ego's speed, its signed offset past the human's rest point and the human
        path's turn around the trip point. ROWS = pai_corridor's per-rollout scorer rows (score, fail, corridor flag)
  pai_hold_diag.py report --facts F.csv.gz --out F.md          classes, upper bound, launch risk (Mac or box)

$P = AlpaSim's venv python (rf_tax.loader needs alpasim_runtime). The latch rules are static_hold.py's constants, copied here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T0_FGT, T_CALL, DT_CALL = 1.7, 1.6, 0.1          # s: PAI force-GT end; serve_fix.ACC_T_HAND (the hold's handoff call); PAI call spacing
V_LATCH, A_STOP, T_STOP, LEASH, CORR = 0.3, 0.5, 3.0, 3.0, 4.0   # static_hold.py: V_LATCH, A_STOP, T_STOP; leash3 / stop3 m; scorer corridor
V_STILL, V_FAST, TURN_DEG = 0.3, 20.0, 30.0       # human "stopped"; "high speed" (d328's drift zeros were 27-38 m/s); turn around the trip


def _speed(T, P, t, h=0.1):
    x = lambda tt: np.array([np.interp(tt, T, P[:, 0]), np.interp(tt, T, P[:, 1])])   # noqa: E731
    return float(np.hypot(*(x(t + h / 2) - x(t - h / 2))) / h)


def _job(job):
    sys.path.insert(0, str(HERE))
    import polars as pl
    import rf_tax as TX
    cfg, items = job
    L, out = TX.loader(cfg), []
    for sim, s, rid, row in items:
        tr = L.get_data_source(s).rig.trajectory
        T, P, Y = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
        m = pl.read_parquet(f"{sim}/rollouts/{s}/{rid}/metrics.parquet").select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
        ser = {n: np.array(sorted(zip(g["timestamps_us"].to_list(), [np.nan if v is None else v for v in g["values"].to_list()])), float)
               for (n,), g in m.group_by("name")}
        d = ser["dist_to_gt_trajectory"]
        t0 = d[0, 0]
        C = TX.ctrl(sim, rid)
        r = dict(id=s, rid=rid, **row)
        # the hold's handoff inputs: force-GT makes the ego's measured speed the human's
        tc = t0 + T_CALL * 1e6
        v, vp = _speed(T, P, tc), _speed(T, P, tc - DT_CALL * 1e6)
        a = (v - vp) / DT_CALL
        v5, a5 = _speed(T, P, tc, 0.5), (_speed(T, P, tc, 0.5) - _speed(T, P, tc - 0.5e6, 0.5)) / 0.5
        xy0 = np.array([np.interp(tc, T, P[:, 0]), np.interp(tc, T, P[:, 1])])
        leash = v < V_LATCH
        stop = (not leash) and a < -A_STOP and v / -a <= T_STOP
        cap = LEASH if leash else (LEASH + v * v / (-2 * a) if stop else np.nan)
        # the human after the handoff
        aft = T >= tc
        disp = np.hypot(*(P[aft] - xy0).T)
        seg = np.hypot(*np.diff(P, axis=0).T)
        vh = np.r_[seg / np.maximum(np.diff(T) / 1e6, 1e-3), 0.0]
        mov = np.flatnonzero(vh > V_STILL)
        t_stop = (T[mov[-1] + 1] - t0) / 1e6 if len(mov) and mov[-1] + 1 < len(T) else np.nan   # human still from here to the end
        rest = P[-1]
        r.update(v_call=v, v_prev=vp, a_call=a, v_call5=v5, a_call5=a5, latch_leash=leash, latch_stop=stop, cap=cap,
                 h_disp_max=float(disp.max()) if len(disp) else 0.0, h_rest_from_xy0=float(np.hypot(*(rest - xy0))),
                 h_v_end=_speed(T, P, T[-1] - 0.1e6, 0.2), h_t_stop=t_stop, h_seg_last_mm=float(seg[-1] * 1e3), gt_len=float(seg.sum()),
                 h_still_end=bool(seg[-1] * 1e3 < 1), h_yaw_tot=float(np.degrees(abs(Y[-1] - Y[0]))))
        # the corridor trip
        k = np.flatnonzero(d[:, 1] >= CORR)
        if len(k) and C is not None:
            tt = d[k[0], 0]
            i = int(np.argmin(abs(C["t"] - tt)))
            ex = np.array([C["x"][i], C["y"][i]])
            hy = Y[-1]
            j = int(np.argmin(np.hypot(*(P - ex).T)))                      # nearest GT point to the ego at the trip
            win = (T >= T[j] - 3e6) & (T <= T[j] + 3e6)
            lat = ser.get("lateral_dist_to_gt_trajectory")
            r.update(t_trip=(tt - t0) / 1e6, ego_v_trip=float(np.hypot(C["vx"][i], C["vy"][i])),
                     ego_past_rest=float((ex - rest) @ [np.cos(hy), np.sin(hy)]), ego_rest_dist=float(np.hypot(*(ex - rest))),
                     gt_frac_trip=float(j / max(len(P) - 1, 1)), turn_deg=float(np.degrees(np.ptp(Y[win]))) if win.any() else 0.0,
                     ego_yaw_vs_gt=float(np.degrees(abs((C["yaw"][i] - Y[j] + np.pi) % (2 * np.pi) - np.pi))),
                     lat_trip=float(lat[k[0], 1]) if lat is not None else np.nan, h_v_at_trip=_speed(T, P, tt, 0.2))
            if not np.isnan(t_stop):                                       # the ego when the human came to rest
                ts = t0 + t_stop * 1e6
                q = int(np.argmin(abs(C["t"] - ts)))
                r.update(ego_v_at_hstop=float(np.hypot(C["vx"][q], C["vy"][q])),
                         ego_gap_at_hstop=float((rest - np.array([C["x"][q], C["y"][q]])) @ [np.cos(hy), np.sin(hy)]))
        if C is not None:                                                  # ego after the handoff
            sel = C["t"] >= t0 + T0_FGT * 1e6
            r.update(ego_arc_after=float(np.hypot(*np.diff(np.c_[C["x"][sel], C["y"][sel]], axis=0).T).sum()) if sel.sum() > 1 else 0.0,
                     ego_maxd_xy0=float(np.hypot(C["x"][sel] - xy0[0], C["y"][sel] - xy0[1]).max()) if sel.any() else 0.0)
        out.append(r)
    return out


def cmd_extract(a):
    import pandas as pd
    from multiprocessing import Pool
    rows = pd.read_csv(a.rows)
    jobs = {}
    for _, x in rows.iterrows():
        sim = Path(x.run)
        rid = [p.name for p in (sim / "rollouts" / x.id).iterdir() if p.is_dir()]
        if not rid:
            continue
        row = dict(score=float(x.score), fail=x.fail if isinstance(x.fail, str) else "", corridor=int(x.corridor))
        jobs.setdefault(str(sim / "generated-user-config-0.yaml"), []).append((str(sim), x.id, rid[0], row))
    J = [(c, it[i::6]) for c, it in jobs.items() for i in range(6)]
    with Pool(a.workers) as p:
        R = [r for rr in p.map(_job, J) for r in rr]
    pd.DataFrame(R).to_csv(a.out, index=False)
    print(json.dumps(dict(rollouts=len(R))))


def classify(x) -> str:
    if x.fail not in ("left_corridor_laterally",):
        return "-"
    if x.h_still_end:
        if x.latch_leash:
            return "A still from handoff to end (leash3 latch)"
        if x.latch_stop:
            return "B braking to a stop at handoff (stop3 latch)"
        return "C human stops mid-scene, ego rolls past (no latch)"
    if x.turn_deg >= TURN_DEG:
        return "D turn / turn exit"
    if x.ego_v_trip >= V_FAST:
        return "E high-speed lateral drift"
    return "F other (moving human, mid-path, < 20 m/s, straight)"


def cmd_report(a):
    import pandas as pd
    F = pd.read_csv(a.facts)
    F["fail"] = F.fail.fillna("")
    F["cls"] = F.apply(classify, axis=1)
    n = len(F)
    z = F[F.score == 0]
    L = ["# PAI-HOLD-DIAG per-rollout tables (ACC-GAP, decision 330 stage 2)", "",
         f"Rollouts {n}; zeros {len(z)}; by failure: " + ", ".join(f"{k} {v}" for k, v in z.fail.replace('', 'none').value_counts().items()), "",
         "## Corridor zeros by class", "", "| class | n | median ego v at trip (m/s) | median human v at handoff | median ego past rest (m) |",
         "|---|--:|--:|--:|--:|"]
    c = z[z.fail == "left_corridor_laterally"]
    for k, g in c.groupby("cls"):
        L.append(f"| {k} | {len(g)} | {g.ego_v_trip.median():.1f} | {g.v_call.median():.1f} | {g.ego_past_rest.median():+.1f} |")
    cols = ["id", "score", "fail", "cls", "v_call", "a_call", "a_call5", "latch_leash", "latch_stop", "cap", "h_disp_max", "h_rest_from_xy0",
            "h_v_end", "h_t_stop", "h_still_end", "gt_len", "t_trip", "ego_v_trip", "ego_past_rest", "ego_v_at_hstop", "ego_gap_at_hstop",
            "turn_deg", "gt_frac_trip", "ego_maxd_xy0"]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else (v[7:15] if isinstance(v, str) and v.startswith("clipgt-") else str(v))  # noqa: E731
    for title, g in (("Corridor zeros", c.sort_values(["cls", "id"])), ("Offroad zeros", z[z.fail == "offroad"].sort_values("id")),
                     ("Other zeros", z[~z.fail.isin(["left_corridor_laterally", "offroad"])].sort_values("id")),
                     ("Every latched rollout (leash3 or stop3)", F[F.latch_leash | F.latch_stop].sort_values(["latch_leash", "id"]))):
        L += ["", f"## {title} ({len(g)})", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        L += ["| " + " | ".join(fmt(getattr(x, k)) for k in cols) + " |" for x in g.itertuples()]
    # offline hold estimate on latched rollouts: within cap of xy0 for the session
    L += ["", "## Offline hold estimate (latched rollouts only change)", "",
          "Held ego stays within `cap` of the handoff point xy0. Fix = the human is still at the end and its rest point is within cap + 4 m "
          "of xy0 (the ego can end inside the 4 m disc) -> score 1 if GT < 5 m else min(1, cap / (0.8 GT)) as an upper bound. Launch = the "
          "human later goes farther than cap from xy0 -> held score <= min(1, cap / (0.8 GT)) (progress bound, collisions not modelled).", "",
          "| rule | latched | human still to end | fix candidates (score 0 now) | gain | launches | of which scored > 0 now | cost | net / 281 |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, msk in (("leash3", F.latch_leash), ("stop3 extra", F.latch_stop), ("stop3 total", F.latch_leash | F.latch_stop)):
        g = F[msk]
        held = np.where(g.gt_len < 5, 1.0, np.minimum(1.0, g.cap / (0.8 * g.gt_len)))
        fixable = g.h_still_end & (g.h_rest_from_xy0 <= g.cap + CORR)
        launch = g.h_disp_max > g.cap
        gain = float(np.clip(held - g.score, 0, None)[fixable & (g.score == 0)].sum())
        cost = float(np.clip(g.score - held, 0, None)[launch].sum())
        L.append(f"| {name} | {len(g)} | {int(g.h_still_end.sum())} | {int((fixable & (g.score == 0)).sum())} | {gain:+.2f} | {int(launch.sum())} | "
                 f"{int((launch & (g.score > 0)).sum())} | {-cost:.2f} | {(gain - cost) / n:+.4f} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("extract")
    p.add_argument("--rows", required=True), p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=48)
    p.add_argument("--runs", nargs="*", default=[])
    p = sp.add_parser("report")
    p.add_argument("--facts", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(extract=cmd_extract, report=cmd_report)[a.cmd](a)
