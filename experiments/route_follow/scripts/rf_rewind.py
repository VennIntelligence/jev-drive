#!/usr/bin/env python3
"""RECOVERY-EXPERT REWIND (prereg experiments/route_follow/plans/2026-10-16-recovery-rewind-prereg.md; runner rf_rewind.sh; driver
experiments/onpol_pai/lib/onp_log_driver.py ONP_CELLS rewind cells). RSG-L18-s0 drives each PAI strict-turn scene from its start (box, gains 20,
JEV_VCONT=1.0 JEV_LEAD=1, JEV_TRACK=0); on the scenes where it fails the turn, the SAME session is replayed (box rollouts are bit-identical,
results/p4_self/box_det.md) and the rule expert (experiments/alpasim/lib/rule_expert.py, XP-L12-E46) takes over at decision k_fail - 10 x Delta,
Delta in 1 / 2 / 3 s, with the policy's own pose, speed, controller state and history; the expert drives to the rollout's end.

  cells   (plain python3)  rf_rewind.py cells --scene <id> --phase A|B --pack <expert pack dir> --out <dir> [--tfail <tfail.json>] [--smoke]
          one cells file per launch (one rollout each): A = `id` (the policy alone: the control) and `xp0` (the expert from decision 0: the
          reference); B = `xpD1` / `xpD2` / `xpD3` at h = k_fail - 10 Delta (only when the identity failed the turn and h >= 17, the end of
          the runtime's forced-log window); --smoke adds `idB` (a second identity session: box determinism). Prints the tags.
  extract (AlpaSim venv)   rf_rewind.py extract --sd <scene dir> --out <pkl>   per launch <sd>/<tag>/sim: rf_pai_follow.scene_job (map lanes,
          log, controller trace, zero event, rear contact, AlpaSim's MAP route rebuilt along the rollout) with the label = the cell tag, plus
          the session's decisions from drive.jsonl (k, now, t0, anchor, the model's plan, the served plan, expert on, measured speed).
  tfail   (plain python3)  rf_rewind.py tfail --pkl <pkl> --out <json>   the identity rollout's strict turn (rf_pai_follow.one): outcome;
          t_fail = the first zero-score event (rf_pai_follow's tz) for `event`, the first controller sample with route station >= entry - 20 m
          and |lateral| > 2.5 m for `off`, none otherwise (completed / short: no rewind); k_fail = the last decision with now <= t_fail.
  read    (plain python3)  rf_rewind.py read --root <run root> [--scenes ids] --out <json>   per scene and cell: rf_pai_follow.one (completed,
          outcome, zero kind), the handover state (t_h, ego speed / lateral / heading error to the route, the identity's at the same instant),
          the rewind check (served plans and controller track equal to the identity's before h), settle time (|lateral| <= 1 m held 1 s),
          speed at h + 3 s and at the end, rear contact after h, at-fault, the decision-293 shape gate on the post-handover track.
  table   (plain python3)  rf_rewind.py table --read <json> [--tokyo <turns.json>] --md <md> --csv <csv>   per Delta: expert vs policy, paired
          bootstrap, the registered lines.
  fig     (Mac .venv)      rf_rewind.py fig --pkl-dir <dir> --read <json> --scenes s8,s8,s8 --out <png>   BEV of recovered / failed paths.

nuPlan train 64 (prereg experiments/route_follow/plans/2026-10-16-rewind-train64-prereg.md; runner rf_rewind.sh train64; driver
experiments/alpasim/lib/sh30_driver.py with JEV_EXPERT + JEV_EXPERT_FROM): the held-read serving (rm_chain.sh readlist, 18 decisions at 2 Hz,
plain RSG-L18-s0) on the 64 strict turns of the nuPlan @18 train pool. Launches per stage: `id` (+ `idB` in the smoke), `xp0` (the expert from
the start, stratum C) and `D1` / `D2` / `D3` (the policy to h = k_ref - 2 Delta, the expert from h; A and B units share a launch).
  n-units (plain python3)  rf_rewind.py n-units --follow <scenes.csv> --label RSG-L18-s0@18 --out <txt>   the 64 turn scenes with the stored outcome
  n-tfail (AlpaSim venv)   rf_rewind.py n-tfail --runs <id run dirs, comma> --out <stage dir>   per turn scene of the identity: stratum A
          (event / off / short) k_fail, stratum B (completed) k_viol = the decision of the first decision-293 shape violation, h per Delta;
          writes tfail.json, from_D<d>.txt (`<scene> <h>`) and list_D<d>_c<i>.txt chunks
  n-read  (AlpaSim venv)   rf_rewind.py n-read --stage <stage dir> --out <json> [--pkl <dir>]   every rollout of the stage: rf_follow.one (turn,
          completed, outcome, k_z), the 10 Hz controller track on rf_follow's route, the rewind check, settle, creep, rear / at-fault, the gate
          on the post-handover track, B's before / after profile, the row yield; --pkl keeps the per-scene tracks (figures)
  n-table (plain python3)  rf_rewind.py n-table --read <json> --md <md> --csv <csv>   per stratum and Delta, the registered lines
  n-fig   (Mac .venv)      rf_rewind.py n-fig --pkl <scene pkl> --read <json> --out <png>   BEV of one unit (id, xp0, D1-D3)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
sys.path[:0] = [str(HERE)]
import rf_pai_follow as F  # noqa: E402

DELTAS, FGT_K, STEP = (1, 2, 3), 17, 100_000
PRE, SETTLE_LAT, SETTLE_S, WIN_END, V_CREEP = 20.0, 1.0, 1.0, 3.0, 2.0


def cell(tag, xp):
    return dict(tag=tag, src="rew", switch=0, D=None, g=1.0, lon="raw", xp=xp)


# ---------------------------------------------------------------- cells
def cells(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.phase == "A":
        cl = [cell("id", None), cell("xp0", 0)]
    else:
        T = json.loads(Path(a.tfail).read_text())
        cl = [cell(f"xpD{d}", T["h"][str(d)]) for d in DELTAS if T.get("h", {}).get(str(d)) is not None]
        if a.smoke:
            cl.append(cell("idB", None))
    for c in cl:
        (out / f"cells_{c['tag']}.json").write_text(json.dumps(dict(xp_pack=a.pack, scenes={}, cells={a.scene: [c]})))
    print(" ".join(c["tag"] for c in cl))


# ---------------------------------------------------------------- extract (AlpaSim venv)
def drive_table(f):
    """drive.jsonl -> {session: [decision dict]} (infer records only)."""
    out = {}
    for line in open(f):
        if '"infer": true' not in line:
            continue
        r = json.loads(line)
        p = r.get("p4") or {}
        out.setdefault(r["session"], []).append(dict(
            k=r["k"], now=r["now"], t0=r["t0"], anchor=r["anchor"], poses=r["poses"], served=r.get("poses_branch", r["poses"]),
            on=bool(p.get("lat")), v=p.get("v_meas"), tag=p.get("tag"), xp=p.get("xp"), expert=p.get("expert")))
    return {k: sorted(v, key=lambda d: d["k"]) for k, v in out.items()}


def extract(a):
    sd = Path(a.sd)
    scene = (sd / "scene.txt").read_text().split()[0]
    rolls, drv, cfg = [], {}, None
    for sim in sorted(glob.glob(str(sd / "*" / "sim"))):
        f = Path(sim) / "aggregate" / "results-summary.json"
        dj = Path(sim) / "driver-logs" / "drive.jsonl"
        if not f.exists() or not dj.exists():
            continue
        cfg = cfg or str(Path(sim) / "generated-user-config-0.yaml")
        D = drive_table(dj)
        for r in json.loads(f.read_text())["rollouts"]:
            dec = D.get(r["rollout_id"], [])
            if not dec or r["clipgt_id"] != scene:
                continue
            tag = dec[0]["tag"]
            rolls.append(dict(label=tag, sim=sim, rid=r["rollout_id"], score=r.get("score"), metrics=r.get("metrics") or {},
                              dec=[(d["t0"], d["k"], None) for d in dec]))
            drv[tag] = dec
    if not rolls:
        raise SystemExit(f"{sd}: no rollouts")
    out = F.scene_job((scene, cfg, rolls))
    for tag, o in out["rolls"].items():
        o["drv"] = drv[tag]
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(scene[7:15], sorted(out["rolls"]), "->", a.out)


# ---------------------------------------------------------------- shared geometry
def geo(o):
    """Rollout -> (route R, stations S, ctrl t, P, vx, yaw, st, lat)."""
    C = o["ctrl"]
    R = F.dedup(o["route"])
    S = F.stations(R)
    P = np.c_[C["x"], C["y"]]
    st, lat = F.track(R, S, P)
    return R, S, np.asarray(C["t"], float), P, np.asarray(C["vx"], float), np.asarray(C["yaw"], float), st, lat


def tz_of(o):
    return o["tz"] if o["tz"] is not None and not o["score"] else np.inf


def tfail_of(sc, o):
    """Identity rollout -> dict(outcome, t_fail, kind of t_fail, k_fail, h per Delta, turn)."""
    row, _, _ = F.one(sc, "id", o, zeros=False)
    res = dict(scene=sc["scene"], outcome=row.get("outcome"), turn=bool(row.get("turn")), entry=row.get("entry"), exit=row.get("exit"),
               t_fail=None, src=None, k_fail=None, h={str(d): None for d in DELTAS}, score=row["score"], kind=row["kind"])
    if not row.get("turn") or row["outcome"] not in ("event", "off"):
        return res
    R, S, t, P, vx, yaw, st, lat = geo(o)
    tz = tz_of(o)
    if row["outcome"] == "event":
        tf, src = tz, "zero"
    else:
        i = np.flatnonzero((st >= row["entry"] - PRE) & (np.abs(lat) > F.LAT_OK) & (t <= tz))
        if not len(i):
            return res
        tf, src = float(t[i[0]]), "off"
    ks = [d for d in o["drv"] if d["now"] <= tf]
    if not ks:
        return res
    kf = ks[-1]["k"]
    res.update(t_fail=float(tf), src=src, k_fail=int(kf), t_fail_rel_s=round((tf - o["drv"][0]["now"]) * 1e-6, 2),
               h={str(d): (int(kf - 10 * d) if kf - 10 * d >= FGT_K else None) for d in DELTAS})
    return res


def tfail(a):
    sc = pickle.load(open(a.pkl, "rb"))
    res = tfail_of(sc, sc["rolls"]["id"])
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


# ---------------------------------------------------------------- read
def interp(t, x, q):
    return float(np.interp(q, t, x)) if t[0] <= q <= t[-1] else np.nan


def heading_err(R, S, st, yaw, i):
    s, h = F.heading(R, S)
    return float(np.degrees((yaw[i] - np.interp(st[i], s, h) + np.pi) % (2 * np.pi) - np.pi))


def read_scene(sc, T):
    import rf_p4 as P4
    import rf_shape as SH
    J = json.loads((RES / "p3_branch" / "shape.json").read_text())
    topo = json.loads((RES / "turn_gallery" / "plan_strict36.json").read_text())["scenes"].get(sc["scene"], {}).get("topo", "other")
    gt = P4.gate_of(topo, J)
    ido = sc["rolls"].get("id")
    gi = geo(ido) if ido is not None and ido.get("ctrl") is not None else None
    rows = []
    for tag, o in sorted(sc["rolls"].items()):
        if o.get("ctrl") is None:
            rows.append(dict(scene=sc["scene"], tag=tag, missing=True))
            continue
        row, _, _ = F.one(sc, tag, o, zeros=False)
        R, S, t, P, vx, yaw, st, lat = geo(o)
        tz = tz_of(o)
        x = dict(scene=sc["scene"], tag=tag, missing=False, topo=topo, score=row["score"], kind=row["kind"], zflag=row["zflag"],
                 turn=bool(row.get("turn")), completed=bool(row.get("completed")), outcome=row.get("outcome"), entry=row.get("entry"),
                 exit=row.get("exit"), lat_cross=row.get("lat_cross"), at_fault=bool(row["kind"] == "collision"),
                 t_end_rel_s=round((t[-1] - t[0]) * 1e-6, 2))
        dec = o["drv"]
        h = next((d["k"] for d in dec if d["on"]), None) if tag.startswith("xp") else None
        x["h"] = h
        if h is not None:
            dh = next(d for d in dec if d["k"] == h)
            th = float(dh["now"])
            i = int(np.clip(np.searchsorted(t, th), 0, len(t) - 1))
            x.update(t_h=th, t_h_rel_s=round((th - dec[0]["now"]) * 1e-6, 2), v_h=interp(t, vx, th), v_meas_h=dh["v"], lat_h=float(lat[i]),
                     herr_h=heading_err(R, S, st, yaw, i), st_h_rel_entry=(float(st[i]) - row["entry"]) if row.get("entry") is not None else None,
                     v_trace=[interp(t, vx, th + q * 1e6) for q in (0.0, 0.25, 0.5, 0.75, 1.0)],
                     v_win_end=interp(t, vx, th + WIN_END * 1e6), v_end=float(vx[-1]),
                     rear_after=bool(o["trear"] is not None and o["trear"] >= th), zero_before_h=bool(np.isfinite(tz) and tz < th),
                     xp_stop_share=float(np.mean([d["expert"] is not None and d["expert"].get("stop") is not None for d in dec if d["on"]])))
            m = (t >= th) & (t < tz)
            ok = (np.abs(lat) <= SETTLE_LAT) & m
            run = np.convolve(ok.astype(int), np.ones(int(SETTLE_S * 10) + 1, int), "valid")
            j = np.flatnonzero(run == int(SETTLE_S * 10) + 1)
            x["settle_s"] = round((t[j[0]] - th) * 1e-6, 2) if len(j) else None
            if gi is not None:                                   # the rewind check against the identity session of the same scene
                Rd = {d["k"]: d for d in ido["drv"]}
                pre = [d for d in dec if d["k"] < h and d["k"] in Rd]
                x["pre_plan_maxdiff"] = max((float(np.max(np.abs(np.subtract(d["served"], Rd[d["k"]]["served"])))) for d in pre), default=None)
                x["pre_n"] = len(pre)
                ti, Pi, vxi, yawi = gi[2], gi[3], gi[4], gi[5]
                n = int(min(np.searchsorted(t, th, side="right"), np.searchsorted(ti, th, side="right")))
                x["pre_track_maxdiff"] = float(max(np.abs(P[:n] - Pi[:n]).max(), np.abs(vx[:n] - vxi[:n]).max(), np.abs(yaw[:n] - yawi[:n]).max())) if n else None
                x["pre_t_equal"] = bool(n and np.array_equal(t[:n], ti[:n]))
                x["id_v_h"] = interp(ti, vxi, th)
                x["id_v_trace"] = [interp(ti, vxi, th + q * 1e6) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
            if row.get("turn"):
                shp = SH.track_shape(t, P, vx, st, row["entry"] - SH.WIN_PRE, row["exit"], tz, t_lo=th)
                spd = SH.speeds(t, st, vx, row["entry"], row["exit"], tz)
                ent_after = x["st_h_rel_entry"] is not None and x["st_h_rel_entry"] < 0
                gk = dict(kmax=bool(shp["kmax"] <= gt["kmax"]), dk=bool(shp["dk"] <= gt["dk"]), alat=bool(shp["alat"] <= gt["alat"]),
                          v_entry=bool(not ent_after or (np.isfinite(spd["v_0"]) and spd["v_0"] <= gt["v_entry"])))
                x.update(kmax=shp["kmax"], dk=shp["dk"], alat=shp["alat"], v_0=spd["v_0"], gate=gk, shape_ok=bool(x["completed"] and all(gk.values())))
        rows.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in x.items()})
    return dict(scene=sc["scene"], tfail=T, rows=rows)


def read(a):
    out = []
    for sd in sorted(glob.glob(os.path.join(a.root, "clipgt-*"))):
        if a.scenes and Path(sd).name[7:15] not in a.scenes.split(","):
            continue
        f, tf = Path(sd) / "all.pkl", Path(sd) / "tfail.json"
        if not f.exists() or not tf.exists():
            continue
        out.append(read_scene(pickle.load(open(f, "rb")), json.loads(tf.read_text())))
    Path(a.out).write_text(json.dumps(dict(scenes=out), indent=1, default=float))
    print(len(out), "scenes ->", a.out)


# ---------------------------------------------------------------- table
def pct(b):
    return "-" if not b["n"] else f"{100 * b['rate']:.1f} [{100 * b['lo']:.1f}, {100 * b['hi']:.1f}]"


def table(a):
    X = json.loads(Path(a.read).read_text())["scenes"]
    by = {s["scene"]: {r["tag"]: r for r in s["rows"]} for s in X}
    T = {s["scene"]: s["tfail"] for s in X}
    fail = [s for s in by if T[s].get("t_fail") is not None]
    L = [f"# Recovery-expert rewind: RSG-L18-s0 failures on the PAI strict turns, expert XP-L12-E46 from the rewound state (box, JEV_TRACK 0)", "",
         f"Scenes read {len(by)}; identity outcome: " + ", ".join(f"{k} {v}" for k, v in sorted(_count(T[s]['outcome'] for s in by).items())) +
         f"; failed (rewound) {len(fail)} (t_fail from the zero event {sum(T[s]['src'] == 'zero' for s in fail)}, from leaving 2.5 m "
         f"{sum(T[s]['src'] == 'off' for s in fail)}).", ""]
    L += ["| Delta | n | expert completed | policy (identity) completed | paired diff [95 % CI] | expert settle <= 1 m (median s, n settled) | "
          "v(h + 3 s) >= 2 m/s | rear-ended after h | at-fault zero | shape gate (of completed) | lat at h (median m) | v at h (median m/s) |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    csv = ["scene,topo,id_outcome,t_fail_src,t_fail_rel_s,k_fail,delta,h,lat_h,herr_h,v_h,id_v_h,xp_completed,xp_outcome,xp_kind,settle_s,v_win_end,"
           "v_end,rear_after,at_fault,shape_ok,pre_plan_maxdiff,pre_track_maxdiff,xp0_completed"]
    summ = {}
    for d in DELTAS:
        U = [s for s in fail if f"xpD{d}" in by[s] and not by[s][f"xpD{d}"].get("missing")]
        xr = [by[s][f"xpD{d}"] for s in U]
        xc = np.array([r["completed"] for r in xr], float)
        pc = np.array([by[s]["id"]["completed"] for s in U], float)
        b, bd = F.boot(xc), F.boot(xc - pc)
        st_ = [r["settle_s"] for r in xr if r.get("settle_s") is not None]
        vw = [r["v_win_end"] for r in xr if np.isfinite(r.get("v_win_end", np.nan))]
        comp = [r for r in xr if r["completed"]]
        summ[d] = dict(n=len(U), xp=b, diff=bd, creep_ok=float(np.mean([v >= V_CREEP for v in vw])) if vw else np.nan,
                       shape=sum(r.get("shape_ok", False) for r in comp), n_comp=len(comp))
        L.append(f"| {d} s | {len(U)} | {int(xc.sum())} / {len(U)} = {pct(b)} % | {int(pc.sum())} / {len(U)} | {pct(bd)} pp | "
                 f"{np.median(st_) if st_ else float('nan'):.1f} ({len(st_)}) | {sum(v >= V_CREEP for v in vw)} / {len(vw)} | "
                 f"{sum(r['rear_after'] for r in xr)} | {sum(r['at_fault'] for r in xr)} | {summ[d]['shape']} / {len(comp)} | "
                 f"{np.median([abs(r['lat_h']) for r in xr]) if xr else float('nan'):.2f} | {np.median([r['v_h'] for r in xr]) if xr else float('nan'):.2f} |")
        for s in U:
            r, i, t = by[s][f"xpD{d}"], by[s]["id"], T[s]
            csv.append(",".join(str(v) for v in (s[7:15], r["topo"], t["outcome"], t["src"], t.get("t_fail_rel_s"), t["k_fail"], d, r["h"],
                                                   _f(r["lat_h"]), _f(r["herr_h"]), _f(r["v_h"]), _f(r.get("id_v_h")), int(r["completed"]),
                                                   r["outcome"], r["kind"], r.get("settle_s"), _f(r["v_win_end"]), _f(r["v_end"]), int(r["rear_after"]),
                                                   int(r["at_fault"]), int(r.get("shape_ok", False)), _f(r.get("pre_plan_maxdiff"), 6),
                                                   _f(r.get("pre_track_maxdiff"), 6), int(by[s].get("xp0", {}).get("completed", False)))))
    L += ["", "Exploratory (Amendment 1): expert outcome per Delta, and completion among units the expert was not rear-ended in", "",
          "| Delta | completed | event (collision / offroad / corridor) | off | short_rear | short_slow | completed / (n - short_rear) |", "|--:|--:|--:|--:|--:|--:|--:|"]
    for d in DELTAS:
        xr = [by[s][f"xpD{d}"] for s in fail if f"xpD{d}" in by[s] and not by[s][f"xpD{d}"].get("missing")]
        c = _count(r["outcome"] for r in xr)
        ev = _count(r["kind"] for r in xr if r["outcome"] == "event")
        nr = len(xr) - c.get("short_rear", 0)
        L.append(f"| {d} s | {c.get('completed', 0)} | {c.get('event', 0)} ({ev.get('collision', 0)} / {ev.get('offroad', 0)} / {ev.get('corridor', 0)}) | "
                 f"{c.get('off', 0)} | {c.get('short_rear', 0)} | {c.get('short_slow', 0)} | {c.get('completed', 0)} / {nr} |")
    L += ["", "Read: units = strict turns the identity (RSG-L18-s0 alone, the same box session) failed (`event` / `off`) whose handover "
          "h = k_fail - 10 Delta is >= 17. Policy control = the identity rollout (box sessions are bit-identical, so the policy handed back the "
          "rewound state reproduces it; checked per session below). CI: bootstrap over scenes (10 000)."]
    # reference: the expert from the start
    x0 = [s for s in by if "xp0" in by[s] and not by[s]["xp0"].get("missing")]
    f0 = [s for s in fail if s in x0]
    L += ["", "## Reference: the expert from the scene's start (xp0) and the policy alone (id)", "",
          f"- all read scenes: expert {sum(by[s]['xp0']['completed'] for s in x0)} / {len(x0)}, policy {sum(by[s]['id']['completed'] for s in x0)} / {len(x0)}",
          f"- on the rewound (failed) units: expert from start {sum(by[s]['xp0']['completed'] for s in f0)} / {len(f0)}"]
    both = [s for s in f0 if not by[s]["xp0"]["completed"]]
    L.append("- failed units the expert also fails from the start: " + (", ".join(f"{s[7:15]} ({by[s]['xp0']['outcome']}/{by[s]['xp0']['kind'] or '-'})" for s in both) or "none"))
    # rewind / velocity check
    chk = [(s, r) for s in fail for tg, r in by[s].items() if tg.startswith("xpD") and not r.get("missing")]
    pm = [r.get("pre_plan_maxdiff") for _, r in chk if r.get("pre_plan_maxdiff") is not None]
    tm = [r.get("pre_track_maxdiff") for _, r in chk if r.get("pre_track_maxdiff") is not None]
    dv = [abs(r["v_h"] - r["id_v_h"]) for _, r in chk if np.isfinite(r.get("id_v_h", np.nan))]
    L += ["", "## Rewind check (expert sessions against the identity session of the same scene, before h)", "",
          f"- sessions {len(chk)}; served plans before h bit-identical {sum(x == 0 for x in pm)} / {len(pm)} (max {max(pm, default=float('nan')):.2g});"
          f" controller track before h identical {sum(x == 0 for x in tm)} / {len(tm)} (max {max(tm, default=float('nan')):.2g}); "
          f"|v(h) expert - v(h) identity| max {max(dv, default=float('nan')):.3g} m/s"]
    rep = [(s, tg) for s in by for tg in ("D1", "D2", "D3") if tg in by[s] and by[s][tg].get("h") is None and "id" in by[s]]
    if rep:                                                  # D launches replay the identity on scenes without a handover at that Delta
        L.append(f"- identity replays inside the D launches (no handover at that Delta): score and outcome equal to id "
                 f"{sum(by[s][tg]['score'] == by[s]['id']['score'] and by[s][tg]['outcome'] == by[s]['id']['outcome'] for s, tg in rep)} / {len(rep)}")
    ib = [s for s in by if "idB" in by[s]]
    if ib:
        L.append(f"- smoke idB (second identity session): scores equal {sum(by[s]['idB']['score'] == by[s]['id']['score'] for s in ib)} / {len(ib)}")
    if a.tokyo:
        tk = {r["scene"]: r for r in json.loads(Path(a.tokyo).read_text())["rows"] if r["label"] == "rsg"}
        agree = [s for s in by if s in tk and not tk[s].get("missing")]
        L += ["", f"Box identity vs Tokyo r281/rsg outcome on {len(agree)} scenes: equal {sum(tk[s].get('outcome') == T[s]['outcome'] for s in agree)}; "
              + ", ".join(f"{s[7:15]} {tk[s].get('outcome')}->{T[s]['outcome']}" for s in agree if tk[s].get("outcome") != T[s]["outcome"])]
    # lines
    L += ["", "## Registered lines", ""]
    best = [d for d in DELTAS if summ[d]["n"] and summ[d]["xp"]["rate"] >= 0.6 and summ[d]["diff"]["rate"] >= 0.3]
    for d in DELTAS:
        s_ = summ[d]
        L.append(f"- Delta {d} s: expert {pct(s_['xp'])} % (line >= 60), paired {pct(s_['diff'])} pp (line >= +30); creep check "
                 f"{100 * s_['creep_ok']:.0f} % with v(h + 3 s) >= 2 m/s (line >= 80); shape gate {s_['shape']} / {s_['n_comp']} completed")
    L.append(f"- RECOVERS at Delta {', '.join(f'{d} s' for d in best)}" if best else "- no Delta passes both lines")
    Path(a.md).write_text("\n".join(L) + "\n")
    Path(a.csv).write_text("\n".join(csv) + "\n")
    Path(a.md).with_suffix(".json").write_text(json.dumps(dict(summary=summ, fail=fail), indent=1, default=float))
    print("\n".join(L))


def _count(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _f(x, n=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


# ---------------------------------------------------------------- fig
def fig(a):
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parents[2] / "research"))
    import plot_style as PS
    PS.apply()
    X = {s["scene"][7:15]: s for s in json.loads(Path(a.read).read_text())["scenes"]}
    pick = a.scenes.split(",")
    figr, axs = plt.subplots(1, len(pick), figsize=(6.875, 6.875 / len(pick) + 0.4))
    col = {"id": "#999999", "xp0": "#000000", "xpD1": "#E69F00", "xpD2": "#56B4E9", "xpD3": "#009E73"}
    for ax, s8 in zip(np.atleast_1d(axs), pick):
        sc = pickle.load(open(glob.glob(os.path.join(a.pkl_dir, f"clipgt-{s8}*", "all.pkl"))[0], "rb"))
        rows = {r["tag"]: r for r in X[s8]["rows"]}
        for lane in sc["lanes"].values():
            c = lane["c"]
            ax.plot(c[:, 0], c[:, 1], color="#dddddd", lw=0.4, zorder=0)
        ax.plot(*sc["log"]["P"].T, color="#0072B2", lw=1.0, ls="--", label="human log", zorder=1)
        lim = []
        for tag in ("id", "xp0", "xpD3", "xpD2", "xpD1"):
            o = sc["rolls"].get(tag)
            if o is None or o.get("ctrl") is None:
                continue
            C = o["ctrl"]
            tz = tz_of(o)
            m = np.asarray(C["t"]) <= tz
            r = rows[tag]
            lab = {"id": "policy (identity)", "xp0": "expert from start"}.get(tag, f"expert from -{tag[-1]} s")
            ax.plot(C["x"][m], C["y"][m], color=col[tag], lw=1.1, label=f"{lab}: {'done' if r['completed'] else r['outcome']}", zorder=2)
            if tag.startswith("xpD") and r.get("t_h") is not None:
                i = int(np.searchsorted(C["t"], r["t_h"]))
                ax.plot(C["x"][i], C["y"][i], "o", color=col[tag], ms=3, zorder=3)
            if not m.all():
                ax.plot(C["x"][m][-1], C["y"][m][-1], "x", color=col[tag], ms=4, zorder=3)
            lim.append(np.c_[C["x"][m], C["y"][m]])
        Q = np.concatenate(lim)
        c0, r0 = Q.mean(0), max(np.ptp(Q, 0).max() / 2 + 5, 25)
        ax.set_xlim(c0[0] - r0, c0[0] + r0)
        ax.set_ylim(c0[1] - r0, c0[1] + r0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])  # noqa: E702
        ax.text(0.02, 0.98, f"{s8} ({X[s8]['tfail']['outcome']})", transform=ax.transAxes, va="top", fontsize=7)
        ax.legend(fontsize=5.5, loc="lower right", frameon=False)
    figr.tight_layout()
    PS.save(figr, str(Path(a.out).with_suffix("")))
    print("->", a.out)


# ---------------------------------------------------------------- nuPlan train 64
N_STEP, N_KEND, N_HMIN, N_PRE, N_WIN, N_PAST, N_LAT = 2, 17, 1, 20.0, 40.0, 10.0, 2.5   # 2 Hz; last decision of the read; first served decision
N_TAGS = ("id", "idB", "xp0", "D1", "D2", "D3")
TOL_PLAN, TOL_TRACK, TOL_V = 0.15, 0.05, 0.05                      # Amendment 2: handover check against the identity (m, m, m/s)                   # (force_gt_duration 0.5 s = decision 0); rf_follow's junction pre; d293 window


def n_shape(t, xy, v, st, lo, hi, tz=np.inf, t_lo=-np.inf):
    """rf_shape.track_shape, NaN when fewer than two samples lie before tz (a zero at the start; rf_shape.resample fails on an empty track)."""
    import rf_shape as SH
    if (np.asarray(t, float) < tz).sum() < 2:
        return dict(kmax=np.nan, alat=np.nan, dk=np.nan, dec=np.nan, arc=0.0, st_kpk=np.nan, st_apk=np.nan, v_apk=np.nan)
    return SH.track_shape(t, xy, v, st, lo, hi, tz, t_lo=t_lo)


def n_gate():
    """Decision-293 thresholds for the nuPlan turns: no topo class here, so the all-turn human p95 (rf_p4.gate_of of a class below
    MIN_CLASS_N) and the default entry speed V_E_OTHER + 1 m/s."""
    import rf_p4 as P4
    return P4.gate_of("other", json.loads((RES / "p3_branch" / "shape.json").read_text()))


def n_units(a):
    import csv as C
    rows = [r for r in C.DictReader(open(a.follow)) if r["label"] == a.label and r["turn"] == "True"]
    Path(a.out).write_text("".join(f"{r['scene']} {r['outcome']} {r['kz']}\n" for r in sorted(rows, key=lambda r: r["scene"])))
    print(len(rows), "turn scenes ->", a.out)


def n_roll(job):
    """(tag, run dir, scene, cfg, want_log) -> one rollout: rf_follow's row and decisions, its route, the 10 Hz controller track on it,
    zero time / kind, rear contact, the driver records (served plan, expert flag), optionally the human log on the same route."""
    tag, D, scene, cfg, want_log = job
    import rf_follow as RF
    from ao_e1 import drive_records, loader
    import long_read as LR  # noqa: F401  (rf_follow.one imports it)
    recs = drive_records(D)[scene]
    summ = {r["clipgt_id"]: r for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]}[scene]
    cap = {}
    reg = RF.regen_full

    def regen(*x):
        cap["r"] = reg(*x)
        return cap["r"]
    RF.regen_full = regen
    try:
        row, dec = RF.one((tag, D, scene, recs, dict(score=summ.get("score"), **(summ.get("metrics") or {})), cfg, "none"))
    finally:
        RF.regen_full = reg
    R = F.dedup(cap["r"][3])
    S = F.stations(R)
    C = F.ctrl(D, summ["rollout_id"])
    o = dict(tag=tag, scene=scene, run=D, row=row, dec=dec, R=R, ctrl=C, rid=summ["rollout_id"],
             drv=[dict(k=int(r["k"]), now=int(r["now"]), t0=int(r["t0"]), served=r["poses"], on="expert" in r or "perturb" in r,
                       expert=r.get("expert"), perturb=r.get("perturb")) for r in recs])
    kz = row["kz"] if row["kz"] >= 0 else None
    o["tz"] = RF.T0 + RF.DT * kz if kz is not None else np.inf
    o["zkind"] = None
    if kz is not None:                                   # the zero's kind at k_z from the evaluator series (rf_follow's rule: atf or corridor)
        import polars as pl
        f = glob.glob(f"{D}/rollouts/{scene}/*/metrics.parquet")
        m = pl.read_parquet(f[0]).filter(pl.col("timestamps_us") == RF.T0 + RF.DT * kz).select("name", "values")
        v = {n: (x or 0.0) > 0 for n, x in zip(m["name"].to_list(), m["values"].to_list())}
        o["zkind"] = "collision" if v.get("collision_front") or v.get("collision_lateral") else "offroad" if v.get("offroad") else "corridor"
    o["trear"] = F.events(D, scene, summ["rollout_id"])[2]
    if C is not None:
        o["st"], o["lat"] = F.track(R, S, np.c_[C["x"], C["y"]])
    if want_log:
        tr = loader(str(cfg)).get_data_source(scene).rig.trajectory
        Tt, P = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2]
        v = np.r_[0.0, np.hypot(*np.diff(P, axis=0).T) / np.maximum(np.diff(Tt) * 1e-6, 1e-6)]
        v[0] = v[1] if len(v) > 1 else 0.0
        st, lat = F.track(R, S, P)
        o["log"] = dict(t=Tt, P=P, v=v, st=st)
    return o


def n_runs(stage, tag):
    return sorted(glob.glob(os.path.join(stage, tag, "c*", "run")))


def n_jobs(stage, tags, log_tags=("id",), dirs=None, only=None):
    """Rollout jobs; each read with its own run's generated config (its cache and window: route, log and map as the simulator had them)."""
    jobs = []
    for tag in tags:
        for D in (dirs if dirs is not None else n_runs(stage, tag)):
            f = Path(D) / "aggregate" / "results-summary.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text())["rollouts"]:
                if only is None or r["clipgt_id"] in only:
                    jobs.append((tag, D, r["clipgt_id"], str(Path(D) / "generated-user-config-0.yaml"), tag in log_tags))
    return jobs


def n_paths():
    R_ = HERE.parents[2]
    for d in ("lib", "experiments/route_mem/lib", "experiments/route_mem/scripts", "experiments/alpasim/scripts", "experiments/alpasim/lib",
              "experiments/anchor_offset/scripts", "experiments/eyes_vs_head/scripts", "experiments/alpasim_long/scripts"):
        if str(R_ / d) not in sys.path:
            sys.path.append(str(R_ / d))
    if str(R_) not in sys.path:
        sys.path.insert(0, str(R_))


def n_violation(o, gt):
    """Completed identity turn -> (t of the first decision-293 shape violation inside [entry - 40, exit] before the zero, which items) or
    (None, []). Per 10 Hz sample on rf_shape's 0.25 m arc grid: |kappa| > kmax, |dkappa/ds| > dk, v^2 |kappa| > alat; and the entry speed
    (box-smoothed controller speed where the track first crosses the entry) > v_entry, timed at that crossing."""
    import rf_shape as SH
    C, row = o["ctrl"], o["row"]
    t, P, st = np.asarray(C["t"], float), np.c_[C["x"], C["y"]], o["st"]
    v = SH.box5(C["vx"])
    lo, hi = row["entry"] - N_WIN, row["exit"]
    pre = t < o["tz"]
    Rr = SH.resample(P[pre], dict(t=t[pre], v=v[pre], st=st[pre])) if pre.sum() >= 2 else None
    hits = []
    if Rr is not None:
        s, g, E = Rr
        _, k, dk = SH.kappa_ops(s, g)
        m = (E["st"] >= lo) & (E["st"] <= hi)
        for name, x, thr in (("kmax", np.abs(k), gt["kmax"]), ("dk", np.abs(dk), gt["dk"]), ("alat", np.abs(E["v"] ** 2 * k), gt["alat"])):
            j = np.flatnonzero(m & np.isfinite(x) & (x > thr))
            if len(j):
                hits.append((float(E["t"][j[0]]), name))
    i = np.flatnonzero((st >= row["entry"]) & pre)
    if len(i) and F.speed_at(t, st, v, row["entry"], o["tz"]) > gt["v_entry"]:
        hits.append((float(t[i[0]]), "v_entry"))
    if not hits:
        return None, []
    tv = min(h[0] for h in hits)
    return tv, sorted({h[1] for h in hits if h[0] == tv})


def n_ref(o, gt):
    """Identity rollout -> the unit: stratum, k_ref (A: failure decision, B: violation decision), h per Delta (registered definitions)."""
    row, dec = o["row"], o["dec"]
    u = dict(scene=o["scene"], turn=bool(row.get("turn")), outcome=row.get("outcome"), kz=row["kz"], k_cross=row.get("k_cross"),
             entry=row.get("entry"), exit=row.get("exit"), side=row.get("side"), stratum=None, k_ref=None, src=None, items=[],
             h={str(d): None for d in DELTAS})
    if not row.get("turn"):
        return u
    oc = row["outcome"]
    if oc == "event":
        u.update(stratum="A", k_ref=int(row["kz"]), src="zero")
    elif oc == "off":
        kz = row["kz"] if row["kz"] >= 0 else 10 ** 6
        i = [d["k"] for d in dec if d["st"] >= row["entry"] - N_PRE and abs(d["lat"]) > N_LAT and d["k"] < kz]
        if i:
            u.update(stratum="A", k_ref=int(i[0]), src="off")
        else:
            u.update(stratum="A", k_ref=int(row["k_cross"]) if row.get("k_cross", -1) >= 0 else N_KEND, src="off_cross")
    elif oc == "short":
        u.update(stratum="A", k_ref=N_KEND, src="short_end")
    elif oc == "completed":
        u["stratum"] = "B"
        tv, items = n_violation(o, gt) if o["ctrl"] is not None else (None, [])
        if tv is not None:
            ks = [d["k"] for d in o["drv"] if d["t0"] <= tv]
            u.update(k_ref=int(ks[-1]) if ks else 0, src="viol", items=items, t_viol=tv)
        else:
            u["src"] = "clean"
    if u["k_ref"] is not None:
        u["h"] = {str(d): (u["k_ref"] - N_STEP * d if N_HMIN <= u["k_ref"] - N_STEP * d <= N_KEND else None) for d in DELTAS}
    return u


def n_tfail(a):
    """Amendment 2: the units, strata and k_ref come from the STORED 18-decision train-pool rollouts of RSG-L18-s0 (--runs), once for
    every stage; writes <out>/tfail.json and from_D<d>.txt. rewind_onpol_all: --runs = the stage's own 30-decision identity runs, --kend 29
    (with RF_K_END=29 in the env: rf_follow's turn window), --deltas 2,3, --rewind event,off (short and B listed, never handed over)."""
    global N_KEND, DELTAS
    N_KEND, DELTAS = a.kend, tuple(int(x) for x in a.deltas.split(","))
    n_paths()
    from jevdrive import par
    stage = Path(a.stage)
    only = set(Path(a.scenes).read_text().split())
    res = par.pmap(n_roll, n_jobs(stage, ["ref18"], log_tags=(), dirs=a.runs.split(","), only=only), workers=a.workers, desc="stored rollouts")
    if res.errors:
        print("errors:", {res.items[i][2]: str(e)[-300:] for i, e in res.errors.items()}, file=sys.stderr)
    gt = n_gate()
    U = {o["scene"]: n_ref(o, gt) for o in res.values if o is not None}
    rw = set(a.rewind.split(","))
    for u in U.values():                                 # units whose src is not rewound keep k_ref, lose h
        if u["src"] is not None and not ({"zero": "event", "off": "off", "off_cross": "off", "short_end": "short", "viol": "viol"}.get(u["src"]) in rw):
            u["h"] = {str(d): None for d in DELTAS}
    (stage / "tfail.json").write_text(json.dumps(dict(gate=gt, units=U), indent=1, default=float))
    for d in DELTAS:
        hs = sorted((s, u["h"][str(d)]) for s, u in U.items() if u["h"][str(d)] is not None)
        (stage / f"from_D{d}.txt").write_text("".join(f"{s} {h}\n" for s, h in hs))
    c = _count((u["stratum"], u["src"]) for u in U.values())
    print(len(U), "units", c, {d: sum(u["h"][str(d)] is not None for u in U.values()) for d in DELTAS})


def n_metrics(o, ido, u, gt):
    """One rollout against its identity -> the read row (handover state, rewind check, recovery, gate, profile, yield)."""
    import rf_shape as SH
    row, C = o["row"], o["ctrl"]
    x = dict(scene=o["scene"], tag=o["tag"], stratum=u["stratum"], unit_src=u["src"], score=row["score"], turn=bool(row.get("turn")),
             completed=bool(row.get("completed")), outcome=row.get("outcome"), kz=row["kz"], zkind=o["zkind"],
             at_fault=bool(o["zkind"] == "collision"), entry=row.get("entry"), exit=row.get("exit"), lat_cross=row.get("lat_cross"))
    x["n_dec"] = len(o["drv"])
    if C is None or not row.get("turn"):
        x["missing"] = C is None
        return x
    t, P, vx, st, lat = np.asarray(C["t"], float), np.c_[C["x"], C["y"]], np.asarray(C["vx"], float), o["st"], o["lat"]
    tz, ent, ext = o["tz"], row["entry"], row["exit"]
    on = [d for d in o["drv"] if d["on"]]
    h = on[0]["k"] if on and o["tag"] != "xp0" else (0 if o["tag"] == "xp0" else None)
    x["h"] = h
    t_lo = -np.inf
    if h is not None and o["tag"] != "xp0":
        dh = next(d for d in o["drv"] if d["k"] == h)
        th = float(dh["now"])
        t_lo = th
        i = int(np.clip(np.searchsorted(t, th), 0, len(t) - 1))
        x.update(t_h=th, t_h_rel_s=round((th - o["drv"][0]["now"]) * 1e-6, 2), v_h=interp(t, vx, th), lat_h=float(lat[i]),
                 st_h_rel_entry=float(st[i]) - ent, v_trace=[interp(t, vx, th + q * 1e6) for q in (0.0, 0.25, 0.5, 0.75, 1.0)],
                 v_win_end=interp(t, vx, th + WIN_END * 1e6), v_end=float(vx[-1]), t_left_s=round((t[-1] - th) * 1e-6, 2),
                 rear_after=bool(o["trear"] is not None and o["trear"] >= th), zero_before_h=bool(np.isfinite(tz) and tz < th))
        m = (t >= th) & (t < tz)
        ok = (np.abs(lat) <= SETTLE_LAT) & m
        run = np.convolve(ok.astype(int), np.ones(int(SETTLE_S * 10) + 1, int), "valid")
        j = np.flatnonzero(run == int(SETTLE_S * 10) + 1)
        x["settle_s"] = round((t[j[0]] - th) * 1e-6, 2) if len(j) else None
        if ido is not None and ido["ctrl"] is not None:      # the rewind check against the identity of the same scene
            Rd = {d["k"]: d for d in ido["drv"]}
            pre = [d for d in o["drv"] if d["k"] < h and d["k"] in Rd]
            x["pre_plan_maxdiff"] = max((float(np.max(np.abs(np.subtract(d["served"], Rd[d["k"]]["served"])))) for d in pre), default=None)
            x["pre_n"] = len(pre)
            Ci = ido["ctrl"]
            ti = np.asarray(Ci["t"], float)
            n = int(min(np.searchsorted(t, th, side="right"), np.searchsorted(ti, th, side="right")))
            x["pre_track_maxdiff"] = float(max(np.abs(P[:n] - np.c_[Ci["x"], Ci["y"]][:n]).max(), np.abs(vx[:n] - np.asarray(Ci["vx"])[:n]).max())) if n else None
            x["id_v_h"] = interp(ti, np.asarray(Ci["vx"], float), th)
            x["id_v_trace"] = [interp(ti, np.asarray(Ci["vx"], float), th + q * 1e6) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
            dv = abs(x["v_h"] - x["id_v_h"]) if np.isfinite(x["v_h"]) and np.isfinite(x["id_v_h"]) else np.inf
            x["dv_h"] = dv                                   # Amendment 2: within the identity's run-to-run spread
            x["within_tol"] = bool((x["pre_plan_maxdiff"] or 0.0) <= TOL_PLAN and (x["pre_track_maxdiff"] or 0.0) <= TOL_TRACK and dv <= TOL_V)
            if u["stratum"] == "B":                           # B: the profile after h, expert against the identity over the same time span
                for lab, tt, ll, PP, vv, ss, zz in (("xp", t, lat, P, vx, st, tz), ("id", ti, ido["lat"], np.c_[Ci["x"], Ci["y"]], np.asarray(Ci["vx"], float), ido["st"], ido["tz"])):
                    w = (tt >= th) & (tt < zz) & (ss >= ent - N_WIN) & (ss <= ext)
                    sh = n_shape(tt, PP, vv, ss, ent - N_WIN, ext, zz, t_lo=th)
                    x[f"post_{lab}"] = dict(lat_med=float(np.median(np.abs(ll[w]))) if w.any() else np.nan,
                                            lat_max=float(np.max(np.abs(ll[w]))) if w.any() else np.nan, kmax=sh["kmax"], dk=sh["dk"], alat=sh["alat"])
    shp = n_shape(t, P, vx, st, ent - N_WIN, ext, tz, t_lo=t_lo)
    spd = SH.speeds(t, st, vx, ent, ext, tz)
    ent_after = not np.isfinite(t_lo) or (x.get("st_h_rel_entry") is not None and x["st_h_rel_entry"] < 0)
    gk = dict(kmax=bool(shp["kmax"] <= gt["kmax"]), dk=bool(shp["dk"] <= gt["dk"]), alat=bool(shp["alat"] <= gt["alat"]),
              v_entry=bool(not ent_after or (np.isfinite(spd["v_0"]) and spd["v_0"] <= gt["v_entry"])))
    x.update(kmax=shp["kmax"], dk=shp["dk"], alat=shp["alat"], v_0=spd["v_0"], gate=gk, shape_ok=bool(x["completed"] and all(gk.values())))
    kz = row["kz"] if row["kz"] >= 0 else 10 ** 6
    hh = h if h is not None else 0
    x["rows_k"] = [int(d["k"]) for d in o["dec"] if d["k"] >= hh and d["k"] < kz and ent - N_PRE <= d["st"] <= ext + N_PAST]  # rf_follow's junction stretch
    x["rows_turn"] = len(x["rows_k"])
    x.update(run=o["run"], rid=o["rid"], ent_after=bool(ent_after))
    x["rows_after"] = int(sum(d["k"] >= hh and d["k"] < kz for d in o["dec"]))
    if o.get("log") is not None:                             # calibration: the human log through the same gate
        L = o["log"]
        lsh = n_shape(L["t"], L["P"], L["v"], L["st"], ent - N_WIN, ext)
        lsp = SH.speeds(L["t"], L["st"], L["v"], ent, ext)
        x["log_gate"] = dict(kmax=bool(lsh["kmax"] <= gt["kmax"]), dk=bool(lsh["dk"] <= gt["dk"]), alat=bool(lsh["alat"] <= gt["alat"]),
                             v_entry=bool(np.isfinite(lsp["v_0"]) and lsp["v_0"] <= gt["v_entry"]))
        x["log_shape"] = dict(kmax=lsh["kmax"], dk=lsh["dk"], alat=lsh["alat"], v_0=lsp["v_0"])
    return x


def n_read(a):
    n_paths()
    from jevdrive import par
    U, gt, jobs = {}, None, []                            # the units (stored 18-decision rollouts, Amendment 2; or one tfail per stage)
    for f in a.ref.split(","):
        T = json.loads(Path(f).read_text())
        U.update(T["units"])
        gt = T["gate"]
    only = {s for s, u in U.items() if u["turn"]}
    for stage in a.stage.split(","):                     # several stages (pilot + rest) read as one set
        jobs += n_jobs(Path(stage), N_TAGS, only=only)
    import csv as C
    span = {r["scene"]: r for r in C.DictReader(open(HERE.parents[2] / "experiments/onpol_nuplan/results/long/span.csv"))}
    res = par.pmap(n_roll, jobs, workers=a.workers, desc="rollouts")
    err = {f"{res.items[i][0]} {res.items[i][2]}": str(e)[-300:] for i, e in res.errors.items()}
    by = {}
    for o in res.values:
        if o is not None:
            by.setdefault(o["scene"], {})[o["tag"]] = o
    out = []
    for sc, R in sorted(by.items()):
        u = U.get(sc)
        if u is None or not u["turn"]:
            continue
        ido = R.get("id")
        rows = [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in n_metrics(o, ido, u, gt).items()} for _, o in sorted(R.items())]
        sp = span.get(sc, {})
        u = dict(u, log_after_s=float(sp.get("log_after_s") or "nan"), video_after_s=float(sp.get("video_after_s") or "nan"))
        out.append(dict(scene=sc, unit=u, rows=rows))
        if a.pkl:
            Path(a.pkl).mkdir(parents=True, exist_ok=True)
            keep = {tg: dict(ctrl=o["ctrl"], R=o["R"], tz=o["tz"], log=o.get("log")) for tg, o in R.items()}
            pickle.dump(dict(scene=sc, unit=u, rolls=keep), open(Path(a.pkl) / f"{sc}.pkl", "wb"), protocol=4)
    Path(a.out).write_text(json.dumps(dict(gate=gt, errors=err, scenes=out, nonturn=sorted(s for s, u in U.items() if not u["turn"])), indent=1, default=float))
    print(len(out), "units,", len(err), "errors ->", a.out)


def n_table(a):
    n_paths()
    from jevdrive import stats
    X = json.loads(Path(a.read).read_text())
    gt = X["gate"]
    by = {s["scene"]: {r["tag"]: r for r in s["rows"]} for s in X["scenes"]}
    U = {s["scene"]: s["unit"] for s in X["scenes"]}
    logs = lambda ss: np.array([s.rsplit("-", 1)[0] for s in ss])  # noqa: E731
    br = lambda v, ss: stats.bootstrap(np.asarray(v, float), logs(ss), n_boot=10_000) if len(ss) else dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan)  # noqa: E731
    ci = lambda r: "-" if not r["n"] else f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"  # noqa: E731
    A = sorted(s for s, u in U.items() if u["stratum"] == "A")
    B = sorted(s for s, u in U.items() if u["stratum"] == "B")
    Bv = [s for s in B if U[s]["src"] == "viol"]
    L = [f"# Recovery-expert rewind on the nuPlan @18 train turns: RSG-L18-s0 (box, held-read serving), expert XP-L12-E46", "",
         f"Turn scenes read {len(U)}; identity outcome " + ", ".join(f"{k} {v}" for k, v in sorted(_count(u["outcome"] for u in U.values()).items())) +
         f". Stratum A (failed) {len(A)}: " + ", ".join(f"{k} {v}" for k, v in sorted(_count(U[s]["src"] for s in A).items())) +
         f"; stratum B (completed) {len(B)}: first-violation {len(Bv)}, no violation (control) {len(B) - len(Bv)}. Gate thresholds: kmax "
         f"{gt['kmax']:.3f} 1/m, dk {gt['dk']:.4f} 1/m^2, alat {gt['alat']:.2f} m/s^2, entry speed {gt['v_entry']:.1f} m/s.", ""]
    csv = ["scene,stratum,unit_src,id18_outcome,k_ref,items,tag,h,lat_h,v_h,id_v_h,completed,outcome,zkind,settle_s,v_win_end,t_left_s,rear_after,"
           "at_fault,shape_ok,gate_fail,rows_turn,rows_after,pre_plan_maxdiff,pre_track_maxdiff,xp0_completed,dv_h,within_tol,n_dec,id_completed,"
           "id_outcome,log_after_s,video_after_s"]
    oc = lambda rr: "/".join(str(sum(r["outcome"] == k for r in rr)) for k in ("completed", "event", "off", "short"))  # noqa: E731
    L += ["Outcome columns are completed / event / off / short (rf_follow; short = still on the route when the rollout ended, its own outcome).", "",
          "| stratum | Delta | units (h >= 1) | expert completes [95 % CI] | expert c / e / o / s | policy (identity, same length) c / e / o / s | "
          "paired expert - policy [95 % CI] | handover within tolerance | settle <= 1 m (n, median s) | v(h + 3 s) >= 2 m/s | rear after h | "
          "at-fault zero | gate pass / completed | gate pass / handovers | rows in turn window after h (sum; completed + gated) |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    summ = {}
    for st_, units in (("A", A), ("A-event", [s for s in A if U[s]["src"] == "zero"]), ("A-off", [s for s in A if U[s]["src"].startswith("off")]),
                       ("A-short", [s for s in A if U[s]["src"] == "short_end"]), ("B", Bv)):
        for d in DELTAS:
            uu = [s for s in units if f"D{d}" in by[s] and not by[s][f"D{d}"].get("missing") and by[s][f"D{d}"].get("h") is not None and "id" in by[s]]
            xr = [by[s][f"D{d}"] for s in uu]
            ir = [by[s]["id"] for s in uu]
            b = br([r["completed"] for r in xr], uu)
            pdf = stats.paired([r["completed"] for r in xr], [r["completed"] for r in ir], logs(uu), n_boot=10_000) if uu else dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0)
            comp = [r for r in xr if r["completed"]]
            gated = [r for r in comp if r.get("shape_ok")]
            vw = [r["v_win_end"] for r in xr if r.get("v_win_end") is not None and np.isfinite(r["v_win_end"])]
            stl = [r["settle_s"] for r in xr if r.get("settle_s") is not None]
            summ[f"{st_}/{d}"] = dict(n=len(uu), n_units=len(units), xp=b, policy=float(np.mean([r["completed"] for r in ir])) if ir else np.nan,
                                     paired=pdf, n_comp=len(comp), n_gated=len(gated), gate_share=len(gated) / len(uu) if uu else np.nan,
                                     creep=(sum(v >= V_CREEP for v in vw), len(vw)), tol=(sum(bool(r.get("within_tol")) for r in xr), len(xr)),
                                     outcomes=oc(xr), policy_outcomes=oc(ir),
                                     rows=sum(r.get("rows_turn", 0) for r in xr), rows_gated=sum(r.get("rows_turn", 0) for r in gated))
            S_ = summ[f"{st_}/{d}"]
            L.append(f"| {st_} | {d} s | {len(uu)} / {len(units)} | {sum(r['completed'] for r in xr)} = {ci(b)} | {oc(xr)} | {oc(ir)} | "
                     f"{ci(pdf) if uu else '-'} | {S_['tol'][0]} / {S_['tol'][1]} | {len(stl)}, {np.median(stl) if stl else float('nan'):.1f} | "
                     f"{S_['creep'][0]} / {S_['creep'][1]} | {sum(bool(r.get('rear_after')) for r in xr)} | {sum(r['at_fault'] for r in xr)} | "
                     f"{len(gated)} / {len(comp)} | {len(gated)} / {len(uu)} | {S_['rows']}; {S_['rows_gated']} |")
    for s in sorted(A + B):
        u = U[s]
        for tg in ("D1", "D2", "D3", "xp0"):
            r = by[s].get(tg)
            if r is None or r.get("missing") or (tg != "xp0" and r.get("h") is None):
                continue
            gf = "+".join(k for k, v in (r.get("gate") or {}).items() if not v)
            csv.append(",".join(str(v) for v in (s, u["stratum"], u["src"], u["outcome"], u["k_ref"], "+".join(u.get("items") or []), tg, r.get("h"),
                                                   _f(r.get("lat_h")), _f(r.get("v_h")), _f(r.get("id_v_h")), int(r["completed"]), r["outcome"], r["zkind"] or "",
                                                   r.get("settle_s") if r.get("settle_s") is not None else "", _f(r.get("v_win_end")), r.get("t_left_s", ""),
                                                   int(bool(r.get("rear_after"))), int(r["at_fault"]), int(bool(r.get("shape_ok"))), gf, r.get("rows_turn", ""),
                                                   r.get("rows_after", ""), _f(r.get("pre_plan_maxdiff"), 6), _f(r.get("pre_track_maxdiff"), 6),
                                                   int(by[s].get("xp0", {}).get("completed", False)), _f(r.get("dv_h"), 4),
                                                   "" if r.get("within_tol") is None else int(r["within_tol"]), r.get("n_dec", ""),
                                                   int(by[s].get("id", {}).get("completed", False)), by[s].get("id", {}).get("outcome", ""),
                                                   _f(u.get("log_after_s"), 1), _f(u.get("video_after_s"), 1))))
    L += ["", "Read: units, strata and k_ref from the stored 18-decision train-pool rollouts; h = k_ref - 2 Delta >= 1 (decision 0 is the runtime's "
          "forced-log step). Every session of this read runs 30 decisions (15 s; Amendment 2). Policy = RSG-L18-s0 alone at the same length (`id`), "
          "so it is no longer 0 / 1 by construction. Handover within tolerance = before h served plans <= 0.15 m and track <= 5 cm from `id`, "
          "|v(h) - v_id(h)| <= 0.05 m/s. CI: bootstrap over logs (scenes of one log drawn together), 10 000. Rows = expert decisions k >= h with "
          "route station in [entry - 20, exit + 10] m before any zero (rf_follow's junction stretch, 2 Hz).", ""]
    L += ["## Expert outcomes per stratum and Delta", "", "| stratum | Delta | completed | event (collision / offroad / corridor) | off | short |", "|:--|--:|--:|--:|--:|--:|"]
    for st_, units in (("A", A), ("B", Bv)):
        for d in DELTAS:
            xr = [by[s][f"D{d}"] for s in units if f"D{d}" in by[s] and not by[s][f"D{d}"].get("missing") and by[s][f"D{d}"].get("h") is not None]
            c = _count(r["outcome"] for r in xr)
            ev = _count(r["zkind"] for r in xr if r["outcome"] == "event")
            L.append(f"| {st_} | {d} s | {c.get('completed', 0)} | {c.get('event', 0)} ({ev.get('collision', 0)} / {ev.get('offroad', 0)} / "
                     f"{ev.get('corridor', 0)}) | {c.get('off', 0)} | {c.get('short', 0)} |")
    # B profile
    L += ["", "## Stratum B: the turn after the handover, expert against the identity over the same time span (median over handovers)", "",
          "| Delta | n | median abs lateral id / expert (m) | max abs lateral id / expert (m) | p95-peak kappa id / expert (1/m) | peak a_lat id / expert (m/s^2) | "
          "violated item at k_ref |", "|--:|--:|--:|--:|--:|--:|:--|"]
    for d in DELTAS:
        xr = [by[s][f"D{d}"] for s in Bv if f"D{d}" in by[s] and by[s][f"D{d}"].get("post_xp")]
        md = lambda lab, k: float(np.nanmedian([r[f"post_{lab}"][k] for r in xr])) if xr else float("nan")  # noqa: E731
        L.append(f"| {d} s | {len(xr)} | {md('id', 'lat_med'):.2f} / {md('xp', 'lat_med'):.2f} | {md('id', 'lat_max'):.2f} / {md('xp', 'lat_max'):.2f} | "
                 f"{md('id', 'kmax'):.3f} / {md('xp', 'kmax'):.3f} | {md('id', 'alat'):.2f} / {md('xp', 'alat'):.2f} | "
                 + ", ".join(f"{k} {v}" for k, v in sorted(_count("+".join(U[s]["items"]) for s in Bv).items())) + " |")
    # C
    C_ = sorted(s for s in by if "xp0" in by[s] and not by[s]["xp0"].get("missing"))
    bc = br([by[s]["xp0"]["completed"] for s in C_], C_)
    summ["C"] = dict(n=len(C_), xp=bc)
    c0 = _count(by[s]["xp0"]["outcome"] for s in C_)
    L += ["", "## Stratum C: the expert from the scene start (xp0) on every turn", "",
          f"- expert completes {sum(by[s]['xp0']['completed'] for s in C_)} / {len(C_)} = {ci(bc)} (policy {sum(by[s]['id']['completed'] for s in C_)}); outcomes "
          + ", ".join(f"{k} {v}" for k, v in sorted(c0.items())),
          f"- on A units {sum(by[s]['xp0']['completed'] for s in A if s in C_)} / {len(A)}, on B units {sum(by[s]['xp0']['completed'] for s in B if s in C_)} / {len(B)}; "
          f"gate pass (of completed) {sum(by[s]['xp0'].get('shape_ok', False) for s in C_)} / {sum(by[s]['xp0']['completed'] for s in C_)}"]
    hum = [by[s]["id"]["log_gate"] for s in by if "id" in by[s] and by[s]["id"].get("log_gate")]
    if hum:
        L.append(f"- calibration, the human log through the same gate: all items {sum(all(g.values()) for g in hum)} / {len(hum)}; "
                 + ", ".join(f"{k} {sum(g[k] for g in hum)}" for k in ("kmax", "dk", "alat", "v_entry")))
    # rewind check
    chk = [r for s in by for tg, r in by[s].items() if tg.startswith("D") and not r.get("missing") and r.get("h") is not None]
    pm = [r["pre_plan_maxdiff"] for r in chk if r.get("pre_plan_maxdiff") is not None]
    tm = [r["pre_track_maxdiff"] for r in chk if r.get("pre_track_maxdiff") is not None]
    dv = [abs(r["v_h"] - r["id_v_h"]) for r in chk if r.get("id_v_h") is not None and np.isfinite(r["id_v_h"]) and np.isfinite(r["v_h"])]
    L += ["", "## Rewind check (each D session against the identity of the same scene, before h)", "",
          f"- sessions {len(chk)}; served plans before h bit-identical {sum(x == 0 for x in pm)} / {len(pm)} (max {max(pm, default=float('nan')):.2g}); "
          f"controller track before h identical {sum(x == 0 for x in tm)} / {len(tm)} (max {max(tm, default=float('nan')):.2g}); "
          f"|v(h) - v_id(h)| max {max(dv, default=float('nan')):.3g} m/s; within the registered tolerance {sum(bool(r.get('within_tol')) for r in chk)} / {len(chk)}"]
    rep = [(s, tg) for s in by for tg in ("D1", "D2", "D3") if tg in by[s] and by[s][tg].get("h") is None and "id" in by[s]]
    if rep:                                                  # D launches replay the identity on scenes without a handover at that Delta
        L.append(f"- identity replays inside the D launches (no handover at that Delta): score and outcome equal to id "
                 f"{sum(by[s][tg]['score'] == by[s]['id']['score'] and by[s][tg]['outcome'] == by[s]['id']['outcome'] for s, tg in rep)} / {len(rep)}")
    ib = [s for s in by if "idB" in by[s]]
    if ib:
        L.append(f"- idB (second identity session): scores equal {sum(by[s]['idB']['score'] == by[s]['id']['score'] for s in ib)} / {len(ib)}, outcomes equal "
                 f"{sum(by[s]['idB']['outcome'] == by[s]['id']['outcome'] for s in ib)} / {len(ib)}")
    ag = [s for s in U if "id" in by[s]]
    tr = _count(f"{U[s]['outcome']}->{by[s]['id']['outcome']}" for s in ag)
    L.append(f"- the identity at 30 decisions against the stored 18-decision outcome ({len(ag)} turns): " + ", ".join(f"{k} {v}" for k, v in sorted(tr.items())))
    nd = [(s, by[s]["id"].get("n_dec")) for s in ag]
    L += ["", "## Flags (Amendment 2)", "",
          f"- decisions run by the identity: 30 on {sum(n == 30 for _, n in nd)} / {len(nd)}; fewer (the log ends, the runtime clips): "
          + (", ".join(f"{s[-8:]} {n}" for s, n in nd if n != 30) or "none"),
          f"- MTGS camera coverage after the token (span.csv video_after_s): median {np.nanmedian([U[s]['video_after_s'] for s in ag]):.1f} s, >= 14 s on "
          f"{sum(U[s]['video_after_s'] >= 14 for s in ag)} / {len(ag)}. Past it the renderer draws outside the reconstruction: the policy's frames "
          "(identity, and the model before h) are confounded there; the expert reads the map, not the frames."]
    # lines
    a2, b2 = summ["A/2"], summ["B/2"]
    L += ["", "## Registered lines", "",
          f"- A (primary Delta 2 s): expert completion {ci(a2['xp'])} on {a2['n']} units (line >= 0.60): **{'PASS' if a2['n'] and a2['xp']['mean'] >= 0.6 else 'FAIL'}**; "
          f"Delta 1 s {ci(summ['A/1']['xp'])}, 3 s {ci(summ['A/3']['xp'])}",
          f"- B (Delta 2 s): gate pass on {b2['n_gated']} / {b2['n']} handovers = {b2['gate_share']:.2f} (line >= 0.5): **{'PASS' if b2['n'] and b2['gate_share'] >= 0.5 else 'FAIL'}**; "
          f"Delta 1 s {summ['B/1']['n_gated']} / {summ['B/1']['n']}, 3 s {summ['B/3']['n_gated']} / {summ['B/3']['n']}",
          f"- C: expert from start {ci(bc)} (line >= 0.80): **{'PASS' if bc['n'] and bc['mean'] >= 0.8 else 'FAIL'}**"]
    if X.get("errors"):
        L += ["", f"Read errors: {X['errors']}"]
    Path(a.md).write_text("\n".join(L) + "\n")
    Path(a.csv).write_text("\n".join(csv) + "\n")
    Path(a.md).with_suffix(".json").write_text(json.dumps(summ, indent=1, default=float))
    print("\n".join(L))


def n_fig(a):
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parents[2] / "research"))
    import plot_style as PS
    PS.apply()
    X = {s["scene"]: s for s in json.loads(Path(a.read).read_text())["scenes"]}
    pk = a.pkl.split(",")
    figr, axs = plt.subplots(1, len(pk), figsize=(6.875, 6.875 / len(pk) + 0.4))
    col = {"id": "#999999", "xp0": "#000000", "D1": "#E69F00", "D2": "#56B4E9", "D3": "#009E73"}
    for ax, f in zip(np.atleast_1d(axs), pk):
        sc = pickle.load(open(f, "rb"))
        rows = {r["tag"]: r for r in X[sc["scene"]]["rows"]}
        R = sc["rolls"]["id"]["R"]
        ax.plot(R[:, 0], R[:, 1], color="#dddddd", lw=3.0, zorder=0, label="route")
        L = sc["rolls"]["id"].get("log")
        if L is not None:
            ax.plot(L["P"][:, 0], L["P"][:, 1], color="#0072B2", lw=1.0, ls="--", label="human log", zorder=1)
        lim = []
        for tag in ("id", "xp0", "D3", "D2", "D1"):
            o = sc["rolls"].get(tag)
            if o is None or o.get("ctrl") is None or tag not in rows or (tag.startswith("D") and rows[tag].get("h") is None):
                continue                                     # a D session without a handover at that Delta replays the identity
            C = o["ctrl"]
            m = np.asarray(C["t"]) <= o["tz"]
            r = rows[tag]
            lab = {"id": "policy (identity)", "xp0": "expert from start"}.get(tag, f"expert from -{tag[-1]} s")
            gate = " gated" if r.get("shape_ok") else ""
            ax.plot(C["x"][m], C["y"][m], color=col[tag], lw=1.1, label=f"{lab}: {'done' + gate if r['completed'] else r['outcome']}", zorder=2)
            if tag.startswith("D") and r.get("t_h") is not None:
                i = int(np.searchsorted(C["t"], r["t_h"]))
                ax.plot(C["x"][i], C["y"][i], "o", color=col[tag], ms=3, zorder=3)
            if not m.all():
                ax.plot(C["x"][m][-1], C["y"][m][-1], "x", color=col[tag], ms=4, zorder=3)
            lim.append(np.c_[C["x"][m], C["y"][m]])
        Q = np.concatenate(lim)
        c0, r0 = Q.mean(0), max(np.ptp(Q, 0).max() / 2 + 5, 20)
        ax.set_xlim(c0[0] - r0, c0[0] + r0)
        ax.set_ylim(c0[1] - r0, c0[1] + r0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])  # noqa: E702
        u = sc["unit"]
        ax.text(0.02, 0.98, f"{sc['scene'][-8:]} ({u['stratum']}: {u['outcome']}" + (f", {'+'.join(u['items'])}" if u.get("items") else "") + ")",
                transform=ax.transAxes, va="top", fontsize=7)
        ax.legend(fontsize=5.5, loc="lower right", frameon=False)
    figr.tight_layout()
    PS.save(figr, str(Path(a.out).with_suffix("")))
    print("->", a.out)


# ---------------------------------------------------------------- self rewind: the policy's own frontier (prereg plans/2026-10-16-rewind-self-prereg.md)
S_DELTAS, S_NCTRL, S_N = (3, 5, 8), 10, 2                          # Delta (s); control units; perturbed decisions (2 Hz)
S_VARIANTS = tuple((g, s) for g in (0.7, 0.85, 1.15, 1.3) for s in (0.9, 1.1))
B_LAT, B_YAW, B_VR, B_DV0, B_V0 = 0.5, 5.0, 0.1, 0.1, 1.0          # bound at the window end: |lateral| m, |heading| deg, |v / v_id - 1|;
                                                                     # |v - v_id| m/s when v_id < B_V0
MOVED = (0.02, 0.2, 0.01)                                           # a variant that left the identity: lateral m, heading deg, speed ratio
V_E_OTHER = 4.6                                                     # rf_p4.V_E_OTHER: the nuPlan (no topo class) entry speed
V_CRAWL = 0.5 * V_E_OTHER                                           # G-crawl (d297, rf_selfpool CRAWL 0.5 v_e) on the approach after h


def vtag(g, s):
    return f"g{g:g}s{s:g}"


def parse_variants(spec):
    import re
    return [(float(m[1]), float(m[2])) for m in re.finditer(r"g([\d.]+)s([\d.]+)", spec)] if spec else list(S_VARIANTS)


def n_self_units(a):
    """Units of the self read from the STORED 18-decision rollouts (as n-tfail): failed = d316's stratum A without `short` (k_ref as
    d316); controls = the S_NCTRL completed turns with the latest entrance crossing (k_ref = the first decision whose route station
    >= entry). h per Delta = k_ref - 2 Delta, 1 <= h; controls at Delta 3 take h = max(1, k_ref - 6) (they cross the entry at k 4-8, so
    the effective Delta is 1.5-3.5 s, `delta_eff`). -> <out>/self.json, <out>/scenes.txt"""
    n_paths()
    from jevdrive import par
    out = Path(a.out)
    only = set(Path(a.scenes).read_text().split())
    res = par.pmap(n_roll, n_jobs(out, ["ref18"], log_tags=(), dirs=a.runs.split(","), only=only), workers=a.workers, desc="stored rollouts")
    if res.errors:
        print("errors:", {res.items[i][2]: str(e)[-300:] for i, e in res.errors.items()}, file=sys.stderr)
    gt = n_gate()
    T64 = json.loads(Path(a.tfail).read_text())["units"]
    U = {}
    for o in res.values:
        if o is None:
            continue
        u = n_ref(o, gt)
        if not u["turn"]:
            continue
        ke = [d["k"] for d in o["dec"] if d["st"] >= o["row"]["entry"]]
        u["k_entry"] = int(ke[0]) if ke else None
        q = T64.get(o["scene"], {})
        u["same_as_d316"] = bool(q.get("k_ref") == u["k_ref"] and q.get("src") == u["src"])
        U[o["scene"]] = u
    fail = sorted(s for s, u in U.items() if u["stratum"] == "A" and u["src"] != "short_end")
    comp = sorted((s for s, u in U.items() if u["stratum"] == "B" and u["k_entry"] is not None), key=lambda s: (-U[s]["k_entry"], s))
    ctrl = sorted(comp[:S_NCTRL])
    units = {}
    for s in fail + ctrl:
        u = dict(U[s], role="fail" if s in fail else "ctrl")
        if u["role"] == "ctrl":
            u["k_ref"], u["src"] = u["k_entry"], "entry"
        u["hs"] = {str(d): (u["k_ref"] - N_STEP * d if u["k_ref"] - N_STEP * d >= N_HMIN else None) for d in S_DELTAS}
        if u["role"] == "ctrl" and u["k_ref"] - 1 >= N_HMIN:   # controls cross the entry early: at Delta 3 h = max(1, k_entry - 6) (prereg)
            u["hs"][str(S_DELTAS[0])] = max(N_HMIN, u["k_ref"] - N_STEP * S_DELTAS[0])
        u["delta_eff"] = {d: (u["k_ref"] - h) / N_STEP for d, h in u["hs"].items() if h is not None}
        units[s] = u
    (out / "self.json").write_text(json.dumps(dict(gate=gt, units=units, completed_k_entry={s: U[s]["k_entry"] for s in comp},
                                                   short=sorted(s for s, u in U.items() if u["src"] == "short_end")), indent=1, default=float))
    (out / "scenes.txt").write_text("".join(f"{s}\n" for s in fail + ctrl))
    print(len(fail), "failed,", len(ctrl), "controls (k_entry", [U[s]["k_entry"] for s in ctrl], ")", "same k_ref as d316:",
          sum(u["same_as_d316"] for u in units.values()), "/", len(units),
          {d: (sum(u["hs"][str(d)] is not None for u in units.values() if u["role"] == "fail"),
               sum(u["hs"][str(d)] is not None for u in units.values() if u["role"] == "ctrl")) for d in S_DELTAS})


def n_self_files(a):
    """One stage's launches -> <out>/launches.txt (`tag mode file`: id = the policy alone on the list; xp = the expert from h
    (JEV_EXPERT_FROM file); pt = one perturbation variant (JEV_PERTURB file)), the files beside it."""
    T = json.loads(Path(a.units).read_text())["units"]
    L = [x.split()[0] for x in open(a.list) if x.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"id id {Path(a.list).resolve()}"] + ([f"idB id {Path(a.list).resolve()}"] if a.smoke else [])
    for d in S_DELTAS:
        hs = [(s, T[s]["hs"][str(d)]) for s in L if s in T and T[s]["hs"][str(d)] is not None]
        if not hs:
            continue
        f = out / f"xp_D{d}.txt"
        f.write_text("".join(f"{s} {h}\n" for s, h in hs))
        lines.append(f"xpD{d} xp {f.resolve()}")
        for g, sc in parse_variants(a.variants):
            f = out / f"pt_D{d}_{vtag(g, sc)}.txt"
            f.write_text("".join(f"{s} {h} {g:g} {sc:g} {a.n}\n" for s, h in hs))
            lines.append(f"D{d}-{vtag(g, sc)} pt {f.resolve()}")
    (out / "launches.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def s_state(C, tq):
    return np.array([np.interp(tq, C["t"], C[k]) for k in ("x", "y", "yaw", "vx")])


def s_disp(Ca, Cb, tq):
    """Displacement of track a from the identity b at time tq: lateral / longitudinal in b's heading frame, heading (deg), speed ratio."""
    a, b = s_state(Ca, tq), s_state(Cb, tq)
    d = a[:2] - b[:2]
    lat, lon = float(-np.sin(b[2]) * d[0] + np.cos(b[2]) * d[1]), float(np.cos(b[2]) * d[0] + np.sin(b[2]) * d[1])
    yaw = float(np.degrees((a[2] - b[2] + np.pi) % (2 * np.pi) - np.pi))
    vr = float(a[3] / b[3]) if b[3] >= B_V0 else np.nan
    dv = float(a[3] - b[3])
    spd_ok = abs(vr - 1) <= B_VR if np.isfinite(vr) else abs(dv) <= B_DV0
    moved = abs(lat) > MOVED[0] or abs(yaw) > MOVED[1] or (abs(vr - 1) > MOVED[2] if np.isfinite(vr) else abs(dv) > MOVED[2])
    return dict(lat=lat, lon=lon, yaw=yaw, vr=vr, dv=dv, v_id=float(b[3]), in_bound=bool(abs(lat) <= B_LAT and abs(yaw) <= B_YAW and spd_ok),
                moved=bool(moved))


def s_strict(o, th):
    """Strict completion from a handover time: rf_follow completed AND v(th + 3 s) >= 2 m/s AND no G-crawl (box-smoothed speed below
    0.5 v_e on the approach, route station < entry, after th and before any zero)."""
    import rf_shape as SH
    C, row = o["ctrl"], o["row"]
    t, vx = np.asarray(C["t"], float), np.asarray(C["vx"], float)
    vs = SH.box5(vx)
    v3 = interp(t, vx, th + WIN_END * 1e6)
    m = (t >= th) & (t < o["tz"]) & (o["st"] < row["entry"])
    vmin = float(vs[m].min()) if m.any() else None
    crawl = bool(vmin is not None and vmin < V_CRAWL)
    return dict(v3=v3, crawl=crawl, v_min_approach=vmin,
                strict=bool(row.get("completed") and np.isfinite(v3) and v3 >= V_CREEP and not crawl))


def n_self_extra(o, ido, u, tg):
    x = {}
    if o["ctrl"] is None or not o["row"].get("turn"):
        return x
    if tg in ("id", "idB"):                                  # the policy's own continuation, read from every Delta's h
        for d in S_DELTAS:
            h = u["hs"][str(d)]
            dh = next((r for r in o["drv"] if r["k"] == h), None) if h is not None else None
            if dh is not None:
                x[f"strict_D{d}"] = s_strict(o, float(dh["now"]))
        return x
    on = [r for r in o["drv"] if r["on"]]
    if not on:
        return dict(no_handover=True)
    th = float(on[0]["now"])
    x.update(s_strict(o, th))
    p = on[0].get("perturb")
    if p:
        x.update(g=p["g"], s=p["s"], n=p["n"], k0=p["k0"], n_on=len(on), on_ks=[r["k"] for r in on])
        rw = next((r for r in o["drv"] if r["k"] == p["k0"] + p["n"]), None)
        tw = float(rw["now"]) if rw is not None else th + p["n"] * STEP * 5
        x["t_w_rel_h_s"] = round((tw - th) * 1e-6, 2)
        if ido is not None and ido.get("ctrl") is not None:
            x["disp"] = s_disp(o["ctrl"], ido["ctrl"], tw)
            x["disp_h"] = s_disp(o["ctrl"], ido["ctrl"], th)
            x["disp_1s"] = s_disp(o["ctrl"], ido["ctrl"], tw + 1e6)
    return x


def s_vs(new, old, u):
    """The new identity session against a reference identity (stored d316 or idB): before the largest h of the unit."""
    hs = [h for h in u["hs"].values() if h is not None]
    hmax = max(hs) if hs else u["k_ref"]
    Rd = {d["k"]: d for d in old["drv"]}
    pre = [d for d in new["drv"] if d["k"] < hmax and d["k"] in Rd]
    plan = max((float(np.max(np.abs(np.subtract(d["served"], Rd[d["k"]]["served"])))) for d in pre), default=None)
    dn = {d["k"]: float(d["now"]) for d in new["drv"]}
    th = dn.get(hmax, np.inf)
    Ca, Cb = new["ctrl"], old["ctrl"]
    ta, tb = np.asarray(Ca["t"], float), np.asarray(Cb["t"], float)
    n = int(min(np.searchsorted(ta, th, side="right"), np.searchsorted(tb, th, side="right")))
    track = float(max(np.abs(np.c_[Ca["x"], Ca["y"]][:n] - np.c_[Cb["x"], Cb["y"]][:n]).max(), np.abs(np.asarray(Ca["vx"])[:n] - np.asarray(Cb["vx"])[:n]).max())) if n else None
    dv = {d: abs(interp(ta, np.asarray(Ca["vx"], float), dn[h]) - interp(tb, np.asarray(Cb["vx"], float), dn[h])) for d, h in u["hs"].items() if h is not None and h in dn}
    ok = (plan or 0.0) <= TOL_PLAN and (track or 0.0) <= TOL_TRACK and all(v <= TOL_V for v in dv.values())
    return dict(hmax=hmax, pre_n=len(pre), plan_maxdiff=plan, track_maxdiff=track, dv_h=dv, within_tol=bool(ok),
                outcome=new["row"].get("outcome"), ref_outcome=old["row"].get("outcome"), score=new["row"].get("score"), ref_score=old["row"].get("score"))


def n_self_read(a):
    n_paths()
    from jevdrive import par
    T = json.loads(Path(a.units).read_text())
    U, gt, jobs = T["units"], T["gate"], []
    for stage in a.stage.split(","):
        tags = sorted({Path(p).parts[-3] for p in glob.glob(os.path.join(stage, "*", "c0", "run"))})
        jobs += n_jobs(Path(stage), tags)
    have = {j[2] for j in jobs}
    for st in (a.stored.split(",") if a.stored else []):  # d316's identities (smoke: the unset switch against them)
        jobs += [("id316", *j[1:3], j[3], False) for j in n_jobs(Path(st), ["id"], log_tags=(), only=have)]
    res = par.pmap(n_roll, jobs, workers=a.workers, desc="rollouts")
    err = {f"{res.items[i][0]} {res.items[i][2]}": str(e)[-300:] for i, e in res.errors.items()}
    by = {}
    for o in res.values:
        if o is not None:
            by.setdefault(o["scene"], {})[o["tag"]] = o
    out = []
    for sc, R in sorted(by.items()):
        u = U.get(sc)
        if u is None or "id" not in R:
            continue
        ido, uu, rows = R["id"], dict(u, stratum=u["role"]), []
        for tg, o in sorted(R.items()):
            if tg == "id316":
                continue
            x = n_metrics(o, ido, uu, gt)
            x.update(n_self_extra(o, ido, u, tg))
            rows.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in x.items()})
        vs = {ref: s_vs(R["id"], R[ref], u) for ref in ("id316", "idB") if ref in R and R[ref]["ctrl"] is not None and ido["ctrl"] is not None}
        out.append(dict(scene=sc, unit=u, rows=rows, vs=vs))
        if a.pkl:
            Path(a.pkl).mkdir(parents=True, exist_ok=True)
            keep = {tg: dict(ctrl=o["ctrl"], R=o["R"], tz=o["tz"], log=o.get("log")) for tg, o in R.items() if tg != "id316"}
            pickle.dump(dict(scene=sc, unit=u, rolls=keep), open(Path(a.pkl) / f"{sc}.pkl", "wb"), protocol=4)
    Path(a.out).write_text(json.dumps(dict(gate=gt, errors=err, scenes=out), indent=1, default=float))
    print(len(out), "units,", len(err), "errors ->", a.out)


def s_rows(X):
    by = {s["scene"]: {r["tag"]: r for r in s["rows"]} for s in X["scenes"]}
    return by, {s["scene"]: s["unit"] for s in X["scenes"]}, {s["scene"]: s.get("vs", {}) for s in X["scenes"]}


def s_ok(r):
    """A variant in the registered denominator: perturbation applied, pre-h check within d316's tolerance, displacement inside the bound."""
    return bool(r.get("disp") and r["disp"]["in_bound"] and r.get("within_tol"))


def n_self_check(a):
    """Smoke / pilot gate: exit 1 when the switch-off identity leaves d316's tolerance (smoke), a perturbed session's pre-h check fails
    too often, or rollouts are missing."""
    X = json.loads(Path(a.read).read_text())
    by, U, VS = s_rows(X)
    bad, L = [], []
    want = [x.split()[0] for x in open(a.launches) if x.strip()] if a.launches else []
    for s in by:
        for ref, v in VS[s].items():
            L.append(f"{s[-8:]} id vs {ref}: plan {v['plan_maxdiff']}, track {v['track_maxdiff']}, dv(h) {v['dv_h']}, within {v['within_tol']}, "
                     f"outcome {v['outcome']} / {v['ref_outcome']}")
            if ref == "id316" and not v["within_tol"]:
                bad.append(f"{s[-8:]} id outside d316 tolerance against the stored identity")
    pts = [(s, tg, r) for s in by for tg, r in by[s].items() if tg.startswith("D")]
    tol = [r for _, _, r in pts if r.get("within_tol")]
    inb = [r for _, _, r in pts if r.get("disp") and r["disp"]["in_bound"]]
    for s, tg, r in sorted(pts):
        d = r.get("disp") or {}
        L.append(f"{s[-8:]} {tg}: h {r.get('h')} on {r.get('on_ks')} pre plan {_f(r.get('pre_plan_maxdiff'), 4)} track {_f(r.get('pre_track_maxdiff'), 4)} "
                 f"dv {_f(r.get('dv_h'), 4)} | window end lat {_f(d.get('lat'))} yaw {_f(d.get('yaw'), 2)} vr {_f(d.get('vr'))} in_bound {d.get('in_bound')} "
                 f"moved {d.get('moved')} | {r['outcome']} strict {r.get('strict')}")
    missing = []
    for s in by:                                             # every launch of the stage that should hold this scene
        for w in want:
            d = w[3:] if w.startswith("xpD") else w[1:].split("-")[0] if w.startswith("D") else None
            if (d is None or U[s]["hs"].get(d) is not None) and w not in by[s]:
                missing.append(f"{s[-8:]} {w}")
    if pts and len(tol) < 0.8 * len(pts):
        bad.append(f"pre-h check within tolerance {len(tol)} / {len(pts)} < 80 %")
    if a.stage and pts and len(inb) < 0.5 * len(pts):
        bad.append(f"in-bound variants {len(inb)} / {len(pts)} < 50 %")
    if X.get("errors"):
        bad.append(f"read errors {list(X['errors'])[:5]}")
    if missing:
        bad.append(f"{len(missing)} rollouts missing")
    L.append(f"perturbed sessions {len(pts)}: pre-h within tolerance {len(tol)}, in bound {len(inb)}; missing {missing[:10]}")
    L += [f"FAIL: {b}" for b in bad] or ["PASS"]
    print("\n".join(L))
    if a.out:
        Path(a.out).write_text("\n".join(L) + "\n")
    sys.exit(1 if bad else 0)


def n_self_table(a):
    n_paths()
    from jevdrive import stats
    X = json.loads(Path(a.read).read_text())
    by, U, VS = s_rows(X)
    d316 = {s["scene"]: {r["tag"]: r for r in s["rows"]} for s in json.loads(Path(a.d316).read_text())["scenes"]} if a.d316 else {}
    logs = lambda ss: np.array([s.rsplit("-", 1)[0] for s in ss])  # noqa: E731
    br = lambda v, ss: stats.bootstrap(np.asarray(v, float), logs(ss), n_boot=10_000) if len(ss) else dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan)  # noqa: E731
    ci = lambda r: "-" if not r["n"] else f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"  # noqa: E731
    F_ = sorted(s for s in U if U[s]["role"] == "fail")
    K_ = sorted(s for s in U if U[s]["role"] == "ctrl")
    unit_csv = ["scene,role,src,k_ref,delta,h,variants,in_denominator,out_of_bound,out_of_tol,strict,plain,p_hat,p_hat_plain,gated,rows_strict,"
                "id_strict,id_outcome,xp_strict,xp_completed,xp_outcome,d316_xp_completed"]
    var_csv = ["scene,role,delta,tag,g,s,h,lat_w,yaw_w,vr_w,dv_w,lat_1s,in_bound,moved,within_tol,pre_plan_maxdiff,completed,outcome,zkind,v3,crawl,"
               "v_min_approach,strict,shape_ok,rows_turn"]
    P = {}                                                   # (scene, Delta) -> per-unit numbers
    for s in F_ + K_:
        for d in S_DELTAS:
            if U[s]["hs"][str(d)] is None:
                continue
            V = {tg: r for tg, r in by[s].items() if tg.startswith(f"D{d}-")}
            ok = [r for r in V.values() if s_ok(r)]
            xp = by[s].get(f"xpD{d}", {})
            idr = (by[s].get("id", {}).get(f"strict_D{d}") or {})
            q = dict(n=len(V), ok=len(ok), oob=sum(bool(r.get("disp")) and not r["disp"]["in_bound"] for r in V.values()),
                     oot=sum(not r.get("within_tol") for r in V.values()), strict=sum(r.get("strict", False) for r in ok),
                     plain=sum(r["completed"] for r in ok), gated=sum(bool(r.get("strict") and r.get("shape_ok")) for r in ok),
                     rows=sum(r.get("rows_turn", 0) for r in ok if r.get("strict")),
                     rows_gated=sum(r.get("rows_turn", 0) for r in ok if r.get("strict") and r.get("shape_ok")),
                     id_strict=bool(idr.get("strict")), id_outcome=by[s].get("id", {}).get("outcome"), xp_strict=xp.get("strict"),
                     xp_completed=xp.get("completed"), xp_outcome=xp.get("outcome"),
                     d316=(d316.get(s, {}).get("D3", {}).get("completed") if d == 3 else None), ok_rows=ok)
            q["p"] = q["strict"] / q["ok"] if q["ok"] else np.nan
            q["p_plain"] = q["plain"] / q["ok"] if q["ok"] else np.nan
            P[s, d] = q
            u = U[s]
            unit_csv.append(",".join(str(v) for v in (s, u["role"], u["src"], u["k_ref"], d, u["hs"][str(d)], q["n"], q["ok"], q["oob"], q["oot"],
                                                       q["strict"], q["plain"], _f(q["p"]), _f(q["p_plain"]), q["gated"], q["rows"], int(q["id_strict"]),
                                                       q["id_outcome"], "" if q["xp_strict"] is None else int(q["xp_strict"]),
                                                       "" if q["xp_completed"] is None else int(q["xp_completed"]), q["xp_outcome"] or "",
                                                       "" if q["d316"] is None else int(q["d316"]))))
            for tg, r in sorted(V.items()):
                D_, D1 = r.get("disp") or {}, r.get("disp_1s") or {}
                var_csv.append(",".join(str(v) for v in (s, u["role"], d, tg, r.get("g"), r.get("s"), r.get("h"), _f(D_.get("lat")), _f(D_.get("yaw"), 2),
                                                          _f(D_.get("vr")), _f(D_.get("dv")), _f(D1.get("lat")), int(bool(D_.get("in_bound"))),
                                                          int(bool(D_.get("moved"))), int(bool(r.get("within_tol"))), _f(r.get("pre_plan_maxdiff"), 4),
                                                          int(r["completed"]), r["outcome"], r["zkind"] or "", _f(r.get("v3")), int(bool(r.get("crawl"))),
                                                          _f(r.get("v_min_approach")), int(bool(r.get("strict"))), int(bool(r.get("shape_ok"))),
                                                          r.get("rows_turn", ""))))
    bins = (("0", lambda p: p == 0), ("(0, 0.2)", lambda p: 0 < p < 0.2), ("[0.2, 0.8]", lambda p: 0.2 <= p <= 0.8), ("(0.8, 1)", lambda p: 0.8 < p < 1),
            ("1", lambda p: p == 1))
    L = ["# Self rewind: does RSG-L18-s0 have a frontier on its own failed nuPlan @18 train turns?", "",
         f"Units: {len(F_)} failed (d316 stratum A without short: " + ", ".join(f"{k} {v}" for k, v in sorted(_count(U[s]['src'] for s in F_).items())) +
         f"), {len(K_)} controls (completed turns with the latest entrance crossing, k_ref = that decision). Variants per (unit, Delta): curvature gain g x "
         f"arc scale s on the served plan for n decisions from h = k_ref - 2 Delta, then the raw plan. Strict completion = rf_follow completed AND "
         f"v(h + 3 s) >= {V_CREEP:g} m/s AND no G-crawl (approach speed < {V_CRAWL:.1f} m/s after h). p-hat = strict completions / variants in the "
         f"denominator (perturbation inside the bound at the window end: |lateral| <= {B_LAT} m, |heading| <= {B_YAW:g} deg, speed ratio "
         f"{1 - B_VR:g}-{1 + B_VR:g}; and the pre-h check within d316's tolerance). CI: bootstrap over logs, 10 000.", ""]
    summ = {}
    L += ["## p-hat per Delta", "",
          "| role | Delta | units rewindable / all | units with >= 1 variant in the denominator | p-hat bins 0 / (0, 0.2) / [0.2, 0.8] / (0.8, 1) / 1 | "
          "mean p-hat [95 % CI] | mean plain p-hat | units with p-hat > 0 | identity strict (same h) | expert strict / plain at the same h | d316 expert (D3) |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for role, units in (("failed", F_), ("control", K_)):
        for d in S_DELTAS:
            uu = [s for s in units if (s, d) in P]
            ud = [s for s in uu if P[s, d]["ok"]]
            ps = [P[s, d]["p"] for s in ud]
            b = br(ps, ud)
            hb = " / ".join(str(sum(f(p) for p in ps)) for _, f in bins)
            xs = [P[s, d] for s in uu if P[s, d]["xp_completed"] is not None]
            dd = [P[s, d]["d316"] for s in uu if P[s, d]["d316"] is not None]
            summ[f"{role}/{d}"] = dict(n=len(uu), n_all=len(units), n_den=len(ud), p=b, bins=hb, n_mid=sum(0.2 <= p <= 0.8 for p in ps),
                                       n_pos=sum(p > 0 for p in ps), id_strict=sum(P[s, d]["id_strict"] for s in uu),
                                       xp_strict=sum(bool(q["xp_strict"]) for q in xs), xp_plain=sum(bool(q["xp_completed"]) for q in xs), n_xp=len(xs),
                                       p_plain=float(np.mean([P[s, d]["p_plain"] for s in ud])) if ud else np.nan)
            S_ = summ[f"{role}/{d}"]
            L.append(f"| {role} | {d} s | {len(uu)} / {len(units)} | {len(ud)} | {hb} | {ci(b)} | {S_['p_plain']:.2f} | {S_['n_pos']} | "
                     f"{S_['id_strict']} / {len(uu)} | {S_['xp_strict']} / {S_['xp_plain']} of {len(xs)} | {sum(map(bool, dd))} / {len(dd) if dd else '-'} |")
    L += ["", "Non-rewindable (unit, Delta) pairs (h < 1): " + ", ".join(
        f"Delta {d} s: failed {sum(U[s]['hs'][str(d)] is None for s in F_)}, controls {sum(U[s]['hs'][str(d)] is None for s in K_)}" for d in S_DELTAS), ""]
    # variant types (failed units)
    L += ["## Completions by variant type (failed units, variants in the denominator)", "",
          "| Delta | g < 1 | g > 1 | s < 1 | s > 1 | strict completions from s < 1 | strict + d293 gate | rows (strict; strict + gate) | units giving rows |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for d in S_DELTAS:
        rr = [r for s in F_ if (s, d) in P for r in P[s, d]["ok_rows"]]
        typ = lambda f: f"{sum(bool(r.get('strict')) for r in rr if f(r))} / {sum(1 for r in rr if f(r))}"  # noqa: E731
        st = [r for r in rr if r.get("strict")]
        slow = sum(r["s"] < 1 for r in st)
        summ[f"failed/{d}"].update(n_strict=len(st), n_slow=slow, slow_share=slow / len(st) if st else np.nan,
                                   gated=sum(bool(r.get("shape_ok")) for r in st),
                                   rows=sum(P[s, d]["rows"] for s in F_ if (s, d) in P), rows_gated=sum(P[s, d]["rows_gated"] for s in F_ if (s, d) in P),
                                   units_rows=sum(P[s, d]["rows"] > 0 for s in F_ if (s, d) in P))
        S_ = summ[f"failed/{d}"]
        L.append(f"| {d} s | {typ(lambda r: r['g'] < 1)} | {typ(lambda r: r['g'] > 1)} | {typ(lambda r: r['s'] < 1)} | {typ(lambda r: r['s'] > 1)} | "
                 f"{slow} / {len(st)} | {S_['gated']} | {S_['rows']}; {S_['rows_gated']} | {S_['units_rows']} |")
    # displacement
    L += ["", "## Displacement from the identity at the end of the perturbation window (all units)", "",
          "| Delta | variant | n | median / max abs lateral (m) | median / max abs heading (deg) | median speed ratio [min, max] | in bound | moved | "
          "pre-h within tolerance | abs lateral 1 s later (median) |", "|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    allv = [(d, tg.split("-", 1)[1], r) for s in by for tg, r in by[s].items() if tg.startswith("D") and (d := int(tg[1:].split("-")[0]))]
    for d in S_DELTAS:
        for v in sorted({v for dd, v, _ in allv if dd == d}):
            rr = [r for dd, vv, r in allv if dd == d and vv == v and r.get("disp")]
            if not rr:
                continue
            la, ya = np.abs([r["disp"]["lat"] for r in rr]), np.abs([r["disp"]["yaw"] for r in rr])
            vr = np.array([r["disp"]["vr"] for r in rr], float)
            l1 = np.abs([r["disp_1s"]["lat"] for r in rr if r.get("disp_1s")])
            L.append(f"| {d} s | {v} | {len(rr)} | {np.median(la):.3f} / {la.max():.3f} | {np.median(ya):.2f} / {ya.max():.2f} | "
                     f"{np.nanmedian(vr):.3f} [{np.nanmin(vr):.3f}, {np.nanmax(vr):.3f}] | {sum(r['disp']['in_bound'] for r in rr)} | "
                     f"{sum(r['disp']['moved'] for r in rr)} | {sum(bool(r.get('within_tol')) for r in rr)} | {np.median(l1) if len(l1) else float('nan'):.3f} |")
    # identity checks
    v316 = [v["id316"] for v in VS.values() if "id316" in v]
    if v316:
        L += ["", f"Switch-off identity against d316's stored identity: within tolerance {sum(v['within_tol'] for v in v316)} / {len(v316)}; "
              f"outcome equal {sum(v['outcome'] == v['ref_outcome'] for v in v316)} / {len(v316)}"]
    vb = [v["idB"] for v in VS.values() if "idB" in v]
    if vb:
        L.append(f"idB against id: within tolerance {sum(v['within_tol'] for v in vb)} / {len(vb)}")
    # registered lines
    fr = [d for d in S_DELTAS if summ[f"failed/{d}"]["n_mid"] >= 8]
    cpool = br([P[s, d]["p"] for s in K_ for d in S_DELTAS if (s, d) in P and P[s, d]["ok"]],
               [s for s in K_ for d in S_DELTAS if (s, d) in P and P[s, d]["ok"]])
    cm = {d: (summ[f"control/{d}"]["p"]["mean"] if summ[f"control/{d}"]["n_den"] else cpool["mean"]) for d in S_DELTAS}
    front = [d for d in fr if cm[d] >= 0.7]
    none = all(P[s, d]["p"] < 0.1 for s in F_ for d in S_DELTAS if (s, d) in P and P[s, d]["ok"])
    st_front = [r for d in front for s in F_ if (s, d) in P for r in P[s, d]["ok_rows"] if r.get("strict")]
    slow_share = sum(r["s"] < 1 for r in st_front) / len(st_front) if st_front else np.nan
    verdict = ("SLOW-ONLY" if front and slow_share >= 0.8 else "FRONTIER" if front else "NONE" if none else "UNDECIDED")
    summ["verdict"] = dict(verdict=verdict, frontier_deltas=front, mid_deltas=fr, control_mean=cm, control_pooled=cpool, slow_share=slow_share)
    L += ["", "## Registered lines", "",
          "- FRONTIER: at some Delta >= 8 failed units with p-hat in [0.2, 0.8] and the controls' mean p-hat >= 0.7 (same Delta; pooled over Delta when "
          "no control is rewindable there). " + "; ".join(f"Delta {d} s: {summ[f'failed/{d}']['n_mid']} units in [0.2, 0.8], controls "
                                                          f"{cm[d]:.2f}" + ("" if summ[f'control/{d}']['n_den'] else " (pooled)") for d in S_DELTAS),
          f"- NONE: every failed unit p-hat < 0.1 at every Delta: {none}",
          f"- SLOW-ONLY: frontier and >= 80 % of its strict completions from s < 1: share {slow_share:.2f}" if front else "- SLOW-ONLY: no frontier",
          f"- controls pooled over Delta: {ci(cpool)}", "", f"**Verdict: {verdict}**"]
    if X.get("errors"):
        L += ["", f"Read errors: {X['errors']}"]
    Path(a.md).write_text("\n".join(L) + "\n")
    Path(a.csv).write_text("\n".join(unit_csv) + "\n")
    Path(a.csv).with_name(Path(a.csv).stem + "_variants.csv").write_text("\n".join(var_csv) + "\n")
    Path(a.md).with_suffix(".json").write_text(json.dumps(summ, indent=1, default=float))
    print("\n".join(L))


def n_self_fig(a):
    """BEV of 2-3 units at one Delta each: route, human log, identity, the expert from h, the variants (solid = strict completion)."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parents[2] / "research"))
    import plot_style as PS
    PS.apply()
    X = {s["scene"]: s for s in json.loads(Path(a.read).read_text())["scenes"]}
    items = [x.split(":") for x in a.units.split(",")]          # <pkl>:<Delta>
    figr, axs = plt.subplots(1, len(items), figsize=(6.875, 6.875 / len(items) + 0.6))
    for ax, (f, d) in zip(np.atleast_1d(axs), items):
        sc = pickle.load(open(f, "rb"))
        rows = {r["tag"]: r for r in X[sc["scene"]]["rows"]}
        R = sc["rolls"]["id"]["R"]
        ax.plot(R[:, 0], R[:, 1], color="#dddddd", lw=3.0, zorder=0, label="route")
        lim = []
        n_ok = n_st = 0
        for tag, o in sorted(sc["rolls"].items()):
            if not tag.startswith(f"D{d}-") or o.get("ctrl") is None or tag not in rows:
                continue
            r, C = rows[tag], o["ctrl"]
            m = np.asarray(C["t"]) <= o["tz"]
            good = bool(r.get("strict")) and s_ok(r)
            n_ok += s_ok(r)
            n_st += good
            ax.plot(C["x"][m], C["y"][m], color="#009E73" if good else "#D55E00", lw=0.7, alpha=0.8, ls="-" if s_ok(r) else ":", zorder=2)
            lim.append(np.c_[C["x"][m], C["y"][m]])
        for tag, col, lab in (("id", "#555555", "policy (identity)"), (f"xpD{d}", "#000000", f"expert from -{d} s")):
            o = sc["rolls"].get(tag)
            if o is None or o.get("ctrl") is None or tag not in rows:
                continue
            C, r = o["ctrl"], rows[tag]
            m = np.asarray(C["t"]) <= o["tz"]
            ax.plot(C["x"][m], C["y"][m], color=col, lw=1.3, ls="--" if tag == "id" else "-", zorder=3,
                    label=f"{lab}: {'strict done' if r.get('strict') or (tag == 'id' and (r.get(f'strict_D{d}') or {}).get('strict')) else r['outcome']}")
            if not m.all():
                ax.plot(C["x"][m][-1], C["y"][m][-1], "x", color=col, ms=4, zorder=4)
            if tag != "id" and r.get("t_h") is not None:
                i = int(np.searchsorted(C["t"], r["t_h"]))
                ax.plot(C["x"][i], C["y"][i], "o", color="#0072B2", ms=4, zorder=5, label=f"h (k {r.get('h')})")
            lim.append(np.c_[C["x"][m], C["y"][m]])
        ax.plot([], [], color="#009E73", lw=0.7, label=f"self variants, strict done ({n_st} / {n_ok})")
        ax.plot([], [], color="#D55E00", lw=0.7, label="self variants, failed")
        Q = np.concatenate(lim)
        c0, r0 = Q.mean(0), max(np.ptp(Q, 0).max() / 2 + 5, 20)
        ax.set_xlim(c0[0] - r0, c0[0] + r0)
        ax.set_ylim(c0[1] - r0, c0[1] + r0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])  # noqa: E702
        u = sc["unit"]
        ax.text(0.02, 0.98, f"{sc['scene'][-8:]} ({u['role']}: {u['outcome']}, k_ref {u['k_ref']}, Delta {d} s)", transform=ax.transAxes, va="top", fontsize=7)
        ax.legend(fontsize=5.0, loc="lower right", frameon=False)
    figr.tight_layout()
    PS.save(figr, str(Path(a.out).with_suffix("")))
    print("->", a.out)



def n_yield(a):
    """rewind_onpol_all (plans/2026-10-16-rewind-onpol-all-prereg.md): per Delta, the expert's completions from h on the failed units, the
    d293 gate with the PAI thresholds and with thresholds recalibrated to the nuPlan human p95 of the read's turns, zeros / at-fault after
    the handover, creep, the identity tolerance, and the label rows (2 Hz, k >= h, junction stretch) of zero-free completed continuations
    per category; writes <md>, <csv> (unit x Delta), <units> (one line per turn), <rows> (one line per label row), <json> (summary)."""
    n_paths()
    from jevdrive import stats
    import csv as C
    X = json.loads(Path(a.read).read_text())
    gp = X["gate"]
    U = {s["scene"]: s["unit"] for s in X["scenes"]}
    by = {s["scene"]: {r["tag"]: r for r in s["rows"]} for s in X["scenes"]}
    Ds = [int(d) for d in a.deltas.split(",")]
    lg = lambda s: s.rsplit("-", 1)[0]  # noqa: E731
    fin = lambda v: v is not None and np.isfinite(v)  # noqa: E731
    H = [by[s]["id"]["log_shape"] for s in U if by[s].get("id", {}).get("log_shape")]
    gn = {k: float(np.nanpercentile([h[q] for h in H], 95)) for k, q in (("kmax", "kmax"), ("dk", "dk"), ("alat", "alat"), ("v_entry", "v_0"))}

    def gate(x, g, ent_check=True):
        return bool(all(fin(x.get(k)) and x[k] <= g[k] for k in ("kmax", "dk", "alat"))
                    and (not ent_check or (fin(x.get("v_0")) and x["v_0"] <= g["v_entry"])))
    hum = dict(n=len(H), pai=sum(gate(h, gp) for h in H), nuplan=sum(gate(h, gn) for h in H))
    turns = sorted(U)
    A = [s for s in turns if U[s]["stratum"] == "A"]
    rew = [s for s in A if any(U[s]["h"].get(str(d)) is not None for d in Ds)]
    br = lambda v, ss: stats.bootstrap(np.asarray(v, float), np.array([lg(s) for s in ss]), n_boot=10_000) if len(ss) else dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan)  # noqa: E731
    ci = lambda r: "-" if not r["n"] else f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"  # noqa: E731
    cats = ("pai", "nuplan", "nogate")
    out, rows, summ = [], [], {}
    for d in Ds:
        S = dict(units=0, valid=0, tol=0, completed=[], oc={}, zero_after=0, at_fault=0, rear=0, crawl=[0, 0], gate_pai=0, gate_nuplan=0,
                 cat={c: dict(units=[], rows=0, per={}) for c in cats}, missing=[], not_rewindable=0)
        for s in A:
            u, h = U[s], U[s]["h"].get(str(d))
            if h is None:
                S["not_rewindable"] += int(u["src"] in ("zero", "off", "off_cross"))
                continue
            S["units"] += 1
            x = by[s].get(f"D{d}")
            if x is None or x.get("h") is None:
                S["missing"].append(s)
                continue
            valid = bool(x["h"] == h and not x.get("zero_before_h"))
            za = x["kz"] >= 0
            c3 = bool(valid and x["completed"] and not za)
            gpai, gnu = bool(c3 and x.get("shape_ok")), bool(c3 and gate(x, gn, x.get("ent_after", True)))
            S["valid"] += valid
            S["tol"] += bool(valid and x.get("within_tol"))
            if valid:
                S["completed"].append((s, float(x["completed"])))
                S["oc"][x["outcome"]] = S["oc"].get(x["outcome"], 0) + 1
                S["zero_after"] += za
                S["at_fault"] += bool(x.get("at_fault"))
                S["rear"] += bool(x.get("rear_after"))
                if (x.get("t_left_s") or 0) >= WIN_END:
                    S["crawl"][0] += bool(fin(x.get("v_win_end")) and x["v_win_end"] >= V_CREEP)
                    S["crawl"][1] += 1
                S["gate_pai"] += bool(x["completed"] and x.get("shape_ok"))
                S["gate_nuplan"] += bool(x["completed"] and gate(x, gn, x.get("ent_after", True)))
            ks = x.get("rows_k") or []
            for c, ok in zip(cats, (gpai, gnu, c3)):
                if ok:
                    S["cat"][c]["units"].append(s)
                    S["cat"][c]["rows"] += len(ks)
                    S["cat"][c]["per"][s] = len(ks)
            out.append(dict(scene=s, log=lg(s), src=u["src"], id_outcome=u["outcome"], k_ref=u["k_ref"], delta=d, h=h, valid=valid,
                            within_tol=x.get("within_tol"), pre_plan_maxdiff=x.get("pre_plan_maxdiff"), pre_track_maxdiff=x.get("pre_track_maxdiff"),
                            dv_h=x.get("dv_h"), v_h=x.get("v_h"), lat_h=x.get("lat_h"), st_h_rel_entry=x.get("st_h_rel_entry"),
                            completed=x["completed"], outcome=x["outcome"], kz=x["kz"], zkind=x.get("zkind"), zero_after=za,
                            at_fault=x.get("at_fault"), rear_after=x.get("rear_after"), v_win_end=x.get("v_win_end"), t_left_s=x.get("t_left_s"),
                            kmax=x.get("kmax"), dk=x.get("dk"), alat=x.get("alat"), v_0=x.get("v_0"), ent_after=x.get("ent_after"),
                            gate_pai=gpai, gate_nuplan=gnu, label=c3, rows=len(ks) if c3 else 0, run=x.get("run"), rid=x.get("rid")))
            if valid:                                    # every valid handover's junction rows before any zero; label = completed and zero_free
                rows += [dict(scene=s, log=lg(s), delta=d, h=h, k=k, run=x.get("run"), rollout_id=x.get("rid"), completed=bool(x["completed"]),
                              zero_free=not za, label=c3, gate_pai=gpai, gate_nuplan=gnu, within_tol=x.get("within_tol")) for k in ks]
        cs = [c[0] for c in S["completed"]]
        S["rate"] = br([c[1] for c in S["completed"]], cs)
        for c in cats:
            per = S["cat"][c]["per"]
            top = sorted(per.values(), reverse=True)[:3]
            S["cat"][c].update(turns=len(per), logs=len({lg(s) for s in per}), top3=(sum(top) / S["cat"][c]["rows"]) if S["cat"][c]["rows"] else np.nan)
        summ[d] = S
    uni = {}
    for c in cats:
        per = {}
        for d in Ds:
            for s, n in summ[d]["cat"][c]["per"].items():
                per[s] = per.get(s, 0) + n
        top = sorted(per.values(), reverse=True)[:3]
        tot = sum(per.values())
        uni[c] = dict(rows=tot, turns=len(per), logs=len({lg(s) for s in per}), top3=sum(top) / tot if tot else np.nan)
    oc = _count(U[s]["outcome"] for s in turns)
    src = _count(U[s]["src"] for s in A)
    L = [f"# Recovery-expert rewind on every onpol-train turn: RSG-L18-s0 (box, held-read serving, 30 decisions), expert XP-L12-E46", "",
         f"Scenes with a strict turn in the identity's 15 s window: {len(turns)} ({len({lg(s) for s in turns})} logs); identity outcome "
         + ", ".join(f"{k} {v}" for k, v in sorted(oc.items())) + f". Failed (A) {len(A)}: " + ", ".join(f"{k} {v}" for k, v in sorted(src.items()))
         + f"; handed over (event / off with h >= 1 for some Delta) {len(rew)}; short listed, not rewound.", "",
         f"Gate thresholds: PAI (d293 all-turn human p95) kmax {gp['kmax']:.3f} 1/m, dk {gp['dk']:.4f} 1/m^2, alat {gp['alat']:.2f} m/s^2, entry "
         f"{gp['v_entry']:.1f} m/s; nuPlan (human p95 of these {hum['n']} turns) kmax {gn['kmax']:.3f}, dk {gn['dk']:.4f}, alat {gn['alat']:.2f}, "
         f"entry {gn['v_entry']:.1f}. Human logs passing: PAI {hum['pai']} / {hum['n']}, nuPlan {hum['nuplan']} / {hum['n']}.", "",
         "## Expert from h = k_ref - 2 Delta (failed units)", "",
         "| Delta | units (not rewindable) | valid handovers | within tolerance | completed (log CI) | completed / event / off / short | any zero after h | at-fault after h | rear after h | creep v(h+3 s) >= 2 | completed + gate PAI | completed + gate nuPlan |",
         "|--:|--:|--:|--:|:--|:--|--:|--:|--:|:--|--:|--:|"]
    for d in Ds:
        S = summ[d]
        o = S["oc"]
        L.append(f"| {d} s | {S['units']} ({S['not_rewindable']}) | {S['valid']} | {S['tol']} | {ci(S['rate'])} | "
                 f"{o.get('completed', 0)} / {o.get('event', 0)} / {o.get('off', 0)} / {o.get('short', 0)} | {S['zero_after']} | {S['at_fault']} | {S['rear']} | "
                 f"{S['crawl'][0]} / {S['crawl'][1]} | {S['gate_pai']} | {S['gate_nuplan']} |")
    L += ["", "## Label rows (2 Hz, k >= h, rf_follow junction stretch, zero-free completed continuations)", "",
          "| Delta | category | units | rows | unique turns | logs | top-3 turns' share |", "|--:|:--|--:|--:|--:|--:|--:|"]
    nm = dict(pai="completed + gate (PAI)", nuplan="completed + gate (nuPlan)", nogate="completed, zero-free, no gate")
    for d in [*Ds, "both"]:
        for c in cats:
            r = uni[c] if d == "both" else summ[d]["cat"][c]
            L.append(f"| {d if d == 'both' else f'{d} s'} | {nm[c]} | {'-' if d == 'both' else len(r['units'])} | {r['rows']} | {r['turns']} | {r['logs']} | "
                     f"{r['top3']:.2f} |")
    if summ and any(summ[d]["missing"] for d in Ds):
        L += ["", "Missing launches (unit with h, no rollout): " + "; ".join(f"D{d}: " + ", ".join(x[:15] for x in summ[d]["missing"]) for d in Ds if summ[d]["missing"])]
    if X.get("errors"):
        L += ["", f"Read errors: {len(X['errors'])} (" + ", ".join(list(X["errors"])[:10]) + ")"]
    if a.d316:
        T = json.loads(Path(a.d316).read_text())["units"]
        old = {s for s, u in T.items() if u.get("stratum") == "A" and u.get("src") in ("zero", "off", "off_cross")}
        new = {s for s in A if U[s]["src"] in ("zero", "off", "off_cross")}
        old &= set(U) | set(X.get("nonturn", []))           # only the scenes this read covers
        L += ["", f"d316's 64 train turns (stored 18-decision units), those in this read: event / off units {len(old)}; also event / off here "
              f"{len(old & new)}; turn here but not failed {len({s for s in old if s in U and s not in new})}; no strict turn here {len(old - set(U))}."]
    zl = {d: _count((r["zkind"], "after exit" if r["kz"] > (by[r["scene"]][f"D{d}"].get("rows_k") or [-1])[-1] + 1 else "in junction")
                    for r in out if r["delta"] == d and r["valid"] and r["completed"] and r["zero_after"]) for d in Ds}
    L += ["", "Completed but excluded by a zero after h (zero kind, after the last junction row or inside it): "
          + "; ".join(f"{d} s: " + (", ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(zl[d].items())) or "none") for d in Ds)]
    Path(a.md).write_text("\n".join(L) + "\n")
    with open(a.csv, "w", newline="") as f:
        if out:
            w = C.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    with open(a.units, "w", newline="") as f:
        w = C.writer(f)
        w.writerow(["scene", "log", "outcome", "stratum", "src", "k_ref", "kz", *[f"h_D{d}" for d in Ds], "entry", "exit", "side", "n_dec",
                    "log_after_s", "video_after_s"])
        for s in turns:
            u, x = U[s], by[s].get("id", {})
            w.writerow([s, lg(s), u["outcome"], u["stratum"], u["src"], u["k_ref"], u["kz"], *[u["h"].get(str(d)) for d in Ds], u["entry"],
                        u["exit"], u["side"], x.get("n_dec"), u.get("log_after_s"), u.get("video_after_s")])
    if a.rows:
        Path(a.rows).parent.mkdir(parents=True, exist_ok=True)
        with open(a.rows, "w", newline="") as f:
            w = C.DictWriter(f, fieldnames=["scene", "log", "delta", "h", "k", "run", "rollout_id", "completed", "zero_free", "label", "gate_pai",
                                            "gate_nuplan", "within_tol"])
            w.writeheader()
            w.writerows(rows)
    J = dict(gate_pai=gp, gate_nuplan=gn, human=hum, turns=len(turns), outcome=oc, A_src=src, rewound=len(rew), union=uni,
             per_delta={d: {k: v for k, v in S.items() if k not in ("completed",)} for d, S in summ.items()})
    if a.json:
        Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("cells")
    p.add_argument("--scene", required=True); p.add_argument("--phase", choices=("A", "B"), required=True)  # noqa: E702
    p.add_argument("--pack", required=True); p.add_argument("--out", required=True)  # noqa: E702
    p.add_argument("--tfail"); p.add_argument("--smoke", action="store_true")  # noqa: E702
    p = sp.add_parser("extract")
    p.add_argument("--sd", required=True); p.add_argument("--out", required=True)  # noqa: E702
    p = sp.add_parser("tfail")
    p.add_argument("--pkl", required=True); p.add_argument("--out", required=True)  # noqa: E702
    p = sp.add_parser("read")
    p.add_argument("--root", required=True); p.add_argument("--scenes", default=""); p.add_argument("--out", required=True)  # noqa: E702
    p = sp.add_parser("table")
    p.add_argument("--read", required=True); p.add_argument("--tokyo"); p.add_argument("--md", required=True); p.add_argument("--csv", required=True)  # noqa: E702
    p = sp.add_parser("fig")
    p.add_argument("--pkl-dir", required=True); p.add_argument("--read", required=True); p.add_argument("--scenes", required=True)  # noqa: E702
    p.add_argument("--out", required=True)
    p = sp.add_parser("n-units")
    p.add_argument("--follow", required=True); p.add_argument("--label", default="RSG-L18-s0@18"); p.add_argument("--out", required=True)  # noqa: E702
    p = sp.add_parser("n-tfail")
    p.add_argument("--stage", required=True); p.add_argument("--runs", required=True); p.add_argument("--scenes", required=True)  # noqa: E702
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--kend", type=int, default=N_KEND); p.add_argument("--deltas", default="1,2,3")  # noqa: E702
    p.add_argument("--rewind", default="event,off,short,viol")
    p = sp.add_parser("n-yield")
    p.add_argument("--read", required=True); p.add_argument("--md", required=True); p.add_argument("--csv", required=True)  # noqa: E702
    p.add_argument("--units", required=True); p.add_argument("--rows", default=""); p.add_argument("--json", default="")  # noqa: E702
    p.add_argument("--deltas", default="2,3"); p.add_argument("--d316", default="")  # noqa: E702
    p = sp.add_parser("n-read")
    p.add_argument("--stage", required=True); p.add_argument("--ref", required=True); p.add_argument("--out", required=True)  # noqa: E702
    p.add_argument("--pkl", default="")
    p.add_argument("--workers", type=int, default=16)
    p = sp.add_parser("n-table")
    p.add_argument("--read", required=True); p.add_argument("--md", required=True); p.add_argument("--csv", required=True)  # noqa: E702
    p.add_argument("--stored", default="")
    p = sp.add_parser("n-self-units")
    p.add_argument("--out", required=True); p.add_argument("--runs", required=True); p.add_argument("--scenes", required=True)  # noqa: E702
    p.add_argument("--tfail", required=True); p.add_argument("--workers", type=int, default=16)  # noqa: E702
    p = sp.add_parser("n-self-files")
    p.add_argument("--units", required=True); p.add_argument("--list", required=True); p.add_argument("--out", required=True)  # noqa: E702
    p.add_argument("--variants", default=""); p.add_argument("--n", type=int, default=S_N); p.add_argument("--smoke", action="store_true")  # noqa: E702
    p = sp.add_parser("n-self-read")
    p.add_argument("--stage", required=True); p.add_argument("--units", required=True); p.add_argument("--out", required=True)  # noqa: E702
    p.add_argument("--stored", default=""); p.add_argument("--pkl", default=""); p.add_argument("--workers", type=int, default=16)  # noqa: E702
    p = sp.add_parser("n-self-check")
    p.add_argument("--read", required=True); p.add_argument("--launches", default=""); p.add_argument("--out", default="")  # noqa: E702
    p.add_argument("--stage", action="store_true")
    p = sp.add_parser("n-self-table")
    p.add_argument("--read", required=True); p.add_argument("--md", required=True); p.add_argument("--csv", required=True)  # noqa: E702
    p.add_argument("--d316", default="")
    p = sp.add_parser("n-self-fig")
    p.add_argument("--read", required=True); p.add_argument("--units", required=True); p.add_argument("--out", required=True)  # noqa: E702
    p = sp.add_parser("n-fig")
    p.add_argument("--pkl", required=True); p.add_argument("--read", required=True); p.add_argument("--out", required=True)  # noqa: E702
    a = ap.parse_args()
    dict(cells=cells, extract=extract, tfail=tfail, read=read, table=table, fig=fig, **{"n-units": n_units, "n-tfail": n_tfail, "n-read": n_read,
         "n-table": n_table, "n-fig": n_fig, "n-self-units": n_self_units, "n-self-files": n_self_files, "n-self-read": n_self_read,
         "n-self-check": n_self_check, "n-self-table": n_self_table, "n-self-fig": n_self_fig, "n-yield": n_yield})[a.cmd](a)


if __name__ == "__main__":
    main()
