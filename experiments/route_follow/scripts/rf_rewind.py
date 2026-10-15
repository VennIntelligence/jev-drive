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
    a = ap.parse_args()
    dict(cells=cells, extract=extract, tfail=tfail, read=read, table=table, fig=fig)[a.cmd](a)


if __name__ == "__main__":
    main()
