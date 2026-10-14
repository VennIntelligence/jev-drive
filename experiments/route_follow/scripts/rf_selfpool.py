#!/usr/bin/env python3
"""Turning school phase 2 (P4) row builder: completing self-branch rollouts of the AlpaSim PAI simulator -> the on-policy pool `on1_<name>`
(contract experiments/onpol_nuplan/plans/on_pool_format.md, as rf_pait.py pack writes it), with gates. Row definition and gate rules:
research/next-round/turning-school-rows.md, section "on1_SELF". Input = the run dirs of a branch search (P3: rf_p3.sh, P4: rf_p4.sh), their
read.json (P3: rf_branch.py read; P4: rf_p4.py read / merge) and an rf_p4.py plan.json (route grid, strict turn entry / exit, topo, v_e, gate).

  build    the whole chain (below), resumable stage by stage; one command per backend:
             tokyo  docker images (alpasim-base for the replay / route, the run's driver image for the cold-start slots), /data/envs/op-train
             box    native (AlpaSim .venv, envs/op-train); submit it to the GPU pool (the token encoder needs a card)
           rf_selfpool.py build --roots <run root> [...] --tag on1_SELF [--read <read.json> ...] [--plan <plan.json>] [--served <code dir>]
  select   (numpy)  read.json rows -> <work>/branches.json: every completing rollout (`done`) with its run dir, cell, handover and the
           decision-293 shape gate G-shape on its realized track (rf_shape.track_shape / speeds on the plan route, cut at the first zero
           event; thresholds = the plan's `gate`: class p95 kappa / dkappa (all-turn p95 for classes with < 5 human turns), a_lat <= all-turn
           p95 4.42, realized entry speed <= v_e + 1); <work>/failing.json: the failing rollouts per scene (a later critic line). Refuses
           hard_exam / control scenes unless --allow-exam (dev only).
  sim      (AlpaSim env)  per branch (= one session): onp_rows.scene_rows (the inputs replayed through pai_driver from rollout.asl, labels)
           with c1_extract's metrics read pinned to the session (a P4 launch runs several sessions of one scene) -> <stage>/<key>.npz,
           key = <scene>~<session[:8]>; then rf_pait.route_one on it (the runtime's route generator at the row anchors: route0_d, labels)
           -> <work>/route/<key>/route/<scene>.npz
  cold     (driver env)  onp_rows.py cold: the cold-start rows' slots as served, the final <work>/rows/self/<key>.npz
  cache    (op-train, GPU)  onp_cache.py build --pass self: Cinque tokens + teacher of every row -> <work>/cache/self/{tok.npy, tab.npz}
  pack     (op-train, CPU)  rows -> pool, gates -> <out>/pool_gates_<name>.json, coverage.md, the BEV sheet.

Row = one decision of a completing branch, inside the window [k0, exit + PAD] where k0 = the handover h (src log: the ego followed the log
to h) or the earlier of h and the decision at which the ego's route station crosses entry - 40 m (src own: the policy drove itself there);
anchor = the policy's own state at the decision (fed inputs as served, tokens of the frames it saw); `fut` = the realized track (the driver's
egomotion anchors) at t0 + 0.5 k s, k = 1..8, in the anchor's rig frame (speed in the spacing); `served` = the policy's raw plan.
Kept only with 8 valid slots, before any scorer event / end, the 4 s future inside the rollout and before its first scorer event.
Route memory input = rmem.Tracker on the driver's logged route messages (rf_pai_pool's path; gate: equal to the served `rmem`); route
labels = rm_build.label on the regenerated route (rf_pait); G-route = regenerated first route point vs the logged route0.
Tags (tab, strings): log = scene, branch = <scene[7:15]>~<session[:8]> (20 chars: rf_pait merge keeps string tags as U20), cell (g<gain>-s<scale> P3, g<gain>-<raw|ve|ve08> P4), D (D40 .. D0, Did = no handover),
src (own | log), phase (approach < entry <= inside < exit <= exit), push (on: the served plan was transformed at this decision), topo, part;
d_entry (float, entry - station at the anchor).

  rf_selfpool.py build --roots /data/runs/route_night/p3 --plan <plan_p3.json> --served /data/runs/route_night/p3/code --tag on1_SELFdev \
      --allow-exam --work /data/runs/route_night/p4_self/dev --out experiments/route_follow/results/p4_self/pool_dev
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R = Path(os.environ.get("SELF_REPO") or Path(__file__).resolve().parents[3])
HERE = R / "experiments/route_follow/scripts"
sys.path[:0] = [str(R), str(HERE), str(R / "experiments/onpol_pai/scripts"), str(R / "experiments/onpol_pai/lib"), str(R / "experiments/route_mem/lib"),
                str(R / "experiments/route_mem/scripts"), str(R / "experiments/anchor_offset/scripts")]
D = Path(os.environ.get("DATA_DIR", "/data"))
CR, FEAT = D / "runs/op_parity/cache", D / "runs/route_mem/feat"
EXAM = R / "experiments/alpasim/results/pai_exam"
SHAPE = HERE.parent / "results/p3_branch/shape.json"
PRE, PAD, CRAWL, ARC, R0TOL, RMEM_TOL = 40.0, 10.0, 0.5, 0.10, 0.05, 1e-3   # RMEM_TOL: the log rounds rmem to 5 and the message poses to 6 decimals
PASS = "self"


def jload(f):
    return json.load(open(f))


def scenes_in(*files):
    out = set()
    for f in files:
        if Path(f).exists():
            out |= {w for l in open(f) for w in l.split() if w.startswith("clipgt-")}
    return out


def part_of(sc):
    for p in ("hard_exam", "control", "hard_train"):
        if sc in scenes_in(EXAM / f"{p}.txt"):
            return p
    return "train"


# ---------------------------------------------------------------- select
def cell_of(r):
    """A read.json row (P3 rf_branch / P4 rf_p4) -> dict(cell, g, lon, D, src, h); P3 arms are `d<D>@g<g>s<s>`, P4 tags `D<D>-<src>-g<g>-<lon>`."""
    if "arm" in r:
        ho, tf = r["arm"].split("@")
        g, s = tf[1:].split("s")
        return dict(cell=f"g{float(g):g}-s{float(s):g}", g=float(g), lon=f"s{float(s):g}", D=f"D{int(ho[1:])}", src="log", h=int(r["k"]))
    tag = r["tag"]
    if tag == "id":
        return dict(cell="g1-raw", g=1.0, lon="raw", D="Did", src="own", h=None)
    _, src, g, lon = tag.split("-")
    return dict(cell=f"{g}-{lon}", g=float(g[1:]), lon=lon, D=f"D{int(r['D'])}", src="own" if src == "self" else "log", h=r.get("h"))


def run_index(roots):
    """rollout id -> run dir (the dir holding sim/ and driver/) under the roots (walk stops at a run dir)."""
    out = {}
    for root in roots:
        for d, sub, _ in os.walk(root):
            if "sim" in sub and os.path.exists(os.path.join(d, "sim", "aggregate", "results-summary.json")):
                for r in jload(os.path.join(d, "sim", "aggregate", "results-summary.json"))["rollouts"]:
                    out.setdefault(r["rollout_id"], (d, r))
                sub[:] = []
            else:
                sub[:] = [s for s in sub if s not in ("rollouts", "dump", "driver", "driver-logs", "nre-cache", "pool", "parts")]
    return out


def gate_of(p, J):
    """The plan's own gate (rf_p4.plan: P4's instrument), else the same rule on shape.json (decision 293)."""
    if p.get("gate") and p.get("topo"):
        return dict(p["gate"], v_e=float(p["v_e"]), src_gate="plan")
    topo = p.get("topo") or "all"
    n = J["counts"]["topo"].get(topo, 0)
    b = J["band_topo"][topo] if n >= 5 else J["band_all"]
    v_e = J["entry_targets"].get(topo, J["entry_targets"]["all"])["0"]["median"]
    return dict(kmax=b["kmax"][1], dk=b["dk"][1], alat=J["band_all"]["alat"][1], v_entry=v_e + 1.0, v_e=v_e, src="class" if n >= 5 else "all", src_gate="shape.json")


def cmd_select(a):
    import rf_pai_follow as F
    import rf_shape as SH
    PL, J = jload(a.plan)["scenes"], jload(a.shape)
    rows = [dict(r, _read=f) for f in a.read for r in jload(f)["rows"]]
    idx = run_index(a.roots)
    exam = scenes_in(EXAM / "hard_exam.txt", EXAM / "control.txt")
    keep, failing, skipped = [], {}, []
    for r in rows:
        sc, rid = r["scene"], r["rid"]
        c = cell_of(r)
        if sc not in PL:
            skipped.append((sc, rid, "scene not in the plan"))
            continue
        if not r.get("done"):
            failing.setdefault(sc, []).append(dict(rid=rid, **c, score=r.get("score"), zero=r.get("zero") or r.get("end"), t_zero=r.get("t_zero"),
                                                  run=idx.get(rid, (None,))[0], read=r["_read"]))
            continue
        if rid not in idx:
            skipped.append((sc, rid, "run dir not found under --roots"))
            continue
        assert a.allow_exam or sc not in exam, f"{sc} is a hard_exam / control scene: never a training row (--allow-exam: dev only)"
        run, res = idx[rid]
        p = PL[sc]
        Rg, Sg = np.c_[p["x"], p["y"]], np.asarray(p["u"], float)
        C = F.ctrl(os.path.join(run, "sim"), rid)
        t = np.asarray(C["t"], float)
        xy = np.c_[C["x"], C["y"]]
        tz = t[0] + r["t_zero"] * 1e6 if r.get("t_zero") is not None else np.inf
        sp, _ = F.track(Rg, Sg, xy)
        shp = SH.track_shape(t, xy, C["vx"], sp, p["s_entry"] - SH.WIN_PRE, p["s_exit"], tz)
        spd = SH.speeds(t, sp, C["vx"], p["s_entry"], p["s_exit"], tz)
        g = gate_of(p, J)
        gk = dict(kmax=bool(shp["kmax"] <= g["kmax"]), dk=bool(shp["dk"] <= g["dk"]), alat=bool(shp["alat"] <= g["alat"]),
                  v_entry=bool(np.isfinite(spd["v_0"]) and spd["v_0"] <= g["v_entry"]))
        rec = json.load(open(os.path.join(run, "run.json"))) if os.path.exists(os.path.join(run, "run.json")) else {}
        keep.append(dict(key=f"{sc}~{rid[:8]}", scene=sc, rid=rid, run=run, **c, topo=p.get("topo") or "all", side=p.get("side"), part=part_of(sc),
                         s_entry=p["s_entry"], s_exit=p["s_exit"], score=r.get("score"), t_zero=r.get("t_zero"), tag=rec.get("tag", "RSG-L18-s0"),
                         image=rec.get("image"), gate=g, shape={k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in
                                                                  dict(kmax=shp["kmax"], alat=shp["alat"], dk=shp["dk"], v_entry=spd["v_0"], v_min=spd["v_min"]).items()},
                         gate_parts=gk, g_shape=all(gk.values()), read=r["_read"]))
    W = Path(a.work)
    W.mkdir(parents=True, exist_ok=True)
    out = dict(plan=str(a.plan), shape=str(a.shape), roots=[str(x) for x in a.roots], read=[str(x) for x in a.read], branches=keep, skipped=skipped)
    (W / "branches.json").write_text(json.dumps(out, indent=1))
    (W / "failing.json").write_text(json.dumps(dict(note="failing rollouts (read.json done false) per scene, for a later critic line; no rows built",
                                                    scenes=failing), indent=1))
    n_f = sum(map(len, failing.values()))
    print(f"select: {len(keep)} completing rollouts over {len({b['scene'] for b in keep})} scenes, G-shape passes {sum(b['g_shape'] for b in keep)}; "
          f"{n_f} failing rollouts over {len(failing)} scenes -> failing.json; skipped {len(skipped)}: {skipped[:5]}")


# ---------------------------------------------------------------- sim (AlpaSim env): rows stage 1 + route generator
def sim_one(j):
    """One branch: onp_rows.scene_rows on its session (metrics pinned), the staged npz, then rf_pait.route_one on it."""
    import glob as _glob
    import traceback
    import types
    t = time.time()
    try:
        import onp_rows as OR
        import c1_extract as X
        met = f"{j['run']}/sim/rollouts/{j['scene']}/{j['session']}/metrics.parquet"
        X.glob = types.SimpleNamespace(glob=lambda p: [met] if p.endswith("/metrics.parquet") else _glob.glob(p))
        stg = Path(j["stage"]) / f"{j['key']}.npz"
        if not stg.exists():
            Z = OR.scene_rows(j)
            Z["meta"] = json.dumps(dict(json.loads(Z["meta"]), secs_stage1=round(time.time() - t, 2), key=j["key"]))
            tmp = stg.with_name(f".{j['key']}.{os.getpid()}.tmp.npz")
            np.savez(tmp, **Z)
            os.replace(tmp, stg)
        import rf_pait
        ro = Path(j["route"]) / j["key"]
        (ro / "route").mkdir(parents=True, exist_ok=True)
        if not (ro / "route" / f"{j['scene']}.npz").exists():
            rf_pait.route_one((stg, ro))
        m = json.loads(str(np.load(stg)["meta"]))
        return j["key"], m, None
    except Exception:
        return j["key"], None, traceback.format_exc()


def cmd_sim(a):
    import onp_rows as OR
    B = jload(Path(a.work) / "branches.json")["branches"]
    stage, rows = Path(a.stage) / PASS, Path(a.work) / "rows" / PASS
    stage.mkdir(parents=True, exist_ok=True), rows.mkdir(parents=True, exist_ok=True)
    say = OR._say(Path(a.work))
    jobs = []
    for b in B:
        if (rows / f"{b['key']}.npz").exists() and (Path(a.work) / "route" / b["key"] / "route" / f"{b['scene']}.npz").exists():
            continue
        sm = Path(b["run"]) / "sim"
        res = next(r for r in jload(sm / "aggregate/results-summary.json")["rollouts"] if r["rollout_id"] == b["rid"])
        ev = sm / "eval-config.yaml"
        cor = next((float(x.split(":")[1]) for x in ev.read_text().splitlines() if "max_dist_to_gt_trajectory" in x), np.nan) if ev.exists() else np.nan
        jobs.append(dict(run=b["run"], scene=b["scene"], session=b["rid"], key=b["key"], tag=b["tag"], summary=res, corridor=cor, out=str(rows.parent),
                         stage=str(stage), route=str(Path(a.work) / "route"), cold=a.cold, lht=False, nurec=a.nurec, **{"pass": PASS}))
    n = max(1, min(a.jobs, len(jobs)))
    say(f"sim: {len(B)} branches, {len(B) - len(jobs)} done, {len(jobs)} to build with {n} workers", True)
    bad, dt = OR._pool(sim_one, jobs, n, say, lambda k, m: f"{m['n_rows']} rows, G-in dump {m['gin_dump_eq']}/{m['gin_dump_n']} rec {m['gin_rec_eq']}/{m['gin_rec_n']}",
                       "sim") if jobs else ([], 0.0)
    say(f"sim done: {len(jobs) - len(bad)} staged + routed, {len(bad)} failed, {dt:.0f} s", True)
    if bad:
        raise SystemExit(f"sim failed for {bad}")


# ---------------------------------------------------------------- pack (op-train)
def drive_info(run, sessions):
    """drive.jsonl of a run dir -> per session: route messages [(ts, pose, wp)], served rmem {t0: f}, pushed decisions {k}."""
    out = {s: dict(msgs=[], served={}, push=set()) for s in sessions}
    for line in open(Path(run) / "driver" / "drive.jsonl"):
        if not any(s in line for s in sessions):
            continue
        r = json.loads(line)
        o = out.get(r.get("session"))
        if o is None:
            continue
        if r.get("kind") == "route":
            o["msgs"].append((int(r["ts"]), np.array(r["pose"], float), np.array(r["wp"], float).reshape(-1, 2)))
        elif r.get("kind") == "drive":
            if "rmem" in r:
                o["served"][int(r["t0"])] = np.array(r["rmem"], np.float32)
            if "poses_branch" in r:
                o["push"].add(int(r["k"]))
    for o in out.values():
        o["msgs"].sort(key=lambda x: x[0])
    return out


def realized(t0, A, tq):
    """Anchor track (t0 (N,), A (N, 3) local x, y, yaw) at times tq (N, m) -> local poses (N, m, 3), valid mask, the anchors with the same
    unwrapped yaw (so the relative yaw of the future is continuous)."""
    Au = np.c_[A[:, :2], np.unwrap(A[:, 2])]
    P = np.stack([np.interp(tq, t0, Au[:, i]) for i in range(3)], -1)
    return P, (tq >= t0[0]) & (tq <= t0[-1]), Au


def to_rig(P, a):
    c, s = np.cos(a[2]), np.sin(a[2])
    d = P[..., :2] - a[:2]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], P[..., 2] - a[2]], -1)


def bev(path, picks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rmem as RM
    fig, axs = plt.subplots(2, 3, figsize=(13, 10.5))
    for ax, q in zip(axs.flat, picks):
        for e in q["E"][q["eo"]]:
            c, s, hl = np.cos(e[2]), np.sin(e[2]), e[3] / 2
            ax.plot([-(e[1] - s * hl), -(e[1] + s * hl)], [e[0] - c * hl, e[0] + c * hl], "-", color="#888888", lw=1.0)
        ax.plot(-q["path"][q["po"], 1], q["path"][q["po"], 0], "-", color="#1f77b4", lw=1.0, alpha=0.5, label="human log path")
        lb = q["lab"][np.isfinite(q["lab"]).all(1)]
        ax.plot(-lb[:, 1], lb[:, 0], "--", color="#bcbd22", lw=1.0, label="route (label)")
        f = q["f"].reshape(RM.N_PT, RM.NF)
        v = f[:, 0] > 0.5
        ax.plot(-f[v, 2] * 20, f[v, 1] * 20, ".", color="#ff7f0e", ms=3, label="route memory (input, as served)")
        for b in q["B"][q["bo"]]:
            c, s = np.cos(b[2]), np.sin(b[2])
            cx, cy = np.array([1, 1, -1, -1, 1]) * b[3] / 2, np.array([1, -1, -1, 1, 1]) * b[4] / 2
            ax.plot(-(b[1] + s * cx + c * cy), b[0] + c * cx - s * cy, "-", color="#9467bd", lw=0.7)
        sv = q["served"]
        ax.plot(np.r_[0, -sv[:, 1]], np.r_[0, sv[:, 0]], "o-", color="#2ca02c", ms=3, lw=1.0, label="policy's own plan (raw)")
        fu = q["fut"]
        ax.plot(np.r_[0, -fu[:, 1]], np.r_[0, fu[:, 0]], "s-", color="#d62728", ms=3, lw=1.4, label="demo = realized branch 4 s (target)")
        for k in range(8):
            ax.annotate(f"{(k + 1) * 0.5:g}", (-fu[k, 1], fu[k, 0]), fontsize=5.5, color="#d62728", xytext=(3, 0), textcoords="offset points")
        ax.plot([0], [0], "k^", ms=8)
        ax.set_xlim(-30, 30), ax.set_ylim(-10, 70), ax.set_aspect("equal")
        ax.set_title(q["title"], fontsize=7.5)
        ax.set_xlabel("-y (m, right +)"), ax.set_ylabel("x (m)")
    axs.flat[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def counts(x):
    u, c = np.unique(np.asarray(x).astype(str), return_counts=True)
    return {str(k): int(v) for k, v in sorted(zip(u, c), key=lambda z: -z[1])}


def cmd_pack(a):
    import rf_pai_follow as F
    import rf_pait as P
    import rmem as RM
    W, out = Path(a.work), Path(a.out)
    BJ = jload(W / "branches.json")
    B = {b["key"]: b for b in BJ["branches"]}
    PL = jload(BJ["plan"])["scenes"]
    T = np.load(W / "cache" / PASS / "tab.npz")
    stems = T["scenes"][T["sid"]]
    tok = np.load(W / "cache" / PASS / "tok.npy", mmap_mode="r")
    feat_src = a.feat
    keys = [k for k in B if k in set(stems.tolist())]
    missing = [k for k in B if k not in set(stems.tolist())]
    di = {}
    for run in sorted({B[k]["run"] for k in keys}):
        di.update(drive_info(run, [B[k]["rid"] for k in keys if B[k]["run"] == run]))
    sel, cols, per_b = [], {c: [] for c in ("fut", "F", "Freg", "lab", "r0d", "dent", "phase", "push", "ctrl4", "rmemd")}, {}
    gin = dict(dump_n=0, dump_eq=0, rec_n=0, rec_eq=0)
    for key in keys:
        b, p = B[key], PL[B[key]["scene"]]
        idx = np.flatnonzero(stems == key)
        idx = idx[np.argsort(T["k"][idx])]
        Z = np.load(W / "rows" / PASS / f"{key}.npz")
        t0, A, kk = Z["t0_us"].astype(np.int64), Z["anchor"].astype(np.float64), Z["k"].astype(int)
        assert np.array_equal(T["t0_us"][idx], t0) and np.array_equal(T["k"][idx], kk), f"{key}: cache rows != row file"
        m = json.loads(str(Z["meta"]))
        for q in ("dump_n", "dump_eq", "rec_n", "rec_eq"):
            gin[q] += int(m[f"gin_{q}"])
        Rg, Sg = np.c_[p["x"], p["y"]], np.asarray(p["u"], float)
        sig, _ = F.track(Rg, Sg, A[:, :2])
        se, sx = float(p["s_entry"]), float(p["s_exit"])
        # window
        h = b["h"]
        kh = np.flatnonzero(kk >= h)[0] if h is not None and (kk >= h).any() else len(kk)
        if b["src"] == "own":
            c40 = np.flatnonzero(sig >= se - PRE)
            k0 = min(kh, c40[0] if len(c40) else len(kk))
        else:
            k0 = kh
        past = np.flatnonzero(sig > sx + a.pad)
        k1 = past[0] if len(past) else len(kk)
        win = np.zeros(len(kk), bool)
        win[k0:k1] = True
        # realized future (the driver's own egomotion anchors) and the controller trace (independent read of the same track)
        tq = t0[:, None] + 500_000 * np.arange(1, 9)[None]
        Pf, okf, Au = realized(t0, A, tq)
        fut = np.stack([to_rig(Pf[n], Au[n]) for n in range(len(kk))]).astype(np.float32)
        C = F.ctrl(Path(b["run"]) / "sim", b["rid"])
        cxy = np.stack([np.interp(tq, C["t"], C["x"]), np.interp(tq, C["t"], C["y"])], -1)
        c0 = np.stack([np.interp(t0, C["t"], C["x"]), np.interp(t0, C["t"], C["y"])], -1)
        ctrl4 = np.stack([to_rig(np.concatenate([cxy[n], np.zeros((8, 1))], -1), A[n])[:, :2] for n in range(len(kk))])
        ctrl0 = np.stack([to_rig(np.r_[c0[n], 0.0][None], A[n])[0, :2] for n in range(len(kk))])
        tev = int(Z["t_event_us"].min())
        ok = (T["usable"][idx] & ~T["after_event"][idx] & ~T["after_end"][idx] & okf.all(1) & (tq[:, -1] < tev))
        # G-crawl: realized speed at the anchor and on the future's segments that end before the entry
        sig_f = np.interp(tq, t0, sig)
        vseg = np.hypot(*np.diff(np.concatenate([np.zeros((len(kk), 1, 2)), fut[..., :2]], 1), axis=1).transpose(2, 0, 1)) / 0.5
        v0 = T["v0"][idx].astype(float)
        vpre = np.where(sig_f < se, vseg, np.inf).min(1)
        crawl = (sig < se) & (np.minimum(v0, vpre) < CRAWL * b["gate"]["v_e"])
        # route: input from the logged messages (as served), labels / route0 from the regenerated route
        info = di[b["rid"]]
        tr, pose = RM.Tracker(), {ts: q for ts, q, _ in info["msgs"]}
        for ts, _, wp in info["msgs"]:
            tr.add(ts, wp)
        Fl = np.stack([tr.features(int(t0[n]), pose.get, A[n]) for n in range(len(kk))])
        rmemd = np.array([float(np.abs(info["served"][int(t0[n])] - Fl[n]).max()) if int(t0[n]) in info["served"] else np.nan for n in range(len(kk))])
        Zr = np.load(W / "route" / key / "route" / f"{b['scene']}.npz")
        pos = {int(t): i for i, t in enumerate(Zr["t0_us"])}
        jr = np.array([pos[int(t)] for t in t0])
        phase = np.where(sig < se, "approach", np.where(sig < sx, "inside", "exit"))
        push = np.array(["on" if int(k) in info["push"] else "off" for k in kk])
        n_win, n_ok = int(win.sum()), int((win & ok).sum())
        keep = win & ok & b["g_shape"] & ~crawl
        per_b[key] = dict(scene=b["scene"], cell=b["cell"], D=b["D"], src=b["src"], h=h, k0=int(kk[k0]) if k0 < len(kk) else None,
                          k1=int(kk[k1 - 1]) if k1 > 0 else None, g_shape=b["g_shape"], gate_parts=b["gate_parts"], shape=b["shape"],
                          rows_window=n_win, rows_usable=n_ok, rows_crawl=int((win & ok & crawl).sum()), rows=int(keep.sum()),
                          rows_shape_dropped=0 if b["g_shape"] else int((win & ok & ~crawl).sum()),
                          ctrl_vs_anchor_m=float(np.median(np.hypot(*ctrl0[win].T))) if win.any() else None)
        sel.append(idx[keep])
        for c, v in (("fut", fut), ("F", Fl), ("Freg", Zr["f"][jr]), ("lab", Zr["lab"][jr]), ("r0d", Zr["route0_d"][jr]), ("dent", (se - sig).astype(np.float32)),
                     ("phase", phase), ("push", push), ("ctrl4", ctrl4), ("rmemd", rmemd)):
            cols[c].append(v[keep])
    sel = np.concatenate(sel)
    X = {c: np.concatenate(v) for c, v in cols.items()}
    n = len(sel)
    assert n, "no rows"
    st, kk = stems[sel], T["k"][sel].astype(int)
    sc = np.array([s.split("~")[0] for s in st.tolist()])
    bk = lambda f: np.array([B[s][f] for s in st.tolist()])  # noqa: E731
    names = np.array([f"on:{PASS}:{s}:{k}" for s, k in zip(st, kk)])
    assert len(set(names.tolist())) == n
    fut, F_in = X["fut"], (X["F"] if feat_src == "logged" else X["Freg"]).astype(np.float32)
    assert np.isfinite(fut).all()
    # ---- gates
    arc_d, arc_c = P.chord(fut), P.chord(np.concatenate([X["ctrl4"], np.zeros((n, 8, 1))], -1))
    sp_ok = np.where(arc_c >= 1.0, np.abs(arc_d / np.maximum(arc_c, 1e-6) - 1) <= ARC, np.abs(arc_d - arc_c) <= 0.1)
    E, eo = T["edge_box"][sel], T["edge_ok"][sel]
    xd, xs = P.crosses(fut, E, eo), P.crosses(T["poses"][sel].astype(np.float32), E, eo)
    r0 = X["r0d"]
    rmd = X["rmemd"][np.isfinite(X["rmemd"])]
    regd = np.abs(X["Freg"] - X["F"]).max(1)
    nb = {k: v for k, v in per_b.items()}
    G = dict(name=a.name, pool=str(Path(a.pool_root) / f"on1_{a.name}"), rows=n, branches=len(B), branches_with_rows=int(sum(v["rows"] > 0 for v in nb.values())),
             branches_missing_in_cache=missing, scenes=int(len(np.unique(sc))), feat_source=feat_src,
             G_in=dict(gin, pass_=gin["dump_n"] == gin["dump_eq"] and gin["rec_n"] == gin["rec_eq"] and gin["rec_n"] > 0),
             G_route=dict(rows_with_route0=int(np.isfinite(r0).sum()), share_le_005=float((r0[np.isfinite(r0)] <= R0TOL).mean()) if np.isfinite(r0).any() else None,
                          max=float(np.nanmax(r0)) if np.isfinite(r0).any() else None),
             G_rmem=dict(rows_compared=int(len(rmd)), max_abs_diff=float(rmd.max()) if len(rmd) else None, tol=RMEM_TOL,
                         regen_vs_served_share_over_002=float((regd > 0.02).mean()), regen_vs_served_p99=float(np.quantile(regd, 0.99))),
             G_road=dict(demo_cross_share=float(xd.mean()), served_plan_cross_share=float(xs.mean()), rows_with_edges=int(eo.any(1).sum())),
             G_speed=dict(within_10pct=float(sp_ok.mean()), median_ratio=float(np.median(arc_d[arc_c >= 1] / arc_c[arc_c >= 1])), stopped_rows=int((arc_c < 1).sum()),
                          v0_median=float(np.median(T["v0"][sel])), demo_v_mean_median=float(np.median(arc_d / 4.0)),
                          ctrl_vs_anchor_median_m=float(np.nanmedian([v["ctrl_vs_anchor_m"] for v in nb.values() if v["ctrl_vs_anchor_m"] is not None]))),
             G_shape=dict(branches=len(B), pass_=int(sum(b["g_shape"] for b in B.values())),
                          fail_parts={q: int(sum(not b["gate_parts"][q] for b in B.values())) for q in ("kmax", "dk", "alat", "v_entry")},
                          rows_dropped=int(sum(v["rows_shape_dropped"] for v in nb.values())),
                          gate_source=sorted({b["gate"]["src_gate"] for b in B.values()})),
             G_crawl=dict(rows_dropped=int(sum(v["rows_crawl"] for v in nb.values() if v["g_shape"])), rule=f"approach rows: min(v0, future segment speeds before the entry) >= {CRAWL} v_e"),
             coverage=dict(turns_with_rows=int(len(np.unique(sc))), rows_by_phase=counts(X["phase"]), rows_by_D=counts(bk("D")), rows_by_cell=counts(bk("cell")),
                           rows_by_src=counts(bk("src")), rows_by_push=counts(X["push"]), rows_by_topo=counts(bk("topo")), rows_by_part=counts(bk("part")),
                           rows_by_scene=counts(sc), rows_by_branch=counts(st),
                           rows_by_d_entry={"> 40 m": int((X["dent"] > 40).sum()), "40 .. 20": int(((X["dent"] <= 40) & (X["dent"] > 20)).sum()),
                                            "20 .. 0": int(((X["dent"] <= 20) & (X["dent"] > 0)).sum()), "inside / exit (<= 0)": int((X["dent"] <= 0).sum())}),
             per_branch=nb)
    G["G_route"]["pass_"] = G["G_route"]["share_le_005"] is not None and G["G_route"]["share_le_005"] >= 0.99
    G["G_rmem"]["pass_"] = G["G_rmem"]["max_abs_diff"] is not None and G["G_rmem"]["max_abs_diff"] <= RMEM_TOL
    G["G_road"]["pass_"] = G["G_road"]["demo_cross_share"] <= 0.01
    G["G_speed"]["pass_"] = G["G_speed"]["within_10pct"] >= 0.99
    tags = dict(log=sc, branch=np.array([f"{x[7:15]}~{x.split('~')[1]}" for x in st.tolist()]).astype("U20"), cell=bk("cell").astype("U20"), D=bk("D").astype("U8"), src=bk("src").astype("U8"), phase=X["phase"].astype("U10"),
                push=X["push"].astype("U4"), topo=bk("topo").astype("U20"), part=bk("part").astype("U12"))
    pool, warp = Path(a.pool_root) / f"on1_{a.name}", Path(a.pool_root) / f"on1_{a.name}@warp"
    pool.mkdir(parents=True, exist_ok=True), warp.mkdir(parents=True, exist_ok=True)
    fdir = Path(a.feat_root)
    fdir.mkdir(parents=True, exist_ok=True)
    np.savez(pool / "tab.npz", names=names, ego=T["ego_in"][sel].astype(np.float32), fut=fut, cam=T["cam_t"][sel].astype(np.float32), lht=T["lht"][sel].astype(bool),
             speed=T["v0"][sel].astype(np.float32), served=T["poses"][sel].astype(np.float32), d_entry=X["dent"], **tags)
    fr = np.lib.format.open_memmap(warp / "front.npy", "w+", np.float16, (n, 8) + tok.shape[1:])
    tix = T["tix"][sel]
    for i in range(0, n, 512):
        fr[i:i + 512] = tok[tix[i:i + 512]]
    fr.flush()
    del fr
    np.savez(warp / "teacher.npz", out=T["t_out"][sel].astype(np.float32), plan=np.zeros((n, 33, 15), np.float32), di=T["di"], pi=T["pi"])
    np.savez(pool / "agent.npz", tokens=names, ok=np.ones(n, bool), box=T["box"][sel].astype(np.float32), valid=T["box_ok"][sel].astype(bool))
    np.savez(pool / "road.npz", tokens=names, ok=np.zeros(n, bool), sdf=np.zeros((n, 128, 96), np.float16))
    LB = X["lab"]
    fok = F_in[:, 0::5].sum(1) > 0
    sm = RM.summary(F_in)
    np.savez(pool / "route.npz", tokens=names, ok=fok & (np.isfinite(LB).all(-1).sum(1) >= 2), poly=LB, seglen=np.linalg.norm(np.diff(LB, axis=1), axis=-1).astype(np.float32),
             dz=np.ones(n, np.float32), cov20=np.asarray(sm["c20"], bool) & fok)
    np.savez(fdir / f"on1_{a.name}.npz", names=names, f=F_in.astype(np.float16), k=kk, ok=fok)
    if set(tags["part"].tolist()) & {"hard_exam", "control"}:
        (pool / "EXAM_SCENES_INSIDE").write_text("dev pool: hard_exam / control scenes inside; never train on it\n")
    fr = np.load(warp / "front.npy", mmap_mode="r")
    rs = np.random.default_rng(0).choice(n, min(n, 256), replace=False)
    G["G_r1"] = dict(front_equal=bool(all(np.array_equal(fr[i], tok[tix[i]]) for i in rs)), rows_checked=int(len(rs)),
                     ego_equal=bool(np.array_equal(np.load(pool / "tab.npz")["ego"], T["ego_in"][sel].astype(np.float32))),
                     feat_f16_max_diff=float(np.abs(np.load(fdir / f"on1_{a.name}.npz")["f"].astype(np.float32) - F_in).max()))
    G["G_r1"]["pass_"] = G["G_r1"]["front_equal"] and G["G_r1"]["ego_equal"]
    # ---- BEV sheet: 6 rows of 6 different branches, two per phase where possible
    rng = np.random.default_rng(0)
    pick, used = [], set()
    for ph in ("approach", "inside", "exit") * 2:
        c = [i for i in np.flatnonzero(X["phase"] == ph) if st[i] not in used]
        if c:
            i = int(rng.choice(c))
            pick.append(i), used.add(st[i])
    while len(pick) < min(6, n):
        i = int(rng.integers(n))
        if i not in pick:
            pick.append(i)
    figs = Path(a.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    fig = figs / a.fig
    bev(fig, [dict(E=E[i], eo=eo[i], path=T["path"][sel[i]], po=T["path_ok"][sel[i]], lab=LB[i], f=F_in[i], B=T["box"][sel[i], 0], bo=T["box_ok"][sel[i], 0],
                   served=T["poses"][sel[i]], fut=fut[i],
                   title=f"{sc[i][7:15]} {bk('D')[i]} {bk('cell')[i]} k {kk[i]} {X['phase'][i]} ({X['dent'][i]:+.0f} m to entry), push {X['push'][i]}\n"
                         f"v0 {T['v0'][sel[i]]:.1f} m/s, arc4 demo {arc_d[i]:.1f} / plan {P.chord(T['poses'][sel[i]][None])[0]:.1f} m") for i in pick])
    G["G_bev"] = dict(fig=str(fig), rows=[str(names[i]) for i in pick])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"pool_gates_{a.name}.json").write_text(json.dumps(G, indent=1, default=float))
    md = coverage_md(G)
    (out / f"coverage_{a.name}.md").write_text(md)
    for f in ("branches.json", "failing.json"):
        (out / f).write_text((W / f).read_text())
    print(md)


def coverage_md(G):
    f = lambda d: ", ".join(f"{k} {v}" for k, v in d.items())  # noqa: E731
    pb = G["per_branch"]
    L = [f"# Pool on1_{G['name']}: gates and coverage (rf_selfpool.py pack)", "", "| gate | value | pass |", "|:--|:--|:--|",
         f"| G-in (replayed inputs = served: PAI_DUMP files / drive.jsonl records) | dumps {G['G_in']['dump_eq']} / {G['G_in']['dump_n']}, records {G['G_in']['rec_eq']} / {G['G_in']['rec_n']} | {G['G_in']['pass_']} |",
         f"| G-route (regenerated first route point vs the logged route0 <= 0.05 m) | {G['G_route']['share_le_005']} of {G['G_route']['rows_with_route0']} rows, max {G['G_route']['max']:.3f} m | {G['G_route']['pass_']} |",
         f"| G-rmem (route memory input from the logged messages vs the served rmem) | max abs diff {G['G_rmem']['max_abs_diff']} over {G['G_rmem']['rows_compared']} rows | {G['G_rmem']['pass_']} |",
         f"| G-road (demo crosses a road-edge segment; <= 1 %) | {G['G_road']['demo_cross_share']:.4f} (served plan {G['G_road']['served_plan_cross_share']:.4f}) | {G['G_road']['pass_']} |",
         f"| G-speed (demo 4 s chord vs the controller trace within 10 %; >= 99 %) | {G['G_speed']['within_10pct']:.4f}, median ratio {G['G_speed']['median_ratio']:.4f} | {G['G_speed']['pass_']} |",
         f"| G-r1 (pool tokens / ego bit-identical to the cache) | front {G['G_r1']['front_equal']} ({G['G_r1']['rows_checked']} rows), ego {G['G_r1']['ego_equal']}, feat f16 max diff {G['G_r1']['feat_f16_max_diff']:.2e} | {G['G_r1']['pass_']} |",
         f"| G-shape (decision 293 on the branch's realized track) | {G['G_shape']['pass_']} / {G['G_shape']['branches']} branches pass; fails: {f(G['G_shape']['fail_parts'])}; rows dropped {G['G_shape']['rows_dropped']} | - |",
         f"| G-crawl ({G['G_crawl']['rule']}) | rows dropped {G['G_crawl']['rows_dropped']} | - |", "",
         f"Rows {G['rows']} from {G['branches_with_rows']} branches over {G['scenes']} scenes (turns). Route memory input: {G['feat_source']}; "
         f"regenerated memory vs served: {G['G_rmem']['regen_vs_served_share_over_002']:.3f} of rows differ by > 0.02 (p99 {G['G_rmem']['regen_vs_served_p99']:.3f}).", "",
         "| coverage | rows |", "|:--|:--|"]
    for k, v in G["coverage"].items():
        if k != "turns_with_rows" and k != "rows_by_branch":
            L.append(f"| {k[8:] if k.startswith('rows_by_') else k} | {f(v)} |")
    L += ["", "| branch | cell | D | src | h | k0 .. k1 | shape kmax / alat / dk / v_entry | G-shape | window | usable | crawl | rows |",
          "|:--|:--|:--|:--|--:|:--|:--|:--|--:|--:|--:|--:|"]
    for k, v in sorted(pb.items()):
        s = v["shape"]
        L.append(f"| {k[7:15]}~{k.split('~')[1]} | {v['cell']} | {v['D']} | {v['src']} | {v['h']} | {v['k0']} .. {v['k1']} | {s['kmax']} / {s['alat']} / {s['dk']} / {s['v_entry']} | "
                 f"{'pass' if v['g_shape'] else 'fail: ' + ','.join(q for q, ok in v['gate_parts'].items() if not ok)} | {v['rows_window']} | {v['rows_usable']} | {v['rows_crawl']} | {v['rows']} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- build (orchestrator)
def cmd_build(a):
    from jevdrive.run import Run
    if a.tag:
        a.name = a.tag[4:] if a.tag.startswith("on1_") else a.tag
    a.read = a.read or [str(Path(r) / "read.json") for r in a.roots]
    if not a.plan:
        c = [Path(r) / "plan.json" for r in a.roots if (Path(r) / "plan.json").exists()]
        assert c, "--plan: no <root>/plan.json (rf_p4.py plan format) under the roots"
        a.plan = str(c[0])
    assert "u" in next(iter(jload(a.plan)["scenes"].values())), f"{a.plan}: not an rf_p4.py plan (no route grid)"
    W = Path(a.work or D / "runs/route_follow" / f"selfpool_{a.name}")
    a.work, a.out = str(W), a.out or str(W / "results")
    tokyo = a.backend == "tokyo"
    a.stage = a.stage or (f"/tmp/selfpool_{a.name}" if tokyo else str(D / "tmp" / f"selfpool_{a.name}"))   # the box's /tmp is the 30 GB system disk
    served = Path(a.served or R)
    env = dict(os.environ, DATA_DIR=str(D), SELF_REPO=str(R), JEV_REPO=str(served))
    py_train = str(D / "envs/op-train/bin/python")

    def sh(cmd, **kw):
        run.info("$ " + (cmd if isinstance(cmd, str) else " ".join(map(str, cmd))))
        subprocess.run(cmd, check=True, env=kw.pop("env", env), shell=isinstance(cmd, str), **kw)
    with Run("route_follow", f"selfpool-{a.name}", config=vars(a)) as run:
        common = ["--work", a.work]
        sh([sys.executable, __file__, "select", *common, "--plan", a.plan, "--shape", a.shape, "--roots", *a.roots, "--read", *a.read]
           + (["--allow-exam"] if a.allow_exam else []))
        Bs = jload(W / "branches.json")["branches"]
        img = sorted({b["image"] for b in Bs if b.get("image")}) or ["jev-alpasim:p2h10-f-s0-6f3d05d1"]
        assert len(img) == 1, f"branches served from several images {img}: build them separately"
        simargs = ["sim", *common, "--stage", a.stage, "--jobs", str(a.jobs), "--nurec", str(D / "datasets/nurec"), "--cold", a.cold]
        if tokyo:
            S = D / "third_party/alpasim"
            base = f"alpasim-base:{a.base_version}"
            mounts = f"-v {D}:{D}:ro -v {D}/datasets/nurec:/mnt/nre-data:ro -v {W}:{W} -v {a.stage}:{a.stage} -v {R}:/self:ro -v {served}:/jev:ro"
            Path(a.stage).mkdir(parents=True, exist_ok=True)
            sh(f"docker run --rm --network none --user 0:0 --entrypoint bash -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e DATA_DIR={D} -e SELF_REPO=/self -e JEV_REPO=/jev "
               f"-e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 -e PYTHONPATH={S}/src/utils:/jev {mounts} -w /repo {base} "
               f"-c 'umask 0000; exec /repo/.venv/bin/python /self/experiments/route_follow/scripts/rf_selfpool.py {' '.join(simargs)}'")
            sh(["bash", str(R / "experiments/onpol_pai/scripts/onp_env.sh"), "--drv", "scripts/onp_rows.py", "cold", "--pass", PASS, "--jobs", str(a.jobs)],
               env=dict(env, ONP_DEV=str(R / "experiments/onpol_pai"), JEV=str(served), ONP_OUT=str(W / "rows"), ONP_STAGE=a.stage, IMG=img[0]))
        else:
            S = D / "third_party/alpasim"
            sh([str(S / ".venv/bin/python"), __file__, *simargs], env=dict(env, PYTHONPATH=f"{S}/src/utils:{served}"))
            sh([py_train, str(R / "experiments/onpol_pai/scripts/onp_rows.py"), "cold", "--pass", PASS, "--jobs", str(a.jobs), "--out", str(W / "rows"), "--stage", a.stage],
               env=dict(env, PYTHONPATH=f"{served}/experiments/alpasim/lib:{served}"))
        sh([py_train, str(R / "experiments/onpol_pai/scripts/onp_cache.py"), "build", "--pass", PASS, "--rows", str(W / "rows" / PASS), "--out", str(W / "cache")],
           env=dict(env, **({"CUDA_VISIBLE_DEVICES": a.card} if a.card else {})), cwd=str(R))
        sh([py_train, __file__, "pack", *common, "--name", a.name, "--out", a.out, "--feat", a.feat, "--pool-root", a.pool_root, "--feat-root", a.feat_root,
            "--fig-dir", a.fig_dir, "--fig", a.fig or f"self_bev_{a.name}.png", "--pad", str(a.pad)])
        G = jload(Path(a.out) / f"pool_gates_{a.name}.json")
        run.summary.update(rows=G["rows"], scenes=G["scenes"], **{g: G[g]["pass_"] for g in ("G_in", "G_route", "G_rmem", "G_road", "G_speed", "G_r1")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cpu = max(1, min(16, (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4) - 4))
    for c in ("select", "sim", "pack", "build"):
        p = sub.add_parser(c)
        p.add_argument("--work", default="", help="work dir (branches.json, rows/, route/, cache/)")
    for c in ("select", "build"):
        p = sub.choices[c]
        p.add_argument("--roots", nargs="+", required=True, help="run roots of the branch search (searched for run dirs by rollout id)")
        p.add_argument("--read", nargs="*", default=[], help="read.json files (default <root>/read.json)")
        p.add_argument("--plan", default="", help="rf_p4.py plan.json (default <root>/plan.json)")
        p.add_argument("--shape", default=str(SHAPE))
        p.add_argument("--allow-exam", action="store_true", help="dev only: hard_exam / control scenes allowed (the pool is marked)")
    for c in ("pack", "build"):
        p = sub.choices[c]
        p.add_argument("--name", default="SELF"), p.add_argument("--out", default="", help="gates / coverage / branches copies")
        p.add_argument("--feat", choices=("logged", "regen"), default="logged", help="route memory input: the logged messages (as served) or the regenerated route")
        p.add_argument("--pool-root", default=str(CR)), p.add_argument("--feat-root", default=str(FEAT))
        p.add_argument("--fig-dir", default=str(R / "experiments/route_follow/figs")), p.add_argument("--fig", default="")
        p.add_argument("--pad", type=float, default=PAD, help="window end = turn exit + pad m (phase `exit`)")
    for c in ("sim", "build"):
        p = sub.choices[c]
        p.add_argument("--stage", default="", help="staging dir for the uncompressed row files (host /tmp)")
        p.add_argument("--jobs", type=int, default=cpu), p.add_argument("--cold", default="backwarp", help="PAI_COLD the rollouts were served with")
    sub.choices["sim"].add_argument("--nurec", default=str(D / "datasets/nurec"))
    p = sub.choices["build"]
    p.add_argument("--tag", default="", help="pool name with or without on1_ (on1_SELF)")
    p.add_argument("--backend", choices=("tokyo", "box"), default="tokyo" if Path("/data/runs/route_night").is_dir() else "box")
    p.add_argument("--served", default="", help="code snapshot the rollouts were served from (pai_driver for the replay; default this repo)")
    p.add_argument("--card", default="", help="CUDA_VISIBLE_DEVICES for the token encoder (Tokyo; the pool sets it on the box)")
    p.add_argument("--base-version", default="0.89.0")
    a = ap.parse_args()
    if a.cmd != "build" and not a.work:
        ap.error("--work is required")
    dict(select=cmd_select, sim=cmd_sim, pack=cmd_pack, build=cmd_build)[a.cmd](a)
