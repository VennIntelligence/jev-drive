#!/usr/bin/env python3
"""HUMAN-LABEL pack builder: select PAI closed-loop clips and write a self-contained pack (per rollout one mp4 + one JSON) for hl_server.py.
Runs on Tokyo inside alpasim-base:0.89.0 (scene loader, polars, ffmpeg; CPU only, run dirs read, never written) via hl_pack.sh.

  hl_pack.py --out <pack dir> --arm rsg=<run root>[,<run root>...] --arm ... --lists <scene list> ... --exclude <scene list> ...

Batch "sudden_turn_v1" (research/turning-school-checks.md section 1, recomputed here from the scene's logged ego track ds.rig.trajectory):
  turn scene  the log's yaw over its first 20.2 s, resampled at 1 m of arc: some 30 m window changes yaw >= 60 deg (signed max = `ang`,
              left +); 60 <= |ang| <= 125 (U-turns above 125 are excluded).
  entry       the first 30 m window reaching 60 deg; entry = the first point in it whose yaw departs 5 deg from the window start
              (arc length from the log start, the log's arrival time); exit = where the change first reaches 90 % of |ang|.
  sudden      decision 0's command (t = 0, still force-GT, so arm-independent) is not the turn's (S instead of L / R), OR entry >= 15 m.
  pool        the --lists scenes minus every --exclude scene (the exam never shows).
Per selected scene every --arm that has a rollout of it is packed; the first such arm (in --arm order) is the scene's primary.

Rule verdicts per rollout (no map read):
  completed   rf_pai281's proxy: score > 0 and the rollout's own yaw (controller, until the first zero event) turned >= 60 deg the turn's way.
  cls         ego against the human log path in stations [entry - 10, exit + 10] (tracked projection, left +): short (never reached
              entry - 5 m before the zero / end), straight (heading reached < 25 % of the turn there), early-inside (inside offset > 1 m and
              larger than the outside one), late-wide (outside offset > 1 m), on-path (completed, both offsets <= 1 m), else partial.
Time base: t = 0 at decision 0 (`now` of k 0); video frame i shows sim time t_first + i / fps (t_first = -0.1 s, checked per rollout from
the controller CSV's last row and the frame count).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

WIN, TURN_DEG, UTURN_DEG, DEPART_DEG, MIN_ENTRY = 30, 60.0, 125.0, 5.0, 15.0
LOG_WIN_S = 20.2
ZERO = ("collision_front", "collision_lateral", "offroad", "left_corridor_laterally")
CMD = {0: "L", 1: "S", 2: "R"}
_L = {}


def rd(f):
    return [x.split()[0] for x in open(f) if x.strip() and not x.startswith("#")]


def r2(a, n=2):
    return np.round(np.asarray(a, float), n).tolist()


def clean(o):
    """Non-finite floats -> None (route0 is NaN once the route runs out), recursively."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


# ---------------------------------------------------------------- scenes (log track, turn rule)
def loader(cfg):
    if cfg not in _L:
        import logging
        from alpasim_runtime.config import UserSimulatorConfig
        from alpasim_runtime.scene_loader import build_scene_loader
        from omegaconf import OmegaConf
        logging.disable(logging.WARNING)
        _L[cfg] = build_scene_loader(OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(cfg)))
    return _L[cfg]


def turn_of(s, y):
    """1 m arc grid s, yaw deg y -> (signed max change in any WIN m window, entry arc, exit arc) (tsc / ana2 rule)."""
    n = len(y)
    best = 0.0
    for i in range(n):
        seg = y[i:i + WIN + 1] - y[i]
        j = int(np.argmax(np.abs(seg)))
        if abs(seg[j]) > abs(best):
            best = float(seg[j])
    ent = ext = None
    if abs(best) >= TURN_DEG:
        for i0 in range(n):
            seg = y[i0:i0 + WIN + 1] - y[i0]
            if np.abs(seg).max() >= TURN_DEG:
                ent = i0 + int(np.argmax(np.abs(seg) > DEPART_DEG))
                far = y[i0:i0 + 2 * WIN + 1] - y[i0]
                ext = i0 + int(np.argmax(np.abs(far) >= 0.9 * abs(best)))
                break
    return best, (None if ent is None else float(s[ent])), (None if ext is None else float(s[ext]))


def scene_job(job):
    scene, cfg = job
    tr = loader(cfg).get_data_source(scene).rig.trajectory
    T, P, Y = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    v = np.hypot(*np.gradient(P, axis=0).T) / np.maximum(np.gradient(T) * 1e-6, 1e-3)
    m = T <= T[0] + LOG_WIN_S * 1e6
    sg = np.arange(0, d[m][-1], 1.0)
    ang, ent, ext = turn_of(sg, np.degrees(np.interp(sg, d, Y))) if len(sg) > 2 else (0.0, None, None)
    o = dict(scene=scene, T=T, P=P, Y=Y, v=v, d=d, ang=ang, ent_s=ent, ext_s=ext, v17=float(np.interp(T[0] + 1.7e6, T, v)))
    if ent is not None:
        o["ent_T"], o["ext_T"] = float(np.interp(ent, d, T)), float(np.interp(ext, d, T))
        o["ent_xy"] = [float(np.interp(ent, d, P[:, i])) for i in (0, 1)]
        o["ext_xy"] = [float(np.interp(ext, d, P[:, i])) for i in (0, 1)]
    return o


# ---------------------------------------------------------------- geometry
def to_local(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, float).reshape(-1, 2)
    return np.c_[a[0] + c * X[:, 0] - s * X[:, 1], a[1] + s * X[:, 0] + c * X[:, 1]]


def project(R, S, p, lo=-np.inf, hi=np.inf):
    """Point p -> (station, signed lateral, left +) on polyline R (stations S) within stations [lo, hi]."""
    a, b = R[:-1], R[1:]
    dd = b - a
    L2 = np.maximum((dd ** 2).sum(1), 1e-12)
    u = np.clip(((p - a) * dd).sum(1) / L2, 0, 1)
    s = S[:-1] + u * np.sqrt(L2)
    dist = np.hypot(*(p - (a + u[:, None] * dd)).T)
    ok = (s >= lo) & (s <= hi)
    if not ok.any():
        ok[:] = True
    i = int(np.flatnonzero(ok)[np.argmin(dist[ok])])
    cr = dd[i, 0] * (p[1] - a[i, 1]) - dd[i, 1] * (p[0] - a[i, 0])
    return float(s[i]), float(np.sign(cr) * dist[i])


def track(R, S, P):
    st, la, hint = np.empty(len(P)), np.empty(len(P)), None
    for i, p in enumerate(P):
        st[i], la[i] = project(R, S, p, *((-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)))
        hint = st[i]
    return st, la


def merge_route(batches):
    """Route message batches (scene frame, oldest first) -> merged polyline and its length after each batch (rmem.merge's rule)."""
    M, n = None, []
    for B in batches:
        B = B[np.isfinite(B).all(1)]
        if len(B):
            if M is None:
                M = B.copy()
            elif len(M) >= 2:
                a, b = M[-2], M[-1]
                u = (b - a) / max(np.hypot(*(b - a)), 1e-9)
                add = B[((B - b) @ u) > 1.0]
                if len(add):
                    M = np.r_[M, add]
            elif len(B) > len(M):
                M = B.copy()
        n.append(0 if M is None else len(M))
    return M, n


# ---------------------------------------------------------------- rollouts
def ctrl(sim, rid):
    f = Path(sim) / f"controller/alpasim_controller_{rid}.csv"
    if not f.exists():
        return None
    rows = list(csv.reader(open(f)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    if not len(A):
        return None
    c = lambda k: A[:, h[k]]  # noqa: E731
    qx, qy, qz, qw = c("qx"), c("qy"), c("qz"), c("qw")
    yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return dict(t=c("timestamp_us"), x=c("x"), y=c("y"), yaw=yaw, vx=c("vx"), wz=c("wz"))


def events(sim, scene, rid):
    """(first zero-event time us, its flag, first rear-contact time us) from the evaluator's series (rf_pai_follow.events)."""
    import polars as pl
    f = glob.glob(f"{sim}/rollouts/{scene}/{rid}/metrics.parquet")
    if not f:
        return None, None, None
    m = pl.read_parquet(f[0]).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    first = {}
    for (n,), x in m.group_by("name"):
        if n in ZERO + ("collision_rear",):
            t = np.array([np.nan if q is None else q for q in x["timestamps_us"].to_list()], float)
            v = np.array([np.nan if q is None else q for q in x["values"].to_list()], float)
            hit = t[(np.nan_to_num(v) > 0) & np.isfinite(t)]
            if len(hit):
                first[n] = float(hit.min())
    z = {k: v for k, v in first.items() if k in ZERO}
    k = min(z, key=z.get) if z else None
    return (z[k] if k else None), k, first.get("collision_rear")


def kind_of(r):
    m = r.get("metrics") or {}
    if r.get("score") is None:
        return "failed"
    if r["score"]:
        return ""
    return "collision" if m.get("collision_at_fault") else "offroad" if m.get("offroad") else "corridor" if m.get("left_corridor_laterally") else "other"


def probe_video(mp4):
    o = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets,r_frame_rate,width,height", "-of", "json", mp4], capture_output=True, text=True, check=True)
    s = json.loads(o.stdout)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return int(s["nb_read_packets"]), float(a) / float(b), int(s["width"]), int(s["height"])


def camera_crop(mp4, w, h, at=5.0):
    """Bounding box (x, y, w, h) of the camera image under AlpaSim's BEV / metrics panels: non-white pixels below mid-height."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", mp4, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    im = np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(int)
    dark = (im.min(2) < 200)
    y0 = h // 2
    on = np.r_[0, (dark[y0:].mean(1) > 0.5).astype(int), 0]
    a, b = np.flatnonzero(np.diff(on) == 1), np.flatnonzero(np.diff(on) == -1)
    i = int(np.argmax(b - a))                                      # the longest run of dark rows (panel borders are single lines)
    r0, r1 = y0 + a[i], y0 + b[i]
    cols = np.flatnonzero(dark[r0:r1].mean(0) > 0.5)
    return [int(cols[0]), int(r0), int(cols[-1] - cols[0] + 1), int(r1 - r0)]


def roll_job(job):
    """One chunk's rollouts of selected scenes -> clip JSON + remuxed mp4 in the pack; returns manifest rows / errors."""
    run, items, out, scenes = job
    sessions = {it["rid"] for it in items}
    D, R = {}, {}
    for line in open(Path(run) / "driver/drive.jsonl"):
        i = line.find('"session": "')
        if i < 0 or line[i + 12:i + 48] not in sessions:
            continue
        r = json.loads(line)
        if r["kind"] == "drive":
            D.setdefault(r["session"], {})[r["k"]] = r
        elif r["kind"] == "route":
            R.setdefault(r["session"], []).append(r)
    rows, errs = [], []
    for it in items:
        try:
            rows.append(one_roll(it, D.get(it["rid"], {}), R.get(it["rid"], []), out, scenes[it["scene"]]))
        except Exception as e:  # noqa: BLE001
            errs.append((it["scene"], it["arm"], repr(e)[:300]))
    return rows, errs


def one_roll(it, D, Rm, out, sc):
    sim, scene, rid, arm = it["sim"], it["scene"], it["rid"], it["arm"]
    if not D:
        raise RuntimeError("no drive records")
    dec = [D[k] for k in sorted(D)]
    t0 = float(dec[0]["now"])
    rel = lambda us: None if us is None else round((float(us) - t0) / 1e6, 3)  # noqa: E731
    C = ctrl(sim, rid)
    if C is None:
        raise RuntimeError("no controller csv")
    tz, zflag, trear = events(sim, scene, rid)
    kind = kind_of(it["row"])
    tz_end = tz if (tz is not None and kind) else None            # a flag that did not zero the score ends nothing
    # ego trajectory (decision-0 anchor + controller rows), scene frame
    a0 = dec[0]["anchor"]
    v0 = 10.0 * float(np.hypot(dec[0]["ego"][4], dec[0]["ego"][5]))
    et = np.r_[t0, C["t"]]
    ex, ey = np.r_[a0[0], C["x"]], np.r_[a0[1], C["y"]]
    eyaw = np.unwrap(np.r_[a0[2], C["yaw"]])
    evx, ewz = np.r_[v0, C["vx"]], np.r_[0.0, C["wz"]]
    # human log path, its stations, the ego tracked on it
    LP, Ld = sc["P"], sc["d"]
    st, lat = track(LP, Ld, np.c_[ex, ey])
    turn = dict(dir="L" if sc["ang"] > 0 else "R", ang=round(sc["ang"], 1), ent_s=sc["ent_s"], ext_s=sc["ext_s"], ent_xy=r2(sc["ent_xy"]),
                ext_xy=r2(sc["ext_xy"]), t_log_ent=rel(sc["ent_T"]), t_log_ext=rel(sc["ext_T"]))
    sg = 1.0 if sc["ang"] > 0 else -1.0
    d_ent = sc["ent_s"] - st
    hit = np.flatnonzero(st >= sc["ent_s"])
    turn["t_ego_ent"] = rel(et[hit[0]]) if len(hit) else None
    pre = et <= (tz_end if tz_end is not None else np.inf)
    # verdicts
    yaw_net = float(np.degrees(eyaw[pre][-1] - eyaw[0])) if pre.any() else 0.0
    completed = bool(kind == "" and sg * yaw_net >= TURN_DEG)
    win = pre & (st >= sc["ent_s"] - 10) & (st <= sc["ext_s"] + 10)
    if not (pre & (st >= sc["ent_s"] - 5)).any():
        cls = "short"
        mi = mo = frac = None
    else:
        y_ref = float(np.interp(sc["ent_s"] - 10, Ld, sc["Y"]))
        last = int(np.flatnonzero(pre)[-1])
        frac = float(sg * (eyaw[last] - y_ref) / np.radians(abs(sc["ang"])))
        mi = float((sg * lat[win]).max()) if win.any() else 0.0
        mo = float((-sg * lat[win]).max()) if win.any() else 0.0
        cls = ("straight" if frac < 0.25 else "early-inside" if mi > 1.0 and mi >= mo else "late-wide" if mo > 1.0
               else "on-path" if completed else "partial")
    # decisions: served plan, command, speed, route known so far
    msgs = sorted(Rm, key=lambda r: r["ts"])
    batches = [to_local(r["pose"], np.array(r["wp"], float)) for r in msgs]
    M, nk = merge_route(batches)
    mts = [r["ts"] for r in msgs]
    decs = []
    for r in dec:
        a = r["anchor"]
        pl_ = np.array(r["poses"], float)
        xy = to_local(a, pl_[:, :2])
        j = int(np.searchsorted(mts, r["now"], side="right")) - 1
        decs.append(dict(k=r["k"], t=rel(r["now"]), cmd=CMD.get(r["cmd"], "?"), v=round(10.0 * float(np.hypot(r["ego"][4], r["ego"][5])), 2),
                         plan=r2(np.c_[xy, pl_[:, 2] + a[2]]),
                         route0=r2(to_local(a, r["route0"])[0]) if r.get("route0") and np.isfinite(r["route0"]).all() else None,
                         n_route=nk[j] if j >= 0 else 0))
    flips = [dict(t=decs[i]["t"], k=decs[i]["k"], frm=decs[i - 1]["cmd"], to=decs[i]["cmd"]) for i in range(1, len(decs)) if decs[i]["cmd"] != decs[i - 1]["cmd"]]
    # video
    mp4 = glob.glob(f"{sim}/rollouts/{scene}/{rid}/*.mp4")
    if not mp4:
        raise RuntimeError("no mp4")
    nf, fps, w, h = probe_video(mp4[0])
    t_first = round((C["t"][-1] - (nf - 1) * 1e6 / fps - t0) / 1e6, 3)
    cid = f"{scene[7:15]}-{arm}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4[0], "-c", "copy", "-movflags", "+faststart", f"{out}/video/{cid}.mp4"], check=True)
    # human log (10 Hz) with times relative to decision 0
    T = sc["T"]
    lg = dict(t=r2((T - t0) / 1e6, 2), x=r2(LP[:, 0]), y=r2(LP[:, 1]), yaw=r2(sc["Y"], 4), v=r2(sc["v"], 2), s=r2(Ld, 1))
    o = dict(id=cid, scene=scene, arm=arm, tag=it["tag"], rollout=rid, split=it["split"], primary=it["primary"], batch=it["batch"],
             score=None if it["row"].get("score") is None else round(float(it["row"]["score"]), 4), kind=kind, zflag=zflag,
             t_zero=rel(tz_end), t_rear=rel(trear), cmd0=decs[0]["cmd"], turn=turn,
             verdict=dict(completed=completed, yaw_net=round(yaw_net, 1), cls=cls, frac=None if frac is None else round(frac, 2),
                          max_in=None if mi is None else round(mi, 2), max_out=None if mo is None else round(mo, 2)),
             metrics={k: (None if v is None else round(float(v), 3)) for k, v in (it["row"].get("metrics") or {}).items()
                      if isinstance(v, (int, float)) or v is None},
             video=dict(fps=fps, n=nf, t_first=t_first, w=w, h=h),
             ego=dict(t=r2((et - t0) / 1e6, 2), x=r2(ex), y=r2(ey), yaw=r2(eyaw, 4), vx=r2(evx, 2), wz=r2(ewz, 3), s=r2(st, 1), lat=r2(lat, 2),
                      d_ent=r2(d_ent, 1)),
             log=lg, route=r2(M if M is not None else np.zeros((0, 2)), 1), decisions=decs, flips=flips, handoff=1.7)
    Path(f"{out}/clips/{cid}.json").write_text(json.dumps(clean(o), separators=(",", ":"), allow_nan=False))
    return dict(id=cid, scene=scene, arm=arm, tag=it["tag"], rollout=rid, split=it["split"], primary=it["primary"], score=o["score"], kind=kind,
                t_zero=o["t_zero"], dir=turn["dir"], ang=turn["ang"], ent_s=round(sc["ent_s"], 1), t_log_ent=turn["t_log_ent"], cmd0=o["cmd0"],
                completed=completed, cls=cls, n_frames=nf, fps=fps, t_first=t_first, n_flips=len(flips))


# ---------------------------------------------------------------- main
def arm_rollouts(name, roots):
    """{scene: rollout item} of one arm (several roots allowed; a later valid rollout replaces an earlier one without score or video)."""
    out = {}
    for root in roots.split(","):
        for f in sorted(glob.glob(f"{root}/runs/*/sim/aggregate/results-summary.json")):
            sim = f.rsplit("/aggregate/", 1)[0]
            run = sim.rsplit("/sim", 1)[0]
            tag = Path(run).name.rsplit("_c", 1)[0]
            for r in json.load(open(f))["rollouts"]:
                ok = r.get("score") is not None and bool(glob.glob(f"{sim}/rollouts/{r['clipgt_id']}/{r['rollout_id']}/*.mp4"))
                if r["clipgt_id"] not in out or ok:
                    out[r["clipgt_id"]] = dict(arm=name, tag=tag, sim=sim, run=run, rid=r["rollout_id"], row=r, ok=ok, scene=r["clipgt_id"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=ROOT[,ROOT...]; order = primary priority")
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--batch", default="sudden_turn_v1")
    ap.add_argument("--min-entry", type=float, default=MIN_ENTRY)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N selected scenes only (smoke)")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out)
    for d in ("clips", "video"):
        (out / d).mkdir(parents=True, exist_ok=True)
    split = {}
    for f in a.lists:
        for s in rd(f):
            split.setdefault(s, Path(f).stem)
    excl = set().union(*[set(rd(f)) for f in a.exclude]) if a.exclude else set()
    pool = [s for s in split if s not in excl]
    arms = [x.split("=", 1) for x in a.arm]
    A = {n: arm_rollouts(n, r) for n, r in arms}
    cfg = {}
    for n, _ in arms:
        for s, it in A[n].items():
            cfg.setdefault(s, f"{it['sim']}/generated-user-config-0.yaml")
    have = [s for s in pool if s in cfg]
    print(f"pool {len(pool)} scenes ({len(split)} listed, {len(split) - len(pool)} excluded), with a rollout: {len(have)}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        S = {o["scene"]: o for o in ex.map(scene_job, [(s, cfg[s]) for s in have], chunksize=4)}
    turns = [s for s in have if TURN_DEG <= abs(S[s]["ang"]) <= UTURN_DEG and S[s]["ent_s"] is not None]
    uturn = [s for s in have if abs(S[s]["ang"]) > UTURN_DEG]
    # decision-0 command per scene and arm (force-GT: should agree across arms)
    need = {}
    for s in turns:
        for n, _ in arms:
            if s in A[n]:
                need.setdefault(A[n][s]["run"], {})[A[n][s]["rid"]] = (s, n)
    cmd0 = {}
    for run, sess in need.items():
        for line in open(f"{run}/driver/drive.jsonl"):
            if '"k": 0,' not in line or '"kind": "drive"' not in line:
                continue
            r = json.loads(line)
            if r["session"] in sess and r["k"] == 0:
                s, n = sess[r["session"]]
                cmd0.setdefault(s, {})[n] = CMD.get(r["cmd"], "?")
    sel, why = [], {}
    for s in turns:
        d = "L" if S[s]["ang"] > 0 else "R"
        c = next((cmd0.get(s, {}).get(n) for n, _ in arms if n in cmd0.get(s, {})), None)
        unlit, far = c is not None and c != d, S[s]["ent_s"] >= a.min_entry
        if unlit or far:
            sel.append(s)
            why[s] = "+".join(x for x, y in (("unlit", unlit), ("far", far)) if y)
    print(f"turn scenes 60-125 deg: {len(turns)} (U-turns > 125 excluded: {len(uturn)}); selected (cmd0 not lit OR entry >= {a.min_entry:g} m): {len(sel)}",
          flush=True)
    if len(sel) < 15:
        sel = [s for s in turns if S[s]["ent_s"] >= a.min_entry]
        why = {s: "far" for s in sel}
        print(f"fewer than 15: relaxed to entry >= {a.min_entry:g} m alone -> {len(sel)}", flush=True)
    if a.limit:
        sel = sel[: a.limit]
    jobs, items = {}, []
    for s in sel:
        prim = next(n for n, _ in arms if s in A[n])
        for n, _ in arms:
            if s in A[n] and A[n][s]["ok"]:
                it = dict(A[n][s], split=split[s], primary=n == prim, batch=a.batch)
                it.pop("ok")
                jobs.setdefault(it["run"], []).append(it)
                items.append(it)
    sc_light = {s: S[s] for s in sel}
    rows, errs = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for r, e in ex.map(roll_job, [(run, its, str(out), {it["scene"]: sc_light[it["scene"]] for it in its}) for run, its in jobs.items()]):
            rows += r
            errs += e
    rows.sort(key=lambda r: (r["scene"], not r["primary"], [n for n, _ in arms].index(r["arm"])))
    # camera crop from one video (all share AlpaSim's layout)
    crop = camera_crop(str(out / "video" / f"{rows[0]['id']}.mp4"), *[json.load(open(out / "clips" / f"{rows[0]['id']}.json"))["video"][k] for k in ("w", "h")]) if rows else None
    cnt = lambda key, rr: dict(sorted(Counter(key(r) for r in rr).items()))  # noqa: E731
    prim_rows = [r for r in rows if r["primary"]]
    counts = dict(scenes=len(sel), rollouts=len(rows), by_arm=cnt(lambda r: r["arm"], rows), by_split=cnt(lambda r: r["split"], prim_rows),
                  primary_by_dir=cnt(lambda r: r["dir"], prim_rows), by_dir_all=cnt(lambda r: r["dir"], rows),
                  outcome_primary=cnt(lambda r: r["kind"] or "scored", prim_rows), outcome_all=cnt(lambda r: r["kind"] or "scored", rows),
                  completed_primary=cnt(lambda r: r["completed"], prim_rows), cls_primary=cnt(lambda r: r["cls"], prim_rows),
                  cls_all=cnt(lambda r: r["cls"], rows), why=cnt(lambda s: why[s], sel), cmd0_primary=cnt(lambda r: r["cmd0"], prim_rows),
                  cmd0_disagree=[s for s in sel if len(set(cmd0.get(s, {}).values())) > 1], errors=errs)
    man = dict(batch=a.batch, created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
               rule=(f"turn scene: logged ego yaw changes >= {TURN_DEG:g} deg within {WIN} m of arc (first {LOG_WIN_S} s of the log), "
                     f"|angle| <= {UTURN_DEG:g}; selected if decision 0's command is not the turn's OR the turn entry is >= {a.min_entry:g} m "
                     f"from the start; pool = {' + '.join(Path(f).stem for f in a.lists)} minus {' + '.join(Path(f).stem for f in a.exclude) or 'nothing'}"),
               arms={n: r for n, r in arms}, video_crop=crop, counts=counts, clips=rows,
               scenes={s: dict(dir="L" if S[s]["ang"] > 0 else "R", ang=round(S[s]["ang"], 1), ent_s=round(S[s]["ent_s"], 1), why=why[s],
                               split=split[s], cmd0=cmd0.get(s, {})) for s in sel})
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    (out / "labels.jsonl").touch()
    print(json.dumps(counts, indent=1))
    print(f"{len(rows)} clips, {len(errs)} errors -> {out} ({time.time() - t_start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
