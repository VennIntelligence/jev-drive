#!/usr/bin/env python3
"""ONPOL-NUPLAN context labeler (CPU, read-only on logs and maps): the LONG-PROBE context (experiments/long_contact/scripts/lc_ctx.py, imported
unedited) for the 1491 public AlpaSim nuPlan scenes, one row per scene `<log>-<token>`, from the NAVSIM test logs.

  $DATA_DIR/envs/navsim2/bin/python experiments/onpol_nuplan/scripts/on_ctx.py [--limit N] [--workers W]
  -> $DATA_DIR/runs/onpol_nuplan/ctx/ctx.npz, gate.json

What differs from the navtrain context (module constants of lc are overridden in this process only, no file is edited):
  origin   the route starts at the logged rear-axle pose BACK = 4 frames (2 s) before the scene token (or at the log's first frame), so a
           rollout that falls behind the log, or a scene that starts before its token, still projects onto the route with a positive arc.
  NR       261 (0 .. 260 m at 1 m) instead of 131: 2 s back + 5.5 s of rollout + a 4 s plan + up to 40 m ahead of the log.
  NK       24 light-state keyframes (origin + 0.5 k, 0 .. 11.5 s) instead of 9, so every dense step of every decision has its keyframe.
           `ln_elig` of a light line therefore asks for a known state somewhere in 11.5 s, not 4 s.
  extras   origin (3,) float64 global x, y, yaw of the route frame; tok_k (frames from the origin to the scene token); kf_ts (NKF,) int64 log
           timestamps (us) of the frames origin + k, kf_pose (NKF, 3) float64 their global rear-axle poses, kf_s (NKF,) their arc along the route,
           kf_v (NKF,) their speed (nan / -1 past the log's end). The packer aligns a pool row to the log through these.
Everything else (route / rhead / rlen / hlen / vlog / tlog, ln_*, tn_*) is lc_ctx's, in the ORIGIN frame (x forward, y left).
Unchanged limits: the hindsight route is the next 30 s of the log + 30 m straight; only the first >= 45 deg turn ahead of the origin is kept.

Gate (self-consistency; lc_ctx's own gate compares with the navtrain cache and does not apply to test logs): every scene labelled; every logged
keyframe on the hindsight path projects onto the route within 0.25 m laterally, and within 0.5 m of its own arc at the 99.9th percentile
(frames, origin and the longer grid; the 1 m grid is a chord resampling of the 2 Hz polyline and its index is used as the arc; a log that
stands still accumulates jitter in its own arc, such scenes are listed in the gate as `kf_arc_outliers`).
Frames are not always 0.5 s apart (12 of 1491 scenes have a 1 s gap inside the window): align by `kf_ts`, never by k x 0.5 s.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
EXP = HERE.parents[2]
sys.path[:0] = [str(HERE.parents[3]), str(EXP / "long_contact/lib"), str(EXP / "long_contact/scripts"), str(EXP / "lowboard_diag/scripts")]
import lc as LC  # noqa: E402

NR, NK, BACK, NKF = 261, 24, 4, 24
LC.NR, LC.NK = NR, NK                            # process-local override, read by lc.project / lc_ctx.label_log at call time
import lc_ctx as CTX  # noqa: E402
import b1 as B  # noqa: E402
import stoplbl_label as SL  # noqa: E402

LOGDIR = B.data_dir() / "datasets/navsim/navsim_logs/test"
LISTS = B.data_dir() / "runs/alpasim/c0b/lists"
LIST_FILES = ("all.txt", "cf1fresh.txt", "ot3new.txt")          # 700 + 391 + 400 = the 1491 public scenes (prereg, Data)


def out_dir() -> Path:
    return B.data_dir() / "runs/onpol_nuplan/ctx"


def scenes() -> list:
    s = [x.strip() for f in LIST_FILES for x in (LISTS / f).read_text().split() if x.strip()]
    assert len(s) == len(set(s)), "duplicate scene ids across the lists"
    return sorted(s)


def work(job):
    """(log, [(row, scene token)]) -> list of (row, dict)."""
    log, want = job
    path = LOGDIR / f"{log}.pkl"
    fr = pickle.load(open(path, "rb"))
    n = len(fr)
    idx = {f["token"]: i for i, f in enumerate(fr)}
    meta, want2 = {}, []
    for gi, tok in want:
        i = idx.get(tok)
        if i is None:
            continue
        i0 = max(0, i - BACK)
        meta[gi] = (i, i0)
        want2.append((gi, fr[i0]["token"]))
    xy = np.array([f["ego2global_translation"][:2] for f in fr], float)
    yaw = np.unwrap(np.array([SL.yaw_of(f["ego2global_rotation"]) for f in fr]))
    sp = np.array([np.hypot(*f["ego_dynamic_state"][:2]) for f in fr], float)
    ts = np.array([f["timestamp"] for f in fr], np.int64)
    out = []
    for gi, r in CTX.label_log((path, want2)):
        i, i0 = meta[gi]
        kk = np.arange(i0, min(i0 + NKF, n))
        m = len(kk)
        kp, ks, kv, kt = np.full((NKF, 3), np.nan), np.full(NKF, np.nan, np.float32), np.full(NKF, np.nan, np.float32), np.full(NKF, -1, np.int64)
        kp[:m] = np.c_[xy[kk], yaw[kk]]
        ks[:m] = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy[kk], axis=0).T))] if m > 1 else 0.0
        kv[:m], kt[:m] = sp[kk], ts[kk]
        r |= dict(origin=np.array([xy[i0, 0], xy[i0, 1], yaw[i0]]), tok_k=np.int8(i - i0), kf_ts=kt, kf_pose=kp, kf_s=ks, kf_v=kv,
                  n_after=np.int32(n - 1 - i0))
        out.append((gi, r))
    return out


def main(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    with Run("onpol_nuplan", "ctx", config=vars(a)) as run:
        sc = scenes()
        if a.limit:
            sc = sc[:: max(1, len(sc) // a.limit)][: a.limit]
        run.info("%d scenes (lists %s; the lane's train / held split is not used here: context labels for every public scene)", len(sc), LIST_FILES)
        by = {}
        for g, s in enumerate(sc):
            log, tok = s.rsplit("-", 1)
            by.setdefault(log, []).append((g, tok))
        jobs = sorted(by.items())
        t0 = time.time()
        res = par.pmap(work, jobs, run=run, workers=min(a.workers or n_cpus(), len(jobs)), desc="logs")
        res.raise_if_failed()
        rows = [x for part in res.values if part for x in part]
        N = len(sc)
        run.info("labelled %d of %d scenes from %d logs in %.0f s", len(rows), N, len(jobs), time.time() - t0)
        proto = rows[0][1]
        Z = {k: np.zeros((N,) + np.shape(v), np.asarray(v).dtype) for k, v in proto.items()}
        for k in Z:
            if Z[k].dtype.kind == "f":
                Z[k][:] = np.nan
            elif k in ("ln_st", "ln_stcross", "kf_ts"):
                Z[k][:] = -1
        ok = np.zeros(N, bool)
        for g, r in rows:
            ok[g] = True
            for k, v in r.items():
                Z[k][g] = v
        Z["ok"], Z["scene"] = ok, np.array(sc)
        # gate: logged keyframes against the route (origin frame)
        kp = Z["kf_pose"]
        o = Z["origin"]
        loc, _ = CTX.local(kp[..., :2], kp[..., 2], o[:, None, 0], o[:, None, 1], o[:, None, 2])
        on = np.isfinite(kp[..., 0]) & (Z["kf_s"] <= Z["hlen"][:, None] + 1e-3) & (Z["kf_s"] <= NR - 2) & ok[:, None]
        s, lat = LC.project(Z["route"].astype(np.float32), np.nan_to_num(loc).astype(np.float32))
        ds, dl = np.abs(s - Z["kf_s"])[on], np.abs(lat)[on]
        k_dec = np.minimum(Z["tok_k"].astype(int) + 11, NKF - 1)                  # last decision of a 5.5 s scene that starts at its token
        s_last = np.take_along_axis(Z["kf_s"], k_dec[:, None], 1)[:, 0]
        v_last = np.take_along_axis(Z["kf_v"], k_dec[:, None], 1)[:, 0]
        need = s_last + 4.0 * np.nan_to_num(v_last) + 40.0 + LC.SW.FRONT
        dt = np.diff(Z["kf_ts"], axis=1)[(Z["kf_ts"][:, 1:] > 0)]
        gate = dict(scenes=int(N), labelled=int(ok.sum()), logs=len(jobs), NR=NR, NK=NK, BACK=BACK, kf_checked=int(on.sum()),
                    kf_arc_max=float(ds.max()), kf_lat_max=float(dl.max()), kf_arc_p999=float(np.quantile(ds, 0.999)),
                    tok_k_counts={int(k): int(v) for k, v in zip(*np.unique(Z["tok_k"][ok], return_counts=True))},
                    kf_dt_ms_min=float(dt.min() / 1e3), kf_dt_ms_max=float(dt.max() / 1e3),
                    frames_after_origin_min=int(Z["n_after"][ok].min()), scenes_log_ends_before_tok_plus_9p5s=int((Z["n_after"][ok] < Z["tok_k"][ok] + 19).sum()),
                    hlen_min=float(np.nanmin(Z["hlen"])), hlen_median=float(np.nanmedian(Z["hlen"])),
                    arc_at_last_decision_max=float(np.nanmax(s_last)), reach_needed_max=float(np.nanmax(need)),
                    scenes_reach_beyond_grid=int((need > NR - 1).sum()),
                    lines=int((Z["ln_type"] > 0).sum()), tl_lines=int((Z["ln_type"] == 2).sum()), tl_elig=int((Z["ln_elig"] & (Z["ln_type"] == 2)).sum()),
                    tl_elig_red_somewhere=int((Z["ln_elig"] & (Z["ln_type"] == 2) & (Z["ln_st"] == 1).any(-1)).sum()),
                    ss_lines=int((Z["ln_type"] == 1).sum()), ss_elig=int((Z["ln_elig"] & (Z["ln_type"] == 1)).sum()),
                    turns=int(Z["tn_ok"].sum()), turns_elig=int((Z["tn_ok"] & (Z["tn_vpk"] >= LC.C_VLOG_MIN)).sum()))
        dsm = np.where(on, np.abs(s - Z["kf_s"]), 0).max(1)
        gate["kf_arc_outliers"] = {str(Z["scene"][i]): float(dsm[i]) for i in np.where(dsm > 0.4)[0]}
        gate["scenes_with_frame_gap"] = int(((np.diff(Z["kf_ts"], axis=1) > 6e5) & (Z["kf_ts"][:, 1:] > 0)).any(1).sum())
        gate["pass"] = bool(gate["labelled"] == N and gate["kf_arc_p999"] < 0.5 and gate["kf_lat_max"] < 0.25)
        out = out_dir() / ("ctx.npz" if not a.limit else f"ctx-first{a.limit}.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **Z)
        tmp.rename(out)
        (out.parent / ("gate.json" if not a.limit else f"gate-first{a.limit}.json")).write_text(json.dumps(gate, indent=1))
        run.summary.update(gate)
        run.info("gate %s", json.dumps(gate))
        assert gate["pass"], "context gate failed"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    main(ap.parse_args())
