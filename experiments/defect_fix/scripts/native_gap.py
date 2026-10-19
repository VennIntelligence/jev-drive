#!/usr/bin/env python3
"""NATIVE-GAP: what separates the native (no serve hold) arena held reads from ~0.93 (results/native_gap.md). No training.

  $AP native_gap.py extract [--workers N]     (box, CPU pool job, AlpaSim venv) every driver decision of the stored native arena held reads
        (RSG-L18-s0 = the arena's stored read, JTN-L18-s0 / s1 and NTV0-L18-s0 @off = jt.sh aread), joined per window with the human's
        logged track and the logged traffic (the window's data source, as arena_resid offroad) and the scorer's per-step metrics
        -> $O/dec.parquet (one row per tag / window / decision), $O/win.parquet (one row per tag / window), $O/hum.pkl (human tracks)
  native_gap.sh cfdump <tag> <shard>           (box, GPU pool job) the arena closed loop of <tag> natively (jt.sh aread's command) with
        SH30_DUMP on the counterfactual windows of one cache shard (lists from native_gap.py cflist) -> $O/dump/<tag>/c<shard>/run
  python3 native_gap.py cflist                  (box) the counterfactual windows: held, non-static strata, human >= 10 m in the window
  $TR native_gap.py cf --tag T                (box, GPU pool job) open-loop counterfactual on the dumped inputs: the same frames, the fed
        ego row's speed history (vx, vy, ax, the three past poses) replaced by the human's speed at the ego's station at each history key
        (positions re-spaced along the ego's own history polyline, vhist.respace's rule) -> $O/cf/<tag>.parquet
  python3 native_gap.py report                (Mac, .venv) the tables of results/native_gap.md from the pulled parquets in tmp/native_gap/

Definitions (all decisions at 0.5 s; k = 0 .. 9; force-GT until 1.017 s, so k = 0, 1 are not executed by the policy):
  v           fed speed |ego[4:6]| x 10 (m/s), the measured body speed at the decision
  vp(t)       the plan's speed at t s: central difference of the plan poses (rear axle), vp(0.5) = |p(1.0) - p(0)| / 1 s
  s_e         the ego anchor's station on the human's logged track (nearest point of the 0.05 s polyline)
  vh          the human's speed when it was at s_e (first passage); vh_now = the human's speed at the decision time
  step k->k+1 (k >= 2, both valid): log speed change d = log(v[k+1] / v[k]) - log(vh[k+1] / vh[k]) (ego against the human at the same
              station) = plan term log(vp(0.5) / v[k]) - log(vh[k+1] / vh[k]) + execution term log(v[k+1] / vp(0.5))
  lead        nearest logged object within 1.8 m + W / 2 of the human track ahead of the ego (0 .. 40 m along it), heading within 60 deg of
              the track or slower than 0.5 m/s; gap = along-track distance from the ego's front bumper to its rear
  valid       decision before the first kept-step event (any collision, offroad, d >= 4 m from the human track), as the scorer truncates
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
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/alpasim_proxy/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
AR = DATA / "runs/alpasim_proxy/arena"
J = DATA / "runs/defect_fix/jt/arena"
O = DATA / "runs/defect_fix/native_gap"
READS = {"RSG": AR / "reads/RSG-L18-s0", "JTN1": J / "JTN-L18-s1@off", "JTN0": J / "JTN-L18-s0@off", "NTV0": J / "NTV0-L18-s0@off"}
CKPT = {"RSG": "RSG-L18-s0", "JTN1": "JTN-L18-s1", "JTN0": "JTN-L18-s0", "NTV0": "NTV0-L18-s0"}
FRONT, LEAD_LAT, LEAD_MAX = 3.9, 1.8, 40.0        # rear axle -> front bumper (Pacifica, m); lead corridor half width; look-ahead (m)
T_H = np.array([1.5, 1.0, 0.5, 0.0])               # history keys, s before t0, oldest first


# ---------------------------------------------------------------- extract
def drives(run):
    rows = []
    f = Path(run) / "driver-logs/drive.jsonl"
    for line in open(f):
        if '"kind": "drive"' not in line:
            continue
        r = json.loads(line)
        e = r["ego"]
        rows.append(dict(id=r["scene"], k=int(r["k"]), t0=int(r["t0"]), v=float(np.hypot(e[4], e[5]) * 10), axf=float(e[6] * 3), cmd=int(r.get("cmd") or 0),
                         x=r["anchor"][0], y=r["anchor"][1], yaw=r["anchor"][2], poses=np.asarray(r["poses"], np.float32)[:, :3].ravel()))
    return rows


def results(run):
    import arena as A
    out = {}
    f = Path(run) / "aggregate/results-summary.json"
    for r in json.load(open(f))["rollouts"]:
        m = r.get("metrics") or {}
        out[r["clipgt_id"]] = dict(score=float(r.get("score") or 0.0), type=A.failure_type(r), dist=float(m.get("dist_traveled_m") or 0.0),
                                   gt=float(m.get("gt_dist_traveled_m") or 0.0))
    return out


def _read(job):
    import arena_resid as RZ
    tag, run = job
    S = RZ._steps(run)
    return tag, run, drives(run), results(run), S


class Track:
    """The human's logged rear-axle track on a fine grid: station, time, speed, heading."""

    def __init__(self, T, P, Y):
        self.T, self.P, self.Y = T, P, np.unwrap(Y)
        seg = np.hypot(*np.diff(P, axis=0).T)
        self.S = np.r_[0.0, np.cumsum(seg)]
        dt = np.diff(T) / 1e6
        v = seg / np.maximum(dt, 1e-3)
        self.V = np.r_[v[0], 0.5 * (v[1:] + v[:-1]), v[-1]] if len(v) > 1 else np.zeros(len(T))

    def station(self, xy):
        d = np.hypot(*(self.P[None] - np.asarray(xy, float)[:, None]).transpose(2, 0, 1))
        i = d.argmin(1)
        return self.S[i], d[np.arange(len(i)), i], i

    def v_at_station(self, s):
        """speed and time at the first passage of station s (still parts: the first index reaching s)."""
        i = np.clip(np.searchsorted(self.S, s, side="left"), 0, len(self.S) - 1)
        return self.V[i], self.T[i]

    def v_at_time(self, t):
        return np.interp(t, self.T, self.V)

    def heading_change(self, s, ahead):
        i0 = np.clip(np.searchsorted(self.S, s), 0, len(self.S) - 1)
        i1 = np.clip(np.searchsorted(self.S, s + ahead), 0, len(self.S) - 1)
        return np.degrees(np.abs(self.Y[i1] - self.Y[i0]))


def lead_at(tr, objs, t, s_ego):
    """the nearest logged object in the human-track corridor ahead of the ego at time t -> (gap m, speed m/s) or (nan, nan)."""
    best = (np.nan, np.nan)
    hd = np.interp(np.clip(s_ego, tr.S[0], tr.S[-1]), tr.S, tr.Y)
    for o in objs.values():
        T = o["T"]
        if t < T[0] - 1e5 or t > T[-1] + 1e5:
            continue
        p = np.array([np.interp(t, T, o["P"][:, 0]), np.interp(t, T, o["P"][:, 1])])
        s, lat, i = tr.station(p[None])
        s, lat = float(s[0]), float(lat[0])
        ahead = s - s_ego
        if not (0 < ahead < LEAD_MAX + FRONT) or lat > LEAD_LAT + o["W"] / 2:
            continue
        ta, tb = max(t - 5e5, T[0]), t if t - 5e5 >= T[0] else min(t + 5e5, T[-1])
        v = np.hypot(np.interp(tb, T, o["P"][:, 0]) - np.interp(ta, T, o["P"][:, 0]), np.interp(tb, T, o["P"][:, 1]) - np.interp(ta, T, o["P"][:, 1])) / max((tb - ta) * 1e-6, 1e-3)
        dh = np.degrees(abs(np.angle(np.exp(1j * (np.interp(t, T, o["Y"]) - tr.Y[i[0]])))))
        if dh > 60 and v > 0.5:
            continue
        gap = ahead - FRONT - o["L"] / 2
        if np.isnan(best[0]) or gap < best[0]:
            best = (gap, float(v))
    return best


def _window(job):
    sys.path.insert(0, str(REPO / "experiments/route_follow/scripts"))
    from ao_e1 import loader
    from rf_tax import logged_objects
    wid, cfg, D = job
    try:
        ds = loader(cfg).get_data_source(wid)
        t = ds.rig.trajectory
        T, P, Y = np.asarray(t.timestamps_us, float), np.asarray(t.positions, float)[:, :2], np.asarray(t.yaws, float)
        tr = Track(T, P, Y)
        objs = logged_objects(ds)
        out = []
        for tag, rows in D.items():
            rows = sorted(rows, key=lambda r: r["k"])
            xy = np.array([[r["x"], r["y"]] for r in rows])
            s, lat, _ = tr.station(xy)
            vh, th = tr.v_at_station(s)
            for r, si, li, vi, ti in zip(rows, s, lat, vh, th):
                g, vl = lead_at(tr, objs, r["t0"], si)
                out.append(dict(tag=tag, id=wid, k=r["k"], t0=r["t0"], v=r["v"], axf=r["axf"], cmd=r["cmd"], x=r["x"], y=r["y"], yaw=r["yaw"],
                                poses=r["poses"], s_e=float(si), lat_e=float(li), vh=float(vi), t_h=float(ti), vh_now=float(tr.v_at_time(r["t0"])),
                                s_h_now=float(np.interp(r["t0"], T, tr.S)), turn20=float(tr.heading_change(si, 20.0)), lead_gap=g, lead_v=vl))
        hum = dict(T=T, P=P, Y=Y)
        return out, (wid, hum), None
    except Exception as e:  # noqa: BLE001
        return [], (wid, None), f"{wid}: {e!r}"


def cmd_extract(a):
    import pandas as pd
    from multiprocessing import Pool
    global O
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(AR / "set/set.csv.gz").set_index("id")
    held = set(S.index[S.split == "held"])
    jobs = [(t, r) for t, d in READS.items() for r in sorted(glob.glob(f"{d}/c*/run"))[:a.limit or None]]
    with Pool(a.workers) as p:
        res = p.map(_read, jobs)
    D, W, ST, cfg = {}, [], [], {}
    for tag, run, dr, rs, st in res:
        for r in dr:
            if r["id"] in held:
                D.setdefault(r["id"], {}).setdefault(tag, []).append(r)
                cfg.setdefault(r["id"], f"{run}/generated-user-config-0.yaml")
        for i, x in rs.items():
            if i in held:
                W.append(dict(tag=tag, id=i, chunk=Path(run).parent.name, **x))
        st = st[st.id.isin(held)].copy()
        st.insert(0, "tag", tag)
        ST.append(st)
    print(f"reads {len(jobs)}, held windows with decisions {len(D)}", flush=True)
    with Pool(a.workers) as p:
        out = p.map(_window, [(i, cfg[i], D[i]) for i in sorted(D)[:a.limit_win or None]], chunksize=4)
    rows = [r for o, _, _ in out for r in o]
    errs = [e for _, _, e in out if e]
    dec = pd.DataFrame(rows)
    dec["poses"] = dec.poses.map(lambda x: np.asarray(x, np.float32).tolist())
    dec.to_parquet(O / "dec.parquet")
    pd.DataFrame(W).to_parquet(O / "win.parquet")
    pd.concat(ST, ignore_index=True).to_parquet(O / "steps.parquet")
    pickle.dump({i: h for _, (i, h), _ in out if h is not None}, open(O / "hum.pkl", "wb"), protocol=4)
    # census labels (red_light_readers.arena's join) and the set's features
    Wd = pd.read_parquet(DATA / "runs/wait_cap/census/windows/20261017-142817/windows.parquet")
    Wd = Wd[Wd.kind == "uniform"].copy()
    Wd["kq"] = (Wd.t0 / 0.5).round().astype(int)
    Sx = S.reset_index()
    Sx["kq"] = (Sx.t0 / 0.5).round().astype(int)
    M = Sx.merge(Wd, on=["log", "kq"], how="left", suffixes=("", "_c"))
    M.to_parquet(O / "set_census.parquet")
    print(json.dumps(dict(decisions=len(dec), windows=int(dec.id.nunique()), errors=len(errs), first_errors=errs[:3])), flush=True)


# ---------------------------------------------------------------- GPU: dump + counterfactual
def cmd_cf(a):
    """open-loop counterfactual on the dumped inputs of <tag>'s own closed loop (cfdump): F = as served (gate: = the dumped plan), H = the
    speed-history entries rewritten to the human's speed at the ego's station at each history key."""
    import pandas as pd
    import torch
    sys.path[:0] = [str(REPO / "experiments/alpasim/lib"), str(REPO / "lib")]
    import sh30_core as SC
    import vhist as VH
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    hum = pickle.load(open(O / "hum.pkl", "rb"))
    src = O / "dump" / a.tag
    recs = {}
    for f in glob.glob(str(src / "c*/run/driver-logs/drive.jsonl")):
        sess = {}
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "start":
                sess[r["session"]] = (r["n"], Path(f).parent / "dump")
            elif r["kind"] == "drive":
                n, d = sess[r["session"]]
                recs[(r["scene"], int(r["k"]))] = (d / f"s{n:02d}_k{r['k']}.npz", r["anchor"], int(r["t0"]))
    core = SC.Core(CKPT[a.tag], "cuda", "backwarp")
    tracks = {i: Track(h["T"], h["P"], h["Y"]) for i, h in hum.items()}
    out = []
    for (sc, k), (f, anc, t0) in sorted(recs.items()):
        if not f.exists() or sc not in tracks:
            continue
        d = np.load(f)
        tr = tracks[sc]
        ego = d["ego"].astype(np.float32)
        # the history keys' positions in the world frame: rear-axle poses of the fed row (t0 frame) -> world via the anchor
        Ph = ego[8:20].reshape(4, 3).astype(np.float64) * [10.0, 10.0, 1.0]
        c, s_ = np.cos(anc[2]), np.sin(anc[2])
        xy = np.c_[anc[0] + c * Ph[:, 0] - s_ * Ph[:, 1], anc[1] + s_ * Ph[:, 0] + c * Ph[:, 1]]
        st, _, _ = tr.station(xy)
        vh, _ = tr.v_at_station(st)                                # human speed at the station of each history key (oldest first)
        v = float(np.hypot(ego[4], ego[5]) * 10)
        H = ego.copy()
        H[4:6] = ego[4:6] * (vh[-1] / v) if v >= 0.5 else (vh[-1] / 10.0, 0.0)
        H[6] = np.clip((vh[-1] - vh[-2]) / 0.5, -9, 9) / 3.0
        arcs = np.cumsum(0.25 * (vh[1:][::-1] + vh[:-1][::-1]))   # arc from t0 back to each past key (newest first), trapezoid
        Q = Ph[::-1]
        dd = Q[1:, :2] - Q[:-1, :2]
        L = np.linalg.norm(dd, axis=1)
        u = dd[-1] / L[-1] if L[-1] > 1e-4 else np.array([-np.cos(Q[-1, 2]), -np.sin(Q[-1, 2])])
        Qe = np.r_[Q, [[*(Q[-1, :2] + 100.0 * u), Q[-1, 2]]]]
        cc = np.r_[0.0, np.cumsum(L), L.sum() + 100.0]
        newp = np.stack([np.interp(arcs, cc, Qe[:, j]) for j in range(3)], 1)[::-1]
        H[8:17] = (newp / [10.0, 10.0, 1.0]).reshape(9)
        cur, valid = d["cur"], d["valid"]
        prev = np.concatenate([np.zeros((1,) + cur.shape[1:], np.uint8), cur[:-1]])
        res = {}
        with torch.no_grad():
            pp, cu = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
            Hf = core.model.net.run_batched(core.A.vision_feeds(pp, cu), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *core.A.H_SHAPE)
            for mode, row in (("F", ego), ("H", H)):
                r_ = np.r_[row, np.zeros(core.rm_dim, np.float32)] if core.rm else row
                o = core.model(Hf, torch.from_numpy(r_[None].astype(np.float32)).to(core.dev), torch.tensor([[1.0, 0.0]], device=core.dev))
                o = (o[0] if isinstance(o, tuple) else o).float()
                mu = o[0, core.pi].reshape(33, 15).cpu().numpy()
                res[mode] = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, d["cam_t"][:2].astype(np.float64), Z.T_OUT, "lever")[:, :3]
        out.append(dict(id=sc, k=k, t0=t0, v=v, vh_keys=vh.tolist(), gate=float(np.abs(res["F"][:, :2] - d["poses"][:, :2]).max()),
                        F=res["F"].ravel().tolist(), H=res["H"].ravel().tolist(), served=np.asarray(d["poses"])[:, :3].ravel().tolist()))
    (O / "cf").mkdir(exist_ok=True)
    pd.DataFrame(out).to_parquet(O / "cf" / f"{a.tag}.parquet")
    g = np.array([x["gate"] for x in out])
    print(json.dumps(dict(tag=a.tag, decisions=len(out), gate_max=float(g.max()) if len(g) else None, gate_p99=float(np.quantile(g, 0.99)) if len(g) else None)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("extract")
    p.add_argument("--workers", type=int, default=32), p.add_argument("--limit", type=int, default=0), p.add_argument("--limit-win", type=int, default=0)
    p.add_argument("--out", default=str(O))
    p = sp.add_parser("cf")
    p.add_argument("--tag", required=True)
    a = ap.parse_args()
    dict(extract=cmd_extract, cf=cmd_cf)[a.cmd](a)
