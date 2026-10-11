#!/usr/bin/env python3
"""LONG-PROBE context labeler (CPU, read-only): per navtrain token, the hindsight route, its stop lines with light states and the first
>= 45 deg turn, in the token's logged rear-axle frame (lib/lc.py docstring). Map access and the stop-line / connector matching are
experiments/lowboard_diag/scripts/stoplbl_label.py's (decision 238), imported unchanged.

  $DATA_DIR/envs/navsim2/bin/python experiments/long_contact/scripts/lc_ctx.py run [--limit-logs N] [--workers W]
  -> $DATA_DIR/runs/long_contact/ctx/ctx.npz (one row per navtrain token, body1 global row order), gate.json

Gate: the 8 logged future poses rebuilt from the log pkl must equal the cache's `fut` (max |diff| < 0.05 m) and the t0 speed its `speed`.
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
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "lowboard_diag/scripts")]
import lc as LC  # noqa: E402
import b1 as B  # noqa: E402
import stoplbl_label as SL  # noqa: E402

LOGDIR = B.data_dir() / "datasets/navsim/navsim_logs/trainval"
FRONT = LC.SW.FRONT


def local(xy, yaw, x0, y0, h0):
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = xy[..., 0] - x0, xy[..., 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], -1), yaw - h0


def first_arrival(s_k, v_k, g):
    """Arc s_k (n,) non-decreasing of the logged frames (0.5 s apart), speeds v_k; grid g -> (time, speed) of the first arrival, nan if never."""
    k = np.searchsorted(s_k, g, "left")
    t, v = np.full(len(g), np.nan), np.full(len(g), np.nan)
    z = k == 0
    t[z], v[z] = 0.0, v_k[0]
    m = (k > 0) & (k < len(s_k))
    km = k[m]
    den = np.maximum(s_k[km] - s_k[km - 1], 1e-9)
    f = np.clip((g[m] - s_k[km - 1]) / den, 0, 1)
    t[m], v[m] = 0.5 * (km - 1 + f), v_k[km - 1] + f * (v_k[km] - v_k[km - 1])
    return t, v


def turn(th, g, s_k, v_k):
    """Heading th (n,) on the 1 m grid g -> the first run of |kappa| > 0.025 / m (merged over gaps <= 5 m) turning >= 45 deg whose peak is
    ahead of the origin (stoplbl_label's rule, edge-padded smoothing) -> dict or None."""
    if len(g) < 12:
        return None
    ths = np.convolve(np.pad(th, 1, mode="edge"), np.ones(3) / 3, "valid")
    k = np.gradient(ths, 1.0)
    k = np.convolve(np.pad(k, 2, mode="edge"), np.ones(5) / 5, "valid")
    on = np.abs(k) > 0.025
    runs, a0 = [], None
    for q, o in enumerate(on):
        if o and a0 is None:
            a0 = q
        if (not o) and a0 is not None:
            runs.append([a0, q])
            a0 = None
    if a0 is not None:
        runs.append([a0, len(on)])
    mg = []
    for ru in runs:
        if mg and ru[0] - mg[-1][1] <= 5:
            mg[-1][1] = ru[1]
        else:
            mg.append(ru)
    for a_, b_ in mg:
        b_ = min(b_, len(th) - 1)
        dth = th[b_] - th[a_]
        pk = a_ + int(np.abs(k[a_:b_ + 1]).argmax())
        if np.degrees(abs(dth)) >= LC.TURN_DEG and g[pk] >= 1.0:
            (tpk, ta), (vpk, va) = first_arrival(s_k, v_k, np.array([g[pk], g[a_]]))
            return dict(sa=g[a_], sb=g[b_], spk=g[pk], kpk=float(k[pk]), dth=float(np.degrees(dth)), vpk=vpk, va=va, tpk=tpk)
    return None


def label_log(job):
    """(log pkl path, [(global row, token)]) -> list of (global row, dict of per-token arrays)."""
    from shapely.geometry import LineString
    path, want = job
    fr = pickle.load(open(path, "rb"))
    mp = SL.get_map(fr[0]["map_location"])
    n = len(fr)
    xy = np.array([f["ego2global_translation"][:2] for f in fr], float)
    yaw = np.unwrap(np.array([SL.yaw_of(f["ego2global_rotation"]) for f in fr]))
    sp = np.array([np.hypot(*f["ego_dynamic_state"][:2]) for f in fr], float)
    tld = [{str(a): bool(b) for a, b in (f.get("traffic_lights") or [])} for f in fr]
    idx = {f["token"]: i for i, f in enumerate(fr)}
    g_full = np.arange(LC.NR, dtype=float)
    out = []
    for gi, tok in want:
        i = idx.get(tok)
        if i is None:
            continue
        j = min(n - 1, i + LC.HOR)
        p = xy[i:j + 1]
        seg = np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)
        s_k = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(s_k[-1])
        keep = np.r_[True, seg > 1e-3]                                         # strictly increasing vertices for the polyline / heading maps
        pv, sv, hv = p[keep], s_k[keep], yaw[i:j + 1][keep]
        h_end = yaw[j]
        pe = np.vstack([pv, pv[-1] + LC.EXT * np.array([np.cos(h_end), np.sin(h_end)])])
        se = np.r_[sv, L + LC.EXT]
        gx = np.minimum(g_full, se[-1])
        rx, ry = np.interp(gx, se, pe[:, 0]), np.interp(gx, se, pe[:, 1])
        beyond = g_full - gx                                                    # straight continuation past the extension
        rx, ry = rx + beyond * np.cos(h_end), ry + beyond * np.sin(h_end)
        rh = np.where(g_full <= L, np.interp(np.minimum(g_full, L), sv, hv) if len(sv) > 1 else hv[0], h_end)
        R, RH = local(np.stack([rx, ry], -1), rh, xy[i, 0], xy[i, 1], yaw[i])
        tl_, vl_ = first_arrival(s_k, sp[i:j + 1], g_full)
        tl_[g_full > L], vl_[g_full > L] = np.nan, np.nan
        r = dict(route=R.astype(np.float32), rhead=RH.astype(np.float32), rlen=np.float32(L + LC.EXT), hlen=np.float32(L),
                 vlog=vl_.astype(np.float16), tlog=tl_.astype(np.float16), v0=np.float32(sp[i]))
        # logged future (gate against the cache's fut)
        fut = np.full((8, 3), np.nan, np.float32)
        kk = np.arange(i + 1, min(i + 9, n))
        if len(kk):
            fl, fy = local(xy[kk], yaw[kk], xy[i, 0], xy[i, 1], yaw[i])
            fut[:len(kk)] = np.c_[fl, fy]
        r["fut"] = fut
        # stop lines of type 1 / 2 crossed by the route (incl. the extension), nearest first
        line = LineString(pe)
        hits = []
        for c in mp["tree"].query(line.buffer(3.0)):
            ty = int(mp["ty"][c])
            if ty not in (1, 2):
                continue
            d = SL.first_cross(line, mp["sp"][c])
            if d is not None and FRONT - 0.5 < d <= LC.NR - 1:
                hits.append((d, ty, c))
        hits.sort()
        ln = dict(ln_type=np.zeros(LC.NL, np.int8), ln_d=np.full(LC.NL, np.nan, np.float32), ln_st=np.full((LC.NL, LC.NK), -1, np.int8),
                  ln_conn=np.zeros(LC.NL, bool), ln_tstop=np.full(LC.NL, np.nan, np.float32), ln_tcross=np.full(LC.NL, np.nan, np.float32),
                  ln_stcross=np.full(LC.NL, -1, np.int8), ln_elig=np.zeros(LC.NL, bool))
        for q, (d, ty, c) in enumerate(hits[:LC.NL]):
            ln["ln_type"][q], ln["ln_d"][q] = ty, d
            kc = int(np.searchsorted(s_k, d - FRONT, "left"))                  # first logged frame whose front bumper is at / past the line
            crossed = kc < len(s_k)
            if crossed:
                ln["ln_tcross"][q] = first_arrival(s_k, sp[i:j + 1], np.array([d - FRONT]))[0][0]
            zone = (s_k >= d - LC.STOP_ZONE) & (s_k <= d - FRONT + 1.5) & (sp[i:j + 1] < LC.V_LOG_STOP)
            zone &= np.arange(len(s_k)) < (kc if crossed else len(s_k))
            if zone.any():
                ln["ln_tstop"][q] = 0.5 * int(np.argmax(zone))
            if ty == 2:
                conns = mp["sp_conn"].get(mp["sp_ids"][c], [])
                best, bd = None, 1e9
                for cid in conns:
                    gg = mp["lcg"][cid]
                    qs = [line.interpolate(min(d + a, line.length)) for a in (4, 8, 12)]
                    dd = float(np.mean([gg.distance(x) for x in qs]))
                    if dd < bd:
                        best, bd = cid, dd
                if best is not None and bd < 3.0:
                    ln["ln_conn"][q] = True
                    key = str(best)
                    st = lambda k: {True: 1, False: 0, None: -1}[tld[i + k].get(key)] if i + k < n else -1  # noqa: E731
                    ln["ln_st"][q] = [st(k) for k in range(LC.NK)]
                    if crossed:
                        ln["ln_stcross"][q] = st(kc)
                    ln["ln_elig"][q] = (ln["ln_st"][q] >= 0).any() and ln["ln_stcross"][q] != 1
            else:                                                              # stop sign: a future stop of the logged driver, not one already made
                ln["ln_elig"][q] = bool(np.isfinite(ln["ln_tstop"][q]) and ln["ln_tstop"][q] > 0.5)
        r |= ln
        gg = g_full[g_full <= L]
        tn = turn(RH[:len(gg)], gg, s_k, sp[i:j + 1]) if L > 12 else None
        r |= dict(tn_ok=tn is not None, **{f"tn_{k}": np.float32(tn[k] if tn else np.nan) for k in ("sa", "sb", "spk", "kpk", "dth", "vpk", "va", "tpk")})
        out.append((gi, r))
    return out


def cmd_run(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("long_contact", "ctx", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        toks = np.load(B.root() / "labels" / "tokens.npy")
        logs = np.load(B.root() / "labels" / "log.npy")
        N = len(toks)
        by = {}
        for g, (t, l) in enumerate(zip(toks.tolist(), logs.tolist())):
            by.setdefault(l, []).append((g, t))
        names = sorted(by)
        if a.limit_logs:
            names = names[:: max(1, len(names) // a.limit_logs)][: a.limit_logs]
        jobs = [(LOGDIR / f"{l}.pkl", by[l]) for l in names]
        t0 = time.time()
        res = par.pmap(label_log, jobs, run=run, workers=min(a.workers or n_cpus(), len(jobs)), desc="logs")
        res.raise_if_failed()
        rows = [x for part in res.values if part for x in part]
        run.info("labelled %d of %d tokens from %d logs in %.0f s", len(rows), N, len(jobs), time.time() - t0)
        proto = rows[0][1]
        Z = {k: np.zeros((N,) + np.shape(v), np.asarray(v).dtype) for k, v in proto.items()}
        for k in Z:
            if Z[k].dtype.kind == "f":
                Z[k][:] = np.nan
            elif k in ("ln_st", "ln_stcross"):
                Z[k][:] = -1
        ok = np.zeros(N, bool)
        for g, r in rows:
            ok[g] = True
            for k, v in r.items():
                Z[k][g] = v
        Z["ok"] = ok
        Z["tokens"] = toks
        # gate: the cache's fut / speed
        off = B.shard_offsets()
        fut = np.concatenate([B.tab("log", k)["fut"] for k in range(B.NSH)])
        spd = np.concatenate([B.tab("log", k)["speed"] for k in range(B.NSH)])
        m = ok & np.isfinite(fut).all((1, 2)) & np.isfinite(Z["fut"]).all((1, 2))
        df = np.abs(Z["fut"][m] - fut[m])
        df[..., 2] = np.abs(np.angle(np.exp(1j * (Z["fut"][m][..., 2] - fut[m][..., 2]))))
        gate = dict(tokens=int(N), labelled=int(ok.sum()), fut_rows=int(m.sum()), fut_max_xy=float(df[..., :2].max()), fut_max_yaw=float(df[..., 2].max()),
                    fut_p999_xy=float(np.quantile(df[..., :2].max((1, 2)), 0.999)), speed_max=float(np.abs(Z["v0"][ok] - spd[ok]).max()),
                    lines=int((Z["ln_type"] > 0).sum()), lines_elig=int(Z["ln_elig"].sum()), tl_elig=int((Z["ln_elig"] & (Z["ln_type"] == 2)).sum()),
                    ss_elig=int((Z["ln_elig"] & (Z["ln_type"] == 1)).sum()), turns=int(Z["tn_ok"].sum()), offsets=off.tolist())
        out = LC.ctx_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **Z)
        tmp.rename(out)
        (out.parent / "gate.json").write_text(json.dumps(gate, indent=1))
        run.summary.update(gate)
        run.info("gate %s", json.dumps(gate))
        assert gate["fut_max_xy"] < 0.05 and gate["speed_max"] < 0.05, "the log pkl does not reproduce the cache's fut / speed"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--limit-logs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    cmd_run(a)
