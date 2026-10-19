#!/usr/bin/env python3
"""WARP-LOOP (plans/2026-10-19-warp-loop-prereg.md, results/warp_loop.md): does the speed ratchet run through the synthesised slots? No
training. Open loop on the stored native dumps: the 6 synthesised policy slots re-warped along alternative ego tracks (keyframes and the
ego row fixed), against the same alternative tracks fed only through the ego row's speed history.

  $TR warp_loop.py cf --tag T --run <dump run dir>   (box, GPU pool job) every dumped decision of that run -> $O/cf/<tag>__<name>.parquet
  .venv/bin/python warp_loop.py report               (box) the tables of results/warp_loop.md -> stdout and $O/report.json

Per decision (all from the dump, nothing re-simulated):
  keys      decisions are 0.5 s apart and the keyframes are the earlier decisions' t0 frames: key j of decision k = slot 7 of decision
            k - 3 + j (k = 0..2: 1..3 keys, the cold start as served)
  history   P (4, 3) = the fed row's poses (ego[8:20], fill_history already applied); V (4, 2) body velocity of each key = ego[4:6] of the
            decision whose t0 it is (back-filled keys: the oldest real key's, as fill_history)
  gate      the native track's lattice against the dumped cur (bytes), and the plan of the dumped frames against the dumped plan
  profiles  positions re-spaced along the ego's own history polyline (newest to oldest, straight beyond the oldest pose; yaw interpolated
            along it; native_gap cf's rule), body velocity direction kept and its size replaced:
            hum   the human's logged speed at the station of each key (arc = trapezoid; = native_gap cf H)
            const the current measured speed at every key
            stand all four poses at t0, speed 0 (each synthesised slot = its nearest keyframe as is)
            s08 / s12  the ego's own arcs and speeds x 0.8 / 1.2
  variants  F (as served); W_p (frames warped along p, row as fed); R_p (dumped frames, the row's speed history = p: vx vy, ax, the 3
            past poses); B_p (both)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
import native_gap as NG  # noqa: E402

DATA = NG.DATA
O = DATA / "runs/defect_fix/warp_loop"
PROF = ("hum", "const", "stand", "s08", "s12")
MINN = 20
VARS = ("F",) + tuple(f"{x}_{p}" for p in PROF for x in "WRB")


def sessions(drive):
    """drive.jsonl -> {session: (n, scene, {k: record})}"""
    S = {}
    for line in open(drive):
        r = json.loads(line)
        if r["kind"] == "start":
            S[r["session"]] = (r["n"], r["scene"], {})
        elif r["kind"] == "drive":
            S[r["session"]][2][int(r["k"])] = r
    return S


def respace(P, arcs):
    """P (4, 3) oldest first; arcs (3,) from t0 back to keys 2, 1, 0 (newest first) -> (4, 3), the t0 pose kept."""
    Q = P[::-1]
    dd = Q[1:, :2] - Q[:-1, :2]
    L = np.linalg.norm(dd, axis=1)
    u = dd[-1] / L[-1] if L[-1] > 1e-4 else np.array([-np.cos(Q[-1, 2]), -np.sin(Q[-1, 2])])
    Qe = np.r_[Q, [[*(Q[-1, :2] + 100.0 * u), Q[-1, 2]]]]
    cc = np.r_[0.0, np.cumsum(L), L.sum() + 100.0]
    out = P.copy()
    out[:3] = np.stack([np.interp(arcs, cc, Qe[:, j]) for j in range(3)], 1)[::-1]
    return out


def profiles(P, V, ax, vh):
    """-> {name: (P', V', ax')} (ax' m/s^2 at t0); vh (4,) the human's speed at each key's station or None."""
    L = np.linalg.norm(np.diff(P[::-1, :2], axis=0), axis=1)
    a = np.cumsum(L)                                            # native arcs to keys 2, 1, 0
    sp = np.linalg.norm(V, axis=1)
    dirs = np.where(sp[:, None] >= 0.5, V / np.maximum(sp, 1e-9)[:, None], [1.0, 0.0])
    out = {}
    for c, n in ((0.8, "s08"), (1.2, "s12")):
        out[n] = (respace(P, c * a), c * V, c * ax)
    v0 = sp[-1]
    out["const"] = (respace(P, v0 * np.array([0.5, 1.0, 1.5])), dirs * v0, 0.0)
    out["stand"] = (np.zeros((4, 3)), np.zeros((4, 2)), 0.0)
    if vh is not None:
        arcs = np.cumsum(0.25 * (vh[1:][::-1] + vh[:-1][::-1]))
        out["hum"] = (respace(P, arcs), dirs * vh[:, None], float(np.clip((vh[-1] - vh[-2]) / 0.5, -9, 9)))
    return out


def row_of(ego, Pp, Vp, axp):
    r = ego.copy()
    r[4:6] = Vp[-1] / 10.0
    r[6] = axp / 3.0
    r[8:17] = (Pp[:3] / [10.0, 10.0, 1.0]).reshape(9)
    return r.astype(np.float32)


def ubar(Pp):
    return float(np.linalg.norm(np.diff(Pp[:, :2], axis=0), axis=1).sum() / 1.5)


def arc4(p):
    p = np.asarray(p)[:, :2]
    return float(np.linalg.norm(np.diff(np.r_[[[0.0, 0.0]], p], axis=0), axis=1).sum())


def cmd_cf(a):
    import pandas as pd
    import torch
    sys.path[:0] = [str(REPO / "experiments/alpasim/lib"), str(REPO / "lib")]
    import sh30_core as SC
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    SC._POOL = ThreadPoolExecutor(a.threads)
    run = Path(a.run)
    name = "_".join(run.parts[-3:-1]) if run.name == "run" else run.name
    hum = pickle.load(open(NG.O / "hum.pkl", "rb"))
    tracks = {}
    core = SC.Core(NG.CKPT[a.tag], "cuda", "backwarp")
    out, nogate = [], 0
    S = sessions(run / "driver-logs/drive.jsonl")
    outer = ThreadPoolExecutor(len(PROF) + 1)

    def enc(cur, valid):
        prev = np.concatenate([np.zeros((1,) + cur.shape[1:], np.uint8), cur[:-1]])
        pp, cu = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
        return core.model.net.run_batched(core.A.vision_feeds(pp, cu), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *core.A.H_SHAPE)

    def pol(H, row, rmem, cam_t):
        r_ = np.r_[row, np.asarray(rmem, np.float32)] if core.rm else row
        o = core.model(H, torch.from_numpy(r_[None].astype(np.float32)).to(core.dev), torch.tensor([[1.0, 0.0]], device=core.dev))
        o = (o[0] if isinstance(o, tuple) else o).float()
        mu = o[0, core.pi].reshape(33, 15).cpu().numpy()
        return I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")[:, :3]

    for sess, (n, sc, recs) in sorted(S.items(), key=lambda x: x[1][0])[:a.limit or None]:
        d = {}
        for k in sorted(recs):
            f = run / "driver-logs/dump" / f"s{n:02d}_k{k}.npz"
            if f.exists():
                z = np.load(f)
                d[k] = {x: z[x] for x in ("cur", "valid", "poses", "ego", "cam_t")}
        if sc in hum and sc not in tracks:
            h = hum[sc]
            tracks[sc] = NG.Track(h["T"], h["P"], h["Y"])
        for k in sorted(d):
            r, z = recs[k], d[k]
            m = min(k + 1, 4)
            if any((k - j) not in d for j in range(m)) or any(recs[k - j]["t0"] != r["t0"] - 500000 * j for j in range(m)) or r["n_keys"] != m:
                nogate += 1
                continue
            e = 4 - m
            K = SC.stack_keys([d[k - m + 1 + j]["cur"][7] for j in range(m)])
            ego = z["ego"].astype(np.float32)
            P = ego[8:20].reshape(4, 3).astype(np.float64) * [10.0, 10.0, 1.0]
            V = np.zeros((4, 2))
            for j in range(e, 4):
                V[j] = d[k - 3 + j]["ego"][4:6].astype(np.float64) * 10.0
            V[:e] = V[e]
            cam_t = z["cam_t"].astype(np.float64)
            vh = None
            if sc in tracks:
                anc = r["anchor"]
                c, s_ = np.cos(anc[2]), np.sin(anc[2])
                xy = np.c_[anc[0] + c * P[:, 0] - s_ * P[:, 1], anc[1] + s_ * P[:, 0] + c * P[:, 1]]
                vh, _ = tracks[sc].v_at_station(tracks[sc].station(xy)[0])
            prof = profiles(P, V, float(ego[6]) * 3.0, vh)
            jobs = {"N": (P, V)} | {p: (q[0], q[1]) for p, q in prof.items()}
            if a.synth == "gpu":                                  # the serve default (sh30_driver SH30_SYNTH=gpu)
                lat = {q: (lambda c_v: (c_v[0].cpu().numpy(), c_v[1]))(SC.lattice_gpu(K, e, I.track_navsim(*pv), cam_t, "backwarp", core.dev))
                       for q, pv in jobs.items()}
            else:
                lat = dict(zip(jobs, outer.map(lambda pv: SC.lattice(K, e, I.track_navsim(*pv), cam_t, "backwarp"), jobs.values())))
            curN, validN = lat["N"]
            byte_diff = int((curN != z["cur"]).sum()) if np.array_equal(validN, z["valid"]) else -1
            res = {}
            with torch.no_grad():
                Hd = enc(z["cur"], z["valid"])
                res["F"] = pol(Hd, ego, r.get("rmem"), cam_t)
                gate_n = 0.0
                if byte_diff:
                    gate_n = float(np.abs(pol(enc(curN, validN), ego, r.get("rmem"), cam_t)[:, :2] - res["F"][:, :2]).max())
                for p, (Pp, Vp, axp) in prof.items():
                    Hp = enc(*lat[p])
                    row = row_of(ego, Pp, Vp, axp)
                    res[f"W_{p}"] = pol(Hp, ego, r.get("rmem"), cam_t)
                    res[f"R_{p}"] = pol(Hd, row, r.get("rmem"), cam_t)
                    res[f"B_{p}"] = pol(Hp, row, r.get("rmem"), cam_t)
            rec = dict(tag=a.tag, src=name, id=sc, k=k, t0=int(r["t0"]), v=float(np.linalg.norm(V[-1])), u_N=ubar(P),
                       byte_diff=byte_diff, gate_n=gate_n, gate_f=float(np.abs(res["F"][:, :2] - z["poses"][:, :2]).max()),
                       vh_keys=None if vh is None else np.asarray(vh, float).tolist())
            for p, (Pp, _, _) in prof.items():
                rec[f"u_{p}"] = ubar(Pp)
            for x, q in res.items():
                rec[f"A_{x}"] = arc4(q)
                rec[f"P_{x}"] = np.asarray(q, np.float32).ravel().tolist()
            out.append(rec)
    (O / "cf").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out)
    df.to_parquet(a.out or O / "cf" / f"{a.tag}__{name}.parquet")
    print(json.dumps(dict(tag=a.tag, run=str(run), decisions=len(df), skipped=nogate, bytes_equal=int((df.byte_diff == 0).sum()),
                          gate_n_max=float(df.gate_n.max()), gate_f_max=float(df.gate_f.max()), gate_f_p99=float(df.gate_f.quantile(0.99)))))


# ---------------------------------------------------------------- report
def boot(fn, groups, B=2000, seed=0):
    """window-clustered bootstrap of fn(index array) -> (point, lo, hi)"""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(u))]
    pt = fn(np.arange(len(groups)))
    bs = [fn(np.concatenate([idx[i] for i in rng.integers(0, len(u), len(u))])) for _ in range(B)]
    return pt, *np.nanquantile(bs, [0.025, 0.975])


def cmd_report(a):
    import pandas as pd
    C = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(O / "cf/*.parquet")))], ignore_index=True)
    dec = pd.read_parquet(NG.O / "dec.parquet")
    S = pd.read_csv(NG.AR / "set/set.csv.gz").set_index("id")
    win = pd.read_parquet(NG.O / "win.parquet")
    W = pd.read_csv(DATA / "runs/defect_fix/redline_probe/windows.csv").set_index("id")
    C = C.merge(dec[["tag", "id", "k", "lead_gap", "vh", "lat_e"]], on=["tag", "id", "k"], how="left")
    C["stratum"] = S.stratum.reindex(C.id).values
    vmin = dec.sort_values("k").groupby(["tag", "id"]).v.apply(lambda s: np.minimum.accumulate(s.values))
    C["prior_stop"] = [bool(k > 0 and vmin[(t, i)][k - 1] < 0.5) if (t, i) in vmin.index else False for t, i, k in zip(C.tag, C.id, C.k)]
    C["lead"] = C.lead_gap <= 30
    C["dstrat"] = np.where(C.prior_stop, "after stop", np.where(C.lead, "lead <= 30 m", "free"))
    C["cls"] = W.cls.reindex(C.id).values
    sc = win.set_index(["tag", "id"]).score
    C["zero"] = [sc.get((t, i), np.nan) == 0 for t, i in zip(C.tag, C.id)]
    J = {"gate": C.groupby("tag").agg(n=("k", "size"), bytes_equal=("byte_diff", lambda x: float((x == 0).mean())), gate_n_max=("gate_n", "max"),
                                      gate_f_max=("gate_f", "max"), gate_f_p99=("gate_f", lambda x: float(x.quantile(0.99)))).to_dict("index")}
    print("gate\n", pd.DataFrame(J["gate"]).T.to_string())
    stat = C.stratum.eq("static")
    mov = C[~stat & C.src.str.startswith(("JTN1_", "RSG_", "NTV0_")) & (C.k >= 2) & (C.v >= 2)].copy()

    def gain(X, D):
        num = (D[f"A_{X}_s12"] - D[f"A_{X}_s08"]).values
        den = 4.0 * (D.u_s12 - D.u_s08).values
        return lambda i: num[i].sum() / den[i].sum()

    rows = []
    for tag, g0 in mov.groupby("tag"):
        for st, g in [("all", g0)] + list(g0.groupby("dstrat")) + [(f"win {s}", x) for s, x in g0.groupby("stratum")]:
            if len(g) < MINN:
                continue
            ids = g.id.values
            r = dict(tag=tag, stratum=st, n=len(g), windows=g.id.nunique())
            for X in "WRB":
                r[f"G_{X}"] = boot(gain(X, g), ids)
            num = lambda X: (g[f"A_{X}_s12"] - g[f"A_{X}_s08"]).values  # noqa: E731
            den = 4.0 * (g.u_s12 - g.u_s08).values
            r["G_W-G_R"] = boot(lambda i: num("W")[i].sum() / den[i].sum() - num("R")[i].sum() / den[i].sum(), ids)
            # human swap (A3): plan speed = 2 s arc / 2 against the human at the ego's station
            h = g[(g.vh >= 2) & g.A_W_hum.notna()]
            if len(h) >= MINN:
                vpf = np.array([arc4(np.asarray(p).reshape(8, 3)[:4]) / 2 for p in h.P_F])
                for X in "WRB":
                    vpx = np.array([arc4(np.asarray(p).reshape(8, 3)[:4]) / 2 for p in h[f"P_{X}_hum"]])
                    F, Xr, vh_, v = vpf / h.vh.values, vpx / h.vh.values, h.vh.values, h.v.values
                    r[f"rec_{X}"] = boot(lambda i: (Xr[i] - F[i]).sum() / (1 - F[i]).sum(), h.id.values)
                    r[f"hgain_{X}"] = boot(lambda i: (vpx[i] - vpf[i]).sum() / (vh_[i] - v[i]).sum(), h.id.values)
                r["n_hum"], r["planF/h"], r["ego/h"] = len(h), float(np.median(F)), float(np.median(v / vh_))
            for p in ("const", "stand"):
                for X in "WRB":
                    r[f"dA_{X}_{p}"] = float((g[f"A_{X}_{p}"] - g.A_F).median())
            r["A_F"] = float(g.A_F.median())
            rows.append(r)
    J["moving"] = rows
    f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]" if isinstance(t, tuple) else (f"{t:.3f}" if isinstance(t, float) else t)  # noqa: E731
    print("\nmoving\n" + pd.DataFrame([{k: f3(v) for k, v in r.items()} for r in rows]).to_string())
    # static windows (JTN1, redline_probe dumps): plan arc by k per class and outcome
    s = C[stat & (C.tag == "JTN1") & C.cls.notna()].copy()
    srows = []
    for (cls, z), g in s.groupby(["cls", "zero"]):
        for k, gk in g.groupby("k"):
            srows.append(dict(cls=cls, zero=bool(z), k=k, n=len(gk), v=float(gk.v.median()), **{f"A_{x}": float(gk[f"A_{x}"].median()) for x in
                              ("F", "W_stand", "R_stand", "B_stand", "W_const", "W_s08", "W_s12", "W_hum")},
                              lt25_F=float((gk.A_F < 2.5).mean()), lt25_Wst=float((gk.A_W_stand < 2.5).mean())))
    SR = pd.DataFrame(srows)
    J["static"] = srows
    if not len(SR):
        json.dump(J, open(O / "report.json", "w"), indent=1, default=str)
        return
    for cls, g in SR.groupby("cls"):
        print(f"\nstatic {cls}\n" + g.round(3).to_string(index=False))
    # creep growth: mean over k = 5..9 of the median A(k) - A(0)
    cg = []
    for (cls, z), g in SR.groupby(["cls", "zero"]):
        g = g.set_index("k")
        if 0 not in g.index:
            continue
        r = dict(cls=cls, zero=z, windows=int(g.n.max()))
        for x in ("F", "W_stand", "B_stand", "R_stand", "W_const"):
            r[x] = float(g.loc[g.index >= 5, f"A_{x}"].mean() - g.loc[0, f"A_{x}"])
        r["W_stand/F"] = r["W_stand"] / r["F"] if r["F"] else np.nan
        cg.append(r)
    J["creep"] = cg
    print("\ncreep growth (mean_k5..9 median A - A(k0))\n" + pd.DataFrame(cg).round(3).to_string(index=False))
    allz = s[s.zero & (s.k >= 2)]
    J["static_zero_k2"] = dict(n=len(allz), lt25_F=float((allz.A_F < 2.5).mean()), lt25_Wst=float((allz.A_W_stand < 2.5).mean()),
                               lt25_Bst=float((allz.A_B_stand < 2.5).mean()), lt25_Rst=float((allz.A_R_stand < 2.5).mean()))
    print("\nstatic zeros k >= 2:", J["static_zero_k2"])
    json.dump(J, open(O / "report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("cf")
    p.add_argument("--tag", required=True), p.add_argument("--run", required=True), p.add_argument("--threads", type=int, default=12)
    p.add_argument("--limit", type=int, default=0), p.add_argument("--out", default="")
    p.add_argument("--synth", choices=("gpu", "cpu"), default="gpu")
    sp.add_parser("report")
    a = ap.parse_args()
    dict(cf=cmd_cf, report=cmd_report)[a.cmd](a)
