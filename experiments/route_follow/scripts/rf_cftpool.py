#!/usr/bin/env python3
"""Turning school, demonstration source 2: pool `on1_CFT` = rf_cfpool's counterfactual rows (on1_CF: same tokens, same counterfactual route
memory, same road raster, same route.npz for the record) with a TRAJECTORY-shaped imitation target instead of the route hinge
(research/next-round/route-follow.md 3.3 R-CF as first proposed; research/next-round/cft-rows.md; prereg
experiments/route_follow/plans/2026-10-13-cft-prereg.md). CPU only, AlpaSim venv (numpy + the corridor lane-graph Line; no map API needed:
the counterfactual centrelines are rf_cfpool geom's).

  demo   --data navtrain_full.s2of12 ..   per rf_cfpool build row (cf/rows/<data>.npz, every row, ok or not), in the token's rear-axle frame:
                          C   = the geom stage's counterfactual centreline (junction: lane-graph centreline of the alternative exit from the ego's
                                projection; lc: the t0 lane run, "stay in lane"), 1 m samples
                          S_k = the LOG's arc length at t_k = 0, 0.5 .. 4 s (polyline through the origin and tab `fut`): the human's speed profile
                          path P(u) = C(s0 + u) + d(u) n(s0 + u): s0, d0 = the origin's station / left offset on C, d(u) the cubic Hermite from
                                (d0, tan(ego heading - C heading)) to (0, 0) over Lc = clip(4.5 s x S_8 / 4 s, 10, 60) m, 0 beyond: P starts AT the
                                token's pose with its heading (C1 branch, no jump) and merges onto the centreline
                          fut_cft[k] = P at its own arc length S_{k+1}, yaw = P's tangent (a stopped human -> a stopped demo)
                          diagnostics: cont = max |demo - log| at 0.5 / 1.0 s, gap4 = |demo - log| at 4 s, arc4 = demo / log chord arc at 4 s,
                          alat = peak |v x yaw rate| over the 8 half-second steps (demo and log), fchk = max |log frames - tab fut| (frame check)
                          -> $DATA_DIR/runs/route_follow/cf/cft/<data>.npz (aligned to the build rows)
  pack   --data .. [--name CFT]          candidates = on1_CF's rule (build ok, finite logged future, log not in navsim/op-parity-full-dev) with a
                          finite demo; gates on the candidates; registered exclusions (prereg): e1 cont > 0.5 m, e2 demo alat > 4 m/s^2,
                          e3 the demo footprint (nuPlan Pacifica box) overlaps a valid actor box at the same time (agent_labels navtrain_all-k32),
                          e4 a demo sample inside the raster extent with sdf_road <= 0; the rest -> rf_cfpool.write_pool(prefix "cft", fut = demo):
                          cache/on1_<name>/{tab, agent (ok False), road, route}.npz, cache/on1_<name>@warp/{front.npy, teacher.npz},
                          runs/route_mem/feat/on1_<name>.npz; no dev pool (G-swap reads on1_CFdev). The trainer's `--on-mode imit|both` is the only
                          switch (no --on-lon: route.npz, dz included, is never read). Gates -> cf/cft_gates.json + experiments/route_follow/results/,
                          figures experiments/route_follow/figs/cft_bev_{0..5}.png:
                          G-cft1  demo samples inside the raster extent with sdf > 0 >= 99 % (candidates)
                          G-cft2  rows with cont <= 0.5 m >= 95 % (candidates; the failing rows are e1)
                          G-cft3  6 BEV figures (3 junction, 3 lane change, pool rows)
                          G-cft4  rows with |arc4 - 1| <= 10 % (log arc < 1 m: |demo - log arc| <= 0.1 m) >= 99 % (candidates)

  P=$DATA_DIR/third_party/alpasim/.venv/bin/python
  $P experiments/route_follow/scripts/rf_cftpool.py demo --data navtrain_full.s{0..11}of12 && $P .. pack --data navtrain_full.s{0..11}of12
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_cfpool as CF  # noqa: E402  (repo paths, DATA_DIR, map env)

D, CR, OUT, R = CF.D, CF.CR, CF.OUT, CF.R
AGENTS = D / "runs/op_parity/agent_labels/navtrain_all-k32.npz"
T_LC, LC_MIN, LC_MAX, DU = 4.5, 10.0, 60.0, 0.1                       # s of travel, m, m: lateral merge length (chosen on s2, prereg); m path step
CONT, ALAT, ARC = 0.5, 4.0, 0.10                                      # e1 / G-cft2 m, e2 m/s^2, G-cft4 share
EGO_L, EGO_W, EGO_RC = 5.176, 2.297, 1.461                            # nuPlan Pacifica: length, width, rear axle -> box centre


# ---------------------------------------------------------------- demo
def demo(C, fut):
    """C (m, 2) counterfactual centreline in the row frame, fut (8, 3) logged rear-axle future -> (fut_cft (8, 3), diag dict)."""
    import lanegraph as LG
    ln = LG.Line(C)
    s0, d0 = (float(v[0]) for v in ln.frenet(np.zeros((1, 2))))
    psi = float(LG.wrap(-np.interp(s0, ln.S, ln.h)))
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], fut[:, :2]], axis=0).T))]
    Lc = float(np.clip(T_LC * S[8] / 4.0, LC_MIN, LC_MAX))
    u = np.arange(0.0, 1.5 * S[8] + Lc + 5.0, DU)
    x = np.clip(u / Lc, 0, 1)
    d = d0 * (2 * x ** 3 - 3 * x ** 2 + 1) + np.tan(np.clip(psi, -0.6, 0.6)) * Lc * (x ** 3 - 2 * x ** 2 + x)
    B = ln.at(s0 + u)
    P = B[:, :2] + d[:, None] * np.c_[-np.sin(B[:, 2]), np.cos(B[:, 2])]
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    hd = np.unwrap(np.arctan2(*np.diff(P, axis=0).T[::-1]))
    hd = np.r_[hd[0], 0.5 * (hd[1:] + hd[:-1]), hd[-1]]               # tangent at the vertices
    out = np.stack([np.interp(S[1:], cum, P[:, 0]), np.interp(S[1:], cum, P[:, 1]), np.interp(S[1:], cum, hd)], 1).astype(np.float32)
    v = np.diff(S) / 0.5

    def alat(y):
        return float(np.abs(v * np.diff(np.r_[0.0, np.unwrap(y)]) / 0.5).max())

    arc_d = float(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], out[:, :2]], axis=0).T).sum())
    return out, dict(d0=d0, psi=psi, Lc=Lc, cont=float(np.hypot(*(out[:2, :2] - fut[:2, :2]).T).max()), gap4=float(np.hypot(*(out[7, :2] - fut[7, :2]))),
                     arc_log=float(S[8]), arc_demo=arc_d, alat=alat(out[:, 2]), alat_log=alat(fut[:, 2]))


DIAG = ("d0", "psi", "Lc", "cont", "gap4", "arc_log", "arc_demo", "alat", "alat_log", "fchk")


def demo_one(u):
    import corr_geom as CG
    log, items = u
    fr, xy, yaw = CG.log_frames(CF.split_dir(log), log)
    pos = {f["token"]: i for i, f in enumerate(fr)}
    res = []
    for m, tok, cl, fut in items:
        i = pos.get(tok)
        if i is None or cl is None or not np.isfinite(fut).all():
            res.append((m, None, None))
            continue
        C = CG.to_ego(cl, xy[i], yaw[i])
        out, dg = demo(C, fut.astype(np.float64))
        dg["fchk"] = float(np.hypot(*(CG.to_ego(xy[i + 1:i + 9], xy[i], yaw[i]) - fut[:, :2]).T).max()) if i + 8 < len(xy) else np.nan
        res.append((m, out, dg))
    return res


def cmd_demo(a):
    from jevdrive import par
    from jevdrive.run import Run
    with Run("route_follow", "cftpool-demo", config=vars(a)) as run:
        for data in a.data:
            dst = OUT / "cft" / f"{data}.npz"
            if dst.exists() and not a.force:
                run.info(f"{data}: exists")
                continue
            Z = np.load(OUT / "rows" / f"{data}.npz")
            G = {(r["token"], r["cls"]): r["cl"] for r in pickle.load(open(OUT / "geom" / f"{data}.pkl", "rb"))}
            fut = np.load(CR / data / "tab.npz")["fut"]
            by = {}
            for m, (t, c, lg, r) in enumerate(zip(Z["token"].tolist(), Z["cls"].tolist(), Z["log"].tolist(), Z["row"].tolist())):
                by.setdefault(lg, []).append((m, t, G.get((t, c)), fut[r]))
            res = par.pmap(demo_one, sorted(by.items(), key=lambda x: -len(x[1])), run=run, workers=a.workers or None, desc=f"demo {data}")
            n = len(Z["token"])
            F = np.full((n, 8, 3), np.nan, np.float32)
            dg = {k: np.full(n, np.nan, np.float32) for k in DIAG}
            for v in res.values:
                for m, out, d in v or []:
                    if out is not None:
                        F[m] = out
                        for k in DIAG:
                            dg[k][m] = d[k]
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, token=Z["token"], cls=Z["cls"], fut=F, **dg)
            ok = np.isfinite(F).all((1, 2))
            run.info(f"{data}: {n} rows, demo {ok.sum()}, failed logs {len(res.errors)}, cont<=0.5 {np.mean(dg['cont'][ok] <= CONT):.3f}, "
                     f"fchk max {np.nanmax(dg['fchk']):.3f} m")
            run.summary[data] = dict(rows=n, demo=int(ok.sum()), failed=len(res.errors))


# ---------------------------------------------------------------- pack
def boxes_overlap(A, B):
    """Oriented boxes A (..., 5) and B (..., 5) = cx, cy, yaw, length, width (broadcast) -> bool overlap (separating axis test)."""
    def corners(X):
        c, s = np.cos(X[..., 2]), np.sin(X[..., 2])
        hl, hw = X[..., 3] / 2, X[..., 4] / 2
        dx = np.stack([hl, hl, -hl, -hl], -1)
        dy = np.stack([hw, -hw, -hw, hw], -1)
        return np.stack([X[..., :1] + c[..., None] * dx - s[..., None] * dy, X[..., 1:2] + s[..., None] * dx + c[..., None] * dy], -1)
    A, B = np.broadcast_arrays(A, B)
    ca, cb = corners(A), corners(B)
    sep = np.zeros(A.shape[:-1], bool)
    for X in (A, B):
        for th in (X[..., 2], X[..., 2] + np.pi / 2):
            ax = np.stack([np.cos(th), np.sin(th)], -1)[..., None, :]
            pa, pb = (ca * ax).sum(-1), (cb * ax).sum(-1)
            sep |= (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    return ~sep


def ego_box(F):
    """Rear-axle poses (..., 3) -> ego boxes (..., 5)."""
    return np.concatenate([F[..., :2] + EGO_RC * np.stack([np.cos(F[..., 2]), np.sin(F[..., 2])], -1), F[..., 2:3],
                           np.full(F.shape[:-1] + (1,), EGO_L), np.full(F.shape[:-1] + (1,), EGO_W)], -1)


def collide(F, tokens, AG):
    """Per row: the ego box at F (n, 8, 3) overlaps a valid actor box at the same time (t + 0.5 .. 4 s); nan-free rows only; no label -> False."""
    pos = {t: j for j, t in enumerate(AG["tokens"].tolist())}
    out, lab = np.zeros(len(F), bool), np.zeros(len(F), bool)
    for m, t in enumerate(tokens.tolist()):
        j = pos.get(t, -1)
        if j < 0 or not AG["ok"][j]:
            continue
        lab[m] = True
        bx, va = AG["box"][j, 1:], AG["valid"][j, 1:]                 # (8, 32, 5), (8, 32)
        out[m] = bool((boxes_overlap(ego_box(F[m])[:, None], bx) & va).any())
    return out, lab


def figure(path, Zr, sdf, flog, fcft, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rmem as RM
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    yy, xx = np.meshgrid(-24 + 0.5 * (np.arange(96) + 0.5), -8 + 0.5 * (np.arange(128) + 0.5))
    ax.contourf(-yy, xx, sdf.astype(np.float32), levels=[0, 1e3], colors=["#dddddd"])
    ax.contour(-yy, xx, sdf.astype(np.float32), levels=[0], colors=["#888888"], linewidths=0.6)
    if np.isfinite(Zr["real"]).any():
        ax.plot(-Zr["real"][:, 1], Zr["real"][:, 0], "-", color="#1f77b4", lw=1.5, alpha=0.7, label="real route (log track)")
    ax.plot(-Zr["poly"][:, 1], Zr["poly"][:, 0], "-", color="#ff9896", lw=1.5, label="counterfactual route")
    f = Zr["f"].astype(np.float32).reshape(RM.N_PT, RM.NF)
    v = f[:, 0] > 0.5
    ax.plot(-f[v, 2] * 20, f[v, 1] * 20, ".", color="#ff9896", ms=3, label="cf memory (input)")
    ax.plot(np.r_[0, -flog[:, 1]], np.r_[0, flog[:, 0]], "o-", color="#2ca02c", ms=3, lw=1.2, label="logged future 4 s")
    ax.plot(np.r_[0, -fcft[:, 1]], np.r_[0, fcft[:, 0]], "s-", color="#d62728", ms=3, lw=1.4, label="CFT demo 4 s (target)")
    for k in (1, 3, 7):                                                # heading ticks of the demo at 1, 2, 4 s
        ax.plot([-fcft[k, 1], -fcft[k, 1] - 2 * np.sin(fcft[k, 2])], [fcft[k, 0], fcft[k, 0] + 2 * np.cos(fcft[k, 2])], "-", color="#d62728", lw=0.8)
    ax.plot([0], [0], "k^", ms=8)
    ax.set_xlim(-24, 24), ax.set_ylim(-8, 56), ax.set_aspect("equal")
    ax.set_xlabel("-y (m, right +)"), ax.set_ylabel("x (m)")
    ax.set_title(f"{Zr['token'][:10]} {Zr['cls']} cmd={'LSRU'[Zr['cmd']]} dh {Zr['dh']:.0f} deg branch {Zr['branch']:.0f} m\n{info}", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_pack(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    dvs = splits.load(CF.DEV)
    with Run("route_follow", f"cftpool-pack-{a.name}", config=vars(a)) as run:
        run.use_split(dvs)
        zs = [dict(np.load(OUT / "rows" / f"{d}.npz")) for d in a.data]
        ds = [dict(np.load(OUT / "cft" / f"{d}.npz")) for d in a.data]
        for z, q in zip(zs, ds):
            assert (z["token"] == q["token"]).all() and (z["cls"] == q["cls"]).all()
        Z = {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}
        Q = {k: np.concatenate([q[k] for q in ds]) for k in ds[0]}
        Z["data"] = np.concatenate([np.full(len(z["row"]), s) for s, z in enumerate(zs)])
        futs = [np.load(CR / d / "tab.npz")["fut"] for d in a.data]
        flog = np.stack([futs[s][r] for s, r in zip(Z["data"], Z["row"])])
        dev_logs = set()
        for d in a.data:
            t = np.load(CR / d / "tab.npz")
            dev_logs |= set(t["log"][dvs.mask(t["names"])].tolist())
        is_dev = np.isin(Z["log"], list(dev_logs))
        cf_good = Z["ok"] & np.isfinite(flog).all((1, 2)) & ~is_dev        # exactly on1_CF's rows
        cand = np.flatnonzero(cf_good & np.isfinite(Q["fut"]).all((1, 2)))
        road = np.load(CF.ROAD)
        road = dict(tokens=road["tokens"], ok=road["ok"], sdf=road["sdf"])
        rpos = {t: j for j, t in enumerate(road["tokens"].tolist())}
        ri = np.array([rpos.get(t, -1) for t in Z["token"][cand].tolist()])
        sd, sl = [], []
        for m, j in enumerate(ri):
            s = road["sdf"][j] if j >= 0 and road["ok"][j] else None
            sd.append(CF.sdf_at(s, Q["fut"][cand[m], :, :2]) if s is not None else np.full(8, np.nan))
            sl.append(CF.sdf_at(s, flog[cand[m], :, :2]) if s is not None else np.full(8, np.nan))
        sd, sl = np.array(sd), np.array(sl)
        AG = np.load(AGENTS)
        AG = dict(tokens=AG["tokens"], ok=AG["ok"], box=AG["box"], valid=AG["valid"])
        col, lab = collide(Q["fut"][cand], Z["token"][cand], AG)
        col_log, _ = collide(flog[cand], Z["token"][cand], AG)
        cls = Z["cls"][cand]
        cont, alat, al_log = Q["cont"][cand], Q["alat"][cand], Q["alat_log"][cand]
        arc_l, arc_d = Q["arc_log"][cand], Q["arc_demo"][cand]
        arc_ok = np.where(arc_l >= 1.0, np.abs(arc_d / np.maximum(arc_l, 1e-6) - 1) <= ARC, np.abs(arc_d - arc_l) <= 0.1)
        e1, e2, e3, e4 = cont > CONT, alat > ALAT, col, (np.nan_to_num(sd, nan=1.0) <= 0).any(1)
        keep = ~(e1 | e2 | e3 | e4)
        fin = np.isfinite(sd)
        bycls = lambda x: {c: (float(x[cls == c].mean()) if (cls == c).any() else None) for c in ("j0", "j1", "lc")}  # noqa: E731
        G = {"data": a.data,
             "rows": {"cf_rows": int(cf_good.sum()), "candidates": int(len(cand)), "pool": int(keep.sum()),
                      "pool_by_class": {c: int((keep & (cls == c)).sum()) for c in ("j0", "j1", "lc")},
                      "candidates_by_class": {c: int((cls == c).sum()) for c in ("j0", "j1", "lc")},
                      "tokens": int(len(set(Z["token"][cand][keep].tolist()))), "logs": int(len(set(Z["log"][cand][keep].tolist())))},
             "exclusions": {"e1_cont": int(e1.sum()), "e2_alat": int(e2.sum()), "e3_actor_overlap": int(e3.sum()), "e4_offroad": int(e4.sum()),
                            "any": int((~keep).sum()), "by_class": {c: {k: int((x & (cls == c)).sum()) for k, x in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4))}
                                                                    for c in ("j0", "j1", "lc")},
                            "actor_label_rows": int(lab.sum()), "log_future_actor_overlap": int(col_log.sum())},
             "G_cft1": {"samples": int(fin.sum()), "demo_inside": float((sd[fin] > 0).mean()), "log_inside": float((sl[np.isfinite(sl)] > 0).mean()),
                        "demo_inside_by_class": {c: float((sd[cls == c][fin[cls == c]] > 0).mean()) if (cls == c).any() else None for c in ("j0", "j1", "lc")},
                        "pool_demo_inside": float((sd[keep][fin[keep]] > 0).mean())},
             "G_cft2": {"rows_cont_le_0p5": float((cont <= CONT).mean()), "by_class": bycls(cont <= CONT), "median_cont": float(np.median(cont)),
                        "p95_cont": float(np.quantile(cont, 0.95))},
             "G_cft4": {"rows_arc_within_10pct": float(arc_ok.mean()), "median_ratio": float(np.median(arc_d[arc_l >= 1] / arc_l[arc_l >= 1])),
                        "stopped_rows_log_arc_lt_1m": int((arc_l < 1).sum())},
             "diag": {"gap4_median_by_class": {c: float(np.median(Q["gap4"][cand][cls == c])) if (cls == c).any() else None for c in ("j0", "j1", "lc")},
                      "gap4_ge_1m_pool": float((Q["gap4"][cand][keep] >= 1).mean()), "alat_p95_demo": float(np.quantile(alat, 0.95)),
                      "alat_p95_log": float(np.quantile(al_log, 0.95)), "d0_abs_median": float(np.median(np.abs(Q["d0"][cand]))),
                      "psi_abs_median_deg": float(np.degrees(np.median(np.abs(Q["psi"][cand])))), "frame_check_log_vs_tab_fut_max_m": float(np.nanmax(Q["fchk"][cand]))}}
        G["G_cft1"]["pass"] = G["G_cft1"]["demo_inside"] >= 0.99
        G["G_cft2"]["pass"] = G["G_cft2"]["rows_cont_le_0p5"] >= 0.95
        G["G_cft4"]["pass"] = G["G_cft4"]["rows_arc_within_10pct"] >= 0.99
        sel = cand[keep]
        if not a.gates_only:
            CF.write_pool(a.name, Z, sel, a.data, road, run, prefix="cft", fut=Q["fut"][sel])
        figs = R / "experiments/route_follow/figs"
        figs.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        kc = cls[keep]
        pick = [m for c in ("j", "lc") for m in rng.permutation(np.flatnonzero(np.char.startswith(kc.astype(str), c)))[:3]]
        G["G_cft3"] = []
        for q, m in enumerate(pick):
            g = sel[m]
            j = rpos.get(Z["token"][g], -1)
            p = figs / f"cft_bev_{q}.png"
            info = f"cont {Q['cont'][g]:.2f} m, gap4 {Q['gap4'][g]:.1f} m, arc4 log {Q['arc_log'][g]:.1f} / demo {Q['arc_demo'][g]:.1f} m, alat {Q['alat'][g]:.1f}"
            figure(p, {k: v[g] for k, v in Z.items()}, road["sdf"][j] if j >= 0 else np.zeros((128, 96), np.float16), flog[g], Q["fut"][g], info)
            G["G_cft3"].append(dict(fig=str(p.relative_to(R)), token=str(Z["token"][g]), cls=str(Z["cls"][g])))
        tag = "" if a.name == "CFT" else f"_{a.name}"
        (OUT / f"cft_gates{tag}.json").write_text(json.dumps(G, indent=1))
        shutil.copy(OUT / f"cft_gates{tag}.json", R / f"experiments/route_follow/results/cft_gates{tag}.json")
        run.summary.update(G)
        print(json.dumps(G, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--data", nargs="+", required=True), p.add_argument("--workers", type=int, default=0), p.add_argument("--force", action="store_true")
    p = sub.add_parser("pack")
    p.add_argument("--data", nargs="+", required=True), p.add_argument("--name", default="CFT"), p.add_argument("--gates-only", action="store_true")
    a = ap.parse_args()
    dict(demo=cmd_demo, pack=cmd_pack)[a.cmd](a)
