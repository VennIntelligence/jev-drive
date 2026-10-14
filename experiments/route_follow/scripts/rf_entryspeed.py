#!/usr/bin/env python3
"""ENTRY-SPEED ROWS (plans/2026-10-14-entry-speed-rows-prereg.md, results/entry_speed_rows/README.md): are the hinge rows of fast turn
entries waste (just before / after a terminal zero, or far off the route) and those of slow entries recoverable? CPU only, stored rollouts
and row pools read, never written. Definitions: the prereg (copied in short below).

  extract-nup  (box, .venv: numpy + pandas)  rf_entryspeed.py extract-nup --pool L18RB0=RB0-F-s0@18=<rf_follow dir> [...] --out <dir>
               per pool: the rf_follow turn scenes of that label, their pool rows (meta / active / route.npz) and every decision (all.npz
               + rf_follow decisions.parquet) -> <dir>/nup_<pool>.pkl
  extract-p3   (Tokyo, plain python3 + numpy)  rf_entryspeed.py extract-p3 --arms /data/runs/route_night/p3/arms --out <pkl>
               per P3 rollout: controller trace (t, x, y, yaw, vx) and per decision (k, now, anchor, adapter plan `poses_model`, push on)
  report       (Mac, .venv: torch for on_route)  rf_entryspeed.py report --nup <dir> --p3 <pkl> --read <p3 read.json> --plan <plan_p3.json>
               --out <results dir> --figs <figs dir>

Cell = scene x rollout x strict turn (N: rf_follow's turn; P: plan_p3's). Decisions of a cell: anchor station in [entry - 40, exit + 10] m
(N: pool rows; P: 10 Hz decisions from the handover k_eff on). hinge-active: the trainers' own values on the row's own plan (N: active.npz
agent / road > 0, on_route.loss > 0 on meta.poses with the pool's route.npz; P: on_route.loss on poses_model with rm_build.label's polyline,
dz 1.0). recoverable: not terminated, no zero event within T s, |lat to the route| <= L, |heading error| <= H. usable = both.
v_ratio = v_entry / v_h (the log's speed at the same entry station; P: plan v_log_entry).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
R_ = HERE.parents[2]
sys.path[:0] = [str(R_), str(HERE), str(R_ / "experiments/onpol_nuplan/lib"), str(R_ / "experiments/route_mem/lib"),
                str(R_ / "experiments/route_mem/scripts")]

PRE, PAD, DT_N = 40.0, 10.0, 0.5
T0, L0, H0 = 3.0, 1.0, 20.0                                   # primary T (s), L (m), H (deg)
GRID = [(T, L, H) for T in (1.5, 3.0, 5.0) for L in (0.5, 1.0, 2.0) for H in (10.0, 20.0, 30.0)]
EDGES_FIXED = (0.9, 1.1, 1.3)
ABS_EDGES = (3.0, 5.0, 7.0)
VH_MIN, NEAR_ENTRY = 2.0, 10.0
B_BOOT, B_GLM = 10_000, 2000


# ---------------------------------------------------------------- extract-nup (box)
def extract_nup(a):
    import pandas as pd
    D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.pool:
        pool, label, fdir = spec.split("=")
        S = pd.read_csv(Path(fdir) / "scenes.csv")
        S = S[(S.label == label) & (S.turn == True)]                                   # noqa: E712
        Dx = pd.read_parquet(Path(fdir) / "decisions.parquet")
        Dx = Dx[(Dx.label == label) & Dx.scene.isin(S.scene)]
        rd = D / "runs/onpol_nuplan/rows" / pool
        m = np.load(rd / "meta.npz", allow_pickle=True)
        act = np.load(rd / "active.npz", allow_pickle=True)
        al = np.load(rd / "all.npz", allow_pickle=True)
        rt = np.load(D / "runs/op_parity/cache" / f"on1_{pool}" / "route.npz", allow_pickle=True)
        assert (m["names"] == act["names"]).all()
        pos = {t: i for i, t in enumerate(rt["tokens"].tolist())}
        keep = np.flatnonzero(np.isin(m["scene"], S.scene.to_numpy()))
        ri = np.array([pos.get(n, -1) for n in m["names"][keep]])
        rows = dict(name=m["names"][keep], scene=m["scene"][keep], k=m["k"][keep], t_us=m["t_us"][keep], poses=m["poses"][keep].astype(np.float32),
                    logfut0=m["logfut"][keep][:, 0], v_log=m["v_log"][keep], speed=m["speed"][keep], e0=m["e0"][keep], psi_e=m["psi_e"][keep],
                    agent=act["agent"][keep], road=act["road"][keep], r_ok=np.where(ri >= 0, rt["ok"][np.maximum(ri, 0)], False),
                    poly=rt["poly"][np.maximum(ri, 0)], seglen=rt["seglen"][np.maximum(ri, 0)], dz=rt["dz"][np.maximum(ri, 0)])
        ka = np.isin(al["scene"], S.scene.to_numpy())
        alld = dict(scene=al["scene"][ka], k=al["k"][ka], t_us=al["t_us"][ka], excl=al["excl"][ka], row=al["row"][ka])
        J = dict(pool=pool, label=label, follow=str(fdir), scenes=S.to_dict("records"), dec=Dx.to_dict("list"), rows=rows, all=alld)
        pickle.dump(J, open(out / f"nup_{pool}.pkl", "wb"), protocol=4)
        print(pool, label, "turn scenes", len(S), "rows", len(keep), "route rows", int((ri >= 0).sum()), "decisions", len(Dx), flush=True)


# ---------------------------------------------------------------- extract-p3 (Tokyo, no pandas)
def extract_p3(a):
    import csv
    out = []
    for arm_dir in sorted(p for p in glob.glob(os.path.join(a.arms, "*")) if os.path.isdir(p)):
        arm = os.path.basename(arm_dir)
        for run in sorted(glob.glob(os.path.join(arm_dir, "runs", "*"))):
            f = os.path.join(run, "sim", "aggregate", "results-summary.json")
            if not os.path.exists(f):
                continue
            dec = {}
            for line in open(os.path.join(run, "driver", "drive.jsonl")):
                if '"kind": "drive"' not in line:
                    continue
                r = json.loads(line)
                if "poses" not in r or "anchor" not in r:
                    continue
                dec.setdefault(r["session"], []).append((r["k"], r["now"], r["anchor"], r.get("poses_model", r["poses"]),
                                                         bool((r.get("branch") or {}).get("on"))))
            for ro in json.load(open(f))["rollouts"]:
                rid = ro["rollout_id"]
                cf = os.path.join(run, "sim", "controller", f"alpasim_controller_{rid}.csv")
                if not os.path.exists(cf):
                    continue
                rows = list(csv.reader(open(cf)))
                h = {k: i for i, k in enumerate(rows[0])}
                A = np.array(rows[1:], float)
                c = lambda k: A[:, h[k]]  # noqa: E731
                qx, qy, qz, qw = c("qx"), c("qy"), c("qz"), c("qw")
                Dd = sorted(dec.get(rid, []), key=lambda x: x[0])
                out.append(dict(arm=arm, scene=ro["clipgt_id"], rid=rid, score=ro.get("score"),
                                ctrl=dict(t=c("timestamp_us"), x=c("x"), y=c("y"), yaw=np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))),
                                          vx=c("vx")),
                                k=np.array([d[0] for d in Dd]), now=np.array([d[1] for d in Dd], float), anchor=np.array([d[2] for d in Dd], float),
                                plan=np.array([d[3] for d in Dd], np.float32), on=np.array([d[4] for d in Dd])))
        print(arm, len([o for o in out if o["arm"] == arm]), flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(out), "rollouts ->", a.out)


# ---------------------------------------------------------------- shared geometry (report)
def route_terms(poses, poly, seglen, dz, ok):
    """The trainer's route hinge (on_route.loss) on the row's own plan, and the anchor's distance to the label (on_route.near at the origin)."""
    import torch
    import on_route as OR
    t = lambda x: torch.as_tensor(np.asarray(x))  # noqa: E731
    n = len(poses)
    pack = dict(poly=t(poly.astype(np.float32)), seglen=t(seglen.astype(np.float32)), ok=t(ok.astype(bool)), dz=t(dz.astype(np.float32)),
                drop=t(np.zeros(n, bool)))
    rows = torch.arange(n)
    with torch.no_grad():
        h = OR.loss(t(poses.astype(np.float32)), rows, pack)["route"].numpy()
        lat = OR.near(torch.zeros(n, 1, 3), rows, pack)[:, 0].numpy().astype(float)
    lat[(lat >= 1e3) | ~ok] = np.nan
    P = poly.astype(float)                                          # offset 0 m (index 5) = the anchor's projection; tangent over +-2 m
    head = np.degrees(np.abs(np.arctan2(P[:, 6, 1] - P[:, 4, 1], P[:, 6, 0] - P[:, 4, 0])))
    head[~ok | ~np.isfinite(head)] = np.nan
    return h, lat, head


def along(poly, q):
    """Along-route offset (m) of row-frame points q (n, 2) relative to the label's offset-0 point (index 5), nearest interior segment."""
    out = np.full(len(q), np.nan)
    for i in range(len(q)):
        P = poly[i]
        ok = np.isfinite(P).all(1)
        idx = np.flatnonzero(ok[:-1] & ok[1:])
        if not len(idx) or not np.isfinite(q[i]).all():
            continue
        a, b = P[idx], P[idx + 1]
        d = b - a
        L2 = np.maximum((d * d).sum(1), 1e-12)
        tt = np.clip(((q[i] - a) * d).sum(1) / L2, 0, 1)
        j = int(np.argmin(np.hypot(*(a + tt[:, None] * d - q[i]).T)))
        out[i] = -10.0 + 2.0 * idx[j] + tt[j] * np.sqrt(L2[j])
    return out


def cross_speed(st, v, s_target):
    """Speed where the station track first crosses s_target (linear in station), else at the nearest decision within NEAR_ENTRY m."""
    st, v = np.asarray(st, float), np.asarray(v, float)
    i = np.flatnonzero(st >= s_target)
    if len(i) and i[0] > 0:
        j = i[0]
        w = (s_target - st[j - 1]) / max(st[j] - st[j - 1], 1e-9)
        return float((1 - w) * v[j - 1] + w * v[j])
    j = int(np.nanargmin(np.abs(st - s_target))) if np.isfinite(st).any() else -1
    return float(v[j]) if j >= 0 and abs(st[j] - s_target) <= NEAR_ENTRY else np.nan


# ---------------------------------------------------------------- report: N
def nup_table(f):
    """One pool pkl -> (cells, decisions) DataFrames; decisions = the pool rows inside the cell windows, with every label input."""
    import pandas as pd
    J = pickle.load(open(f, "rb"))
    S = pd.DataFrame(J["scenes"]).set_index("scene")
    Dx = pd.DataFrame(J["dec"])
    R = J["rows"]
    h, lat, head = route_terms(R["poses"], R["poly"], R["seglen"], R["dz"], R["r_ok"])
    X = pd.DataFrame(dict(scene=R["scene"], k=R["k"].astype(int), route=h, lat=lat, head=head, agent=R["agent"], road=R["road"], r_ok=R["r_ok"],
                          v_log=R["v_log"], speed=R["speed"]))
    X["log_off"] = along(R["poly"].astype(float), R["logfut0"][:, :2].astype(float))
    st = Dx.set_index(["scene", "k"]).st
    X["st"] = [st.get((s, k), np.nan) for s, k in zip(X.scene, X.k)]
    cells, rows = [], []
    for sc, g in X.groupby("scene"):
        s = S.loc[sc]
        d = Dx[Dx.scene == sc].sort_values("k")
        ent, ext, kz = float(s.entry), float(s.exit), int(s.kz) if int(s.kz) >= 0 else 10 ** 6
        v_entry = cross_speed(d.st, d.v, ent)
        g = g.sort_values("k")
        sl = (g.st + g.log_off).to_numpy()
        ok = np.isfinite(sl)
        o = np.argsort(sl[ok])
        v_h = np.nan
        if ok.sum() >= 2 and sl[ok][o][0] <= ent + 1.0 and sl[ok][o][-1] >= ent:
            v_h = float(np.interp(ent, sl[ok][o], g.v_log.to_numpy()[ok][o]))
        elif ok.any() and abs(sl[ok][o][0] - ent) <= 5.0:
            v_h = float(g.v_log.to_numpy()[ok][o][0])
        win = (g.st >= ent - PRE) & (g.st <= ext + PAD)
        w = g[win].copy()
        w["dt_zero"] = (kz - w.k) * DT_N if kz < 10 ** 6 else np.inf
        w["term"] = w.k >= kz
        al = J["all"]
        ka = (al["scene"] == sc)
        alk = pd.DataFrame(dict(k=al["k"][ka], excl=al["excl"][ka]))
        alk["st"] = [st.get((sc, k), np.nan) for k in alk.k]
        n_excl = int(((alk.excl != "") & (alk.st >= ent - PRE) & (alk.st <= ext + PAD)).sum())
        mid = (ent + ext) / 2
        v_turn = d[(d.st >= ent) & (d.st <= ext)].v.mean() if ((d.st >= ent) & (d.st <= ext)).any() else np.nan
        cid = f"{J['pool']}|{sc}"
        cells.append(dict(cell=cid, src="N", pool=J["pool"], label=J["label"], scene=sc, held=J["label"].endswith("@held"), side=s.side,
                          entry=ent, exit=ext, kz=kz if kz < 10 ** 6 else -1, completed=bool(s.completed), outcome=s.outcome, v_entry=v_entry,
                          v_h=v_h, v_turn=v_turn, n_win=int(len(w)), n_excl=n_excl, rear=False, mid=mid))
        w["cell"] = cid
        rows.append(w)
    return pd.DataFrame(cells), pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- report: P
def p3_table(p3_pkl, read_json, plan_json):
    import pandas as pd
    import rf_pai_follow as F
    import rf_shape as SH
    os.environ.setdefault("DATA_DIR", str(R_ / "tmp"))                 # rm_build reads it at import; nothing is written there
    from rm_build import label
    PL = json.load(open(plan_json))["scenes"]
    rd = {(r["arm"], r["scene"]): r for r in json.load(open(read_json))["rows"]}
    cells, rows = [], []
    for o in pickle.load(open(p3_pkl, "rb")):
        r = rd.get((o["arm"], o["scene"]))
        p = PL.get(o["scene"])
        if r is None or p is None:
            continue
        Rg, Sg = np.c_[p["x"], p["y"]], np.asarray(p["u"], float)
        ent, ext = float(p["s_entry"]), float(p["s_exit"])
        C = o["ctrl"]
        t0 = C["t"][0]
        stc, _ = F.track(Rg, Sg, np.c_[C["x"], C["y"]])
        sp = SH.speeds(C["t"], stc, C["vx"], ent, ext)
        m = (stc >= ent) & (stc <= ext)
        v_turn = float(np.mean(C["vx"][m])) if m.any() else np.nan
        sta, _ = F.track(Rg, Sg, o["anchor"][:, :2])
        k_eff = int(r["k"])
        sel = np.flatnonzero((o["k"] >= k_eff) & (sta >= ent - PRE) & (sta <= ext + PAD))
        tz = r["t_zero"] if r.get("t_zero") is not None else np.inf
        ho, tf = o["arm"].split("@")
        g, s = tf[1:].split("s")
        cid = f"P3|{o['arm']}|{o['scene']}"
        cells.append(dict(cell=cid, src="P", pool="P3", label=o["arm"], scene=o["scene"], held=False, side=p["side"], entry=ent, exit=ext,
                          kz=-1, completed=bool(r["done"]), outcome=r.get("zero") or r.get("end"), v_entry=sp["v_0"] if np.isfinite(sp["v_0"]) else np.nan,
                          v_h=float(p["v_log_entry"]), v_e=float(p["v_e"]), v_turn=v_turn, n_win=int(len(sel)), n_excl=0, rear=r.get("end") == "rear",
                          ho=ho, g=float(g), s=float(s), t_zero=tz))
        if not len(sel):
            continue
        A = o["anchor"][sel]
        poly = np.stack([label(Rg, a) for a in A]).astype(np.float32)
        seglen = np.linalg.norm(np.diff(poly, axis=1), axis=-1).astype(np.float32)
        okp = np.isfinite(poly).all(-1).sum(1) >= 2
        h, lat, head = route_terms(o["plan"][sel], poly, seglen, np.ones(len(sel), np.float32), okp)
        tdec = (o["now"][sel] - t0) * 1e-6
        rows.append(pd.DataFrame(dict(cell=cid, scene=o["scene"], k=o["k"][sel], route=h, lat=lat, head=head, agent=np.nan, road=np.nan,
                                      r_ok=okp, st=sta[sel], dt_zero=tz - tdec, term=tdec >= tz, on=o["on"][sel])))
    return pd.DataFrame(cells), pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- labels and statistics
def label_rows(X, T, L, H, terms=("route", "road", "agent")):
    act = np.zeros(len(X), bool)
    for t in terms:
        act |= (X[t].fillna(0).to_numpy() > 0)
    term = X.term.to_numpy(bool)
    near = (~term) & (X.dt_zero.to_numpy() <= T)
    off = ~((X.lat.to_numpy() <= L) & (X["head"].to_numpy() <= H))           # NaN geometry counts as off
    rec = ~term & ~near & ~off
    return act, rec, term, near, off


def cell_counts(cells, X, T=T0, L=L0, H=H0, terms=("route", "road", "agent")):
    import pandas as pd
    act, rec, term, near, off = label_rows(X, T, L, H, terms)
    Y = pd.DataFrame(dict(cell=X.cell, act=act, use=act & rec, w_term=act & term, w_near=act & near & ~term, w_off=act & off & ~term & ~near))
    G = Y.groupby("cell").sum()
    c = cells.set_index("cell").join(G).fillna({k: 0 for k in G.columns})
    c["share"] = np.where(c.act > 0, c.use / c.act.where(c.act > 0, 1), np.nan)
    return c.reset_index()


def boot_idx(n, B, seed=0):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def bin_stats(c, by, labels, B=B_BOOT):
    """Per bin: usable share (ratio of sums over hinge-active rows), usable count per cell, completion; scene-cluster bootstrap; and the
    contrast first bin - last bin with one draw of scenes for both."""
    sc = np.array(sorted(c.scene.unique()))
    si = {s: i for i, s in enumerate(sc)}
    nb = len(labels)
    U, A, N, Cc, Uc = (np.zeros((len(sc), nb)) for _ in range(5))
    for _, r in c.iterrows():
        if r[by] not in labels:
            continue
        i, j = si[r.scene], labels.index(r[by])
        U[i, j] += r.use
        A[i, j] += r.act
        N[i, j] += 1
        Cc[i, j] += r.completed
        Uc[i, j] += r.use
    I = boot_idx(len(sc), B)
    sU, sA, sN, sC = U[I].sum(1), A[I].sum(1), N[I].sum(1), Cc[I].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share_b, comp_b, cnt_b = sU / sA, sC / sN, sU / sN
    q = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))  # noqa: E731
    out = []
    for j, lab in enumerate(labels):
        a_, n_ = A[:, j].sum(), N[:, j].sum()
        out.append(dict(bin=lab, cells=int(n_), scenes=int((N[:, j] > 0).sum()), active=int(a_), usable=int(U[:, j].sum()),
                        share=float(U[:, j].sum() / a_) if a_ else np.nan, share_ci=q(share_b[:, j]),
                        count=float(U[:, j].sum() / n_) if n_ else np.nan, count_ci=q(cnt_b[:, j]),
                        completion=float(Cc[:, j].sum() / n_) if n_ else np.nan, completion_ci=q(comp_b[:, j])))
    D = dict(share=float(out[0]["share"] - out[-1]["share"]), share_ci=q(share_b[:, 0] - share_b[:, -1]),
             count=float(out[0]["count"] - out[-1]["count"]), count_ci=q(cnt_b[:, 0] - cnt_b[:, -1]),
             completion=float(out[0]["completion"] - out[-1]["completion"]), completion_ci=q(comp_b[:, 0] - comp_b[:, -1]))
    pooled = float(Cc.sum() / N.sum())
    return out, D, pooled


def binize(x, edges):
    names = [f"< {edges[0]:.2f}"] + [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])] + [f">= {edges[-1]:.2f}"]
    lab = np.array([None] * len(x), object)
    ok = np.isfinite(x)
    lab[ok] = [names[i] for i in np.searchsorted(edges, x[ok], side="right")]
    return lab, names


def glm(y, n, x, groups=None, ridge=1e-3, iters=50):
    """Binomial GLM (logit) of y successes of n trials on x (+ group fixed effects): IRLS, returns the slope on x."""
    cols = [np.ones(len(x)), x]
    if groups is not None:
        u, gi = np.unique(groups, return_inverse=True)
        Z = np.zeros((len(x), len(u)))
        Z[np.arange(len(x)), gi] = 1.0
        cols = [x[:, None], Z]
        Xm = np.hstack(cols)
        pen = np.r_[0.0, np.full(len(u), ridge)]
    else:
        Xm = np.stack(cols, 1)
        pen = np.zeros(2)
    b = np.zeros(Xm.shape[1])
    for _ in range(iters):
        eta = np.clip(Xm @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = n * p * (1 - p) + 1e-9
        g = Xm.T @ (y - n * p) - pen * b
        Hm = (Xm * W[:, None]).T @ Xm + np.diag(pen + 1e-9)
        step = np.linalg.solve(Hm, g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return float(b[0] if groups is not None else b[1])


def glm_boot(c, B=B_GLM, fe=False):
    c = c[np.isfinite(c.v_ratio) & (c.act > 0)]
    sc = np.array(sorted(c.scene.unique()))
    by = {s: c[c.scene == s] for s in sc}
    y, n, x = c.use.to_numpy(float), c.act.to_numpy(float), c.v_ratio.to_numpy(float)
    est = glm(y, n, x, c.scene.to_numpy() if fe else None)
    I = boot_idx(len(sc), B)
    bs = []
    for row in I:
        parts = [by[sc[i]].assign(scene=f"{sc[i]}#{j}") for j, i in enumerate(row)]
        d = np.concatenate([p.use.to_numpy(float) for p in parts]), np.concatenate([p.act.to_numpy(float) for p in parts]), \
            np.concatenate([p.v_ratio.to_numpy(float) for p in parts]), np.concatenate([p.scene.to_numpy() for p in parts])
        bs.append(glm(d[0], d[1], d[2], d[3] if fe else None))
    return dict(slope=est, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), cells=int(len(c)), scenes=int(len(sc)))


def p3_contrast(c, B=B_BOOT, s_set=(0.8, 0.6)):
    """Per scene x handover x g in {1, 1.5}: share(s < 1) - share(s = 1); mean over pairs; scene-cluster bootstrap of the pair mean."""
    pairs = []
    for (sc, ho, g), x in c[c.g.isin([1.0, 1.5])].groupby(["scene", "ho", "g"]):
        base = x[x.s == 1.0]
        if not len(base) or not np.isfinite(base.share.iloc[0]):
            continue
        for s in s_set:
            y = x[x.s == s]
            if len(y) and np.isfinite(y.share.iloc[0]):
                pairs.append(dict(scene=sc, ho=ho, g=g, s=s, d=float(y.share.iloc[0] - base.share.iloc[0]),
                                  d_use=float(y.use.iloc[0] - base.use.iloc[0]), d_act=float(y.act.iloc[0] - base.act.iloc[0]),
                                  d_vturn=float(y.v_turn.iloc[0] - base.v_turn.iloc[0]), d_ventry=float(y.v_entry.iloc[0] - base.v_entry.iloc[0]),
                                  d_comp=float(y.completed.iloc[0]) - float(base.completed.iloc[0])))
    if not pairs:
        return dict(n=0), pairs
    import pandas as pd
    P = pd.DataFrame(pairs)
    sc = np.array(sorted(P.scene.unique()))
    S = np.array([[P[P.scene == s][k].sum() for k in ("d", "d_use", "d_act", "d_vturn", "d_comp")] + [float((P.scene == s).sum())] for s in sc])
    I = boot_idx(len(sc), B)
    Sb = S[I].sum(1)
    res = dict(n=int(len(P)), scenes=int(len(sc)))
    for j, k in enumerate(("d", "d_use", "d_act", "d_vturn", "d_comp")):
        res[k] = float(P[k].mean())
        bb = Sb[:, j] / Sb[:, -1]
        res[k + "_ci"] = (float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5)))
    return res, pairs


def fmt(v, ci=None, p=3):
    if v is None or not np.isfinite(v):
        return "-"
    s = f"{v:.{p}f}"
    return s + (f" [{ci[0]:.{p}f}, {ci[1]:.{p}f}]" if ci is not None and np.isfinite(ci[0]) else "")


# ---------------------------------------------------------------- report
def report(a):
    import pandas as pd
    out, figs = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True)
    cn, xn = zip(*[nup_table(f) for f in sorted(glob.glob(str(Path(a.nup) / "nup_*.pkl")))])
    CN, XN = pd.concat(cn, ignore_index=True), pd.concat(xn, ignore_index=True)
    CP, XP = p3_table(a.p3, a.read, a.plan)
    for c in (CN, CP):
        c["v_ratio"] = np.where(c.v_h >= VH_MIN, c.v_entry / c.v_h, np.nan)
    J = dict(prereg="plans/2026-10-14-entry-speed-rows-prereg.md", primary=dict(T=T0, L=L0, H=H0))
    # gates / inputs
    J["inputs"] = dict(N_cells=int(len(CN)), N_scenes=int(CN.scene.nunique()), N_cells_by_pool=CN.pool.value_counts().to_dict(),
                       N_rows_in_windows=int(len(XN)), N_rows_no_station=int(XN.st.isna().sum()), N_route_ok=float(XN.r_ok.mean()),
                       N_cells_no_ventry=int(CN.v_entry.isna().sum()), N_cells_no_vh=int((~(CN.v_h >= VH_MIN)).sum()),
                       N_excluded_decisions_in_windows=int(CN.n_excl.sum()),
                       P_cells=int(len(CP)), P_rows=int(len(XP)), P_cells_no_ventry=int(CP.v_entry.isna().sum()), P_route_ok=float(XP.r_ok.mean()),
                       N_lat_median=float(XN.lat.median()), P_lat_median=float(XP.lat.median()))
    # primary cell counts
    cN = cell_counts(CN, XN)
    cP = cell_counts(CP, XP, terms=("route",))
    cNr = cell_counts(CN, XN, terms=("route",))
    vr = cN.v_ratio.to_numpy(float)
    lab_f, names_f = binize(vr, EDGES_FIXED)
    okb = all(((lab_f == n).sum() >= 15 and cN[lab_f == n].scene.nunique() >= 8) for n in names_f)
    edges = EDGES_FIXED if okb else tuple(float(np.nanpercentile(vr, q)) for q in (25, 50, 75))
    J["bins"] = dict(edges=edges, rule="fixed" if okb else "quartiles", fixed_counts={n: int((lab_f == n).sum()) for n in names_f},
                     fixed_scenes={n: int(cN[lab_f == n].scene.nunique()) for n in names_f}, v_ratio_quantiles=[float(np.nanpercentile(vr, q)) for q in (5, 25, 50, 75, 95)])
    cN["bin"], names = binize(vr, edges)
    cNr["bin"] = cN["bin"].to_numpy()
    cN["abin"], anames = binize(cN.v_entry.to_numpy(float), ABS_EDGES)
    J["N_bins"], J["N_D"], J["N_completion_pooled"] = bin_stats(cN, "bin", names)
    J["N_bins_route"], J["N_D_route"], _ = bin_stats(cNr, "bin", names)
    J["N_abs"], J["N_abs_D"], _ = bin_stats(cN, "abin", anames)
    for t in ("road", "agent"):
        ct = cell_counts(CN, XN, terms=(t,))
        ct["bin"] = cN["bin"].to_numpy()
        J[f"N_bins_{t}"], J[f"N_D_{t}"], _ = bin_stats(ct, "bin", names)
    J["N_pool"] = {}
    for p in sorted(cN.pool.unique()):
        J["N_pool"][p] = bin_stats(cN[cN.pool == p], "bin", names, B=2000)[:2]
    J["N_split"] = {sp: bin_stats(cN[cN.held == (sp == "held")], "bin", names, B=2000)[:2] for sp in ("train", "held")}
    # waste decomposition per bin (hinge-active rows)
    J["N_waste"] = {b: dict(active=int(g.act.sum()), after=int(g.w_term.sum()), within_T=int(g.w_near.sum()), off_LH=int(g.w_off.sum()),
                            usable=int(g.use.sum())) for b, g in cN.groupby("bin")}
    J["N_cells_no_active"] = {b: int((g.act == 0).sum()) for b, g in cN.groupby("bin")}
    J["N_cells_zero"] = {b: dict(cells=int(len(g)), zero=int((g.kz >= 0).sum())) for b, g in cN.groupby("bin")}
    # regression
    J["N_glm_pooled"] = glm_boot(cN)
    J["N_glm_fe"] = glm_boot(cN, fe=True)
    # floor
    pooled = J["N_completion_pooled"]
    fl = [b for b in J["N_bins"] if b["cells"] and b["completion_ci"][1] >= pooled]
    J["floor"] = dict(pooled_completion=pooled, lowest_ok_bin=fl[0]["bin"] if fl else None)
    fla = [b for b in J["N_abs"] if b["cells"] and b["completion_ci"][1] >= pooled]
    J["floor_abs"] = fla[0]["bin"] if fla else None
    # P3
    cP["bin"], _ = binize(cP.v_ratio.to_numpy(float), edges)
    J["P_bins"], J["P_D"], J["P_completion_pooled"] = bin_stats(cP, "bin", names, B=2000)
    J["P_arms"] = {k: dict(cells=int(len(g)), active=int(g.act.sum()), usable=int(g.use.sum()),
                           share=float(g.use.sum() / g.act.sum()) if g.act.sum() else np.nan, count=float(g.use.mean()),
                           completion=float(g.completed.mean()), v_entry=float(g.v_entry.median()), v_turn=float(g.v_turn.median()),
                           rear=int(g.rear.sum()), zero=int(np.isfinite(g.t_zero).sum()), after=int(g.w_term.sum()), within_T=int(g.w_near.sum()),
                           off_LH=int(g.w_off.sum()))
                 for k, g in cP.groupby(["ho", "g", "s"]) for k in [f"{k[0]}@g{k[1]:g}s{k[2]:g}"]}
    J["P_contrast"], pairs = p3_contrast(cP)
    J["P_contrast_by_s"] = {s: p3_contrast(cP, s_set=(s,))[0] for s in (0.8, 0.6)}
    J["P_contrast_by_ho"] = {ho: p3_contrast(cP[cP.ho == ho])[0] for ho in ("d10", "d0")}
    # sensitivity
    sens = []
    for T, L, H in GRID:
        c1 = cell_counts(CN, XN, T, L, H)
        c1["bin"] = cN["bin"].to_numpy()
        b1, D1, _ = bin_stats(c1, "bin", names, B=2000)
        p1, _ = p3_contrast(cell_counts(CP, XP, T, L, H, terms=("route",)).assign(), B=2000)
        sens.append(dict(T=T, L=L, H=H, low=b1[0]["share"], high=b1[-1]["share"], D=D1["share"], D_ci=D1["share_ci"],
                         P=p1.get("d", np.nan), P_ci=p1.get("d_ci", (np.nan, np.nan))))
    J["sensitivity"] = sens
    # verdict
    D = J["N_D"]
    cap = D["share"] >= 0.20 and D["share_ci"][0] > 0 and J["P_contrast"].get("d", np.nan) > 0
    against = []
    if D["count_ci"][1] < 0:
        against.append("lowest bin has fewer usable rows per cell than the highest (CI below 0)")
    if D["completion_ci"][1] < 0:
        against.append("lowest bin completes fewer turns (CI below 0)")
    pc = J["P_contrast"]
    if pc.get("d_act_ci", (0, 0))[1] < 0 or pc.get("d_use_ci", (0, 0))[1] < 0:
        against.append("slow P3 cells lose hinge-active / usable rows (CI below 0)")
    J["verdict"] = dict(rule="speed cap worth building" if cap else "flat", against=against)
    cN.to_csv(out / "cells_nup.csv", index=False, float_format="%.4f")
    cP.to_csv(out / "cells_p3.csv", index=False, float_format="%.4f")
    pd.DataFrame(pairs).to_csv(out / "p3_pairs.csv", index=False, float_format="%.4f")
    (out / "entry_speed.json").write_text(json.dumps(J, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
    tables(J, out / "tables.md", names, anames)
    fig(cN, cP, J, names, figs)
    print(json.dumps(dict(inputs=J["inputs"], bins=J["bins"], N_bins=J["N_bins"], N_D=J["N_D"], P_contrast=J["P_contrast"], verdict=J["verdict"],
                          glm=(J["N_glm_pooled"], J["N_glm_fe"]), floor=J["floor"]), indent=1, default=str))


def tables(J, path, names, anames):
    L = ["# ENTRY-SPEED ROWS tables (rf_entryspeed.py report)", "",
         f"Primary T {T0} s, L {L0} m, H {H0} deg. share = usable / hinge-active rows (ratio of sums), count = usable rows per cell, CIs: "
         "scene-cluster bootstrap 95 %. D = lowest bin - highest bin (one draw of scenes).", ""]

    def bt(title, rows, D=None):
        out = [f"## {title}", "", "| bin | cells | scenes | hinge-active rows | usable | share | usable / cell | completion |", "|:--|--:|--:|--:|--:|:--|:--|:--|"]
        for b in rows:
            out.append(f"| {b['bin']} | {b['cells']} | {b['scenes']} | {b['active']} | {b['usable']} | {fmt(b['share'], b['share_ci'])} | "
                       f"{fmt(b['count'], b['count_ci'], 2)} | {fmt(b['completion'], b['completion_ci'])} |")
        if D:
            out.append(f"| **D (low - high)** | | | | | {fmt(D['share'], D['share_ci'])} | {fmt(D['count'], D['count_ci'], 2)} | "
                       f"{fmt(D['completion'], D['completion_ci'])} |")
        return out + [""]
    L += bt("N, v_entry / v_human bins, any hinge (primary)", J["N_bins"], J["N_D"])
    L += bt("N, route hinge only", J["N_bins_route"], J["N_D_route"])
    L += bt("N, road hinge only", J["N_bins_road"], J["N_D_road"])
    L += bt("N, agent hinge only", J["N_bins_agent"], J["N_D_agent"])
    L += bt("N, absolute v_entry (m/s)", J["N_abs"], J["N_abs_D"])
    for p, (b, D) in J["N_pool"].items():
        L += bt(f"N, pool {p}", b, D)
    for sp, (b, D) in J["N_split"].items():
        L += bt(f"N, {sp} scenes", b, D)
    L += ["## N, waste decomposition of hinge-active rows (primary labels)", "", "| bin | active | after termination | within T before | off L / H only | usable | cells with no active row | cells with a zero |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for b in names:
        w = J["N_waste"].get(b)
        if w:
            L.append(f"| {b} | {w['active']} | {w['after']} | {w['within_T']} | {w['off_LH']} | {w['usable']} | {J['N_cells_no_active'][b]} | "
                     f"{J['N_cells_zero'][b]['zero']} / {J['N_cells_zero'][b]['cells']} |")
    L += ["", "## P3 per arm (route hinge only)", "", "| arm | cells | active | usable | share | usable / cell | completion | v_entry med | v_turn med | zero / rear | after / within T / off |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, v in J["P_arms"].items():
        L.append(f"| {k} | {v['cells']} | {v['active']} | {v['usable']} | {fmt(v['share'])} | {v['count']:.1f} | {v['completion']:.3f} | {v['v_entry']:.2f} | "
                 f"{v['v_turn']:.2f} | {v['zero']} / {v['rear']} | {v['after']} / {v['within_T']} / {v['off_LH']} |")
    L += [""] + bt("P3 cells binned by v_entry / v_human (secondary)", J["P_bins"], J["P_D"])
    pc = J["P_contrast"]
    L += ["## P3 within-scene contrast, s < 1 minus s = 1 (same scene, handover, g)", "", "| set | pairs | scenes | d share | d usable rows | d hinge-active rows | d v_turn (m/s) | d completion |",
          "|:--|--:|--:|:--|:--|:--|:--|:--|"]
    for nm, r in [("pooled", pc)] + [(f"s {s}", v) for s, v in J["P_contrast_by_s"].items()] + [(f"handover {h}", v) for h, v in J["P_contrast_by_ho"].items()]:
        if r.get("n"):
            L.append(f"| {nm} | {r['n']} | {r['scenes']} | {fmt(r['d'], r['d_ci'])} | {fmt(r['d_use'], r['d_use_ci'], 1)} | {fmt(r['d_act'], r['d_act_ci'], 1)} | "
                     f"{fmt(r['d_vturn'], r['d_vturn_ci'], 2)} | {fmt(r['d_comp'], r['d_comp_ci'])} |")
    g1, g2 = J["N_glm_pooled"], J["N_glm_fe"]
    L += ["", "## N, binomial GLM of the usable share on v_entry / v_human (logit slope per unit ratio)", "",
          f"- pooled: {fmt(g1['slope'], (g1['lo'], g1['hi']))} ({g1['cells']} cells, {g1['scenes']} scenes)",
          f"- scene fixed effects (within-scene): {fmt(g2['slope'], (g2['lo'], g2['hi']))}", "",
          "## Sensitivity (N: D low - high, any hinge; P3: pooled within-scene d share, route hinge)", "",
          "| T (s) | L (m) | H (deg) | N low | N high | N D | P3 d |", "|--:|--:|--:|--:|--:|:--|:--|"]
    for r in J["sensitivity"]:
        L.append(f"| {r['T']:g} | {r['L']:g} | {r['H']:g} | {fmt(r['low'])} | {fmt(r['high'])} | {fmt(r['D'], r['D_ci'])} | {fmt(r['P'], r['P_ci'])} |")
    Path(path).write_text("\n".join(L) + "\n")


def fig(cN, cP, J, names, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(names))
    for a_, key, lab in ((ax[0], "share", "usable share of hinge-active rows"), (ax[1], "count", "usable rows per cell"),
                         (ax[2], "completion", "turn completion")):
        for rows, off, col, nm in ((J["N_bins"], -0.12, "#2a6fdb", "N any hinge"), (J["N_bins_route"], 0.12, "#e07b39", "N route hinge")):
            if key == "completion" and nm == "N route hinge":
                continue
            v = np.array([r[key] for r in rows])
            ci = np.array([r[f"{key}_ci"] for r in rows])
            a_.errorbar(x + off, v, yerr=np.abs(ci.T - v), fmt="o", color=col, capsize=3, label=nm)
        a_.set_xticks(x, names)
        a_.set_xlabel("v_entry / v_human")
        a_.set_title(lab, fontsize=10)
        a_.grid(alpha=0.3)
    for i, r in enumerate(J["N_bins"]):
        ax[0].annotate(f"{r['cells']} cells", (i, 0.02), ha="center", fontsize=7, color="0.4")
    ax[0].set_ylim(0, 1)
    ax[2].set_ylim(0, 1)
    ax[0].legend(fontsize=8)
    fig.suptitle("nuPlan @18 turn cells (6 row pools), bins of entry speed over the human's; 95 % scene-cluster CI", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "entry_speed_rows_bins.png", dpi=100)
    plt.close(fig)
    # P3: per scene, s = 1 vs s < 1 shares (same handover, g), and the N cell scatter
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    c = cP.set_index(["scene", "ho", "g", "s"])
    for (sc, ho, g), _ in cP[cP.g.isin([1.0, 1.5])].groupby(["scene", "ho", "g"]):
        try:
            b = c.loc[(sc, ho, g, 1.0)]
        except KeyError:
            continue
        for s, col in ((0.8, "#7b4fd6"), (0.6, "#d64f8a")):
            if (sc, ho, g, s) in c.index:
                y = c.loc[(sc, ho, g, s)]
                if np.isfinite(b.share) and np.isfinite(y.share):
                    ax[0].plot([0, 1], [b.share, y.share], "-o", color=col, alpha=0.5, ms=3)
    ax[0].plot([], [], "-o", color="#7b4fd6", label="s 0.8")
    ax[0].plot([], [], "-o", color="#d64f8a", label="s 0.6")
    ax[0].set_xticks([0, 1], ["s = 1 (raw speed)", "s < 1 (slowed)"])
    ax[0].set_ylabel("usable share of route-hinge-active rows")
    ax[0].set_title(f"P3: same scene / handover / g, d = {J['P_contrast'].get('d', np.nan):+.3f}", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    m = np.isfinite(cN.v_ratio) & np.isfinite(cN.share)
    for p, col in zip(sorted(cN.pool.unique()), plt.cm.tab10.colors):
        q = m & (cN.pool == p)
        ax[1].scatter(cN.v_ratio[q], cN.share[q], s=10 + cN.act[q], color=col, alpha=0.6, label=p, edgecolor="none")
    for e in J["bins"]["edges"]:
        ax[1].axvline(e, color="0.6", lw=0.8, ls=":")
    ax[1].set_xlabel("v_entry / v_human")
    ax[1].set_ylabel("usable share (cell)")
    ax[1].set_title("nuPlan cells (marker size = hinge-active rows)", fontsize=10)
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "entry_speed_rows_cells.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract-nup")
    e.add_argument("--pool", nargs="+", required=True, help="POOL=LABEL=<rf_follow dir>")
    e.add_argument("--out", required=True)
    e = sp.add_parser("extract-p3")
    e.add_argument("--arms", required=True)
    e.add_argument("--out", required=True)
    r = sp.add_parser("report")
    r.add_argument("--nup", required=True)
    r.add_argument("--p3", required=True)
    r.add_argument("--read", required=True)
    r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--figs", required=True)
    a = ap.parse_args()
    dict(**{"extract-nup": extract_nup, "extract-p3": extract_p3, "report": report})[a.cmd](a)
