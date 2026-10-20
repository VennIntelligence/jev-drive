#!/usr/bin/env python3
"""LEAD-BREAK (plans/2026-10-20-lead-break-prereg.md, results/lead_break/): why does RB0-F-s0's own plan not brake for a closing lead where
shipped Cinque (P0) does? Open-loop arms on RB0-F's own PAI states (the 50 AF-front zero scenes of its native PAI 281 read, rows rebuilt from
the stored rollout.asl by onp_rows.py) and the navtrain content under the same lead definition. No training, no rollout. Box, repo root.

  $TR lead_break.py select     the 50 scenes -> $O/scenes.txt (lineage_pai281.json, RB0-NAT zeros of class AF-front*)
  $TR lead_break.py cf         (GPU pool job) P0 / RB0-F-s0 / NTV0-L18-s0 x arms on the BRK / FOL decisions -> $O/cf.parquet, $O/rows.parquet, gate.json
  $TR lead_break.py nav        (CPU) navtrain_full rows under the same BRK definition -> $O/nav.parquet
  $TR lead_break.py report     (CPU) the registered tables and verdicts -> results/lead_break/report.{md,json}

Arms (prereg): F as fed; OFF present = 0 (the anchor / cons condition: adapter and route bias x 0); ZR route memory zeroed; ZE the 16 ego-state
entries zeroed; AXN fed ax = -2 m/s^2; KW the trained slot protocol (pai_core.key_slots: 2 Hz keys + ego-motion warp), row as fed; KWOFF KW + OFF.
P0 reads no row: F and KW only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/alpasim/lib"), str(REPO / "lib"), str(REPO / "experiments/route_follow/scripts")]
import inertia as IN  # noqa: E402

DATA = IN.DATA
O = DATA / "runs/defect_fix/lead_break"
RES = REPO / "experiments/defect_fix/results/lead_break"
LIN = DATA / "runs/defect_fix/lineage"
ROWS = O / "rows/RB0"
TAGS = ("P0", "RB0-F-s0", "NTV0-L18-s0")
ARMS = ("F", "OFF", "ZR", "ZE", "AXN", "KW", "KWOFF")
P0_ARMS = ("F", "KW")
LAT, GAP_MAX, CLOSE_MIN, NAV_FRONT = 1.75, 80.0, 0.5, 4.049    # prereg; NAV_FRONT = NAVSIM Pacifica rear axle -> front bumper
TK = 0.5 * np.arange(9)                                          # box times t0 + 0 .. 4 s


# ---------------------------------------------------------------- geometry (shared by PAI rows and navtrain)
def project(pts, path):
    """pts (m, 2), path (n, 2) polyline -> arc along the path from its first point (m,), absolute lateral distance (m,)."""
    A, B = path[:-1], path[1:]
    d = B - A
    L2 = np.maximum((d * d).sum(1), 1e-9)
    t = np.clip(((pts[:, None] - A[None]) * d[None]).sum(-1) / L2[None], 0.0, 1.0)
    q = A[None] + t[..., None] * d[None]
    dist = np.hypot(*(pts[:, None] - q).transpose(2, 0, 1))
    j = dist.argmin(1)
    c = np.r_[0.0, np.cumsum(np.sqrt(L2))]
    r = np.arange(len(pts))
    return c[j] + t[r, j] * np.sqrt(L2[j]), dist[r, j]


def lead_of(box, ok, path, front, v0):
    """box (9, K, 5) x y yaw length width (centre, the t0 frame), ok (9, K), path (n, 2) through the ego, front (rear axle -> front bumper),
    v0 -> None, or dict(gap, v_lead, closing, ttc, tail (9,) lead-tail arc rel. the ego's rear axle (nan where absent), cvhit, s0)."""
    s0, _ = project(np.zeros((1, 2)), path)
    s0 = float(s0[0])
    c = np.flatnonzero(ok[0])
    if not len(c):
        return None
    a, lat = project(box[0, c, :2].astype(np.float64), path)
    a = a - s0
    tail0 = a - box[0, c, 3] / 2
    m = (lat < LAT) & (tail0 - front > -1.0) & (a > 0)
    if not m.any():
        return None
    j = c[np.flatnonzero(m)[np.argmin(a[m])]]
    tail = np.full(9, np.nan)
    for k in range(9):
        if ok[k, j]:
            tail[k] = project(box[k, j, None, :2].astype(np.float64), path)[0][0] - s0 - box[k, j, 3] / 2
    k1 = 2 if np.isfinite(tail[2]) else 1
    if not np.isfinite(tail[k1]):
        return None
    vl = (tail[k1] - tail[0]) / TK[k1]
    gap = tail[0] - front
    cl = v0 - vl
    cv = v0 * TK[1:] + front
    hit = bool(np.any(np.isfinite(tail[1:]) & (cv > tail[1:])))
    return dict(gap=float(gap), v_lead=float(vl), closing=float(cl), ttc=float(min(gap / cl, 10.0)) if cl > 0 else 10.0, tail=tail, cvhit=hit, s0=s0)


def plan_margin(P, path, s0, tail, front):
    """plan (8, 3) t0 frame -> margin = min_k (lead tail - plan front) over the 0.5 .. 4 s points with a lead pose."""
    sp, _ = project(np.asarray(P, np.float64)[:, :2], path)
    d = tail[1:] - (sp - s0 + front)
    return float(np.nanmin(d)) if np.isfinite(d).any() else np.nan


def window(L, v0):
    if L is None or not (0 < L["gap"] <= GAP_MAX):
        return ""
    if L["closing"] >= CLOSE_MIN and L["cvhit"]:
        return "BRK"
    if 5 <= L["gap"] <= 40 and abs(L["closing"]) < 1.0 and not L["cvhit"]:
        return "FOL"
    return ""


# ---------------------------------------------------------------- select
def cmd_select(a):
    J = json.load(open(LIN / "lineage_pai281.json"))["RB0-NAT"]
    sc = sorted(s for s, t in J["types"].items() if J["scores"][s] == 0 and t.startswith("AF-front"))
    O.mkdir(parents=True, exist_ok=True)
    (O / "scenes.txt").write_text("".join(f"{s}\n" for s in sc))
    print(len(sc), "scenes")


# ---------------------------------------------------------------- cf
def _scene_rows(job):
    f, every = job
    z = np.load(f, allow_pickle=True)
    ev = int(np.min(z["t_event_us"]))
    vs = z["veh_sim"].astype(np.float64)
    front = float(vs[2] + vs[0] / 2)
    out = []
    for n, k in enumerate(z["k"]):
        v0 = float(z["v0"][n])
        t0 = int(z["t0_us"][n])
        if n % every or t0 >= ev or z["after_end"][n] or (z["slot"][n] < 0).any() or not z["valid"][n].all() or v0 < 2.0 \
                or not abs(float(np.nan_to_num(z["e0"][n], nan=99))) <= 1.5:
            continue
        pok = z["path_ok"][n]
        path = z["path"][n][pok].astype(np.float64)
        if len(path) < 30:
            continue
        L = lead_of(z["box"][n].astype(np.float64), z["box_ok"][n], path, front, v0)
        w = window(L, v0)
        if not w:
            continue
        out.append(dict(scene=str(z["scene"]), session=str(z["session"]), run=str(z["run"]), k=int(k), t0=t0, t_ev=ev, v0=v0, lht=bool(z["lht"]),
                        cam_t=z["cam_t"].astype(np.float64), ego=z["ego_in"][n].astype(np.float32), P=z["P"][n], V=z["V"][n],
                        poses_log=z["poses_model"][n].astype(np.float32), sl=z["slot"][n].astype(int), path=path, front=front, win=w,
                        ax=float(z["ego_in"][n][6] * 3.0), file=f, **{x: L[x] for x in ("gap", "v_lead", "closing", "ttc", "s0")}, tail=L["tail"]))
    return out


def cmd_cf(a):
    import pandas as pd
    import torch
    import pai_core as PC
    import sh30_core as SC
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    from jevdrive.run import Run
    O.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", "lead-break-cf", config=vars(a)) as run:
        files = sorted(glob.glob(str(ROWS / "*.npz")))[:a.limit or None]
        with ProcessPoolExecutor(a.workers) as ex:
            R = [r for out in ex.map(_scene_rows, [(f, a.every) for f in files], chunksize=1) for r in out]
        rm = IN.rmem_of(sorted({r["run"] for r in R}))
        R = [r for r in R if (r["session"], r["k"]) in rm]
        run.info(f"{len(R)} rows in {len({r['scene'] for r in R})} scenes: " + str(pd.Series([r["win"] for r in R]).value_counts().to_dict()))
        cores = {t: SC.Core(t, "cuda", "backwarp") for t in TAGS}
        enc = cores["RB0-F-s0"]
        net, A, dev = enc.model.net, enc.A, enc.dev

        def encode(cur):
            cur = cur if torch.is_tensor(cur) else torch.from_numpy(cur).to(dev)
            prev = torch.cat([torch.zeros_like(cur[:1]), cur[:-1]])
            with torch.no_grad():
                return net.run_batched(A.vision_feeds(prev, cur), ["view_39"])["view_39"].reshape(8, *A.H_SHAPE)

        recs, meta = [], []
        by = defaultdict(list)
        for i, r in enumerate(R):
            by[r["file"]].append(i)
        for nf, (f, idx) in enumerate(sorted(by.items())):
            zz = np.load(f, allow_pickle=True)
            fr, fts = zz["frames"], zz["frame_ts"].astype(np.int64)
            fd = {int(t): fr[j] for j, t in enumerate(fts)}
            for i in idx:
                r = R[i]
                Hf = encode(fr[r["sl"]])
                cur, kval, _ = PC.key_slots(fd, r["t0"], r["P"], r["V"], r["cam_t"], "backwarp", dev)
                Hk = encode(cur) if bool(np.asarray(kval).all()) else None
                e, mem = r["ego"], rm[(r["session"], r["k"])]
                off = e.copy(); off[0] = 0.0
                ze = e.copy(); ze[4:20] = 0.0
                axn = e.copy(); axn[6] = -2.0 / 3.0
                feeds = dict(F=("f", e, mem), OFF=("f", off, mem), ZR=("f", e, np.zeros_like(mem)), ZE=("f", ze, mem), AXN=("f", axn, mem))
                if Hk is not None:
                    feeds |= dict(KW=("k", e, mem), KWOFF=("k", off, mem))
                tc1 = [[0.0, 1.0] if r["lht"] else [1.0, 0.0]]
                gate = None
                for t, core in cores.items():
                    names = [n for n in feeds if t != "P0" or n in P0_ARMS]
                    H = torch.stack([Hf if feeds[n][0] == "f" else Hk for n in names])
                    row = np.stack([np.r_[feeds[n][1], feeds[n][2]] if core.rm else feeds[n][1] for n in names]).astype(np.float32)
                    with torch.no_grad():
                        o = core.model(H, torch.from_numpy(row).to(dev), torch.tensor(tc1 * len(names), device=dev))
                    mu = (o[0] if isinstance(o, tuple) else o).float()[:, core.pi].reshape(len(names), 33, 15).cpu().numpy()
                    for n, m in zip(names, mu):
                        p = np.asarray(I.to_rear(m[:, 0:3], m[:, 11], I.T_IDXS, r["cam_t"][:2], Z.T_OUT, "lever")[:, :3], np.float64)
                        if t == "RB0-F-s0" and n == "F":
                            gate = (float(np.abs(p[:, :2] - r["poses_log"][:, :2]).max()), IN.gate_ok(p[:, :2], r["poses_log"][:, :2]))
                        recs.append(dict(i=i, tag=t, arm=n, P=p.astype(np.float32).ravel(),
                                         margin=plan_margin(p, r["path"], r["s0"], r["tail"], r["front"])))
                meta.append({k: r[k] for k in ("scene", "k", "t0", "t_ev", "v0", "ax", "win", "gap", "v_lead", "closing", "ttc")}
                            | dict(i=i, kw_ok=Hk is not None, gate=gate[0], gate2=gate[1]))
            if nf % 10 == 0:
                run.info(f"{nf + 1} / {len(by)} scenes, {len(meta)} rows")
        M = pd.DataFrame(meta)
        C = pd.DataFrame(recs)
        Pm = np.stack(C.P.values).reshape(-1, 8, 3)
        for k, v in IN.metrics(Pm).items():
            C[k] = v
        C["a02"] = (np.hypot(Pm[:, 3, 0], Pm[:, 3, 1]) - 2.0 * M.set_index("i").v0.reindex(C.i).values) / 2.0   # 2 (s(2 s) - 2 v0) / 2^2
        C.drop(columns="P").to_parquet(O / "cf.parquet")
        np.save(O / "cf_poses.npy", Pm.astype(np.float32))
        M.to_parquet(O / "rows.parquet")
        g = dict(rows=len(M), gate_p99=float(M.gate.quantile(0.99)), gate_max=float(M.gate.max()), gate2_share=float(M.gate2.mean()),
                 pass_=bool(M.gate2.mean() >= 0.99 and M.gate.max() <= 0.15), kw_ok=float(M.kw_ok.mean()))
        json.dump(g, open(O / "gate.json", "w"), indent=1)
        run.summary.update(g)
        run.info(f"gate {g}")


# ---------------------------------------------------------------- navtrain
def _nav_shard(d):
    z = np.load(DATA / "runs/op_parity/cache" / d / "tab.npz")
    AL = np.load(DATA / "runs/op_parity/agent_labels/navtrain_all-k32.npz")
    ix = {t: i for i, t in enumerate(AL["tokens"])}
    names = set(map(str, z["names"]))
    keep = np.array(sorted(i for t, i in ix.items() if t in names), np.int64)          # load only this shard's label rows, once
    ABOX, AVAL, ACLS, AOK = (AL[k][keep] for k in ("box", "valid", "cls", "ok"))
    ix = {str(AL["tokens"][i]): n for n, i in enumerate(keep)}
    out = []
    for n, tok in enumerate(z["names"]):
        v0 = float(z["speed"][n])
        j = ix.get(str(tok))
        fut = z["fut"][n].astype(np.float64)
        if v0 < 2.0 or j is None or not AOK[j] or np.isnan(fut).any():
            continue
        yaw = fut[-1, 2]
        ext = fut[-1, :2] + np.arange(1, 81)[:, None] * np.array([np.cos(yaw), np.sin(yaw)])
        path = np.r_[np.zeros((1, 2)), fut[:, :2], ext]
        box, ok = ABOX[j].astype(np.float64), AVAL[j] & (ACLS[j] >= 0)[None]
        L = lead_of(box, ok, path, NAV_FRONT, v0)
        if L is None or not (0 < L["gap"] <= GAP_MAX):
            continue
        hm = plan_margin(np.c_[fut[:, :2], fut[:, 2]], path, L["s0"], L["tail"], NAV_FRONT)
        s2 = float(np.hypot(*np.diff(np.r_[np.zeros((1, 2)), fut[:4, :2]], axis=0).T).sum())
        out.append(dict(token=str(tok), log=str(z["log"][n]), v0=v0, ax=float(z["ego"][n][6] * 3.0), gap=L["gap"], v_lead=L["v_lead"],
                        closing=L["closing"], ttc=L["ttc"], cvhit=L["cvhit"], win=window(L, v0), h_margin=hm, h_a02=(s2 - 2.0 * v0) / 2.0))
    return out


def cmd_nav(a):
    import pandas as pd
    ds = [f"navtrain_full.s{i}of12" for i in range(12)]
    with ProcessPoolExecutor(a.workers) as ex:
        R = [r for out in ex.map(_nav_shard, ds) for r in out]
    D = pd.DataFrame(R)
    O.mkdir(parents=True, exist_ok=True)
    D.to_parquet(O / "nav.parquet")
    tot = sum(len(np.load(DATA / "runs/op_parity/cache" / d / "tab.npz")["names"]) for d in ds)
    json.dump(dict(rows_total=int(tot), rows_lead=len(D), **D.win.value_counts().to_dict()), open(O / "nav.json", "w"), indent=1)
    print(tot, len(D), D.win.value_counts().to_dict())


# ---------------------------------------------------------------- report
def cmd_report(a):
    import pandas as pd
    g = json.load(open(O / "gate.json"))
    M = pd.read_parquet(O / "rows.parquet").set_index("i")
    C = pd.read_parquet(O / "cf.parquet")
    C["clear"] = (C.margin >= 0).astype(float)
    C.loc[~np.isfinite(C.margin), "clear"] = np.nan
    W = {c: C.pivot_table(index="i", columns=["tag", "arm"], values=c, dropna=False) for c in ("clear", "a02", "A4", "margin")}
    out, L = dict(gate=g), []
    f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"  # noqa: E731
    p3 = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
    L.append(f"G0 (RB0-F-s0 F against the logged plan): p99 {g['gate_p99']:.4f} m, max {g['gate_max']:.4f} m, within 2 fp16 steps "
             f"{g['gate2_share']:.3f}, rows {g['rows']}, KW usable {g['kw_ok']:.2f} -> {'PASS' if g['pass_'] else 'FAIL'}\n")
    cols = [(t, x) for t in TAGS for x in (P0_ARMS if t == "P0" else ARMS)]
    for win in ("BRK", "FOL"):
        m = M[M.win == win]
        if not len(m):
            continue
        ids, sc = m.index.values, m.scene.values
        X = {c: W[c].reindex(ids) for c in W}
        kw = m.kw_ok.values
        cell = dict(rows=len(m), scenes=int(m.scene.nunique()), v0=float(m.v0.median()), gap=float(m.gap.median()), closing=float(m.closing.median()),
                    ttc=float(m.ttc.median()), ax=float(m.ax.median()), arms={})
        L += [f"## {win}: {len(m)} decisions in {cell['scenes']} scenes (median v0 {cell['v0']:.1f} m/s, gap {cell['gap']:.1f} m, closing "
              f"{cell['closing']:.1f} m/s, TTC {cell['ttc']:.1f} s, fed ax {cell['ax']:+.2f} m/s^2)\n",
              "| model | arm | clear | a02 (m/s^2) | 4 s arc (m) | margin median (m) |", "|---|---|--:|--:|--:|--:|"]
        for t, x in cols:
            if (t, x) not in X["clear"].columns:
                continue
            msk = kw if x.startswith("KW") else np.ones(len(m), bool)
            cl, a2, a4, mg = (X[c][(t, x)].values for c in ("clear", "a02", "A4", "margin"))
            mk = msk & np.isfinite(cl)
            r = dict(clear=IN.boot(lambda i: float(np.nanmean(cl[i][mk[i]])), sc), a02=float(np.nanmean(a2[mk])), A4=float(np.nanmean(a4[mk])),
                     margin_med=float(np.nanmedian(mg[mk])))
            cell["arms"][f"{t}/{x}"] = r
            L.append(f"| {t} | {x} | {p3(r['clear'])} | {r['a02']:+.2f} | {r['A4']:.1f} | {r['margin_med']:+.1f} |")
        # paired differences (same decisions; KW-arm differences on the KW-usable rows)
        D = {}

        def dif(t1, x1, t0_, x0, c="clear"):
            u, v = X[c][(t1, x1)].values, X[c][(t0_, x0)].values
            mk = np.isfinite(u) & np.isfinite(v) & (kw if (x1.startswith("KW") or x0.startswith("KW")) else True)
            return u, v, mk

        def pair(t1, x1, t0_, x0, c="clear"):
            u, v, mk = dif(t1, x1, t0_, x0, c)
            return IN.boot(lambda i: float(np.mean(u[i][mk[i]] - v[i][mk[i]])), sc)

        def ratio(t1, x1, t0_, x0):
            u, v, mk = dif(t1, x1, t0_, x0)
            p, q = X["clear"][("P0", "F")].values, X["clear"][("RB0-F-s0", "F")].values
            mk = mk & np.isfinite(p) & np.isfinite(q)
            return IN.boot(lambda i: float(np.mean(u[i][mk[i]] - v[i][mk[i]]) / max(np.mean(p[i][mk[i]] - q[i][mk[i]]), 1e-6)), sc)
        D["gap P0 F - RB0 F"] = pair("P0", "F", "RB0-F-s0", "F")
        D["NTV0 F - RB0 F"] = pair("NTV0-L18-s0", "F", "RB0-F-s0", "F")
        for x in ("OFF", "ZR", "ZE", "AXN", "KW", "KWOFF"):
            D[f"RB0 {x} - RB0 F"] = pair("RB0-F-s0", x, "RB0-F-s0", "F")
            D[f"R RB0 {x}"] = ratio("RB0-F-s0", x, "RB0-F-s0", "F")
        D["P0 KW - P0 F"] = pair("P0", "KW", "P0", "F")
        D["R P0 KW"] = ratio("P0", "KW", "P0", "F")
        D["RB0 OFF - P0 F"] = pair("RB0-F-s0", "OFF", "P0", "F")
        D["RB0 KWOFF - P0 KW"] = pair("RB0-F-s0", "KWOFF", "P0", "KW")
        D["NTV0 OFF - NTV0 F"] = pair("NTV0-L18-s0", "OFF", "NTV0-L18-s0", "F")
        D["a02: RB0 OFF - RB0 F"] = pair("RB0-F-s0", "OFF", "RB0-F-s0", "F", "a02")
        D["a02: P0 F - RB0 F"] = pair("P0", "F", "RB0-F-s0", "F", "a02")
        cell["diff"] = D
        L += ["", "| paired (same decisions) | value, 95 % scene-cluster CI |", "|---|--:|"] + [f"| {k} | {f3(v)} |" for k, v in D.items()] + [""]
        out[win] = cell
    # verdicts (prereg)
    b = out.get("BRK")
    V = {}
    if b:
        gp = b["diff"]["gap P0 F - RB0 F"]
        V["G1"] = bool(gp[0] >= 0.15 and gp[1] > 0)
        cut = lambda r, lo: "supported" if r[0] >= 0.5 and lo > 0 else "partial" if r[0] >= 0.2 else "not supported"  # noqa: E731
        V["b_adapter_override"] = cut(b["diff"]["R RB0 OFF"], b["diff"]["RB0 OFF - RB0 F"][1])
        V["a1_slot_mismatch"] = cut(b["diff"]["R RB0 KW"], b["diff"]["RB0 KW - RB0 F"][1])
        r = b["diff"]["R P0 KW"][0]
        V["a2_cue_lost_in_warp"] = "cue lost" if r <= -0.5 else "cue kept" if r > -0.2 else "partial"
    # (c) navtrain
    nf = O / "nav.parquet"
    if nf.exists() and b:
        N = pd.read_parquet(nf)
        nj = json.load(open(O / "nav.json"))
        brk = N[N.win == "BRK"]
        init = float((brk.ax > -0.5).mean())
        clos = N[N.closing >= CLOSE_MIN]
        m = M[M.win == "BRK"]
        cnt, brate = [], []
        cv, cc, ct = clos.v0.values, clos.closing.values, clos.ttc.values
        hb = (clos.h_a02.values <= -1.0)
        for r_ in m.itertuples():
            k_ = (np.abs(cv - r_.v0) <= 2) & (np.abs(cc - r_.closing) <= 1.5) & (np.abs(ct - r_.ttc) <= 1.0)
            cnt.append(int(k_.sum()))
            brate.append(float(hb[k_].mean()) if k_.any() else np.nan)
        cnt = np.array(cnt)
        bi = brk[brk.ax > -0.5]
        nav = dict(rows_total=nj["rows_total"], rows_lead=nj["rows_lead"], brk=len(brk), brk_share=len(brk) / nj["rows_total"],
                   brk_logs=int(brk.log.nunique()), init_share=init, init_rows=len(bi), brk_h_clear=float((brk.h_margin >= 0).mean()),
                   brk_h_a02=float(brk.h_a02.median()), init_h_a02=float(bi.h_a02.median()) if len(bi) else np.nan,
                   brk_ax_med=float(brk.ax.median()), pai_brk_ax_med=float(m.ax.median()), pai_brk_v0=float(m.v0.median()), nav_brk_v0=float(brk.v0.median()),
                   pai_brk_closing=float(m.closing.median()), nav_brk_closing=float(brk.closing.median()), pai_brk_ttc=float(m.ttc.median()),
                   nav_brk_ttc=float(brk.ttc.median()), neigh_median=float(np.median(cnt)), neigh_p10=float(np.percentile(cnt, 10)),
                   neigh_zero_share=float((cnt == 0).mean()), neigh_brake_rate=float(np.nanmean(brate)),
                   init_rows_ax_gt0=int((brk.ax > 0).sum()))
        out["nav"] = nav
        V["c1_out_of_distribution"] = bool(nav["neigh_median"] < 20)
        V["c2_initiation_rare"] = bool(init < 0.25)
        L += ["## navtrain (same lead / BRK definition; path = the human's future + straight extension)\n"] + [f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in nav.items()] + [""]
    out["verdicts"] = V
    L += ["## Verdicts (prereg rules)\n"] + [f"- {k}: {v}" for k, v in V.items()]
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / "report.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    (RES / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    global O, TAGS
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("select")
    c = sp.add_parser("cf")
    c.add_argument("--every", type=int, default=2)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--tags", default=",".join(TAGS), help="models (must hold RB0-F-s0: it encodes the frames and carries G0)")
    c.add_argument("--name", default="", help="output subdir of $O (default: $O itself)")
    n = sp.add_parser("nav")
    n.add_argument("--workers", type=int, default=12)
    sp.add_parser("report")
    a = p.parse_args()
    if a.cmd == "cf" and (a.name or a.tags != ",".join(TAGS)):         # lon-anchor supplement 3: the pilot read on other checkpoints
        O, TAGS = O / a.name if a.name else O, tuple(a.tags.split(","))
        assert "RB0-F-s0" in TAGS, "--tags: RB0-F-s0 encodes the frames"
    {"select": cmd_select, "cf": cmd_cf, "nav": cmd_nav, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
