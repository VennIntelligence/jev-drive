#!/usr/bin/env python3
"""ROUTE-NIGHT G-swap (research/next-round/route-night.md, stage 3 "G-swap", line L2): does a ROUTE-MEM checkpoint's plan follow a different
route when its route input changes? Offline, no simulator. envs/op-train, one GPU (pool job: --vram 20 --cpu 4 --ram 40).

  rf_gates.py --tag RB-F-s0 [--pool on1_CFdev] [--n-st 1024] [--out JSON]
      -> $DATA_DIR/runs/route_follow/gates/<tag>.json and experiments/route_follow/results/gates_<tag>.json (box repo; commit from the Mac)
  rf_gates.py selftest                                                       the geometry on synthetic rows (CPU)

Model: rmem.load_pmodel(tag) (sidecar route_mem.json: attach the branch, load the whole state, as rm_train's --rm-from path); a tag without
the branch is refused unless --allow-plain (then a zero branch is attached: plans equal the plain model's; code-path smoke only, its numbers
are not results). Forward as rm_train.rows_ade: model(front, ego (20 ++ 200 route features), tc, None, None) -> plan rows A.plan_index ->
pp_train.rear on the 8 poses 0.5 .. 4 s (rear axle, left +).

Rows. Counterfactual (CF) rows = the pool `<pool>` (rf_cfpool.py pack; dev-split logs only, never trained), names on:cf:<token>:<j0|j1|lc>:
  real    the navtrain row of the token: its tab ego (real command) ++ its memory from runs/route_mem/feat/<shard>.npz (rm_build nav), both
          looked up by token through runs/route_follow/cf/rows/<shard>.npz (row, token, cls; also the real route polyline `real`, same
          arc grid as the label: -10 .. +90 m around the ego's projection every 2 m)
  cf      the pool row: tab ego (counterfactual command) ++ feat/<pool>.npz (counterfactual memory); label polyline = <pool>/route.npz poly
  memcf   diagnostic: the real ego row (real command) ++ the counterfactual memory (the branch's own share of the swap)
  The front tokens are the pool's <pool>@warp/front.npy row (the token's own) for every variant.
Quantities on CF rows with ok (route.npz ok, real memory ok, real polyline present):
  ratio   poses at 1.5 and 2.0 s: s_i = arc length of the REAL plan to pose i; d_i = C(s_i) - R(s_i) (counterfactual minus real polyline at that
          arc); move_i = (P_cf,i - P_real,i) . d_i / |d_i|; ratio = sum move_i / sum |d_i| (rows with mean |d_i| >= 0.5 m; 1 = the plan moved all
          the way to the counterfactual route, 0 = not at all)
  side    the cf plan's 4 s pose is nearer the counterfactual polyline than the real one (rows whose gap at the 4 s arc is >= 1 m)
Straight rows (G-ST) = dev rows of the 12 navtrain shards (feat dev flag, memory ok, no junction within 60 m, route turn over 40 m < 5 deg),
--n-st of them (rng 0): |lateral move| of the 1.5 / 2.0 s poses (mean of the two) when the real memory is replaced by the same route shifted
sideways by +0.5 / -0.5 m (rmem.transform_rows) and by no memory.
Line L2 (junction rows j0 / j1): median ratio >= 0.50, side share >= 0.70, G-ST: median move <= 0.10 m for the shifted and the masked memory.
CIs: log-clustered bootstrap (rf_cf.boot_med for medians, jevdrive.stats for shares).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/route_mem/lib"),
                str(REPO / "experiments/route_follow/scripts")]
import rmem as RM  # noqa: E402

TI = (2, 3)                                    # plan poses at 1.5 and 2.0 s
S_BACK, S_STEP = -10.0, 2.0                    # route.npz label grid (rf_cfpool.py arc_window)
MIN_GAP, MIN_GAP4, SHIFT = 0.5, 1.0, 0.5
HEAD = dict(d=128, layers=2, heads=4)


# ---------------------------------------------------------------- geometry (numpy, CPU)
def at_arc(Pl, s):
    """Label-grid polylines Pl (n, M, 2) (NaN outside) at arc s (n, k) -> (n, k, 2), NaN where s is outside the valid points."""
    n, M, _ = Pl.shape
    grid = S_BACK + S_STEP * np.arange(M)
    out = np.full(s.shape + (2,), np.nan)
    for i in range(n):
        ok = np.isfinite(Pl[i]).all(1)
        if ok.sum() < 2:
            continue
        g = grid[ok]
        inside = (s[i] >= g[0]) & (s[i] <= g[-1])
        for c in range(2):
            out[i, inside, c] = np.interp(s[i][inside], g, Pl[i][ok, c])
    return out


def plan_arc(P):
    """Plans (n, 8, >= 2) -> arc length from the rear axle to every pose (n, 8)."""
    Q = np.concatenate([np.zeros((len(P), 1, 2)), P[:, :, :2]], 1)
    return np.cumsum(np.hypot(*np.diff(Q, axis=1).transpose(2, 0, 1)), 1)


def pl_dist(q, M):
    """Point q (2,) to polyline M (m, 2) (NaN rows dropped)."""
    M = M[np.isfinite(M).all(1)]
    if len(M) < 2:
        return np.nan
    a, ab = M[:-1], np.diff(M, axis=0)
    t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return float(np.hypot(*(a + t[:, None] * ab - q).T).min())


def swap_read(Pr, Pc, C, Rr):
    """Real / cf plans (n, 8, 2), cf / real polylines (n, M, 2) -> ratio (n,), mean gap at 1.5-2 s (n,), side (n,) bool, gap at the 4 s arc (n,)."""
    s = plan_arc(Pr)
    d = at_arc(C, s) - at_arc(Rr, s)                                         # (n, 8, 2)
    g = np.hypot(d[..., 0], d[..., 1])
    e = d / np.maximum(g, 1e-9)[..., None]
    mv = ((Pc[:, :, :2] - Pr[:, :, :2]) * e).sum(-1)
    gap = g[:, TI].mean(1)
    ratio = mv[:, TI].sum(1) / np.maximum(g[:, TI].sum(1), 1e-9)
    ratio[~(gap >= MIN_GAP)] = np.nan
    side = np.array([pl_dist(Pc[i, 7, :2], C[i]) < pl_dist(Pc[i, 7, :2], Rr[i]) for i in range(len(Pc))])
    return ratio, gap, side, g[:, 7]


def selftest():
    """A straight real route and a counterfactual that bends left by 3.5 m over 20 .. 40 m: a plan halfway there reads 0.5, side checks."""
    M = 51
    s = S_BACK + S_STEP * np.arange(M)
    Rr = np.c_[s, np.zeros(M)][None]
    C = np.c_[s, np.clip((s - 5) / 10, 0, 1) * 3.5][None]                    # +3.5 m from 15 m on
    Pr = np.c_[np.arange(1, 9) * 5.0, np.zeros(8)][None]                    # 10 m/s straight
    gap_true = np.interp(plan_arc(Pr)[0, list(TI)], s, C[0, :, 1])
    Pc = Pr.copy()
    Pc[0, :, 1] = 0.5 * np.interp(plan_arc(Pr)[0], s, C[0, :, 1])
    r, gap, side, g4 = swap_read(Pr, Pc, C, Rr)
    assert abs(r[0] - 0.5) < 1e-6 and abs(gap[0] - gap_true.mean()) < 1e-6, (r, gap)
    assert not side[0] and g4[0] > 1                                         # halfway at 4 s: 1.75 m from both -> not nearer the cf
    Pc[0, :, 1] = np.interp(plan_arc(Pr)[0], s, C[0, :, 1])
    r, gap, side, _ = swap_read(Pr, Pc, C, Rr)
    assert abs(r[0] - 1.0) < 1e-6 and side[0]
    Pr2 = Pr * np.array([0.2, 1.0])                                          # slow: 1.5-2 s poses at 3 and 4 m, the routes still equal there
    r, gap, side, _ = swap_read(Pr2, Pr2, C, Rr)
    assert np.isnan(r[0]), r
    print("rf_gates selftest ok")


# ---------------------------------------------------------------- model passes (GPU)
def plans(model, front, rows, ego, tc, cam_x, pi, W, dev, bs=128):
    """Rows of a memory-mapped front (N, 8, 32, 512) with fed ego rows (n, 220) -> rear-axle plans (n, 8, 3) numpy."""
    import torch
    import pp_train as T
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            o = np.argsort(r)
            H = torch.from_numpy(np.ascontiguousarray(front[r[o]])).to(dev)[torch.from_numpy(np.argsort(o)).to(dev)]
            e = torch.from_numpy(ego[i:i + bs]).to(dev)
            p = model(H, e, tc[i:i + bs], None, None).float()[:, pi].view(-1, 33, 15)
            x, y, psi = T.rear(p, cam_x[i:i + bs], W)
            out.append(torch.stack([x, y, psi], -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main(a):
    import torch
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive import op_adapt as A
    from jevdrive import stats
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    from rf_cf import boot_med
    D = data_dir()
    CR, FEAT, ROWS = D / "runs/op_parity/cache", D / "runs/route_mem/feat", D / "runs/route_follow/cf/rows"
    dev = torch.device("cuda")
    with Run("route_follow", f"gates-{a.tag}", config=vars(a)) as run:
        run.use_split(splits.load("navsim/op-parity-full-dev"))
        if RM.has_branch(a.tag):
            model = RM.load_pmodel(a.tag, dev)
        else:
            assert a.allow_plain, f"{a.tag}: no {RM.SIDECAR} (not a ROUTE-MEM checkpoint); --allow-plain only for a code-path smoke"
            model = RM.attach(T.load_pmodel(a.tag, dev), **HEAD).eval()
            run.info(f"{a.tag}: plain checkpoint + zero branch (code-path smoke, not a result)")
        pi = torch.as_tensor(A.plan_index(model.net.slices), device=dev)
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev, dtype=torch.float32)

        def tcs(lht):
            return torch.as_tensor(np.where(np.asarray(lht, bool)[:, None], [[0.0, 1.0]], [[1.0, 0.0]]), dtype=torch.float32, device=dev)

        # ---- counterfactual rows
        tb, rt = dict(np.load(CR / a.pool / "tab.npz")), dict(np.load(CR / a.pool / "route.npz"))
        fz = np.load(FEAT / f"{a.pool}.npz")
        names = tb["names"]
        assert (rt["tokens"] == names).all() and (fz["names"] == names).all(), "pool files disagree on the row order"
        tok = np.array([n.split(":")[2] for n in names])
        cls = np.array([n.split(":")[3] for n in names])
        n = len(names)
        f_real, ego_real, Rr, okr = np.zeros((n, RM.DIM), np.float32), np.zeros((n, RM.EGO_DIM), np.float32), np.full((n, 51, 2), np.nan), np.zeros(n, bool)
        key = {(t, c): i for i, (t, c) in enumerate(zip(tok, cls))}
        for rf in sorted(ROWS.glob("*.npz")):
            data = rf.stem
            Z = np.load(rf)
            hit = [(i, key[t, c]) for i, (t, c) in enumerate(zip(Z["token"].tolist(), Z["cls"].tolist())) if (t, c) in key]
            if not hit:
                continue
            zi, pj = map(np.array, zip(*hit))
            src = Z["row"][zi]
            st, ft = np.load(CR / data / "tab.npz"), np.load(FEAT / f"{data}.npz")
            assert (st["names"][src] == tok[pj]).all() and (ft["names"][src] == tok[pj]).all(), f"{data}: token rows disagree"
            ego_real[pj], f_real[pj] = st["ego"][src], ft["f"][src].astype(np.float32)
            Rr[pj], okr[pj] = Z["real"][zi], ft["ok"][src] & (np.isfinite(Z["real"][zi]).all(-1).sum(-1) >= 2)
        ok = rt["ok"] & fz["ok"] & okr
        run.info(f"{a.pool}: {n} rows, ok {ok.sum()} (route {rt['ok'].sum()}, cf memory {fz['ok'].sum()}, real memory + polyline {okr.sum()})")
        rows = np.flatnonzero(ok)
        front = np.load(CR / f"{a.pool}@warp" / "front.npy", mmap_mode="r")
        tc, cam_x = tcs(tb["lht"][rows]), torch.as_tensor(tb["cam"][rows, 0], dtype=torch.float32, device=dev)
        f_cf = fz["f"][rows].astype(np.float32)
        t0 = time.time()
        P = {v: plans(model, front, rows, np.c_[e, f].astype(np.float32), tc, cam_x, pi, W, dev)
             for v, e, f in (("real", ego_real[rows], f_real[rows]), ("cf", tb["ego"][rows], f_cf), ("memcf", ego_real[rows], f_cf),
                             ("cmdcf", tb["ego"][rows], f_real[rows]))}
        run.info(f"CF rows: {len(rows)} x {len(P)} variants in {time.time() - t0:.0f} s")
        C = rt["poly"][rows].astype(np.float64)
        logs = tb["log"][rows]
        res = dict(tag=a.tag, plain=not RM.has_branch(a.tag), pool=a.pool, rows=int(n), rows_ok=int(len(rows)), classes={})
        per = {}
        for v in ("cf", "memcf", "cmdcf"):
            per[v] = swap_read(P["real"][:, :, :2], P[v][:, :, :2], C, Rr[rows])
        sr = swap_read(P["real"][:, :, :2], P["real"][:, :, :2], C, Rr[rows])[2]     # the real plan's own 4 s side (sanity: near the real route)
        for cn, m in (("junction", np.isin(cls[rows], ["j0", "j1"])), ("lc", cls[rows] == "lc"), ("all", np.ones(len(rows), bool))):
            r = dict(rows=int(m.sum()))
            for v, (ratio, gap, side, g4) in per.items():
                s4 = m & (g4 >= MIN_GAP4)
                r[v] = dict(ratio=boot_med(ratio[m], logs[m], B=a.B), gap_med=float(np.nanmedian(gap[m])) if m.any() else np.nan,
                            side=(stats.bootstrap(side[s4].astype(float), logs[s4], n_boot=a.B) if s4.any() else dict(n=0, mean=np.nan)),
                            side_real_plan=float(sr[s4].mean()) if s4.any() else np.nan)
            res["classes"][cn] = r
            run.info(f"{cn}: {json.dumps({v: [r[v]['ratio']['v'], r[v]['side'].get('mean')] for v in per})}")
        # ---- straight rows (G-ST)
        dvs = splits.load("navsim/op-parity-full-dev")
        cand = []
        for i in range(12):
            data = f"navtrain_full.s{i}of12"
            z = np.load(FEAT / f"{data}.npz")
            m = dvs.mask(z["names"]) & z["ok"] & ~np.isfinite(z["jdist"]) & np.isfinite(z["turn40"]) & (z["turn40"] < 5)
            cand += [(data, int(j)) for j in np.flatnonzero(m)]
        sel = sorted(np.random.default_rng(0).choice(len(cand), min(a.n_st, len(cand)), replace=False).tolist())
        st_rows = {}
        for j in sel:
            st_rows.setdefault(cand[j][0], []).append(cand[j][1])
        mv = {"shift+": [], "shift-": [], "mask": []}
        st_logs = []
        for data, rr in sorted(st_rows.items()):
            rr = np.array(rr)
            stb, z = np.load(CR / data / "tab.npz"), np.load(FEAT / f"{data}.npz")
            fr = np.load(CR / f"{data}@warp" / "front.npy", mmap_mode="r")
            e, f = stb["ego"][rr], z["f"][rr].astype(np.float32)
            tc_, cx = tcs(stb["lht"][rr]) if "lht" in stb.files else tcs(np.zeros(len(rr), bool)), torch.as_tensor(stb["cam"][rr, 0], dtype=torch.float32, device=dev)
            base = plans(model, fr, rr, np.c_[e, f].astype(np.float32), tc_, cx, pi, W, dev)
            for v, ff in (("shift+", RM.transform_rows(f, np.tile([[SHIFT, 0.0]], (len(rr), 1)))), ("shift-", RM.transform_rows(f, np.tile([[-SHIFT, 0.0]], (len(rr), 1)))),
                          ("mask", np.zeros_like(f))):
                q = plans(model, fr, rr, np.c_[e, ff].astype(np.float32), tc_, cx, pi, W, dev)
                mv[v].append(np.abs(q[:, TI, 1] - base[:, TI, 1]).mean(1))
            st_logs.append(stb["log"][rr])
        st_logs = np.concatenate(st_logs)
        mv = {k: np.concatenate(v) for k, v in mv.items()}
        mv["shift"] = np.maximum(mv["shift+"], mv["shift-"])
        res["straight"] = dict(candidates=len(cand), rows=len(st_logs), **{k: dict(med=boot_med(v, st_logs, B=a.B), p90=float(np.quantile(v, 0.9)), mean=float(v.mean()))
                                                                          for k, v in mv.items()})
        J = res["classes"]["junction"]["cf"]
        res["L2"] = dict(ratio_med_ge_050=bool(J["ratio"]["v"] >= 0.5), side_ge_070=bool(J["side"].get("mean", np.nan) >= 0.7),
                         st_shift_le_010=bool(res["straight"]["shift"]["med"]["v"] <= 0.10), st_mask_le_010=bool(res["straight"]["mask"]["med"]["v"] <= 0.10))
        res["L2"]["pass"] = all(res["L2"].values())
        out = Path(a.out) if a.out else D / "runs/route_follow/gates" / f"{a.tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        txt = json.dumps(res, indent=1, default=float) + "\n"
        out.write_text(txt)
        if not a.out:
            (REPO / "experiments/route_follow/results").mkdir(parents=True, exist_ok=True)
            (REPO / "experiments/route_follow/results" / f"gates_{a.tag}.json").write_text(txt)
        run.summary.update(gates=res)
        run.info(f"L2 {json.dumps(res['L2'])} -> {out}")
        print(json.dumps(dict(junction=res["classes"]["junction"], straight={k: v["med"]["v"] for k, v in res["straight"].items() if isinstance(v, dict)},
                              L2=res["L2"]), indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        selftest()
        raise SystemExit(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True), ap.add_argument("--pool", default="on1_CFdev")
    ap.add_argument("--n-st", type=int, default=1024), ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", default="", help="JSON path (default runs/route_follow/gates/<tag>.json + the results copy)")
    ap.add_argument("--allow-plain", action="store_true", help="code-path smoke on a checkpoint without the branch (zero branch attached)")
    main(ap.parse_args())
