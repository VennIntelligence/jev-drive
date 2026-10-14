#!/usr/bin/env python3
"""ROUTE v2 pilot gates (experiments/route_follow/plans/2026-10-15-route-v2-prereg.md, "Gates"): offline, no simulator; one pilot checkpoint
against its base (RSG-L18-s0 by default). envs/op-train, one GPU (pool job: --vram 20 --cpu 4 --ram 48).

  rf_v2gate.py --tag RV2P-L18-s0 [--base RSG-L18-s0] [--n-st 1024] [--B 10000]
      -> $DATA_DIR/runs/route_follow/v2gate/<tag>.json + .md, and experiments/route_follow/results/v2gate/<tag>.{json,md} (box repo)

Models: rmem.load_pmodel (sidecar route_mem.json); each model is fed its own layout: the old window (feat/<data>.npz) or, for a sidecar head
with `order`, rmem.to_order(old window, trav) (trav from feat/<data>.o.npz; the counterfactual rows take their token's NAVSIM row's trav,
same k and poses). Fakes are built in the old layout and converted the same way (rm_train's fake_shift / fake_straight, the trainer's own).
Plans: rf_gates.plans (rear axle, left +, 0.5 .. 4 s).

Rows. dev = the dev rows (navsim/op-parity-full-dev) of the 12 navtrain shards with a logged future; junction = the route enters a junction
within 40 m (rm_build jdist), straight = no junction within 60 m and < 5 deg route turn over 40 m (rm_train.rely's split); curve = no junction
within 60 m and >= 15 deg route turn over 40 m. CF = the pool on1_CFdev (rf_cfpool pack, dev logs, never trained): junction rows j0 / j1 and
lane-change rows lc with their real twin (rf_gates' lookup).

  G-rely   junction dev rows: mean ADE(memory masked) - mean ADE(memory on) >= 0.01 m (pilot)
  G-swap   CF junction rows, counterfactual memory + command vs real: median share of the needed move at 1.5 - 2.0 s >= 0.50 and the 4 s pose
           nearer the counterfactual route >= 70 % (rf_gates.swap_read)
  G-ST     |lateral move| of the 1.5 / 2.0 s poses (mean of the two), median <= 0.10 m for (a) straight dev rows with the memory shifted
           sideways by +-3.5 m (max of the two signs per row), (b) curve dev rows with the memory replaced by the straight continuation
  G-LC     CF lane-change rows: |y(2 s) stay-lane route - y(2 s) real route| median >= 0.50 m
  G-real   dev ADE (memory on) <= base + 0.01 m
  G-over   junction dev rows: share whose plan heading at 2 s exceeds the memory route's heading at the same arc by > 10 deg toward the inside
           of the route's turn (rm_train.over_turn > 0) not above the base's
Bootstrap CIs (log-clustered, B) are printed for the medians / shares; the lines read the point estimates.
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
                str(REPO / "experiments/route_mem/scripts"), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/onpol_nuplan/lib")]
import rmem as RM  # noqa: E402

TI = (2, 3)
LINES = dict(rely=0.01, swap_ratio=0.50, swap_side=0.70, st=0.10, lc=0.50, real=0.01)


def feed(order, f, trav):
    return RM.to_order(f, trav) if order else np.asarray(f, np.float32)


def main(a):
    import torch
    import pp_train as T
    import rf_gates as G
    import rm_train as RT
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
    with Run("route_follow", f"v2gate-{a.tag}", config=vars(a)) as run:
        dvs = splits.load("navsim/op-parity-full-dev")
        run.use_split(dvs)
        M = {}
        for tg in (a.base, a.tag):
            assert RM.has_branch(tg), f"{tg}: no {RM.SIDECAR}"
            M[tg] = (RM.load_pmodel(tg, dev), RM.head_order(tg))
        run.info(f"models: {[(t, 'ordered' if o else 'old') for t, (_, o) in M.items()]}")
        m0 = next(iter(M.values()))[0]
        pi = torch.as_tensor(A.plan_index(m0.net.slices), device=dev)
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev, dtype=torch.float32)

        def tcs(lht):
            return torch.as_tensor(np.where(np.asarray(lht, bool)[:, None], [[0.0, 1.0]], [[1.0, 0.0]]), dtype=torch.float32, device=dev)

        def run_plans(tg, front, rows, ego, f_old, trav, tc, cx):
            model, order = M[tg]
            return G.plans(model, front, rows, np.c_[ego, feed(order, f_old, trav)].astype(np.float32), tc, cx, pi, W, dev)

        # ---------------------------------------------------------------- dev rows (G-rely, G-real, G-over, G-ST)
        tmap, dv = {}, []
        for i in range(12):
            d = f"navtrain_full.s{i}of12"
            z, zo, tb = np.load(FEAT / f"{d}.npz"), np.load(FEAT / f"{d}.o.npz"), np.load(CR / d / "tab.npz")
            tmap.update(zip(zo["names"].tolist(), zo["trav"].tolist()))
            r = np.flatnonzero(dvs.mask(z["names"]) & np.isfinite(tb["fut"][:, 0, 0]))
            if len(r):
                f = z["f"][r].astype(np.float32)
                f[~z["ok"][r]] = 0
                dv.append(dict(d=d, r=r, f=f, trav=zo["trav"][r], jd=z["jdist"][r], tu=z["turn40"][r], ego=tb["ego"][r], fut=tb["fut"][r, :, :2],
                               lht=tb["lht"][r], cam=tb["cam"][r, 0], log=tb["log"][r]))
        cat = lambda k: np.concatenate([x[k] for x in dv])  # noqa: E731
        JD, TU, LOG, FUT, F0 = cat("jd"), cat("tu"), cat("log"), cat("fut"), cat("f")
        J = np.isfinite(JD) & (JD <= 40)
        ST = ~np.isfinite(JD) & np.isfinite(TU) & (TU < 5)
        CU = ~np.isfinite(JD) & np.isfinite(TU) & (TU >= 15) & (RM.pts(F0)[:, RM.I0, 0] > 0.5)
        run.info(f"dev rows {len(JD)} ({len(set(LOG.tolist()))} logs): junction {J.sum()}, straight {ST.sum()}, curve {CU.sum()}")
        rng = np.random.default_rng(0)
        st_pick = np.zeros(len(JD), bool)
        st_pick[rng.choice(np.flatnonzero(ST & (RM.pts(F0)[:, RM.I0, 0] > 0.5)), min(a.n_st, int(ST.sum())), replace=False)] = True
        cu_pick = np.zeros(len(JD), bool)
        cu_pick[rng.choice(np.flatnonzero(CU), min(a.n_st, int(CU.sum())), replace=False)] = True

        def dev_variant(tg, fn):
            """fn(f_old (n, DIM), mask) -> fed old-layout features; plans (N, 8, 3) over all dev rows (rows outside mask keep their memory)."""
            out, o = [], 0
            for x in dv:
                n = len(x["r"])
                fr = np.load(CR / f"{x['d']}@warp" / "front.npy", mmap_mode="r")
                f = fn(x["f"], slice(o, o + n))
                out.append(run_plans(tg, fr, x["r"], x["ego"], f, x["trav"], tcs(x["lht"]), torch.as_tensor(x["cam"], dtype=torch.float32, device=dev)))
                o += n
            return np.concatenate(out)

        def sel_fake(kind, pick):
            def fn(f, sl):
                f = f.copy()
                m = pick[sl]
                if m.any():
                    ft = torch.as_tensor(f[m], device=dev)
                    if kind == "straight":
                        f[m] = RT.fake_straight(ft).cpu().numpy()
                    else:
                        s = torch.full((int(m.sum()),), float(kind), device=dev)
                        f[m] = RT.fake_shift(ft, s).cpu().numpy()
                return f
            return fn

        t0 = time.time()
        P = {}
        for tg in M:
            P[tg] = dict(on=dev_variant(tg, lambda f, sl: f), mask=dev_variant(tg, lambda f, sl: np.zeros_like(f)),
                         **{k: dev_variant(tg, sel_fake(v, pk)) for k, v, pk in (("sh+", 3.5, st_pick), ("sh-", -3.5, st_pick), ("straight", "straight", cu_pick))})
        run.info(f"dev plans: {len(M)} models x 5 variants in {time.time() - t0:.0f} s")

        def ade(Pl):
            return np.hypot(Pl[:, :, 0] - FUT[..., 0], Pl[:, :, 1] - FUT[..., 1]).mean(1)

        Ft = torch.as_tensor(F0, device=dev)
        Pt = RT.old_poly(Ft)
        res = dict(tag=a.tag, base=a.base, layouts={t: ("ordered" if o else "old") for t, (_, o) in M.items()}, dev_rows=int(len(JD)),
                   junction_rows=int(J.sum()), straight_rows=int(st_pick.sum()), curve_rows=int(cu_pick.sum()), models={})
        for tg in M:
            on, mk = ade(P[tg]["on"]), ade(P[tg]["mask"])
            xyp = torch.as_tensor(P[tg]["on"], dtype=torch.float32, device=dev)
            ov = (RT.over_turn(xyp, Pt, RT.OLD_GRID0, 10.0) > 0).cpu().numpy()
            mv = lambda v, pk: np.abs(P[tg][v][pk][:, TI, 1] - P[tg]["on"][pk][:, TI, 1]).mean(1)  # noqa: E731
            shm = np.maximum(mv("sh+", st_pick), mv("sh-", st_pick))
            stm = mv("straight", cu_pick)
            res["models"][tg] = dict(
                ade=float(on.mean()), ade_junction=float(on[J].mean()), rely_junction=float(mk[J].mean() - on[J].mean()),
                rely_straight=float(mk[ST].mean() - on[ST].mean()),
                over_share=stats.bootstrap(ov[J].astype(float), LOG[J], n_boot=a.B),
                st_shift=boot_med(shm, LOG[st_pick], B=a.B), st_straight=boot_med(stm, LOG[cu_pick], B=a.B),
                st_mask=boot_med(np.abs(P[tg]["mask"][st_pick][:, TI, 1] - P[tg]["on"][st_pick][:, TI, 1]).mean(1), LOG[st_pick], B=a.B))
            run.info(f"{tg}: {json.dumps({k: (v if not isinstance(v, dict) else v.get('v', v.get('mean'))) for k, v in res['models'][tg].items()}, default=float)}")

        # ---------------------------------------------------------------- CF rows (G-swap, G-LC)
        tb, rt = dict(np.load(CR / a.pool / "tab.npz")), dict(np.load(CR / a.pool / "route.npz"))
        fz = np.load(FEAT / f"{a.pool}.npz")
        names = tb["names"]
        assert (rt["tokens"] == names).all() and (fz["names"] == names).all()
        tok = np.array([n.split(":")[2] for n in names])
        cls = np.array([n.split(":")[3] for n in names])
        n = len(names)
        f_real, ego_real, Rr, okr = np.zeros((n, RM.DIM), np.float32), np.zeros((n, RM.EGO_DIM), np.float32), np.full((n, 51, 2), np.nan), np.zeros(n, bool)
        key = {(t, c): i for i, (t, c) in enumerate(zip(tok, cls))}
        for rf in sorted(ROWS.glob("*.npz")):
            Z = np.load(rf)
            hit = [(i, key[t, c]) for i, (t, c) in enumerate(zip(Z["token"].tolist(), Z["cls"].tolist())) if (t, c) in key]
            if not hit:
                continue
            zi, pj = map(np.array, zip(*hit))
            src = Z["row"][zi]
            st_, ft = np.load(CR / rf.stem / "tab.npz"), np.load(FEAT / f"{rf.stem}.npz")
            assert (st_["names"][src] == tok[pj]).all() and (ft["names"][src] == tok[pj]).all()
            ego_real[pj], f_real[pj] = st_["ego"][src], ft["f"][src].astype(np.float32)
            Rr[pj], okr[pj] = Z["real"][zi], ft["ok"][src] & (np.isfinite(Z["real"][zi]).all(-1).sum(-1) >= 2)
        ok = rt["ok"] & fz["ok"] & okr & np.array([t in tmap for t in tok])
        rows = np.flatnonzero(ok)
        trav = np.array([tmap.get(t, 0.0) for t in tok], np.float32)[rows]
        front = np.load(CR / f"{a.pool}@warp" / "front.npy", mmap_mode="r")
        tc, cx = tcs(tb["lht"][rows]), torch.as_tensor(tb["cam"][rows, 0], dtype=torch.float32, device=dev)
        C, logs, cl = rt["poly"][rows].astype(np.float64), tb["log"][rows], cls[rows]
        res["cf_rows"] = dict(rows=int(n), ok=int(len(rows)), junction=int(np.isin(cl, ["j0", "j1"]).sum()), lc=int((cl == "lc").sum()))
        for tg in M:
            pr = run_plans(tg, front, rows, ego_real[rows], f_real[rows], trav, tc, cx)
            pc = run_plans(tg, front, rows, tb["ego"][rows], fz["f"][rows].astype(np.float32), trav, tc, cx)
            ratio, gap, side, g4 = G.swap_read(pr[:, :, :2], pc[:, :, :2], C, Rr[rows])
            jm = np.isin(cl, ["j0", "j1"])
            s4 = jm & (g4 >= G.MIN_GAP4)
            lm = cl == "lc"
            sw = np.abs(pc[:, 3, 1] - pr[:, 3, 1])
            res["models"][tg] |= dict(swap_ratio=boot_med(ratio[jm], logs[jm], B=a.B), swap_side=stats.bootstrap(side[s4].astype(float), logs[s4], n_boot=a.B),
                                      lc_swing2=boot_med(sw[lm], logs[lm], B=a.B), lc_ratio=boot_med(ratio[lm], logs[lm], B=a.B),
                                      lc_rows=int(lm.sum()))
        # ---------------------------------------------------------------- lines
        p, b = res["models"][a.tag], res["models"][a.base]
        v = lambda x: x.get("v", x.get("mean")) if isinstance(x, dict) else x  # noqa: E731
        res["gates"] = {
            "G-rely": dict(value=p["rely_junction"], line=">= 0.01", base=b["rely_junction"], pass_=p["rely_junction"] >= LINES["rely"]),
            "G-swap": dict(value=[v(p["swap_ratio"]), v(p["swap_side"])], line=">= 0.50 / >= 0.70", base=[v(b["swap_ratio"]), v(b["swap_side"])],
                           pass_=bool(v(p["swap_ratio"]) >= LINES["swap_ratio"] and v(p["swap_side"]) >= LINES["swap_side"])),
            "G-ST": dict(value=[v(p["st_shift"]), v(p["st_straight"])], line="<= 0.10 / <= 0.10", base=[v(b["st_shift"]), v(b["st_straight"])],
                         pass_=bool(v(p["st_shift"]) <= LINES["st"] and v(p["st_straight"]) <= LINES["st"])),
            "G-LC": dict(value=v(p["lc_swing2"]), line=">= 0.50", base=v(b["lc_swing2"]), pass_=bool(v(p["lc_swing2"]) >= LINES["lc"])),
            "G-real": dict(value=p["ade"], line=f"<= {b['ade'] + LINES['real']:.4f}", base=b["ade"], pass_=bool(p["ade"] <= b["ade"] + LINES["real"])),
            "G-over": dict(value=v(p["over_share"]), line=f"<= {v(b['over_share']):.4f}", base=v(b["over_share"]), pass_=bool(v(p["over_share"]) <= v(b["over_share"]))),
        }
        res["all_pass"] = all(g["pass_"] for g in res["gates"].values())
        fmt = lambda x: (", ".join(f"{y:.3f}" for y in x) if isinstance(x, list) else f"{x:.4f}")  # noqa: E731
        md = [f"# Route v2 gates: {a.tag} vs {a.base}", "", f"dev rows {res['dev_rows']} (junction {res['junction_rows']}, straight picks {res['straight_rows']}, "
              f"curve picks {res['curve_rows']}); CF rows ok {res['cf_rows']['ok']} (junction {res['cf_rows']['junction']}, lc {res['cf_rows']['lc']}); layouts "
              f"{res['layouts']}", "", "| gate | pilot | base | line | pass |", "|:--|--:|--:|:--|:--|"]
        md += [f"| {k} | {fmt(g['value'])} | {fmt(g['base'])} | {g['line']} | {'PASS' if g['pass_'] else 'FAIL'} |" for k, g in res["gates"].items()]
        md += ["", "CIs (log-clustered): " + "; ".join(f"{k} {json.dumps({q: (x.get('lo'), x.get('hi')) for q, x in m.items() if isinstance(x, dict) and 'lo' in x}, default=float)}"
                                                       for k, m in res["models"].items())]
        out = D / "runs/route_follow/v2gate"
        out.mkdir(parents=True, exist_ok=True)
        txt = json.dumps(res, indent=1, default=float) + "\n"
        (out / f"{a.tag}.json").write_text(txt)
        (out / f"{a.tag}.md").write_text("\n".join(md) + "\n")
        rd = REPO / "experiments/route_follow/results/v2gate"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / f"{a.tag}.json").write_text(txt)
        (rd / f"{a.tag}.md").write_text("\n".join(md) + "\n")
        run.summary.update(gates={k: g["pass_"] for k, g in res["gates"].items()}, all_pass=res["all_pass"])
        print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True), ap.add_argument("--base", default="RSG-L18-s0"), ap.add_argument("--pool", default="on1_CFdev")
    ap.add_argument("--n-st", type=int, default=1024), ap.add_argument("--B", type=int, default=10000)
    main(ap.parse_args())
