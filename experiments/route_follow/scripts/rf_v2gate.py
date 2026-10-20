#!/usr/bin/env python3
"""ROUTE v2 pilot gates (experiments/route_follow/plans/2026-10-15-route-v2-prereg.md, "Gates"): offline, no simulator; one pilot checkpoint
against its base (RSG-L18-s0 by default). envs/op-train, one GPU (pool job: --vram 20 --cpu 4 --ram 48).

  rf_v2gate.py --tag RV2P-L18-s0 [--base RSG-L18-s0] [--n-st 1024] [--B 10000]
      -> $DATA_DIR/runs/route_follow/v2gate/<tag>.json + .md, and experiments/route_follow/results/v2gate/<tag>.{json,md} (box repo)

Models: rmem.load_pmodel (sidecar route_mem.json); each model is fed its own layout: the old window (feat/<data>.npz) or, for a sidecar head
with `order`, the ordered memory (feat/<data>.o.npz, rm_build order / rf_cfpool build --order). Fakes are built on the model's own point block
(rm_train's fake_shift / fake_straight, the trainer's own) and, for the ordered layout, re-scored with rmem.to_order_t. The over-turn read uses
the ordered block (slots at their true route arc) as the route for both models. Plans: rf_gates.plans (rear axle, left +, 0.5 .. 4 s).
G-slot (a property of the input layout, not of a model): `rf_v2gate.py slotmem` / `slot` below; its JSON is passed with --slot and shown.

Rows. dev = the dev rows (navsim/op-parity-full-dev) of the 12 navtrain shards with a logged future; junction = the route enters a junction
within 40 m (rm_build jdist), straight = no junction within 60 m and < 5 deg route turn over 40 m (rm_train.rely's split); curve = no junction
within 60 m and >= 15 deg route turn over 40 m. Both with a valid slot ahead in both layouts (the fakes need one). CF = the pool on1_CFdev (rf_cfpool pack, dev logs, never trained): junction rows j0 / j1 and
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
  G-slot   (layout, decision 299 Amendment 2's cases) the 53 turn zeros of experiments/route_follow/results/progress/zeros.csv whose served
           first valid slot was ahead of the ego (first_off_m3 > 0), at the decision 3 s before the failure (k_mem_m3): the arc at which the
           layout puts the memory's start (-s_e) against its true route arc (station of the merged memory's first point on rf_tax's rebuilt
           route minus the decision anchor's station); median |error| <= 0.5 m for the ordered layout (the projection layout reproduced alongside)
Bootstrap CIs (log-clustered, B) are printed for the medians / shares; the lines read the point estimates.

  rf_v2gate.py slotmem --set pai --zeros zeros.csv --runs <PAI root> .. --out J   (Tokyo, python3 + numpy: driver route messages)
  rf_v2gate.py slotmem --set held_s0 --zeros zeros.csv --runs <read run> .. --out J  (box, AlpaSim venv: rm_build regen at the anchors)
      per case: the old Tracker's window (checked against the served rmem), memory_o's arcs (s_geo = projection, s_e = progress anchor), the
      merged memory's first point in the local frame, the first valid slot of either layout
  rf_v2gate.py slot --mem J .. --tax NAME=<rf_tax pkl> .. --out results/v2gate/slot.json   (Mac .venv: the true arcs on the rebuilt route)
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
from jevdrive.tokens import open_front  # noqa: E402
import rmem as RM  # noqa: E402

TI = (2, 3)
LINES = dict(rely=0.01, swap_ratio=0.50, swap_side=0.70, st=0.10, lc=0.30, real=0.01)   # lc 0.50 -> 0.30: prereg Amendment 4


def feed(order, f, fo):
    """The model's own layout: old window f or ordered fo (both (n, ...) rows of the same decisions)."""
    return np.asarray(fo if order else f, np.float32)


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

        def run_plans(tg, front, rows, ego, f, fo, tc, cx):
            model, order = M[tg]
            return G.plans(model, front, rows, np.c_[ego, feed(order, f, fo)].astype(np.float32), tc, cx, pi, W, dev)

        # ---------------------------------------------------------------- dev rows (G-rely, G-real, G-over, G-ST)
        dv = []
        for i in range(12):
            d = f"navtrain_full.s{i}of12"
            z, zo, tb = np.load(FEAT / f"{d}.npz"), np.load(FEAT / f"{d}.o.npz"), np.load(CR / d / "tab.npz")
            r = np.flatnonzero(dvs.mask(z["names"]) & np.isfinite(tb["fut"][:, 0, 0]))
            if len(r):
                f, fo = z["f"][r].astype(np.float32), zo["f"][r].astype(np.float32)
                f[~z["ok"][r]], fo[~zo["ok"][r]] = 0, 0
                dv.append(dict(d=d, r=r, f=f, fo=fo, trav=zo["trav"][r], jd=z["jdist"][r], tu=z["turn40"][r], ego=tb["ego"][r], fut=tb["fut"][r, :, :2],
                               lht=tb["lht"][r], cam=tb["cam"][r, 0], log=tb["log"][r]))
        cat = lambda k: np.concatenate([x[k] for x in dv])  # noqa: E731
        JD, TU, LOG, FUT, F0 = cat("jd"), cat("tu"), cat("log"), cat("fut"), cat("fo")[:, :RM.DIM]
        J = np.isfinite(JD) & (JD <= 40)
        ST = ~np.isfinite(JD) & np.isfinite(TU) & (TU < 5)
        V0 = (RM.pts(F0)[:, RM.I0:, 0] > 0.5).any(1) & (RM.pts(cat("f"))[:, RM.I0:, 0] > 0.5).any(1)   # a valid slot ahead in both layouts
        CU = ~np.isfinite(JD) & np.isfinite(TU) & (TU >= 15) & V0
        run.info(f"dev rows {len(JD)} ({len(set(LOG.tolist()))} logs): junction {J.sum()}, straight {ST.sum()}, curve {CU.sum()}")
        rng = np.random.default_rng(0)
        st_pick = np.zeros(len(JD), bool)
        st_pick[rng.choice(np.flatnonzero(ST & V0), min(a.n_st, int((ST & V0).sum())), replace=False)] = True
        cu_pick = np.zeros(len(JD), bool)
        cu_pick[rng.choice(np.flatnonzero(CU), min(a.n_st, int(CU.sum())), replace=False)] = True

        def dev_variant(tg, fn):
            """fn(fed features (n, DIM | DIM_O), trav, mask, order) -> fed features; plans (N, 8, 3) over all dev rows."""
            out, o = [], 0
            order = M[tg][1]
            for x in dv:
                n = len(x["r"])
                fr = open_front(CR / f"{x['d']}@warp" / "front.npy")
                f = fn(feed(order, x["f"], x["fo"]), x["trav"], slice(o, o + n), order)
                out.append(run_plans(tg, fr, x["r"], x["ego"], f, f, tcs(x["lht"]), torch.as_tensor(x["cam"], dtype=torch.float32, device=dev)))
                o += n
            return np.concatenate(out)

        def sel_fake(kind, pick):
            def fn(f, trav, sl, order):
                f = f.copy()
                m = pick[sl]
                if m.any():
                    ft = torch.as_tensor(f[m, :RM.DIM], device=dev)
                    if kind == "straight":
                        fk = RT.fake_straight(ft)
                    else:
                        fk = RT.fake_shift(ft, torch.full((int(m.sum()),), float(kind), device=dev))
                    f[m] = (RM.to_order_t(fk, torch.as_tensor(trav[m], device=dev)) if order else fk).cpu().numpy()
                return f
            return fn

        t0 = time.time()
        P = {}
        for tg in M:
            P[tg] = dict(on=dev_variant(tg, lambda f, tv, sl, o: f), mask=dev_variant(tg, lambda f, tv, sl, o: np.zeros_like(f)),
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
        fz, fzo = np.load(FEAT / f"{a.pool}.npz"), np.load(FEAT / f"{a.pool}.o.npz")
        names = tb["names"]
        assert (fzo["names"] == names).all()
        assert (rt["tokens"] == names).all() and (fz["names"] == names).all()
        tok = np.array([n.split(":")[2] for n in names])
        cls = np.array([n.split(":")[3] for n in names])
        n = len(names)
        f_real, ego_real, Rr, okr = np.zeros((n, RM.DIM), np.float32), np.zeros((n, RM.EGO_DIM), np.float32), np.full((n, 51, 2), np.nan), np.zeros(n, bool)
        fo_real = np.zeros((n, RM.DIM_O), np.float32)
        key = {(t, c): i for i, (t, c) in enumerate(zip(tok, cls))}
        for rf in sorted(ROWS.glob("navtrain_full.s*of12.npz")):
            Z = np.load(rf)
            hit = [(i, key[t, c]) for i, (t, c) in enumerate(zip(Z["token"].tolist(), Z["cls"].tolist())) if (t, c) in key]
            if not hit:
                continue
            zi, pj = map(np.array, zip(*hit))
            src = Z["row"][zi]
            st_, ft, fto = np.load(CR / rf.stem / "tab.npz"), np.load(FEAT / f"{rf.stem}.npz"), np.load(FEAT / f"{rf.stem}.o.npz")
            assert (st_["names"][src] == tok[pj]).all() and (ft["names"][src] == tok[pj]).all()
            ego_real[pj], f_real[pj], fo_real[pj] = st_["ego"][src], ft["f"][src].astype(np.float32), fto["f"][src].astype(np.float32)
            Rr[pj], okr[pj] = Z["real"][zi], ft["ok"][src] & (np.isfinite(Z["real"][zi]).all(-1).sum(-1) >= 2)
        ok = rt["ok"] & fz["ok"] & okr
        rows = np.flatnonzero(ok)
        front = open_front(CR / f"{a.pool}@warp" / "front.npy")
        tc, cx = tcs(tb["lht"][rows]), torch.as_tensor(tb["cam"][rows, 0], dtype=torch.float32, device=dev)
        C, logs, cl = rt["poly"][rows].astype(np.float64), tb["log"][rows], cls[rows]
        res["cf_rows"] = dict(rows=int(n), ok=int(len(rows)), junction=int(np.isin(cl, ["j0", "j1"]).sum()), lc=int((cl == "lc").sum()))
        for tg in M:
            pr = run_plans(tg, front, rows, ego_real[rows], f_real[rows], fo_real[rows], tc, cx)
            pc = run_plans(tg, front, rows, tb["ego"][rows], fz["f"][rows].astype(np.float32), fzo["f"][rows].astype(np.float32), tc, cx)
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
            "G-LC": dict(value=v(p["lc_swing2"]), line=">= 0.30", base=v(b["lc_swing2"]), pass_=bool(v(p["lc_swing2"]) >= LINES["lc"])),
            "G-real": dict(value=p["ade"], line=f"<= {b['ade'] + LINES['real']:.4f}", base=b["ade"], pass_=bool(p["ade"] <= b["ade"] + LINES["real"])),
            "G-over": dict(value=v(p["over_share"]), line=f"<= {v(b['over_share']):.4f}", base=v(b["over_share"]), pass_=bool(v(p["over_share"]) <= v(b["over_share"]))),
        }
        if a.slot and Path(a.slot).exists():                         # the layout gate (slotmem / slot): same for every model with --rm-order
            sl = json.loads(Path(a.slot).read_text())
            res["gates"]["G-slot"] = dict(value=sl["new_med_abs"], line="<= 0.50 (layout, ordered)", base=sl["old_med_abs"],
                                          pass_=bool(sl["new_med_abs"] <= 0.5) if M[a.tag][1] else False)
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


def cases(zeros, name):
    import csv
    out = []
    for r in csv.DictReader(open(zeros)):
        f = lambda k: float(r[k]) if r[k] not in ("", "nan") else np.nan  # noqa: E731
        if r["set"] == name and r["turn_zero"] == "True" and f("first_off_m3") > 0:
            out.append(dict(set=name, scene=r["scene"], rid=r["rid"], k=int(float(r["k_mem_m3"])), s_e_m3=f("s_e_m3"), gnear_m3=f("gnear_m3"),
                            first_off_m3=f("first_off_m3")))
    return out


def first_slot(f):
    v = RM.pts(f)[:, 0] > 0.5
    j = np.flatnonzero(v & (RM.OFFS >= 0))
    return float(RM.OFFS[j[0]]) if len(j) else np.nan


def one_case(c, B, anchor, last, trav, served):
    old = RM.memory(B, anchor)
    info = {}
    new = RM.memory_o(B, anchor, last, trav, info)
    rep = float(np.abs(old - np.asarray(served, np.float32)).max()) if served is not None else np.nan
    return c | dict(anchor=[float(x) for x in anchor], s_geo=info.get("s_geo", np.nan), s_e=info.get("s_e", np.nan),
                    M0=RM.to_local(anchor, info["M0"][None])[0].tolist() if "M0" in info else None, first_old=first_slot(old), first_new=first_slot(new),
                    served_first=first_slot(np.asarray(served, np.float32)) if served is not None else np.nan, rep_err=rep)


def cmd_slotmem(a):
    want = {c["rid"]: c for c in cases(a.zeros, a.set)}
    out = []
    if a.set == "pai":
        for root in a.runs:
            for f in sorted(Path(root).glob("runs/*/driver/drive.jsonl")):
                R, D = {}, {}
                for line in open(f):
                    if not any(rid in line for rid in want):
                        continue
                    r = json.loads(line)
                    if r.get("session") not in want:
                        continue
                    if r["kind"] == "route":
                        R.setdefault(r["session"], {})[int(r["ts"])] = (np.asarray(r["pose"], float), np.asarray(r["wp"], float).reshape(-1, 2))
                    elif r["kind"] == "drive" and "rmem" in r and r.get("infer") is not False and int(r["k"]) == want[r["session"]]["k"]:
                        D.setdefault(r["session"], r)
                for sid, d in D.items():
                    t0, anchor = int(d["t0"]), np.asarray(d["anchor"], float)
                    msgs = sorted((ts, pw) for ts, pw in R.get(sid, {}).items() if ts <= t0)
                    B = [RM.to_local(pw[0], pw[1][np.isfinite(pw[1]).all(1)]) for _, pw in msgs if np.isfinite(pw[1]).all(1).any()]
                    last = [pw[0] for _, pw in msgs if np.isfinite(pw[1]).all(1).any()]
                    trav = RM.travelled([pw[0][:2] for _, pw in msgs], anchor) if msgs else 0.0
                    out.append(one_case(want[sid], B, anchor, last[-1] if last else anchor, trav, d["rmem"]))
            print(root, len(out), "cases", flush=True)
    else:
        sys.path[:0] = [str(REPO / "experiments/alpasim/scripts"), str(REPO / "experiments/alpasim/lib"), str(REPO / "experiments/anchor_offset/scripts"),
                        str(REPO / "experiments/eyes_vs_head/scripts")]
        import ev as EV
        import rm_build as RB
        from ao_e1 import drive_records
        by_scene = {c["scene"]: c for c in want.values()}
        for run in a.runs:
            dr = drive_records(run)
            for sc in set(dr) & set(by_scene):
                c, recs = by_scene[sc], dr[sc]
                W, A = RB.regen(sc, recs, EV.CFG)
                ks = [k for k in sorted(W) if k <= c["k"]]
                B, last = [], None
                for k in ks:
                    wp = W[k][np.isfinite(W[k]).all(1)]
                    if len(wp):
                        B.append(RM.to_local(A[k], wp))
                        last = A[k]
                served = next((r["rmem"] for r in recs if r["k"] == c["k"] and "rmem" in r), None)
                out.append(one_case(c, B, A[c["k"]], last if last is not None else A[c["k"]], RM.travelled([A[k][:2] for k in ks]), served))
            print(run, len(out), "cases", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(set=a.set, wanted=len(want), cases=out), indent=1, default=float))
    print(a.set, len(out), "/", len(want), "cases ->", a.out)


def cmd_slot(a):
    import pickle
    import rf_pai_follow as PF
    rolls = {}
    for spec in a.tax:
        name, fp = spec.split("=", 1)
        rolls[name] = {o["rid"]: o for o in pickle.load(open(fp, "rb"))["rolls"]}
    rows = []
    for fj in a.mem:
        J = json.loads(Path(fj).read_text())
        for c in J["cases"]:
            o = rolls[c["set"]].get(c["rid"])
            if o is None or c["M0"] is None:
                continue
            R = PF.dedup(np.asarray(o["route"], float))
            S = PF.stations(R)
            st_a = PF.project(R, S, np.asarray(c["anchor"][:2]), c["s_e_m3"] - 40.0, c["s_e_m3"] + 10.0)[0]
            st_m = PF.project(R, S, np.asarray(c["M0"]), st_a - 60.0, st_a + 130.0)[0]
            true = st_m - st_a
            rows.append(dict(set=c["set"], scene=c["scene"][7:15], rid=c["rid"][:8], k=c["k"], true=true, start_old=-c["s_geo"], start_new=-c["s_e"],
                             err_old=-c["s_geo"] - true, err_new=-c["s_e"] - true, first_served=c["served_first"], first_old=c["first_old"],
                             first_new=c["first_new"], rep_err=c["rep_err"], gnear_m3=c["gnear_m3"]))
    eo, en = np.array([r["err_old"] for r in rows]), np.array([r["err_new"] for r in rows])
    rep = np.array([r["rep_err"] for r in rows])
    res = dict(n=len(rows), wanted=53, old_med=float(np.median(eo)), old_med_abs=float(np.median(np.abs(eo))), old_p90_abs=float(np.quantile(np.abs(eo), 0.9)),
               new_med=float(np.median(en)), new_med_abs=float(np.median(np.abs(en))), new_p90_abs=float(np.quantile(np.abs(en), 0.9)),
               old_ge5=int((np.abs(eo) >= 5).sum()), new_ge5=int((np.abs(en) >= 5).sum()),
               served_reproduced=float(np.mean(np.nan_to_num(rep, nan=1.0) <= 1e-3)), rows=rows)
    res["pass"] = bool(res["new_med_abs"] <= 0.5)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    md = [f"# G-slot: where the layout puts the memory's start (m of route arc ahead of the ego) vs its true arc", "",
          f"{res['n']} / 53 cases (decision 299's turn zeros with the served first slot ahead, at the decision 3 s before the failure); served window "
          f"reproduced (max |diff| <= 1e-3) {res['served_reproduced']:.2f}.", "",
          "| layout | median error | median abs | p90 abs | abs >= 5 m |", "|:--|--:|--:|--:|--:|",
          f"| projection (served today) | {res['old_med']:+.2f} | {res['old_med_abs']:.2f} | {res['old_p90_abs']:.2f} | {res['old_ge5']} |",
          f"| ordered, progress anchor | {res['new_med']:+.2f} | {res['new_med_abs']:.2f} | {res['new_p90_abs']:.2f} | {res['new_ge5']} |", "",
          f"G-slot (median abs <= 0.5 m): {'PASS' if res['pass'] else 'FAIL'}", "",
          "| set | scene | true | projection start / first slot | ordered start / first slot | served first slot |", "|:--|:--|--:|--:|--:|--:|"]
    md += [f"| {r['set']} | {r['scene']} | {r['true']:.1f} | {r['start_old']:.1f} / {r['first_old']:.0f} | {r['start_new']:.1f} / {r['first_new']:.0f} | {r['first_served']:.0f} |"
           for r in sorted(rows, key=lambda r: -abs(r["err_old"]))]
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    if sys.argv[1:2] in (["slotmem"], ["slot"]):
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"), ap.add_argument("--set", default=""), ap.add_argument("--zeros", default=str(REPO / "experiments/route_follow/results/progress/zeros.csv"))
        ap.add_argument("--runs", nargs="*", default=[]), ap.add_argument("--mem", nargs="*", default=[]), ap.add_argument("--tax", nargs="*", default=[])
        ap.add_argument("--out", required=True)
        a = ap.parse_args()
        (cmd_slotmem if a.cmd == "slotmem" else cmd_slot)(a)
        raise SystemExit(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True), ap.add_argument("--base", default="RSG-L18-s0"), ap.add_argument("--pool", default="on1_CFdev")
    ap.add_argument("--n-st", type=int, default=1024), ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--slot", default=str(REPO / "experiments/route_follow/results/v2gate/slot.json"), help="G-slot JSON (rf_v2gate.py slot)")
    main(ap.parse_args())
