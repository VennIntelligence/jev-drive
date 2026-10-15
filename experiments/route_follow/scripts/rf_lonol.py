#!/usr/bin/env python3
"""LON-DIAG reading 3 (plans/2026-10-15-longitudinal-diag-prereg.md, "读数 3"): open-loop counterfactuals of RSG-L18-s0 on nuPlan offline rows.
Frames untouched, only the fed ego row changes (rm_train EGO_SPEED: vx, vy, ax, ay and the x / y of the 4 history poses). Box, op-train python.

  rf_lonol.py cells  --shards 0 --out DIR      CPU: per row of the shards' navtrain cache, the cell flags from LONG-PROBE's context
                                               (long_contact ctx.npz: route, stop lines, light states) and on_lon.build_rows on body1's
                                               agent labels (A-eligible leads, on_lon's own rule) -> DIR/cells.npz, prints the cell sizes
  rf_lonol.py run    --shards 0 --tag T --out DIR   GPU (one pool job): rm_train.build's store (rf_lonread.py's way: --data the shards,
                                               --yr-mass 0, the RSG pool on1_L18RB0 appended and unused), checkpoint T; the gate, then the
                                               plans of every offline row under 6 fed rows -> DIR/plans.npz, DIR/gate.json
  rf_lonol.py report --cells DIR --run DIR --out RESULTS --figs FIGS   CPU: the registered lines -> RESULTS/openloop.{md,json},
                                               FIGS/lon_diag_history_gain.png, FIGS/lon_diag_red_brake.png

Fed rows (conditions): S06 S08 H S12 S14 = the measured row with the EGO_SPEED entries x 0.6 / 0.8 / 1.0 / 1.2 / 1.4 (vx, vy, ax, ay and the
history poses scaled radially about the t0 rear axle, i.e. along their own direction; present, command, yaws and the route features stay);
F = flat history: vx kept, vy = ax = ay = 0, the 3 past poses re-spaced along their own polyline to v x (0.5, 1.0, 1.5) s at v = vx
(experiments/alpasim/lib/vhist.py respace, the serving hold's rule; its yaw interpolation included).

Quantities: v_p = arc of the plan from the rear axle through its 0.5 .. 2.0 s poses / 2 s (vhist.arc2); v_f = the fed vx (m/s); brake
b = (v_f - v_p) / v_f. Cells (line distances from the front bumper, ln_d - 4.049 m, slp_probe's convention; the first traffic-light line
ahead; human speed = the cache's t0 speed, ax = the cache's t0 longitudinal acceleration):
  free   no A-eligible actor with gap <= 40 m, no stop line of any type <= 60 m, route heading change over 0 .. 60 m < 30 deg, speed >= 2 m/s
  red    the first traffic-light line ahead is LONG-PROBE-eligible (ln_elig, connector matched), red at t0, 10 .. 40 m; speed >= 6; ax < -0.5
  green  the same line state green at t0 (otherwise identical; the prereg's "same conditions")
  lead   the nearest A-eligible actor (on_lon.build_rows a_g0) 0 .. 30 m ahead and closing (its arc speed over 0 .. 0.5 s below the ego's
         t0 speed); speed >= 6; ax < -0.5; no red traffic-light line within 60 m
  + secondary: red_all / green_all / lead_all (without the ax condition), red_nolead (no A-eligible actor before the line)
Statistics: log-clustered percentile bootstrap (B 10 000, seed 0) of medians / ratios of medians / the fixed-effects slope.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R / "experiments/route_mem/scripts"), str(_R / "experiments/onpol_nuplan/lib"), str(_R / "experiments/body1/lib"),
                 str(_R / "experiments/long_contact/lib"), str(_R / "experiments/alpasim/lib"), str(_R), str(_pl.Path(__file__).resolve().parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

FRONT = 4.049
EGO_SPEED = [4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18]
SCALES = (0.6, 0.8, 1.0, 1.2, 1.4)
CONDS = ("S06", "S08", "H", "S12", "S14", "F")
BINS, BIN_NAMES = (8.0, 12.0, 16.0), ("< 8", "8-12", "12-16", ">= 16")
CELLS = ("free", "red", "green", "lead", "red_all", "green_all", "lead_all", "red_nolead")
NB, SEED = 10_000, 0


# ---------------------------------------------------------------- cells (CPU)
def cmd_cells(a):
    import b1 as B
    import on_lon as OL
    L, off = B.labels(), B.shard_offsets()
    z = np.load(B.data_dir() / "runs/long_contact/ctx/ctx.npz")
    keep = ("route", "rhead", "rlen", "vlog", "ln_type", "ln_d", "ln_st", "ln_conn", "ln_elig", "tn_ok", "tn_spk", "tn_vpk", "tokens", "ok")
    Z = {k: z[k] for k in keep}
    kf = np.clip(np.round(np.arange(41) * 0.1 / 0.5).astype(int), 0, 8)              # dense step -> nearest light keyframe
    out = {k: [] for k in ("names", "log", "shard", "v0", "vx", "ax", "dline", "st0", "lead_g0", "lead_v", "closing", "turn60") + CELLS}
    for k in a.shards:
        t = B.tab("log", k)
        g = np.arange(off[k], off[k + 1])
        assert np.array_equal(t["names"], Z["tokens"][g]) and np.array_equal(t["names"], L["tokens"][g]) and Z["ok"][g].all(), f"shard {k}: rows"
        n = len(g)
        v0, vx, ax = t["speed"].astype(float), t["ego"][:, 4] * 10.0, t["ego"][:, 6] * 3.0
        assert np.allclose(ax, t["acc"][:, -1, 0], atol=1e-4), "ego ax column"
        BR = {q: [] for q in ("a_g0", "a_sig", "a_val")}
        for i in range(0, n, 512):                                      # on_lon's A eligibility on the token's own logged agents
            s = g[i:i + 512]
            r = OL.build_rows(Z["route"][s], Z["rhead"][s], Z["rlen"][s], np.asarray(L["box"][s]), np.asarray(L["valid"][s]), np.asarray(L["cls"][s]),
                              Z["ln_type"][s], Z["ln_d"][s], Z["ln_elig"][s], (Z["ln_st"][s][:, :, kf] == 1), Z["tn_ok"][s], Z["tn_spk"][s], Z["tn_vpk"][s],
                              Z["vlog"][s].astype(np.float32), np.zeros((len(s), 8), np.float32))
            for q in BR:
                BR[q].append(r[q])
        a_g0, a_sig, a_val = (np.concatenate(BR[q]) for q in ("a_g0", "a_sig", "a_val"))
        pe = a_g0 > -OL.BIG / 2
        ty, dF, st, conn, el = Z["ln_type"][g], Z["ln_d"][g] - FRONT, Z["ln_st"][g], Z["ln_conn"][g], Z["ln_elig"][g]
        tl = (ty == 2) & np.isfinite(dF) & (dF > 0)
        q = np.where(tl.any(1), tl.argmax(1), 0)
        has = tl.any(1)
        ar = np.arange(n)
        dq, sq = np.where(has, dF[ar, q], np.nan), np.where(has, st[ar, q, 0], -1)
        ok_line = has & el[ar, q] & conn[ar, q] & (dq >= 10) & (dq <= 40)
        red_line, green_line = ok_line & (sq == 1), ok_line & (sq == 0)
        red60 = ((ty == 2) & np.isfinite(dF) & (dF <= 60) & (st[:, :, 0] == 1)).any(1)
        any_line60 = ((ty > 0) & np.isfinite(dF) & (dF <= 60)).any(1)
        rh = Z["rhead"][g][:, :61].astype(float)
        turn60 = np.degrees(np.abs(np.angle(np.exp(1j * (rh - rh[:, :1])))).max(1))
        g0 = np.where(pe[:, 0], a_g0[:, 0], np.inf)
        both = a_val[:, 0, 0] & a_val[:, 0, 5]
        lv = np.where(pe[:, 0] & both, (a_sig[:, 0, 5] - a_sig[:, 0, 0]) / 0.5, np.nan)
        closing = np.isfinite(lv) & (v0 - lv > 0)
        lead_in = lambda m: (pe & (a_g0 <= m)).any(1)                    # noqa: E731
        fast, dec = v0 >= 6.0, ax < -0.5
        has_lead = (g0 > 0) & (g0 <= 30) & closing
        c = dict(free=~lead_in(40.0) & ~any_line60 & (turn60 < 30.0) & (v0 >= 2.0),
                 red=red_line & fast & dec, green=green_line & fast & dec, lead=has_lead & fast & dec & ~red60,
                 red_all=red_line & fast, green_all=green_line & fast, lead_all=has_lead & fast & ~red60,
                 red_nolead=red_line & fast & dec & ~(pe & (a_g0 <= np.nan_to_num(dq, nan=-1)[:, None])).any(1))
        for key, val in dict(names=t["names"], log=t["log"], shard=np.full(n, k), v0=v0, vx=vx, ax=ax, dline=dq, st0=sq, lead_g0=g0, lead_v=lv,
                             closing=closing, turn60=turn60, **c).items():
            out[key].append(val)
        print(f"shard {k}: rows {n}, " + ", ".join(f"{q} {int(c[q].sum())}" for q in CELLS), flush=True)
    O = {k: np.concatenate(v) for k, v in out.items()}
    d = _pl.Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "cells.npz", **O)
    cnt = {q: dict(rows=int(O[q].sum()), logs=len(set(O["log"][O[q]].tolist()))) for q in CELLS}
    (d / "cells.json").write_text(json.dumps(dict(shards=a.shards, rows=int(len(O["names"])), cells=cnt), indent=1))
    print(json.dumps(cnt), flush=True)


# ---------------------------------------------------------------- run (GPU)
def flat_rows(E):
    """E (n, 20) measured ego rows -> F rows (vhist.respace at v = vx)."""
    import vhist as VH
    F = E.astype(np.float64).copy()
    F[:, 5:8] = 0.0
    for i in range(len(F)):
        P = E[i, 8:20].reshape(4, 3).astype(np.float64) * [10.0, 10.0, 1.0]
        F[i, 8:20] = (VH.respace(P, max(float(E[i, 4]) * 10.0, 0.0)) / [10.0, 10.0, 1.0]).reshape(12)
    return F.astype(np.float32)


def cmd_run(a):
    import torch
    from types import SimpleNamespace
    import rm_train as RT
    import pp_train as T
    from rf_lonread import REC
    dev = torch.device("cuda")
    data = [f"navtrain_full.s{k}of12" for k in a.shards]
    ns_a = SimpleNamespace(**(REC | dict(data=data)), road_labels=RT.OT.JT.ROAD, on=a.pool, init="", rm_from=a.tag, tag=a.tag, exp="route_follow",
                           lr_scale=1.0, prefetch=1, dev_card_gb=4.0)
    ns, _, _, _, _ = RT.build(ns_a, dev)
    S, model, on0 = ns.S, ns.model, ns.on0
    W = ns.losses(ns.mode).W
    model.eval()
    names = S.tab["names"][:on0]
    C = np.load(_pl.Path(a.cells) / "cells.npz", allow_pickle=False)
    assert np.array_equal(C["names"], names), "cells rows != store's offline rows (same shards, same order)"
    pi = torch.as_tensor(S.pi, device=dev)
    esp = torch.as_tensor(EGO_SPEED, device=dev)
    E0 = S.ego[:on0, :20].cpu().numpy()
    Fd = torch.as_tensor(flat_rows(E0), device=dev)

    def fwd(rows, cond, bs=256):
        xs, ys = [], []
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                r = torch.as_tensor(rows[i:i + bs], device=dev)
                e = S.ego[r]
                if cond == "F":
                    e = e.clone()
                    e[:, :20] = Fd[r]
                elif cond != "H":
                    e = e.clone()
                    e[:, esp] = e[:, esp] * float(cond[1:]) / 10.0
                p = model(S.front[r], e, S.tc[r], None, None, nv=None if S.nv is None else S.nv[r]).float()[:, pi].view(-1, 33, 15)
                x, y, _ = T.rear(p, S.cam_x[r], W)
                xs.append(x.cpu())
                ys.append(y.cpu())
        return torch.cat(xs).numpy(), torch.cat(ys).numpy()

    # gate: edits off -> the plans of rm_train's dev evaluation (rows_ade, memory on, bs 256), per-row ADE bit for bit, on the shards' dev rows
    dv = ns.dv_o[ns.dv_o < on0]
    dv = dv[S.has_fut[torch.as_tensor(dv, device=dev)].cpu().numpy().astype(bool)]
    ref = RT.rows_ade(model, S, dv, W, False)
    x, y = fwd(dv, "H")
    fut = S.fut[torch.as_tensor(dv, device=dev)].cpu().numpy()
    mine = torch.hypot(torch.as_tensor(x) - torch.as_tensor(fut[..., 0]), torch.as_tensor(y) - torch.as_tensor(fut[..., 1])).mean(1).numpy()
    xs, ys = fwd(dv, "S10")                                           # the x 1.0 scale path must be the measured row
    gate = dict(dev_rows=int(len(dv)), ade_equal_bitwise=bool(np.array_equal(ref, mine)), ade_max_abs_diff=float(np.abs(ref - mine).max()),
                dev_ade_rows_ade=float(ref.mean()), dev_ade_mine=float(mine.mean()), scale1_equal_bitwise=bool(np.array_equal(x, xs) and np.array_equal(y, ys)),
                f_changes_only_speed_cols=bool(np.array_equal(np.delete(Fd.cpu().numpy(), EGO_SPEED + [10, 13, 16], 1), np.delete(E0, EGO_SPEED + [10, 13, 16], 1))))
    print("gate", json.dumps(gate), flush=True)
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    assert gate["ade_equal_bitwise"] and gate["scale1_equal_bitwise"] and gate["f_changes_only_speed_cols"], "gate failed: no counterfactual read"
    rows = np.arange(on0)
    P = {}
    for c in CONDS:
        x, y = fwd(rows, c)
        P[c] = np.stack([x, y], -1).astype(np.float32)                 # (n, 8, 2) rear-axle plan
        print(c, "done", flush=True)
    sp = np.zeros(on0, bool)
    sp[ns.dv_o[ns.dv_o < on0]] = True
    np.savez(out / "plans.npz", names=names, dev=sp, ego=E0, egoF=Fd.cpu().numpy(), **{f"plan_{c}": v for c, v in P.items()})
    print("wrote", out / "plans.npz", flush=True)


# ---------------------------------------------------------------- report (CPU)
def arc2(P):
    p = np.concatenate([np.zeros(P.shape[:-2] + (1, 2)), P[..., :4, :]], -2)
    return np.linalg.norm(np.diff(p, axis=-2), axis=-1).sum(-1)


def cboot(fn, logs, B=NB, seed=SEED):
    """Log-clustered percentile bootstrap: fn(idx) on the row indices of the drawn logs -> (point, lo, hi)."""
    u, inv = np.unique(logs, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(u))]
    rng = np.random.default_rng(seed)
    pt = fn(np.arange(len(logs)))
    bs = np.array([fn(np.concatenate([members[j] for j in rng.integers(0, len(u), len(u))])) for _ in range(B)])
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    return float(pt), float(lo), float(hi)


def fe_slope(vf, vp):
    """Fixed-effects (within-row) OLS slope of v_p on v_f over the scales; vf / vp (n, 5) -> per-row sums."""
    xc, yc = vf - vf.mean(1, keepdims=True), vp - vp.mean(1, keepdims=True)
    return (xc * yc).sum(1), (xc * xc).sum(1)


def cmd_report(a):
    C = dict(np.load(_pl.Path(a.cells) / "cells.npz", allow_pickle=False))
    R = np.load(_pl.Path(a.run) / "plans.npz", allow_pickle=False)
    gate = json.loads((_pl.Path(a.run) / "gate.json").read_text())
    assert np.array_equal(R["names"], C["names"])
    vx = R["ego"][:, 4] * 10.0
    vp = {c: arc2(R[f"plan_{c}"].astype(np.float64)) / 2.0 for c in CONDS}
    logs = C["log"]
    res = dict(prereg="experiments/route_follow/plans/2026-10-15-longitudinal-diag-prereg.md (读数 3)", tag=a.tag, shards=sorted(set(C["shard"].tolist())),
               rows=int(len(vx)), gate=gate, cells={q: dict(rows=int(C[q].sum()), logs=len(set(logs[C[q]].tolist()))) for q in CELLS})
    # (i) history gain
    vfS = np.stack([vx * s for s in SCALES], 1)
    vpS = np.stack([vp[c] for c in CONDS[:5]], 1)
    sxy, sxx = fe_slope(vfS, vpS)
    free = C["free"]
    bins = np.digitize(C["v0"], BINS)

    def gamma(m):
        i = np.flatnonzero(m)
        if len(i) < 2:
            return None
        pt, lo, hi = cboot(lambda j: sxy[i[j]].sum() / sxx[i[j]].sum(), logs[i])
        rs = sxy[i] / np.maximum(sxx[i], 1e-9)
        return dict(rows=int(len(i)), logs=len(set(logs[i].tolist())), gamma=pt, lo=lo, hi=hi, median_row_slope=float(np.median(rs)))
    G = {"all": gamma(free)} | {BIN_NAMES[b]: gamma(free & (bins == b)) for b in range(4)}
    res["history_gain"] = G
    res["history_gain_curve"] = {BIN_NAMES[b]: [float(np.median(vpS[free & (bins == b), j] / vx[free & (bins == b)])) if (free & (bins == b)).any() else None
                                                for j in range(5)] for b in range(4)}
    g = G["all"]["gamma"]
    res["line_i"] = dict(rule="gamma >= 0.6 history-led; <= 0.3 frame-led; between: mixed", gamma=g,
                         verdict="history-led" if g >= 0.6 else "frame-led" if g <= 0.3 else "mixed")
    # (ii) brake source
    bH, bF = (vx - vp["H"]) / vx, (vx - vp["F"]) / vx

    def brake(q):
        i = np.flatnonzero(C[q])
        if len(i) < 3:
            return dict(rows=int(len(i)))
        ratio = cboot(lambda j: np.median(bF[i[j]]) / np.median(bH[i[j]]), logs[i])
        mH, mF = cboot(lambda j: np.median(bH[i[j]]), logs[i]), cboot(lambda j: np.median(bF[i[j]]), logs[i])
        pr = bH[i] > 0.05
        return dict(rows=int(len(i)), logs=len(set(logs[i].tolist())), b_H=mH, b_F=mF, ratio=ratio,
                    median_rowwise_ratio_bH_gt_0p05=float(np.median(bF[i][pr] / bH[i][pr])) if pr.any() else None, rows_bH_gt_0p05=int(pr.sum()),
                    v_f_median=float(np.median(vx[i])), ax_median=float(np.median(C["ax"][i])), dline_median=float(np.nanmedian(C["dline"][i])),
                    lead_g0_median=float(np.median(C["lead_g0"][i])) if q.startswith("lead") else None)
    res["brake"] = {q: brake(q) for q in CELLS if q != "free"}
    rr = res["brake"]["red"].get("ratio", [np.nan])[0]
    lr = res["brake"]["lead"].get("ratio", [np.nan])[0]
    res["line_ii_red"] = dict(rule="median b_F / b_H < 0.3 reads stop from history; 0.3-0.7 partial; >= 0.7 reads the light", ratio=rr,
                              verdict="history" if rr < 0.3 else "light" if rr >= 0.7 else "partial")
    res["line_ii_lead"] = dict(rule="lead ratio >= 0.7 and red < 0.3 -> the light / line channel is what is missing", lead_ratio=lr, red_ratio=rr,
                               verdict=bool(lr >= 0.7 and rr < 0.3))
    gb = res["brake"]["green"].get("b_H", [np.nan])[0]
    res["line_ii_green"] = dict(rule="negative control: b under H near 0 on green lines", b_H_green=gb, b_H_red=res["brake"]["red"].get("b_H", [np.nan])[0])
    # (iii) cap
    capr = vp["H"] / vx
    cap = {}
    for b in range(4):
        i = np.flatnonzero(free & (bins == b))
        cap[BIN_NAMES[b]] = dict(rows=int(len(i)), logs=len(set(logs[i].tolist())), **dict(zip(("median", "lo", "hi"), cboot(lambda j: np.median(capr[i[j]]), logs[i])))) if len(i) >= 3 else dict(rows=int(len(i)))
    res["cap"] = cap
    hi16 = cap[">= 16"].get("median", np.nan)
    res["line_iii"] = dict(rule=">= 16 m/s bin median v_p / v_f < 0.85 -> distribution cap", median_ge16=hi16, verdict=bool(hi16 < 0.85))
    if a.speeds:
        res["store_speed"] = json.loads(_pl.Path(a.speeds).read_text())
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "openloop.json").write_text(json.dumps(res, indent=1))
    (out / "openloop.md").write_text(md(res))
    figs(res, C, vx, vp, bH, bF, _pl.Path(a.figs))
    print(md(res))


def _ci(t, f=".3f"):
    return "n/a" if t is None or not np.isfinite(t[0]) else f"{t[0]:{f}} [{t[1]:{f}}, {t[2]:{f}}]"


def md(r):
    L = [f"# LON-DIAG reading 3: open-loop counterfactuals ({r['tag']})", "",
         f"Prereg: `{r['prereg']}`. Script: `experiments/route_follow/scripts/rf_lonol.py` (docstring = definitions). Store: navtrain_full shard(s) "
         f"{r['shards']} ({r['rows']} offline rows, rm_train.build as rf_lonread.py). CIs: log-clustered percentile bootstrap, B 10 000, seed 0.", "",
         "## Gate", "", "Edits off vs rm_train's dev evaluation (rows_ade, memory on): "
         f"{r['gate']['dev_rows']} dev rows, per-row ADE bit-for-bit equal: **{r['gate']['ade_equal_bitwise']}** (max |diff| {r['gate']['ade_max_abs_diff']:.3g}, "
         f"dev ADE {r['gate']['dev_ade_rows_ade']:.6f}); x 1.0 scale path = measured row bit for bit: {r['gate']['scale1_equal_bitwise']}; F touches only the "
         f"speed columns and the history yaws: {r['gate']['f_changes_only_speed_cols']}.", "",
         "## Cells", "", "| cell | rows | logs |", "|:--|--:|--:|"]
    L += [f"| {q} | {v['rows']} | {v['logs']} |" for q, v in r["cells"].items()]
    L += ["", "## (i) History gain (free straight rows)", "",
          "gamma = fixed-effects OLS slope of v_p on the scaled v_f over x 0.6 .. 1.4 (pooled within-row sums).", "",
          "| human speed | rows | logs | gamma [95 % CI] | median per-row slope | median v_p / v_f at x 0.6 / 0.8 / 1.0 / 1.2 / 1.4 |", "|:--|--:|--:|:--|--:|:--|"]
    for k, v in r["history_gain"].items():
        if v is None:
            L.append(f"| {k} | 0 | | | | |")
            continue
        cv = r["history_gain_curve"].get(k)
        L.append(f"| {k} | {v['rows']} | {v['logs']} | {_ci((v['gamma'], v['lo'], v['hi']))} | {v['median_row_slope']:.3f} | "
                 f"{' / '.join('n/a' if c is None else f'{c:.3f}' for c in cv) if cv else ''} |")
    li = r["line_i"]
    L += ["", f"Line (i) ({li['rule']}): gamma {li['gamma']:.3f} -> **{li['verdict']}**.", "",
          "## (ii) Brake source: b = (v_f - v_p) / v_f under H (own history) vs F (flat history)", "",
          "| cell | rows | logs | median b_H | median b_F | b_F / b_H (ratio of medians) | row-wise median ratio (b_H > 0.05, rows) | v_f | ax | line / lead m |",
          "|:--|--:|--:|:--|:--|:--|:--|--:|--:|--:|"]
    for q, v in r["brake"].items():
        if "b_H" not in v:
            L.append(f"| {q} | {v['rows']} | | | | | | | | |")
            continue
        dd = v["lead_g0_median"] if v["lead_g0_median"] is not None else v["dline_median"]
        rw = "n/a" if v["median_rowwise_ratio_bH_gt_0p05"] is None else f"{v['median_rowwise_ratio_bH_gt_0p05']:.3f}"
        L.append(f"| {q} | {v['rows']} | {v['logs']} | {_ci(v['b_H'])} | {_ci(v['b_F'])} | {_ci(v['ratio'])} | {rw} ({v['rows_bH_gt_0p05']}) | "
                 f"{v['v_f_median']:.1f} | {v['ax_median']:.2f} | {dd:.1f} |")
    a, b, c = r["line_ii_red"], r["line_ii_lead"], r["line_ii_green"]
    L += ["", f"Line (ii) red ({a['rule']}): ratio {a['ratio']:.3f} -> **{a['verdict']}**.",
          f"Positive control ({b['rule']}): lead {b['lead_ratio']:.3f}, red {b['red_ratio']:.3f} -> **{'met' if b['verdict'] else 'not met'}**.",
          f"Negative control ({c['rule']}): b_H green {c['b_H_green']:.3f} vs red {c['b_H_red']:.3f}.", "",
          "## (iii) Cap (free rows, H)", "", "| human speed | rows | logs | median v_p / v_f [95 % CI] |", "|:--|--:|--:|:--|"]
    for k, v in r["cap"].items():
        L.append(f"| {k} | {v['rows']} | {v.get('logs', '')} | {_ci((v['median'], v['lo'], v['hi'])) if 'median' in v else 'n/a'} |")
    li = r["line_iii"]
    L += ["", f"Line (iii) ({li['rule']}): {li['median_ge16']:.3f} -> **{'cap holds' if li['verdict'] else 'no cap'}**."]
    if "store_speed" in r:
        L += ["", "Training store row speed (m/s, the cache's t0 speed):", "", "| rows | n | p50 | p90 | p99 | max | share >= 16 |", "|:--|--:|--:|--:|--:|--:|--:|"]
        L += [f"| {k} | {v['n']} | {v['p50']:.2f} | {v['p90']:.2f} | {v['p99']:.2f} | {v['max']:.2f} | {v['ge16']:.4f} |" for k, v in r["store_speed"].items()]
    L += ["", "Figures: `figs/lon_diag_history_gain.png` (median v_p against the scaled v_f per speed bin, the diagonal = plan follows the history "
          "one for one; look at the slope against the diagonal), `figs/lon_diag_red_brake.png` (b under H and F per cell; look at whether the red "
          "and lead bars keep their H height under F)."]
    return "\n".join(L) + "\n"


def figs(r, C, vx, vp, bH, bF, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys_path = str(_R / "research")
    if sys_path not in _sys.path:
        _sys.path.insert(0, sys_path)
    import plot_style as PS
    PS.apply()
    d.mkdir(parents=True, exist_ok=True)
    free, bins = C["free"], np.digitize(C["v0"], BINS)
    col = [PS.PALETTE[k] for k in ("blue", "green", "orange", "vermillion")]
    f, ax = plt.subplots(figsize=(3.4, 3.0))
    for b in range(4):
        m = free & (bins == b)
        if m.sum() < 3:
            continue
        xf = [np.median(vx[m] * s) for s in SCALES]
        yp = [np.median(vp[c][m]) for c in CONDS[:5]]
        gm = r["history_gain"][BIN_NAMES[b]]
        ax.plot(xf, yp, "o-", color=col[b], ms=3, label=f"{BIN_NAMES[b]} m/s (n {m.sum()}, gamma {gm['gamma']:.2f})")
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color=PS.BASELINE, lw=0.6, ls="--", label="v_p = v_f")
    ax.set_xlabel("fed speed v_f (m/s, history x 0.6 .. 1.4)")
    ax.set_ylabel("plan speed v_p (m/s, median)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title(f"history gain, free rows (gamma all {r['history_gain']['all']['gamma']:.2f})")
    f.tight_layout()
    f.savefig(d / "lon_diag_history_gain.png", dpi=150)
    plt.close(f)
    cells = [q for q in ("red", "lead", "green", "red_all", "lead_all", "green_all") if "b_H" in r["brake"][q]]
    f, ax = plt.subplots(figsize=(4.2, 2.8))
    xx = np.arange(len(cells))
    for j, (k, c) in enumerate((("b_H", PS.PALETTE["blue"]), ("b_F", PS.PALETTE["orange"]))):
        v = np.array([r["brake"][q][k] for q in cells])
        ax.bar(xx + (j - 0.5) * 0.38, v[:, 0], 0.38, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=c, capsize=2, error_kw=dict(lw=0.6),
               label={"b_H": "H: own history", "b_F": "F: flat history"}[k])
    ax.set_xticks(xx, [f"{q}\nn {r['brake'][q]['rows']}\nF/H {r['brake'][q]['ratio'][0]:.2f}" for q in cells], fontsize=6.5)
    PS.zero_line(ax)
    ax.set_ylabel("median brake b = (v_f - v_p) / v_f")
    ax.legend(fontsize=6.5)
    f.tight_layout()
    f.savefig(d / "lon_diag_red_brake.png", dpi=150)
    plt.close(f)


def cmd_speeds(a):
    """The training store's row speeds: navtrain_full shards, op-parity-full train rows (pp_train.split_rows) and all rows."""
    import b1 as B
    from jevdrive.data import splits
    tabs = [B.tab("log", k) for k in range(B.NSH)]
    v = np.concatenate([t["speed"] for t in tabs]).astype(float)
    names = np.concatenate([t["names"] for t in tabs])
    sp = splits.load("navsim/op-parity-full-train")
    tr = sp.mask(np.concatenate([t["log"] for t in tabs]) if sp.unit in ("sequence", "log") else names)
    s0 = np.arange(len(v)) < len(tabs[0]["names"])
    q = lambda x: dict(n=int(len(x)), p50=float(np.quantile(x, 0.5)), p90=float(np.quantile(x, 0.9)), p99=float(np.quantile(x, 0.99)),  # noqa: E731
                       max=float(x.max()), ge16=float((x >= 16).mean()), ge12=float((x >= 12).mean()))
    res = {"train rows (12 shards)": q(v[tr]), "all rows (12 shards)": q(v), "shard 0 (this read)": q(v[s0])}
    _pl.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("cells")
    p.add_argument("--shards", type=int, nargs="+", default=[0]), p.add_argument("--out", required=True)
    p = sp.add_parser("run")
    p.add_argument("--shards", type=int, nargs="+", default=[0]), p.add_argument("--tag", default="RSG-L18-s0")
    p.add_argument("--pool", default="on1_L18RB0"), p.add_argument("--cells", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("speeds")
    p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--cells", required=True), p.add_argument("--run", required=True), p.add_argument("--out", required=True)
    p.add_argument("--figs", required=True), p.add_argument("--tag", default="RSG-L18-s0"), p.add_argument("--speeds", default="")
    a = ap.parse_args()
    dict(cells=cmd_cells, run=cmd_run, report=cmd_report, speeds=cmd_speeds)[a.cmd](a)
