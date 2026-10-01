"""op-adapt r2: diagnosis of the failed V4 / V6 checks (fc65452:todos/2026-09-29-op-adapt-r2-prereg.md, v5 section).
Read-only on the score tables; writes R/checks/diag_*.parquet|json and figures to experiments/op_adapt_r2/results/diag/.

  v6    offset dev slots: per-slot failure reason of `rej` (start pose off-road / on an actor, later DAC / NC / DDC, with
        the failing time and actor), and figures of 20 failing slots against the NAVSIM map
  v4    x+ slots with the pedestrian visible >= 500 px_eq in the corridor: Top-set composition of the misses, per
        scenario instance, with the pass-1 expert's own speed change dv*
Usage (box): $DATA_DIR/envs/jevdrive/bin/python experiments/op_adapt_r2/archive/op_adapt_r2_diag.py v6|v4 [--tag v4]
"""
import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO))
from experiments.op_adapt_r2.archive import op_adapt_r2_data as C  # noqa: E402
from experiments.op_adapt_r2.archive import op_adapt_score as S  # noqa: E402
from experiments.op_adapt_r2.archive import op_adapt_score_data as SD  # noqa: E402

FIG = REPO / "research" / "results" / "op-adapt-r2" / "diag"
KIND = {S.VEH: "vehicle", S.PED: "pedestrian", S.CYC: "cyclist", S.STATIC: "static"}


def _ctx(dom):
    return SD._CTX.get(dom) or SD._CTX.setdefault(dom, SD.SlotContext(dom))


def _eval(dom, u):
    """Slot, candidate names, rollouts, geometry, map areas and NC details of one slot (visible actors)."""
    s = _ctx(dom).slot(int(u))
    names, (P, _) = _ctx(dom).candidates(int(u), s)
    st = S.rollout(P)
    g = S._geom(st, s.ego, s.n)
    ar = s.mapq.areas(S.to_world(g["corners"], s.pose), S.to_world(g["c"], s.pose), S.to_world(g["p"], s.pose), g["h"] + s.pose[2])
    A = s.actors.subset(s.actors.vis)
    ok, fail, _ = S.nc(g, s.ego, A, s.n)
    return s, list(names), P, st, g, ar, A, ok, fail


def _v6_one(u):
    s, names, P, st, g, ar, A, ok, fail = _eval("off", u)
    j = names.index("rej")
    ov0 = (S.obb_overlap(g["c"][j, 0], g["h"][j, 0], s.ego.hl, s.ego.hw, A.c[0], A.h[0], A.hl, A.hw) & A.valid[0]).any() \
        if A.c.shape[1] else False
    # the true start footprint (the offset pose itself: heading 0 in its own frame), independent of any candidate
    c0 = np.array([s.ego.rc, 0.0])
    start_off = bool(s.mapq.areas(S.to_world(S.box_corners(c0[None], np.zeros(1), s.ego.hl, s.ego.hw), s.pose),
                                  S.to_world(c0[None], s.pose), S.to_world(np.zeros((1, 2)), s.pose),
                                  np.full(1, s.pose[2]))["offroad"][0])
    start_ov = bool((S.obb_overlap(c0, 0.0, s.ego.hl, s.ego.hw, A.c[0], A.h[0], A.hl, A.hw) & A.valid[0]).any()) if A.c.shape[1] else False
    off = ar["offroad"][j]
    dd, D = S.ddc({"c": g["c"][j:j + 1]}, {k: v[j:j + 1] for k, v in ar.items()})
    fk = [KIND[int(k)] for k in A.kind[fail[j]]]
    fv = [float(v) for v in A.speed[0][fail[j]]]
    # first NC failure time of rej
    v = g["v"][j] >= S.V_FAULT
    ctr = s.meta["centre"]
    _, d0 = S.project(np.zeros(2), ctr)
    return {"uid": int(u), "v0": s.v0, "start_offroad": start_off, "start_on_actor": start_ov, "rej_t0_offroad": bool(off[0]),
            "rej_t0_on_actor": bool(ov0), "rej_first_offroad": int(np.argmax(off)) if off.any() else -1,
            "rej_dac": bool(~off.any()), "rej_nc": bool(ok[j]), "rej_ddc": float(dd[0]), "rej_D": float(D[0]),
            "nc_kinds": ",".join(sorted(set(fk))), "nc_speed_max": max(fv) if fv else np.nan,
            "d_centre": float(d0), "any_ok": bool(((ok) & ~ar["offroad"].any(-1)).any()),
            "op_dac": bool(~ar["offroad"][0].any()), "rej_arc4": float(st["s"][j, -1])}


def _draw(ax, s, names, P, g, A, show, title, fail_t=None):
    import shapely
    from matplotlib.patches import Polygon as Poly
    pm = s.mapq
    def ego_xy(geom):
        xy = np.asarray(geom.exterior.coords)
        return S.to_ego(xy, s.pose)
    for geom, col, z in ((pm.drive, "#d9d9d9", 0), (pm.route, "#cfe3f5", 1), (pm.inter, "#f3e3c3", 2)):
        if geom is None:
            continue
        for p in getattr(geom, "geoms", [geom]):
            if isinstance(p, shapely.Polygon):
                ax.add_patch(Poly(ego_xy(p), closed=True, fc=col, ec="none", zorder=z))
    ctr = s.meta.get("centre", s.ref)
    ax.plot(ctr[:, 0], ctr[:, 1], "--", c="#555555", lw=0.8, zorder=3, label="centre line")
    cols = {"rej": "#d62728", "op": "#1f77b4", "hold": "#2ca02c", "brake_hard": "#9467bd", "shift_L": "#8c564b", "shift_R": "#e377c2"}
    for k in show:
        j = names.index(k)
        ax.plot(P[j, :s.n, 0], P[j, :s.n, 1], c=cols.get(k, "k"), lw=1.4, zorder=5, label=k)
        for t in range(0, s.n, 10):
            ax.add_patch(Poly(g["corners"][j, t], closed=True, fc="none", ec=cols.get(k, "k"), lw=0.7, zorder=5))
    tt = [0] + ([fail_t] if fail_t is not None and fail_t > 0 else [])
    for t, ls in zip(tt, ("-", ":")):
        for i in range(A.c.shape[1]):
            if not A.valid[t, i]:
                continue
            cr = S.box_corners(A.c[t, i], A.h[t, i], A.hl[i], A.hw[i])
            ax.add_patch(Poly(cr, closed=True, fc="none", ec={S.PED: "#ff7f0e", S.STATIC: "#7f7f7f"}.get(int(A.kind[i]), "k"),
                              lw=0.9, ls=ls, zorder=6))
    ax.set_title(title, fontsize=7)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def v6(tag: str, split: str = "dev"):
    z = S.load_scores("off")
    t = pd.read_parquet(SD.R("offset", "table.parquet")).set_index("uid").loc[z["uid"]]
    dev = z["uid"][(t.split == split).to_numpy()]
    with Pool(32) as p:
        d = pd.DataFrame(p.map(_v6_one, dev, chunksize=8)).set_index("uid")
    d = d.join(t[["e", "psi", "corner", "cmd", "token"]])
    j = list(z["cands"]).index("rej")
    pos = pd.Series(np.arange(len(z["uid"])), index=z["uid"])[d.index].to_numpy()
    d["rej_ok"] = (z["NC"][pos, j] == 1) & (z["DAC"][pos, j] == 1) & (z["DDC"][pos, j] == 1)
    d["valid"] = z["valid"][pos]
    def reason(r):
        if r.rej_ok:
            return "ok"
        if r.start_offroad:
            return "start off drivable area"
        if r.start_on_actor:
            return "start overlaps an actor"
        if not r.rej_dac:
            return "DAC later"
        if not r.rej_nc:
            return "NC later"
        return "DDC"
    d["reason"] = d.apply(reason, axis=1)
    d.to_parquet(SD.R("checks", f"diag_v6_{split}_{tag}.parquet"))
    if split != "dev":
        print(d.groupby(pd.cut(d.e.abs(), [1.0, 1.25, 1.5, 1.75, 2.0]))[["start_offroad", "start_on_actor", "rej_ok"]].mean().round(3))
        print(d.reason.value_counts(normalize=True).round(3).to_dict())
        return
    feas = ~d.start_offroad & ~d.start_on_actor
    res = {"n": len(d), "rej_ok": float(d.rej_ok.mean()), "reasons": d.reason.value_counts().to_dict(),
           "start_offroad": float(d.start_offroad.mean()), "start_on_actor": float(d.start_on_actor.mean()),
           "feasible_start": int(feas.sum()), "rej_ok_feasible": float(d.rej_ok[feas].mean()),
           "any_candidate_ok": float(d.any_ok.mean()), "by_corner": d.groupby("corner").agg(
               start_offroad=("start_offroad", "mean"), rej_ok=("rej_ok", "mean")).round(3).to_dict(),
           "by_cmd": d.groupby("cmd").rej_ok.mean().round(3).to_dict(),
           "nc_kinds_feasible": Counter(d.nc_kinds[feas & ~d.rej_nc]).most_common(8),
           "first_offroad_feasible": d.rej_first_offroad[feas & ~d.rej_dac].value_counts().sort_index().to_dict()}
    SD._save_json(SD.R("checks", f"diag_v6_{tag}.json"), res)
    print(json.dumps(res, indent=1, default=str))
    # figures: 20 failing slots, 5 per corner, preferring distinct reasons
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    pick = []
    for c in range(4):
        f = d[(d.corner == c) & ~d.rej_ok]
        by = [f[f.reason == r].index.to_numpy() for r in f.reason.unique()]
        k = 0
        while len([u for u in pick if d.corner[u] == c]) < 5 and any(len(b) for b in by):
            b = by[k % len(by)]
            if len(b):
                i = rng.integers(len(b))
                pick.append(int(b[i]))
                by[k % len(by)] = np.delete(b, i)
            k += 1
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(4, 5, figsize=(15, 12.5))
    for ax, u in zip(axs.ravel(), pick):
        s, names, P, st, g, ar, A, ok, fail = _eval("off", u)
        r = d.loc[u]
        ft = r.rej_first_offroad if r.reason == "DAC later" else None
        _draw(ax, s, names, P, g, A, ("rej", "op"), f"{u % 100000} e={r.e:+.1f} psi={r.psi:+.2f} v0={r.v0:.1f}\n{r.reason}"
              + (f" t={ft / 10:.1f}s" if ft else "") + (f" ({r.nc_kinds})" if r.reason == "NC later" else ""), ft)
        ax.set_xlim(-15, 45)
        ax.set_ylim(-15, 15)
    axs[0, 0].legend(fontsize=6, loc="lower left")
    fig.suptitle("op-adapt r2 V6: `rej` (red) and `op` (blue) on 20 failing offset dev slots; grey drivable, blue route lanes, "
                 "tan intersections; boxes every 1 s; actors solid at t = 0, dotted at the failing time", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"v6_fail20_{tag}.png", dpi=130)
    print("fig", FIG / f"v6_fail20_{tag}.png")


def v4(tag: str):
    slow_lat = ("op_slow", "op_stop", "brake_hard", "brake_mild", "shift_L", "shift_R", "shift_L_slow", "shift_R_slow",
                "nudge_L", "op_L", "op_R")
    out = {}
    for dom in ("simC", "simK"):
        z = S.load_scores(dom)
        idx = C.load_index(dom).set_index("uid").loc[z["uid"]]
        names = list(z["cands"])
        sl = [names.index(k) for k in slow_lat]
        big = (idx.sign > 0).to_numpy() & (idx.px_eq >= 500).to_numpy() & idx.ped_corr.to_numpy(bool)
        m = big & z["valid"]
        hit = z["top"][:, sl].any(1)
        d = idx[m].copy()
        d["hit"] = hit[m]
        d["top"] = ["+".join(n for n, x in zip(names, r) if x) for r in z["top"][m]]
        d["hold_nc"] = z["NC"][m, names.index("hold")]
        d["op_nc"] = z["NC"][m, 0]
        d["prog_hold"], d["prog_op"] = z["prog"][m, names.index("hold")], z["prog"][m, 0]
        g = d.groupby("inst").agg(n=("hit", "size"), hit=("hit", "mean"), family=("family", "first"),
                                  dv2=("dv_star_2", "median"), hold_safe=("hold_nc", "mean"))
        out[dom] = {"big": int(big.sum()), "big_valid": int(m.sum()), "big_short_horizon": int((big & (z["h"] < 3.0 - 1e-6)).sum()),
                    "big_maxS0": int((big & (z["S"].max(1) <= 0) & (z["h"] >= 3.0 - 1e-6)).sum()),
                    "slot_rate": float(d.hit.mean()), "instances": len(g), "instance_mean": float(g.hit.mean()),
                    "miss_top_sets": Counter(d.top[~d.hit]).most_common(5),
                    "miss_hold_nc_ok": float(d.hold_nc[~d.hit].mean()), "miss_hold_faster_than_op": float((d.prog_hold > d.prog_op + 1)[~d.hit].mean()),
                    "miss_by_instance": d[~d.hit].inst.value_counts().head(5).to_dict(),
                    "dv2_miss": float(np.nanmedian(d.dv_star_2[~d.hit])), "dv2_hit": float(np.nanmedian(d.dv_star_2[d.hit])),
                    "worst_instances": g.sort_values("hit").head(5).round(3).reset_index().to_dict("records")}
        d.drop(columns=["file"], errors="ignore").to_parquet(SD.R("checks", f"diag_v4_{dom}_{tag}.parquet"))
    SD._save_json(SD.R("checks", f"diag_v4_{tag}.json"), out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("v6", "v4"))
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    v6(a.tag, a.split) if a.cmd == "v6" else v4(a.tag)
