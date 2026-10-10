"""BODY1 progress diagnosis, open-loop part (results/progress_diagnosis.md section 1): is the own plan of the loss arm (P2H10B-F) shorter than
P2H10-F's everywhere, or only where an object or the road edge is near the swept path? No training, no simulator.

  states   own plans of both checkpoints and both seeds on the G3 state sets (bd4_g3.py's selection, unchanged): `hold` = on-log / ot1 / yr1 / bd4
           states of navsim/body1-hold-logs, `navtest` = on-log navtest tokens -> $DATA_DIR/runs/body1/prog/ol_<set>.parquet, one row per state:
           4 s arc length of every plan (and at 1 / 2 / 3 s), the logged arc, the plan difference new - base split along / across the base
           plan, and the proximity of the BASE plan (seed 0) to objects and to the NAVSIM raster edge.          (envs/op-train, one GPU)
  tables   arc ratio new / base by speed bin, proximity group, state family, user class -> results/prog/ol_<set>_*.csv   (CPU)

Proximity groups of a state (thresholds fixed here, before any number was read; the dose-response tables show they do not carry the result):
  contact    the base plan touches a counted object or leaves the raster (min footprint margin < -0.20 m): a hinge is certainly active
  lead       an object at t0 ahead of the front bumper inside the ego's lane width (|lateral centre| < 1.5 m, gap < max(10 m, 3 s x v0))
  near obj   time-matched clearance of the base plan to any object < 1.0 m (sides included)
  near edge  min footprint margin of the base plan on the NAVSIM raster < 0.5 m
  open       none of the four
Ratio = mean arc of the arm / mean arc of the base on the same states; interval from jevdrive.stats.paired(groups=log) on the per-state
difference, divided by the base mean. `seed` rows give the same ratio between the two seeds of the base (the noise floor).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import argparse  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

OUT = B.REPO / "experiments/body1/results/prog"
NEW, REF = ("P2H10B-F-s0", "P2H10B-F-s1"), ("P2H10-F-s0", "P2H10-F-s1")
VB = [(-1, 1, "v < 1"), (1, 3, "1-3"), (3, 6, "3-6"), (6, 10, "6-10"), (10, 99, "> 10")]
GROUPS = ("contact", "lead", "near obj", "near edge", "open")


def knots_arc(P):
    """Plans (..., 8, 3) -> cumulative arc length at the 8 knots (..., 8), from the state's origin."""
    d = np.diff(np.concatenate([np.zeros_like(P[..., :1, :2]), P[..., :2]], -2), axis=-2)
    return np.hypot(d[..., 0], d[..., 1]).cumsum(-1)


def along_cross(Pn, Pb):
    """Difference new - base at the 8 knots in the base plan's heading frame -> (along (n, 8), cross (n, 8)); + along = further ahead."""
    d = Pn[..., :2] - Pb[..., :2]
    c, s = np.cos(Pb[..., 2]), np.sin(Pb[..., 2])
    return c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]


def lead_gap(box, valid, cls, off, v0):
    """Gap (m, front bumper to the nearest face) to the nearest object ahead in the ego's lane at t0, in the state's own frame; inf if none."""
    import sweep as SW
    b, ok = box[:, 0].astype(np.float64), valid[:, 0] & (cls >= 0)
    dy, dp = off[:, :1], off[:, 1:]
    c, s = np.cos(dp), np.sin(dp)
    x, y = c * b[..., 0] + s * (b[..., 1] - dy), -s * b[..., 0] + c * (b[..., 1] - dy)
    gap = x - b[..., 3] / 2 - SW.FRONT
    m = ok & (gap > -0.5) & (np.abs(y) < 1.5) & (gap < np.maximum(10.0, 3.0 * v0)[:, None])
    return np.where(m, gap, np.inf).min(1)


def cmd_states(a):
    import pandas as pd
    import torch
    import bd4_g3 as G
    import sweep as SW
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    tags = list(NEW + REF)
    with Run("body1", f"prog-ol-{a.set}", config=vars(a)) as run:
        if a.set == "hold":
            hold = splits.load(B.HOLD)
            run.use_split(hold)
            L = B.labels()
            tax = np.load(B.root() / "taxonomy" / "tax.npz")
            dirs, rows_of, meta, futs = [], [], [], []
            for f in a.fams:
                for k in a.shards:
                    t = B.tab(f, k)
                    rr = np.flatnonzero(hold.mask(t["log"]))
                    g, off = B.state_index(f, k, t)
                    dirs.append(B.cdir(f, k)), rows_of.append(rr)
                    meta.append(pd.DataFrame(dict(fam=f, name=t["names"][rr], grow=g[rr], dy=off[rr, 0], dpsi=off[rr, 1], log=t["log"][rr], v0=t["speed"][rr])))
                    futs.append(knots_arc(np.nan_to_num(t["fut"][rr].astype(np.float64)))[:, -1] if f == "log" and "fut" in t else np.full(len(rr), np.nan))
            M = pd.concat(meta, ignore_index=True)
            g = M.grow.to_numpy()
            box, val, cls, sdf = L["box"][g], L["valid"][g], L["cls"][g], L["sdf"][g]
            from contact_head import classes
            M["cls"], M["dyaw"], M["arc_log"] = np.array(B.CLS)[classes(tax)[0][g]], np.abs(tax["dyaw"][g]), np.concatenate(futs)
        else:
            run.use_split(splits.load("navsim/navtest"))
            t = dict(np.load(B.cache_root() / "lb_navtest" / "tab.npz"))
            ag = np.load(data_dir() / "runs/op_parity/agent_labels/navtest-k32.npz")
            sd = np.load(data_dir() / "runs/op_probe/labels/navtest.npz")
            assert (ag["tokens"] == t["names"]).all() and (sd["tokens"] == t["names"]).all()
            rr = np.flatnonzero(ag["ok"] & sd["ok"])
            dirs, rows_of = ["lb_navtest"], [rr]
            box, val, cls, sdf = ag["box"][rr], ag["valid"][rr], ag["cls"][rr], sd["sdf"][rr]
            fut = np.nan_to_num(t["fut"][rr].astype(np.float64))
            M = pd.DataFrame(dict(fam="navtest", name=t["names"][rr], dy=0.0, dpsi=0.0, log=t["log"][rr], v0=t["speed"][rr], cls="all",
                                  dyaw=np.abs(np.degrees(np.unwrap(fut[:, :, 2], axis=1)[:, -1])), arc_log=knots_arc(fut)[:, -1]))
        P = G.plans(dirs, rows_of, tags, dev)                                        # (4, n, 8, 3)
        off = M[["dy", "dpsi"]].to_numpy(np.float32)
        for m, tag in enumerate(tags):
            s = knots_arc(P[m].astype(np.float64))
            for j, sec in ((1, 1), (3, 2), (5, 3), (7, 4)):
                M[f"arc{sec}|{tag}"] = s[:, j]
            lab = SW.labels(P[m][:, None], off, box, val, cls, sdf)
            M[f"hit|{tag}"], M[f"clr|{tag}"], M[f"lat|{tag}"] = lab["a_hit"][:, 0], lab["a_clr"][:, 0], lab["a_lat"][:, 0]
            M[f"bm|{tag}"], M[f"bt0|{tag}"] = lab["b_margin"][:, 0], lab["b_t0"][:, 0]
        for i in (0, 1):
            al, cr = along_cross(P[i].astype(np.float64), P[2 + i].astype(np.float64))
            M[f"along4|s{i}"], M[f"cross4|s{i}"], M[f"crossrms|s{i}"] = al[:, -1], cr[:, -1], np.sqrt((cr ** 2).mean(1))
        al, cr = along_cross(P[3].astype(np.float64), P[2].astype(np.float64))        # base seed 1 against base seed 0: the noise floor
        M["along4|seed"], M["cross4|seed"], M["crossrms|seed"] = al[:, -1], cr[:, -1], np.sqrt((cr ** 2).mean(1))
        M["lead_gap"] = lead_gap(box, val, cls, off.astype(np.float64), M.v0.to_numpy(np.float64))
        d = B.root() / "prog"
        d.mkdir(parents=True, exist_ok=True)
        M.to_parquet(d / f"ol_{a.set}.parquet")
        np.save(d / f"ol_{a.set}_plans.npy", P)
        run.info(f"{a.set}: {len(M)} states, {M.log.nunique()} logs -> {d / f'ol_{a.set}.parquet'}")
        run.summary.update(n=len(M), logs=int(M.log.nunique()))
    cmd_tables(a)


def group(M, ref=REF[0]):
    """Proximity group of every state from the base plan of seed 0 (first match in GROUPS order)."""
    contact = M[f"hit|{ref}"].to_numpy() | ((M[f"bm|{ref}"].to_numpy() < -0.20) & ~M[f"bt0|{ref}"].to_numpy())
    lead = np.isfinite(M.lead_gap.to_numpy())
    nobj = M[f"clr|{ref}"].to_numpy() < 1.0
    nedge = M[f"bm|{ref}"].to_numpy() < 0.5
    return np.select([contact, lead, nobj, nedge], GROUPS[:4], "open")


def cmd_tables(a):
    import pandas as pd
    from jevdrive import stats
    M = pd.read_parquet(B.root() / "prog" / f"ol_{a.set}.parquet")
    M["grp"] = group(M)
    M["vbin"] = np.select([(M.v0 > lo) & (M.v0 <= hi) for lo, hi, _ in VB], [n for *_, n in VB], "?")
    logs = M.log.to_numpy()
    pairs = [("s0", NEW[0], REF[0]), ("s1", NEW[1], REF[1]), ("seed", REF[1], REF[0])]

    def rows(name, masks, sec=4):
        out = []
        for lab, m in masks:
            if m.sum() < 20:
                continue
            for pn, x, y in pairs:
                r = stats.paired(M[f"arc{sec}|{x}"].to_numpy()[m], M[f"arc{sec}|{y}"].to_numpy()[m], groups=logs[m])
                b = r["mean_b"]
                out.append(dict(split=name, subset=lab, pair=pn, n=int(m.sum()), logs=int(len(np.unique(logs[m]))), arc_base=b, arc_new=r["mean_a"],
                                ratio=r["mean_a"] / b, lo=1 + r["lo"] / b, hi=1 + r["hi"] / b, diff_m=r["mean"],
                                arc_log=float(np.nanmean(M.arc_log.to_numpy()[m])) if np.isfinite(M.arc_log.to_numpy()[m]).any() else np.nan,
                                cross_rms=float(M[f"crossrms|{pn}"].to_numpy()[m].mean()), along4=float(M[f"along4|{pn}"].to_numpy()[m].mean())))
        return out
    T_ = lambda c, vals: [(str(v), (M[c] == v).to_numpy()) for v in vals]  # noqa: E731
    mov = (M.v0 >= 1).to_numpy()
    R = rows("all", [("all", np.ones(len(M), bool)), ("v >= 1", mov), ("v < 1 (launch)", ~mov), ("> 45 deg", (M.dyaw > 45).to_numpy())])
    R += rows("group", T_("grp", GROUPS)) + rows("speed", T_("vbin", [n for *_, n in VB])) + rows("family", T_("fam", M.fam.unique())) + rows("class", T_("cls", sorted(M.cls.unique())))
    R += rows("group x speed", [(f"{g} | {n}", ((M.grp == g) & (M.vbin == n)).to_numpy()) for g in GROUPS for *_, n in VB])
    R += rows("family x group", [(f"{f} | {g}", ((M.fam == f) & (M.grp == g)).to_numpy()) for f in M.fam.unique() for g in GROUPS])
    clr, bm, lg = M[f"clr|{REF[0]}"].to_numpy(), M[f"bm|{REF[0]}"].to_numpy(), M.lead_gap.to_numpy()
    R += rows("clearance of the base plan (m)", [(f"{lo} .. {hi}", (clr > lo) & (clr <= hi)) for lo, hi in ((-99, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 98), (98, 100))])
    R += rows("raster margin of the base plan (m)", [(f"{lo} .. {hi}", (bm > lo) & (bm <= hi)) for lo, hi in ((-99, -0.2), (-0.2, 0.3), (0.3, 0.5), (0.5, 1), (1, 2), (2, 98))])
    R += rows("lead gap (m)", [(f"{lo} .. {hi}", (lg > lo) & (lg <= hi)) for lo, hi in ((-1, 5), (5, 10), (10, 20), (20, 40), (40, 1e9))] + [("no lead", ~np.isfinite(lg))])
    R += rows("open, arc at 1 s", [("open", (M.grp == "open").to_numpy())], sec=1) + rows("open, arc at 2 s", [("open", (M.grp == "open").to_numpy())], sec=2)
    D = pd.DataFrame(R)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / f"ol_{a.set}_arc.csv", index=False, float_format="%.4f")
    # where the shortening is: share of the pooled arc difference carried by each group
    sh = []
    for pn, x, y in pairs[:2]:
        d = (M[f"arc4|{x}"] - M[f"arc4|{y}"]).to_numpy()
        for g in GROUPS:
            m = (M.grp == g).to_numpy()
            sh.append(dict(pair=pn, group=g, n=int(m.sum()), share_states=float(m.mean()), mean_diff_m=float(d[m].mean()) if m.any() else np.nan, share_of_total_diff=float(d[m].sum() / d.sum())))
    pd.DataFrame(sh).to_csv(OUT / f"ol_{a.set}_share.csv", index=False, float_format="%.4f")
    print(D[D.split.isin(["all", "group", "speed", "family"])].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(sh).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["states", "tables"])
    ap.add_argument("--set", default="hold", choices=["hold", "navtest"])
    ap.add_argument("--fams", nargs="+", default=["log", "ot1", "yr1", "bd4"])
    ap.add_argument("--shards", type=int, nargs="+", default=list(range(B.NSH)))
    a = ap.parse_args()
    {"states": cmd_states, "tables": cmd_tables}[a.cmd](a)
