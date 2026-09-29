"""Readouts of the WL-2 representation dry run (todos/2026-09-29-wl-dryrun.md): C1c, C1 brake direction, C2, C3a under the cg
label, and the continuous C4 examinee, with WL-2's definitions (todos/2026-09-29-wl2-prereg.md) on WL-1's 130 eval fork points.
Descriptive only, none of it is a judged result.

  python -m jevdrive.wl_dryrun_report [--n-boot 2000] [--skip-c4]      -> $WL_DRYRUN_OUT or runs/wl/dryrun_results/
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import nq4_w as W
from . import p5_exam as E
from . import wl as WL
from . import wl_c4 as C4
from . import wl_dryrun as D
from . import wl_model as M
from .common import data_dir, get_logger
from .wl_traj import ACTIONS

log = get_logger(__name__)
SEEDS = M.SEEDS
NAMES = ("W1worig", "W1main", "B", "Bi", "T")                # C1 / C1c arms
C23 = ("W1main", "B", "Bi", "T")                             # C2 / C3 arms (7 candidates)
HOLD = {"W1main": "W1holdout", "B": "Bh"}                    # arm -> its C2(b) holdout arm
LABELS = ("cg", "reg")
OUT = Path(os.environ.get("WL_DRYRUN_OUT", data_dir() / "runs" / "wl" / "dryrun_results"))
ARM_DIR = {"W1main": ("model", "main"), "W1worig": ("model", "worig"), "W1holdout": ("model", "holdout"),
           "B": ("dryrun", "B"), "Bh": ("dryrun", "Bh"), "Bi": ("dryrun", "Bi"), "T": ("dryrun", "T")}


def arm_dir(name: str, seed: int) -> Path | None:
    kind, a = ARM_DIR[name]
    fs = sorted((data_dir() / "runs" / "wl" / kind / a / f"seed{seed}").glob("*/model.pt"))
    return fs[-1].parent if fs else None


_CACHE = {}


def load(name: str, seed: int):
    if (name, seed) not in _CACHE:
        d = arm_dir(name, seed)
        _CACHE[name, seed] = None if d is None else (np.load(d / "preds.npz"), pd.read_parquet(d / "forks.parquet"), d)
    return _CACHE[name, seed]


def truth() -> pd.DataFrame:
    t = M._truth()
    hit = t.collision.astype(float).where(t.collision.notna())
    t["cg"] = ((hit > 0) | (t.gap_min_m < 2.0)).astype(float).where(t.unsafe.notna())
    t["reg"] = t.unsafe
    return t


def boot_multi(df: pd.DataFrame, fns: dict, n: int, seed: int = 0) -> dict:
    """Route (base_id) bootstrap of several statistics of one frame on the same resamples."""
    rng = np.random.RandomState(seed)
    keys = df.base_id.unique()
    by = {k: g for k, g in df.groupby("base_id")}
    xs = {k: [] for k in fns}
    for _ in range(n):
        x = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        for k, f in fns.items():
            xs[k].append(f(x))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in xs.items()}


def boot_arms(dfs: dict, fn, n: int, seed: int = 0) -> dict:
    """dfs: name -> list of per-seed frames (all with base_id). fn(dict name -> seed-mean stat) is evaluated on shared route
    resamples; returns the percentile CI of every key of fn's output."""
    rng = np.random.RandomState(seed)
    keys = np.array(sorted(set.union(*[set(f.base_id) for v in dfs.values() for f in v])))
    by = {nm: [{k: g for k, g in f.groupby("base_id")} for f in v] for nm, v in dfs.items()}
    xs = {}
    for _ in range(n):
        pick = rng.choice(keys, len(keys))
        vals = {nm: [pd.concat([m[k] for k in pick if k in m]) for m in ms] for nm, ms in by.items()}
        for k, v in fn(vals).items():
            xs.setdefault(k, []).append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in xs.items()}


# ================================================================ C1 / C1c

def c1_seed(name: str, seed: int, key: pd.DataFrame, exit_pairs: set):
    q, fa, _ = load(name, seed)
    n, A = len(fa), len(ACTIONS)
    d = pd.DataFrame({"fork_id": np.repeat(fa.fork_id.to_numpy(), A), "action": np.tile(ACTIONS, n),
                      "d10": q["p_d_front"][:, :, 9].ravel(), "occ10": q["p_occ"][:, :, 9].ravel()})
    e = d.join(key, on=["fork_id", "action"])
    e = e[(e.split == "eval") & ~e.fork_id.isin(exit_pairs)]
    pv = lambda c: e.pivot_table(index="fork_id", columns="action", values=c, dropna=False)
    d10, dtrue, occp, occt = pv("d10"), pv("d_front"), pv("occ10"), pv("occ")
    c = pd.DataFrame({"brake_gt_hold": d10.brake_hard > d10.hold,
                      "err": ((d10.brake_hard - d10.hold) - (dtrue.brake_hard - dtrue.hold)).abs()}).join(
        e.groupby("fork_id")[["base_id", "cls", "world"]].first())
    for sd in ("L", "R"):
        a = f"shift_{sd}"
        have = occt[a].notna() & occt.op.notna()
        c[f"ok_{sd}"] = ((occp[a] < occp.op) == (occt[a] < occt.op)).where(have)
        c[f"S_{sd}"] = (occt.op > 0.5) & have
        c[f"score_{sd}"] = occp[a]
        c[f"still_{sd}"] = occt[a] > 0.5
    return c.reset_index()


def _stat_c1(x: pd.DataFrame) -> dict:
    S = pd.concat([pd.DataFrame({"score": x[f"score_{s}"][x[f"S_{s}"]], "still": x[f"still_{s}"][x[f"S_{s}"]]}) for s in "LR"])
    return {"brake_gt_hold": x.brake_gt_hold.mean(), "median_abs_err_m": x.err.median(),
            "agree_all": pd.concat([x.ok_L, x.ok_R]).mean(),
            "agree_S": pd.concat([x.ok_L[x.S_L], x.ok_R[x.S_R]]).mean(),
            "auc_S": M._auc(S.still.astype(bool), S.score) if len(S) else np.nan}


def s_counts(x: pd.DataFrame) -> dict:
    S = pd.concat([x.loc[x[f"S_{s}"], ["base_id", f"still_{s}"]].rename(columns={f"still_{s}": "still"}) for s in "LR"])
    return {"n_S": len(S), "n_still": int(S.still.sum()), "n_left": int((~S.still.astype(bool)).sum()), "routes_S": int(S.base_id.nunique())}


def readout_c1(n_boot: int) -> tuple[pd.DataFrame, dict]:
    tr = truth()
    key = tr.set_index(["fork_id", "action"])
    exit_pairs = WL.pair_dropped_forks()
    pts = {nm: [c1_seed(nm, s, key, exit_pairs) for s in SEEDS if load(nm, s) is not None] for nm in NAMES if load(nm, SEEDS[0]) is not None}
    stats = {nm: {k: float(np.nanmean([_stat_c1(c)[k] for c in cs])) for k in _stat_c1(cs[0])} for nm, cs in pts.items()}
    per_seed = {nm: [_stat_c1(c)["auc_S"] for c in cs] for nm, cs in pts.items()}

    def fn(v):
        m = {nm: {k: np.nanmean([_stat_c1(c)[k] for c in cs]) for k in ("auc_S", "agree_S", "agree_all", "brake_gt_hold")} for nm, cs in v.items()}
        out = {f"{nm}|{k}": x for nm, d in m.items() for k, x in d.items()}
        for nm in m:
            for ref in ("W1worig", "W1main"):
                if nm != ref and ref in m:
                    out[f"{nm}-{ref}|auc_S"] = m[nm]["auc_S"] - m[ref]["auc_S"]
                    out[f"{nm}-{ref}|agree_S"] = m[nm]["agree_S"] - m[ref]["agree_S"]
        return out
    ci = boot_arms(pts, fn, n_boot)
    rows = []
    for nm, cs in pts.items():
        sc = s_counts(cs[0])
        r = {"arm": nm, "seeds": len(cs), **stats[nm], **sc, "auc_S_per_seed": [round(float(x), 3) for x in per_seed[nm]],
             "auc_S_ci": ci[f"{nm}|auc_S"], "agree_S_ci": ci[f"{nm}|agree_S"], "agree_all_ci": ci[f"{nm}|agree_all"],
             "brake_gt_hold_ci": ci[f"{nm}|brake_gt_hold"]}
        for ref in ("W1worig", "W1main"):
            if f"{nm}-{ref}|auc_S" in ci:
                r[f"auc_S_minus_{ref}"] = stats[nm]["auc_S"] - stats[ref]["auc_S"]
                r[f"auc_S_minus_{ref}_ci"] = ci[f"{nm}-{ref}|auc_S"]
                r[f"agree_S_minus_{ref}_ci"] = ci[f"{nm}-{ref}|agree_S"]
        rows.append(r)
    return pd.DataFrame(rows), {"key": key, "pts": pts, "exit_pairs": exit_pairs}


def ego_error(key: pd.DataFrame, exit_pairs: set) -> pd.DataFrame:
    """B / Bi / T: the predicted e_y at 2 s against the branch's real e_y, on the shift and op candidates of the eval fork points;
    the persistence baseline is the branch's own e_y at the fork tick."""
    ix = pd.read_parquet(D.WL_INDEX, columns=["route_id", "tick"])
    ok = np.load(M.pdir("z_ok.npy"))
    ey = np.load(D.ddir("ey_idx.npy"))[:, 0][ok]
    ix = ix[ok].assign(ey=ey).set_index(["route_id", "tick"]).ey
    runs = pd.read_parquet(WL.rundir("forks.parquet"))[["fork_id", "action", "route_id", "fork_tick"]]
    runs = runs.assign(ey0=ix.reindex(list(zip(runs.route_id, runs.fork_tick))).to_numpy(),
                       ey10=ix.reindex(list(zip(runs.route_id, runs.fork_tick + W.FUT * W.TICKS))).to_numpy())
    rows = []
    for nm in ("B", "Bi", "T"):
        if load(nm, SEEDS[0]) is None:
            continue
        hat = []
        for s in SEEDS:
            q, fa, d = load(nm, s)
            ck = __import__("torch").load(d / "model.pt", map_location="cpu")
            e_hat = q["z10"][..., -D.S_DIM].astype(np.float32) * float(ck["sd"][-D.S_DIM]) + float(ck["mu"][-D.S_DIM])
            hat.append(pd.DataFrame({"fork_id": np.repeat(fa.fork_id.to_numpy(), len(ACTIONS)), "action": np.tile(ACTIONS, len(fa)), "ey_hat": e_hat.ravel()}))
        h = pd.concat(hat).groupby(["fork_id", "action"]).ey_hat.mean().reset_index()
        e = h.merge(runs, on=["fork_id", "action"]).join(key[["split"]], on=["fork_id", "action"])
        e = e[(e.split == "eval") & ~e.fork_id.isin(exit_pairs) & e.ey10.notna() & e.ey0.notna()]
        for a in ("op", "shift_L", "shift_R"):
            x = e[e.action == a]
            rows.append({"arm": nm, "action": a, "n": len(x), "med_abs_err_m": float((x.ey_hat - x.ey10).abs().median()),
                         "persistence_med_abs_err_m": float((x.ey0 - x.ey10).abs().median()), "true_ey10_median_m": float(x.ey10.median()),
                         "pred_ey10_median_m": float(x.ey_hat.median()), "sign_agree": float((np.sign(x.ey_hat) == np.sign(x.ey10)).mean())})
    return pd.DataFrame(rows)


# ================================================================ critics, C2, C3a

def critic_frame(name: str, label: str, holdout: bool = False, seeds=SEEDS) -> pd.DataFrame:
    """Per (fork_id, action): p_learn / p_q averaged over the seeds' critics fitted on that seed's train-split fork points."""
    tr = truth().set_index(["fork_id", "action"])
    pl, pq, pg = [], [], []
    for s in seeds:
        q, fa, _ = load(name, s)
        n, A = q["z0"].shape[0], len(ACTIONS)
        z0 = np.repeat(q["z0"].astype(np.float32)[:, None], A, 1)
        Xl = np.concatenate([z0, q["z5"].astype(np.float32), q["z10"].astype(np.float32)], -1).reshape(n * A, -1)
        Xq = np.concatenate([z0, q["cmds"].reshape(n, A, -1).astype(np.float32)], -1).reshape(n * A, -1)
        lab = pd.DataFrame({"fork_id": np.repeat(fa.fork_id.to_numpy(), A), "action": np.tile(ACTIONS, n)}).join(tr[[label, "split"]], on=["fork_id", "action"])
        trn = (lab.split == "train").to_numpy() & lab[label].notna().to_numpy()
        if holdout:
            trn &= ~lab.action.isin(M.HOLD_OUT).to_numpy()
        y = lab[label].fillna(0).astype(float).to_numpy()
        pl.append(M._mlp_fit(Xl[trn], y[trn], s, Xl))
        pq.append(M._mlp_fit(Xq[trn], y[trn], s, Xq))
        pg.append((q["p_v"] * W.DT).sum(2).ravel())
    return lab[["fork_id", "action"]].assign(p_learn=np.mean(pl, 0), p_q=np.mean(pq, 0), prog=np.mean(pg, 0))


def _pairwise(x):
    ok = tot = 0
    for _, g in x.groupby("fork_id"):
        u, s = g[g.y.astype(bool)].p_learn.to_numpy(), g[~g.y.astype(bool)].p_learn.to_numpy()
        ok += (u[:, None] > s[None]).sum()
        tot += len(u) * len(s)
    return ok / max(tot, 1)


def evalset(cf: pd.DataFrame, label: str, tr: pd.DataFrame, exit_pairs: set) -> pd.DataFrame:
    k = tr.set_index(["fork_id", "action"])
    e = cf.join(k[[label, "split", "base_id", "cls", "world", "travel_m"]], on=["fork_id", "action"]).rename(columns={label: "y"})
    return e[(e.split == "eval") & e.y.notna() & ~e.fork_id.isin(exit_pairs)].reset_index(drop=True)


def select(e: pd.DataFrame) -> pd.DataFrame:
    sel = []
    for fid, g in e.groupby("fork_id"):
        ok = g[g.p_learn < M.UNSAFE_P]
        pick = ok.loc[ok.prog.idxmax()] if len(ok) else g.loc[g.p_learn.idxmin()]
        op = g[g.action == "op"]
        if not len(op):
            continue
        orc = g.sort_values(["y", "travel_m"], ascending=[True, False]).iloc[0]
        sel.append({"fork_id": fid, "base_id": g.base_id.iloc[0], "world": g.world.iloc[0], "cls": g.cls.iloc[0], "pick": pick.action,
                    "U_pick": float(pick.y), "U_op": float(op.y.iloc[0]), "U_or": float(orc.y)})
    return pd.DataFrame(sel)


def _H(x):
    den = x.U_op.mean() - x.U_or.mean()
    return (x.U_op.mean() - x.U_pick.mean()) / den if den > 1e-9 else np.nan


def c2_c3(name: str, label: str, tr: pd.DataFrame, exit_pairs: set, n_boot: int, hold_name: str | None) -> dict:
    cf = critic_frame(name, label)
    e = evalset(cf, label, tr, exit_pairs)
    auc = lambda x: M._auc(x.y, x.p_learn)
    diff = lambda x: M._auc(x.y, x.p_learn) - M._auc(x.y, x.p_q)
    ci = boot_multi(e, {"auc": auc, "diff": diff}, n_boot)
    r = {"arm": name, "label": label, "n_rows": len(e), "n_unsafe": int(e.y.sum()), "auc_learn": auc(e), "auc_learn_ci": ci["auc"],
         "auc_q": M._auc(e.y, e.p_q), "pairwise": _pairwise(e), "diff_learn_q": diff(e), "diff_ci": ci["diff"]}
    if hold_name:
        h = evalset(critic_frame(hold_name, label, holdout=True), label, tr, exit_pairs)
        h = h[h.action.isin(M.HOLD_OUT)]
        c = boot_multi(h, {"diff": diff}, n_boot)["diff"]
        r.update({"holdout_n": len(h), "holdout_auc_learn": auc(h), "holdout_auc_q": M._auc(h.y, h.p_q), "holdout_diff": diff(h), "holdout_diff_ci": c})
    sel = select(e)
    plus = sel[sel.world == "plus"]
    hci = boot_multi(plus, {"H": _H, "gain": lambda x: x.U_op.mean() - x.U_pick.mean()}, n_boot)
    solv = plus[(plus.U_op > 0.5) & (plus.U_or < 0.5)]
    r.update({"n_plus": len(plus), "U_op": plus.U_op.mean(), "U_sel": plus.U_pick.mean(), "U_or": plus.U_or.mean(), "H": _H(plus), "H_ci": hci["H"],
              "gain_ci": hci["gain"], "n_solvable": len(solv), "routes_solvable": int(solv.base_id.nunique()),
              "rescued": int((solv.U_pick < 0.5).sum()), "harmed": int(((plus.U_op < 0.5) & (plus.U_pick > 0.5)).sum()),
              "n_op_safe": int((plus.U_op < 0.5).sum()), "picks": sel.pick.value_counts().to_dict()})
    return r


# ================================================================ C4, continuous examinee

def c4_hold_risk(names: tuple, n_boot: int = 0) -> dict:
    import torch
    obs, null, pairs = C4.exam_tables()
    frames = sorted(set(obs.fn_plus) | set(obs.fn_minus) | set(null.fn_plus) | set(null.fn_null))
    wm = pd.read_parquet(W.wdir("meta.parquet"))
    wz = np.load(W.wdir("z.npy"), mmap_mode="r")
    key = pd.Series(np.arange(len(wm)), index=wm.set.astype(str) + "|" + wm.frame_name)
    rows_of, ok = {}, []
    for f in frames:
        i = key.get(f"{C4.SET}|{f}")
        if i is None or i < W.HIST - 1:
            continue
        h = np.arange(i - (W.HIST - 1), i + 1)
        if (wm.adir.to_numpy()[h] == wm.adir.to_numpy()[i]).all() and (np.diff(wm.frame.to_numpy()[h]) == W.TICKS).all():
            rows_of[f] = (int(i), h)
            ok.append(f)
    fi = C4.frame_inputs(ok)
    hold = ACTIONS.index("hold")
    cmds = C4.candidates_for(fi)[:, hold: hold + 1]
    ey_wm = np.load(D.ddir("ey_wm.npy"))
    wmc = wm.assign(a_prev=wm.a_prev.fillna(0).clip(-15, 15), w_prev=wm.w_prev.fillna(0).clip(-15, 15), v=wm.v.fillna(0))
    S_wm = D.ego_cols(wmc, ey_wm)

    class Ext:
        def __init__(s, base, S):
            s.base, s.S, s.shape = base, S, (base.shape[0], base.shape[1] + S.shape[1])

        def __getitem__(s, i):
            return np.concatenate([np.asarray(s.base[i]), s.S[i].astype(np.float16)], 1)
    tr = truth()
    out = {}
    risk = {}
    for nm in names:
        ext = nm not in ("W1main", "W1worig")
        prep = C4.prepare_rows(ok, rows_of, wm, Ext(wz, S_wm) if ext else wz)
        meta, z, anchors = prep
        ps = []
        for s in SEEDS:
            d = arm_dir(nm, s)
            ck = torch.load(d / "model.pt", map_location="cuda")
            data = D.Data(meta, z, np.arange(min(len(z) - W.WIN, 4096)))
            data.mu, data.sd = ck["mu"].cuda(), ck["sd"].cuda()
            for lo in range(0, len(z), 8192):
                data.Z[lo: lo + 8192] = ((torch.as_tensor(z[lo: lo + 8192], device="cuda").float() - data.mu) / data.sd).bfloat16()
            model = M.build(z.shape[1]).cuda()
            model.load_state_dict(ck["state"])
            starts = torch.as_tensor(anchors - (W.HIST - 1), device="cuda")
            pred = M.predict(model, data, starts, torch.as_tensor(cmds[:, 0], device="cuda"), src=1)
            z0, z5, z10 = data.Z[starts + W.HIST - 1].float(), pred[:, 4], pred[:, 9]
            # the seed's C-learn fitted on that seed's train-split fork points (cg label), read on the exam rollouts
            q, fa, _ = load(nm, s)
            n, A = q["z0"].shape[0], len(ACTIONS)
            Xl = np.concatenate([np.repeat(q["z0"].astype(np.float32)[:, None], A, 1), q["z5"].astype(np.float32), q["z10"].astype(np.float32)], -1).reshape(n * A, -1)
            lab = pd.DataFrame({"fork_id": np.repeat(fa.fork_id.to_numpy(), A), "action": np.tile(ACTIONS, n)}).join(
                tr.set_index(["fork_id", "action"])[["cg", "split"]], on=["fork_id", "action"])
            trn = (lab.split == "train").to_numpy() & lab.cg.notna().to_numpy()
            Xe = torch.cat([z0, z5, z10], -1).cpu().numpy().astype(np.float32)
            ps.append(M._mlp_fit(Xl[trn], lab.cg.fillna(0).astype(float).to_numpy()[trn], s, Xe))
            del data, model, pred
            torch.cuda.empty_cache()
        risk[nm] = {f"s{s}": pd.Series(-p, index=ok) for s, p in zip(SEEDS, ps)}
        risk[nm]["mean"] = pd.Series(-np.mean(ps, 0), index=ok)
    have = set(ok)
    ex = {}
    cols = []
    for tag, df, a, b in (("obs", obs, "fn_plus", "fn_minus"), ("null", null, "fn_plus", "fn_null")):
        df = df[df[a].isin(have) & df[b].isin(have)].copy()
        for nm in names:
            for k, sr in risk[nm].items():
                c = f"{nm} risk {k}"
                df[c] = sr[df[a]].to_numpy() - sr[df[b]].to_numpy()
                if tag == "obs":
                    cols.append(c)
        ex[tag] = df
    rows = []
    for cls, fams in (("ped", WL.PED_FAM), ("cutin", WL.CUTIN_FAM)):
        o, n_, pr = ex["obs"][ex["obs"].family.isin(fams)], ex["null"][ex["null"].family.isin(fams)], pairs[pairs.family.isin(fams)]
        fl = E.exam(o, n_, pr, cols)["flips"]
        rows.append(fl[fl.scope == "pooled"].assign(cls=cls, n_obs=len(o), n_null=len(n_)))
    fl = pd.concat(rows)
    fl["margin_ok"] = fl.flip_lo > fl.false_flip_null_oos + C4.FLIP_PED_MARGIN
    return {"flips": fl[["cls", "examinee", "flip_rate", "flip_lo", "flip_hi", "false_flip_null_oos", "tau_model", "n_reactive", "margin_ok", "n_obs", "n_null"]],
            "n_frames": len(ok)}


# ================================================================ summary

def _f(x, k=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{k}f}" if isinstance(x, (int, float, np.floating)) else str(x))


def _ci(c, k=3):
    return f"[{_f(c[0], k)}, {_f(c[1], k)}]"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-c4", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    c1, ctx = readout_c1(a.n_boot)
    c1.to_csv(OUT / "c1.csv", index=False, float_format="%.4f")
    ee = ego_error(ctx["key"], ctx["exit_pairs"])
    ee.to_csv(OUT / "ego_error.csv", index=False, float_format="%.4f")
    log.info("C1 / C1c done %.0f s", time.time() - t0)
    tr = truth()
    rows = []
    for nm in C23:
        if load(nm, SEEDS[0]) is None:
            continue
        for lb in LABELS:
            hn = HOLD.get(nm)
            rows.append(c2_c3(nm, lb, tr, ctx["exit_pairs"], min(a.n_boot, 1000), hn if hn and load(hn, SEEDS[0]) is not None else None))
            log.info("C2/C3 %s %s done %.0f s", nm, lb, time.time() - t0)
            pd.DataFrame(rows).to_csv(OUT / "c2_c3.csv", index=False, float_format="%.4f")
    c23 = pd.DataFrame(rows)
    c4 = None
    if not a.skip_c4:
        c4 = c4_hold_risk(tuple(n for n in ("W1main", "B") if load(n, SEEDS[0]) is not None))
        c4["flips"].to_csv(OUT / "c4_flips.csv", index=False, float_format="%.4f")
        log.info("C4 done %.0f s", time.time() - t0)
    (OUT / "results.json").write_text(json.dumps({"c1": c1.to_dict("records"), "ego_error": ee.to_dict("records"), "c2_c3": c23.to_dict("records"),
                                                  "c4": None if c4 is None else c4["flips"].to_dict("records")}, indent=1, default=float))
    log.info("all done %.0f s -> %s", time.time() - t0, OUT)


if __name__ == "__main__":
    main()
