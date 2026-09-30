"""WL-2 readouts, exactly as registered (todos/2026-09-29-wl2-prereg.md, "判据"): C1a-c, C2, C3a-c, C4, the arm comparisons with
route-bootstrap CIs, the slow-shift analysis with the realised offset, on the WL-2 eval fork points (dataset 2; WL-1's
130 eval fork points, dataset 1, as the second, descriptive eval set).

The same code runs on WL-1 data and predictions (`--validate-wl1`): dataset 1, 7 candidates, the WL-1 dry-run arms; the numbers must
reproduce research/results/wl-dryrun/ (`--validate-wl1` compares them).

  python -m jevdrive.wl2_report [--n-boot 2000] [--skip-c4] [--out DIR]
  python -m jevdrive.wl2_report --validate-wl1
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("WL_NAME", "wl2")
from . import nq4_w as W  # noqa: E402
from . import p5_exam as E  # noqa: E402
from . import wl as WL  # noqa: E402
from . import wl2_model as M2  # noqa: E402
from . import wl_c4 as C4  # noqa: E402
from . import wl_model as M  # noqa: E402
from .common import data_dir, get_logger  # noqa: E402
from .wl_traj import ACTIONS, ACTIONS11  # noqa: E402

log = get_logger(__name__)
DD = data_dir()
GID = M2.GID
LABELS = ("cg", "reg")
UNSAFE_P = M.UNSAFE_P
SHIFT_SIDES = {"L": "shift_L", "R": "shift_R", "Ls": "shift_L_slow", "Rs": "shift_R_slow", "nL": "nudge_L"}
MAIN_ARMS = ("vrep", "A", "B", "W", "Bs", "T")               # C1 / C1c arms
C2_ARMS = ("vrep", "A", "B", "Bs", "T", "Bh", "Bhc")
HOLD_ARM = {"B": "Bh", "A": None}
OUT = Path(os.environ.get("WL2_RESULTS", DD / "runs" / "wl2" / "results"))


# ================================================================ labels and truth

def label_cg(t: pd.DataFrame) -> pd.Series:
    """WL-2's main label: collision or in-lane gap < 2 m (the registered unsafe without the TTC < 1 s term); NaN where unsafe is."""
    hit = t.collision.astype(float).where(t.collision.notna())
    return ((hit > 0) | (t.gap_min_m < 2.0)).astype(float).where(t.unsafe.notna())


def truth(ds: int, P: Path, R: Path) -> pd.DataFrame:
    """Per fork run of a dataset: outcome, labels (cg main, reg = the registered unsafe), the branch's own labels at fork tick + 2 s."""
    runs = pd.read_parquet(R / "forks.parquet")
    o = pd.read_parquet(P / "outcomes.parquet")
    t = pd.read_parquet(P / "index.parquet", columns=["route_id", "tick", "d_front", "occ", "ped"])
    t2 = runs[["route_id", "fork_tick"]].assign(tick=runs.fork_tick + W.FUT * W.TICKS).merge(t, on=["route_id", "tick"], how="left")
    t = runs.merge(o, on="route_id", how="left").merge(t2[["route_id", "d_front", "occ", "ped"]], on="route_id", how="left")
    for c in ("occ", "ped", "unsafe"):
        t[c] = t[c].astype(float)
    t["cg"], t["reg"] = label_cg(t), t.unsafe
    if "unsafe_cg" in t:
        ok = t.unsafe_cg.notna()
        assert (t.unsafe_cg[ok].astype(float) == t.cg[ok]).all(), "outcome.unsafe_cg differs from the cg label"
    t["gid"] = ds * GID + t.fork_id
    t["ds"] = ds
    if "seed" not in t:
        t["seed"] = 0
    return t


def pair_exit(R: Path, ds: int) -> set:
    f = R / "drops.json"
    return {ds * GID + int(i) for i in json.loads(f.read_text())["pair_dropped_fork_ids"]} if f.exists() else set()


def nonped_both2(t: pd.DataFrame) -> set:
    """Amendment (b) as WL.nonped_both, but per seed (WL-2 has two seeds of the same base route and k)."""
    x = t[(t.cls == "ped") & t.action.isin(["hold", "op"])].copy()
    ty = x.collision_types.map(lambda v: json.loads(v) if isinstance(v, str) else [])
    x["nonped"] = ty.map(lambda v: any(not str(c).startswith("walker.") for c in v))
    w = x.groupby(["base_id", "seed", "k_name", "world"]).nonped.any().unstack("world")
    if not {"plus", "minus"} <= set(w):
        return set()
    both = w[w.plus.fillna(False).astype(bool) & w.minus.fillna(False).astype(bool)].index
    return set(x[x.set_index(["base_id", "seed", "k_name"]).index.isin(both)].gid.unique())


# ================================================================ arm loading

class Store:
    """arm name -> seed -> run dir; preds and fork tables loaded lazily. `actions` follows the arm's preds (7 or 11 candidates)."""

    def __init__(self, dirs: dict, ds_default=2):
        self.dirs, self.cache = dirs, {}

    def seeds(self, arm):
        return sorted(self.dirs.get(arm, {}))

    def has(self, arm):
        return bool(self.dirs.get(arm))

    def load(self, arm, seed):
        if (arm, seed) not in self.cache:
            d = self.dirs[arm][seed]
            q = np.load(d / "preds.npz")
            fa = pd.read_parquet(d / "forks.parquet")
            q = {k: q[k] for k in q.files}
            gid = q.pop("gid") if "gid" in q else q.pop("fork_id") + GID          # WL-1 dry-run preds: dataset 1 only
            if "gid" not in fa:
                fa["gid"] = fa.fork_id + GID
            A = q["p_occ"].shape[1]
            actions = ACTIONS11 if A == 11 else ACTIONS
            self.cache[arm, seed] = (gid, q, fa, actions, d)
        return self.cache[arm, seed]


def wl2_store() -> Store:
    dirs = {}
    for arm in list(M2.SEEDS) + ["vrep"]:
        for s in range(5):
            d = M2.latest(arm, s)
            if d is not None:
                dirs.setdefault(arm, {})[s] = d
    return Store(dirs)


def wl1_store() -> Store:
    """The WL-1 dry-run arms (research/results/wl-dryrun): W1worig, W1main, B, Bi, T, and the two holdout arms."""
    R1 = DD / "runs" / "wl"
    spec = {"W1worig": ("model", "worig"), "W1main": ("model", "main"), "W1holdout": ("model", "holdout"), "B": ("dryrun", "B"),
            "Bh": ("dryrun", "Bh"), "Bi": ("dryrun", "Bi"), "T": ("dryrun", "T")}
    dirs = {}
    for nm, (kind, a) in spec.items():
        for s in range(3):
            fs = sorted((R1 / kind / a / f"seed{s}").glob("*/model.pt"))
            if fs:
                dirs.setdefault(nm, {})[s] = fs[-1].parent
    return Store(dirs)


# ================================================================ bootstrap

def boot_multi(df, fns: dict, n: int, seed: int = 0) -> dict:
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
    """dfs: name -> list of per-seed frames (all with base_id). fn(dict name -> list of resampled frames) -> {key: value},
    evaluated on shared route resamples; returns the percentile CI of every key."""
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

def c1_seed(store: Store, arm: str, seed: int, key: pd.DataFrame, ds: int, exit_pairs: set) -> pd.DataFrame:
    """Per eval fork point of the dataset: brake / hold direction, the shift occupancy agreement and, per shift-type candidate, the
    S membership (op branch's lane occupied at 2 s and both branches labelled), the score (rollout `occ` probe at step 10) and the
    branch's true occupancy at 2 s."""
    gid, q, fa, actions, _ = store.load(arm, seed)
    n, A = len(gid), len(actions)
    d = pd.DataFrame({"gid": np.repeat(gid, A), "action": np.tile(actions, n), "d10": q["p_d_front"][:, :, 9].ravel(),
                      "occ10": q["p_occ"][:, :, 9].ravel()})
    e = d.join(key, on=["gid", "action"])
    e = e[(e.split == "eval") & (e.ds == ds) & ~e.gid.isin(exit_pairs)]
    pv = lambda c: e.pivot_table(index="gid", columns="action", values=c, dropna=False)
    d10, dtrue, occp, occt = pv("d10"), pv("d_front"), pv("occ10"), pv("occ")
    c = pd.DataFrame({"brake_gt_hold": d10.brake_hard > d10.hold,
                      "err": ((d10.brake_hard - d10.hold) - (dtrue.brake_hard - dtrue.hold)).abs()}).join(
        e.groupby("gid")[["base_id", "cls", "world", "k_name"]].first())
    for sd, a in SHIFT_SIDES.items():
        if a not in occt:
            continue
        have = occt[a].notna() & occt.op.notna()
        c[f"ok_{sd}"] = ((occp[a] < occp.op) == (occt[a] < occt.op)).where(have)
        c[f"S_{sd}"] = (occt.op > 0.5) & have
        c[f"score_{sd}"] = occp[a]
        c[f"still_{sd}"] = occt[a] > 0.5
    return c.reset_index()


def _S(x: pd.DataFrame, sides: str) -> pd.DataFrame:
    parts = [pd.DataFrame({"score": x[f"score_{s}"][x[f"S_{s}"]], "still": x[f"still_{s}"][x[f"S_{s}"]], "base_id": x.base_id[x[f"S_{s}"]]})
             for s in ((sides,) if sides in SHIFT_SIDES else tuple(sides.split(","))) if f"S_{s}" in x]
    return pd.concat(parts) if parts else pd.DataFrame({"score": [], "still": [], "base_id": []})


def stat_c1(x: pd.DataFrame) -> dict:
    S = _S(x, "L,R")
    both = lambda col, sel=None: pd.concat([(x[f"{col}_{s}"][x[f"S_{s}"]] if sel else x[f"{col}_{s}"]) for s in "LR"])
    out = {"brake_gt_hold": x.brake_gt_hold.mean(), "median_abs_err_m": x.err.median(), "agree_all": both("ok").mean(),
           "agree_S": both("ok", True).mean(), "auc_S": M._auc(S.still.astype(bool), S.score) if len(S) else np.nan}
    for tag, sides in (("slow", "Ls,Rs"), ("nudge", "nL")):
        Sx = _S(x, sides)
        out[f"auc_S_{tag}"] = M._auc(Sx.still.astype(bool), Sx.score) if len(Sx) and 0 < Sx.still.sum() < len(Sx) else np.nan
    return out


def s_counts(x: pd.DataFrame) -> dict:
    S = _S(x, "L,R")
    return {"n_S": len(S), "n_still": int(S.still.sum()), "n_left": int((~S.still.astype(bool)).sum()), "routes_S": int(S.base_id.nunique())}


def readout_c1(store: Store, key, ds: int, exit_pairs: set, arms, n_boot: int, pairs=()) -> pd.DataFrame:
    pts = {a: [c1_seed(store, a, s, key, ds, exit_pairs) for s in store.seeds(a)] for a in arms if store.has(a)}
    keys = ("brake_gt_hold", "median_abs_err_m", "agree_all", "agree_S", "auc_S", "auc_S_slow", "auc_S_nudge")
    seed_stats = {a: [stat_c1(c) for c in cs] for a, cs in pts.items()}
    mean = {a: {k: float(np.nanmean([s[k] for s in ss])) for k in keys} for a, ss in seed_stats.items()}

    def fn(v):
        m = {a: {k: np.nanmean([stat_c1(c)[k] for c in cs]) for k in keys} for a, cs in v.items()}
        out = {f"{a}|{k}": x for a, d in m.items() for k, x in d.items()}
        for a, b in pairs:
            if a in m and b in m:
                for k in ("auc_S", "agree_S", "brake_gt_hold"):
                    out[f"{a}-{b}|{k}"] = m[a][k] - m[b][k]
        return out
    ci = boot_arms(pts, fn, n_boot)
    rows = []
    for a, cs in pts.items():
        r = {"arm": a, "seeds": len(cs), "n_fork_points": len(cs[0]), **mean[a], **s_counts(cs[0]),
             "auc_S_per_seed": [round(float(s["auc_S"]), 3) for s in seed_stats[a]]}
        for k in keys:
            r[f"{k}_ci"] = ci[f"{a}|{k}"]
        for x, y in pairs:
            if x == a and f"{x}-{y}|auc_S" in ci:
                for k in ("auc_S", "agree_S", "brake_gt_hold"):
                    r[f"{k}_minus_{y}"] = mean[x][k] - mean[y][k]
                    r[f"{k}_minus_{y}_ci"] = ci[f"{x}-{y}|{k}"]
        rows.append(r)
    return pd.DataFrame(rows)


def ego_error(store: Store, key, ds: int, exit_pairs: set, arms, ey_of) -> pd.DataFrame:
    """Arms with s: the predicted e_y at 2 s (last-but-4 column of z10, de-standardised) against the branch's real e_y at fork + 2 s;
    persistence = the branch's own e_y at the fork tick. ey_of: (route_id, tick) -> e_y."""
    import torch
    R = M2.R2 if ds == 2 else M2.R1
    runs = pd.read_parquet(R / "forks.parquet")[["fork_id", "action", "route_id", "fork_tick"]]
    runs["gid"] = ds * GID + runs.fork_id
    runs = runs.assign(ey0=ey_of.reindex(list(zip(runs.route_id, runs.fork_tick))).to_numpy(),
                       ey10=ey_of.reindex(list(zip(runs.route_id, runs.fork_tick + W.FUT * W.TICKS))).to_numpy())
    rows = []
    for a in arms:
        if not store.has(a) or a not in M2.WITH_S:
            continue
        hat = []
        for s in store.seeds(a):
            gid, q, fa, actions, d = store.load(a, s)
            ck = torch.load(d / "model.pt", map_location="cpu")
            e_hat = q["z10"][..., -M2.S_DIM].astype(np.float32) * float(ck["sd"][-M2.S_DIM]) + float(ck["mu"][-M2.S_DIM])
            hat.append(pd.DataFrame({"gid": np.repeat(gid, len(actions)), "action": np.tile(actions, len(gid)), "ey_hat": e_hat.ravel()}))
        h = pd.concat(hat).groupby(["gid", "action"]).ey_hat.mean().reset_index()
        e = h.merge(runs, on=["gid", "action"]).join(key[["split"]], on=["gid", "action"])
        e = e[(e.split == "eval") & (e.gid // GID == ds) & ~e.gid.isin(exit_pairs) & e.ey10.notna() & e.ey0.notna()]
        for act in ("op", "shift_L", "shift_R", "shift_L_slow", "shift_R_slow", "nudge_L"):
            x = e[e.action == act]
            if len(x):
                rows.append({"arm": a, "action": act, "n": len(x), "med_abs_err_m": float((x.ey_hat - x.ey10).abs().median()),
                             "persistence_med_abs_err_m": float((x.ey0 - x.ey10).abs().median()), "true_ey10_median_m": float(x.ey10.median()),
                             "pred_ey10_median_m": float(x.ey_hat.median()), "sign_agree": float((np.sign(x.ey_hat) == np.sign(x.ey10)).mean())})
    return pd.DataFrame(rows)


# ================================================================ critics, C2, C3

_CRIT = {}


def critic_frame(store: Store, arm: str, label: str, tr: pd.DataFrame, holdout=None, train_ds=None, extra=None) -> pd.DataFrame:
    """Per (gid, action): p_learn / p_q averaged over the arm's seeds; each seed's critics (2-layer MLP, WL-1's recipe) are fitted on that
    seed's train-split fork points (of the datasets in train_ds; actions in `holdout` left out). extra: {seed: (Xl, Xq)} extra rows scored
    by the same fits, returned as attribute `.attrs['extra']` (the C4 exam frames)."""
    mk = (id(store), arm, label, tuple(holdout or ()), tuple(train_ds or ()))
    if extra is None and mk in _CRIT:
        return _CRIT[mk]
    tk = tr.set_index(["gid", "action"])
    pl, pq, pg, pc, ex_l, ex_q = [], [], [], [], [], []
    for s in store.seeds(arm):
        gid, q, fa, actions, _ = store.load(arm, s)
        n, A = len(gid), len(actions)
        z0 = np.repeat(q["z0"].astype(np.float32)[:, None], A, 1)
        Xl = np.concatenate([z0, q["z5"].astype(np.float32), q["z10"].astype(np.float32)], -1).reshape(n * A, -1)
        Xq = np.concatenate([z0, q["cmds"].reshape(n, A, -1).astype(np.float32)], -1).reshape(n * A, -1)
        lab = pd.DataFrame({"gid": np.repeat(gid, A), "action": np.tile(actions, n)}).join(tk[[label, "split", "ds"]], on=["gid", "action"])
        trn = (lab.split == "train").to_numpy() & lab[label].notna().to_numpy()
        if train_ds is not None:
            trn &= lab.ds.isin(train_ds).to_numpy()
        if holdout:
            trn &= ~lab.action.isin(holdout).to_numpy()
        y = lab[label].fillna(0).astype(float).to_numpy()
        xl_e = Xl if extra is None else np.concatenate([Xl, extra[s][0]])
        xq_e = Xq if extra is None else np.concatenate([Xq, extra[s][1]])
        p1, p2 = M._mlp_fit(Xl[trn], y[trn], s, xl_e), M._mlp_fit(Xq[trn], y[trn], s, xq_e)
        pl.append(p1[: n * A])
        pq.append(p2[: n * A])
        if extra is not None:
            ex_l.append(p1[n * A:])
            ex_q.append(p2[n * A:])
        pg.append((q["p_v"] * W.DT).sum(2).ravel())
        if "v0" in fa:                                          # the candidate's own commanded progress over 2 s (Q's rule)
            v0 = fa.set_index("gid").v0.reindex(gid).to_numpy()
            v = np.maximum(v0[:, None, None] + np.cumsum(q["cmds"][..., 0] * W.DT, -1), 0)
            pc.append((v * W.DT).sum(-1).ravel())
    cf = lab[["gid", "action"]].assign(p_learn=np.mean(pl, 0), p_q=np.mean(pq, 0), prog=np.mean(pg, 0))
    if pc:
        cf["prog_cmd"] = np.mean(pc, 0)
    if extra is not None:
        cf.attrs["extra"] = {"learn": ex_l, "q": ex_q}
    else:
        _CRIT[mk] = cf
    return cf


def _pairwise(x):
    ok = tot = 0
    for _, g in x.groupby("gid"):
        u, s = g[g.y.astype(bool)].p_learn.to_numpy(), g[~g.y.astype(bool)].p_learn.to_numpy()
        ok += (u[:, None] > s[None]).sum()
        tot += len(u) * len(s)
    return ok / max(tot, 1)


def evalset(cf, label, tr, ds, exit_pairs, actions=None) -> pd.DataFrame:
    k = tr.set_index(["gid", "action"])
    e = cf.join(k[[label, "split", "base_id", "cls", "world", "travel_m", "k_name", "ds"]], on=["gid", "action"]).rename(columns={label: "y"})
    e = e[(e.split == "eval") & (e.ds == ds) & e.y.notna() & ~e.gid.isin(exit_pairs)]
    if actions is not None:
        e = e[e.action.isin(actions)]
    return e.reset_index(drop=True)


def select(e: pd.DataFrame, pcol="p_learn", prog="prog") -> pd.DataFrame:
    """The registered rule per fork point: among candidates with p < 0.2 the largest progress, else the smallest p; with the op /
    oracle references (oracle: safe first, then the longest 3 s travel)."""
    sel = []
    for gid, g in e.groupby("gid"):
        op = g[g.action == "op"]
        if not len(op):
            continue
        ok = g[g[pcol] < UNSAFE_P]
        pick = ok.loc[ok[prog].idxmax()] if len(ok) else g.loc[g[pcol].idxmin()]
        orc = g.sort_values(["y", "travel_m"], ascending=[True, False]).iloc[0]
        sel.append({"gid": gid, "base_id": g.base_id.iloc[0], "world": g.world.iloc[0], "cls": g.cls.iloc[0], "k_name": g.k_name.iloc[0],
                    "pick": pick.action, "U_pick": float(pick.y), "U_op": float(op.y.iloc[0]), "U_or": float(orc.y),
                    "travel_pick": float(pick.travel_m), "travel_op": float(op.travel_m.iloc[0])})
    return pd.DataFrame(sel)


def H(x) -> float:
    den = x.U_op.mean() - x.U_or.mean()
    return (x.U_op.mean() - x.U_pick.mean()) / den if den > 1e-9 else np.nan


def _solvable(p):
    return p[(p.U_op > 0.5) & (p.U_or < 0.5)]


def c2_c3(store: Store, arm: str, label: str, tr, ds: int, exit_pairs: set, n_boot: int, hold_arm=None, cf=None, train_ds=None,
          actions=None) -> tuple[dict, pd.DataFrame | None]:
    """C2 (AUC, pairwise, C-learn - Q, 2(b) with the holdout arm) and C3 (H, x- progress / unsafe, C3c) of one arm under one label,
    with the WL-2 candidates `actions` (default: what the arm has)."""
    cf = cf if cf is not None else critic_frame(store, arm, label, tr, train_ds=train_ds)
    e = evalset(cf, label, tr, ds, exit_pairs, actions)
    auc = lambda x: M._auc(x.y, x.p_learn)
    diff = lambda x: M._auc(x.y, x.p_learn) - M._auc(x.y, x.p_q)
    ci = boot_multi(e, {"auc": auc, "diff": diff}, n_boot)
    r = {"arm": arm, "label": label, "ds": ds, "n_rows": len(e), "n_unsafe": int(e.y.sum()), "auc_learn": auc(e), "auc_learn_ci": ci["auc"],
         "auc_q": M._auc(e.y, e.p_q), "pairwise": _pairwise(e), "diff_learn_q": diff(e), "diff_ci": ci["diff"]}
    if hold_arm and store.has(hold_arm):
        hset = M2.HOLD.get(hold_arm, M.HOLD_OUT)
        h = evalset(critic_frame(store, hold_arm, label, tr, holdout=hset, train_ds=train_ds), label, tr, ds, exit_pairs, actions)
        h = h[h.action.isin(hset)]
        c = boot_multi(h, {"diff": diff}, n_boot)["diff"]
        r.update({"holdout_arm": hold_arm, "holdout_actions": "+".join(hset), "holdout_n": len(h), "holdout_auc_learn": auc(h),
                  "holdout_auc_q": M._auc(h.y, h.p_q), "holdout_diff": diff(h), "holdout_diff_ci": c})
    sel = select(e)
    plus, minus = sel[sel.world == "plus"], sel[sel.world == "minus"]
    hci = boot_multi(plus, {"H": H, "gain": lambda x: x.U_op.mean() - x.U_pick.mean()}, n_boot)
    solv = _solvable(plus)
    r.update({"n_plus": len(plus), "U_op": plus.U_op.mean(), "U_sel": plus.U_pick.mean(), "U_or": plus.U_or.mean(), "H": H(plus), "H_ci": hci["H"],
              "gain": plus.U_op.mean() - plus.U_pick.mean(), "gain_ci": hci["gain"], "n_solvable": len(solv), "routes_solvable": int(solv.base_id.nunique()),
              "rescued": int((solv.U_pick < 0.5).sum()), "harmed": int(((plus.U_op < 0.5) & (plus.U_pick > 0.5)).sum()),
              "n_op_safe": int((plus.U_op < 0.5).sum()), "picks": sel.pick.value_counts().to_dict(),
              "n_minus": len(minus), "travel_ratio_minus": float((minus.travel_pick / minus.travel_op.clip(lower=0.1)).mean()) if len(minus) else np.nan,
              "U_pick_minus": minus.U_pick.mean(), "U_op_minus": minus.U_op.mean()})
    if "prog_cmd" in e:                                          # C3c: Q with the same rule, progress = the candidate's commanded 2 s progress
        selq = select(e, "p_q", "prog_cmd")
        m = sel[["gid", "base_id", "world", "U_pick", "U_op", "U_or"]].merge(selq[["gid", "U_pick"]].rename(columns={"U_pick": "U_pick_q"}), on="gid")
        mp = m[m.world == "plus"]
        Hq = lambda x: H(x.assign(U_pick=x.U_pick_q))
        d3 = lambda x: H(x) - Hq(x)
        c3c = boot_multi(mp, {"d": d3}, n_boot)["d"]
        r.update({"H_q": Hq(mp), "H_learn_minus_q": d3(mp), "H_learn_minus_q_ci": c3c})
    return r, sel


def descriptions(e: pd.DataFrame, sel: pd.DataFrame, tr: pd.DataFrame, n_boot: int) -> dict:
    """C3 descriptions: by k / class, the 7-candidate rule, the best fixed candidate that satisfies C3b, the non-pedestrian-hazard forks."""
    out = {}
    plus = sel[sel.world == "plus"]
    out["by_k"] = {k: {"n": len(g), "U_op": g.U_op.mean(), "U_sel": g.U_pick.mean(), "U_or": g.U_or.mean()} for k, g in plus.groupby("k_name")}
    out["by_cls"] = {k: {"n": len(g), "U_op": g.U_op.mean(), "U_sel": g.U_pick.mean(), "U_or": g.U_or.mean()} for k, g in plus.groupby("cls")}
    e7 = e[e.action.isin(ACTIONS)]
    s7 = select(e7)
    out["rule_7_candidates"] = {"H": H(s7[s7.world == "plus"]), "n_plus": int((s7.world == "plus").sum())}
    fixed = {}
    for a in sorted(e.action.unique()):
        g = e[e.action == a].drop_duplicates("gid")
        u = g.merge(e[e.action == "op"][["gid", "y", "travel_m"]].rename(columns={"y": "y_op", "travel_m": "t_op"}), on="gid")
        orc = e.sort_values(["y", "travel_m"], ascending=[True, False]).groupby("gid").y.first()
        p, mi = u[u.world == "plus"], u[u.world == "minus"]
        if not len(p):
            continue
        den = p.y_op.mean() - orc.reindex(p.gid).mean()
        ratio = float((mi.travel_m / mi.t_op.clip(lower=0.1)).mean()) if len(mi) else np.nan
        fixed[a] = {"H": float((p.y_op.mean() - p.y.mean()) / den) if den > 1e-9 else np.nan, "travel_ratio_minus": ratio,
                    "unsafe_minus": float(mi.y.mean()) if len(mi) else np.nan, "unsafe_op_minus": float(mi.y_op.mean()) if len(mi) else np.nan,
                    "c3b_ok": bool(ratio >= 0.9 and mi.y.mean() <= mi.y_op.mean() + 0.02) if len(mi) else False}
    out["fixed_candidate"] = fixed
    npb = nonped_both2(tr)
    sub = sel[sel.gid.isin(npb)]
    out["nonped_both"] = {"n": len(sub), **{f"{c}_{w}": float(sub[sub.world == w][c].mean()) if (sub.world == w).any() else None
                                            for c in ("U_pick", "U_op", "U_or") for w in ("plus", "minus")}}
    return out


# ================================================================ slow shifts with the realised offset

def slow_shift(store: Store, key: pd.DataFrame, off: pd.DataFrame, exit_pairs: set, arms) -> pd.DataFrame:
    """Shift-type branches of the WL-2 eval fork points: the executed offset (jevdrive.wl.shift_offsets' definition, from the poses),
    its distribution per candidate, and how the arms' rollouts relate to it: Spearman(score, offset) against Spearman(true occupancy
    at 2 s, offset) over the branches of S (op lane occupied), and the C1c AUC split by the realised-offset band."""
    from scipy.stats import spearmanr
    rows = []
    o = off.dropna(subset=["offset_left_m"]) if "offset_left_m" in off else off
    ok = key.reset_index().merge(o[["route_id", "action", "offset_left_m"]], on=["route_id", "action"]) if "route_id" in key.reset_index() else None
    ok = ok[(ok.split == "eval") & ~ok.gid.isin(exit_pairs)]
    ok["signed"] = np.where(ok.action.str.contains("_R"), -ok.offset_left_m, ok.offset_left_m)     # >0: moved to the commanded side
    for a, g in ok.groupby("action"):
        rows.append({"section": "offset", "arm": "-", "action": a, "n": len(g), "median_m": g.signed.median(), "q25_m": g.signed.quantile(.25),
                     "q75_m": g.signed.quantile(.75), "share_ge_2m": float((g.signed >= 2).mean()), "share_1_2m": float(((g.signed >= 1) & (g.signed < 2)).mean()),
                     "sign_ok": float((g.signed > 0).mean())})
    K = ok.set_index(["gid", "action"])
    for arm in arms:
        if not store.has(arm) or "shift_L_slow" not in store.load(arm, store.seeds(arm)[0])[3]:
            continue
        d = []
        for s in store.seeds(arm):
            gid, q, fa, actions, _ = store.load(arm, s)
            d.append(pd.DataFrame({"gid": np.repeat(gid, len(actions)), "action": np.tile(actions, len(gid)), "score": q["p_occ"][:, :, 9].ravel()}))
        d = pd.concat(d).groupby(["gid", "action"]).score.mean().reset_index()
        opocc = key.reset_index().query("action == 'op'").set_index("gid").occ
        for a in ("shift_L", "shift_R", "shift_L_slow", "shift_R_slow", "nudge_L"):
            g = d[d.action == a].join(K[["offset_left_m", "occ", "signed"]], on=["gid", "action"]).dropna(subset=["signed", "occ"])
            g = g[g.gid.map(opocc) > 0.5]                             # S: the op lane is occupied at 2 s
            if len(g) < 8:
                continue
            rs, rt = spearmanr(g.score, g.signed)[0], spearmanr(g.occ, g.signed)[0]
            r = {"section": "arm", "arm": arm, "action": a, "n": len(g), "n_still": int((g.occ > 0.5).sum()), "spearman_score_offset": rs,
                 "spearman_true_occ_offset": rt}
            for lo, hi, nm in ((-9, 1, "lt1"), (1, 2, "1to2"), (2, 99, "ge2")):
                b = g[(g.signed >= lo) & (g.signed < hi)]
                r[f"n_{nm}"], r[f"still_{nm}"] = len(b), float((b.occ > 0.5).mean()) if len(b) else np.nan
                r[f"score_{nm}"] = float(b.score.mean()) if len(b) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


# ================================================================ C4, continuous examinee

def exam_rollouts(store: Store, arm: str, hold_idx: int = 3):
    """Per seed: the `hold` rollout (10 steps) of every exam frame of the 20 eval routes with the arm's predictor, and the critic inputs."""
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
    cmds = C4.candidates_for(fi)[:, hold_idx: hold_idx + 1]
    ey_wm = np.load(M2.P1 / "dryrun" / "ey_wm.npy")
    wmc = wm.assign(a_prev=wm.a_prev.fillna(0).clip(-15, 15), w_prev=wm.w_prev.fillna(0).clip(-15, 15), v=wm.v.fillna(0))
    from . import wl_dryrun as D
    S_wm = D.ego_cols(wmc, ey_wm)
    with_s = arm in M2.WITH_S

    class Ext:
        def __init__(s, base, S):
            s.base, s.S, s.shape = base, S, (base.shape[0], base.shape[1] + S.shape[1])

        def __getitem__(s, i):
            return np.concatenate([np.asarray(s.base[i]), s.S[i].astype(np.float16)], 1)
    meta, z, anchors = C4.prepare_rows(ok, rows_of, wm, Ext(wz, S_wm) if with_s else wz)
    res = {}
    for s in store.seeds(arm):
        d = store.dirs[arm][s]
        ck = torch.load(d / "model.pt", map_location="cuda")
        data = M2.Data2(meta, [z], np.arange(0), moments=(ck["mu"], ck["sd"]))
        model = M.build(z.shape[1]).cuda()
        model.load_state_dict(ck["state"])
        starts = torch.as_tensor(anchors - (W.HIST - 1), device="cuda")
        pred = M.predict(model, data, starts, torch.as_tensor(cmds[:, 0], device="cuda"), src=1)
        z0 = data.Z[starts + W.HIST - 1].float()
        Xl = torch.cat([z0, pred[:, 4], pred[:, 9]], -1).cpu().numpy().astype(np.float32)
        Xq = np.concatenate([z0.cpu().numpy(), cmds[:, 0].reshape(len(ok), -1)], -1).astype(np.float32)
        res[s] = (Xl, Xq)
        del data, model, pred
        torch.cuda.empty_cache()
    return ok, res, (obs, null, pairs)


def c4_arm(store: Store, arm: str, tr: pd.DataFrame) -> dict:
    ok, ex, (obs, null, pairs) = exam_rollouts(store, arm)
    cf = critic_frame(store, arm, "cg", tr, extra=ex)
    pl, pq = cf.attrs["extra"]["learn"], cf.attrs["extra"]["q"]
    risk = {f"{arm} risk s{s}": pd.Series(-p, index=ok) for s, p in zip(store.seeds(arm), pl)}
    risk[f"{arm} risk mean"] = pd.Series(-np.mean(pl, 0), index=ok)
    risk[f"{arm} Q risk mean"] = pd.Series(-np.mean(pq, 0), index=ok)
    return {"ok": ok, "risk": risk, "obs": obs, "null": null, "pairs": pairs}


def c4_flips(store: Store, arms: tuple, tr: pd.DataFrame) -> pd.DataFrame:
    parts = [c4_arm(store, a, tr) for a in arms if store.has(a)]
    obs, null, pairs = parts[0]["obs"], parts[0]["null"], parts[0]["pairs"]
    have = set.intersection(*[set(p["ok"]) for p in parts])
    cols = []
    ex = {}
    for tag, df, a, b in (("obs", obs, "fn_plus", "fn_minus"), ("null", null, "fn_plus", "fn_null")):
        df = df[df[a].isin(have) & df[b].isin(have)].copy()
        for p in parts:
            for c, sr in p["risk"].items():
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
    return fl[["cls", "examinee", "flip_rate", "flip_lo", "flip_hi", "false_flip_null_oos", "tau_model", "n_reactive", "margin_ok", "n_obs", "n_null"]]


# ================================================================ driver

def _f(x, k=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{k}f}" if isinstance(x, (int, float, np.floating)) else str(x))


def run_wl2(a) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    store = wl2_store()
    tr2 = truth(2, M2.P2, M2.R2)
    tr1 = truth(1, M2.P1, M2.R1)
    tr = pd.concat([tr1, tr2], ignore_index=True)
    key = tr.set_index(["gid", "action"])
    ex2, ex1 = pair_exit(M2.R2, 2), pair_exit(M2.R1, 1)
    exit_all = ex1 | ex2
    res = {"arms": {arm: store.seeds(arm) for arm in store.dirs}}
    pairs = (("B", "W"), ("B", "A"), ("A", "vrep"), ("Bs", "B"), ("T", "Bs"), ("T", "B"))
    for ds, ex, tag in ((2, ex2, "main"), (1, ex1, "wl1eval")):
        c1 = readout_c1(store, key, ds, ex, MAIN_ARMS, a.n_boot, pairs)
        c1.to_csv(OUT / f"c1_{tag}.csv", index=False, float_format="%.4f")
        R = M2.P2 if ds == 2 else M2.P1
        ey = np.load(R / ("ey.npy" if ds == 2 else "dryrun/ey_idx.npy"))[:, 0]
        ix = pd.read_parquet(R / "index.parquet", columns=["route_id", "tick"])
        ey_of = pd.Series(ey, index=pd.MultiIndex.from_arrays([ix.route_id, ix.tick]))
        ee = ego_error(store, key, ds, ex, ("B", "Bs", "T", "Bh", "Bhc"), ey_of[~ey_of.index.duplicated()])
        ee.to_csv(OUT / f"ego_error_{tag}.csv", index=False, float_format="%.4f")
        res[f"c1_{tag}"] = c1.to_dict("records")
        log.info("C1 %s done %.0f s", tag, time.time() - t0)
    rows, sels, descr = [], {}, {}
    for ds, ex, tag in ((2, ex2, "main"), (1, ex1, "wl1eval")):
        for arm in C2_ARMS:
            if not store.has(arm) or arm in ("Bh", "Bhc"):
                continue
            for lb in LABELS:
                hold = {"B": "Bh"}.get(arm)
                td = (1,) if arm == "vrep" else None
                r, sel = c2_c3(store, arm, lb, tr, ds, ex, min(a.n_boot, 1000), hold if tag == "main" else None, train_ds=td)
                r["set"] = tag
                rows.append(r)
                if tag == "main" and lb == "cg":
                    e = evalset(critic_frame(store, arm, lb, tr, train_ds=td), lb, tr, ds, ex)
                    descr[arm] = descriptions(e, sel, tr, a.n_boot)
                    sels[arm] = sel
                if arm == "B" and tag == "main" and lb == "cg" and store.has("Bhc"):
                    hc = evalset(critic_frame(store, "Bhc", lb, tr, holdout=M2.HOLD["Bhc"]), lb, tr, ds, ex)
                    hc = hc[hc.action.isin(M2.HOLD["Bhc"])]
                    dif = lambda x: M._auc(x.y, x.p_learn) - M._auc(x.y, x.p_q)
                    res["combo_holdout"] = {"n": len(hc), "diff": dif(hc), "diff_ci": boot_multi(hc, {"d": dif}, 1000)["d"],
                                            "auc_learn": M._auc(hc.y, hc.p_learn), "auc_q": M._auc(hc.y, hc.p_q)}
                log.info("C2/C3 %s %s %s done %.0f s", tag, arm, lb, time.time() - t0)
                pd.DataFrame(rows).to_csv(OUT / "c2_c3.csv", index=False, float_format="%.4f")
    res["c2_c3"] = rows
    res["descriptions"] = descr
    pd.concat(sels.values(), keys=sels.keys()).reset_index(level=0).rename(columns={"level_0": "arm"}).to_csv(OUT / "c3_selection_cg.csv", index=False)
    off = pd.read_parquet(M2.P2 / "offsets.parquet")
    ss = slow_shift(store, key.loc[key.ds == 2], off, ex2, ("B", "A", "vrep", "T", "Bs"))
    ss.to_csv(OUT / "slow_shift.csv", index=False, float_format="%.4f")
    if not a.skip_c4:
        fl = c4_flips(store, tuple(x for x in ("B", "A", "W") if store.has(x)), tr)
        fl.to_csv(OUT / "c4_flips.csv", index=False, float_format="%.4f")
        res["c4"] = fl.to_dict("records")
        log.info("C4 done %.0f s", time.time() - t0)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    log.info("all done %.0f s -> %s", time.time() - t0, OUT)
    return res


def validate_wl1(a) -> dict:
    """The generalised readouts on WL-1 data (dataset 1, 7 candidates, the dry-run arms) against research/results/wl-dryrun/."""
    OUT.mkdir(parents=True, exist_ok=True)
    store = wl1_store()
    tr = truth(1, M2.P1, M2.R1)
    key = tr.set_index(["gid", "action"])
    ex = pair_exit(M2.R1, 1)
    ref = Path(__file__).resolve().parents[1] / "research" / "results" / "wl-dryrun"
    c1 = readout_c1(store, key, 1, ex, ("W1worig", "W1main", "B", "Bi", "T"), a.n_boot, (("B", "W1worig"), ("B", "W1main"), ("T", "Bi")))
    c1.to_csv(OUT / "validate_c1.csv", index=False, float_format="%.4f")
    old = pd.read_csv(ref / "c1.csv").set_index("arm")
    rep = []
    for r in c1.itertuples():
        o = old.loc[r.arm]
        for k in ("brake_gt_hold", "agree_S", "auc_S"):
            rep.append({"what": f"c1 {r.arm} {k}", "new": getattr(r, k), "old": o[k], "diff": getattr(r, k) - o[k]})
    rows = []
    old2 = pd.read_csv(ref / "c2_c3.csv")
    for arm in ("B", "W1main", "Bi", "T"):
        for lb in LABELS:
            hold = {"B": "Bh", "W1main": "W1holdout"}.get(arm)
            r, _ = c2_c3(store, arm, lb, tr, 1, ex, min(a.n_boot, 1000), hold)
            o = old2[(old2.arm == arm) & (old2.label == lb)].iloc[0]
            for k in ("auc_learn", "auc_q", "pairwise", "diff_learn_q", "H", "n_solvable"):
                rep.append({"what": f"c2c3 {arm} {lb} {k}", "new": r[k], "old": o[k], "diff": r[k] - o[k]})
            if hold:
                rep.append({"what": f"c2c3 {arm} {lb} holdout_diff", "new": r["holdout_diff"], "old": o["holdout_diff"], "diff": r["holdout_diff"] - o["holdout_diff"]})
            rows.append(r)
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / "validate_report.csv", index=False, float_format="%.5f")
    print(rep.to_string())
    print("max |diff|:", rep["diff"].abs().max())
    return {"max_abs_diff": float(rep["diff"].abs().max())}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-c4", action="store_true")
    ap.add_argument("--validate-wl1", action="store_true")
    a = ap.parse_args()
    print(json.dumps(validate_wl1(a) if a.validate_wl1 else {"done": bool(run_wl2(a))}, default=float))


if __name__ == "__main__":
    main()
