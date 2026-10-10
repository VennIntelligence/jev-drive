#!/usr/bin/env python3
"""HEAD1 stage-1 reads and gate (experiments/corridor, plans/2026-10-10-head1-prereg.md). CPU, .venv.

  feats   plan features of the stored policy plans (4 s arc length, 4 s heading, heading profile on the label grid) for
          (H) navtrain fold-0 dev logs x CF5f0-F-s0 and (T) navtest x SH30-F-s0/s1, GH0-F-s0/s1 -> $H1/report/feats.npz
  read    reads (a)-(e), arm selection on (H), gate lines G1-G3 on (T) -> $H1/report/{tables.md, reads.json, gate.json, tok.npz}
  fig     figures from reads.json / tok.npz -> $H1/report/figs/
Sets    H = held-out navtrain logs of fold 0 (head and fold policy both never saw them); T = navtest (never fitted, never selected on).
The logged future and the map label are the truth / a privileged reference here; the head reads frozen vision tokens + ego + command only.
Statistics are RMS / correlation / R2 with the log-cluster bootstrap of jevdrive.stats (same draws: default_rng(0).integers(G, (B, G))).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_parity/scripts"), str(REPO / "research"), str(Path(__file__).parent)]
from jevdrive.common import data_dir  # noqa: E402

D = data_dir()
H1 = D / "runs/corridor/head1"
RP = H1 / "report"
CR = D / "runs/op_parity/cache"
OL = D / "runs/bench/ol"
NSH, NB, FOLD = 12, 10_000, 0
GRID = np.r_[np.arange(0.0, 40.001, 2.5), 45.0, 50.0, 60.0, 70.0, 80.0]
POL = {"H": ["CF5f0-F-s0"], "T": ["SH30-F-s0", "SH30-F-s1", "GH0-F-s0", "GH0-F-s1"]}
ARMS, FIXED = ("L", "M", "C"), (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
BK = {"all": 0.0, ">20": 20.0, ">45": 45.0}
QS_GAIN, R2_PASS, R2_FAIL = 0.93, 0.5 / 0.93, 0.3 / 0.93          # decision 204: shape-only oracle +0.93; predicted gain = 0.93 x R2
_P = {}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def pname(arm, frac=1.0, seed=0):
    return f"{arm}-f{FOLD}-p{int(round(frac * 1000)):04d}-s{seed}"


def at(prof, s):
    """prof (n, 22) on GRID, s (n,) -> linear interpolation at arc length s (clamped to the grid); nan where a neighbour is nan."""
    s = np.clip(s, 0, GRID[-1])
    j = np.clip(np.searchsorted(GRID, s, side="right") - 1, 0, len(GRID) - 2)
    r = np.arange(len(s))
    f = (s - GRID[j]) / (GRID[j + 1] - GRID[j])
    lo, hi = prof[r, j], prof[r, j + 1]
    return np.where(f < 1e-9, lo, lo * (1 - f) + hi * f)


# ---------------------------------------------------------------- data
def load_set(S):
    """-> dict(names, log, fut, lab (label npz rows of the set), plans {model: (n, 8, 3)})."""
    if S == "T":
        t = np.load(CR / "lb_navtest/tab.npz")
        names, log, fut = t["names"].astype(str), t["log"].astype(str), t["fut"].astype(np.float64)
        lab = dict(np.load(H1 / "labels/navtest.npz"))
        assert (lab["names"].astype(str) == names).all()
        plans = {}
        for m in POL[S]:
            z = np.load(OL / "lb_navtest/preds" / f"{m}-warp__base.npz")
            pos = {k: i for i, k in enumerate(z["tokens"].tolist())}
            plans[m] = z["poses"][[pos[k] for k in names]].astype(np.float64)
        return dict(names=names, log=log, fut=fut, lab=lab, plans=plans)
    from jevdrive.data import splits
    tabs = [np.load(CR / f"navtrain_full.s{k}of{NSH}/tab.npz") for k in range(NSH)]
    names, log, fut = (np.concatenate([t[k] for t in tabs]) for k in ("names", "log", "fut"))
    m = splits.load(f"navsim/op-parity-cf5f{FOLD}-dev").mask(log)
    lab = {k: (v[m] if getattr(v, "shape", ()) and len(v) == len(names) else v) for k, v in np.load(H1 / "labels/navtrain.npz").items()}
    plans = {}
    for mod in POL[S]:
        P = []
        for k in range(NSH):
            z = np.load(OL / f"navtrain_full.s{k}of{NSH}/preds" / f"{mod}-warp__base.npz")
            assert (z["tokens"] == tabs[k]["names"]).all()
            P.append(z["poses"])
        plans[mod] = np.concatenate(P)[m].astype(np.float64)
    return dict(names=names[m].astype(str), log=log[m].astype(str), fut=fut[m].astype(np.float64), lab=lab, plans=plans)


def _feat_chunk(c):
    import pt_swap as PS
    P = _P["P"]
    out = np.full((len(c), 2 + len(GRID)), np.nan)
    for n, i in enumerate(c):
        if not np.isfinite(P[i]).all():
            continue
        cv = PS.Curve(P[i])
        sp = float(cv.sv[-1])
        out[n, 0], out[n, 1] = sp, np.unwrap(np.r_[0, P[i][:, 2]])[-1]
        g = GRID <= sp + 1e-9
        if g.any() and cv.L > 0:
            out[n, 2:][g] = cv.at(GRID[g])[:, 2]
        else:
            out[n, 2] = 0.0
    return c, out


def cmd_feats(a):
    from jevdrive import par
    from jevdrive.run import Run
    from scipy.interpolate import CubicSpline  # noqa: F401  (before the fork)
    with Run("corridor", "head1/feats", config=vars(a)) as run:
        out = {}
        for S in ("H", "T"):
            d = load_set(S)
            for m, P in d["plans"].items():
                _P["P"] = P
                idx = np.arange(len(P))
                res = par.pmap(_feat_chunk, [idx[k::128] for k in range(128)], run=run, desc=f"{S} {m}")
                res.raise_if_failed()
                F = np.full((len(P), 2 + len(GRID)), np.nan)
                for c, o in res.values:
                    F[c] = o
                out[f"{S}|{m}"] = F.astype(np.float32)
                run.info("%s %s: %d plans, median 4 s arc %.2f m", S, m, len(P), np.nanmedian(F[:, 0]))
        RP.mkdir(parents=True, exist_ok=True)
        np.savez(RP / "feats.npz", **out)


# ---------------------------------------------------------------- statistics
class CB:
    """Log-cluster bootstrap of a statistic that is a function of moment sums of two paired columns (a, b)."""

    def __init__(s, logs):
        import pandas as pd
        s.codes, u = pd.factorize(np.asarray(logs))
        s.G = len(u)
        idx = np.random.default_rng(0).integers(s.G, size=(NB, s.G))
        s.C = np.zeros((NB, s.G))
        np.add.at(s.C, (np.arange(NB)[:, None], idx), 1.0)

    def __call__(s, stat, a, b=None, mask=None):
        b = a if b is None else b
        ok = np.isfinite(a) & np.isfinite(b) & (True if mask is None else mask)
        a, b, c = a[ok].astype(np.float64), b[ok].astype(np.float64), s.codes[ok]
        M = np.stack([np.ones_like(a), a, b, a * a, b * b, a * b], 1)
        S = np.stack([np.bincount(c, M[:, k], s.G) for k in range(6)], 1)
        bs = stat((s.C @ S).T)
        lo, hi = np.nanquantile(bs, [0.025, 0.975])
        return dict(v=float(stat(S.sum(0))), lo=float(lo), hi=float(hi), n=int(ok.sum()), logs=int(len(np.unique(c))))


rms_a = lambda m: np.sqrt(m[3] / m[0])  # noqa: E731
rms_diff = lambda m: np.sqrt(m[3] / m[0]) - np.sqrt(m[4] / m[0])  # noqa: E731
mean_a = lambda m: m[1] / m[0]  # noqa: E731
mean_diff = lambda m: (m[1] - m[2]) / m[0]  # noqa: E731


def corr(m):
    n, a, b, aa, bb, ab = m
    return (ab / n - a * b / n ** 2) / np.sqrt(np.maximum((aa / n - (a / n) ** 2) * (bb / n - (b / n) ** 2), 1e-18))


def r2(m):
    """corr(a, a - b)^2: the share of the variance of a (the policy's error) a linear read of the disagreement a - b explains."""
    n, a, b, aa, bb, ab = m
    va, vb, cab = aa / n - (a / n) ** 2, bb / n - (b / n) ** 2, ab / n - a * b / n ** 2
    return (va - cab) ** 2 / np.maximum(va * (va + vb - 2 * cab), 1e-18)


def shift_fit(X, L, sp):
    """(b): delta (m, late +) minimising RMS_s[X(s) - L(s - delta)] over the grid points <= min(sp, 40 m); L(u < 0) = 0. nan when < 3 points."""
    g = GRID[:17]
    X, L = X[:, :17], L[:, :17]
    use = (g[None] <= np.minimum(sp, 40.0)[:, None] + 1e-9) & np.isfinite(X)
    dl = np.round(np.arange(-6, 6.001, 0.1), 2)
    best, arg = np.full(len(X), np.inf), np.full(len(X), np.nan)
    r = np.arange(len(X))[:, None]
    for d_ in dl:
        u = g[None] - d_
        j = np.clip(np.floor(u / 2.5).astype(int), 0, 15)
        f = u / 2.5 - j
        v = np.where(u < 0, 0.0, L[r, np.broadcast_to(j, X.shape)] * (1 - f) + np.where(f > 1e-9, L[r, np.broadcast_to(j + 1, X.shape)] * f, 0.0))
        v = np.where(u > 40.0, np.nan, v)
        m = use & np.isfinite(v)
        e = np.where(m, (X - v) ** 2, 0.0).sum(1) / np.maximum(m.sum(1), 1)
        e = np.where(m.sum(1) >= 3, e, np.inf)
        upd = e < best - 1e-12
        best[upd], arg[upd] = e[upd], d_
    return arg


def fmt(r, f="{:.2f}", ci=True):
    return "n/a" if r is None or not np.isfinite(r["v"]) else (f.format(r["v"]) + (f" [{f.format(r['lo'])}, {f.format(r['hi'])}]" if ci else ""))


def md(rows, cols):
    L = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] * len(cols)) + "|"]
    return "\n".join(L + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]) + "\n"


# ---------------------------------------------------------------- reads
def cannot_turn(names, fut):
    """SH30 seeds' cannot-make-the-turn flags on navtest (decisions 207 / 240: DAC fail, not an inside cut, turn gain < 0.9)."""
    import pandas as pd
    import fd_navsim as FD
    from jevdrive.bench.navsim import SUBS
    PT = D / "runs/op_parity/pt_swap"
    df = pd.read_csv(PT / "score_all.csv")
    rp = pd.read_parquet(PT / "replay.parquet").set_index(["key", "token"])
    Z = np.load(PT / "poses.npz")
    assert (Z["tokens"].astype(str) == names).all()
    sgn, out = np.sign(np.degrees(fut[:, -1, 2])), {}
    for m, mod in (("sh0", "SH30-F-s0"), ("sh1", "SH30-F-s1")):
        dac = df[df.key == f"{m}_pp"].set_index("token").reindex(names)[SUBS["DAC"]].to_numpy(float) < 1
        side = rp.loc[f"{m}_pp"].reindex(names).lqr_side.to_numpy(float)
        with np.errstate(invalid="ignore"):
            under = FD.plan_kin(Z[f"{m}_pp"].astype(np.float64), fut)["gain"] < 0.9
        out[mod] = dac & ~(dac & (side == sgn)) & under
    return out


def cmd_read(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("corridor", "head1/read", config=vars(a)) as run:
        for s in ("navsim/navtest", f"navsim/op-parity-cf5f{FOLD}-dev", f"navsim/op-parity-cf5f{FOLD}-train"):
            run.use_split(splits.load(s))
        FE = np.load(RP / "feats.npz")
        R, TOK, L = {}, {}, []
        P = L.append
        preds = {}

        def pred(name, S):
            if name not in preds:
                preds[name] = np.load(H1 / "pred" / f"{name}.npz")
            z = preds[name]
            return z["names_dev" if S == "H" else "names_test"].astype(str), z["dev" if S == "H" else "test"].astype(np.float64)

        heads = {f"{arm} s{s}": (pname(arm, 1.0, s), 1 if arm == "M" else 0) for arm in ARMS for s in (0, 1)} | {"blind": (pname("blind"), 0), "blind (M channel)": (pname("blind"), 1),
                                                                                                                  "C s0 (M channel)": (pname("C"), 1), "C s1 (M channel)": (pname("C", 1.0, 1), 1)}
        heads = {k: v for k, v in heads.items() if (H1 / "pred" / f"{v[0]}.npz").exists()}
        DS = {}
        for S in ("H", "T"):
            d = load_set(S)
            n = len(d["names"])
            d["dy"] = np.abs(np.degrees(d["fut"][:, 7, 2]))
            d["bk"] = {k: d["dy"] > v if v else np.isfinite(d["dy"]) for k, v in BK.items()}
            d["h4"] = d["fut"][:, 7, 2]
            d["pol"] = {m: dict(sp=FE[f"{S}|{m}"][:, 0].astype(np.float64), e=wrap(FE[f"{S}|{m}"][:, 1] - d["h4"]), prof=FE[f"{S}|{m}"][:, 2:].astype(np.float64)) for m in POL[S]}
            d["head"] = {}
            for k, (nm, ch) in heads.items():
                nmz, pz = pred(nm, S)
                assert (nmz == d["names"]).all(), f"{nm}: prediction rows are not the set's rows"
                d["head"][k] = pz[..., ch]
            d["cb1"] = CB(d["log"])
            DS[S] = d
            run.info("%s: %d tokens, %d logs; buckets %s", S, n, d["cb1"].G, {k: int(v.sum()) for k, v in d["bk"].items()})
        assert not set(DS["H"]["log"]) & set(DS["T"]["log"])
        deg = np.degrees

        def pooled(S, mods, fn):
            """Concatenate fn(model) -> (cols...) over the policy seeds, with the tiled logs / buckets and a bootstrap object."""
            d = DS[S]
            cols = [fn(m) for m in mods]
            key = (S, len(mods))
            if key not in _P:
                _P[key] = CB(np.tile(d["log"], len(mods)))
            return [np.concatenate(c) for c in zip(*cols)], {k: np.tile(v, len(mods)) for k, v in d["bk"].items()}, _P[key]

        # ---- (a) heading error at the plan's own 4 s arc length; (c) independence; (d) floor
        groups = {"H": {"CF5f0": POL["H"]}, "T": {"SH30-F": POL["T"][:2], "GH0-F (pilot)": POL["T"][2:]}}
        A = []
        for S, gs in groups.items():
            d = DS[S]
            for gname, mods in gs.items():
                (ep,), bk, cb = pooled(S, mods, lambda m: (deg(d["pol"][m]["e"]),))
                (emap,), _, _ = pooled(S, mods, lambda m: (deg(wrap(at(d["lab"]["M"].astype(np.float64), d["pol"][m]["sp"]) - d["h4"])),))
                (elog,), _, _ = pooled(S, mods, lambda m: (deg(wrap(at(d["lab"]["L"].astype(np.float64), d["pol"][m]["sp"]) - d["h4"])),))
                eh = {k: pooled(S, mods, lambda m, k=k: (deg(wrap(at(d["head"][k], d["pol"][m]["sp"]) - d["h4"])),))[0][0] for k in d["head"]}
                eh4 = {k: deg(wrap(at(d["head"][k], d["lab"]["s4"].astype(np.float64)) - d["h4"])) for k in d["head"]}
                for b, mk in bk.items():
                    key = f"{S}|{gname}|{b}"
                    R[key] = dict(policy=cb(rms_a, ep, mask=mk), map_ref=cb(rms_a, emap, mask=mk), log_profile_at_plan_arc=cb(rms_a, elog, mask=mk), heads={})
                    A.append({"set": S, "policy": gname, "bucket": b, "predictor": "policy's own plan", "n": R[key]["policy"]["n"], "RMS deg": fmt(R[key]["policy"])})
                    A.append({"set": S, "policy": gname, "bucket": b, "predictor": "map label M at the plan's arc (privileged)", "n": R[key]["map_ref"]["n"], "RMS deg": fmt(R[key]["map_ref"])})
                    A.append({"set": S, "policy": gname, "bucket": b, "predictor": "logged profile L at the plan's arc (privileged)", "n": R[key]["log_profile_at_plan_arc"]["n"],
                              "RMS deg": fmt(R[key]["log_profile_at_plan_arc"])})
                    for k in d["head"]:
                        h = dict(rms=cb(rms_a, eh[k], mask=mk), diff_policy=cb(rms_diff, eh[k], ep, mask=mk), corr=cb(corr, ep, eh[k], mask=mk), r2=cb(r2, ep, eh[k], mask=mk),
                                 rms_at_log_arc=d["cb1"](rms_a, eh4[k], mask=d["bk"][b]))
                        if "blind" in eh and k != "blind":
                            h["diff_blind"] = cb(rms_diff, eh[k], eh["blind"], mask=mk)
                        R[key]["heads"][k] = h
                        A.append({"set": S, "policy": gname, "bucket": b, "predictor": f"head {k}", "n": h["rms"]["n"], "RMS deg": fmt(h["rms"]), "minus policy": fmt(h["diff_policy"], "{:+.2f}"),
                                  "minus blind": fmt(h.get("diff_blind"), "{:+.2f}"), "corr(e_head, e_policy)": fmt(h["corr"]), "R2": fmt(h["r2"], "{:.3f}"),
                                  "predicted gain 0.93 x R2": f"{QS_GAIN * h['r2']['v']:+.2f}", "RMS at the logged 4 s arc (privileged arc)": fmt(h["rms_at_log_arc"])})
                TOK[f"{S}|{gname}"] = dict(ep=ep, dy=np.tile(d["dy"], len(mods)), **{f"eh|{k}": v for k, v in eh.items()})
        # ---- selection on H, gate on T
        selH = {arm: R["H|CF5f0|>45"]["heads"][f"{arm} s0"]["rms"]["v"] for arm in ARMS if f"{arm} s0" in DS["H"]["head"]}
        sel = min(selH, key=selH.get)
        hk = f"{sel} s0"
        g1, g3 = R["T|SH30-F|>45"]["heads"][hk], R["T|GH0-F (pilot)|all"]["heads"][hk]
        gate = dict(selected_arm=sel, selection_rms_H_gt45=selH,
                    G1=dict(rms_head=g1["rms"], rms_policy=R["T|SH30-F|>45"]["policy"], diff=g1["diff_policy"], met=bool(g1["diff_policy"]["hi"] < 0)),
                    G2=dict(diff_blind=g1.get("diff_blind"), met=bool(g1.get("diff_blind") is not None and g1["diff_blind"]["hi"] < 0)),
                    G3=dict(r2=g3["r2"], corr=g3["corr"], predicted_gain=QS_GAIN * g3["r2"]["v"], line_r2=R2_PASS, met=bool(g3["r2"]["v"] >= R2_PASS),
                            band="pass" if g3["r2"]["v"] >= R2_PASS else ("marginal" if g3["r2"]["v"] >= R2_FAIL else "clear fail")))
        gate["stage1"] = bool(gate["G1"]["met"] and gate["G2"]["met"] and gate["G3"]["met"])
        gate["failed_part"] = None if gate["stage1"] else ("accuracy" if not (gate["G1"]["met"] and gate["G2"]["met"]) else "independence")
        # ---- (a) fixed arc lengths, equal arc: head vs the logged profile, next to the policy's plan curve
        Fx = []
        for S, gs in groups.items():
            d = DS[S]
            for gname, mods in gs.items():
                for s_ in FIXED:
                    j = int(np.flatnonzero(np.isclose(GRID, s_))[0])
                    lab = d["lab"]["L"][:, j].astype(np.float64)
                    (epc,), bk, cb = pooled(S, mods, lambda m: (deg(wrap(d["pol"][m]["prof"][:, j] - lab)),))
                    for k in (hk, "blind"):
                        if k not in d["head"]:
                            continue
                        ehc = np.tile(deg(wrap(d["head"][k][:, j] - lab)), len(mods))
                        for b, mk in bk.items():
                            both = mk & np.isfinite(epc)
                            r_ = dict(head=cb(rms_a, ehc, mask=both), policy=cb(rms_a, epc, mask=both), diff=cb(rms_diff, ehc, epc, mask=both), corr=cb(corr, epc, ehc, mask=both),
                                      r2=cb(r2, epc, ehc, mask=both), head_all_covered=d["cb1"](rms_a, deg(wrap(d["head"][k][:, j] - lab)), mask=d["bk"][b]))
                            R[f"fixed|{S}|{gname}|{k}|{s_:g}|{b}"] = r_
                            Fx.append({"set": S, "policy": gname, "head": k, "arc m": f"{s_:g}", "bucket": b, "n (plan reaches the arc)": r_["head"]["n"], "head RMS deg": fmt(r_["head"]),
                                       "plan curve RMS deg": fmt(r_["policy"]), "head minus plan": fmt(r_["diff"], "{:+.2f}"), "corr": fmt(r_["corr"], ci=False), "R2": fmt(r_["r2"], "{:.3f}", ci=False),
                                       "head RMS, every token the log covers": fmt(r_["head_all_covered"])})
        # ---- (b) turn-in shift
        Bt = []
        ct = cannot_turn(DS["T"]["names"], DS["T"]["fut"])
        for S, gs in groups.items():
            d = DS[S]
            Lab = d["lab"]["L"].astype(np.float64)
            for gname, mods in gs.items():
                (dh, dp), bk, cb = pooled(S, mods, lambda m: (shift_fit(d["head"][hk], Lab, d["pol"][m]["sp"]), shift_fit(d["pol"][m]["prof"], Lab, d["pol"][m]["sp"])))
                sets = {">45": bk[">45"], ">20": bk[">20"]}
                if gname == "SH30-F":
                    sets["SH30 cannot-make-the-turn, > 45"] = np.concatenate([ct[m] for m in mods]) & bk[">45"]
                for sn, mk in sets.items():
                    both = mk & np.isfinite(dh) & np.isfinite(dp)
                    r_ = dict(head_mean=cb(mean_a, dh, mask=both), policy_mean=cb(mean_a, dp, mask=both), head_abs=cb(mean_a, np.abs(dh), mask=both), policy_abs=cb(mean_a, np.abs(dp), mask=both),
                              abs_diff=cb(mean_diff, np.abs(dh), np.abs(dp), mask=both), head_median=float(np.median(dh[both])) if both.any() else np.nan,
                              policy_median=float(np.median(dp[both])) if both.any() else np.nan, corr=cb(corr, dp, dh, mask=both))
                    R[f"shift|{S}|{gname}|{sn}"] = r_
                    Bt.append({"set": S, "policy": gname, "tokens": sn, "n": r_["head_mean"]["n"], "head median delta m": f"{r_['head_median']:+.2f}", "policy median delta m": f"{r_['policy_median']:+.2f}",
                               "head mean delta": fmt(r_["head_mean"], "{:+.2f}"), "policy mean delta": fmt(r_["policy_mean"], "{:+.2f}"), "head mean abs delta": fmt(r_["head_abs"]),
                               "policy mean abs delta": fmt(r_["policy_abs"]), "abs: head minus policy": fmt(r_["abs_diff"], "{:+.2f}"), "corr(delta_head, delta_policy)": fmt(r_["corr"])})
                TOK[f"shift|{S}|{gname}"] = dict(dh=dh, dp=dp)
        # ---- (e) learning curve (arm C, L channel), and the own-label fit of each arm on H (does the head learn its label at all)
        Lc = []
        for fr in (0.125, 0.25, 0.5, 1.0):
            nm = pname("C", fr)
            if not (H1 / "pred" / f"{nm}.npz").exists():
                continue
            row = {"share of fit logs": fr}
            for S, gname, mods in (("H", "CF5f0", POL["H"]), ("T", "SH30-F", POL["T"][:2])):
                d = DS[S]
                pz = pred(nm, S)[1][..., 0]
                (e, ep), bk, cb = pooled(S, mods, lambda m: (deg(wrap(at(pz, d["pol"][m]["sp"]) - d["h4"])), deg(d["pol"][m]["e"])))
                for b in ("all", ">45"):
                    r_ = cb(rms_a, e, mask=bk[b])
                    R[f"lc|{fr}|{S}|{b}"] = r_
                    row[f"{S} {b} RMS deg"] = fmt(r_)
                    R[f"lc_r2|{fr}|{S}|{b}"] = cb(r2, ep, e, mask=bk[b])
            Lc.append(row)
        Ow = []
        for k, (nm, ch) in heads.items():
            for S in ("H", "T"):
                d = DS[S]
                lab = d["lab"]["M" if ch else "L"].astype(np.float64)
                e = deg(wrap(d["head"][k] - lab))[:, 1:17]
                for b in ("all", ">45"):
                    x = e[d["bk"][b]]
                    Ow.append({"head": k, "label": "M" if ch else "L", "set": S, "bucket": b, "RMS deg over the grid 2.5-40 m": f"{np.sqrt(np.nanmean(x ** 2)):.2f}"})
        # ---- write
        RP.mkdir(parents=True, exist_ok=True)
        P("# HEAD1 stage 1: reads and gate\n")
        P(f"Selected arm on (H), > 45 deg RMS at the fold policy's 4 s arc: **{sel}** ({', '.join(f'{k} {v:.2f}' for k, v in selH.items())}). Gate head = `{pname(sel)}` (fold 0, seed 0).\n")
        P("## Gate (navtest)\n")
        P(md([{"line": "G1 accuracy: RMS(head) - RMS(SH30-F plan), > 45 deg, CI high < 0", "read": f"{fmt(g1['rms'])} vs {fmt(R['T|SH30-F|>45']['policy'])}; diff {fmt(g1['diff_policy'], '{:+.2f}')}",
               "met": gate["G1"]["met"]},
              {"line": "G2 above the blind floor: RMS(head) - RMS(blind), > 45 deg, CI high < 0", "read": fmt(g1.get("diff_blind"), "{:+.2f}"), "met": gate["G2"]["met"]},
              {"line": f"G3 independence: R2 >= {R2_PASS:.3f} (0.93 x R2 >= +0.5), all tokens, pilot policy GH0-F", "read": f"R2 {fmt(g3['r2'], '{:.3f}')}, corr {fmt(g3['corr'])}, predicted gain "
               f"{gate['G3']['predicted_gain']:+.2f} ({gate['G3']['band']})", "met": gate["G3"]["met"]}], ["line", "read", "met"]))
        P(f"**Stage 1 {'PASSES' if gate['stage1'] else 'FAILS'}**" + ("" if gate["stage1"] else f" (failed part: {gate['failed_part']})") + "\n")
        P("## (a) (c) (d) Heading error at the plan's own 4 s arc length against the logged 4 s heading\n")
        P(md(A, ["set", "policy", "bucket", "predictor", "n", "RMS deg", "minus policy", "minus blind", "corr(e_head, e_policy)", "R2", "predicted gain 0.93 x R2", "RMS at the logged 4 s arc (privileged arc)"]))
        P("## (a) Fixed arc lengths (equal arc; tokens where the plan and the log both reach the arc)\n")
        P(md(Fx, ["set", "policy", "head", "arc m", "bucket", "n (plan reaches the arc)", "head RMS deg", "plan curve RMS deg", "head minus plan", "corr", "R2", "head RMS, every token the log covers"]))
        P("## (b) Turn-in shift along the arc (late +)\n")
        P(md(Bt, list(Bt[0])))
        P("## (e) Learning curve (arm C, L channel)\n")
        P(md(Lc, list(Lc[-1])) if Lc else "not run\n")
        P("## Own-label fit of every head (held out)\n")
        P(md(Ow, list(Ow[0])))
        (RP / "tables.md").write_text("\n".join(L))
        js = lambda o: json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))  # noqa: E731
        (RP / "reads.json").write_text(json.dumps(js(R), indent=1))
        (RP / "gate.json").write_text(json.dumps(js(gate), indent=1))
        np.savez(RP / "tok.npz", **{f"{k}|{q}": v.astype(np.float32) for k, dct in TOK.items() for q, v in dct.items()})
        run.summary.update(gate=js(gate))
        run.info("\n".join(L[:12]))


def cmd_fig(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as PSY
    PSY.apply()
    src = Path(a.src) if a.src else RP
    R, gate, T = json.loads((src / "reads.json").read_text()), json.loads((src / "gate.json").read_text()), np.load(src / "tok.npz")
    hk, col = f"{gate['selected_arm']} s0", PSY.PALETTE
    out = Path(a.out) if a.out else RP / "figs"
    out.mkdir(parents=True, exist_ok=True)
    # 1: (a) bars per bucket on navtest
    fig, axs = plt.subplots(1, 2, figsize=(PSY.DOUBLE_COLUMN_IN, 2.5), sharey=True)
    for ax, g in zip(axs, ("SH30-F", "GH0-F (pilot)")):
        names = [("policy's plan", lambda r: r["policy"], PSY.BASELINE), ("blind head", lambda r: r["heads"]["blind"]["rms"], col["yellow"])] + \
                [(f"head {arm}", (lambda r, arm=arm: r["heads"][f"{arm} s0"]["rms"]), c) for arm, c in zip(ARMS, (col["blue"], col["green"], col["vermillion"]))] + \
                [("map label (privileged)", lambda r: r["map_ref"], col["purple"])]
        for i, (nm, fn, c) in enumerate(names):
            for j, b in enumerate(BK):
                r = fn(R[f"T|{g}|{b}"])
                ax.bar(j + (i - 2.5) * 0.14, r["v"], 0.13, color=c, label=nm if j == 0 else None, yerr=[[r["v"] - r["lo"]], [r["hi"] - r["v"]]], error_kw=dict(lw=0.5))
        ax.set_xticks(range(3), ["all", "> 20 deg", "> 45 deg"]); ax.set_xlabel(f"navtest, 4 s arc length of {g}"); PSY.bars(ax)
    axs[0].set_ylabel("4 s heading error, RMS (deg)"); axs[0].legend(fontsize=6.5)
    fig.tight_layout(pad=0.4); PSY.save(fig, out / "h1_heading")
    # 2: error scatter on > 45 deg
    fig, axs = plt.subplots(1, 2, figsize=(PSY.DOUBLE_COLUMN_IN, 3.2))
    for ax, g in zip(axs, ("SH30-F", "GH0-F (pilot)")):
        m = T[f"T|{g}|dy"] > 45
        x, y = T[f"T|{g}|ep"][m], T[f"T|{g}|eh|{hk}"][m]
        ax.plot(x, y, ".", ms=1.5, color=col["blue"], alpha=0.5, rasterized=True)
        ax.plot([-60, 60], [-60, 60], color="#999999", lw=0.5)
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect("equal")
        r = R[f"T|{g}|>45"]["heads"][hk]
        ax.set_xlabel(f"{g} plan: 4 s heading error (deg)"); ax.text(0.03, 0.97, f"corr {r['corr']['v']:.2f}, $R^2$ {r['r2']['v']:.2f}", transform=ax.transAxes, va="top")
    axs[0].set_ylabel(f"head {hk}: error at the plan's arc (deg)")
    fig.tight_layout(pad=0.4); PSY.save(fig, out / "h1_scatter")
    # 3: learning curve + fixed arcs
    fig, axs = plt.subplots(1, 2, figsize=(PSY.DOUBLE_COLUMN_IN, 2.5))
    fr = [f for f in (0.125, 0.25, 0.5, 1.0) if f"lc|{f}|T|>45" in R]
    for S, c in (("H", col["orange"]), ("T", col["blue"])):
        for b, ls in ((">45", "-"), ("all", "--")):
            v = [R[f"lc|{f}|{S}|{b}"] for f in fr]
            axs[0].errorbar(fr, [x["v"] for x in v], yerr=[[x["v"] - x["lo"] for x in v], [x["hi"] - x["v"] for x in v]], fmt="o" + ls, ms=2.5, color=c, lw=0.9,
                            label=f"{'held-out navtrain' if S == 'H' else 'navtest'}, {b if b == 'all' else '> 45 deg'}")
    axs[0].set_xscale("log", base=2); axs[0].set_xticks(fr, [f"{100 * f:g}%" for f in fr]); axs[0].set_xlabel("share of the training logs"); axs[0].set_ylabel("head C, 4 s heading error RMS (deg)")
    axs[0].legend(fontsize=6.5)
    for k, c, nm in ((hk, col["blue"], f"head {hk}"), ("blind", col["yellow"], "blind head")):
        v = [R.get(f"fixed|T|SH30-F|{k}|{s:g}|>45") for s in FIXED]
        ok = [i for i, x in enumerate(v) if x and x["head"]["n"] > 50]
        axs[1].plot([FIXED[i] for i in ok], [v[i]["head"]["v"] for i in ok], "o-", ms=2.5, color=c, label=nm)
        if k == hk:
            axs[1].plot([FIXED[i] for i in ok], [v[i]["policy"]["v"] for i in ok], "o-", ms=2.5, color=PSY.BASELINE, label="SH30-F plan curve")
    axs[1].set_xlabel("arc length (m)"); axs[1].set_ylabel("heading error at equal arc, > 45 deg (deg)"); axs[1].legend(fontsize=6.5)
    fig.tight_layout(pad=0.4); PSY.save(fig, out / "h1_curves")
    print(out)


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("feats", "read"):
        cli_args(sub.add_parser(c))
    f = sub.add_parser("fig"); f.add_argument("--src", default=""); f.add_argument("--out", default="")
    a = ap.parse_args()
    {"feats": cmd_feats, "read": cmd_read, "fig": cmd_fig}[a.cmd](a)
