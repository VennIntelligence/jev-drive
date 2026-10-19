#!/usr/bin/env python3
"""SLOW-RESCORE (CPU, no new rollouts): frozen board-mix PCS (decision 382) next to the arena held mean for stored native reads, with the
slow-partial and near-1 counts, plus the cause breakdown of JTN-L18-s1 native's slow partials (decision 372 labels).

Mac, in the pcs_local env (drive-irt not needed, scipy + pyarrow):
  tmp/pcs_local/.venv/bin/python experiments/defect_fix/scripts/slow_rescore.py --rows tmp/pcs_local/rows_1.csv.gz tmp/pcs_local/rows_3.csv.gz \
      tmp/slow/cont_rows.csv.gz [--native-gap tmp/native_gap] [--out experiments/defect_fix/results/slow]

Definitions (per window of one read, arena held, route-fail in either arm dropped):
- zero: score <= 1e-4. near-1: 0.98 <= score < 1 - 1e-4 (arena type "ok" but not exact 1).
- slow partial: 0 < score < 0.98, no collision (arena types slow / creep / no-launch: progress short of 0.8 of the human), window stratum not
  static; "slow" = non-launch strata (decision 382's "slow progress" class), "launch-short" = launch stratum.
- PCS: items frozen (results/pcs_local/boardmix.npz, boardmix-g15.npz), one read = one more submission, ability = MAP under the fitted
  ability mean / sd (on a 0.01 grid, parabolic refinement), mapped to board PCS through the 6 anchors (pcs_local.Scored).
  A read missing items (the d378 6-chunk pilot) is scored two ways: "fill" = the missing items take JTN-L18-s0 native's scores in both arms
  (the difference is then the subset's alone, at a realistic ability level), and "subset" = MAP on the read's items only.
  CI: log-clustered bootstrap (2 000) of the paired PCS difference with items fixed; the mean CI is jevdrive.stats.paired by log.
- Gate for a FREE-M dose pilot, fixed before this ran: FREEM1K - JTNS1K >= +30 PCS (fill) under board-mix AND g15, subset MAP same sign.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/alpasim_proxy/scripts"), str(REPO / "experiments/defect_fix/scripts")]
from jevdrive import stats  # noqa: E402
import pcs_local as PL  # noqa: E402

FITS = {"boardmix": REPO / "experiments/alpasim_proxy/results/pcs_local/boardmix.npz",
        "g15": REPO / "experiments/alpasim_proxy/results/pcs_local/boardmix-g15.npz"}
PAIRS = [("FREEM1K-L18-s0@off", "JTNS1K-L18-s0@off", "d378 pilot (6 chunks)"),
         ("BNC2K-L18-s1@off", "JTN-L18-s1@off", "d379 BN control +2K, s1"),
         ("BNC2K-L18-s0", "JTN-L18-s0", "d379 suppl. 1, s0"),
         ("BNC4K-L18-s1", "JTN-L18-s1", "d379 suppl. 1, 4K s1"),
         ("JTN-L18-s1@off", "RSG-L18-s0", "native JTN1 vs RSG"),
         ("JTN-L18-s1@off", "NTV0-L18-s0@off", "native JTN1 vs NTV0"),
         ("NTV0-L18-s0@off", "RSG-L18-s0", "native NTV0 vs RSG")]
FILL = "JTN-L18-s0@off"
NB = 2000
SLOWT = ("slow", "creep", "no-launch")


class Frozen:
    def __init__(self, path, sc):
        z = PL.Scored(dict(np.load(path, allow_pickle=True)), sc)
        self.F, self.items, self.mu, self.sd, self.A = z.f, z.items, z.mu, z.sd, z.A
        self.grid = np.arange(-8, 8.0001, 0.01)
        self.prior = -0.5 * ((self.grid - self.mu) / self.sd) ** 2

    def L(self, y):
        """per-item log-likelihood on the grid [G, N]; missing items contribute 0."""
        y = np.asarray(y, float)
        out = np.zeros((len(self.grid), len(y)))
        ok = np.isfinite(y)
        Fo = {k: np.asarray(self.F[k])[ok] for k in ("a", "b", "g0", "g1", "prec")}
        for i in range(0, len(self.grid), 200):
            th = self.grid[i:i + 200]
            out[i:i + 200, ok] = item_ll(th, y[ok], Fo)
        return out

    def pcs(self, Lm, w):
        """PCS at item weights w ([N] or [N, B])."""
        tot = Lm @ w
        tot = tot + (self.prior[:, None] if tot.ndim == 2 else self.prior)
        j = np.clip(np.argmax(tot, 0), 1, len(self.grid) - 2)
        if tot.ndim == 1:
            y0, y1, y2 = tot[j - 1], tot[j], tot[j + 1]
        else:
            c = np.arange(tot.shape[1]); y0, y1, y2 = tot[j - 1, c], tot[j, c], tot[j + 1, c]
        den = y0 - 2 * y1 + y2
        off = np.clip(np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0), -1, 1)   # at the grid edge (MAP >= 8) the vertex is outside: cap at one step
        th = self.grid[j] + off * 0.01
        return np.polyval(self.A, th)


def item_ll(th, y, F):
    from scipy.special import betaln, expit, log_expit
    th = np.asarray(th, float)[:, None]
    a, b, g0, g1, o = F["a"][None], F["b"][None], F["g0"][None], F["g1"][None], 2 * np.log(F["prec"])[None]
    at = a * th; eta = at - b
    al, be = np.exp(np.clip((eta + o) / 2, -13.8, 13.8)), np.exp(np.clip((-eta + o) / 2, -13.8, 13.8))
    z0, z1 = g0 - at, g1 - at
    lp0, lp1 = log_expit(z0), log_expit(-z1)
    lp_c = np.log(np.clip(expit(z1) - expit(z0), 1e-12, 1))
    yy = np.clip(y, 1e-6, 1 - 1e-6)[None]
    lb = (al - 1) * np.log(yy) + (be - 1) * np.log1p(-yy) - betaln(al, be)
    is0, is1 = (y <= 1e-4)[None], (y >= 1 - 1e-4)[None]
    return np.where(is0, lp0, np.where(is1, lp1, lp_c + lb))


def classes(r, S):
    st = S.stratum.reindex(r.index)
    sc, ty = r.score, r.type.astype(str)
    short = (sc > 1e-4) & (sc < 0.98) & ty.isin(SLOWT) & (st != "static")
    return dict(zeros=int((sc <= 1e-4).sum()), slow=int((short & (st != "launch")).sum()), launch_short=int((short & (st == "launch")).sum()),
                near1=int(((sc >= 0.98) & (sc < 1 - 1e-4)).sum()), exact1=int((sc >= 1 - 1e-4).sum()),
                af_front=int(ty.str.startswith("AF-front").sum()), static_zeros=int(((sc <= 1e-4) & (st == "static")).sum()))


def boot_w(logs, rng):
    codes, uniq = pd.factorize(logs)
    draw = rng.integers(len(uniq), size=(NB, len(uniq)))
    cnt = np.stack([np.bincount(d, minlength=len(uniq)) for d in draw], 1)       # [U, B]
    return codes, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--native-gap", default=str(REPO / "tmp/native_gap"))
    ap.add_argument("--out", default=str(REPO / "experiments/defect_fix/results/slow"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(Path(a.native_gap) / "set.csv.gz").drop_duplicates("id").set_index("id")
    held = S.index[S.split == "held"]
    R = pd.concat([pd.read_csv(f) for f in a.rows]).drop_duplicates(["tag", "id"]).set_index(["tag", "id"])
    rd = {t: g.droplevel(0) for t, g in R.groupby(level=0)}
    sc_pool, _, _ = PL.load(REPO / "experiments/alpasim_proxy/results/pcs_local")
    FZ = {k: Frozen(p, sc_pool) for k, p in FITS.items()}
    items = FZ["boardmix"].items
    assert all(list(f.items) == items for f in FZ.values())
    vec = lambda t: rd[t].score.reindex(items).to_numpy(float)  # noqa: E731
    rng = np.random.default_rng(0)
    res, absrows = [], {}
    Lcache = {}

    def Lof(k, key, y):
        if (k, key) not in Lcache:
            Lcache[(k, key)] = FZ[k].L(y)
        return Lcache[(k, key)]

    for ta, tb, what in PAIRS:
        A, B = rd[ta], rd[tb]
        ids = A.index.intersection(B.index).intersection(held)
        ids = ids[(A.type.reindex(ids) != "route-fail").values & (B.type.reindex(ids) != "route-fail").values]
        m = stats.paired(A.score.reindex(ids).values, B.score.reindex(ids).values, S.log.reindex(ids).values)
        row = dict(pair=f"{ta} - {tb}", what=what, windows=len(ids), mean_a=m["mean_a"], mean_b=m["mean_b"], d_mean=m["mean"], d_lo=m["lo"], d_hi=m["hi"],
                   a=classes(A.loc[ids], S), b=classes(B.loc[ids], S))
        ya, yb = vec(ta), vec(tb)
        sub = np.isfinite(ya) & np.isfinite(yb)
        full = bool(sub.all())
        logs = S.log.reindex(items).to_numpy()
        codes, cnt = boot_w(logs[sub], rng)
        for k, fz in FZ.items():
            modes = {"full": (ya, yb)} if full else {"fill": tuple(np.where(sub, y, vec(FILL)) for y in (ya, yb)), "subset": (ya, yb)}
            for mode, (y1, y2) in modes.items():
                L1, L2 = Lof(k, (ta, mode), y1), Lof(k, (tb, mode), y2)
                w0 = np.where(sub, 1.0, 1.0 if mode == "fill" else 0.0)
                p1, p2 = fz.pcs(L1, w0), fz.pcs(L2, w0)
                W = np.repeat(w0[:, None], NB, 1); W[sub] = cnt[codes]
                d = fz.pcs(L1, W) - fz.pcs(L2, W)
                lo, hi = np.quantile(d, [0.025, 0.975])
                row[f"pcs_{k}_{mode}"] = dict(a=float(p1), b=float(p2), d=float(p1 - p2), lo=float(lo), hi=float(hi), items=int(sub.sum()))
                if mode in ("full",):
                    absrows[ta] = absrows.get(ta, {}) | {k: float(p1)}; absrows[tb] = absrows.get(tb, {}) | {k: float(p2)}
        res.append(row)
        print(json.dumps(row)[:600], flush=True)
    gate = None
    r0 = res[0]
    gate = dict(boardmix=r0["pcs_boardmix_fill"]["d"], g15=r0["pcs_g15_fill"]["d"], subset_boardmix=r0["pcs_boardmix_subset"]["d"],
                subset_g15=r0["pcs_g15_subset"]["d"])
    gate["pass"] = bool(gate["boardmix"] >= 30 and gate["g15"] >= 30 and gate["subset_boardmix"] > 0 and gate["subset_g15"] > 0)
    causes = slow_causes(a.native_gap, rd["JTN-L18-s1@off"], S, held, FZ, vec("JTN-L18-s1@off"), items, Lof)
    json.dump(dict(pairs=res, abs=absrows, gate=gate, causes=causes), open(out / "rescore.json", "w"), indent=1)
    print(json.dumps(gate)); print(json.dumps(causes, indent=1))


def slow_causes(src, r, S, held, FZ, y, items, Lof):
    """JTN-L18-s1 native slow / launch-short windows by cause (first match), from native_gap's decision table (tag JTN1)."""
    import native_gap_report as NG
    D, _, _, _ = NG.load(Path(src))
    D = D[(D.tag == "JTN1")]
    ids = r.index.intersection(held)
    ids = ids[(r.type.reindex(ids) != "route-fail").values]
    st = S.stratum.reindex(ids)
    sc, ty = r.score.reindex(ids), r.type.reindex(ids).astype(str)
    short = ids[((sc > 1e-4) & (sc < 0.98) & ty.isin(SLOWT) & (st != "static")).values]
    q = D[D.id.isin(short) & D.valid & (D.k >= 2)].copy()
    q["ratio"] = q.v / q.vh
    rows = []
    for i in short:
        g = q[q.id == i].sort_values("k")
        mv = g[g.vh >= 2]
        stop = bool(((g.vh >= 2) & (g.v < 0.3) & (g.v.cummax().shift(1) >= 1)).any())
        lead = (mv.lead_gap < 25).mean() if len(mv) else np.nan
        last = mv.ratio.iloc[-1] if len(mv) else np.nan
        first = mv.ratio.iloc[0] if len(mv) else np.nan
        if S.loc[i, "v0"] < 1.0:
            c = "late launch (human starts < 1 m/s)"
        elif stop:
            c = "stop / lock (ego < 0.3 m/s where the human moves >= 2)"
        elif len(mv) and lead >= 0.5:
            c = "following a lead (lead < 25 m on >= half the decisions)"
        elif len(mv) and last < 0.9:
            c = "moving ratchet (free / turn, last ratio < 0.9)"
        else:
            c = "other (ratio >= 0.9 or no valid moving decision)"
        rows.append(dict(id=i, stratum=st[i], score=float(sc[i]), type=ty[i], cause=c, ratio_first=first, ratio_last=last, lead_share=lead,
                         n_dec=len(mv), turn=bool((mv.turn20 >= 15).mean() >= 0.5) if len(mv) else False))
    C = pd.DataFrame(rows)
    C.to_csv(REPO / "experiments/defect_fix/results/slow/slow_causes_jtn1.csv", index=False, float_format="%.4f")
    out = {}
    for k, fz in FZ.items():
        L = Lof(k, ("JTN-L18-s1@off", "full"), y)
        w = np.ones(len(items))
        base = fz.pcs(L, w)
        out[f"base_{k}"] = float(base)
        for grp in [None] + sorted(C.cause.unique()) + ["all slow (non-launch)", "all short"]:
            if grp is None:
                continue
            sel = C if grp == "all short" else C[C.stratum != "launch"] if grp == "all slow (non-launch)" else C[C.cause == grp]
            y2 = y.copy(); y2[np.isin(items, sel.id)] = 1.0
            out.setdefault(grp, {})[k] = float(fz.pcs(fz.L(y2), w) - base)
    for grp, g in C.groupby("cause"):
        out[grp] |= dict(windows=len(g), nonlaunch=int((g.stratum != "launch").sum()), points=float((1 - g.score).sum()),
                         ratio_last=float(g.ratio_last.median()), strata=g.stratum.value_counts().to_dict())
    out["all short"] |= dict(windows=len(C), points=float((1 - C.score).sum()))
    out["all slow (non-launch)"] |= dict(windows=int((C.stratum != "launch").sum()))
    return out


if __name__ == "__main__":
    main()
