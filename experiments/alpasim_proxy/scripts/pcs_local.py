#!/usr/bin/env python3
"""PCS-LOCAL: the AlpaSim nuPlan leaderboard's Policy Capability Score (Drive-IRT zoib = Molenaar zero-and-one-inflated Beta 2PL,
drive-irt 82ddd5a) refit on the ARENA held split from every stored arena read, to see which scenes cost us PCS.

Runs on the Mac in an isolated uv env with drive-irt installed (tmp/pcs_local/.venv, `uv pip install <drive-irt 82ddd5a> pandas`).
Inputs: `arena.py collect` tables of every arena read root on the box (rows_*.csv.gz) and the arena set (results/arena/set.csv.gz).

  pcs_local.py build --rows DIR --out DIR          held valid windows x subjects -> DIR/{scores.csv.gz, types.csv.gz, windows.csv.gz}
  pcs_local.py fit --data DIR --out DIR            official-setting fits (pool variants, 1PL, seeds) -> DIR/fits/<variant>.npz + fits.csv
  pcs_local.py score --rows ROWS.csv.gz [--subset] a new arena held read (arena.py collect) scored against the frozen board-mix items
                                                   (--subset: MAP on the windows every tag has, e.g. a 6-chunk read; also slow / near-1 counts)
  pcs_local.py report --data DIR --out DIR [--primary POOL]  calibration against the board, item parameters by stratum, deficit decomposition,
                                                   counterfactuals, subject-count curve -> DIR/report.md + tables

Official settings (e2e_challenge/local_evaluation/evaluate.py @ alpasim 0bb4c4b): BetaIrt(epochs=1000, lr=0.1, lr_decay=0.975,
num_particles=16, seed=42, clip_norm=10, average_iterate_tail_percentage=0.025), 2PL; zoib needs >= S + 5N observations (else the
arithmetic mean); a run whose scene set differs from the canonical set is excluded (complete matrix). PCS = positive affine map of the
ability fixed by two anchor subjects; the board's anchors are not published, so local PCS is mapped onto the board by least squares on
the subjects whose private PCS is known (the 6 anchors of decision 355 + our two 10-19 probes).
Counterfactuals hold the item parameters fixed and re-estimate one subject's ability (MAP under a Normal prior with the fitted ability
mean / sd), the way one submission among 130 barely moves the items.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OFFICIAL = dict(epochs=1000, lr=0.1, lr_decay=0.975, log_every=10_000, num_particles=16, clip_norm=10.0, average_iterate_tail_percentage=0.025)

# arena read tag -> subject label; family: ours (openpilot-student line) / third (third-party or host-like)
LABEL = {"RSG-L18-s0": "RSG", "NR1F-L18-s0": "NR1F", "NR1FB10-L18-s0": "NR1FB10", "NR1HV-L18-s0": "NR1HV", "NR1HV2-L18-s0": "NR1HV2",
         "YR10m10-F-s0": "YR", "APY10m10-AB-s0": "APY", "DD-sample": "DD", "LTF-sample": "LTF", "starter": "go-straight", "G0014": "garage",
         "full": "RSG+full", "lead": "RSG+lead", "leash3": "RSG+leash3", "stop3": "RSG+stop3",
         "RSG-L18-s0@stop3": "RSG+stop3+ACC", "JTN-L18-s0@stop3#acc": "JTN-s0+stop3+ACC", "JTN-L18-s1@warp-const": "JTN-s1+warpconst"}
THIRD = {"DD", "LTF", "go-straight", "garage"}
# private board 2026-10-19 (tmp lb_nuplan_1019.json): PCS / mean scene score of the subjects we have arena reads for
BOARD = {"garage": (1543, 0.848), "YR": (1030, 0.800), "go-straight": (1300, 0.799), "DD": (1080, 0.772), "LTF": (1014, 0.738),
         "APY": (779, 0.640), "JTN-s1": (1325, 0.9155), "JTN-s1+stop3": (1463, 0.950)}
ANCHORS6 = ["garage", "YR", "go-straight", "DD", "LTF", "APY"]
ZERO_T = ("AF", "offroad", "corridor")


def label(tag):
    if tag in LABEL:
        return LABEL[tag]
    t, _, v = tag.partition("@")
    short = t.replace("-L18-s", "-s")
    return short if v == "off" else f"{short}+{v}"


# ---------------------------------------------------------------- build
def cmd_build(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(HERE.parent / "results/arena/set.csv.gz")
    R = []
    for f in sorted(glob.glob(f"{a.rows}/rows_*.csv.gz")):
        r = pd.read_csv(f)
        if "nuacc/jtn" in open(f"{a.rows}/sources.txt").read().splitlines()[int(Path(f).stem.split("_")[1].split(".")[0]) - 1]:
            r["tag"] = r.tag + "#acc"
        R.append(r)
    R = pd.concat(R)
    R = R[~R.tag.str.endswith("-id")]                                   # 12-window identity checks
    main = R[R.tag.isin(["RSG-L18-s0", "YR10m10-F-s0", "APY10m10-AB-s0", "DD-sample", "starter", "NR1F-L18-s0"])]
    bad = set(main[main.type == "route-fail"].id)                         # arena prereg 4.1: dropped where a read failed before driving
    held = S[(S.split == "held") & ~S.id.isin(bad)]
    R = R[R.id.isin(set(held.id))].copy()
    R["subject"] = R.tag.map(label)
    R["type"] = R.type.str.replace(r"AF-(front|side|rear)", "AF", regex=True)
    sc = R.pivot_table(index="subject", columns="id", values="score", aggfunc="first")
    ty = R.pivot_table(index="subject", columns="id", values="type", aggfunc="first")
    sc = sc[held.id]; ty = ty[held.id]
    sc.to_csv(out / "scores.csv.gz"); ty.to_csv(out / "types.csv.gz")
    w = held.set_index("id")[["log", "stratum", "v0", "v_end", "maxdisp", "dyaw", "lead", "lead_gap", "near"]]
    w.to_csv(out / "windows.csv.gz")
    n = sc.notna().sum(1).sort_values()
    print(f"held valid windows {sc.shape[1]}, subjects {sc.shape[0]}"); print(n.to_string())


# ---------------------------------------------------------------- fitting
def load(data):
    sc = pd.read_csv(Path(data) / "scores.csv.gz", index_col=0)
    ty = pd.read_csv(Path(data) / "types.csv.gz", index_col=0)
    w = pd.read_csv(Path(data) / "windows.csv.gz", index_col=0)
    w["cell"] = w.stratum.where(w.stratum != "static", np.where((w.lead == True) & (w.lead_gap <= 8), "static-lead", "static-free"))  # noqa: E712
    return sc, ty.reindex_like(sc), w.loc[sc.columns]


def fit(M, subjects, items, seed=42, fix=False, **kw):
    import io, contextlib, logging
    from drive_irt.irt import BetaIrt
    logging.disable(logging.INFO)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        r = BetaIrt(seed=seed, fix_discrimination=fix, **(OFFICIAL | kw)).fit(np.asarray(M, np.float32), list(subjects), list(items))
    th = dict(zip(r.subject_ids, r.scores)); sd = dict(zip(r.subject_ids, r.score_uncertainty))
    return dict(theta=np.array([th[s] for s in subjects]), theta_sd=np.array([sd[s] for s in subjects]), a=r.discriminations, b=r.difficulties,
                prec=r.precision, g0=r.g0_loc, g1=r.g1_loc, loss=r.best_loss)


def guard(M):
    S, N = M.shape
    return int(np.isfinite(M).sum()) >= S + 5 * N


BOARD_MIX = {"garage": 30, "LTF": 10, "DD": 5, "go-straight": 2, "APY": 1, "YR": 1, "JTN-s1": 1, "JTN-s1+stop3": 1}
# the 130-entry board 2026-10-19 by family: ~30 py123d-garage / 0014 clones and garage-based team entries, ~10 LTF / GTRS / TransFuser-like,
# ~5 DiffusionDrive, 2 go-straight, and our 4 entries (APY, YR, JTN-s1 native, JTN-s1 + stop3); WAM / VaVAM / world-model entries have no
# arena stand-in and are left out. Replicated rows are exact copies (the board itself holds 6 identical 0014 runs).


def pool(sc, name):
    """-> (row subjects with repeats, item columns). The canonical item set = garage's 2106 windows (complete matrix, as evaluate.py)."""
    cols = sc.columns[sc.loc["garage"].notna()]
    comp = [s for s in sc.index if sc.loc[s, cols].notna().all()]
    if name == "complete":
        return comp, cols
    if name == "no-third":                                           # without host-like / third-party subjects
        return [s for s in comp if s not in THIRD], cols
    if name == "dedup":                                              # one subject per near-identical family member
        return ["RSG", "YR", "APY", "NR1F", "NTV0-s0", "JTN-s1", "JTN-s1+stop3", "RSG+stop3"] + sorted(THIRD), cols
    if name.startswith("garage"):                                    # every complete subject + garage repeated to k rows
        return comp + ["garage"] * (int(name[6:]) - 1), cols
    if name.startswith("boardmix"):                                  # board composition; boardmix-g<k>: garage weight k
        mix = dict(BOARD_MIX)
        if "-g" in name:
            mix["garage"] = int(name.split("-g")[1])
        return [s for s, k in mix.items() for _ in range(k)], cols
    raise SystemExit(f"unknown pool {name}")


def matrix(sc, subs, cols):
    seen = {}
    names = []
    for s in subs:
        names.append(s if s not in seen else f"{s}#{seen[s]}"); seen[s] = seen.get(s, 0) + 1
    return sc.loc[list(subs), cols].to_numpy(float), names


VARIANTS = {  # name: (pool, fix_discrimination, seed)
    "official": ("complete", False, 42),
    "seed1": ("complete", False, 1), "seed2": ("complete", False, 2), "seed3": ("complete", False, 3),
    "1pl": ("complete", True, 42),
    "no-third": ("no-third", False, 42),
    "dedup": ("dedup", False, 42),
    "garage10": ("garage10", False, 42),
    "garage30": ("garage30", False, 42),
    "boardmix": ("boardmix", False, 42),
    "boardmix-s1": ("boardmix", False, 1), "boardmix-s2": ("boardmix", False, 2),
    "boardmix-1pl": ("boardmix", True, 42),
    "boardmix-g15": ("boardmix-g15", False, 42),
    "boardmix-g60": ("boardmix-g60", False, 42),
}


def cmd_fit(a):
    sc, _, _ = load(a.data)
    out = Path(a.out) / "fits"; out.mkdir(parents=True, exist_ok=True)
    for v in (a.variants.split(",") if a.variants else VARIANTS):
        pname, fix, seed = VARIANTS[v]
        subs, cols = pool(sc, pname)
        M, names = matrix(sc, subs, cols)
        assert guard(M), v
        f = fit(M, names, cols, seed=seed, fix=fix)
        np.savez(out / f"{v}.npz", subjects=np.array(names), items=np.array(cols, dtype=str), **f)
        rows = []
        for s, t, d in zip(names, f["theta"], f["theta_sd"]):
            if "#" not in s:
                rows.append(dict(variant=v, subject=s, theta=t, theta_sd=d, mean=np.nanmean(sc.loc[s, cols])))
        print(v, len(names), "rows", len(cols), "items", flush=True)
        pd.DataFrame(rows).to_csv(Path(a.out) / f"fits/{v}.csv", index=False)


# ---------------------------------------------------------------- one-subject ability under fixed items
def loglik(theta, y, F):
    """zoib log-likelihood of response vector y (len N) at abilities theta (len T) under fitted items F -> [T]."""
    from scipy.special import betaln, expit, log_expit
    th = np.asarray(theta, float)[:, None]
    a, b, g0, g1, o = F["a"][None], F["b"][None], F["g0"][None], F["g1"][None], 2 * np.log(F["prec"])[None]
    at = a * th; eta = at - b
    al, be = np.exp(np.clip((eta + o) / 2, -13.8, 13.8)), np.exp(np.clip((-eta + o) / 2, -13.8, 13.8))
    z0, z1 = g0 - at, g1 - at
    lp0, lp1 = log_expit(z0), log_expit(-z1)
    lp_c = np.log(np.clip(expit(z1) - expit(z0), 1e-12, 1))
    yy = np.clip(y, 1e-6, 1 - 1e-6)[None]
    lb = (al - 1) * np.log(yy) + (be - 1) * np.log1p(-yy) - betaln(al, be)
    is0, is1 = (y <= 1e-4)[None], (y >= 1 - 1e-4)[None]
    ok = np.isfinite(y)[None]
    ll = np.where(is0, lp0, np.where(is1, lp1, lp_c + lb))
    return np.where(ok, ll, 0).sum(1)


GRID = np.linspace(-8, 8, 161)


def theta_map(y, F, mu, sd):
    """MAP ability of response vector y under fixed items: coarse grid, then bounded Brent around the best grid point."""
    from scipy.optimize import minimize_scalar
    nlp = lambda t: -(loglik(np.atleast_1d(t), y, F) - 0.5 * ((np.atleast_1d(t) - mu) / sd) ** 2)  # noqa: E731
    t0 = GRID[np.argmin(nlp(GRID))]
    return minimize_scalar(lambda t: float(nlp(t)[0]), bounds=(t0 - 0.1, t0 + 0.1), method="bounded", options=dict(xatol=1e-5)).x


# ---------------------------------------------------------------- report
def kendall(x, y):
    from scipy.stats import kendalltau
    return kendalltau(x, y).statistic


def outcome(y, t, c):
    """Outcome class of one cell: exact 0 by cause, partial by cause, exact 1."""
    if y >= 1 - 1e-4:
        return "1"
    if y <= 1e-4:
        if "AF" in t:
            return "0 at-fault collision"
        if "offroad" in t:
            return "0 offroad"
        return "0 corridor, static window" if c.startswith("static") else "0 corridor, moving window"
    if c == "launch":
        return "partial, launch window (late / no launch, creep, rear-hit)"
    if t == "rear-hit":
        return "partial, struck from behind"
    if t == "ok":
        return "partial >= 0.98"
    return "partial, slow progress"


class Scored:
    """One fit (items fixed) + every subject's ability: the fitted posterior mean for pool rows, the MAP under the fitted items for the
    rest (scored as one more submission), and the least-squares map of ability onto board PCS through the 6 private anchors."""

    def __init__(self, f, sc):
        self.f, self.items = f, list(f["items"])
        names = [str(s) for s in f["subjects"]]
        self.mu, self.sd = float(f["theta"].mean()), float(f["theta"].std())
        self.fitted = {s: float(t) for s, t in zip(names, f["theta"]) if "#" not in s}
        self.theta = {s: self.fitted.get(s, self.map(sc.loc[s, self.items].to_numpy(float))) for s in sc.index if sc.loc[s, self.items].notna().all()}
        self.check = {s: self.map(sc.loc[s, self.items].to_numpy(float)) - t for s, t in self.fitted.items()}
        x = np.array([self.theta[k] for k in ANCHORS6]); y = np.array([BOARD[k][0] for k in ANCHORS6])
        self.A = np.polyfit(x, y, 1)
        self.pcs = {s: float(np.polyval(self.A, t)) for s, t in self.theta.items()}

    def map(self, y):
        return float(theta_map(y, self.f, self.mu, self.sd))

    def pcs_of(self, y):
        return float(np.polyval(self.A, self.map(y)))


def cmd_report(a):
    from scipy.stats import spearmanr
    sc, ty, w = load(a.data)
    out = Path(a.out)
    L = []
    P = lambda *s: L.append(" ".join(str(x) for x in s))  # noqa: E731
    vs = [v for v in VARIANTS if (out / f"fits/{v}.npz").exists()]
    S = {v: Scored(dict(np.load(out / f"fits/{v}.npz", allow_pickle=True)), sc) for v in vs}
    known = list(BOARD)
    items0 = S["official"].items
    mean = {s: float(sc.loc[s, items0].mean()) for s in sc.index}

    # 1. calibration
    P("## 1. Does local PCS reproduce the board?\n")
    P(f"Arena held windows in the fit: {len(items0)} (garage's complete set). Board-mapped PCS = least squares of ability on the 6 anchors' "
      "private PCS; JTN-s1 native / + stop3 are out of sample. Kendall tau over the 8 subjects with a private PCS.\n")
    P("| variant | rows (subjects) | tau, 8 known | tau, 6 anchors | RMSE on 6 anchors | JTN-s1 | JTN-s1 + stop3 | garage | go-straight | stop3 - garage (board -80) | native - go-straight (board +25) |")
    P("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for v, z in S.items():
        th, pc = z.theta, z.pcs
        rmse = np.sqrt(np.mean([(pc[k] - BOARD[k][0]) ** 2 for k in ANCHORS6]))
        P(f"| {v} | {len(z.f['subjects'])} ({len(z.fitted)}) | {kendall([th[k] for k in known], [BOARD[k][0] for k in known]):+.2f} | "
          f"{kendall([th[k] for k in ANCHORS6], [BOARD[k][0] for k in ANCHORS6]):+.2f} | {rmse:.0f} | {pc['JTN-s1']:.0f} | {pc['JTN-s1+stop3']:.0f} | "
          f"{pc['garage']:.0f} | {pc['go-straight']:.0f} | {pc['JTN-s1+stop3'] - pc['garage']:+.0f} | {pc['JTN-s1'] - pc['go-straight']:+.0f} |")
    P(f"\nBoard: JTN-s1 1325, JTN-s1 + stop3 1463, garage 1543, go-straight 1300. The arena held mean orders the 8 with tau "
      f"{kendall([mean[k] for k in known], [BOARD[k][0] for k in known]):+.2f} (board mean scene score: "
      f"{kendall([BOARD[k][1] for k in known], [BOARD[k][0] for k in known]):+.2f}).")
    worst = max(abs(d) for z in S.values() for d in z.check.values())
    P(f"Out-of-pool scoring check: MAP ability under fixed items minus the fitted posterior mean, pool subjects, all variants: max |diff| {worst:.2f}.\n")

    P("### Every subject: arena held outcomes and local PCS (official pool and board-mix pool)\n")
    P("| subject | family | held mean | exact 1 | exact 0 | partial | PCS official | PCS board-mix | board PCS (mean) |"); P("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    pm = S[a.primary]
    for s in sorted(pm.theta, key=lambda s: -pm.pcs[s]):
        y = sc.loc[s, items0].to_numpy(float)
        bp = f"{BOARD[s][0]} ({BOARD[s][1]:.3f})" if s in BOARD else ""
        P(f"| {s} | {'third' if s in THIRD else 'ours'} | {np.mean(y):.3f} | {np.mean(y >= 1 - 1e-4):.3f} | {np.mean(y <= 1e-4):.3f} | "
          f"{np.mean((y > 1e-4) & (y < 1 - 1e-4)):.3f} | {S['official'].pcs[s]:.0f} | {pm.pcs[s]:.0f} | {bp} |")
    pd.DataFrame({v: z.pcs for v, z in S.items()}).to_csv(out / "pcs_by_variant.csv")

    # 2. items by stratum
    for v in ("official", a.primary):
        f = S[v].f
        it = pd.DataFrame(dict(a=f["a"], b=f["b"], g0=f["g0"], g1=f["g1"], prec=f["prec"]), index=items0).join(w)
        it["third"] = sc.loc[sorted(THIRD), items0].mean(0); it["garage"] = sc.loc["garage", items0]
        it["ours"] = sc.loc[[s for s in pm.theta if s not in THIRD], items0].mean(0)
        P(f"\n## 2{'a' if v == 'official' else 'b'}. Item parameters by cell, {v} pool\n")
        P("Cells: arena strata; static split by a lead within 8 m at the window start. a = discrimination; g0 / g1 = exact-0 / exact-1 thresholds "
          "(P(0) = sigmoid(g0 - a theta), P(1) = sigmoid(a theta - g1)).\n")
        P("| cell | windows | a median [q25, q75] | share a < 0.5 | b median | g0 median | g1 median | third-party mean | garage | ours mean | JTN-s1 | JTN-s1 + stop3 |")
        P("|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for c, g in it.groupby("cell"):
            q = g.a.quantile([.25, .5, .75])
            P(f"| {c} | {len(g)} | {q[.5]:.2f} [{q[.25]:.2f}, {q[.75]:.2f}] | {np.mean(g.a < 0.5):.2f} | {g.b.median():+.2f} | {g.g0.median():+.2f} | {g.g1.median():+.2f} | "
              f"{g.third.mean():.3f} | {g.garage.mean():.3f} | {g.ours.mean():.3f} | {sc.loc['JTN-s1', g.index].mean():.3f} | {sc.loc['JTN-s1+stop3', g.index].mean():.3f} |")
        P(f"\nSpearman over items of a against (third-party mean - ours mean): {spearmanr(it.a, it.third - it.ours).statistic:+.2f}; "
          f"against the garage score: {spearmanr(it.a, it.garage).statistic:+.2f}.")
        it.to_csv(out / f"items_{v}.csv.gz")

    # 3. decomposition under the primary pool
    z = pm; cell = w.loc[items0, "cell"].to_numpy()
    gy = sc.loc["garage", items0].to_numpy(float)
    P(f"\n## 3. What costs us PCS ({a.primary} pool, items fixed, board-mapped PCS)\n")
    for s in ["JTN-s1+stop3", "JTN-s1"]:
        y0 = sc.loc[s, items0].to_numpy(float); typ = ty.loc[s, items0].to_numpy(object)
        base = z.pcs_of(y0)
        P(f"\n### {s}: PCS {base:.0f} (garage {z.pcs['garage']:.0f}, gap {base - z.pcs['garage']:+.0f})\n")
        P("PCS change when this subject's scores on one cell are replaced (all other cells unchanged):\n")
        P("| cell | windows | ours 0 / partial / 1 | garage 0 / partial / 1 | take garage's scores | max(ours, garage) | zeros -> 1 | partials -> 1 | all -> 1 |")
        P("|---|--:|---|---|--:|--:|--:|--:|--:|")
        for c in sorted(set(cell)) + ["all"]:
            m = np.ones(len(items0), bool) if c == "all" else cell == c

            def cf(fn):
                y = y0.copy(); y[m] = fn(y0[m], gy[m]); return z.pcs_of(y) - base
            cnt = lambda v: f"{np.sum(v[m] <= 1e-4)} / {np.sum((v[m] > 1e-4) & (v[m] < 1 - 1e-4))} / {np.sum(v[m] >= 1 - 1e-4)}"  # noqa: E731
            P(f"| {c} | {m.sum()} | {cnt(y0)} | {cnt(gy)} | {cf(lambda o, g: g):+.0f} | {cf(np.maximum):+.0f} | "
              f"{cf(lambda o, g: np.where(o <= 1e-4, 1.0, o)):+.0f} | {cf(lambda o, g: np.where((o > 1e-4) & (o < 1 - 1e-4), 1.0, o)):+.0f} | {cf(lambda o, g: np.ones_like(o)):+.0f} |")
        oc = np.array([outcome(y, t, c) for y, t, c in zip(y0, typ, cell)])
        P(f"\n{s} by outcome class (those windows set to 1):\n")
        P("| outcome | windows | mean-score points lost | PCS if fixed | PCS per window | garage scores 1 on |"); P("|---|--:|--:|--:|--:|--:|")
        for k in sorted(set(oc) - {"1"}):
            m = oc == k; y = y0.copy(); y[m] = 1.0; g = z.pcs_of(y) - base
            P(f"| {k} | {m.sum()} | {np.sum(1 - y0[m]):.1f} | {g:+.0f} | {g / m.sum():+.2f} | {np.sum(gy[m] >= 1 - 1e-4)} |")
        combos = {"slow-progress partials -> 1 (progress >= 0.8 of the human)": oc == "partial, slow progress",
                  "+ partials >= 0.98 and struck-from-behind -> 1": np.isin(oc, ["partial, slow progress", "partial >= 0.98", "partial, struck from behind"]),
                  "+ offroad zeros -> 1": np.isin(oc, ["partial, slow progress", "partial >= 0.98", "partial, struck from behind", "0 offroad"]),
                  "static-window corridor zeros -> 1 (native: stop at rest)": oc == "0 corridor, static window",
                  "launch-window partials -> 1": oc == "partial, launch window (late / no launch, creep, rear-hit)",
                  "every partial -> 1 (zeros kept)": (y0 > 1e-4) & (y0 < 1 - 1e-4),
                  "every zero -> 1 (partials kept)": y0 <= 1e-4}
        P(f"\n{s} combined counterfactuals:\n")
        P("| change | windows | mean score after | PCS after | gain |"); P("|---|--:|--:|--:|--:|")
        for k, m in combos.items():
            y = y0.copy(); y[m] = 1.0; p = z.pcs_of(y)
            P(f"| {k} | {m.sum()} | {y.mean():.3f} | {p:.0f} | {p - base:+.0f} |")
        if s == "JTN-s1+stop3":
            # marginal value of one window
            gains = np.full(len(items0), np.nan)
            for j in np.where(y0 < 1 - 1e-4)[0]:
                y = y0.copy(); y[j] = 1.0; gains[j] = z.pcs_of(y) - base
            th = z.theta[s]; f = z.f
            m = np.isfinite(gains)
            d = pd.DataFrame(dict(gain=gains[m], kind=np.where(y0[m] <= 1e-4, "exact 0", np.where(y0[m] >= 0.9, "partial >= 0.9", "partial < 0.9")),
                                  garage=np.where(gy[m] >= 1 - 1e-4, "garage 1", np.where(gy[m] <= 1e-4, "garage 0", "garage partial")), a=f["a"][m]))
            P("\nMarginal PCS of fixing one window (JTN-s1 + stop3):\n")
            P("| our outcome | garage on the window | windows | median PCS gain | mean | median a |"); P("|---|---|--:|--:|--:|--:|")
            for (k, g), dd in d.groupby(["kind", "garage"]):
                P(f"| {k} | {g} | {len(dd)} | {dd.gain.median():+.1f} | {dd.gain.mean():+.1f} | {dd.a.median():.2f} |")
            # value of moving 0.95 -> 1 vs 0 -> 0.95
            P("\nThe three-way likelihood on one typical window (median-a launch-cell item): ability change per outcome, holding the rest:")
            for c in ("launch", "other"):
                j = np.where((cell == c) & (y0 >= 1 - 1e-4))[0]
                if len(j):
                    j = j[np.argsort(f["a"][j])[len(j) // 2]]
                    vals = []
                    for v in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0):
                        y = y0.copy(); y[j] = v; vals.append(f"{v:g} -> {z.pcs_of(y) - base:+.1f}")
                    P(f"- {c} item (a {f['a'][j]:.2f}): " + ", ".join(vals))

    # 3c. other candidates scored against the primary pool
    P(f"\n### Our stored candidates scored as one more submission against the {a.primary} pool\n")
    P("| subject | held mean | PCS | zeros | partials |"); P("|---|--:|--:|--:|--:|")
    for s in sorted([s for s in z.theta if s not in THIRD], key=lambda s: -z.pcs[s]):
        y = sc.loc[s, items0].to_numpy(float)
        P(f"| {s} | {np.mean(y):.3f} | {z.pcs[s]:.0f} | {np.sum(y <= 1e-4)} | {np.sum((y > 1e-4) & (y < 1 - 1e-4))} |")

    # 4. subject-count curve
    if (out / "subcurve.csv").exists():
        P("\n## 4. How many subjects a local PCS needs\n")
        P(pd.read_csv(out / "subcurve.csv").to_markdown(index=False, floatfmt=".2f"))
    (out / ("report.md" if a.primary == "boardmix" else f"report_{a.primary}.md")).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_subcurve(a):
    """Board-mix pool with the garage weight thinned / the pool shrunk: is the JTN pair's PCS stable? Pools: the 8 known subjects with
    garage repeated k times (k = 1 .. 30), plus our other complete subjects added singly."""
    sc, _, _ = load(a.data)
    _, cols = pool(sc, "complete")
    rows = []
    for k in [1, 3, 10, 30]:
        for extra in (False, True):
            mix = dict(BOARD_MIX, garage=k)
            subs = [s for s, n in mix.items() for _ in range(n)]
            if extra:
                subs += [s for s in pool(sc, "complete")[0] if s not in mix]
            M, names = matrix(sc, subs, cols)
            f = fit(M, names, cols)
            z = Scored(dict(f, subjects=np.array(names), items=np.array(cols)), sc)
            rows.append(dict(garage_rows=k, our_other_subjects=extra, rows=len(names),
                             tau8=kendall([z.theta[s] for s in BOARD], [BOARD[s][0] for s in BOARD]),
                             native=z.pcs["JTN-s1"], stop3=z.pcs["JTN-s1+stop3"], garage=z.pcs["garage"], go_straight=z.pcs["go-straight"]))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(Path(a.out) / "subcurve.csv", index=False)


def cmd_score(a):
    """Score new arena reads (arena.py collect table) as one more submission against frozen items (default results/pcs_local/boardmix.npz)."""
    sc, _, _ = load(HERE.parent / "results/pcs_local")
    z = Scored(dict(np.load(a.fit, allow_pickle=True)), sc)
    R = pd.read_csv(a.rows)
    keep = None
    if a.subset:                                                     # NR1STAND: only the windows every tag of the table has (a chunk subset)
        keep = set.intersection(*(set(g.id[g.score.notna()]) for _, g in R.groupby("tag"))) & set(z.items)
    for t, g in R.groupby("tag"):
        y = g.drop_duplicates("id").set_index("id").score.reindex(z.items).to_numpy(float)
        if keep is not None:
            y = np.where(np.isin(z.items, list(keep)), y, np.nan)
        miss = int(np.isnan(y).sum())
        cnt = f", slow {int((g.type == 'slow').sum())}, near-1 {int(((g.score >= 0.98) & (g.score <= 0.999)).sum())}" if "type" in g else ""
        if keep is not None:                                         # fixed items, MAP on the subset's items only (NaN = not observed)
            print(f"{t}: subset windows {len(z.items) - miss} / {len(z.items)}, mean {np.nanmean(y):.4f}{cnt}, subset PCS {z.pcs_of(y):.0f}")
        else:
            print(f"{t}: windows {len(z.items) - miss} / {len(z.items)}, mean {np.nanmean(y):.4f}, " + ("incomplete (needs every held arena window)" if miss else f"PCS {z.pcs_of(y):.0f}"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    b = s.add_parser("build"); b.add_argument("--rows", required=True); b.add_argument("--out", required=True)
    f = s.add_parser("fit"); f.add_argument("--data", required=True); f.add_argument("--out", required=True); f.add_argument("--variants", default="")
    r = s.add_parser("report"); r.add_argument("--data", required=True); r.add_argument("--out", required=True); r.add_argument("--primary", default="boardmix")
    c = s.add_parser("subcurve"); c.add_argument("--data", required=True); c.add_argument("--out", required=True); c.add_argument("--reps", type=int, default=1)
    k = s.add_parser("score"); k.add_argument("--rows", required=True); k.add_argument("--fit", default=str(HERE.parent / "results/pcs_local/boardmix.npz"))
    k.add_argument("--subset", action="store_true", help="score on the windows every tag has (a chunk subset); not the board's complete set")
    a = p.parse_args()
    dict(build=cmd_build, fit=cmd_fit, report=cmd_report, subcurve=cmd_subcurve, score=cmd_score)[a.cmd](a)


if __name__ == "__main__":
    main()
