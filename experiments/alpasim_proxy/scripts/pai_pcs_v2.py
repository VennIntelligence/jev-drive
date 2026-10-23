#!/usr/bin/env python3
"""PAI-PCS-V2: the decision-392 PAI PCS ruler refit under AlpaSim's fixed corridor scorer (e2e_challenge 1b024f0, PR #215; the PAI
board was recomputed with it, snapshot leaderboard_pai_20261023.json).

Same settings as pai_pcs_local.py (zoib 2PL, official optimiser settings, the d392 item set of 275 scenes, board bands -> stand-ins,
three-anchor least squares, scene bootstrap of the whole pipeline). What changes:
- per-scene scores are the new-scorer rescores (experiments/alpasim/scripts/scorer_v2.py; decision 453), the anchors are fresh
  garage 0014 / go-straight re-reads with their zero ASLs kept (`pai_pcs_v2/{G0014,GS}`), rescored the same way;
- anchor PCS from the new board (garage 1837, go-straight 1441, RSG + ACC + GAP 2189); band cuts shifted with the board (`CUTS`);
- the strong band's stand-ins are the d392 band reads that are complete under the new scorer (RSG-s0, NTV0-s0 + ACC + GAP). Partial
  reads (zero-score ASLs deleted after the read, so some 0 -> 1 flips are invisible) are lower bounds: out of the pool by default,
  pooled as-is only in the `lb` variant. Band rows are spread evenly over the band reads (d392 rounded k / n per read).

  pai_pcs_v2.py collect --out ROWS.csv.gz           box, repo root, .venv python: new / old score + RPC flag + partial flag per read and scene
  pai_pcs_v2.py fit --rows ROWS --out DIR [--variants a,b] [--boot N --boot0 K]   Mac (tmp/pcs_local/.venv, drive-irt 82ddd5a)
  pai_pcs_v2.py report --rows ROWS --out DIR [--gap scenes.csv]                   -> DIR/report.md, DIR/pai_pcs_v2.json, DIR/pred.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE)]
RES = HERE.parent / "results"

# scorer_v2 roots (under $DATA_DIR/runs) whose new/ and old/ rollouts.csv hold the rescored reads
ROOTS = ("alpasim/scorer_v2", "alpasim_proxy/pai_pcs_v2/rescore", "alpasim_proxy/pai_pcs_v2/rescore5")
RENAME = {"G0014-v2": "garage", "GS-v2": "go-straight", "GS5-v2": "go-straight-rerun5"}
# A rollout the runtime could not run (thread start failed at the box's pids cap) has no metrics and a stored 0: infrastructure, masked
# like an RPC error. GS-v2 lost 5 scenes this way (4 in the d392 items); `--gs` says how go-straight is completed on them:
#   fill  = the first go-straight read (pai_pcs/GS, d392 rows, old scorer; the starter kit is near-deterministic)   [default]
#   mask  = left out: those items drop from the item set (271 of 275)
#   rerun = the 5 scenes re-read alone (pai_pcs_v2/GS5, same settings) and rescored
INFRA = ("can't start new thread", "Resource temporarily unavailable")
ANCHORS = {"garage": 1837, "go-straight": 1441, "RSG-L18-s0-ag": 2189}          # PAI board 2026-10-23 (new scorer)
BOARD_MEAN = {"garage": 0.4649, "go-straight": 0.1746, "RSG-L18-s0-ag": 0.6802}
VALID = {"NTV0-L18-s0-ag": 2251}                                                 # ujs-iv p1021a, not an anchor
OLD_ANCHORS = {"garage": 1819, "go-straight": 1446, "RSG-L18-s0-ag": 2089}       # board 2026-10-19 (old scorer)
REF = "NTV0-L18-s0-ag"
# d392's cuts 1940 / 1750 / 1440 moved with the board: local linear fit of new on old PCS over the 104 submissions on both boards
# within +-60 PCS of each cut -> 1961 / 1772 / 1441.
CUTS = (1961, 1772, 1441)
BAND = ["RSG-L18-s0-ag", "NTV0-L18-s0-ag"]                                       # d392 band reads, complete under the new scorer
BAND_LB = ["NTV0-L18-s1-ag", "JTN-L18-s0-ag", "JTN-L18-s1-ag", "BNC2K-L18-s1-ag", "JTXS-L18-s1-ag", "NTVX-L18-s0-ag", "NTVXS-L18-s0-ag"]
# no `dedup` variant: one row per subject is 4 rows (garage, go-straight, 2 band reads), below the fit guard of S + 5 N cells
VARIANTS = {"v2": 42, "v2-s1": 1, "v2-s2": 2, "garagex0.5": 42, "garagex2": 42, "bandx0.5": 42, "bandx2": 42, "gsx0.5": 42, "gsx2": 42,
            "low-as-gs": 42, "lb": 42, "allag": 42, "oldcuts": 42, "v2-1pl": 42}


def bands(board, cuts=CUTS):
    p = np.array([r["policy_capability_score"] for r in board])
    return dict(band=int((p >= cuts[0]).sum()), garage=int(((p >= cuts[1]) & (p < cuts[0])).sum()),
                gs=int(((p >= cuts[2]) & (p < cuts[1])).sum()), low=int((p < cuts[2]).sum()))


# ---------------------------------------------------------------- box: collect
def cmd_collect(a):
    sys.path[:0] = [str(REPO / "experiments/defect_fix/scripts"), str(REPO / "experiments/alpasim/scripts")]
    from pai281 import load
    from scorer_v2 import READS, stored_scores
    runs = Path(os.environ["DATA_DIR"]) / "runs"
    out = []
    for root in ROOTS:
        r = runs / root
        if not (r / "report/reads.csv").exists():
            print(f"{root}: no report, skipped"); continue
        part = pd.read_csv(r / "report/reads.csv").set_index("read").partial.to_dict()
        S = {v: pd.read_csv(r / f"{v}/rollouts.csv").drop_duplicates(["read", "scene"], keep="last").set_index(["read", "scene"]).score
             for v in ("new", "old")}
        for lab in S["new"].index.get_level_values(0).unique():
            _, bad = load(str(runs / READS[lab][0]))
            infra = {sc for sc, (_, _, fr) in stored_scores(lab).items() if any(t in str(fr) for t in INFRA)}
            bad = set(bad) | infra
            n, o = S["new"][lab], S["old"][lab]
            for sc in sorted(set(n.index) | set(bad)):
                out.append(dict(subject=RENAME.get(lab, lab), read=lab, scene=sc, score=n.get(sc, np.nan), score_old=o.get(sc, np.nan),
                                rpc=sc in bad, infra=sc in infra, partial=bool(part[lab])))
            print(f"{lab}: {len(n)} scenes, {len(bad)} RPC / infra ({len(infra)} infra), new mean {n.mean():.4f} old {o.mean():.4f}, partial {part[lab]}", flush=True)
    pd.DataFrame(out).to_csv(a.out, index=False)


# ---------------------------------------------------------------- Mac: matrix, pools, fits
GS_MODE = "fill"


def table(rows, col="score", gs=None):
    R = pd.read_csv(rows)
    sc = R.pivot_table(index="subject", columns="scene", values=col, aggfunc="first", dropna=False)
    rpc = R.pivot_table(index="subject", columns="scene", values="rpc", aggfunc="first").fillna(False).astype(bool)
    sc = sc.where(~rpc.reindex_like(sc).fillna(False).astype(bool))                # RPC error = infrastructure, not behaviour (d327)
    part = R.groupby("subject").partial.first()
    gs = gs or GS_MODE
    hole = R[(R.subject == "go-straight") & R.infra.astype(bool)].scene.tolist()
    if gs == "fill":
        v1 = pd.read_csv(RES / "pai_pcs_local/rows.csv.gz").query("subject == 'go-straight'").set_index("scene").score
        sc.loc["go-straight", hole] = v1.reindex(hole).to_numpy()
    elif gs == "rerun":
        sc.loc["go-straight", hole] = sc.loc["go-straight-rerun5", hole].to_numpy(float) if col == "score" else \
            R[R.subject == "go-straight-rerun5"].set_index("scene")[col].reindex(hole).to_numpy(float)
    return sc.drop(index=[s for s in sc.index if s.startswith("go-straight-")]), part


def d392_items():
    """The decision-392 item set: complete matrix over its paimix pool (garage, go-straight, 13 band reads) -> 275 scenes."""
    import pai_pcs_local as P1
    sc, _, fam = P1.table(RES / "pai_pcs_local/rows.csv.gz")
    nb = P1.bands(json.load(open(RES / "leaderboard_pai_20261019.json"))["leaderboard"])
    it = list(P1.items_of(sc, P1.pool(sc, fam, "paimix", nb)))
    assert len(it) == 275, len(it)
    return it


def items_for(sc):
    """The d392 items, minus any the pool's anchors miss (only under --gs mask)."""
    it = d392_items()
    keep = [s for s in it if sc.loc[["garage", "go-straight"], s].notna().all()]
    if len(keep) < len(it):
        print(f"items: {len(keep)} of {len(it)} (go-straight masked on {len(it) - len(keep)})", flush=True)
    return keep


def spread(reads, k):
    """k rows over the reads, as evenly as possible (the first k mod n reads get one more)."""
    n = len(reads)
    return [s for i, s in enumerate(reads) for _ in range(k // n + (i < k % n))] if k >= n else list(reads)


def pool(sc, part, name, nb):
    k = dict(band=nb["band"], garage=nb["garage"], gs=nb["gs"])
    if name == "low-as-gs":
        k["gs"] += nb["low"]
    for key in ("garage", "band", "gs"):
        if name.startswith(f"{key}x"):
            k[key] = round(k[key] * float(name.split("x")[1]))
    band = list(BAND)
    if name == "lb":
        band += BAND_LB
    if name == "allag":                                              # every complete ACC + GAP (Harmonizer off) read we have
        it = d392_items()
        band = [s for s in sc.index if s.endswith("-ag") and not part.get(s, True) and sc.loc[s, it].notna().all()]
    if name == "dedup":
        return ["garage", "go-straight"] + band
    return ["garage"] * k["garage"] + ["go-straight"] * k["gs"] + spread(band, k["band"])


def base_of(v):
    return "v2" if v.startswith("v2") else v


def cmd_fit(a):
    from pcs_local import fit, guard, matrix
    col = "score_old" if a.old else "score"
    sc, part = table(a.rows, col)
    board = json.load(open(RES / "leaderboard_pai_20261023.json"))["leaderboard"]     # --old: same pool as the new fit (scorer effect only)
    items = items_for(sc)
    out = Path(a.out) / ("fits_old" if a.old else "fits"); out.mkdir(parents=True, exist_ok=True)
    for v in ([] if a.variants == "-" else a.variants.split(",") if a.variants else VARIANTS):
        nb = bands(board, (1940, 1750, 1440) if v == "oldcuts" else CUTS)
        subs = pool(sc, part, "v2" if v == "oldcuts" else base_of(v), nb)
        miss = sc.loc[sorted(set(subs)), items].isna().sum(axis=1)
        assert not miss.any(), f"{v}: pool subjects with NaN on the d392 items: {miss[miss > 0].to_dict()}"
        M, names = matrix(sc, subs, items)
        assert guard(M), v
        f = fit(M, names, items, seed=VARIANTS.get(v, 42), fix=v.endswith("1pl"))
        np.savez(out / f"{v}.npz", subjects=np.array(names), items=np.array(items, dtype=str), **f)
        print(v, len(names), "rows", len(items), "items", nb, flush=True)
    if a.boot:
        subs = pool(sc, part, "v2", bands(board, CUTS))
        cols = np.array(items)
        B = []
        for b in range(a.boot0, a.boot0 + a.boot):
            rng = np.random.default_rng(b)
            pick = cols[rng.integers(0, len(cols), len(cols))]
            sb = sc[pick]; sb.columns = [f"{c}#{i}" for i, c in enumerate(pick)]
            M, names = matrix(sb, subs, sb.columns)
            f = fit(M, names, list(sb.columns))
            B.append(Ruler(dict(f, subjects=np.array(names), items=np.array(sb.columns)), sb).pcs)
            print("boot", b, flush=True)
        pd.DataFrame(B).to_csv(Path(a.out) / f"boot_{a.boot0:03d}.csv", index=False)


class Ruler:
    """Fixed items; ability of a pool subject = its fitted posterior mean, of any other = MAP under the items (pai_pcs_local.Ruler);
    PCS = least squares of ability on the anchors' board PCS (`drop`: leave that anchor out of the map)."""

    def __init__(self, f, sc, anchors=ANCHORS, drop=None, mapall=False):
        from pcs_local import theta_map
        self.f, self.items = f, [str(c) for c in f["items"]]
        names = [str(s) for s in f["subjects"]]
        mu, sd = float(np.mean(f["theta"])), float(np.std(f["theta"]))
        fitted = {s: float(t) for s, t in zip(names, f["theta"]) if "#" not in s}
        self.theta, self.check, self.nan = {}, {}, {}
        for s in sc.index:
            y = sc.loc[s, self.items].to_numpy(float)
            if np.isnan(y).mean() > 0.05:
                continue
            if mapall or s not in fitted:
                self.theta[s] = float(theta_map(y, f, mu, sd))
            else:
                self.theta[s] = fitted[s]
                if drop is None:
                    self.check[s] = float(theta_map(y, f, mu, sd)) - fitted[s]
        an = [k for k in anchors if k != drop]
        self.A = np.polyfit([self.theta[k] for k in an], [anchors[k] for k in an], 1)
        self.pcs = {s: float(np.polyval(self.A, t)) for s, t in self.theta.items()}


def load_fit(out, v, sub="fits"):
    return dict(np.load(Path(out) / sub / f"{v}.npz", allow_pickle=True))


# ---------------------------------------------------------------- report
CANDS = [("NTV0-s0", "NTV0-L18-s0")] + [(f"{m}-{s}", f"{m}-L18-{s}") for m in ("R6DAG", "R8K") for s in ("s0", "s1", "s2")]


def cmd_report(a):
    sc, part = table(a.rows)
    so, _ = table(a.rows, "score_old")
    out = Path(a.out)
    Z = {v: Ruler(load_fit(out, v), sc) for v in VARIANTS if (out / f"fits/{v}.npz").exists()}
    z = Z["v2"]
    Z["v2, MAP for all"] = Ruler(dict(z.f), sc, mapall=True)
    nb = bands(json.load(open(RES / "leaderboard_pai_20261023.json"))["leaderboard"])
    L, J = [], dict(bands=nb, cuts=CUTS, anchors=ANCHORS)
    P = L.append
    it = z.items
    mean = sc[it].mean(1)
    meanall = sc.mean(1)
    P(f"Pool v2: {len(z.f['subjects'])} rows ({nb['garage']} garage, {nb['gs']} go-straight, {nb['band']} band rows over {', '.join(BAND)}), "
      f"{len(it)} scenes (the d392 item set). Board bands at cuts {CUTS}: {nb}.\n")

    # ---- anchors, leave-one-out, validation
    P("## Anchor fit and validation\n")
    P("| subject | board PCS (mean) | local 281 mean, new (old) | ability | fitted PCS | residual | leave-this-anchor-out |")
    P("|---|--:|--:|--:|--:|--:|--:|")
    J["anchor_fit"] = {}
    for k in ANCHORS:
        lo = Ruler(dict(z.f), sc, drop=k).pcs[k]
        P(f"| {k} | {ANCHORS[k]} ({BOARD_MEAN[k]:.3f}) | {meanall[k]:.3f} ({so.mean(1)[k]:.3f}) | {z.theta[k]:+.2f} | {z.pcs[k]:.0f} | "
          f"{z.pcs[k] - ANCHORS[k]:+.0f} | {lo:.0f} ({lo - ANCHORS[k]:+.0f}) |")
        J["anchor_fit"][k] = dict(fit=z.pcs[k], loo=lo)
    rm = np.sqrt(np.mean([(z.pcs[k] - ANCHORS[k]) ** 2 for k in ANCHORS]))
    v = REF
    P(f"| **{v} (p1021a, validation)** | {VALID[v]} (0.716) | {meanall[v]:.3f} ({so.mean(1)[v]:.3f}) | {z.theta[v]:+.2f} | **{z.pcs[v]:.0f}** | "
      f"**{z.pcs[v] - VALID[v]:+.0f}** | |")
    P(f"\nMap: PCS = {z.A[0]:.1f} x ability + {z.A[1]:.0f}; anchor RMSE {rm:.0f}. NTV0 - RSG: predicted {z.pcs[REF] - z.pcs['RSG-L18-s0-ag']:+.0f}, "
      f"board {VALID[REF] - ANCHORS['RSG-L18-s0-ag']:+d}.")
    ck = pd.Series(z.check)
    P(f"Pool subjects, MAP under the fitted items minus the posterior mean: max |diff| {ck.abs().max():.2f} ({ck.abs().idxmax()}) = "
      f"{ck.abs().max() * z.A[0]:.0f} PCS.\n")
    J.update(rmse=rm, map=list(z.A), valid=dict(pred=z.pcs[REF], board=VALID[REF], gap_pred=z.pcs[REF] - z.pcs["RSG-L18-s0-ag"]))

    # ---- predictions
    bf = sorted(out.glob("boot_*.csv"))
    boot = pd.concat([pd.read_csv(f) for f in bf], ignore_index=True) if bf else None
    ci = lambda s: (np.nanpercentile(boot[s], 2.5), np.nanpercentile(boot[s], 97.5)) if boot is not None and s in boot else (np.nan, np.nan)  # noqa: E731
    P(f"## Predicted PAI PCS, new scorer\n\nCI: scene bootstrap of the whole pipeline ({0 if boot is None else len(boot)} draws), 2.5-97.5 %. "
      "Paired difference: same draws, against NTV0-s0 with the same Harmonizer setting (OFF -> NTV0-s0 + AG, ON -> NTV0-s0 + AG + H).\n")
    P("| candidate | Harmonizer | local mean (275) | zeros | PCS | 95 % CI | range over variants | vs NTV0-s0 (paired median) | 95 % CI | P(> 0) |")
    P("|---|---|--:|--:|--:|---|---|--:|---|--:|")
    rows = []
    for name, lab in CANDS:
        for mode, suf in (("off", "-ag"), ("on", "-agH")):
            s, ref = lab + suf, "NTV0-L18-s0" + suf
            if s not in z.pcs:
                continue
            vv = [Zv.pcs[s] for Zv in Z.values() if s in Zv.pcs]
            c = ci(s)
            d = (boot[s] - boot[ref]) if boot is not None and s != ref else None
            r = dict(candidate=name, harmonizer=mode, read=s, mean=mean[s], zeros=int((sc.loc[s, it] <= 1e-4).sum()), pcs=z.pcs[s], lo=c[0], hi=c[1],
                     vmin=min(vv), vmax=max(vv), diff=np.nan if d is None else d.median(), dlo=np.nan if d is None else d.quantile(.025),
                     dhi=np.nan if d is None else d.quantile(.975), p_pos=np.nan if d is None else (d > 0).mean())
            rows.append(r)
            star = "**" if name in ("R8K-s2", "R6DAG-s1") else ""
            P(f"| {star}{name}{star} | {mode} | {r['mean']:.3f} | {r['zeros']} | {star}{r['pcs']:.0f}{star} | [{c[0]:.0f}, {c[1]:.0f}] | {r['vmin']:.0f}-{r['vmax']:.0f} | "
              + ("(reference) | | |" if d is None else f"{r['diff']:+.0f} | [{r['dlo']:+.0f}, {r['dhi']:+.0f}] | {r['p_pos']:.2f} |"))
    pred = pd.DataFrame(rows)
    if boot is not None:
        P("\nAverages (same draws): PCS of the mean over the reads, minus the NTV0-s0 counterpart.\n")
        P("| set | PCS | 95 % CI | vs NTV0-s0 | 95 % CI | P(> 0) |"); P("|---|--:|---|--:|---|--:|")
        J["sets"] = {}
        for m in ("R6DAG", "R8K"):
            for tag, sufs in (("OFF", ["-ag"]), ("ON", ["-agH"]), ("(OFF + ON) / 2", ["-ag", "-agH"])):
                for nm, seeds in ((f"{m} 3-seed", ("s0", "s1", "s2")),) + (((f"{m}-s2", ("s2",)),) if m == "R8K" else ((f"{m}-s1", ("s1",)),)):
                    ss = [f"{m}-L18-{sd}{x}" for sd in seeds for x in sufs]
                    rr = [f"NTV0-L18-s0{x}" for x in sufs]
                    if not all(s in boot for s in ss + rr):
                        continue
                    xb, rb = boot[ss].mean(1), boot[rr].mean(1)
                    xp = np.mean([z.pcs[s] for s in ss]); d = xb - rb
                    P(f"| {nm}, {tag} | {xp:.0f} | [{xb.quantile(.025):.0f}, {xb.quantile(.975):.0f}] | {d.median():+.0f} | "
                      f"[{d.quantile(.025):+.0f}, {d.quantile(.975):+.0f}] | {(d > 0).mean():.2f} |")
                    J["sets"][f"{nm}, {tag}"] = dict(pcs=xp, ci=[xb.quantile(.025), xb.quantile(.975)], diff=d.median(), dci=[d.quantile(.025), d.quantile(.975)], p=(d > 0).mean())
        d = boot[REF] - boot["RSG-L18-s0-ag"]
        P(f"\nNTV0-s0 + AG - RSG + AG on the same draws: {d.median():+.0f} [{d.quantile(.025):+.0f}, {d.quantile(.975):+.0f}] (board +62).")
        x = boot["R8K-L18-s2-ag"] / 2 + boot["R8K-L18-s2-agH"] / 2 - boot["R6DAG-L18-s1-ag"] / 2 - boot["R6DAG-L18-s1-agH"] / 2
        P(f"R8K-s2 - R6DAG-s1, (OFF + ON) / 2: {x.median():+.0f} [{x.quantile(.025):+.0f}, {x.quantile(.975):+.0f}], P(> 0) {(x > 0).mean():.2f}.")
    pred.round(4).to_csv(out / "pred.csv", index=False)

    # ---- other reads (out of pool; partial = lower bound)
    P("\n## Other rescored reads (MAP, out of pool unless band)\n")
    P("| read | partial (lower bound) | local mean (275) | PCS | range over variants |"); P("|---|---|--:|--:|---|")
    J["all"] = {}
    for s in sorted(z.pcs, key=lambda s: -z.pcs[s]):
        vv = [Zv.pcs[s] for Zv in Z.values() if s in Zv.pcs]
        P(f"| {s} | {'yes' if part.get(s, False) else ''} | {mean[s]:.3f} | {z.pcs[s]:.0f} | {min(vv):.0f}-{max(vv):.0f} |")
        J["all"][s] = dict(pcs=z.pcs[s], partial=bool(part.get(s, False)), by_variant={v: Zv.pcs.get(s) for v, Zv in Z.items()})

    # ---- sensitivity
    P("\n## Sensitivity to the pool\n")
    keys = ["NTV0-L18-s0-ag", "NTV0-L18-s0-agH", "R6DAG-L18-s1-ag", "R6DAG-L18-s1-agH", "R8K-L18-s2-ag", "R8K-L18-s2-agH", "garage", "go-straight", "RSG-L18-s0-ag"]
    P("| variant | rows | anchor RMSE | p1021a err | NTV0-RSG | " + " | ".join(k.replace("-L18", "") for k in keys) + " |"); P("|---|--:|--:|--:|--:|" + "--:|" * len(keys))
    J["variants"] = {}
    for v, Zv in Z.items():
        r = np.sqrt(np.mean([(Zv.pcs[k] - ANCHORS[k]) ** 2 for k in ANCHORS]))
        P(f"| {v} | {len(Zv.f['subjects'])} | {r:.0f} | {Zv.pcs[REF] - VALID[REF]:+.0f} | {Zv.pcs[REF] - Zv.pcs['RSG-L18-s0-ag']:+.0f} | "
          + " | ".join(f"{Zv.pcs.get(k, np.nan):.0f}" for k in keys) + " |")
        J["variants"][v] = dict(rmse=r, **{k: Zv.pcs.get(k) for k in keys})

    # ---- discrimination: same pool and items, old vs new scores
    if (out / "fits_old/v2.npz").exists():
        fo = load_fit(out, "v2", "fits_old")
        zo = Ruler(fo, so, anchors=OLD_ANCHORS)
        an, ao = pd.Series(z.f["a"], index=it), pd.Series(fo["a"], index=it)
        bn, bo = pd.Series(z.f["b"], index=it), pd.Series(fo["b"], index=it)
        flip = (sc[it] - so[it]).abs().gt(1e-9).sum(0)
        F = flip[flip > 0].index
        P("\n## Item discrimination, old vs new scores (same pool rows, items and seed; the old fit mapped on the 10-19 anchors)\n")
        P(f"Old-score fit as a check: NTV0-s0 + AG {zo.pcs[REF]:.0f}, RSG {zo.pcs['RSG-L18-s0-ag']:.0f} (board then 2154 / 2089).\n")
        P("| items | n | median a old | median a new | median b old | median b new |"); P("|---|--:|--:|--:|--:|--:|")
        grp = [("all", it), ("changed by the fix in some read", list(F)), ("unchanged", [s for s in it if s not in F])]
        if a.gap:
            G = pd.read_csv(a.gap).set_index("scene")
            grp += [(f"stratum {k}", [s for s in it if s in G.index and G.loc[s, "stratum"] == k]) for k in ("static", "launch", "stop/lead", "straight", "turn")]
        J["disc"] = {}
        for g, ss in grp:
            P(f"| {g} | {len(ss)} | {ao[ss].median():.2f} | {an[ss].median():.2f} | {bo[ss].median():+.2f} | {bn[ss].median():+.2f} |")
            J["disc"][g] = dict(n=len(ss), a_old=ao[ss].median(), a_new=an[ss].median())
        P("\nThe changed items (0 -> 1 in some read; reads changed / pool band reads scoring 1, old -> new):\n")
        P("| scene | reads changed | a old | a new | garage old / new | go-straight old / new | RSG old / new | NTV0 old / new |"); P("|---|--:|--:|--:|---|---|---|---|")
        for s in sorted(F, key=lambda s: -flip[s]):
            P(f"| {s[-8:]} | {flip[s]} | {ao[s]:.2f} | {an[s]:.2f} | " + " | ".join(f"{so.loc[k, s]:.2f} / {sc.loc[k, s]:.2f}" for k in ("garage", "go-straight", "RSG-L18-s0-ag", REF)) + " |")
        top = an.sort_values(ascending=False).index[:15]
        P(f"\nTop-15 items by new a: {', '.join(f'{s[-8:]} {an[s]:.1f} (old {ao[s]:.1f})' for s in top)}.")
        P(f"Spearman(a old, a new) over the {len(it)} items: {an.corr(ao, method='spearman'):.2f}.")
        pd.DataFrame(dict(a_old=ao, a_new=an, b_old=bo, b_new=bn, changed_reads=flip.reindex(it))).round(4).to_csv(out / "items.csv")
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "pai_pcs_v2.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("collect"); c.add_argument("--out", required=True)
    f = s.add_parser("fit"); f.add_argument("--rows", required=True); f.add_argument("--out", required=True)
    f.add_argument("--variants", default="", help="comma list; - = none (bootstrap shards only)"); f.add_argument("--boot", type=int, default=0)
    f.add_argument("--boot0", type=int, default=0); f.add_argument("--old", action="store_true", help="fit the old-scorer scores (fits_old/)")
    r = s.add_parser("report"); r.add_argument("--rows", required=True); r.add_argument("--out", required=True); r.add_argument("--gap", default="")
    p.add_argument("--gs", choices=("fill", "mask", "rerun"), default="fill", help="go-straight on GS-v2's 5 thread-cap failures")
    a = p.parse_args()
    global GS_MODE
    GS_MODE = a.gs
    dict(collect=cmd_collect, fit=cmd_fit, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
