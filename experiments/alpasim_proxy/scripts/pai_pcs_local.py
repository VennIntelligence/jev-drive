#!/usr/bin/env python3
"""PAI-PCS-LOCAL: a local ruler for the private PAI-board Policy Capability Score, the PAI analogue of pcs_local.py (decision 382).

The PAI board's PCS is the same Drive-IRT zoib 2PL refit over every submission's per-scene scores; locally the items are the 281 PAI
scenes of the stored PAI 281 reads (one chunk table, hash 28800209) and the pool rows mimic the board's family composition (counted
from leaderboard_pai_20261019.json, `BANDS`). Ability is mapped to board PCS by least squares on the three subjects with a private PAI
PCS: garage model_0014 1819, the host go-straight 1446, ours RSG-L18-s0 + g10 + ACC + GAP 2089. Subjects outside the pool are scored as
one more submission under the fixed items (MAP, pcs_local.theta_map).

  pai_pcs_local.py collect --out ROWS.csv.gz [--only a,b --base OLD.csv.gz]   box (repo root, .venv python): per-scene score / type / RPC
                                                        flag of every source (--only: just those labels, other rows copied from --base)
  pai_pcs_local.py r5 --rows ROWS.csv.gz --out DIR --facts DIR --subjects a=FILE,...   the PCS-LOCAL-R5 report (paired vs NTV0, class values)
  pai_pcs_local.py fit --rows ROWS.csv.gz --out DIR [--variants a,b] [--boot N]    Mac (tmp/pcs_local/.venv with drive-irt 82ddd5a)
  pai_pcs_local.py report --rows ROWS.csv.gz --out DIR [--gap scenes.csv]          -> DIR/report.md, DIR/pai_pcs_local.json
--gap: decision 381's per-scene loss table ($DATA_DIR/runs/route_follow/pai_gap/scenes.csv) for the easy-item cause classes.
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

# label -> (read root under $DATA_DIR/runs, family). family: third = garage / go-straight; band = our submission serve (ACC + GAP 0.7,2.7,
# g10), the stand-ins for the board's strong band; other = our other serve configurations (scored out of pool unless a variant pools them).
SRC = {
    "garage": ("alpasim_proxy/pai_pcs/G0014", "third"),
    "go-straight": ("alpasim_proxy/pai_pcs/GS", "third"),
    "RSG+AG": ("route_follow/acc_gap/read/s2", "band"),
    "NTV0-s0+AG": ("defect_fix/pai_accgap/NTV0-L18-s0", "band"),
    "NTV0-s1+AG": ("defect_fix/pai_accgap/NTV0-L18-s1", "band"),
    "NTV0-s0+AG+hl": ("defect_fix/pai_accgap/NTV0-L18-s0-hl", "band"),
    "JTN-s0+AG": ("defect_fix/pai_accgap/JTN-L18-s0", "band"),
    "JTN-s1+AG": ("defect_fix/pai_accgap/JTN-L18-s1", "band"),
    "NTVX-s0+AG": ("defect_fix/pai_accgap/NTVX-L18-s0", "band"),
    "NTVXS-s0+AG": ("defect_fix/pai_accgap/NTVXS-L18-s0", "band"),
    "JTXS-s1+AG": ("defect_fix/pai_accgap/JTXS-L18-s1", "band"),
    "BNC2K-s1+AG": ("defect_fix/pai_accgap/BNC2K-L18-s1", "band"),
    "STANDFMX-s0+AG": ("defect_fix/stand_b/pai281/STANDFMX-L18-s0", "band"),
    "STANDFMX-s1+AG": ("defect_fix/stand_b/pai281/STANDFMX-L18-s1", "band"),
    "STANDFMXH-s0+AG": ("defect_fix/stand_b/pai281/STANDFMXH-L18-s0", "band"),
    "RSG-g10": ("route_follow/pai_g10/g10", "other"),
    "RSG-g10+hl": ("route_follow/pai_g10/g10hl", "other"),
    "RBSG2-g10": ("route_follow/pai_g10/rbsg2g10", "other"),
    "RSG-g5": ("route_follow/pai_g10/g5", "other"),
    "RSG+ACC": ("route_follow/pai_g10/acc", "other"),
    "RSG-s0+ACC (compile)": ("cl_infra/compile/pai/RSG-L18-s0", "other"),
    "RSG-s1+ACC": ("cl_infra/compile/pai/RSG-L18-s1", "other"),
    "RSGC-s0+ACC": ("cl_infra/compile/pai/RSGC-L18-s0", "other"),
    "RSGC-s1+ACC": ("cl_infra/compile/pai/RSGC-L18-s1", "other"),
    "NR1F-s0+ACC": ("defect_fix/pai/NR1F-L18-s0", "other"),
    "NR1FB10-s0+ACC": ("defect_fix/pai/NR1FB10-L18-s0", "other"),
    "JTN-s0": ("defect_fix/pai_nohold/JTN-L18-s0", "other"),
    "NTV0-s0 (N0)": ("route_follow/pai_input_loop/N0", "other"),
    "NTV0-s0+AG+keywarp": ("route_follow/pai_input_loop/KA", "other"),
    "NTV0-s0+keywarp": ("route_follow/pai_input_loop/KN", "other"),
    "RSG+ACC+desire": ("desire_cap/d0/pai281", "other"),
    "RSG-g20": ("route_follow/recovery_envelope/pai/RSG-L18-s0", "other"),
    "RCVL-g20": ("route_follow/recovery_envelope/pai/RCVL-L18-s0", "other"),
    "RCVR-g20": ("route_follow/recovery_envelope/pai/RCVR-L18-s0", "other"),
    "RCVRL-g20": ("route_follow/recovery_envelope/pai/RCVRL-L18-s0", "other"),
    "RCVLF-g20": ("route_follow/recovery_envelope/pai/RCVLF-L18-s0", "other"),
    "RCVRLF-g20": ("route_follow/recovery_envelope/pai/RCVRLF-L18-s0", "other"),
    # round 2 / round 5 PAI 281 ACC + GAP reads (plans: PAI-PCS-LOCAL-R5): scored out of pool under the d392 ruler
    "R2C-s0+AG": ("round2/pai281/R2C-L18-s0-ag", "other"),
    "R2J-s0+AG": ("round2/pai281/R2J-L18-s0-ag", "other"),
    "R3MV-s0+AG": ("round5/r5_281/pai281/R3MV-L18-s0-ag", "other"),
    # PAI-PARTIALS serve sweep on NTV0-s0 (plans/2026-10-20-pai-partials-sweep-prereg.md): out of pool, so the fit is unchanged
    **{f"PP-{a}": (f"alpasim_proxy/pai_partials/{a}", "other") for a in ("R0", "V1", "V1b", "V2", "V3", "J")},
}
ANCHORS = {"garage": 1819, "go-straight": 1446, "RSG+AG": 2089}          # private PAI PCS, board 2026-10-19 13:49 UTC
BOARD_MEAN = {"garage": 0.436, "go-straight": 0.140, "RSG+AG": 0.632}
TOP3, TOP1 = 2186, 2250
# PAI board 2026-10-19, 104 rows, assigned to the stand-in nearest in PCS and behaviour (counted by `bands` below):
#   >= 1940 (32 rows: ACM, VAIL, the Anonymous host, BASICLAB ... ours, Host internal; mean 0.50-0.74)      -> our ACC + GAP band
#   1750-1939 (35 rows: 17 py123d-garage LTF rows, Ulsan's 0014 copy, and participants scoring like them)   -> garage 0014
#   1440-1749 (14 rows: Alpain mars / jupiter, starter copies, host go-straight; mean 0.14-0.40)           -> go-straight
#   < 1440 (23 rows: VaVAM / WAM / broken drivers, policy_4; mean 0.03-0.19): no stand-in, left out (as d382 left out WAM / VaVAM)
CUTS = (1940, 1750, 1440)


def bands(board):
    p = np.array([r["policy_capability_score"] for r in board])
    return dict(band=int((p >= CUTS[0]).sum()), garage=int(((p >= CUTS[1]) & (p < CUTS[0])).sum()),
                gs=int(((p >= CUTS[2]) & (p < CUTS[1])).sum()), low=int((p < CUTS[2]).sum()))


# ---------------------------------------------------------------- box: collect
def cmd_collect(a):
    sys.path[:0] = [str(REPO / "experiments/defect_fix/scripts")]
    from pai281 import load, ptype
    D = os.environ["DATA_DIR"] + "/runs/"
    rows = []
    only = set(a.only.split(",")) if a.only else None
    for lab, (root, fam) in SRC.items():
        if only is not None and lab not in only:
            continue
        if not os.path.isdir(D + root):
            print(f"{lab}: missing {root}"); continue
        R, bad = load(D + root)
        for s, r in R.items():
            rows.append(dict(subject=lab, family=fam, scene=s, score=float(r.get("score") or 0.0), type=ptype(r), rpc=s in bad,
                             reason=str(r.get("failure_reason") or "").splitlines()[0][:80] if r.get("failure_reason") else ""))
        print(f"{lab}: {len(R)} scenes, {len(bad)} RPC errors, mean {np.mean([float(r.get('score') or 0) for r in R.values()]):.4f}")
    new = pd.DataFrame(rows)
    if only is not None:
        old = pd.read_csv(a.base)
        new = pd.concat([old[~old.subject.isin(only)], new], ignore_index=True)
    new.to_csv(a.out, index=False)


# ---------------------------------------------------------------- Mac: matrix, pools, fits
def table(rows):
    R = pd.read_csv(rows)
    sc = R.pivot_table(index="subject", columns="scene", values="score", aggfunc="first")
    rpc = R.pivot_table(index="subject", columns="scene", values="rpc", aggfunc="first").fillna(False).astype(bool)
    sc = sc.where(~rpc.reindex_like(sc).fillna(False))                     # an RPC-error rollout is infrastructure, not behaviour (d327)
    ty = R.pivot_table(index="subject", columns="scene", values="type", aggfunc="first").reindex_like(sc)
    fam = R.groupby("subject").family.first()
    return sc, ty, fam


def pool(sc, fam, name, nb):
    """-> row subjects with repeats. Weights from the board bands nb; the variant name scales one band."""
    band = [s for s in sc.index if fam[s] == "band"]
    k = dict(band=nb["band"], garage=nb["garage"], gs=nb["gs"])
    if name == "low-as-gs":
        k["gs"] += nb["low"]
    for key, f in (("garage", "garage"), ("band", "band"), ("gs", "gs")):
        if name.startswith(f"{f}x"):
            k[key] = round(k[key] * float(name.split("x")[1]))
    subs = ["garage"] * k["garage"] + ["go-straight"] * k["gs"]
    if name == "allours":                                                # every one of our reads, the band weight spread over all of them
        band = [s for s in sc.index if fam[s] != "third"]
    w = max(1, round(k["band"] / len(band)))
    if name == "dedup":
        return ["garage", "go-straight"] + band
    return subs + [s for s in band for _ in range(w)]


VARIANTS = {"paimix": 42, "paimix-s1": 1, "paimix-s2": 2, "garagex0.5": 42, "garagex2": 42, "bandx0.5": 42, "bandx2": 42, "gsx0.5": 42,
            "gsx2": 42, "low-as-gs": 42, "allours": 42, "dedup": 42, "paimix-1pl": 42}


def items_of(sc, subs):
    """Complete-matrix rule (evaluate.py): scenes every pool subject scored without an RPC error."""
    return sc.columns[sc.loc[sorted(set(subs))].notna().all(axis=0)]


def cmd_fit(a):
    from pcs_local import fit, guard, matrix
    sc, _, fam = table(a.rows)
    nb = bands(json.load(open(HERE.parent / "results/leaderboard_pai_20261019.json"))["leaderboard"])
    out = Path(a.out) / "fits"; out.mkdir(parents=True, exist_ok=True)
    for v in ([] if a.variants == "-" else a.variants.split(",") if a.variants else VARIANTS):
        base = v.split("-s")[0] if v.startswith("paimix-s") else "paimix" if v == "paimix-1pl" else v
        subs = pool(sc, fam, base, nb)
        cols = items_of(sc, subs)
        M, names = matrix(sc, subs, cols)
        assert guard(M), v
        f = fit(M, names, cols, seed=VARIANTS.get(v, 42), fix=v.endswith("1pl"))
        np.savez(out / f"{v}.npz", subjects=np.array(names), items=np.array(cols, dtype=str), **f)
        print(v, len(names), "rows", len(cols), "items", flush=True)
    if a.boot:                                                           # scene bootstrap of the whole pipeline (items resampled, refit)
        subs = pool(sc, fam, "paimix", nb)
        cols = np.array(items_of(sc, subs))
        B = []
        for b in range(a.boot0, a.boot0 + a.boot):
            rng = np.random.default_rng(b)
            pick = cols[rng.integers(0, len(cols), len(cols))]
            sb = sc[pick]; sb.columns = [f"{c}#{i}" for i, c in enumerate(pick)]
            M, names = matrix(sb, subs, sb.columns)
            f = fit(M, names, list(sb.columns))
            z = Ruler(dict(f, subjects=np.array(names), items=np.array(sb.columns)), sb)
            B.append(z.pcs)
            print("boot", b, flush=True)
        pd.DataFrame(B).to_csv(Path(a.out) / f"boot_{a.boot0:03d}.csv", index=False)


class Ruler:
    """Fixed items; ability of a pool subject = its fitted posterior mean, of any other = MAP under the items; PCS = least squares of
    ability on the three anchors' private PCS (`drop`: leave that anchor out of the map)."""

    def __init__(self, f, sc, drop=None, mapall=False):
        from pcs_local import theta_map
        self.f, self.items = f, [str(c) for c in f["items"]]
        names = [str(s) for s in f["subjects"]]
        mu, sd = float(np.mean(f["theta"])), float(np.std(f["theta"]))
        fitted = {s: float(t) for s, t in zip(names, f["theta"]) if "#" not in s}
        self.theta, self.check = {}, {}
        for s in sc.index:
            y = sc.loc[s, self.items].to_numpy(float)
            if np.isnan(y).mean() > 0.05:
                continue
            m = float(theta_map(y, f, mu, sd)) if (mapall or s not in fitted or drop is None) else None
            if s in fitted and m is not None:
                self.check[s] = m - fitted[s]                            # MAP under the fitted items minus the VI posterior mean
            self.theta[s] = m if (mapall or s not in fitted) else fitted[s]
        self.sd = {s: float(d) for s, d in zip(names, f["theta_sd"]) if "#" not in s}
        an = [k for k in ANCHORS if k != drop]
        self.A = np.polyfit([self.theta[k] for k in an], [ANCHORS[k] for k in an], 1)
        self.pcs = {s: float(np.polyval(self.A, t)) for s, t in self.theta.items()}


# ---------------------------------------------------------------- report
def cmd_report(a):
    sc, ty, fam = table(a.rows)
    board = json.load(open(HERE.parent / "results/leaderboard_pai_20261019.json"))["leaderboard"]
    nb = bands(board)
    out = Path(a.out)
    Z = {v: Ruler(dict(np.load(out / f"fits/{v}.npz", allow_pickle=True)), sc) for v in VARIANTS if (out / f"fits/{v}.npz").exists()}
    z = Z["paimix"]
    Z["paimix, MAP for all"] = Ruler(dict(z.f), sc, mapall=True)
    L, J = [], {"bands": nb, "anchors": ANCHORS}
    P = L.append
    mean = sc.mean(1)
    n1 = (sc >= 1 - 1e-4).sum(1); n0 = (sc <= 1e-4).sum(1)
    P(f"Pool paimix: {len(z.f['subjects'])} rows, {len(z.items)} scenes (complete matrix over the pool). Band counts from the board: {nb}.\n")

    # anchor fit and leave-one-anchor-out
    P("## Anchor fit\n")
    P("| anchor | private PCS (mean) | local 281 mean | ability | fitted PCS | residual | leave-this-anchor-out PCS |")
    P("|---|--:|--:|--:|--:|--:|--:|")
    J["anchor_fit"] = {}
    for k in ANCHORS:
        lo = Ruler(dict(np.load(out / "fits/paimix.npz", allow_pickle=True)), sc, drop=k).pcs[k]
        P(f"| {k} | {ANCHORS[k]} ({BOARD_MEAN[k]:.3f}) | {mean[k]:.3f} | {z.theta[k]:+.2f} | {z.pcs[k]:.0f} | {z.pcs[k] - ANCHORS[k]:+.0f} | {lo:.0f} ({lo - ANCHORS[k]:+.0f}) |")
        J["anchor_fit"][k] = dict(fit=z.pcs[k], loo=lo)
    P(f"\nMap: PCS = {z.A[0]:.1f} x ability + {z.A[1]:.0f}. RSG + ACC + GAP is the highest anchor; ability above it is extrapolated.")
    ck = pd.Series(z.check)
    P(f"Out-of-pool scoring check (pool subjects: MAP under the fitted items minus the fitted posterior mean): median {ck.median():+.2f}, "
      f"max |diff| {ck.abs().max():.2f} ({ck.abs().idxmax()}), = {ck.abs().max() * z.A[0]:.0f} PCS.\n")

    # predictions
    bf = sorted(out.glob("boot_*.csv"))
    boot = pd.concat([pd.read_csv(f) for f in bf], ignore_index=True) if bf else None
    P("## Predicted private PAI PCS\n")
    P("CI: scene bootstrap of the whole pipeline (281 scenes resampled, pool refit, anchors re-mapped; "
      f"{len(boot) if boot is not None else 0} draws), 2.5-97.5 %. 'beyond 2089' = extrapolation above the top anchor.\n")
    P("| subject | family | local mean | exact 1 | zeros | ability | PCS paimix | 95 % CI | min-max over pool variants | vs RSG + ACC + GAP | to top-3 2186 |")
    P("|---|---|--:|--:|--:|--:|--:|---|---|--:|--:|")
    J["pred"] = {}
    order = sorted(z.pcs, key=lambda s: -z.pcs[s])
    for s in order:
        vv = [Zv.pcs[s] for Zv in Z.values() if s in Zv.pcs]
        ci = (np.nanpercentile(boot[s], 2.5), np.nanpercentile(boot[s], 97.5)) if boot is not None and s in boot else (np.nan, np.nan)
        P(f"| {s} | {fam[s]} | {mean[s]:.3f} | {n1[s]} | {n0[s]} | {z.theta[s]:+.2f} | **{z.pcs[s]:.0f}** | [{ci[0]:.0f}, {ci[1]:.0f}] | "
          f"{min(vv):.0f}-{max(vv):.0f} | {z.pcs[s] - z.pcs['RSG+AG']:+.0f} | {z.pcs[s] - TOP3:+.0f} |")
        J["pred"][s] = dict(family=fam[s], mean=float(mean[s]), pcs=z.pcs[s], ci=[float(c) for c in ci], vmin=min(vv), vmax=max(vv),
                            by_variant={v: Zv.pcs.get(s) for v, Zv in Z.items()})
    if boot is not None:
        P("\nPaired bootstrap differences (same draws):\n")
        P("| difference | median | 95 % CI | P(> 0) |"); P("|---|--:|---|--:|")
        for x, y in (("NTV0-s0+AG", "RSG+AG"), ("NTV0-s1+AG", "RSG+AG"), ("STANDFMX-s0+AG", "RSG+AG"), ("STANDFMX-s0+AG", "NTV0-s0+AG"),
                     ("JTN-s0+AG", "RSG+AG"), ("NTVX-s0+AG", "RSG+AG")):
            if x in boot and y in boot:
                d = boot[x] - boot[y]
                P(f"| {x} - {y} | {d.median():+.0f} | [{d.quantile(.025):+.0f}, {d.quantile(.975):+.0f}] | {(d > 0).mean():.2f} |")

    # sensitivity
    P("\n## Sensitivity to the pool\n")
    keys = ["NTV0-s0+AG", "NTV0-s1+AG", "STANDFMX-s0+AG", "JTN-s0+AG", "NTVX-s0+AG", "garage", "go-straight", "RSG+AG"]
    P("| variant | rows | scenes | anchor RMSE | " + " | ".join(keys) + " |"); P("|---|--:|--:|--:|" + "--:|" * len(keys))
    for v, Zv in Z.items():
        rm = np.sqrt(np.mean([(Zv.pcs[k] - ANCHORS[k]) ** 2 for k in ANCHORS]))
        P(f"| {v} | {len(Zv.f['subjects'])} | {len(Zv.items)} | {rm:.0f} | " + " | ".join(f"{Zv.pcs.get(k, np.nan):.0f}" for k in keys) + " |")
    J["variants"] = {v: {k: Zv.pcs.get(k) for k in keys} for v, Zv in Z.items()}

    # easy items: garage = 1, NTV0 + ACC + GAP < 1
    if a.gap:
        G = pd.read_csv(a.gap).set_index("scene")
        it = z.items
        g = sc.loc["garage", it]; n = sc.loc["NTV0-s0+AG", it]
        f = z.f
        aa = pd.Series(f["a"], index=it)
        P("\n## Easy items: garage scores 1, NTV0-s0 + ACC + GAP does not (decision 381 cause classes)\n")
        e = [s for s in it if g[s] >= 1 - 1e-4 and n[s] < 1 - 1e-4]
        y0 = sc.loc["NTV0-s0+AG", it].to_numpy(float)
        base = pcs_of(z, y0)
        cls = lambda s: (G.loc[s, "cls"] + (f" / {G.loc[s, 'sub']}" if isinstance(G.loc[s, "sub"], str) else "")) if s in G.index else "?"  # noqa: E731
        df = pd.DataFrame(dict(scene=e, score=[n[s] for s in e], cls=[cls(s) for s in e], a=[aa[s] for s in e],
                               stratum=[G.loc[s, "stratum"] if s in G.index else "?" for s in e]))
        P(f"{len(e)} easy items lost ({(df.score <= 1e-4).sum()} zeros, {(df.score > 1e-4).sum()} partials, {np.sum(1 - df.score):.1f} points); "
          f"garage = 1 on {int((g >= 1 - 1e-4).sum())} of {len(it)} scenes, NTV0 = 1 on {int((n >= 1 - 1e-4).sum())}.\n")
        P("| cause (d381 class / sub) | items | points | median a | PCS if these -> 1 (paimix) |"); P("|---|--:|--:|--:|--:|")
        for c, d in df.groupby("cls"):
            y = y0.copy(); y[np.isin(it, d.scene)] = 1.0
            P(f"| {c} | {len(d)} | {np.sum(1 - d.score):.1f} | {d.a.median():.2f} | {pcs_of(z, y) - base:+.0f} |")
        y = y0.copy(); y[np.isin(it, df.scene)] = 1.0
        P(f"| **all easy items** | {len(df)} | {np.sum(1 - df.score):.1f} | {df.a.median():.2f} | {pcs_of(z, y) - base:+.0f} |")
        hard = [s for s in it if g[s] < 1 - 1e-4 and n[s] < 1 - 1e-4]
        y = y0.copy(); y[np.isin(it, hard)] = 1.0
        P(f"| (non-easy losses: garage < 1 too) | {len(hard)} | {np.sum(1 - n[hard]):.1f} | {aa[hard].median():.2f} | {pcs_of(z, y) - base:+.0f} |")
        win = [s for s in it if n[s] >= 1 - 1e-4 and g[s] < 1 - 1e-4]
        P(f"\nNTV0 + ACC + GAP scores 1 where garage does not on {len(win)} scenes (median a {aa[win].median():.2f}).")
        P("\nItem discrimination by d381 stratum (paimix): " + ", ".join(
            f"{k} {aa[[s for s in it if s in G.index and G.loc[s, 'stratum'] == k]].median():.2f}" for k in ("static", "launch", "stop/lead", "straight", "turn")))
        df.to_csv(out / "easy_items.csv", index=False)
        J["easy"] = df.groupby("cls").size().to_dict()
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "pai_pcs_local.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


# ---------------------------------------------------------------- PCS-LOCAL-R5: more stored reads on the same ruler
CLASSES = [("partial rear-hit", ["rear-hit"]), ("partial slow", ["slow", "creep", "no-launch"]), ("partial other contact", ["hit"]),
           ("zero overshoot (passes the human's rest point)", ["overshoot"]), ("zero route-turn", ["route-turn"]), ("zero offroad", ["offroad"]),
           ("zero other (AF-front, side, drift / lane change)", ["AF-front", "AF-side", "side", "zero-other"])]


def cmd_r5(a):
    sc, _, fam = table(a.rows)
    out = Path(a.out)
    z = Ruler(dict(np.load(out / "fits/paimix.npz", allow_pickle=True)), sc)
    subs = dict(x.split("=") for x in a.subjects.split(","))              # ruler label -> facts file
    bf = sorted(out.glob("boot_*.csv"))
    boot = pd.concat([pd.read_csv(f) for f in bf], ignore_index=True)
    ref = "NTV0-s0+AG"
    P = (L := []).append
    P(f"Pool paimix: {len(z.f['subjects'])} rows, {len(z.items)} scenes. Map: PCS = {z.A[0]:.1f} x ability + {z.A[1]:.0f}; {len(boot)} bootstrap draws.\n")
    P("| subject | scenes scored | local mean | zeros | ability | PCS | 95 % CI | vs NTV0-s0 (paired) | 95 % CI | P(> 0) |"); P("|---|--:|--:|--:|--:|--:|---|--:|---|--:|")
    J = {}
    mean = sc[z.items].mean(1)
    for s in [ref, *subs]:
        if s == ref:
            P(f"| {s} (reference) | {int(sc.loc[s, z.items].notna().sum())} | {mean[s]:.3f} | {int((sc.loc[s, z.items] <= 1e-4).sum())} | {z.theta[s]:+.2f} | "
              f"**{z.pcs[s]:.0f}** | [{np.percentile(boot[s], 2.5):.0f}, {np.percentile(boot[s], 97.5):.0f}] | | | |")
            J[s] = dict(pcs=z.pcs[s], ci=[float(np.percentile(boot[s], q)) for q in (2.5, 97.5)])
            continue
        d = boot[s] - boot[ref]
        P(f"| {s} | {int(sc.loc[s, z.items].notna().sum())} | {mean[s]:.3f} | {int((sc.loc[s, z.items] <= 1e-4).sum())} | {z.theta[s]:+.2f} | **{z.pcs[s]:.0f}** | "
          f"[{np.percentile(boot[s], 2.5):.0f}, {np.percentile(boot[s], 97.5):.0f}] | {d.median():+.0f} (point {z.pcs[s] - z.pcs[ref]:+.0f}) | "
          f"[{d.quantile(.025):+.0f}, {d.quantile(.975):+.0f}] | {(d > 0).mean():.2f} |")
        J[s] = dict(pcs=z.pcs[s], ci=[float(np.percentile(boot[s], q)) for q in (2.5, 97.5)], diff_median=float(d.median()),
                    diff_point=z.pcs[s] - z.pcs[ref], diff_ci=[float(d.quantile(.025)), float(d.quantile(.975))], p_pos=float((d > 0).mean()))
    P("\nPaired differences between the new reads (same draws): " + "; ".join(
        f"{x} - {y} {(boot[x] - boot[y]).median():+.0f} [{(boot[x] - boot[y]).quantile(.025):+.0f}, {(boot[x] - boot[y]).quantile(.975):+.0f}]"
        for x, y in (("R3MV-s0+AG", "R2C-s0+AG"), ("R3MV-s0+AG", "R2J-s0+AG"), ("R2J-s0+AG", "R2C-s0+AG"))) + "\n")
    # class values: set the class's scenes to 1 in that read, items fixed (MAP), PCS delta
    P("## PCS value of each loss class if fixed (scenes -> 1, items fixed, MAP; paimix)\n")
    P("Class = pai_fix.py `fclass` of the stored facts. Read the ranking, not the sizes: the ceiling sits above the anchors.\n")
    P("| class | " + " | ".join(subs_ := [ref, *subs]) + " |"); P("|---|" + "--:|" * len(subs_))
    V, N = {}, {}
    facts = {s: pd.read_csv(Path(a.facts) / (subs[s] if s in subs else NTV0_FACTS)).set_index("scene") for s in subs_}
    for s in subs_:
        y0 = sc.loc[s, z.items].to_numpy(float)
        F = facts[s].reindex(z.items)
        base = pcs_of(z, y0)
        V[s], N[s] = {}, {}
        groups = CLASSES + [("all partials", CLASSES[0][1] + CLASSES[1][1] + CLASSES[2][1]), ("all zeros", sum((c[1] for c in CLASSES[3:]), []))]
        for name, fc in groups:
            m = F.fclass.isin(fc).to_numpy() & np.isfinite(y0)
            y = y0.copy(); y[m] = 1.0
            V[s][name] = pcs_of(z, y) - base; N[s][name] = (int(m.sum()), float(np.sum(1 - y0[m])))
    for name, _ in CLASSES + [("all partials", 0), ("all zeros", 0)]:
        P(f"| {name} | " + " | ".join(f"{N[s][name][0]} sc, {N[s][name][1]:.1f} pt: **{V[s][name]:+.0f}**" for s in subs_) + " |")
    J["class_values"] = V; J["class_counts"] = N
    # pool sensitivity and the items that carry the gap
    Z = {v: Ruler(dict(np.load(out / f"fits/{v}.npz", allow_pickle=True)), sc) for v in VARIANTS if v != "allours" and (out / f"fits/{v}.npz").exists()}
    ks = [ref, *subs, "garage", "go-straight", "RSG+AG"]
    P("\n## Pool sensitivity (PCS)\n")
    P("| variant | anchor RMSE | " + " | ".join(ks) + " | R3MV - NTV0 |"); P("|---|--:|" + "--:|" * (len(ks) + 1))
    for v, Zv in Z.items():
        rm = np.sqrt(np.mean([(Zv.pcs[k] - ANCHORS[k]) ** 2 for k in ANCHORS]))
        P(f"| {v} | {rm:.0f} | " + " | ".join(f"{Zv.pcs[k]:.0f}" for k in ks) + f" | {Zv.pcs['R3MV-s0+AG'] - Zv.pcs[ref]:+.0f} |")
    J["variants"] = {v: {k: Zv.pcs[k] for k in ks} for v, Zv in Z.items()}
    it = z.items; aa = pd.Series(z.f["a"], index=it)
    top = aa.sort_values(ascending=False).index[:12]
    band = [s for s in sc.index if fam[s] == "band"]
    P("\n## The 12 most discriminating items (a >= " + f"{aa[top].min():.1f}" + "): scores of the reads and of the 13 pool band reads\n")
    P("| scene | a | b | NTV0 | R2C | R2J | R3MV | R3MV class | band reads scoring 1 (of 13) |"); P("|---|--:|--:|--:|--:|--:|--:|---|--:|")
    bb = pd.Series(z.f["b"], index=it)
    for sname in top:
        P(f"| {sname[-8:]} | {aa[sname]:.2f} | {bb[sname]:+.1f} | " + " | ".join(f"{sc.loc[k, sname]:.2f}" for k in (ref, *subs)) +
          f" | {facts['R3MV-s0+AG'].fclass.get(sname, '?')} | {int((sc.loc[band, sname] >= 1 - 1e-4).sum())} |")
    lo = aa[aa >= 3].index
    P(f"\n{len(lo)} items have a >= 3. Scored 0 on them: " + ", ".join(f"{k} {int((sc.loc[k, lo] <= 1e-4).sum())}" for k in (ref, *subs)) +
      "; mean score on them: " + ", ".join(f"{k} {sc.loc[k, lo].mean():.3f}" for k in (ref, *subs)) + ".")
    P("\nZeros on the a >= 3 items (all 13 pool band reads score 1 on nearly all of them), by cause, and whether the garage 0014 scores 1 there:\n")
    P("| read | scene | a | fclass | garage | go-straight |"); P("|---|---|--:|---|--:|--:|")
    for k in subs:
        for sname in [x for x in lo if sc.loc[k, x] <= 1e-4]:
            P(f"| {k} | {sname[-8:]} | {aa[sname]:.2f} | {facts[k].fclass.get(sname, '?')} | {sc.loc['garage', sname]:.2f} | {sc.loc['go-straight', sname]:.2f} |")
    # R3MV minus NTV0 class counts are in round5/results/pai281_R3MV_ag.md
    (out / "r5_report.md").write_text("\n".join(L) + "\n")
    (out / "pai_pcs_local_r5.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


NTV0_FACTS = "NTV0-L18-s0-281ag.csv"


def pcs_of(z, y):
    from pcs_local import theta_map
    return float(np.polyval(z.A, theta_map(y, z.f, float(np.mean(z.f["theta"])), float(np.std(z.f["theta"])))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("collect"); c.add_argument("--out", required=True); c.add_argument("--only", default=""); c.add_argument("--base", default="")
    q = s.add_parser("r5"); q.add_argument("--rows", required=True); q.add_argument("--out", required=True); q.add_argument("--facts", required=True)
    q.add_argument("--subjects", required=True, help="label=facts file name,... (pai_fix facts csv with fclass per scene)")
    f = s.add_parser("fit"); f.add_argument("--rows", required=True); f.add_argument("--out", required=True)
    f.add_argument("--variants", default="", help="comma list; - = none (bootstrap shards only)"); f.add_argument("--boot", type=int, default=0)
    f.add_argument("--boot0", type=int, default=0, help="first bootstrap draw (seed), for parallel shards")
    r = s.add_parser("report"); r.add_argument("--rows", required=True); r.add_argument("--out", required=True); r.add_argument("--gap", default="")
    a = p.parse_args()
    dict(collect=cmd_collect, fit=cmd_fit, report=cmd_report, r5=cmd_r5)[a.cmd](a)


if __name__ == "__main__":
    main()
