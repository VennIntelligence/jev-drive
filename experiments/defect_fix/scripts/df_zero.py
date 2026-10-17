#!/usr/bin/env python3
"""Zero overlap and speed relative to the human on the stored held 460 @18 / public 1491 @10 reads (d346 follow-up; descriptive, no prereg line).

  df_zero.py collect [--out F]   (box, repo root, op-train python; read-only on stored runs) one row per (read, tag, scene) of every stored rollout:
                                 score, nx_regress.failure_type, collision / offroad / corridor flags, dist_traveled_m, gt_dist_traveled_m; duplicate
                                 reads of one tag (RSG-L18-s0 held re-reads, YR / APY public run groups) get rep = 1, 2, ...
                                 -> $DATA_DIR/runs/defect_fix/zero/rows.csv.gz
  df_zero.py report ROWS --out DIR   (Mac) results/zero_overlap.md, speed tables, figs/zo_speed_*.png

Definitions (fixed before the tables were read):
  zero class    nx_regress.failure_type of the score-0 rollout, grouped: AF-front, AF-side, AF-rear (at-fault rear), offroad, corridor (corridor only;
                a zero with both offroad and corridor counts as offroad), route-fail, zero-other. Partial classes (0 < score < 0.98): rear-hit (a
                non-at-fault rear contact), hit (other non-at-fault contact), no-launch, creep (barely / crawl: < 30 % of the log), slow.
  older models  RSG-L18-s0, RTB1ARO-L18-s0 (V2), V2AC-L18-s0; "any older" adds P0 (held only), YR10m10-F-s0, APY10m10-AB-s0.
  new zero      zero for a new model and for none of the older models on that read. Seed-consistent: zero for both seeds of the chain.
  speed         dv = (ego arc - human arc) / T over the scored window (T = 5.5 s public, 9.5 s held; the arcs are the scorer's dist_traveled_m and
                gt_dist_traveled_m), progress ratio = ego arc / human arc (scenes with a human arc > 5 m). Scenes scoring 1.0 are excluded.
  strata        from the human's first 5.5 s of the public window (experiments/alpasim_proxy/results/scene_mix/public_windows.csv), first match:
                static (max displacement < 1 m), launch (v0 < 1 m/s), braking (v_end < v0 - 2 m/s), turn (heading change >= 30 deg), cruise
                (v0 >= 5, |v_end - v0| < 1, heading change < 5 deg), other.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
WIN = dict(held=9.5, pub=5.5)
NEW = ("NR1F-L18-s0", "NR1F-L18-s1", "NR1FB10-L18-s0", "NR1FB10-L18-s1")
OLDER = ("RSG-L18-s0", "RTB1ARO-L18-s0", "V2AC-L18-s0")
REF = ("P0", "YR10m10-F-s0", "APY10m10-AB-s0")
LABEL = {"RSG-L18-s0": "RSG", "RTB1ARO-L18-s0": "V2", "V2AC-L18-s0": "V2AC", "NR1F-L18-s0": "NR1F-s0", "NR1F-L18-s1": "NR1F-s1",
         "NR1FB10-L18-s0": "NR1FB10-s0", "NR1FB10-L18-s1": "NR1FB10-s1", "P0": "P0", "YR10m10-F-s0": "YR", "APY10m10-AB-s0": "APY"}
COLS = ("read", "tag", "rep", "src", "scene", "score", "type", "at_fault", "front", "side", "rear", "offroad", "corridor", "dist", "gt")


def sources():
    """(read, tag, rep, [run dirs]) of every stored read used here."""
    R = DATA / "runs"
    held = lambda t: sorted(glob.glob(f"{R}/route_mem/read/{t}/c*/run"))  # noqa: E731
    S = [("held", t, 1, held(t)) for t in OLDER + NEW + ("P0",)]
    S += [("held", "RSG-L18-s0", 2, held("RSG-L18-s0-re1016")), ("held", "RSG-L18-s0", 3, sorted(glob.glob(f"{R}/onpol_long/roll/RSG-L18-s0-held/c*/run"))),
          ("held", "YR10m10-F-s0", 1, [f"{R}/alpasim_long/held/h18/YR10m10-F-s0"]), ("held", "APY10m10-AB-s0", 1, [f"{R}/alpasim_long/held/a18/APY10m10-AB-s0"])]
    S += [("pub", "RSG-L18-s0", 1, sorted(glob.glob(f"{R}/route_follow/guard/read10/RSG-L18-s0/b*")))]
    S += [("pub", t, 1, sorted(glob.glob(f"{R}/wait_cap/next/cont/wide/pub/{t}/b*"))) for t in ("RTB1ARO-L18-s0", "V2AC-L18-s0")]
    S += [("pub", t, 1, sorted(glob.glob(f"{R}/defect_fix/pub/{t}/b*"))) for t in NEW]
    for t in ("YR10m10-F-s0", "APY10m10-AB-s0"):     # AUD1's public reads: the run groups of ot3 / cf1; rep 1 = the rollout AUD1's per_scene.csv used
        S += [("pub", t, 0, sorted(glob.glob(f"{R}/alpasim/ot3/a/runs/{t}*/*")) + sorted(glob.glob(f"{R}/alpasim/cf1/a/runs/{t}*/*")))]
    return S


def collect(a):
    sys.path[:0] = [str(REPO / "experiments/wait_cap/scripts")]
    import nx_regress as NR
    aud = {}
    with open(DATA / "runs/alpasim/aud1_drift/per_scene.csv") as fh:
        for r in csv.DictReader(fh):
            aud[(r["driver"], r["scene"])] = r["rollout_id"]
    rows = []
    for read, tag, rep, dirs in sources():
        seen = {}
        for d in dirs:
            f = Path(d) / "aggregate/results-summary.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text())["rollouts"]:
                s, m = r["clipgt_id"], r.get("metrics") or {}
                if rep:
                    if s in seen:
                        continue
                    k = rep
                else:                                  # AUD1 groups: the rollout AUD1 used is rep 1, every further read of the scene rep 2, 3, ...
                    if r.get("rollout_id", m.get("rollout_id")) == aud.get((tag, s)):
                        k = 1
                    else:
                        k = 2 + sum(1 for x in seen.get(s, []) if x > 1)
                seen.setdefault(s, []).append(k)
                pos = lambda q: int((m.get(q) or 0) > 0)  # noqa: E731
                rows.append(dict(read=read, tag=tag, rep=k, src=str(d).replace(str(DATA) + "/", ""), scene=s, score=float(r.get("score") or 0.0),
                                 type=NR.failure_type(r), at_fault=pos("collision_at_fault"), front=pos("collision_front"), side=pos("collision_lateral"),
                                 rear=pos("collision_rear"), offroad=pos("offroad"), corridor=pos("left_corridor_laterally"),
                                 dist=float(m.get("dist_traveled_m") or 0.0), gt=float(m.get("gt_dist_traveled_m") or 0.0)))
        print(read, tag, rep, len(dirs), "dirs", sum(1 for x in rows if x["read"] == read and x["tag"] == tag), "rows so far", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows))


# ---------------------------------------------------------------- report (Mac)
def zclass(t):
    if t.startswith("AF-"):
        return t.split("+")[0]
    if "offroad" in t:
        return "offroad"
    return t if t in ("corridor", "route-fail", "zero-other") else t


def pclass(r):
    if r.score >= 0.98:
        return "ok"
    if r.type == "hit":
        return "rear-hit" if r.rear else "hit"
    return {"barely": "creep", "crawl": "creep"}.get(r.type, r.type)


def strata(P):
    import numpy as np
    return np.select([P.maxdisp < 1, P.v0 < 1, P.v_end < P.v0 - 2, P.dyaw >= 30, (P.v0 >= 5) & ((P.v_end - P.v0).abs() < 1) & (P.dyaw < 5)],
                     ["static", "launch", "braking", "turn", "cruise"], "other")


def report(a):
    import numpy as np
    import pandas as pd
    from jevdrive import stats
    out, figs = Path(a.out), REPO / "experiments/defect_fix/figs"
    figs.mkdir(exist_ok=True)
    D = pd.read_csv(a.rows)
    D["label"] = D.tag.map(LABEL)
    D.loc[D.groupby(["read", "tag", "scene"]).rep.transform("min").eq(D.rep), "rep"] = 1     # a scene AUD1 did not use (route fold) keeps its one read
    P = pd.read_csv(REPO / "experiments/alpasim_proxy/results/scene_mix/public_windows.csv")
    P = P[P.kind == "public"].dropna(subset=["v0"])
    st = dict(zip(P.scene, strata(P)))
    D["stratum"] = D.scene.map(st).fillna("other")
    D["zero"] = D.score == 0
    D["zc"] = np.where(D.zero, D.type.map(zclass), "")
    D["pc"] = [pclass(r) if r.score > 0 else "" for r in D.itertuples()]
    D["dv"] = (D["dist"] - D["gt"]) / D.read.map(WIN)
    D["ratio"] = np.where(D["gt"] > 5, D["dist"] / D["gt"].clip(lower=1e-6), np.nan)
    main = D[D.rep == 1]
    L = ["# Zero overlap and speed relative to the human: NR1F / NR1FB10 against the older models (stored reads)", "",
         "Descriptive follow-up to [decision 346](../../../research/decisions/346.md); no prereg line. Script: `scripts/df_zero.py` (definitions in its header); "
         "rows `results/zero_rows.csv.gz` (one per read, tag, scene and repeat). Held 460 @18 and public 1491 @10, one read per tag unless stated.", ""]
    Z = {(rd, t): set(g[g.zero].scene) for (rd, t), g in main.groupby(["read", "tag"])}
    N = {(rd, t): len(g) for (rd, t), g in main.groupby(["read", "tag"])}
    tags = [t for t in OLDER + NEW + REF]

    # 1. run-to-run flip rate
    L += ["## 1. Run-to-run flip rate (same checkpoint, read twice)", "",
          "| read | tag | reads compared | common scenes | zeros (a / b) | zero in exactly one | flip rate per scene | zeros shared / union |", "|---|---|---|--:|--:|--:|--:|--:|"]
    flip = {}
    for (rd, t), g in D.groupby(["read", "tag"]):
        reps = sorted(g.rep.unique())
        for r2 in reps[1:]:
            A, B = g[g.rep == 1].set_index("scene"), g[g.rep == r2].set_index("scene")
            c = A.index.intersection(B.index)
            if len(c) < 30:
                continue
            za, zb = A.loc[c, "zero"], B.loc[c, "zero"]
            x = int((za != zb).sum())
            flip.setdefault(rd, []).append(x / len(c))
            srcb = B.src.iloc[0].split("/run")[0]
            L.append(f"| {rd} | {LABEL[t]} | 1 vs {r2} (`{srcb}`) | {len(c)} | {int(za.sum())} / {int(zb.sum())} | {x} | {x / len(c):.3f} | "
                     f"{int((za & zb).sum())} / {int((za | zb).sum())} |")
    L.append("")

    # 2. overlap table
    L += ["## 2. Zeros of each new model against the older ones", "",
          "shared RSG = also zero for RSG; V2-not-RSG = zero for V2 or V2AC but not RSG; new (L18) = zero for none of RSG / V2 / V2AC; "
          "new (any) = additionally not zero for P0 (held only), YR or APY. Seed-consistent = zero for both seeds of the chain. "
          "The route-fail zeros (8 held, 12 public: the map route generation fails before any driving, the same scenes for every model) sit inside every "
          "model's zeros and inside 'shared RSG'.", "",
          "| read | model | scenes | zeros | shared RSG | V2-not-RSG | new (L18) | new (any) | seed-consistent | of them new (L18) |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    ov = {}
    for rd in ("held", "pub"):
        old = set().union(*[Z.get((rd, t), set()) for t in OLDER])
        anyold = old.union(*[Z.get((rd, t), set()) for t in REF])
        for t in NEW:
            z = Z[(rd, t)]
            other = t[:-1] + ("1" if t.endswith("0") else "0")
            both = z & Z[(rd, other)]
            o = ov[(rd, t)] = dict(zeros=len(z), rsg=len(z & Z[(rd, "RSG-L18-s0")]), v2=len((z & (Z[(rd, "RTB1ARO-L18-s0")] | Z[(rd, "V2AC-L18-s0")])) - Z[(rd, "RSG-L18-s0")]),
                                   new=sorted(z - old), new_any=sorted(z - anyold), both=len(both), both_new=sorted(both - old))
            L.append(f"| {rd} | {LABEL[t]} | {N[(rd, t)]} | {o['zeros']} | {o['rsg']} | {o['v2']} | {len(o['new'])} | {len(o['new_any'])} | {o['both']} | {len(o['both_new'])} |")
    L += ["", "Older models' zeros on the same reads: " + "; ".join(
        f"{rd} " + ", ".join(f"{LABEL[t]} {len(Z[(rd, t)])}/{N[(rd, t)]}" for t in tags if (rd, t) in Z and t not in NEW) for rd in ("held", "pub")), ""]
    for rd in ("held", "pub"):
        for c in ("NR1F", "NR1FB10"):
            s0, s1 = Z[(rd, f"{c}-L18-s0")], Z[(rd, f"{c}-L18-s1")]
            old = set().union(*[Z.get((rd, t), set()) for t in OLDER])
            bn = sorted((s0 & s1) - old)
            rows = main[(main.read == rd) & main.scene.isin(bn) & main.tag.isin([f"{c}-L18-s0", f"{c}-L18-s1"])]
            desc = ", ".join(f"{s.rsplit('-', 1)[1][:8]} ({'/'.join(rows[rows.scene == s].zc)}, {st.get(s, 'other')})" for s in bn)
            L.append(f"- {rd} {c}: seed-consistent new zeros {len(bn)}" + (f": {desc}" if bn else ""))
    L.append("")

    # 3. zero classes per model
    classes = ["AF-front", "AF-side", "AF-rear", "offroad", "corridor", "route-fail", "zero-other"]
    pcs = ["rear-hit", "hit", "no-launch", "creep", "slow"]
    for rd in ("held", "pub"):
        L += [f"## 3{'a' if rd == 'held' else 'b'}. Zero and partial classes, {rd}", "",
              "| model | scenes | zeros | " + " | ".join(classes) + " | " + " | ".join(f"partial {p}" for p in pcs) + " |", "|---|--:|--:|" + "--:|" * (len(classes) + len(pcs))]
        for t in tags:
            g = main[(main.read == rd) & (main.tag == t)]
            if g.empty:
                continue
            L.append(f"| {LABEL[t]} | {len(g)} | {int(g.zero.sum())} | " + " | ".join(str(int((g.zc == c).sum())) for c in classes) + " | "
                     + " | ".join(str(int((g.pc == p).sum())) for p in pcs) + " |")
        L.append("")
        # class of the new zeros
        L += [f"Class of the zeros new to each new model ({rd}; new = zero for none of RSG / V2 / V2AC):", ""]
        for t in NEW:
            g = main[(main.read == rd) & (main.tag == t) & main.scene.isin(ov[(rd, t)]["new"])]
            L.append(f"- {LABEL[t]}: " + (", ".join(f"{k} {v}" for k, v in g.zc.value_counts().items()) or "none")
                     + "; strata " + (", ".join(f"{k} {v}" for k, v in g.stratum.value_counts().items()) or "-"))
        L.append("")

    # 4. zero class by stratum (scene counts), held + pub
    L += ["## 4. Zeros by stratum", "", "| read | stratum | scenes | " + " | ".join(LABEL[t] for t in tags) + " |", "|---|---|--:|" + "--:|" * len(tags)]
    for rd in ("held", "pub"):
        g0 = main[(main.read == rd) & (main.tag == "RSG-L18-s0")]
        for sname in ("static", "launch", "braking", "turn", "cruise", "other"):
            ids = set(g0[g0.stratum == sname].scene)
            cells = []
            for t in tags:
                g = main[(main.read == rd) & (main.tag == t) & main.scene.isin(ids)]
                cells.append("-" if g.empty else str(int(g.zero.sum())))
            L.append(f"| {rd} | {sname} | {len(ids)} | " + " | ".join(cells) + " |")
    L.append("")

    # 5. speed relative to the human (route-fail zeros excluded: no rollout, both arcs 0)
    S5 = main[main.type != "route-fail"]
    SP = ["RSG-L18-s0", "NR1F-L18-s0", "NR1F-L18-s1", "NR1FB10-L18-s0", "NR1FB10-L18-s1", "RTB1ARO-L18-s0", "P0"]
    L += ["## 5. Speed relative to the human on scenes that do not score 1.0", "",
          "dv = (ego arc - human arc) / T in m/s (negative = slower than the human); ratio = ego arc / human arc (human arc > 5 m). Per model over its own "
          "non-1.0 scenes, split by zero / partial; quantiles q10 / q25 / median / q75 / q90. Route-fail zeros (no rollout) are left out. "
          "Selection caveat: a scene without a contact scores below 1.0 only through progress, so partial scenes are slower than the human by construction; "
          "the paired rows against RSG (union of both models' non-1.0 scenes) are the comparison.", ""]
    q = lambda x: " / ".join(f"{v:+.2f}" for v in np.nanquantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])) if len(x) else "-"  # noqa: E731
    qr = lambda x: " / ".join(f"{v:.2f}" for v in np.nanquantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])) if np.isfinite(x).sum() else "-"  # noqa: E731
    for rd in ("held", "pub"):
        L += [f"### {rd}", "", "| model | split | scenes | slower / faster than human | dv quantiles (m/s) | ratio quantiles |", "|---|---|--:|--:|---|---|"]
        for t in SP:
            g = S5[(S5.read == rd) & (S5.tag == t) & (S5.score < 1.0)]
            if g.empty:
                continue
            for split, h in (("all < 1.0", g), ("zero", g[g.zero]), ("partial", g[~g.zero])):
                L.append(f"| {LABEL[t]} | {split} | {len(h)} | {int((h.dv < 0).sum())} / {int((h.dv > 0).sum())} | {q(h.dv.values)} | {qr(h.ratio.values)} |")
        L.append("")
        L += [f"By stratum ({rd}, median dv m/s over the model's non-1.0 scenes, n in brackets):", "",
              "| stratum | " + " | ".join(LABEL[t] for t in SP) + " |", "|---|" + "--:|" * len(SP)]
        for sname in ("static", "launch", "braking", "turn", "cruise", "other"):
            cells = []
            for t in SP:
                h = S5[(S5.read == rd) & (S5.tag == t) & (S5.score < 1.0) & (S5.stratum == sname)]
                cells.append("-" if h.empty else f"{h.dv.median():+.2f} ({len(h)})")
            L.append(f"| {sname} | " + " | ".join(cells) + " |")
        L.append("")
        # paired against RSG on scenes where either is below 1.0
        L += [f"Paired against RSG ({rd}): scenes where RSG or the model scores below 1.0; model minus RSG, session-clustered 95 % CI.", "",
              "| model | scenes | dv diff (m/s) | ratio diff | model slower than RSG / faster (by > 0.2 m/s) |", "|---|--:|---|---|--:|"]
        R0 = S5[(S5.read == rd) & (S5.tag == "RSG-L18-s0")].set_index("scene")
        for t in SP[1:]:
            M = S5[(S5.read == rd) & (S5.tag == t)].set_index("scene")
            c = M.index.intersection(R0.index)
            c = c[(M.loc[c, "score"] < 1) | (R0.loc[c, "score"] < 1)]
            if not len(c):
                continue
            grp = np.array(["_".join(s.split("_")[:2]) for s in c])
            d1 = stats.paired(M.loc[c, "dv"].values, R0.loc[c, "dv"].values, groups=grp)
            ok = (M.loc[c, "gt"] > 5).values
            d2 = stats.paired(M.loc[c, "ratio"].values[ok], R0.loc[c, "ratio"].values[ok], groups=grp[ok])
            dd = M.loc[c, "dv"].values - R0.loc[c, "dv"].values
            L.append(f"| {LABEL[t]} | {len(c)} | {stats.fmt(d1)} | {stats.fmt(d2)} | {int((dd < -0.2).sum())} / {int((dd > 0.2).sum())} |")
        L.append("")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"RSG-L18-s0": "#1f5fa8", "NR1F-L18-s0": "#d9822b", "NR1FB10-L18-s0": "#2a9d63", "RTB1ARO-L18-s0": "#8a8a8a", "P0": "#b0457a"}
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    bins = np.linspace(-6, 3, 46)
    for i, rd in enumerate(("held", "pub")):
        for j, split in enumerate(("zero", "partial")):
            ax = axs[i, j]
            for t, c in col.items():
                h = S5[(S5.read == rd) & (S5.tag == t) & (S5.score < 1.0)]
                h = h[h.zero] if split == "zero" else h[~h.zero]
                if len(h):
                    ax.hist(h.dv.clip(-6, 3), bins=bins, histtype="step", lw=1.6, weights=np.full(len(h), 1 / len(h)), color=c, label=f"{LABEL[t]} (n={len(h)}, med {h.dv.median():+.2f})")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{'held 460 @18' if rd == 'held' else 'public 1491 @10'}: {split} scenes", fontsize=10)
            ax.legend(fontsize=7, frameon=False, loc="upper left")
            ax.spines[["top", "right"]].set_visible(False)
            if i == 1:
                ax.set_xlabel("ego minus human mean speed over the window (m/s; clipped to [-6, 3])")
            if j == 0:
                ax.set_ylabel("share of the model's scenes")
    fig.tight_layout()
    fig.savefig(figs / "zo_speed_hist.png", dpi=130)
    L += ["## Figure", "", "![dv histogram](../figs/zo_speed_hist.png)", "",
          "`figs/zo_speed_hist.png`: per model, the histogram of ego-minus-human mean speed over the scenes that do not score 1.0 (rows: held / public; "
          "columns: zero / partial; each curve normalised to its own scene count, n in the legend). What to look at: where the mass sits relative to 0 (left = slower than the human) and whether NR1F / NR1FB10 "
          "shift left of RSG on partial scenes (the longitudinal loss) or on zero scenes (a zero while slow is usually rear-ended or corridor after stalling).", ""]
    (out / "zero_overlap.md").write_text("\n".join(L) + "\n")
    js = {f"{rd}|{t}": dict(zeros=v["zeros"], shared_rsg=v["rsg"], v2_not_rsg=v["v2"], new=v["new"], new_any=v["new_any"], seed_both=v["both"], seed_both_new=v["both_new"])
          for (rd, t), v in ov.items()}
    (out / "zero_overlap.json").write_text(json.dumps(dict(overlap=js, flip=flip), indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("collect")
    p.add_argument("--out", default=str(DATA / "runs/defect_fix/zero/rows.csv.gz"))
    p = sp.add_parser("report")
    p.add_argument("rows"), p.add_argument("--out", default=str(REPO / "experiments/defect_fix/results"))
    a = ap.parse_args()
    dict(collect=collect, report=report)[a.cmd](a)
