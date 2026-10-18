#!/usr/bin/env python3
"""Do the third-party policies read the AlpaSim render's red light? CPU, read-only on stored reads. Box: $DATA_DIR/envs/navsim2/bin/python.

  red_light_readers.py waitset  -> per model and class (red_first, red_queue, other_lead, other_free): variant a wait success / zeros,
                                   variant b launched / n and median launch delay (tables of every wait-set read under reads/*/table)
  red_light_readers.py arena    -> per model mean score / zero rate per census label on the arena windows (held split by default)

Arena labels (census rule of experiments/wait_cap/scripts/nx_census.py `windows`, uniform 0.5 s grid joined on (log, round(t0 / 0.5))):
  red-static   red_start & red_stop & red_cls == C  (human stops at a red light and does not launch in the window)
  red-launch   red_stop & red_cls == G              (human stops at red and launches inside the window)
  nolight-static  static (human maxdisp < 1 m) & no light line ahead (tl False)
  other        everything else (static at a green / unknown light, moving windows, ...)
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

D = os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")
W = f"{D}/runs/alpasim_proxy/wait_set/reads"
A = f"{D}/runs/alpasim_proxy/arena"
CL = ["red_first", "red_queue", "other_lead", "other_free"]


def waitset(tags):
    R = pd.concat([pd.read_csv(f) for t in tags for f in glob.glob(f"{W}/{t}/table/{t}.csv")])
    R["launched"] = R.launch_s.notna()
    out = ["| model | class | n a | wait ok a | zeros a | n b | launched b | median delay b (s) |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for t in tags:
        g = R[R.label == t]
        for c in CL + ["ALL"]:
            h = g if c == "ALL" else g[g.cls == c]
            a, b = h[h.variant == "a"], h[h.variant == "b"]
            d = b.delay_s.dropna()
            out.append(f"| {t} | {c} | {len(a)} | {a.wait_ok.astype(bool).sum()} ({a.wait_ok.astype(bool).mean():.0%}) | {(a.score == 0).sum()} | "
                       f"{len(b)} | {b.launched.sum()} / {len(b)} ({b.launched.mean():.0%}) | {d.median():.2f} |")
    print("\n".join(out))
    # contrast: red_first vs other_free
    out = ["", "| model | wait ok a: red_first | other_free | zeros a: red_first | other_free | launched b: red_first | other_free | median delay b: red_first | other_free |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t in tags:
        g = R[R.label == t]
        r = {}
        for c in ("red_first", "other_free"):
            a, b = g[(g.cls == c) & (g.variant == "a")], g[(g.cls == c) & (g.variant == "b")]
            r[c] = (f"{a.wait_ok.astype(bool).mean():.0%}", f"{(a.score == 0).sum()}/{len(a)}", f"{b.launched.sum()}/{len(b)}", f"{b.delay_s.median():.2f}")
        out.append(f"| {t} | " + " | ".join(r["red_first"][i] + " | " + r["other_free"][i] for i in range(4)) + " |")
    print("\n".join(out))


def arena(split, tags):
    S = pd.read_csv(f"{A}/set/set.csv.gz")
    Wd = pd.read_parquet(f"{D}/runs/wait_cap/census/windows/20261017-142817/windows.parquet")
    Wd = Wd[Wd.kind == "uniform"].copy()
    Wd["k"] = (Wd.t0 / 0.5).round().astype(int)
    S["k"] = (S.t0 / 0.5).round().astype(int)
    M = S.merge(Wd, on=["log", "k"], how="left", suffixes=("", "_c"))
    miss = M.red_stop.isna().sum()
    rs, cl = M.red_stop.fillna(False).astype(bool), M.red_cls.fillna("")
    lab = np.where(rs & (cl == "C") & M.red_start.fillna(False).astype(bool), "red-static",
          np.where(rs & (cl == "G"), "red-launch",
          np.where(M.static.fillna(False).astype(bool) & ~M.tl.fillna(False).astype(bool), "nolight-static", "other")))
    M["label"] = lab
    R = pd.read_csv(f"{A}/report/all2/rows_valid.csv.gz")
    R = R.merge(M[["id", "label"]], on="id", how="left")
    R = R[R.split == split] if split != "all" else R
    base = R[R.tag == "RSG-L18-s0"]
    print(f"unmatched census rows: {miss}; unlabelled arena rows: {R.label.isna().sum()}")
    print("\nwindows per label (RSG rows, valid): " + ", ".join(f"{k} {v}" for k, v in base.label.value_counts().items()))
    LB = ["red-static", "red-launch", "nolight-static", "other"]
    out = ["| model | " + " | ".join(f"{l} mean / zero ({int((base.label == l).sum())})" for l in LB) + " | all mean / zero |", "|---|" + "--:|" * (len(LB) + 1)]
    for t in tags:
        g = R[R.tag == t]
        cells = [f"{h.score.mean():.3f} / {(h.score == 0).mean():.1%}" for h in (g[g.label == l] for l in LB)] + [f"{g.score.mean():.3f} / {(g.score == 0).mean():.1%}"]
        out.append(f"| {t} (n {len(g)}) | " + " | ".join(cells) + " |")
    print("\n".join(out))
    # corridor / front-collision composition of the zeros in the red-light labels
    out = ["", "zero types per label (count of zeros: corridor / at-fault front / offroad):", "", "| model | " + " | ".join(LB) + " |", "|---|" + "--:|" * len(LB)]
    for t in tags:
        g = R[R.tag == t]
        c = []
        for l in LB:
            h = g[(g.label == l) & (g.score == 0)]
            c.append(f"{int(h.corridor.fillna(0).astype(bool).sum())} / {int(h.front.fillna(0).astype(bool).sum())} / {int(h.offroad.fillna(0).astype(bool).sum())}")
        out.append(f"| {t} | " + " | ".join(c) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["waitset", "arena"]); ap.add_argument("--split", default="held")
    ap.add_argument("--tags", nargs="*"); a = ap.parse_args()
    if a.cmd == "waitset":
        waitset(a.tags or sorted(os.path.basename(p) for p in glob.glob(f"{W}/*") if glob.glob(f"{p}/table/*.csv")))
    else:
        arena(a.split, a.tags or ["RSG-L18-s0", "NR1F-L18-s0", "YR10m10-F-s0", "APY10m10-AB-s0", "G0014", "DD-sample", "LTF-sample", "starter"])
