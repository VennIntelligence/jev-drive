"""JT, the joint L18 arm of NR1 prereg supplement 9 (plans/2026-10-17-nr1-prereg.md): one stage from RB0-F-s0 that trains navtrain + yaw-rate
imitation, the big on-policy pool (S hinges, lateral 0.5v2, the two-sided arc term) and trainval imitation rows (reasoned standing / launch
at a dose) in the same batches. Box, repo root; chain in jt.sh.

  weights --pool on1_JT --k-s 32 --k-tv 38 --hold-pct 15 --tv D10 [--no-wait] --out NAME   (op-train, CPU) per-row draw weights of the mixed
      pool for rm_train --on-wfile -> runs/defect_fix/w/NAME.npz (+ grp: S / S_held / the trainval groups of df.py weights) and
      results/jt/w_NAME.json. S rows (`mode` self): mass k_s / (k_s + k_tv), uniform over the rows of the logs not held out
      (crc32(log) % 100 >= hold-pct; rm_train --on-hold-pct applies the same rule and then finds them at weight 0); trainval rows (`mode`
      both): mass k_tv / (k_s + k_tv) split as df.py's file --tv (names on:<token> -> on:tv:<token>); --no-wait: the standing / launch
      groups at 0, the rest renormalised (= df.py weights --wait 0 --launch 0).
  arena --tags T@V .. --out DIR   (alpasim venv, CPU) the arena held-split readout of JT's reads (runs/defect_fix/jt/arena/<tag>@<off|leash3>)
      against the stored RSG-L18-s0 reads (no hold: the arena's reads; leash3: ARENA-HOLD's): per stratum, zero types, partial types, paired
      CIs clustered by log -> DIR/arena.md + rows.csv.gz
  held --tags T ..   (op-train) held 460 @18 against RSG-L18-s0 (df.py cl_read) -> stdout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = DATA / "runs/op_parity/cache"
OUT = DATA / "runs/defect_fix"
RES = REPO / "experiments/defect_fix/results/jt"
ARENA = DATA / "runs/alpasim_proxy/arena"


def held_logs(logs, pct):
    return {g for g in set(logs) if zlib.crc32(g.encode()) % 100 < pct}


def cmd_weights(a):
    tb = np.load(CR / a.pool / "tab.npz")
    names, mode, logs = tb["names"].astype(str), tb["mode"].astype(str), tb["log"].astype(str)
    S = mode == "self"
    hl = held_logs(logs[S], a.hold_pct)
    held = S & np.isin(logs, sorted(hl))
    w = np.zeros(len(names))
    grp = np.where(held, "S_held", np.where(S, "S", "")).astype("U12")
    fs = a.k_s / (a.k_s + a.k_tv)
    w[S & ~held] = fs / (S & ~held).sum()
    tv = np.flatnonzero(~S)
    z = np.load(OUT / "w" / f"{a.tv}.npz")
    pos = {t: i for i, t in enumerate(z["names"].astype(str).tolist())}
    j = np.array([pos["on:" + n.split(":", 2)[2]] for n in names[tv]])
    wt, gt = z["w"][j].astype(np.float64), z["grp"].astype(str)[j]
    assert len(set(j.tolist())) == len(j) == len(z["names"]), "trainval rows != the wfile's rows"
    if a.no_wait:
        wt = np.where(np.isin(gt, ["stand_r", "stand_u", "launch"]), 0.0, wt)
    w[tv] = (1.0 - fs) * wt / wt.sum()
    grp[tv] = gt
    w = w / w.sum()
    k = a.k_s + a.k_tv
    rec = dict(pool=a.pool, rows=len(names), s_rows=int(S.sum()), tv_rows=int((~S).sum()), k_s=a.k_s, k_tv=a.k_tv, on_k=k, hold_pct=a.hold_pct,
               held_logs=sorted(hl), s_logs=len(set(logs[S])), held_rows=int(held.sum()), tv=a.tv, no_wait=a.no_wait,
               share={g: float(w[grp == g].sum()) for g in np.unique(grp)}, per_batch={g: float(w[grp == g].sum() * k) for g in np.unique(grp)},
               rows_by_group={g: int((grp == g).sum()) for g in np.unique(grp)},
               draws_per_row_10k={g: float(w[grp == g].sum() * k * 1e4 / max((grp == g).sum(), 1)) for g in np.unique(grp)},
               s_parts={p: int((S & (tb["pool"].astype(str) == p)).sum()) for p in np.unique(tb["pool"].astype(str)[S])})
    (OUT / "w").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "w" / f"{a.out}.npz", names=tb["names"], w=w, grp=grp)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"w_{a.out}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_arena(a):
    import pandas as pd
    sys.path[:0] = [str(REPO), str(REPO / "experiments/alpasim_proxy/scripts")]
    import arena as A
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.system(f"{sys.executable} {REPO}/experiments/alpasim_proxy/scripts/arena.py collect --reads {a.reads} --out {out}/rows.csv.gz > /dev/null")
    Sx = pd.read_csv(ARENA / "set/set.csv.gz").set_index("id")
    Hs = Sx[Sx.split == "held"]
    R = pd.read_csv(out / "rows.csv.gz")
    ref = {"off": pd.read_csv(a.ref_off), "leash3": pd.read_csv(a.ref_leash3)}
    ref["off"] = ref["off"][ref["off"].tag == "RSG-L18-s0"]
    ref["leash3"] = ref["leash3"][ref["leash3"].tag == "leash3"]
    ref = {k: v[v.id.isin(Hs.index)].drop_duplicates("id").set_index("id") for k, v in ref.items()}
    L = [f"# JT arena held-split readout ({len(Hs)} windows)", "",
         "Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3: ARENA-HOLD's stored RSG + leash3 read) "
         "on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). "
         "Contribution = the stratum's summed paired difference / all common windows.", "",
         "| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    per, typ = [], []
    for tv in a.tags:
        tag, v = tv.split("@")
        X = R[(R.tag == tv) & R.id.isin(Hs.index)].drop_duplicates("id").set_index("id")
        B = ref[v]
        if not len(X):
            L.append(f"| {tv} | 0 | - | - | - | - | - | - |")
            continue
        bad = set(B.index[B.type == "route-fail"]) | set(X.index[X.type == "route-fail"])
        c = X.index.intersection(B.index).difference(bad)
        x, y, g = X.loc[c, "score"].values, B.loc[c, "score"].values, Hs.loc[c, "log"].values
        d, dz = stats.paired(x, y, groups=g), stats.paired((x == 0).astype(float), (y == 0).astype(float), groups=g)
        L.append(f"| {tv} | {len(c)} | {x.mean():.4f} | {y.mean():.4f} | {stats.fmt(d, '+.4f')} | {(x == 0).mean():.3f} | {(y == 0).mean():.3f} | "
                 f"{stats.fmt(dz, '+.4f')} |")
        for s in A.STRATA:
            cs = c[Hs.loc[c, "stratum"].values == s]
            if not len(cs):
                continue
            xs, ys = X.loc[cs, "score"].values, B.loc[cs, "score"].values
            ds = stats.paired(xs, ys, groups=Hs.loc[cs, "log"].values) if len(cs) >= 20 else None
            per.append(f"| {tv} | {s} | {len(cs)} | {ys.mean():.3f} / {(ys == 0).mean():.3f} | {xs.mean():.3f} / {(xs == 0).mean():.3f} | "
                       f"{'-' if ds is None else stats.fmt(ds, '+.3f')} | {(xs - ys).sum() / len(c):+.4f} | "
                       f"{int(((ys > 0) & (xs == 0)).sum())} / {int(((ys == 0) & (xs > 0)).sum())} |")
        for nm, Q in ((f"RSG @ {v}", B.loc[c]), (tv, X.loc[c])):
            zc = Q[Q.score == 0].type.map(A.zclass).value_counts()
            pc = Q[Q.score > 0].type.value_counts()
            typ.append(f"| {nm} | " + " | ".join(str(int(zc.get(k, 0))) for k in A.ZC) + " | " + " | ".join(str(int(pc.get(k, 0))) for k in A.PC) + " |")
    L += ["", "By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):", "",
          "| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |", "|---|---|--:|--:|--:|--:|--:|--:|", *per,
          "", "Zero types and partial-score types on the common windows:", "",
          "| read | " + " | ".join(A.ZC) + " | " + " | ".join(A.PC) + " |", "|---|" + "--:|" * (len(A.ZC) + len(A.PC)), *typ, ""]
    (out / "arena.md").write_text("\n".join(L))
    print("\n".join(L))


def cmd_held(a):
    sys.path[:0] = [str(HERE)]
    import df
    for t in a.tags:
        r = df.cl_read(t, ["RSG-L18-s0"])
        print(json.dumps({k: v for k, v in r.items() if k != "held"} | {"held": {q: r["held"][q] for q in r["held"] if q != "pair"} if r["held"] else None,
                          "pair": r["held"]["pair"] if r["held"] else None}, default=str))
        print("\n".join(df.fmt_cl(t, r, ["RSG-L18-s0"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("weights")
    p.add_argument("--pool", default="on1_JT"), p.add_argument("--k-s", type=int, default=32), p.add_argument("--k-tv", type=int, default=38)
    p.add_argument("--hold-pct", type=int, default=15), p.add_argument("--tv", default="D10"), p.add_argument("--no-wait", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("arena")
    p.add_argument("--tags", nargs="+", required=True), p.add_argument("--out", required=True)
    p.add_argument("--reads", default=str(OUT / "jt/arena"))
    p.add_argument("--ref-off", default=str(ARENA / "hold/report/ref_rows.csv.gz"), help="the arena's stored reads (RSG-L18-s0 rows)")
    p.add_argument("--ref-leash3", default=str(ARENA / "hold/report/rows.csv.gz"), help="ARENA-HOLD's reads (tag leash3 = RSG + leash3)")
    p = sub.add_parser("held")
    p.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    dict(weights=cmd_weights, arena=cmd_arena, held=cmd_held)[a.cmd](a)
