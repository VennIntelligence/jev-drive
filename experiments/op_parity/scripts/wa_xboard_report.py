"""wa-xboard read-out (plans/2026-10-08-wa-xboard-prereg.md): WA-JEPA zero-shot on WOD-E2E val against shipped Cinque, WP2, WLG, P2H10, the log. jevdrive env, CPU.

Conventions of wod_parity / wod_launch (wod_launch_report.Ctx): 479 rater frames, cluster-mean RFS, ADE@3s / @5s on the 1 437 rater + extra frames against the log,
paired bootstrap over sequences (B 4 000, seed 0). WA-JEPA plans: 4 s -> 5 s by xcv (primary) or xca; the `-x4` reference rows get the same operator applied.
  python wa_xboard_report.py [--arms V1 V2 V3 IMG0 STATE0] -> results/wa_xboard/{arms,paired,strata,clusters,sens}.csv + tables.md
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wod_launch_report import B, Ctx, f3, xca, xcv  # noqa: E402

OUT = _R / "experiments/op_parity/results/wa_xboard"
REFS = {"shipped": ["shipped"], "WP2": ["WP2-full-s0", "WP2-full-s1"], "WLG": ["WLG-full-s0", "WLG-full-s1"], "P2H10": ["P2H10-F-s0", "P2H10-F-s1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["V1", "V2", "V3", "IMG0", "STATE0"])
    ap.add_argument("--extra", nargs="*", default=["V1n2"], help="sensitivity arms (not in the main tables)")
    a = ap.parse_args()
    C = Ctx()
    n, N = C.n, len(C.names)
    OUT.mkdir(parents=True, exist_ok=True)
    P = {}                                                                           # name -> (1437, 20, 2) plan; seed groups = mean of the per-frame metrics
    for k, tags in REFS.items():
        P[k] = [C.preds(t, all_frames=True) for t in tags]
    P["log"] = [C.fut.copy()]
    for k in a.arms + a.extra:
        P[f"WA-{k}"] = [C.preds(f"wa-{k}", all_frames=True)]
        P[f"WA-{k}-xca"] = [C.preds(f"wa-{k}-xca", all_frames=True)]
    for k in ("shipped", "WP2", "WLG", "P2H10", "log"):                               # the same 4 s truncation + xcv as WA-JEPA gets
        P[f"{k}-x4"] = [xcv(p) for p in P[k]]
    rfs, ade3, ade5 = {}, {}, {}
    for k, ps in P.items():
        rfs[k] = np.mean([C.rfs(p) for p in ps], 0)
        e = [np.linalg.norm(p - C.fut, axis=-1) for p in ps]
        ade3[k], ade5[k] = np.mean([x[:, :12].mean(1) for x in e], 0), np.mean([x.mean(1) for x in e], 0)

    def ci_all(d, rows=None):                                                         # mean over the 1 437 frames, bootstrap over their sequences
        m = np.ones(N) if rows is None else np.asarray(rows, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (C.Ka @ (m * d)) / (C.Ka @ m)
        return (float((m * d).sum() / m.sum()), *np.nanpercentile(b, [2.5, 97.5]))

    # ---- arms
    rows = []
    for k in P:
        rows.append({"arm": k, "RFS": C.cm(rfs[k]), "ADE@3s": ade3[k].mean(), "ADE@5s": ade5[k].mean()})
    mv = np.linalg.norm(C.fut[:, -1], axis=-1) > 2.0
    for r in rows:
        p = P[r["arm"]][0]
        r["5 s displacement / log (median, log > 2 m)"] = float(np.median(np.linalg.norm(p[:, -1], axis=-1)[mv] / np.linalg.norm(C.fut[:, -1], axis=-1)[mv]))
    arms = pd.DataFrame(rows)
    arms.to_csv(OUT / "arms.csv", index=False)

    # ---- paired
    main = [f"WA-{k}" for k in a.arms]
    refs = ["shipped", "WP2", "WLG", "P2H10", "log", "shipped-x4", "WP2-x4", "WLG-x4"]
    pr = []
    for w in main + [f"WA-{k}-xca" for k in a.arms] + [f"WA-{k}" for k in a.extra]:
        for r in refs:
            d = rfs[w] - rfs[r]
            row = {"A": w, "B": r, "dRFS": f3(C.ci(d)), "dADE@3s": f3(ci_all(ade3[w] - ade3[r])), "dADE@5s": f3(ci_all(ade5[w] - ade5[r]))}
            lo, hi = C.ci(d)[1:]
            row["RFS verdict"] = "wins" if lo > 0 else "loses" if hi < 0 else "n.s."
            pr.append(row)
    for w, r in (("WA-V1", "WA-IMG0"), ("WA-V2", "WA-IMG0"), ("WA-V3", "WA-IMG0"), ("WA-V1", "WA-STATE0"), ("WA-V1", "WA-V2"), ("WA-V1", "WA-V3"), ("WA-V1", "WA-V1n2")):
        if w in rfs and r in rfs:
            d = rfs[w] - rfs[r]
            lo, hi = C.ci(d)[1:]
            pr.append({"A": w, "B": r, "dRFS": f3(C.ci(d)), "dADE@3s": f3(ci_all(ade3[w] - ade3[r])), "dADE@5s": f3(ci_all(ade5[w] - ade5[r])),
                       "RFS verdict": "wins" if lo > 0 else "loses" if hi < 0 else "n.s."})
    paired = pd.DataFrame(pr)
    paired.to_csv(OUT / "paired.csv", index=False)

    # ---- strata (rater frames; ADE strata on the same 479 for the rows they share)
    st = dict(C.st)
    st["intent straight"], st["intent turn (left + right)"] = C.intent == 1, np.isin(C.intent, (2, 3))
    cols = main + ["shipped", "WP2", "WLG", "P2H10", "log"]
    sr = []
    for nm, m in st.items():
        if m.sum() == 0:
            continue
        row = {"stratum": nm, "n": int(m.sum())}
        for k in cols:
            row[k] = C.cm(rfs[k], m)
        for w in main[:3]:
            row[f"{w} - shipped"] = f3(C.ci(rfs[w] - rfs["shipped"], m))
            row[f"{w} - WLG"] = f3(C.ci(rfs[w] - rfs["WLG"], m))
        sr.append(row)
    pd.DataFrame(sr).to_csv(OUT / "strata.csv", index=False)
    clusters = pd.factorize(pd.Series(C.Z.load_sets()["rater"]["cluster"].astype(str)))
    cr = []
    for j, cn in enumerate(clusters[1]):
        m = clusters[0] == j
        row = {"cluster": cn, "n": int(m.sum())}
        for k in cols:
            row[k] = float(rfs[k][m].mean())
        row["WA-V1 - shipped"] = f3(C.ci(rfs["WA-V1"] - rfs["shipped"], m)) if "WA-V1" in rfs else ""
        cr.append(row)
    pd.DataFrame(cr).sort_values("cluster").to_csv(OUT / "clusters.csv", index=False)

    with open(OUT / "tables.md", "w") as f:
        for nm, df in (("arms", arms), ("paired", paired), ("strata", pd.DataFrame(sr)), ("clusters", pd.DataFrame(cr).sort_values("cluster"))):
            f.write(f"## {nm}\n\n{df.to_markdown(index=False, floatfmt='.3f')}\n\n")
    print(open(OUT / "tables.md").read())


if __name__ == "__main__":
    main()
