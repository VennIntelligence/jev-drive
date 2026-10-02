"""Selection and write-up of the vlm_thin study (CPU).

  vlm_thin_report.py select --run RUN    grid of (res, feature, N, head) on val / test / CV + bench latency -> grid.csv,
                                         selection.json (the registered rule, plan section 5)
  vlm_thin_report.py report --run RUN    part 1 table, truncation tables and figure, the two Phase A tables -> results/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_thin_common import (ANS3, RED, CUTS, FOUR_PROMPT, LIGHT3, LIGHTS, RES, light_metrics, log, percentiles, cellp, jwrite)  # noqa: E402
from vlm_arb_phase_a import LINES, evaluate, table  # noqa: E402

LAT_LINE_MS = LINES["latency_p95_ms"]               # 600, registered
TOL = 0.03                                          # registered tolerance on the validation score S
KEYS = ("red_recall", "red_as_green", "green_recall", "nolight_fp", "other_fp")
ANS = np.array([ANS3[c] for c in LIGHT3])


def frames(run):
    return pd.read_csv(Path(run) / "frames.csv", dtype={"route": str})


def bench(run):
    out = {}
    for p in sorted((Path(run) / "bench").glob("*.json")):
        d = json.loads(p.read_text())
        out[d["cfg"]] = d
    return out


def flat(R, prefix=""):
    d = {prefix + "S": R["S"]}
    for k in KEYS:
        r = R[k]
        d.update({prefix + k: r["est"], prefix + k + "_lo": r["lo"], prefix + k + "_hi": r["hi"], prefix + k + "_n": r["n"]})
    return d


def grid(run):
    df, B = frames(run), bench(run)
    part = df.part.to_numpy()
    rows = []
    for res in RES:
        p = Path(run) / "fit" / (res + ".npz")
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=False)
        info = json.loads((Path(run) / "fit" / (res + ".json")).read_text())
        lat = lambda cfg: ({k: B[cfg][k] for k in ("p50", "p95", "p99")} if cfg in B and "p50" in B[cfg] else {})   # noqa: E731
        variants = [("zs", "-", 36, None, np.array(LIGHTS)[z["zs_pred"]], None, "fwd_" + res)]
        for key in z.files:
            if key.startswith("pred|"):
                _, feat, N, head = key.split("|")
                cv = z["cv|" + key[5:]]
                lam = info["%s|%s|%s" % (feat, N, head)].get("lam")
                cfg = "vis_" + res if feat == "vis" else "cut_%s_%s" % (res, N)
                variants.append((feat, head, int(N), lam, ANS[z[key]], ANS[cv], cfg))
        for feat, head, N, lam, ans, cv, cfg in variants:
            base = dict(res=res, tokens=int(res[1:]), feat=feat, N=N, head=head, lam=lam, **{"lat_" + k: v for k, v in lat(cfg).items()})
            for pt in ("val", "test"):
                m = part == pt
                rows.append(dict(base, part=pt, **flat(light_metrics(df[m], ans[m]))))
            if cv is not None:
                rows.append(dict(base, part="cv", **flat(light_metrics(df, cv))))
            else:
                rows.append(dict(base, part="all", **flat(light_metrics(df, ans))))
        log("grid %s done" % res)
    return pd.DataFrame(rows)


def select(a):
    run = Path(a.run)
    g = grid(run)
    g.to_csv(run / "grid.csv", index=False)
    v = g[(g.part == "val") & g.lat_p95.notna() & (g.lat_p95 <= LAT_LINE_MS) & g.S.notna()]
    out = dict(rule="max validation S among variants with measured p95 <= %g ms; within %g of the max, the lowest p95" % (LAT_LINE_MS, TOL))
    if len(v):
        top = v[v.S >= v.S.max() - TOL].sort_values(["lat_p95", "S"], ascending=[True, False])
        c = top.iloc[0]
        out["chosen"] = dict(res=c.res, feat=c.feat, N=int(c.N), head=c["head"], lam=None if pd.isna(c.lam) else float(c.lam),
                             val_S=float(c.S), p50_ms=float(c.lat_p50), p95_ms=float(c.lat_p95))
        out["within_tolerance"] = top[["res", "feat", "N", "head", "S", "lat_p50", "lat_p95"]].head(15).to_dict("records")
    else:
        out["chosen"] = None
    # smallest N within TOL of the best N, per (res, feat, head) of the heads
    fam = []
    for (res, feat, head), gg in g[(g.part == "val") & (g["head"].isin(["lin", "mlp"]))].groupby(["res", "feat", "head"]):
        best = gg.S.max()
        r = gg[gg.S >= best - TOL].sort_values("N").iloc[0]
        fam.append(dict(res=res, feat=feat, head=head, best_val_S=float(best), N_star=int(r.N), val_S_at_N_star=float(r.S),
                        p95_ms=None if pd.isna(r.lat_p95) else float(r.lat_p95)))
    out["N_star_per_family"] = fam
    jwrite(run / "selection.json", out)
    log("selection: %s" % json.dumps(out.get("chosen")))


# ---------------------------------------------------------------------------------------------------- report
def pct(r):
    return "n/a" if not np.isfinite(r) else "%.0f%%" % (100 * r)


def md_table(df):
    return df.to_markdown(index=False)


def part1_table(run, df, B):
    """Training-free variants on the 233 sweep frames, read with the sweep's own labels (so the generate row reproduces the
    earlier sweep row exactly) + the bench latency."""
    from vlm_arb_common import boot_ratio
    from vlm_thin_common import SWEEP_FRAMES
    sw = pd.read_csv(SWEEP_FRAMES, dtype={"route": str})
    A = {}
    for p in sorted((Path(run) / "baseline").glob("shard-*.jsonl")):
        for l in open(p):
            d = json.loads(l)
            A.setdefault(d["cfg"], {})[d["id"]] = d["ans"]
    names = {"gen_ref": "generate, HF reference path (PIL + CPU processor), full resolution",
             "gen_gpu": "generate, GPU preprocessing, full resolution"}

    def cell(hit, den):
        r = boot_ratio((hit & den).astype(float), den.astype(float), sw.route)
        return "%.0f%% [%.0f, %.0f] %d/%d" % (100 * r["est"], 100 * r["lo"], 100 * r["hi"], int((hit & den).sum()), int(den.sum()))
    rows = []
    for cfg in ["gen_ref", "gen_gpu"] + ["fwd_" + r for r in RES] + ["fwd_r4573_compile"]:
        src = A.get("gen_ref" if cfg == "gen_gpu" else "fwd_r4573" if cfg.endswith("_compile") else cfg)
        b = B.get(cfg, {})
        row = {"variant": names.get(cfg, ("one forward pass, option scoring, %s tokens" % cfg.split("_")[1][1:]) + (", torch.compile" if cfg.endswith("_compile") else ""))}
        if src:
            a = pd.Series([src.get(i, "") for i in sw.id])
            t = sw.truth
            row.update({"ego red: red": cell(a == RED, t == "red"), "ego red: green": cell(a == "green_for_ego", t == "red"),
                        "ego green: green": cell(a == "green_for_ego", t == "green"), "no light: red": cell(a == RED, t == "none")})
        row["latency p50 / p95 ms"] = ("%.0f / %.0f" % (b["p50"], b["p95"])) if "p50" in b else ("failed" if b.get("error") else "n/a")
        row["note"] = "accuracy of the variant it shares math with (not re-run)" if cfg.endswith("_compile") or cfg == "gen_gpu" else ""
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_tables(g, B, res_list):
    t = g[g.part == "test"]
    out = {}
    for res in res_list:
        rows = []
        for feat in ("ans", "pool"):
            for N in CUTS:
                for head in ("lin", "mlp"):
                    r = t[(t.res == res) & (t.feat == feat) & (t.N == N) & (t["head"] == head)]
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    v = g[(g.part == "val") & (g.res == res) & (g.feat == feat) & (g.N == N) & (g["head"] == head)].iloc[0]
                    c = g[(g.part == "cv") & (g.res == res) & (g.feat == feat) & (g.N == N) & (g["head"] == head)].iloc[0]
                    rows.append({"feature": feat, "N": N, "head": head, "val S": "%.2f" % v.S, "test red": pct(r.red_recall),
                                 "test red->green": pct(r.red_as_green), "test green": pct(r.green_recall),
                                 "test no-light red": pct(r.nolight_fp), "test other-dir red": pct(r.other_fp),
                                 "CV S": "%.2f" % c.S, "latency p50 / p95 ms": "%.0f / %.0f" % (r.lat_p50, r.lat_p95)})
        vis = t[(t.res == res) & (t.feat == "vis")]
        for _, r in vis.iterrows():
            v = g[(g.part == "val") & (g.res == res) & (g.feat == "vis") & (g["head"] == r["head"])].iloc[0]
            c = g[(g.part == "cv") & (g.res == res) & (g.feat == "vis") & (g["head"] == r["head"])].iloc[0]
            rows.append({"feature": "vision tower only", "N": 0, "head": r["head"], "val S": "%.2f" % v.S, "test red": pct(r.red_recall),
                         "test red->green": pct(r.red_as_green), "test green": pct(r.green_recall), "test no-light red": pct(r.nolight_fp),
                         "test other-dir red": pct(r.other_fp), "CV S": "%.2f" % c.S, "latency p50 / p95 ms": "%.0f / %.0f" % (r.lat_p50, r.lat_p95)})
        z = t[(t.res == res) & (t.feat == "zs")]
        if len(z):
            r = z.iloc[0]
            v = g[(g.part == "val") & (g.res == res) & (g.feat == "zs")].iloc[0]
            c = g[(g.part == "cv") & (g.res == res) & (g.feat == "zs")]
            rows.append({"feature": "zero-shot, one pass, option scoring (no head)", "N": 36, "head": "-", "val S": "%.2f" % v.S, "test red": pct(r.red_recall),
                         "test red->green": pct(r.red_as_green), "test green": pct(r.green_recall), "test no-light red": pct(r.nolight_fp),
                         "test other-dir red": pct(r.other_fp), "CV S": "", "latency p50 / p95 ms": "%.0f / %.0f" % (r.lat_p50, r.lat_p95)})
        out[res] = pd.DataFrame(rows)
    return out


def figure(g, B, path, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = g[(g.part == "test") & (g.feat == "ans") & (g["head"] == "lin")]
    c = g[(g.part == "cv") & (g.feat == "ans") & (g["head"] == "lin")]
    col = {"r4573": "#1f3b73", "r2335": "#3b6ea5", "r1153": "#d98c1f", "r559": "#a33b2d"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    for k, (d, title) in enumerate(((t, "held-out test routes"), (c, "7-fold route-grouped CV, all 21 routes"))):
        for res in RES:
            r = d[d.res == res].sort_values("N")
            if len(r):
                ax[k].plot(r.lat_p50, r.S, "-o", ms=3, color=col[res], label="%s tokens, linear head at layer N" % res[1:])
                for _, q in r.iterrows():
                    if q.N in (4, 12, 20, 36):
                        ax[k].annotate(str(int(q.N)), (q.lat_p50, q.S), fontsize=6, xytext=(2, 3), textcoords="offset points")
        z = g[(g.part == ("test" if k == 0 else "all")) & (g.feat == "zs")]
        ax[k].scatter(z.lat_p50, z.S, marker="s", color=[col[r] for r in z.res], s=22, label="zero-shot one pass")
        ax[k].axvline(LAT_LINE_MS, color="#222", lw=1, ls="--")
        ax[k].set_title(title, fontsize=9, loc="left")
        ax[k].set_xlabel("latency p50, ms (batch 1, quiet card; dashed: registered p95 line)", fontsize=8)
        ax[k].set_xscale("log")
        ax[k].grid(color="#eee", lw=.6)
    ax[0].set_ylabel("S = red recall - red answered green + green recall - no-light false alarm", fontsize=7)
    ax[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)


def side_by_side(df, pa, pb, g, ch):
    """The light readouts on the test frames, zero-shot and head-fitted rows next to each other (not the same kind of evidence:
    the head rows are trained on CARLA frames of other routes)."""
    t = df[df.part == "test"].reset_index(drop=True)
    rows = []

    def add(name, kind, R, lat):
        rows.append({"variant": name, "evidence": kind, "ego red: red": cellp(R["red_recall"]), "ego red: green": cellp(R["red_as_green"]),
                     "ego green: green": cellp(R["green_recall"]), "no light: red": cellp(R["nolight_fp"]), "other dir red: red": cellp(R["other_fp"]),
                     "latency p50 / p95 ms": lat})
    add("openjev, as logged in the closed loop (System One endpoint)", "zero-shot, other model", light_metrics(t, t.oj_light.fillna("")),
        "%.0f / %.0f (logged, under load)" % (np.nanpercentile(t.oj_lat, 50), np.nanpercentile(t.oj_lat, 95)))
    if pa is not None:
        add("(a) Qwen3-VL-4B zero-shot, generate", "zero-shot", light_metrics(pa, pa.a_light), "%.0f / %.0f" % (np.percentile(pa.lat, 50), np.percentile(pa.lat, 95)))
    for res in RES:
        r = g[(g.part == "test") & (g.feat == "zs") & (g.res == res)]
        if len(r):
            r = r.iloc[0]
            f = lambda k: "%.0f%% [%.0f, %.0f] n=%d" % (100 * r[k], 100 * r[k + "_lo"], 100 * r[k + "_hi"], r[k + "_n"])   # noqa: E731
            rows.append({"variant": "Qwen3-VL-4B zero-shot, one pass, %s tokens" % res[1:], "evidence": "zero-shot", "ego red: red": f("red_recall"),
                         "ego red: green": f("red_as_green"), "ego green: green": f("green_recall"), "no light: red": f("nolight_fp"),
                         "other dir red: red": f("other_fp"), "latency p50 / p95 ms": "%.0f / %.0f" % (r.lat_p50, r.lat_p95)})
    if pb is not None:
        add("(b) chosen: %s" % (json.dumps({k: ch[k] for k in ("res", "feat", "N", "head")})), "head fitted on CARLA train routes" if ch["feat"] != "zs" else "zero-shot",
            light_metrics(pb, pb.a_light), "%.0f / %.0f" % (np.percentile(pb.lat, 50), np.percentile(pb.lat, 95)))
    return pd.DataFrame(rows)


def phase_a_df(run, df):
    """Test frames with the answers of (a) zero-shot generate and (b) the chosen variant; both keep the zero-shot joint
    prompt's sign / block / side answers."""
    G = {}
    for p in sorted((Path(run) / "phase_a").glob("gen-*.jsonl")):
        for l in open(p):
            d = json.loads(l)
            G[d["id"]] = d
    S = {}
    for p in sorted((Path(run) / "phase_a").glob("serve-*.jsonl")):
        for l in open(p):
            d = json.loads(l)
            S[d["id"]] = d
    t = df[df.part == "test"].reset_index(drop=True)
    t = t[t.id.isin(G)].reset_index(drop=True)
    j = lambda i, q: (G[i]["joint"].get(q) or "")   # noqa: E731
    base = t.assign(ok=True, a_sign=[j(i, "Q_sign") for i in t.id], a_block=[j(i, "Q_block") for i in t.id],
                    a_side=[j(i, "Q_side") for i in t.id])
    a = base.assign(a_light=[G[i]["light"] for i in t.id], lat=[G[i]["light_ms"] for i in t.id])
    b = None
    if all(i in S for i in t.id):
        b = base.assign(a_light=[S[i]["ans"] for i in t.id], lat=[S[i]["ms"] for i in t.id])
    jl = percentiles([G[i]["joint_ms"] for i in t.id])
    return a, b, jl, G, S


def report(a):
    run = Path(a.run)
    df, B = frames(run), bench(run)
    g = pd.read_csv(run / "grid.csv")
    sel = json.loads((run / "selection.json").read_text())
    ch = sel["chosen"]
    res_out = run / "results"
    res_out.mkdir(exist_ok=True)
    L = ["# vlm_thin: Qwen3-VL-4B light reading with a cut language model and a thin head", ""]
    p1 = part1_table(run, df, B)
    L += ["## 1. Training-free speed levers (full model, no training)", "",
          "Accuracy on the 233 sweep frames with the sweep's own labels (85 ego red, 68 ego green, 80 no light; cell = estimate "
          "[95% route-cluster CI] hits/frames); latency from the bench (100 test frames, batch 1, quiet card).", "", md_table(p1), ""]
    for res, t in sweep_tables(g, B, list(RES)).items():
        L += ["## 2. Truncation sweep at %s visual tokens (two cameras)" % res[1:], "", md_table(t), ""]
    figure(g, B, res_out / "thin_sweep.png", ch)
    pa, pb, jl, G, S = phase_a_df(run, df)
    out = {}
    for name, d in (("a", pa), ("b", pb)):
        if d is None:
            continue
        R, gate = evaluate(d.reset_index(drop=True))
        txt = table(R, gate, d)
        out[name] = dict(readouts=R, gate=gate)
        L += ["## 3%s. Phase A on the test routes: %s" % (name, "(a) full zero-shot Qwen3-VL-4B, generate" if name == "a" else
                                                          "(b) chosen fast variant: %s" % json.dumps(ch)), "", txt, ""]
    L += ["## 4. Light readouts on the test routes side by side", "",
          "Cells: estimate [95% route-cluster CI] n = frames in the denominator. Zero-shot rows have seen no CARLA frame; the head row was "
          "fitted on CARLA frames of other routes (in-domain supervised), so it is a different kind of evidence.", "",
          md_table(side_by_side(df, pa, pb, g, ch)), ""]
    L += ["Joint-prompt latency (the sign / block / side answers of both tables): p50 %.0f ms, p95 %.0f ms." % (jl["p50"], jl["p95"]), ""]
    (res_out / "vlm_thin.md").write_text("\n".join(L) + "\n")
    jwrite(res_out / "vlm_thin_phase_a.json", out)
    g.to_csv(res_out / "vlm_thin_grid.csv", index=False)
    log("report written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "report"])
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    {"select": select, "report": report}[a.cmd](a)
