"""op_parity turn selector bench: identity gate (G-id) and the pooled report of the bench runs (plans/2026-10-08-turn-selector-bench-prereg.md).

  python turn_selbench_report.py idcheck --subset navsim/op-parity-tsbench-smoke
  python turn_selbench.py report                       (calls report(a) below)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

TERMS = ["score", "NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]


def units(bench, key):
    import pandas as pd
    from jevdrive.bench.runner import bench_root
    return pd.read_csv(bench_root(bench, key) / "units.csv").set_index("token")


def cmd_idcheck(a):
    """G-id: ts0 on the smoke logs equals the archived SH30-F-s0 / s1 bench scores token by token (max abs diff <= 1e-9, NaN pattern equal)."""
    from jevdrive.bench.models import data_dir
    tag = a.subset.replace("/", "-")
    res = {}
    for s in (0, 1):
        new, old = units("navtest", f"SH30-F-s{s}@warp_ts0_{tag}"), units("navtest", f"SH30-F-s{s}@warp")
        assert len(new) > 300, f"smoke run has {len(new)} tokens"
        old = old.loc[new.index]
        d = [float(np.nanmax(np.abs(new[c].to_numpy(float) - old[c].to_numpy(float)))) for c in TERMS]
        nan_ok = all((np.isnan(new[c].to_numpy(float)) == np.isnan(old[c].to_numpy(float))).all() for c in TERMS)
        res[f"s{s}"] = dict(n=int(len(new)), max_abs_diff=max(d), nan_pattern_equal=bool(nan_ok), per_term=dict(zip(TERMS, d)))
    res["ok"] = bool(all(v["max_abs_diff"] <= 1e-9 and v["nan_pattern_equal"] for k, v in res.items() if k != "ok"))
    out = data_dir() / "runs/op_parity/turn_selbench/gate_id.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if not res["ok"]:
        raise SystemExit("G-id failed")


FRAMES = {"navtest": "warp", "navhard": "gimm"}
VAR = {"SH30": "", "A": ":tsA", "B": ":tsB"}


def spec(v, s, bench):
    return f"SH30-F-s{s}@{FRAMES[bench]}{VAR[v]}"


def arm(v, bench):
    return f"{v}=" + "+".join(spec(v, s, bench) for s in (0, 1))


def sel_of(v, s, bench):
    from jevdrive.bench.models import resolve
    import turn_selbench as TB
    z = np.load(TB.select_file(resolve(spec(v, s, bench)), bench))
    return {k: z[k] for k in z.files}


def report(a):
    import pandas as pd
    from jevdrive import bench as BE, stats
    from jevdrive.bench import tables as T
    from jevdrive.bench.models import data_dir
    import turn_ceiling as TC
    import turn_dewater as TD
    out, figd = _pl.Path(a.out), _pl.Path(a.figs)
    out.mkdir(parents=True, exist_ok=True)
    cell = TC.cell
    # ---- 1. the bench tables (arms, paired vs SH30 and WA-JEPA, Shapley, strata)
    BE.report("navtest", [arm("B", "navtest"), arm("A", "navtest")], vs=[arm("SH30", "navtest"), "WA-JEPA"], out=str(out / "bench"))
    BE.report("navhard", [arm("B", "navhard"), arm("A", "navhard")], vs=[arm("SH30", "navhard"), "WA-JEPA"], out=str(out / "bench"), strata=False)
    # ---- 2. per-token frames
    U = {(v, s): T.load("navtest", spec(v, s, "navtest"))[0] for v in VAR for s in (0, 1)}
    idx = U["SH30", 0].index
    for u in U.values():
        assert u.index.equals(idx)
    log = U["SH30", 0].log.astype(str).to_numpy()
    ST = T.navtest_strata().loc[idx]
    turn = ST.turn.to_numpy()
    logged = np.isin(turn, ["20-45", ">45"])
    sc = {k: 100 * u.score.to_numpy(float) for k, u in U.items()}
    sm = lambda v: (sc[v, 0] + sc[v, 1]) / 2  # noqa: E731
    SEL = {(v, s): sel_of(v, s, "navtest") for v in "AB" for s in (0, 1)}
    for z in SEL.values():
        assert np.array_equal(z["tokens"].astype(str), idx.to_numpy().astype(str))
    rows = []
    # ---- 3. headline and strata Delta (official, with EC) vs SH30
    masks = {"all navtest": np.ones(len(idx), bool), "logged |dyaw| >= 20": logged, "logged 20-45": turn == "20-45", "logged > 45": turn == ">45",
             "logged < 5": turn == "<5", "logged 5-20": turn == "5-20", "manoeuvre straight": (ST.maneuver == "straight").to_numpy(),
             "left turn": (ST.maneuver == "left turn").to_numpy(), "right turn": (ST.maneuver == "right turn").to_numpy(), "not logged turn": ~logged}
    for v in "BA":
        for name, m in masks.items():
            r = stats.paired(sm(v)[m], sm("SH30")[m], groups=log[m])
            rows.append({"variant": v, "stratum": name, "n tokens": int(m.sum()), "SH30": r["mean_b"], "variant score": r["mean_a"], "delta": cell(r), "_mean": r["mean"], "_lo": r["lo"], "_hi": r["hi"]})
    for v in "BA":                                                       # gated-in / gated-out, pooled over the two seeds
        for name, gsel in (("gated in", True), ("gated out", False)):
            a_ = np.concatenate([sc[v, s][SEL[v, s]["gate"] == gsel] for s in (0, 1)])
            b_ = np.concatenate([sc["SH30", s][SEL[v, s]["gate"] == gsel] for s in (0, 1)])
            g_ = np.concatenate([log[SEL[v, s]["gate"] == gsel] for s in (0, 1)])
            if len(a_):
                r = stats.paired(a_, b_, groups=g_)
                rows.append({"variant": v, "stratum": f"{name} (token-seeds)", "n tokens": int(len(a_)), "SH30": r["mean_b"], "variant score": r["mean_a"], "delta": cell(r), "_mean": r["mean"], "_lo": r["lo"], "_hi": r["hi"]})
    strata = pd.DataFrame(rows)
    stats.write_table(strata.drop(columns=["_mean", "_lo", "_hi"]), out / "official_delta", floatfmt=".2f",
                      note="official navtest EPDMS x 100 (with EC), seed mean per token, delta = variant - SH30, paired log-cluster bootstrap 95% (B 10 000); gated in / out pool the two seeds' token-seeds")
    # ---- 4. gate coverage vs the logged bucket
    rows = []
    for v in "BA":
        for s in (0, 1):
            g, pk = SEL[v, s]["gate"], SEL[v, s]["picks"]
            rows.append({"variant": v, "seed": s, "gate share %": 100 * g.mean(), "precision % (logged turn | gate)": 100 * logged[g].mean() if g.any() else np.nan,
                         "recall % (gate | logged turn)": 100 * g[logged].mean(), "moved % of all": 100 * (pk != 0).mean(), "moved % of gated": 100 * (pk[g] != 0).mean() if g.any() else np.nan,
                         "moved % of logged turns": 100 * (pk[logged] != 0).mean(), "moved % of straight": 100 * (pk[masks["manoeuvre straight"]] != 0).mean()})
    stats.write_table(rows, out / "gate_coverage", floatfmt=".2f", note="navtest, 12 146 tokens; logged turn = |dyaw| >= 20 deg (privileged bucket); gate B = the model's own exported 4 s heading >= 20 deg")
    # ---- 5. decomposition on the logged-turn tokens: pc no-EC -> raw no-EC -> bench no-EC -> official
    C, tok, dyaw, blog, X = TD.load()
    ti = {t: i for i, t in enumerate(idx.tolist())}
    rowsel = np.array([ti[t] for t in tok])
    F = TD.fams(C)
    Ypc = (TC.noec(TD.conv(X, C, "pc"))[:, F["F19"]] - TC.noec(TD.conv(X, C, "pc"))[:, :1]).transpose(0, 2, 1)
    Yraw = (TC.noec(X)[:, F["F19"]] - TC.noec(X)[:, :1]).transpose(0, 2, 1)
    rows = []
    for v in "BA":
        pk = np.stack([SEL[v, s]["picks"][rowsel] for s in (0, 1)])
        steps = {"1 pc no-EC (candidate table, 191 convention)": TD.take(Ypc, pk).mean(0),
                 "2 raw no-EC (candidate table, same picks)": TD.take(Yraw, pk).mean(0)}
        noec = lambda u: TC.noec(u[T.TERMS].to_numpy(float).reshape(-1, 1, 9)[:, 0])  # noqa: E731
        steps["3 bench no-EC (bench sub-scores)"] = np.mean([noec(U[v, s].iloc[rowsel]) - noec(U["SH30", s].iloc[rowsel]) for s in (0, 1)], 0)
        steps["4 bench official (with EC)"] = (sm(v) - sm("SH30"))[rowsel] / 100
        for k, d in steps.items():
            r = stats.paired(100 * d, np.zeros(len(d)), groups=blog)
            rows.append({"variant": v, "step": k, "n": len(d), "delta": cell(r), "_mean": r["mean"]})
    dec = pd.DataFrame(rows)
    stats.write_table(dec.drop(columns=["_mean"]), out / "decomposition", floatfmt=".2f",
                      note="the 3 154 logged |dyaw| >= 20 deg navtest tokens, seed mean, EPDMS x 100 gain over SH30; step 1 values the picks of this run under decision 191's pc convention, "
                           "step 2 takes the candidate scores raw, step 3 recomputes no-EC from the bench run's own sub-scores (must equal step 2), step 4 adds EC (neighbouring-frame comfort)")
    # ---- 6. sub-scores B and A vs SH30
    rows = []
    for v in "BA":
        for t in ["score"] + T.TERMS:
            a_ = np.mean([U[v, s][t].to_numpy(float) for s in (0, 1)], 0)
            b_ = np.mean([U["SH30", s][t].to_numpy(float) for s in (0, 1)], 0)
            r = stats.paired(100 * a_, 100 * b_, groups=log)
            rows.append({"variant": v, "term": "EPDMS" if t == "score" else t, "SH30": r["mean_b"], "variant": r["mean_a"] if False else v, "variant value": r["mean_a"], "delta": cell(r)})
    stats.write_table(rows, out / "subscores", floatfmt=".2f", note="navtest, seed-mean per token, x 100, paired log-cluster 95% CI")
    # ---- 7. headline JSON for the result note
    js = {}
    for v in "BA":
        full = strata[(strata.variant == v) & (strata.stratum == "all navtest")].iloc[0]
        st_ = strata[(strata.variant == v) & (strata.stratum == "manoeuvre straight")].iloc[0]
        br = strata[(strata.variant == v) & (strata.stratum == "logged |dyaw| >= 20")].iloc[0]
        js[v] = dict(full=[full._mean, full._lo, full._hi], straight=[st_._mean, st_._lo, st_._hi], bucket=[br._mean, br._lo, br._hi],
                     entry_real=bool(full._lo > 0 and st_._mean >= -0.10))
    (out / "verdict.json").write_text(json.dumps(js, indent=1, default=float))
    print(json.dumps(js, indent=1, default=float))
    fig(figd, strata, dec)


def fig(figd, strata, dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    figd.mkdir(parents=True, exist_ok=True)
    fig_, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.0, 1.7]))
    ax = axs[0]
    d = dec[dec.variant == "B"].reset_index(drop=True)
    lab = ["pc\nno-EC", "raw\nno-EC", "bench\nno-EC", "official\nwith EC"]
    ax.bar(range(len(d)), d._mean, 0.7, color=[ps.PALETTE["blue"]] * 3 + [ps.PALETTE["vermillion"]])
    for i, m in enumerate(d._mean):
        ax.text(i, m + 0.05, f"{m:+.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(d)), lab, fontsize=7)
    ax.set_ylabel("B - SH30 on logged |dyaw| >= 20 tokens\nEPDMS x 100")
    ps.bars(ax), ps.zero_line(ax)
    ax = axs[1]
    names = ["all navtest", "logged |dyaw| >= 20", "logged 20-45", "logged > 45", "left turn", "right turn", "manoeuvre straight", "logged < 5", "gated in (token-seeds)", "gated out (token-seeds)"]
    x0 = np.arange(len(names))
    for j, (v, c) in enumerate((("B", ps.PALETTE["blue"]), ("A", ps.PALETTE["orange"]))):
        s_ = strata[strata.variant == v].set_index("stratum")
        s_ = s_.reindex(names)
        m = s_._mean.to_numpy(float)
        ax.errorbar(x0 + (j - 0.5) * 0.25, m, yerr=[m - s_._lo.to_numpy(float), s_._hi.to_numpy(float) - m], fmt="o", ms=2.5, lw=0.7, capsize=1.2, color=c, label=f"gate {v}")
    ax.set_xticks(x0, [n.replace(" (token-seeds)", "").replace("manoeuvre ", "").replace("logged ", "") for n in names], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("variant - SH30, official EPDMS x 100 (95% CI)")
    ps.zero_line(ax), ax.legend(fontsize=6.5)
    fig_.savefig(figd / "turn_selector_bench.png", dpi=300)
    plt.close(fig_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("idcheck")
    p.add_argument("--subset", required=True)
    a = ap.parse_args()
    {"idcheck": cmd_idcheck}[a.cmd](a)
