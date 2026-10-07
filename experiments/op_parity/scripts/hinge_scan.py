"""op_parity hinge-scan (plans/2026-10-08-hinge-scan-prereg.md): lambda x margin grid of the drivable hinge at pilot scale, and SH30 on WOD val.

  scan   (op-train env, CPU)  every grid arm vs SHP-F-s0 on navtest: paired log-cluster bootstrap of EPDMS / straight EPDMS / EP / raw-plan out-of-bounds /
                              DAC / inside-cut / cannot-make-turn; the pre-registered winner rule -> results/hinge_scan/, figs/hinge_scan/
  wod    (jevdrive env, CPU)  SH30-F-s0 / s1 served on WOD val (preds op_cinque_<tag>) vs P2H10 and shipped: RFS, ADE, turn-frame inward miss
Scoring itself is jevdrive.bench; the replay (turn_oracle.py replay --name hs) and geometry (rh.py proxy --name <tag>) parquets come from the chain.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_S = _pl.Path(__file__).parent
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "research"), str(_R / "experiments/op_adapt_r2/lib"), str(_S),
                 str(_R / "experiments/op_probe/scripts")]
import argparse, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RES = _R / "experiments/op_parity/results/hinge_scan"
FIG = _R / "experiments/op_parity/figs/hinge_scan"
GEOM = D / "runs/op_parity/replay_hinge"
LAMS, MARGINS = (10, 30, 100), (0.25, 0.5, 1.0)
REF, REF_S1 = "SHP-F-s0", "SHP-F-s1"
NB = 4000
N_CAND = 8
ALPHA_BONF = 0.05 / N_CAND


def tag(lam, m):
    return REF if (lam, m) == (30, 0.5) else f"SC-L{lam}M{int(round(m * 100))}-F-s0"


GRID = {(lam, m): tag(lam, m) for lam in LAMS for m in MARGINS}


def feats(Dt, t):
    f, _ = Dt.one(t)
    g = pd.read_parquet(GEOM / f"geom_{t}.parquet").set_index("token").reindex(Dt.tok)
    f["raw-plan out %"] = g.raw_out.astype(float).to_numpy() * 100
    f["replay out %"] = g.dev_out.astype(float).to_numpy() * 100
    return f


def cmd_scan(a):
    from jevdrive import stats
    import turn_oracle as TO
    Dt = TO.Data(a.replays)
    S5, T20, T45 = Dt.sets["S5 (< 5 deg)"], Dt.sets["T20 (> 20 deg)"], Dt.sets["T45 (> 45 deg)"]
    ALL = Dt.sets["all"]
    tags = list(dict.fromkeys([*GRID.values(), REF_S1]))
    F = {t: feats(Dt, t) for t in tags}
    M = [("EPDMS all", "EPDMS", ALL), ("EPDMS straight (< 5 deg)", "EPDMS", S5), ("EP all", "EP", ALL), ("EP straight", "EP", S5),
         ("raw-plan out % (all)", "raw-plan out %", ALL), ("replay out % (all)", "replay out %", ALL), ("DAC fail % (all)", "DAC fail %", ALL),
         ("raw-plan out % (> 20 deg)", "raw-plan out %", T20), ("inside-cut % (> 45 deg)", "inside-cut %", T45),
         ("cannot-make-turn % (> 45 deg)", "cannot-make-turn %", T45), ("inside-cut % (> 20 deg)", "inside-cut %", T20)]
    rows, vals = [], []

    def contrast(t, ref):
        for name, col, m in M:
            x, y = F[t][col].to_numpy()[m], F[ref][col].to_numpy()[m]
            r = stats.paired(x, y, groups=Dt.log[m], n_boot=NB)
            r99 = stats.paired(x, y, groups=Dt.log[m], n_boot=NB, alpha=ALPHA_BONF)
            rows.append(dict(arm=t, ref=ref, metric=name, n=int(m.sum()), arm_mean=r["mean_a"], ref_mean=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"],
                             lo_bonf=r99["lo"], hi_bonf=r99["hi"]))
    for t in tags:
        for name, col, m in M:
            vals.append(dict(arm=t, metric=name, n=int(m.sum()), value=float(np.nanmean(F[t][col].to_numpy()[m]))))
        if t != REF:
            contrast(t, REF)
    P = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(vals).to_csv(RES / "arms.csv", index=False, float_format="%.4f")
    stats.write_table(P, RES / "paired", floatfmt=".2f", note=f"arm - {REF}, navtest 12 146 tokens (x 100), log-cluster paired bootstrap B {NB}; "
                      f"lo_bonf / hi_bonf: alpha 0.05 / {N_CAND}")
    q = lambda t, met, k: float(P[(P.arm == t) & (P.metric == met)][k].iloc[0])  # noqa: E731
    seed_sp = abs(q(REF_S1, "EPDMS all", "diff"))
    verdict = {}
    for (lam, m), t in GRID.items():
        if t == REF:
            continue
        w1 = q(t, "EPDMS all", "lo_bonf") > 0 and q(t, "EPDMS all", "diff") >= seed_sp
        w2 = all(q(t, k, "diff") >= -0.2 and q(t, k, "hi") >= 0 for k in ("EPDMS straight (< 5 deg)", "EP all"))
        w3 = q(t, "raw-plan out % (all)", "diff") <= 0
        verdict[t] = dict(lam=lam, margin=m, W1=bool(w1), W2=bool(w2), W3=bool(w3), win=bool(w1 and w2 and w3), epdms_diff=q(t, "EPDMS all", "diff"))
    win = [t for t, v in verdict.items() if v["win"]]
    best = max(win, key=lambda t: verdict[t]["epdms_diff"]) if win else None
    summ = dict(seed_spread_epdms=seed_sp, seed_diff_epdms=q(REF_S1, "EPDMS all", "diff"), winners=win, best=best, arms=verdict,
                rule="W1 99.38% lower > 0 and diff >= |SHP s1 - s0|; W2 straight EPDMS & EP diff >= -0.2 and 95% hi >= 0; W3 raw-plan out all diff <= 0")
    (RES / "verdict.json").write_text(json.dumps(summ, indent=1))
    # grid tables (rows lambda, columns margin) of the diffs and the margin-only / lambda-only main effects
    G = {}
    for name in ("EPDMS all", "raw-plan out % (all)", "EPDMS straight (< 5 deg)", "EP all", "inside-cut % (> 45 deg)", "cannot-make-turn % (> 45 deg)"):
        G[name] = np.array([[0.0 if GRID[(lam, m)] == REF else q(GRID[(lam, m)], name, "diff") for m in MARGINS] for lam in LAMS])
    pd.concat({k: pd.DataFrame(v, index=LAMS, columns=MARGINS) for k, v in G.items()}).to_csv(RES / "grid_diffs.csv", float_format="%.3f")
    figure(G, verdict)
    print(json.dumps(summ, indent=1))
    print(P[P.metric.isin(["EPDMS all", "raw-plan out % (all)"])].round(2).to_string())


def figure(G, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.9), constrained_layout=True)
    for ax, (name, title, cmap, sgn) in zip(axs, (("EPDMS all", "navtest EPDMS change vs SHP (30 / 0.5), points", "RdBu", 1),
                                                  ("raw-plan out % (all)", "raw-plan out-of-bounds change vs SHP, pp (lower is better)", "RdBu_r", 1))):
        g = G[name]
        v = max(abs(g).max(), 1e-6)
        ax.imshow(g, cmap=cmap, vmin=-v, vmax=v, origin="lower")
        for i, lam in enumerate(LAMS):
            for j, m in enumerate(MARGINS):
                t = GRID[(lam, m)]
                mark = "ref" if t == REF else ("W" if verdict[t]["win"] else "")
                ax.text(j, i, f"{g[i, j]:+.2f}" + (f"\n{mark}" if mark else ""), ha="center", va="center", fontsize=8)
        ax.set_xticks(range(3), [f"{m}" for m in MARGINS]), ax.set_yticks(range(3), [f"{l}" for l in LAMS])
        ax.set_xlabel("margin (m)"), ax.set_ylabel("lambda"), ax.set_title(title, fontsize=7.5), ax.grid(False)
    fig.savefig(FIG / "grid_heatmap.png", dpi=300)
    fig.savefig(FIG / "grid_heatmap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- WOD
def cmd_wod(a):
    from jevdrive import stats, waymo as W
    from pp_wod import frames, load_preds
    from wod_gap import parts
    S = frames()
    r, x = S["rater"], S["extra"]
    nr = len(r["name"])
    names = np.concatenate([r["name"], x["name"]]).astype(str)
    seq_all = np.concatenate([r["sequence"], x["sequence"]]).astype(str)
    fut = np.concatenate([r["future"], x["future"]])[..., :2].astype(np.float64)
    traj, sc, v0, cl = r["traj"].astype(np.float64), r["scores"].astype(np.float64), W.init_speed(r["past"]), r["cluster"].astype(str)
    arms = {"shipped": ["shipped"], "P2H10": ["P2H10-F-s0", "P2H10-F-s1"], "SH30": ["SH30-F-s0", "SH30-F-s1"]}
    tags = sorted({t for v in arms.values() for t in v})
    P = {t: load_preds(t, names) for t in tags}
    rfs = {t: np.asarray(W.rater_feedback_score(P[t][:nr], traj, sc, v0), float) for t in tags}
    ade3 = {t: np.linalg.norm(P[t] - fut, axis=-1)[:, :12].mean(1) for t in tags}
    ade5 = {t: np.linalg.norm(P[t] - fut, axis=-1).mean(1) for t in tags}
    mean = lambda d, k: np.mean([d[t] for t in arms[k]], 0)  # noqa: E731
    R = {k: mean(rfs, k) for k in arms}
    A3, A5 = ({k: mean(d, k) for k in arms} for d in (ade3, ade5))
    codes_r, ur = pd.factorize(pd.Series(seq_all[:nr]))
    codes_a, ua = pd.factorize(pd.Series(seq_all))
    idx_r = [np.flatnonzero(codes_r == k) for k in range(len(ur))]
    idx_a = [np.flatnonzero(codes_a == k) for k in range(len(ua))]
    rng = np.random.default_rng(0)
    dr = [rng.integers(len(ur), size=len(ur)) for _ in range(NB)]
    da = [rng.integers(len(ua), size=len(ua)) for _ in range(NB)]
    rf = lambda f, i: W.rfs_by_cluster(f[i], cl[i])[0]  # noqa: E731
    allr = np.arange(nr)

    def b_rfs(f, g):
        out = [rf(f, (i := np.concatenate([idx_r[k] for k in d]))) - rf(g, i) for d in dr]
        return np.percentile(out, [2.5, 97.5])

    def b_mean(f, g):
        out = [f[i].mean() - g[i].mean() for i in (np.concatenate([idx_a[k] for k in d]) for d in da)]
        return np.percentile(out, [2.5, 97.5])

    rows = []
    for k in ("P2H10", "SH30"):
        for nm, f_, g_ in (("shipped", R[k], R["shipped"]),):
            lo, hi = b_rfs(f_, g_)
            rows.append(dict(contrast=f"{k} - shipped", metric="RFS", arm=rf(f_, allr), ref=rf(g_, allr), diff=rf(f_, allr) - rf(g_, allr), lo=lo, hi=hi))
        for m, dd in (("ADE@3s", A3), ("ADE@5s", A5)):
            lo, hi = b_mean(dd[k], dd["shipped"])
            rows.append(dict(contrast=f"{k} - shipped", metric=m, arm=dd[k].mean(), ref=dd["shipped"].mean(), diff=dd[k].mean() - dd["shipped"].mean(), lo=lo, hi=hi))
    lo, hi = b_rfs(R["SH30"], R["P2H10"])
    rows.append(dict(contrast="SH30 - P2H10", metric="RFS", arm=rf(R["SH30"], allr), ref=rf(R["P2H10"], allr), diff=rf(R["SH30"], allr) - rf(R["P2H10"], allr), lo=lo, hi=hi))
    for m, dd in (("ADE@3s", A3), ("ADE@5s", A5)):
        lo, hi = b_mean(dd["SH30"], dd["P2H10"])
        rows.append(dict(contrast="SH30 - P2H10", metric=m, arm=dd["SH30"].mean(), ref=dd["P2H10"].mean(), diff=dd["SH30"].mean() - dd["P2H10"].mean(), lo=lo, hi=hi))
    for t in ("SH30-F-s0", "SH30-F-s1", "P2H10-F-s0", "P2H10-F-s1"):
        lo, hi = b_rfs(rfs[t], R["shipped"])
        rows.append(dict(contrast=f"{t} - shipped", metric="RFS", arm=rf(rfs[t], allr), ref=rf(R["shipped"], allr), diff=rf(rfs[t], allr) - rf(R["shipped"], allr), lo=lo, hi=hi))
    lo, hi = b_rfs(rfs["SH30-F-s0"], rfs["SH30-F-s1"])
    rows.append(dict(contrast="SH30 s0 - s1 (seed noise)", metric="RFS", diff=rf(rfs["SH30-F-s0"], allr) - rf(rfs["SH30-F-s1"], allr), lo=lo, hi=hi))
    # turn frames: lateral miss at 5 s against the top-rated trajectory, + = toward the inside of the turn
    top = sc.argmax(1)
    ii = np.arange(nr)
    intent = r["intent"]
    tm = intent >= 2
    sign = np.where(intent == 2, 1, -1)
    lat, lon = {}, {}
    for k in arms:
        qs = [parts(P[t][:nr], traj, sc, v0) for t in arms[k]]
        lat[k] = np.mean([q["e_lat"][ii, top, 1] * sign for q in qs], 0)           # seed mean per frame
        lon[k] = np.mean([q["e_lon"][ii, top, 1] for q in qs], 0)
    trows = []
    for k in arms:
        l, o = lat[k][tm], lon[k][tm]
        near = np.abs(o) < 3.0
        trows.append(dict(arm=k, n_turn_frames=int(tm.sum()), median_inward_m=float(np.median(l)), mean_inward_m=float(l.mean()), inside_gt1m=float((l > 1).sum()),
                          wide_gt1m=float((l < -1).sum()), n_near=int(near.sum()), near_median_inward_m=float(np.median(l[near])), near_inside_gt1m=float((l[near] > 1).sum())))
    for k, ref in (("SH30", "P2H10"), ("SH30", "shipped"), ("P2H10", "shipped")):
        g = codes_r[tm]
        for nm, f in (("mean inward miss (m)", lambda z: z), ("inside > 1 m (%)", lambda z: (z > 1) * 100.0)):
            res = stats.paired(f(lat[k][tm]), f(lat[ref][tm]), groups=g, n_boot=NB)
            trows.append(dict(arm=f"{k} - {ref}", metric=nm, n_turn_frames=int(tm.sum()), mean=res["mean_a"], ref=res["mean_b"], diff=res["mean"], lo=res["lo"], hi=res["hi"]))
    RES.mkdir(parents=True, exist_ok=True)
    stats.write_table(rows, RES / "wod_rfs", floatfmt=".3f", note=f"WOD val, 479 rater frames (ADE: 1 437 frames), seed means, bootstrap over sequences B {NB}")
    stats.write_table(trows, RES / "wod_turn_side", floatfmt=".2f", note="turn frames (intent left / right) of the 479 rater frames; miss at 5 s against the top-rated trajectory, + = inside of the turn; "
                      "per-frame seed means; paired rows bootstrap over sequences")
    print(pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(trows).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("scan")
    p.add_argument("--replays", nargs="+", default=["hs"])
    sp.add_parser("wod")
    a = ap.parse_args()
    {"scan": cmd_scan, "wod": cmd_wod}[a.cmd](a)
