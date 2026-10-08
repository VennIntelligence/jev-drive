"""Report stage of turn_selwod.py (plans/2026-10-08-turn-selwod-prereg.md): RFS of the selector on the 479 WOD val rater frames, paired against the same run's
unmodified plan (ts0), with the four context readings. Repo venv, CPU. Conventions of the WOD lanes (wod_launch_report.Ctx): cluster-mean RFS, paired
bootstrap over sequences (B draws of Ctx), seed mean = per-frame mean of the two seeds' scores.
"""
import json
import sys as _sys

import numpy as np
import pandas as pd

import turn_selwod as TW
from wod_launch_report import Ctx

R = TW._R
OUTD = R / "experiments/op_parity/results"
FIGD = R / "experiments/op_parity/figs"
TURN_DEG = 20.0
NAVTEST_REL = 2.11 / 81.15          # decision 193: gated-in navtest tokens, +2.11 EPDMS x 100 on 81.15 (relative gain)


def f3(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


class Arm:
    """One model family (tags = its two seeds): per-seed candidate RFS tables and the picks."""

    def __init__(self, C, tags, base="tapped"):
        self.C, self.tags = C, tags
        self.S = []
        for t in tags:
            z = np.load(TW.sel_file(t))
            names = z["names"].astype(str)
            assert sorted(names.tolist()) == sorted(C.names[: C.n].tolist())
            idx = np.array([{k: i for i, k in enumerate(names)}[k] for k in C.names[: C.n]])
            d = {k: z[k][idx] for k in ("pred", "pick_free", "dyaw", "W10", "W4", "wod", "arch_wod")}
            if base == "archived":                                  # prereg addendum only: archived waypoints + the candidate's change
                for k in ("W10", "W4"):
                    d[k] = d["arch_wod"][:, None] + (d[k] - d["wod"][:, None])
            d["info"] = json.loads(str(z["info"]))
            d["R10"] = np.stack([C.rfs(d["W10"][:, c].astype(np.float64)) for c in range(d["W10"].shape[1])], 1)      # (n, 19)
            d["R4"] = np.stack([C.rfs(d["W4"][:, c].astype(np.float64)) for c in range(d["W4"].shape[1])], 1)
            d["arch"] = C.rfs(d["arch_wod"].astype(np.float64))
            d["gB"] = d["dyaw"] >= TURN_DEG
            d["pk_B"] = np.where(d["gB"], d["pick_free"], 0)
            self.S.append(d)

    def per_frame(self, f):
        return np.mean([f(d) for d in self.S], 0)

    def pick_rfs(self, key="R10", gate="B"):
        return self.per_frame(lambda d: d[key][np.arange(self.C.n), d["pk_B"] if gate == "B" else d["pick_free"] if gate == "A" else 0 * d["pk_B"]])


def analyze(C, tags, label, out, base_mode="tapped"):
    A = Arm(C, tags, base_mode)
    n = C.n
    ii = np.arange(n)
    res = {"label": label, "tags": tags}
    # ---- G-id: ts0 (the tapped run's own plan) against the archived run of the same weights, per frame
    base = A.per_frame(lambda d: d["R10"][:, 0])
    arch = A.per_frame(lambda d: d["arch"])
    dd = base - arch
    gid = dict(rfs_ts0=C.cm(base), rfs_archived=C.cm(arch), diff=C.cm(dd), frac_frames_within_0p01=float((np.abs(dd) <= 0.01).mean()), frac_frames_within_0p05=float((np.abs(dd) <= 0.05).mean()), max_abs_frame_diff=float(np.abs(dd).max()),
               plan=[d["info"] for d in A.S])
    gid["ok"] = bool(abs(gid["diff"]) <= 0.01 and gid["frac_frames_within_0p05"] >= 0.95)          # prereg: cluster-mean <= 0.01, >= 95% of frames within 0.05
    # G-feat at the waypoint level (the tolerance of the prereg): tapped run vs archived run of the same weights, and vs the other seed's archived run (control)
    gf = []
    for d, t in zip(A.S, tags):
        dw = np.abs(d["arch_wod"].astype(np.float64) - d["wod"]).max((1, 2))
        gf.append(dict(tag=t, rows_over_0p25=float((dw > 0.25).mean()), median=float(np.median(dw)), max=float(dw.max())))
    for i, d in enumerate(A.S):
        dc = np.abs(A.S[1 - i]["arch_wod"].astype(np.float64) - d["wod"]).max((1, 2))
        gf[i].update(control_rows_over_0p25=float((dc > 0.25).mean()), control_median=float(np.median(dc)))
    gid["g_feat"] = gf
    gid["g_feat_ok"] = bool(all(g["rows_over_0p25"] <= 0.02 and g["control_rows_over_0p25"] > 0.5 for g in gf))
    gid["g_eqv"] = [d["info"]["g_eqv_pick_agreement"] for d in A.S]
    gid["g_gen"] = [d["info"]["g_gen_max_abs"] for d in A.S]
    gid["g_map"] = [d["info"]["g_map_max_abs_m"] for d in A.S]
    res["gid"] = gid
    out.joinpath(f"{label}_gates.json").write_text(json.dumps(gid, indent=1))
    if not (gid["ok"] and gid["g_feat_ok"]) and base_mode == "tapped" and label == "SH30":
        raise SystemExit(f"gate failed, stop and report: {json.dumps({k: v for k, v in gid.items() if k != 'plan'})}")
    print(label, "G-id", {k: v for k, v in gid.items() if k != "plan"}, flush=True)
    # ---- masks
    intent = np.asarray(C.intent)
    seq_clusters = C.Z.load_sets()["rater"]["cluster"].astype(str)
    names_c = sorted(set(seq_clusters))
    strata = {"all": np.ones(n, bool), "turn intent (left / right)": intent >= 2, "straight intent": intent == 1, "stopped (v0 < 0.5)": C.st["stopped"], "moving (v0 >= 0.5)": C.st["moving (v>=0.5)"],
              "slow 0.5-5": C.st["slow 0.5-5"], "mid 5-12": C.st["mid 5-12"], "fast >= 12": C.st["fast >=12"], "night": C.st["night"], "day": C.st["day"]}
    strata["gate B in (>= 1 of 2 seeds)"] = np.any([d["gB"] for d in A.S], 0)
    strata["gate B out (both seeds)"] = ~strata["gate B in (>= 1 of 2 seeds)"]
    strata |= {f"cluster: {c}": seq_clusters == c for c in names_c}
    res["n_gate_B"] = [int(d["gB"].sum()) for d in A.S]
    # ---- primary and strata
    rows = []
    for variant, key in (("primary (0.5-5 s)", "R10"), ("H4 (0.5-4 s, held)", "R4")):
        for gate, gl in (("B", "gate B"), ("A", "gate A")):
            sel = A.pick_rfs(key, gate)
            d = sel - base
            for nm, m in strata.items():
                if not m.any():
                    continue
                p, lo, hi = C.ci(d, m)
                rows.append(dict(variant=variant, gate=gl, stratum=nm, n=int(m.sum()), ts0=C.cm(base, m), selector=C.cm(sel, m), d=p, lo=lo, hi=hi))
    T = pd.DataFrame(rows)
    T.to_csv(out / f"{label}_strata.csv", index=False)
    prim = T[(T.variant == "primary (0.5-5 s)") & (T.gate == "gate B") & (T.stratum == "all")].iloc[0]
    per_seed = []
    for s, d in zip(tags, A.S):
        dseed = d["R10"][ii, d["pk_B"]] - d["R10"][:, 0]
        per_seed.append(dict(tag=s, d=C.ci(dseed), ts0=C.cm(d["R10"][:, 0]), selector=C.cm(d["R10"][ii, d["pk_B"]]),
                             n_gate=int(d["gB"].sum()), n_moved=int((d["pk_B"] != 0).sum())))
    from jevdrive import stats
    seq = C.seq[:n]
    fm = stats.paired(base + (A.pick_rfs("R10", "B") - base), base, groups=seq)
    res["primary"] = dict(ts0=prim.ts0, selector=prim.selector, d=prim.d, lo=prim.lo, hi=prim.hi, frame_mean=fm, per_seed=per_seed)
    lo, hi = prim.lo, prim.hi
    both_pos = all(p["d"][0] > 0 for p in per_seed)
    res["verdict"] = "transfers" if (lo > 0 and both_pos) else "harms" if hi < 0 else "does not transfer"
    res["verdict_detail"] = dict(lo_gt0=bool(lo > 0), both_seed_points_gt0=bool(both_pos), hi_lt0=bool(hi < 0))
    # ---- context readings
    import turn_ceiling as TC
    cands = TC.candidates()[:19]
    F = TC.families(cands)
    gain = lambda key: A.per_frame(lambda d: d[key] - d[key][:, [0]])                       # noqa: E731  (n, 19) realised change of every candidate, seed mean
    G10, G4 = gain("R10"), gain("R4")
    gate_m = strata["gate B in (>= 1 of 2 seeds)"]
    ctx = {}
    ceil = {}
    for fam in ("O3", "K3", "V3", "F7", "F19"):
        for variant, G in (("10", G10), ("4", G4)):
            idx = F[fam]
            best = G[:, idx].max(1)
            ceil[f"{fam}_{variant}"] = dict(all=C.ci(best), gated=C.ci(np.where(gate_m, best, 0.0)), gated_only_rows=C.ci(best, gate_m), intent=C.ci(best, strata["turn intent (left / right)"]))
    ctx["ceiling"] = ceil
    # privileged-by-seed ceiling: best per frame inside each seed, then average (the seed-mean plan is not one plan)
    best_seed = A.per_frame(lambda d: (d["R10"] - d["R10"][:, [0]])[:, F["F19"]].max(1))
    ctx["ceiling_F19_seedwise"] = dict(all=C.ci(best_seed), gated=C.ci(np.where(gate_m, best_seed, 0.0)))
    # (2) gate firing and pick marginals
    pk_rows = []
    for t, d in zip(tags, A.S):
        pk = d["pk_B"]
        pf = d["pick_free"]
        mv = pk != 0
        o, k, v = (np.array([cands[c][j] for c in pk]) for j in (1, 2, 3))
        pk_rows.append(dict(tag=t, gate_share=float(d["gB"].mean()), gate_share_turn_intent=float(d["gB"][intent >= 2].mean()), gate_share_straight=float(d["gB"][intent == 1].mean()),
                            moved_share_all=float(mv.mean()), moved_of_gated=float(mv[d["gB"]].mean()) if d["gB"].any() else np.nan,
                            moved_free_all=float((pf != 0).mean()),
                            offset_moved=float((o[mv] != 0).mean()) if mv.any() else np.nan, curvature_moved=float((k[mv] != 1).mean()) if mv.any() else np.nan,
                            speed_moved=float((v[mv] != 1).mean()) if mv.any() else np.nan,
                            speed_lt1=float((v[mv] < 1).mean()) if mv.any() else np.nan, speed_gt1=float((v[mv] > 1).mean()) if mv.any() else np.nan,
                            offset_left=float((o[mv] > 0).mean()) if mv.any() else np.nan, offset_right=float((o[mv] < 0).mean()) if mv.any() else np.nan,
                            hist=np.bincount(pk, minlength=19).tolist()))
    nav_rows = []
    for s in (0, 1):
        z = np.load(TW.D / f"runs/op_parity/turn_selbench/select/SH30-F-s{s}@warp_tsB__navtest.npz")
        pk, gt = z["picks"], z["gate"]
        o, k, v = (np.array([cands[c][j] for c in pk]) for j in (1, 2, 3))
        mv = pk != 0
        nav_rows.append(dict(tag=f"navtest SH30-F-s{s}", gate_share=float(gt.mean()), moved_share_all=float(mv.mean()), moved_of_gated=float(mv[gt].mean()),
                             offset_moved=float((o[mv] != 0).mean()), curvature_moved=float((k[mv] != 1).mean()), speed_moved=float((v[mv] != 1).mean()),
                             speed_lt1=float((v[mv] < 1).mean()), speed_gt1=float((v[mv] > 1).mean()), offset_left=float((o[mv] > 0).mean()), offset_right=float((o[mv] < 0).mean()),
                             hist=np.bincount(pk, minlength=19).tolist()))
    ctx["picks"] = pk_rows + nav_rows
    # (3) rank agreement on gated-in rows, per seed
    from scipy.stats import spearmanr
    ag = []
    for t, d in zip(tags, A.S):
        g = np.flatnonzero(d["gB"])
        Gs = d["R10"] - d["R10"][:, [0]]
        rho = [spearmanr(d["pred"][i, 1:], Gs[i, 1:])[0] for i in g if np.ptp(Gs[i, 1:]) > 0]
        best = Gs[:, 1:].argmax(1) + 1
        pk = d["pk_B"]
        pr = np.array([(Gs[i, 1:] < Gs[i, pk[i]]).mean() if pk[i] else np.nan for i in g])           # share of the 18 moved candidates the pick beats
        tie = np.array([np.ptp(Gs[i, 1:]) == 0 for i in g])
        oracle_gate = np.where(Gs[ii, pk] > 0, Gs[ii, pk], 0.0)
        ag.append(dict(tag=t, n_gated=len(g), n_informative=int((~tie).sum()), spearman_mean=float(np.nanmean(rho)), spearman_median=float(np.nanmedian(rho)),
                       top1_agree=float((pk[g][~tie] == best[g][~tie]).mean()), pick_beats_share_mean=float(np.nanmean(pr[~tie])),
                       pick_gain_gated=float(Gs[g, pk[g]].mean()), pick_gain_rate_pos=float((Gs[g, pk[g]] > 0).mean()), pick_gain_rate_neg=float((Gs[g, pk[g]] < 0).mean()),
                       oracle_gate_d=C.ci(np.where(d["gB"], oracle_gate, 0.0))))
    ctx["rank"] = ag
    # (4) candidate-by-candidate valuation on gated-in rows (seed mean) and the decomposition
    val = []
    for c in range(19):
        npick = [int(((d["pk_B"] == c) & d["gB"]).sum()) for d in A.S]
        gm = G10[gate_m, c].mean() if gate_m.any() else np.nan
        val.append(dict(cand=c, name=cands[c][0], offset=cands[c][1], curv_gain=cands[c][2], speed=cands[c][3], static_gain_gated=float(gm), picked_s0=npick[0], picked_s1=npick[1]))
    ctx["candidates"] = val
    ids = np.zeros(n)
    sel_d = A.pick_rfs("R10", "B") - base
    rand_d = np.where(gate_m, G10[:, 1:].mean(1), 0.0)                                         # a uniformly random moved candidate on gated-in rows
    fixed = int(np.argmax([G10[gate_m, c].mean() for c in range(19)])) if gate_m.any() else 0
    fix_d = np.where(gate_m, G10[:, fixed], 0.0)
    bestc = np.where(gate_m, G10[:, F["F19"]].max(1), 0.0)
    dec = [("identity", ids), ("selector (gate B)", sel_d), ("random moved candidate (gate B)", rand_d), (f"best fixed candidate c{fixed:02d} (privileged choice of one constant)", fix_d),
           ("privileged best of F19 (gate B)", bestc)]
    ctx["decomposition"] = [dict(arm=nm, d_all=C.ci(v), d_gated_rows=C.ci(v, gate_m), d_turn_intent=C.ci(v, strata["turn intent (left / right)"])) for nm, v in dec]
    # ---- power statement (what the data allow)
    se = (hi - lo) / (2 * 1.96)
    mde = 2.8 * se
    gated_ts0 = base[gate_m].mean() if gate_m.any() else np.nan
    exp_gain = gate_m.mean() * gated_ts0 * NAVTEST_REL
    res["power"] = dict(n_rater=n, n_gated_seedmean=int(gate_m.sum()), se_primary=se, mde_80=mde, expected_scaled_navtest=exp_gain, mean_ts0_gated=gated_ts0,
                        sd_pick_diff_gated=float(sel_d[gate_m].std()) if gate_m.any() else np.nan)
    res["ctx"] = ctx
    (out / f"{label}.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return res, A, strata, T


def report(a):
    import matplotlib
    matplotlib.use("Agg")
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = OUTD / a.name
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "turn_selwod/report", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        C = Ctx()
        res = {}
        groups = {"SH30": ["SH30-F-s0", "SH30-F-s1"], "WLG": ["WLG-full-s0", "WLG-full-s1"]}
        for lab, tags in groups.items():
            if not all(TW.sel_file(t).exists() for t in tags):
                continue
            res[lab] = analyze(C, tags, lab, out, a.base)
        run.summary.update({k: dict(verdict=v[0]["verdict"], d=v[0]["primary"]["d"], lo=v[0]["primary"]["lo"], hi=v[0]["primary"]["hi"]) for k, v in res.items()})
        figs(res, C)


def figs(res, C):
    import matplotlib.pyplot as plt
    sys_path = str(R / "research")
    if sys_path not in _sys.path:
        _sys.path.insert(0, sys_path)
    import plot_style as PS
    PS.apply()
    FIGD.joinpath("turn_selector_wod").mkdir(parents=True, exist_ok=True)
    r, A, strata, T = res["SH30"]
    # figure 1: effect by stratum (forest) for gate B, primary variant, seed mean
    sub = T[(T.variant == "primary (0.5-5 s)") & (T.gate == "gate B")]
    order = [s for s in sub.stratum if not s.startswith("cluster") and s not in ("slow 0.5-5", "mid 5-12", "fast >= 12", "night", "day")] + [s for s in sub.stratum if s.startswith("cluster")]
    sub = sub.set_index("stratum").loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(PS.DOUBLE_COLUMN_IN, 3.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    y = np.arange(len(sub))[::-1]
    ax[0].errorbar(sub.d, y, xerr=[sub.d - sub.lo, sub.hi - sub.d], fmt="o", color=PS.PALETTE["blue"], ms=3, capsize=1.5)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{s} (n={n})" for s, n in zip(sub.index, sub.n)], fontsize=6.5)
    ax[0].axvline(0, color="#999999", lw=0.5)
    ax[0].set_xlabel("RFS change, selector gate B minus identity (95% CI)")
    # figure 1b: decomposition on gated-in rows
    dec = r["ctx"]["decomposition"]
    labels = ["selector", "random moved", "best fixed", "privileged best"]
    vals = [dec[1], dec[2], dec[3], dec[4]]
    cols = [PS.PALETTE["blue"], PS.BASELINE, PS.PALETTE["orange"], PS.PALETTE["green"]]
    for i, (v, c) in enumerate(zip(vals, cols)):
        p, lo, hi = v["d_all"]
        ax[1].bar(i, p, color=c, yerr=[[p - lo], [hi - p]], capsize=2, width=0.65)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(labels, rotation=25, ha="right")
    ax[1].axhline(0, color="#999999", lw=0.5)
    ax[1].set_ylabel("RFS change over identity, all 479 frames")
    fig.tight_layout()
    PS.save(fig, FIGD / "turn_selector_wod" / "effect_and_ceiling")
    plt.close(fig)
    # figure 2: pick marginals WOD vs navtest
    pk = r["ctx"]["picks"]
    keys = ["offset_moved", "curvature_moved", "speed_moved", "speed_lt1", "offset_left"]
    names = ["offset moved", "curvature moved", "speed moved", "speed < 1 of moved", "offset left of moved"]
    fig, ax = plt.subplots(figsize=(PS.SINGLE_COLUMN_IN * 1.5, 2.6))
    w = 0.38
    wod_v = [np.nanmean([p[k] for p in pk if not p["tag"].startswith("navtest")]) for k in keys]
    nav_v = [np.nanmean([p[k] for p in pk if p["tag"].startswith("navtest")]) for k in keys]
    ax.bar(np.arange(len(keys)) - w / 2, nav_v, w, color=PS.BASELINE, label="navtest (gate B)")
    ax.bar(np.arange(len(keys)) + w / 2, wod_v, w, color=PS.PALETTE["blue"], label="WOD val (gate B)")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("share of moved plans")
    ax.legend()
    fig.tight_layout()
    PS.save(fig, FIGD / "turn_selector_wod" / "pick_marginals")
    plt.close(fig)
