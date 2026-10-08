"""Closed-loop readout of the turn selector on HUGSIM (plans/2026-10-08-turn-selector-hugsim-prereg.md).
Arms: SH30 = archived SH30-F-s{0,1} (decision 170 runs), tsB = SH30-F-s{0,1}:tsB, optional ts0 = SH30-F-s{0,1}:ts0 (all64 only when it was run).
Writes results/turn_selector_hugsim/*.{md,csv} and figs/turn_selector_hugsim/turn_selector_hugsim.{png,pdf}."""
import json
import pathlib as _pl

import numpy as np
import pandas as pd

import turn_selhug as H
from jevdrive import stats
from jevdrive.bench import sets

SEEDS = (0, 1)


def arm_units(tag):
    return [H.load_units(f"SH30-F-s{s}{tag}") for s in SEEDS]


def seedmean(us, col, idx):
    return sum(u.loc[idx, col].astype(float) for u in us) / len(us)


def count_row(us, idx):
    c = lambda f: float(np.mean([f(u.loc[idx]).sum() for u in us]))  # noqa: E731
    return dict(fg=c(lambda u: u.cls == "fg_coll"), bg=c(lambda u: u.cls == "bg_coll"), off_route=c(lambda u: u.cls == "off_route"),
                stuck=c(lambda u: u.end == "max_steps"), spin=c(lambda u: u.cls == "spin"), launch_stall=c(lambda u: u.launch_stall.astype(bool)),
                complete=c(lambda u: u.cls == "complete"))


def report(a):
    H.RES.mkdir(parents=True, exist_ok=True)
    H.FIG.mkdir(parents=True, exist_ok=True)
    all64 = [_pl.Path(x).stem for x in sets.hugsim_scenarios("all64")]
    turn = [_pl.Path(x).stem for x in sets.hugsim_scenarios("turn23")]
    straight = [s for s in all64 if s not in set(turn)]
    arms = {"SH30": arm_units(""), "tsB": arm_units(":tsB")}
    try:
        arms["ts0"] = arm_units(":ts0")
        have0 = len(arms["ts0"][0]) >= 60
    except Exception:
        have0 = False
    if not have0:
        arms.pop("ts0", None)
    sets_ = {"all64": all64, "turn23": turn, "straight41": straight}
    # ---- HD, paired
    rows, cnt = [], []
    for ref in ["SH30"] + (["ts0"] if "ts0" in arms else []):
        for nm, idx in sets_.items():
            A, B = seedmean(arms["tsB"], "hdscore", idx), seedmean(arms[ref], "hdscore", idx)
            r = stats.paired(A.to_numpy(), B.to_numpy())
            rows.append(dict(contrast=f"tsB - {ref}", set=nm, n=len(idx), tsB=r["mean_a"], ref=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"]))
            per = [stats.paired(arms["tsB"][k].loc[idx, "hdscore"].to_numpy(float), arms[ref][k].loc[idx, "hdscore"].to_numpy(float))["mean"] for k in range(2)]
            rows[-1]["per_seed_diff"] = " / ".join(f"{x:+.3f}" for x in per)
    for nm, idx in sets_.items():
        for lab, us in arms.items():
            cnt.append(dict(set=nm, arm=lab, hd=float(seedmean(us, "hdscore", idx).mean()), **count_row(us, idx)))
    T = stats.write_table(rows, H.RES / "hd_paired", floatfmt=".3f", note="HD-Score per scenario, mean of the two SH30 seeds first, bootstrap over scenarios (B 10000)")
    C = stats.write_table(cnt, H.RES / "failure_counts", floatfmt=".2f", note="seed-mean counts per set; stuck = max_steps end, spin = heading error >= 60 deg, launch stall = peak speed over 40 steps < 1.6 m/s")
    # ---- step statistics
    st = {lab: [H.scen_stats(u.loc[all64]) for u in us] for lab, us in arms.items() if lab in ("tsB", "SH30") or lab == "ts0"}
    gate = []
    for lab in [l for l in ("tsB", "ts0") if l in st]:
        for nm, idx in (("turn23", turn), ("straight41", straight), ("all64", all64)):
            S = [s[k] for s in st[lab] for k in idx]
            g = lambda key: float(sum(x[key] for x in S))  # noqa: E731
            gate.append(dict(arm=lab, set=nm, steps=g("n"), gate_fire=g("n_allowed") / g("n"), applied=g("n_applied") / g("n"), free_pick_moved=g("n_free_moved") / g("n"),
                             moved_given_gate=g("n_applied") / max(g("n_allowed"), 1), switch_given_both_in_gate=g("n_switch_allowed") / max(g("n_pairs_allowed"), 1),
                             switch_all_pairs=g("n_switch_all") / g("n_pairs"), pairs_in_gate=g("n_pairs_allowed")))
    G = stats.write_table(gate, H.RES / "gate_pick", floatfmt=".3f", note="per-step rates pooled over scenarios and both seeds; switch = pick differs between consecutive 0.25 s steps")
    ctl = []
    for nm, idx in (("turn23", turn), ("straight41", straight), ("all64", all64)):
        for met in ("jerk_rms", "dsteer_mean", "dsteer_p95"):
            def per(lab):
                return np.mean([[st[lab][k][s].get(met, np.nan) for s in idx] for k in range(2)], 0)
            r = stats.paired(per("tsB"), per("SH30"))
            ctl.append(dict(set=nm, metric=met, SH30=r["mean_b"], tsB=r["mean_a"], diff=r["mean"], lo=r["lo"], hi=r["hi"]))
    K = stats.write_table(ctl, H.RES / "control_stats", floatfmt=".4f", note="v > 1 m/s steps; jerk = d/dt of v * yaw rate (m/s^3, rms per scenario); dsteer = |step change of ego_steer|; seeds averaged per scenario, bootstrap over scenarios")
    # ---- figure
    import plot_style as ps
    import matplotlib.pyplot as plt
    ps.apply()
    d = (seedmean(arms["tsB"], "hdscore", all64) - seedmean(arms["SH30"], "hdscore", all64))
    fig, ax = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.5), gridspec_kw=dict(width_ratios=[2.2, 1]))
    order = d.sort_values().index
    col = [ps.PALETTE["vermillion"] if s in set(turn) else ps.PALETTE["sky_blue"] for s in order]
    ax[0].bar(range(len(order)), d[order].to_numpy(), color=col, width=0.8)
    ps.zero_line(ax[0])
    ax[0].set_xlabel("scenario (sorted)")
    ax[0].set_ylabel("HD tsB - SH30 (seed mean)")
    ax[0].bar([], [], color=ps.PALETTE["vermillion"], label="turning route")
    ax[0].bar([], [], color=ps.PALETTE["sky_blue"], label="straight route")
    ax[0].legend(loc="upper left")
    ps.panel(ax[0], "(a)")
    gf = {s: sum(x["n_allowed"] for x in (st["tsB"][0][s], st["tsB"][1][s])) / sum(x["n"] for x in (st["tsB"][0][s], st["tsB"][1][s])) for s in all64}
    ax[1].scatter([gf[s] for s in all64], d[all64].to_numpy(), s=9, c=[ps.PALETTE["vermillion"] if s in set(turn) else ps.PALETTE["sky_blue"] for s in all64])
    ps.zero_line(ax[1])
    ax[1].set_xlabel("fraction of steps in gate B")
    ax[1].set_ylabel("HD tsB - SH30")
    ps.panel(ax[1], "(b)")
    fig.tight_layout()
    meta = ps.save(fig, H.FIG / "turn_selector_hugsim")
    (H.RES / "report_meta.json").write_text(json.dumps(dict(fig=meta, per_scenario=d.round(4).to_dict()), indent=1))
    print(T.to_string(), C.to_string(), G.to_string(), K.to_string(), sep="\n")
