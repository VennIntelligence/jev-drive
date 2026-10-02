"""Final closed-loop report of the vlm_arb lane: arm table, paired differences to `drive`, registered lines.

  python vlm_arb_report.py            # writes $DATA_DIR/runs/vlm_arb/results/{report.md, routes.csv, paired.csv, paired_ds.png}

Units: v2-<arm>-s<seed>-<shard> of the chain, plus the first executor's finished drive runs (all traffic seed 0):
eval-drive-s0 is the seed-0 reference where it has the route, the other three are identical repeats used only for
the repeat-noise read. Pairs are (route, seed); intervals resample routes (2000 resamples, seed 0).
The batch is 19 routes x 2 seeds: every number is a diagnostic read. Registered confirmation lines are printed with
their value and marked "not evaluated at this size"; none is reported as passed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import (INFRACTIONS, OBS_ROUTES, LIGHT_ROUTES, OLD_DRIVE, ROUTES, RUN, SEEDS, SIGN_ROUTES, TARGET,  # noqa: E402
                            boot_mean, drive_dir, fmt, route_row, unit_dir)

ARMS = ["drive", "jslow", "dslow", "pred", "pbyp", "vbyp", "vred", "vall"]
MIN_ROUTES = 30          # below this many routes a registered confirmation line is not evaluated (plan D7)


def collect():
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            for rid in ROUTES:
                d = drive_dir(rid, seed) if arm == "drive" else unit_dir(arm, seed, "tgt" if rid not in ROUTES[:10] else "dev")
                r = route_row(d, rid)
                if r is not None:
                    rows.append(dict(arm=arm, seed=seed, unit=d.name, **r))
    return pd.DataFrame(rows)


def repeats():
    rows = []
    for k, u in enumerate(OLD_DRIVE):
        for rid in ROUTES:
            r = route_row(RUN / "arms" / u, rid)
            if r is not None:
                rows.append(dict(rep=k, unit=u, **r))
    return pd.DataFrame(rows)


def paired(df, a, b, routes, col="DS"):
    """Per-route mean over seeds of (a - b) on (route, seed) pairs both arms finished without a program crash."""
    A = df[(df.arm == a) & ~df.crash].set_index(["route", "seed"])
    B = df[(df.arm == b) & ~df.crash].set_index(["route", "seed"])
    idx = [i for i in A.index.intersection(B.index) if i[0] in routes]
    if not idx:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, groups=0, pairs=0)
    d = (A.loc[idx, col] - B.loc[idx, col]).groupby(level="route").mean()
    return dict(boot_mean(d.to_numpy()), pairs=len(idx), mean_a=float(A.loc[idx, col].mean()), mean_b=float(B.loc[idx, col].mean()),
                **{"d_" + k: int(A.loc[idx, k].sum() - B.loc[idx, k].sum()) for k in ("collisions", "red_light", "stop_infraction", "vehicle_blocked")})


def main():
    out = RUN / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(out / "routes.csv", index=False)
    arms = [a for a in ARMS if (df.arm == a).any()]
    doc = ["# vlm_arb closed loop: diagnostic batch (19 routes x 2 traffic seeds)", "",
           "Every number here is a diagnostic read on a small batch, not a confirmation. Registered confirmation lines "
           "are listed with their value and marked not evaluated at this size.", ""]
    g = df.groupby("arm")
    T = g.agg(runs=("route", "count"), crashes=("crash", "sum"), DS=("DS", "mean"), RC=("RC", "mean"), red_light=("red_light", "sum"),
              stop_sign=("stop_infraction", "sum"), collisions=("collisions", "sum"), blocked=("vehicle_blocked", "sum"),
              timeouts=("route_timeout", "sum"), scenario_timeouts=("scenario_timeouts", "sum"), v_mean=("v_mean", "mean")).reindex(arms)
    doc += ["## Arms", "", T.round(2).to_markdown(), "",
            "Expected runs per arm: %d. Official infraction counts (Bench2Drive results.json); `crashes` are program "
            "crashes by official status, excluded from the paired reads." % (len(ROUTES) * len(SEEDS)), ""]
    # paired reads
    P = []
    for a in arms[1:]:
        tgt = TARGET[a]
        non = [r for r in ROUTES if r not in tgt]
        for name, routes in (("all", ROUTES), ("target", tgt), ("non-target", non)):
            if routes:
                P.append(dict(contrast=a + " - drive", routes=name, **paired(df, a, "drive", routes)))
        for name, routes in (("target RC", tgt),):
            P.append(dict(contrast=a + " - drive", routes=name, **paired(df, a, "drive", routes, "RC")))
    if "dslow" in arms and "jslow" in arms:
        P.append(dict(contrast="jslow - dslow", routes="all", **paired(df, "jslow", "dslow", ROUTES)))
    P = pd.DataFrame(P)
    P.to_csv(out / "paired.csv", index=False)
    doc += ["## Paired differences (DS unless the set says RC), mean over routes [95% route-cluster CI]", "",
            "| contrast | routes | difference | routes n | pairs | d collisions | d red light | d stop sign | d blocked |",
            "|:--|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for r in P.to_dict("records"):
        doc.append("| %s | %s | %s | %d | %d | %s | %s | %s | %s |" % (
            r["contrast"], r["routes"], fmt(r), r["groups"], r["pairs"], *[("%+d" % r[k]) if r["groups"] and np.isfinite(r.get(k, np.nan)) else ""
                                                                         for k in ("d_collisions", "d_red_light", "d_stop_infraction", "d_vehicle_blocked")]))
    # speed match
    if "dslow" in arms and "jslow" in arms:
        s = df[df.arm.isin(["jslow", "dslow", "drive"])].pivot_table(index=["route", "seed"], columns="arm", values="v_mean").dropna()
        doc += ["", "Speed match: mean speed drive %.2f, jslow %.2f, dslow %.2f m/s; per run |dslow - jslow| mean %.2f m/s (%d runs)." % (
            s.drive.mean(), s.jslow.mean(), s.dslow.mean(), (s.dslow - s.jslow).abs().mean(), len(s))]
    # retention
    doc += ["", "## Retention of the privileged gain (target routes)", ""]
    for v, p, col in (("vbyp", "pbyp", "RC"), ("vbyp", "pbyp", "DS"), ("vred", "pred", "DS")):
        if v in arms and p in arms:
            a, b = paired(df, v, "drive", TARGET[p], col), paired(df, p, "drive", TARGET[p], col)
            ratio = a["est"] / b["est"] if b["groups"] and abs(b["est"]) > 1e-9 else np.nan
            doc.append("- %s / %s, %s on %d routes: (%+.2f) / (%+.2f) = %s" % (v, p, col, b["groups"], a["est"], b["est"],
                                                                         "n/a" if not np.isfinite(ratio) else "%.2f" % ratio))
        else:
            doc.append("- %s / %s, %s: not run (%s)" % (v, p, col, "gate" if v not in arms else "missing"))
    # repeat noise
    rep = repeats()
    if len(rep):
        piv = rep.pivot_table(index="route", columns="rep", values="DS")
        sd = piv.std(axis=1, ddof=1).dropna()
        doc += ["", "## Repeat noise (identical drive runs, traffic seed 0)", "",
                "%d routes with 2-4 identical runs: per-route DS standard deviation mean %.1f, median %.1f, max %.1f; %d of %d routes "
                "changed DS between repeats. A paired difference inside this spread is not a finding." % (
                    len(sd), sd.mean(), sd.median(), sd.max(), int((sd > 1e-6).sum()), len(sd))]
    # VLM arms: what the table did
    V = df[df.arm.isin(["vbyp", "vred", "vall"])]
    if len(V):
        cols = [c for c in ("vlm_n", "vlm_ok", "lat_p95", "lat_over_L", "n_R1", "n_R2", "n_R3", "n_R4", "n_R5") if c in read_readouts().columns]
        R = read_readouts()
        R = R[R.arm.isin(["vbyp", "vred", "vall"])]
        if len(R) and cols:
            doc += ["", "## VLM arms: requests and row activity (0.5 s samples with the row active)", "",
                    R.groupby("arm")[cols].agg({c: ("sum" if c.startswith("n_") or c == "vlm_n" else "mean") for c in cols}).round(3).to_markdown()]
    # registered lines
    doc += ["", "## Registered lines (plan 4.3)", "", "| line | read | status |", "|:--|:--|:--|"]
    def status(n):
        return "not evaluated at this size (%d routes < %d)" % (n, MIN_ROUTES) if n < MIN_ROUTES else "evaluated"
    for a in arms[1:]:
        non = [r for r in ROUTES if r not in TARGET[a]]
        if non and a not in ("jslow", "dslow"):
            r = paired(df, a, "drive", non)
            doc.append("| harmless, %s on non-target routes: CI lower bound >= -5, no new blocked, no new collision | %s; blocked %+d, collisions %+d | %s |" % (
                a, fmt(r), r.get("d_vehicle_blocked", 0), r.get("d_collisions", 0), status(r["groups"])))
    for v in ("vred", "vbyp"):
        if v in arms:
            r = paired(df, v, "drive", TARGET[v], "DS" if v == "vred" else "RC")
            doc.append("| useful, %s on target routes (%s) | %s; red light %+d | %s |" % (v, "DS, CI lower bound > 0" if v == "vred" else "RC up, bypass completed",
                                                                                  fmt(r), r.get("d_red_light", 0), status(r["groups"])))
        else:
            doc.append("| useful, %s | not run: its Phase A question did not pass | not evaluated |" % v)
    if "jslow" in arms:
        a = paired(df, "jslow", "drive", ROUTES)
        b = paired(df, "jslow", "dslow", ROUTES) if "dslow" in arms else dict(groups=0, est=np.nan)
        doc.append("| jslow separation: vs drive and vs dslow | vs drive %s; vs dslow %s | %s |" % (fmt(a), fmt(b), status(a["groups"])))
    if "vall" in arms:
        c = df[df.arm == "vall"].collisions.sum(), df[df.arm == "drive"].collisions.sum()
        best = max(df[df.arm == x].DS.mean() for x in arms if x in ("jslow", "vred", "vbyp"))
        doc.append("| stacking: vall collisions <= drive, DS >= best single arm | collisions %d vs %d; DS %.1f vs %.1f | %s |" % (
            c[0], c[1], df[df.arm == "vall"].DS.mean(), best, status(len(ROUTES))))
    else:
        doc.append("| stacking, vall | not run: Q-light did not pass | not evaluated |")
    figure(P, out / "paired_ds.png")
    doc += ["", "![paired DS differences](paired_ds.png)", "",
            "Figure: paired DS difference to drive per arm on all, target and non-target routes (dot: mean over routes, bar: 95% "
            "route-cluster CI). Look at whether a bar clears zero and how wide it is against the repeat noise above."]
    for extra in ("phase_a.md", "lightsweep.md"):
        if (out / extra).exists():
            doc += ["", "See also [%s](%s)." % (extra, extra)]
    (out / "report.md").write_text("\n".join(doc) + "\n")
    print("\n".join(doc))


def read_readouts():
    fs = [p for p in (RUN / "readouts").glob("v2-*/routes.csv") if p.stat().st_size > 5]
    return pd.concat([pd.read_csv(p) for p in fs]) if fs else pd.DataFrame(columns=["arm"])


def figure(P, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = P[P.routes.isin(["all", "target", "non-target"]) & (P.groups > 0)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * max(len(P), 3) + 1.2), dpi=150)
    for i, r in P.iterrows():
        y = len(P) - 1 - i
        ax.plot([r.lo, r.hi], [y, y], color="#3b6ea5", lw=2, solid_capstyle="round")
        ax.plot(r.est, y, "o", color="#3b6ea5", ms=6)
    ax.axvline(0, color="#222", lw=1)
    ax.set_yticks(range(len(P)))
    ax.set_yticklabels(["%s, %s (%d routes)" % (r.contrast, r.routes, r.groups) for _, r in P.iterrows()][::-1], fontsize=7)
    ax.set_xlabel("paired DS difference (mean over routes, 95% route-cluster CI)", fontsize=8)
    ax.set_title("Closed loop, diagnostic batch: difference to drive", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee", lw=.6)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
