"""Tables and figure of the launch-lean lane (plans/2026-10-04-launch-lean-prereg.md). op-train python on the box:
    python experiments/hugsim/scripts/lean_report.py $DATA_DIR/runs/hugsim-lean $DATA_DIR/runs/op_adapt_H/rate_probe experiments/hugsim/results/lean
Writes <out>/tables.md, <out>/*.csv and experiments/hugsim/figs/launch-lean.{png,pdf}. Signs: + = left everywhere.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research"))
MODELS = ("O", "pilot-s0", "it_dw3-s0")
RATES = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
SPIN10 = [Path(x).stem for x in open(REPO / "experiments/hugsim/scripts/derot_spin10.txt").read().split()]


def boot(x, f=np.mean, n=4000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = [f(x[np.random.default_rng([seed, i]).integers(0, len(x), len(x))]) for i in range(n)]
    return float(f(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def controller_c(exam_csv):
    """Executed yaw change per 0.25 s step against the sent plan's 1 s direction (both deg), base exam, v < 3 m/s."""
    X, Y = [], []
    for r in csv.DictReader(open(exam_csv)):
        if r["tag"] != "cinque-fixed":
            continue
        L = [json.loads(x) for x in open(Path(r["run_dir"]) / "zs_steps.jsonl")][1:]
        th = np.unwrap([q["theta"] for q in L])
        p = np.array([q["plan"][1] for q in L])
        phi = np.arctan2(p[:, 0], np.maximum(p[:, 1], 1e-3))
        mov = np.linalg.norm(np.array([q["plan"][-1] for q in L]), axis=1) > 1.0
        v = np.array([q["v"] for q in L])
        for k in range(len(L) - 1):
            if mov[k] and v[k] < 3 and abs(phi[k]) < np.radians(40):
                X.append(phi[k])
                Y.append(th[k + 1] - th[k])
    X, Y = np.array(X), np.array(Y)
    return float((X * Y).sum() / (X * X).sum()), len(X)


def loop_sim(c, Hgrid, Fgrid, lean=1.0, n=12, win=6):
    """theta_{k+1} = theta_k + c phi_k, phi_k = lean + F(H_k), H_k = theta_k - theta_{k-win} (deg, + left); F odd, interpolated."""
    th = np.zeros(n + win + 1)
    phis = []
    for k in range(win, win + n):
        H = th[k] - th[k - win]
        F = np.sign(H) * np.interp(abs(H), Hgrid, Fgrid, left=0.0)
        phi = lean + F
        phis.append(phi)
        th[k + 1] = th[k] + c * phi
    return np.array(phis)


def main(L, RP, out):
    L, RP, out = Path(L), Path(RP), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md = []
    D = Path(json.load(open(L / "jobs64.json"))[0]["run_dir"]).parents[3]          # runs/hugsim-exam/scored-op

    # ---------------------------------------------------------------- 1a: G by rate (port probe)
    g = pd.read_csv(RP / "g_rate.csv")
    ra = pd.read_csv(RP / "ratio.csv")
    g.to_csv(out / "g_rate.csv", index=False)
    ra.to_csv(out / "g_ratio.csv", index=False)
    md.append("## 1a. Decision-92 probe G by fake yaw rate (port, nav / WOD / CARLA)\n")
    md.append("G (deg, plan heading at 3 s) mean [95% cluster CI]; gain = G / (1.5 w).\n")
    for bn in ("stop", "low", "mid"):
        md.append(f"\n**{bn}**\n\n| domain | model | n | " + " | ".join(f"{r:g} deg/s" for r in RATES) + " |\n|" + "---|" * (3 + len(RATES)))
        for d in ("pnav", "pwod", "pcarla"):
            for m in MODELS:
                s = g[(g.domain == d) & (g.bin == bn) & (g.model == m)].set_index("rate")
                md.append(f"| {d[1:]} | {m} | {int(s.n.iloc[0])} | " + " | ".join(f"{s.G[r]:.2f} [{s.lo[r]:.2f}, {s.hi[r]:.2f}]" for r in RATES) + " |")
    md.append("\nRatio of mean G, adapted / shipped (cluster bootstrap 95% CI):\n")
    for bn in ("stop", "low", "low+stop"):
        md.append(f"\n**{bn}**\n\n| domain | model | " + " | ".join(f"{r:g}" for r in RATES) + " |\n|" + "---|" * (2 + len(RATES)))
        for d in ("pnav", "pwod", "pcarla"):
            for m in MODELS[1:]:
                s = ra[(ra.domain == d) & (ra.bin == bn) & (ra.model == m)].set_index("rate")
                md.append(f"| {d[1:]} | {m} | " + " | ".join(f"{s.ratio[r]:.2f} [{s.lo[r]:.2f}, {s.hi[r]:.2f}]" for r in RATES) + " |")

    # ---------------------------------------------------------------- 1b: G on HUGSIM frames
    R = json.load(open(L / "rate20.json"))
    scen = [k for k in R if not k.startswith("_")]
    rows = []
    for s in scen:
        for m in MODELS:
            for r in RATES:
                p, q = R[s][f"{m}|{r:+g}"], R[s][f"{m}|{-r:+g}"]
                rows.append(dict(scenario=s, spin=s in SPIN10, model=m, rate=r, v=R[s]["v"][R[s]["k"]],
                                 G_phi1=(p["phi1"] - q["phi1"]) / 2, G_psi3=(p["psi3"] - q["psi3"]) / 2, G_lean10=(p["lean10"] - q["lean10"]) / 2))
    hr = pd.DataFrame(rows)
    hr.to_csv(out / "hugsim_rate.csv", index=False)
    md.append(f"\n## 1b. G on HUGSIM frames (step {R['_meta']['k']}, history de-rotated then fake rate w; n = {len(scen)} scenarios)\n")
    md.append("phi1 = direction of the 1 s plan point (tracked by the controller); s = G_phi1 / (1.2 w) = deg of plan direction per deg of "
              "history yaw over the last 6 steps. Mean [95% bootstrap CI over scenarios].\n")
    md.append("| model | readout | " + " | ".join(f"{r:g}" for r in RATES) + " |\n|" + "---|" * (2 + len(RATES)))
    S = {}
    for m in MODELS:
        for col in ("G_phi1", "G_psi3"):
            cells = []
            for r in RATES:
                x = hr[(hr.model == m) & (hr.rate == r)][col].values
                t = boot(x)
                cells.append(f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]")
                if col == "G_phi1":
                    S[(m, r)] = t[0]
            md.append(f"| {m} | {col} | " + " | ".join(cells) + " |")
    md.append("\nRatio adapted / shipped of the mean G_phi1: " + "; ".join(
        f"{m}: " + ", ".join(f"{r:g}: {S[(m, r)] / S[('O', r)]:.2f}" for r in RATES) for m in MODELS[1:]))

    # ---------------------------------------------------------------- loop gain
    c, nc = controller_c(D / "results.csv")
    md.append(f"\n## Loop gain\n\nController transfer c = {c:.3f} deg of executed yaw per step per deg of the sent 1 s plan direction "
              f"(base exam, v < 3 m/s, n = {nc} steps). Linear loop theta_(k+1) = theta_k + c phi_k, phi_k = lean + s H_k, "
              "H_k = theta_k - theta_(k-6): dominant root z of z^6 (z - 1) = c s (z^6 - 1); z > 1 grows.\n")
    md.append("| model | " + " | ".join(f"s({r:g})" for r in RATES) + " | z at s(0.5) | z at s(1) | z at s(2) | z at s(10) | simulated growth, steps 3-8, lean 1 deg |\n|" + "---|" * (6 + len(RATES)))
    sim = {}
    for m in MODELS:
        s_r = [S[(m, r)] / (1.2 * r) for r in RATES]
        zz = []
        for sv in (s_r[0], s_r[1], s_r[2], s_r[-1]):
            roots = np.roots(np.r_[1.0, -(1 + c * sv), np.zeros(5), c * sv])   # z^7 - (1 + cs) z^6 + cs
            roots = roots[np.abs(roots - 1) > 1e-6]
            zz.append(float(np.max(np.abs(roots))))
        Hg = np.r_[0.0, [1.2 * r for r in RATES]]
        Fg = np.r_[0.0, [S[(m, r)] for r in RATES]]
        ph = loop_sim(c, Hg, Fg)
        sim[m] = ph
        gr = float(np.exp(np.mean(np.diff(np.log(np.abs(ph[2:8]) + 1e-9)))))
        md.append(f"| {m} | " + " | ".join(f"{x:.2f}" for x in s_r) + " | " + " | ".join(f"{x:.2f}" for x in zz) + f" | x{gr:.2f} |")

    # ---------------------------------------------------------------- 2: launch lean
    LO = json.load(open(L / "lean_O64.json"))
    sc = [k for k in LO if not k.startswith("_")]
    vars_ = sorted({k.split("|")[1] for k in LO[sc[0]] if "|" in k})
    rows = []
    for s in sc:
        for v in vars_:
            for k, x in enumerate(LO[s][f"O|{v}"]):
                rows.append(dict(scenario=s, spin=s in SPIN10, tc="L" if LO[s]["tc"] == [0, 1] else "R", var=v, step=k,
                                 **{q: x[q] for q in ("lean10", "phi1", "psi3", "y3")}))
    ln = pd.DataFrame(rows)
    ln.to_csv(out / "lean_O64.csv", index=False)
    rep = []
    for s in sc:
        for k in range(3):
            lg, rp = np.array(LO[s]["logged"][k]), np.array(LO[s]["O|base"][k]["pos"])
            rep.append((abs(lg[3, 1] - rp[3, 1]), abs(np.degrees(np.arctan2(-lg[6, 1], max(lg[6, 0], 1e-3))) - LO[s]["O|base"][k]["lean10"])))
    rep = np.array(rep)
    # readout: mean of steps 1-2
    W = ln[ln.step >= 1].groupby(["scenario", "var"])[["lean10", "phi1", "psi3", "y3"]].mean().unstack("var")
    base = W["lean10"]["base"]
    big = base.abs() >= 1.0
    md.append(f"\n## 2. Launch lean (shipped, 64 scenarios, steps 1-2 of the logged frames, history yaw < 0.05 deg)\n")
    md.append(f"Replay vs exam log (video frames are mp4): |lateral at 2.5 s| median {np.median(rep[:, 0]):.2f} m (p90 {np.percentile(rep[:, 0], 90):.2f}), "
              f"|lean10| median {np.median(rep[:, 1]):.1f} deg (p90 {np.percentile(rep[:, 1], 90):.1f}).\n")
    md.append(f"Base lean10 over 64: mean {fmt(boot(base))}; left-hand-traffic scenes (n {int((ln.groupby('scenario').tc.first() == 'L').sum())}) "
              f"{fmt(boot(base[ln.groupby('scenario').tc.first() == 'L']))}, right-hand {fmt(boot(base[ln.groupby('scenario').tc.first() == 'R']))}; "
              f"spin set {fmt(boot(base[[s in SPIN10 for s in base.index]]))}. |lean| >= 1 deg: {int(big.sum())} scenarios.\n")
    md.append("| variant | flips sign (of |base| >= 1) | median |variant| / |base| | median (variant - base) | mean (variant - base) [CI] |\n|---|---|---|---|---|")
    for v in vars_:
        if v == "base":
            continue
        x = W["lean10"][v]
        fl = float(np.mean(np.sign(x[big]) == -np.sign(base[big])))
        md.append(f"| {v} | {fl:.2f} | {np.median(x[big].abs() / base[big].abs()):.2f} | {np.median((x - base)[big]):+.2f} | {fmt(boot((x - base)[big]))} |")
    md.append("\nSame table with psi3 (3 s heading, deg) for |base psi3| >= 0.5:\n\n| variant | flips | median ratio |\n|---|---|---|")
    bp = W["psi3"]["base"]
    bg = bp.abs() >= 0.5
    for v in vars_:
        if v != "base":
            x = W["psi3"][v]
            md.append(f"| {v} | {np.mean(np.sign(x[bg]) == -np.sign(bp[bg])):.2f} (n {int(bg.sum())}) | {np.median(x[bg].abs() / bp[bg].abs()):.2f} |")
    per = pd.DataFrame({"tc": ln.groupby("scenario").tc.first(), "spin": [s in SPIN10 for s in W.index]}, index=W.index)
    for v in ("base", "mirror", "mirror_tc", "tc", "single", "roll", "mask_L", "mask_R", "front"):
        if v in W["lean10"]:
            per[v] = W["lean10"][v].round(2)
    per.to_csv(out / "lean_per_scenario.csv")
    t = per[per.spin].drop(columns="spin")
    md.append("\nSpin set, lean10 (deg, mean of steps 1-2):\n\n| scenario | " + " | ".join(t.columns) + " |\n|" + "---|" * (len(t.columns) + 1))
    md += [f"| {i} | " + " | ".join(str(x) for x in r) + " |" for i, r in t.iterrows()]

    LA = json.load(open(L / "lean_adapt20.json"))
    md.append("\nAdapted models on the 20-scenario subset (lean10, steps 1-2; O from the 64 run):\n\n| scenario | " +
              " | ".join(f"{m} {v}" for m in MODELS for v in ("base", "mirror_tc")) + " |\n|" + "---|" * 7)
    ad = []
    for s in [k for k in LA if not k.startswith("_")]:
        cells = []
        for m in MODELS:
            src = LO if m == "O" else LA
            for v in ("base", "mirror_tc"):
                cells.append(np.mean([x["lean10"] for x in src[s][f"{m}|{v}"][1:]]))
        ad.append([s] + cells)
        md.append(f"| {s} | " + " | ".join(f"{x:+.1f}" for x in cells) + " |")
    ad = pd.DataFrame(ad, columns=["scenario"] + [f"{m}|{v}" for m in MODELS for v in ("base", "mirror_tc")])
    ad.to_csv(out / "lean_adapt20.csv", index=False)
    md.append("\nMean |lean10| (steps 1-2): " + ", ".join(f"{m} {ad[f'{m}|base'].abs().mean():.2f}" for m in MODELS))

    # ---------------------------------------------------------------- 3: loop replay
    RE = json.load(open(L / "replay.json"))
    md.append("\n## 3. Loop replay (logged frames; normal stepping vs de-rotated replay at every step)\n")
    md.append("H = history yaw over the last 6 steps (deg, + left); rot = phi1(normal) - phi1(derot); s = rot / H on steps with 1 <= |H| <= 15.\n")
    md.append("| log | model | steps | median s [IQR] | median phi1 derot (lean, spin side +) | median phi1 normal (spin side +) |\n|---|---|---|---|---|---|")
    rr = []
    for key in [k for k in RE if not k.startswith("_")]:
        th = np.array(RE[key]["theta"])
        n = len(th)
        H = np.array([-np.degrees(th[k] - th[max(0, k - 6)]) for k in range(n)])
        side = np.sign(-np.degrees(th[-1] - th[0])) or 1.0
        for m in MODELS:
            pn = np.array([x["phi1"] for x in RE[key][f"{m}|normal"]])
            pdr = np.array([x["phi1"] for x in RE[key][f"{m}|derot"]])
            sel = (np.abs(H) >= 1) & (np.abs(H) <= 15)
            sk = (pn - pdr)[sel] / H[sel]
            for k in range(n):
                rr.append(dict(log=key, model=m, step=k, H=H[k], phi_normal=pn[k], phi_derot=pdr[k], v=RE[key]["v"][k]))
            q = np.percentile(sk, [25, 50, 75]) if len(sk) else [np.nan] * 3
            md.append(f"| {key} | {m} | {int(sel.sum())} | {q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}] | {np.median(side * pdr[1:]):+.1f} | {np.median(side * pn[1:]):+.1f} |")
    rr = pd.DataFrame(rr)
    rr.to_csv(out / "replay_steps.csv", index=False)
    (out / "tables.md").write_text("# Launch-lean lane tables (generated by lean_report.py)\n\n" + "\n".join(md) + "\n")
    fig(g, hr, sim, ln, W, rr, c, REPO / "experiments/hugsim/figs/launch-lean")
    print("\n".join(md))


def fig(g, hr, sim, ln, W, rr, c, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as P
    P.apply()
    col = {"O": P.BASELINE, "pilot-s0": P.PALETTE["blue"], "it_dw3-s0": P.PALETTE["vermillion"]}
    lab = {"O": "shipped", "pilot-s0": "pilot", "it_dw3-s0": "it_dw3"}
    fig, ax = plt.subplots(1, 4, figsize=(P.DOUBLE_COLUMN_IN, 1.9))
    a = ax[0]
    for m in col:
        s = g[(g.bin == "low+stop") & (g.model == m)].groupby("rate").G.mean()
        a.plot(s.index, s.values / (1.5 * s.index), "o-", color=col[m], ms=2.5, label=lab[m])
        h = hr[hr.model == m].groupby("rate").G_phi1.mean()
        a.plot(h.index, h.values / (1.2 * h.index), "s--", color=col[m], ms=2.5, lw=0.8)
    a.set_xscale("log")
    a.set_xticks(RATES)
    a.set_xticklabels([f"{r:g}" for r in RATES])
    a.set_xlabel("fake yaw rate (deg/s)")
    a.set_ylabel("gain (deg / deg)")
    a.legend(loc="upper right")
    P.panel(a, "(a)")
    a = ax[1]
    for m in col:
        a.plot(np.arange(1, len(sim[m]) + 1), sim[m], "o-", color=col[m], ms=2.5)
    a.set_yscale("log")
    a.set_xlabel("step")
    a.set_ylabel("plan direction (deg)")
    P.panel(a, "(b)")
    a = ax[2]
    b = W["lean10"]["base"]
    for v, mk, cc in (("mirror_tc", "o", P.PALETTE["blue"]), ("tc", "^", P.PALETTE["orange"])):
        a.scatter(b, W["lean10"][v], s=6, marker=mk, color=cc, label=v.replace("_", "+"))
    lim = float(np.nanpercentile(np.abs(b), 95)) * 1.2
    a.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.5)
    a.plot([-lim, lim], [lim, -lim], color="#999999", lw=0.5, ls=":")
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_xlabel("lean, base (deg)")
    a.set_ylabel("lean, variant (deg)")
    a.legend(loc="lower right")
    P.panel(a, "(c)")
    a = ax[3]
    key = sorted(rr.log.unique())[0]
    for k in rr.log.unique():
        if k.startswith("base|") and "0528" in k:
            key = k
    for m in col:
        q = rr[(rr.log == key) & (rr.model == m)]
        a.plot(q.step, q.phi_normal, "-", color=col[m])
        a.plot(q.step, q.phi_derot, ":", color=col[m])
    q = rr[(rr.log == key) & (rr.model == "O")]
    a.plot(q.step, q.H, color=P.PALETTE["black"], lw=0.7, ls="--")
    a.set_xlabel("step (" + key.split("|")[1].replace("scene-", "") + ")")
    a.set_ylabel("deg (+ left)")
    P.panel(a, "(d)")
    fig.tight_layout(pad=0.3)
    P.save(fig, stem)


if __name__ == "__main__":
    main(*sys.argv[1:4])
