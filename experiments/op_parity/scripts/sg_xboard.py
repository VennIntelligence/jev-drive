"""Stop gate across boards (plans/2026-10-08-stop-gate-xboard-prereg.md, results/stop_gate_xboard.md). Box, jevdrive env (.venv), CPU.

  check    plan-level checks before scoring: SG plans equal the stored baseline plans on every token the gate does not touch;
           touched-token counts; mean-bias norms          -> results/stop_gate_xboard/check.{csv,md}
  navtest  EPDMS / sub-scores, paired cluster bootstrap vs BASE, strata                 -> navtest_*.{csv,md}
  hugsim   HD, launch stall / stuck / spin / collision, paired bootstrap, gate flips, speed traces  -> hugsim_*.{csv,md}, figs/

BASE = stored P2H10-F-s0 / s1 runs (navtest: legacy devkit CSV of the same harness; HUGSIM: stored spec_plan_smooth rr1 / rr2 repeats),
SG / DN = bench runs `P2H10-F-s*@warp:sg|dn` (navtest) and `P2H10-F-s*_spec_plan_smooth-c<hash>` with parity opts stop_gate / sub_bias.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive.bench.models import data_dir  # noqa: E402
from jevdrive.bench.navsim import SUBS, read_devkit_csv  # noqa: E402
from jevdrive.stats import paired  # noqa: E402

OUT = _R / "experiments/op_parity/results/stop_gate_xboard"
D = data_dir() / "runs"
SEEDS = (0, 1)
GATE = 0.5
f3 = lambda r, s=1.0, p=2: f"{r['mean'] * s:+.{p}f} [{r['lo'] * s:+.{p}f}, {r['hi'] * s:+.{p}f}]"  # noqa: E731


def md(df, name, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(df.to_markdown(index=False) + ("\n\n" + note if note else "") + "\n")
    print(f"\n== {name}\n" + df.to_markdown(index=False))


# ---------------------------------------------------------------- navtest
def nav_base(s):
    import glob
    fs = sorted(glob.glob(str(D / f"navsim/eval/v2_navtest_opi_lb_navtest_warp-cinque_PPP2H10-F-s{s}__base/*/*.csv")))
    t, _ = read_devkit_csv(fs[-1])
    return t.rename(columns={v: k for k, v in SUBS.items()}) if "NC" not in t else t


def nav_arm(s, opt):
    u = pd.read_csv(D / f"bench/navtest/P2H10-F-s{s}@warp_{opt}/units.csv").set_index("token")
    return u


def nav_frame(opt):
    """token-indexed frame: per arm the two-seed mean of every score column (score, NC .. EC), arm in {BASE, sg, dn}."""
    toks = pd.read_parquet(D / "op_probe/joint/navtest_tokens.parquet").set_index("token")
    out = {}
    for arm in ("BASE", "sg", "dn"):
        fr = []
        for s in SEEDS:
            if arm == "BASE":
                t = nav_base(s)
                t = pd.DataFrame({"score": t["score"], **{k: t[v] for k, v in SUBS.items()}})
            else:
                t = nav_arm(s, arm)[["score", *SUBS]]
            fr.append(t.reindex(toks.index))
        out[arm] = (fr[0] + fr[1]) / 2, fr
    return toks, out


def navtest(a):
    toks, arms = nav_frame(None)
    tab = np.load(D / "op_parity/cache/lb_navtest/tab.npz", allow_pickle=True)
    assert tab["names"].tolist() == toks.index.tolist()
    fed = pd.Series(tab["ego"][:, 4] * 10.0, index=toks.index)
    touched = fed < GATE
    g = toks.log.to_numpy()
    strata = [("all", np.ones(len(toks), bool)),
              ("v0 < 0.5 m/s", (toks.v0 < GATE).to_numpy()),
              ("launch (v0 < 2, logged 4 s > 5 m)", ((toks.v0 < 2) & (toks.path_len > 5)).to_numpy()),
              ("moving (v0 >= 0.5)", (toks.v0 >= GATE).to_numpy()),
              ("turn > 20 deg", (toks.dyaw.abs() > 20).to_numpy()),
              ("gate touched (fed vx < 0.5)", touched.to_numpy())]
    rows, per = [], []
    for lab, m in strata:
        r = {"stratum": lab, "n": int(m.sum())}
        for arm in ("sg", "dn"):
            d = paired(100 * arms[arm][0].score[m], 100 * arms["BASE"][0].score[m], g[m])
            r |= {"BASE": round(d["mean_b"], 2), arm.upper(): round(d["mean_a"], 2), f"{arm.upper()} - BASE": f3(d)}
            for s in SEEDS:
                ds = paired(100 * arms[arm][1][s].score[m], 100 * arms["BASE"][1][s].score[m], g[m])
                r[f"{arm.upper()} - BASE s{s}"] = f"{ds['mean']:+.2f}"
        rows.append(r)
    md(pd.DataFrame(rows), "navtest_strata")
    # sub-scores, all tokens
    rows = []
    for k in ["score", *SUBS]:
        r = {"metric": "EPDMS" if k == "score" else k}
        for arm in ("BASE", "sg", "dn"):
            r[arm.upper()] = round(100 * float(np.nanmean(arms[arm][0][k])), 2)
        for arm in ("sg", "dn"):
            d = paired(100 * arms[arm][0][k], 100 * arms["BASE"][0][k], g)
            r[f"{arm.upper()} - BASE"] = f3(d, p=3)
        rows.append(r)
    md(pd.DataFrame(rows), "navtest_subscores")
    # tokens touched
    pl = {a: [np.load(D / f"bench/ol/lb_navtest/plans/P2H10-F-s{s}@warp_{a}.npz", allow_pickle=True)["plan_pos"] for s in SEEDS] for a in ("sg", "dn")}
    bl = [np.load(D / f"op_lb/lb_navtest/plans/warp@cinque_PPP2H10-F-s{s}.npz", allow_pickle=True)["plan_pos"] for s in SEEDS]
    diff = {a: [np.abs(pl[a][i] - bl[i]).reshape(len(toks), -1).max(1) > 1e-4 for i in range(2)] for a in pl}
    v0 = toks.v0.to_numpy()
    st = pd.DataFrame([{"quantity": "tokens fed vx < 0.5 m/s (gate fires)", "n": int(touched.sum()), "share %": round(100 * touched.mean(), 2)},
                       {"quantity": "  of which launch tokens", "n": int((touched & strata[2][1]).sum()), "share %": round(100 * (touched & strata[2][1]).mean(), 2)},
                       {"quantity": "  of which logged 4 s distance <= 5 m (stay)", "n": int((touched & ~strata[2][1]).sum()), "share %": round(100 * (touched & ~strata[2][1]).mean(), 2)},
                       {"quantity": "launch tokens (v0 < 2, logged 4 s > 5 m), all", "n": int(strata[2][1].sum()), "share %": round(100 * strata[2][1].mean(), 2)},
                       *[{"quantity": f"SG plan differs from BASE (max |dxy| > 1e-4 m), seed {s}", "n": int(diff['sg'][s].sum()), "share %": round(100 * diff['sg'][s].mean(), 2)} for s in SEEDS],
                       *[{"quantity": f"DN plan differs from BASE, seed {s}", "n": int(diff['dn'][s].sum()), "share %": round(100 * diff['dn'][s].mean(), 2)} for s in SEEDS],
                       *[{"quantity": f"SG plans identical to BASE on untouched tokens, seed {s} (check)", "n": int((~touched.to_numpy() & diff['sg'][s]).sum()), "share %": 0.0} for s in SEEDS]])
    md(st, "navtest_touched", "The last two rows count untouched tokens whose SG plan differs from the stored baseline plan: 0 means BASE reuse is exact.")
    # EPDMS by arm with CI over all tokens for the arm mean itself
    rows = [{"arm": a.upper(), "EPDMS": round(100 * arms[a][0].score.mean(), 2), "s0": round(100 * arms[a][1][0].score.mean(), 2), "s1": round(100 * arms[a][1][1].score.mean(), 2),
             "DAC fail %": round(100 * float((arms[a][0].DAC < 1 - 1e-9).mean()), 2)} for a in ("BASE", "sg", "dn")]
    md(pd.DataFrame(rows), "navtest_arms")


# ---------------------------------------------------------------- HUGSIM
def hs_runs(arm_dirs):
    return [pd.read_csv(D / "bench/hugsim" / d / "units.csv").set_index("scenario") for d in arm_dirs]


def find_run(seed, kind):
    import glob
    ds = sorted(p.parent.name for p in (D / "bench/hugsim").glob(f"P2H10-F-s{seed}_spec_plan_smooth-c*/config.json"))
    for d in ds:
        o = json.loads((D / "bench/hugsim" / d / "config.json").read_text())["opts"].get("parity", {})
        if (kind == "sg" and o.get("stop_gate") == GATE) or (kind == "dn" and o.get("sub_bias")):
            return d
    raise SystemExit(f"no {kind} run for seed {seed}")


def flips(run, scen_dir):
    z = scen_dir / "zs_steps.jsonl"
    g, v, hd = [], [], []
    for line in z.read_text().splitlines():
        r = json.loads(line)
        if "step" in r and r.get("parity"):
            g.append(bool(r["parity"].get("gated", False)))
            v.append(r["parity"]["ego"][4] * 10.0)
    g = np.array(g, int)
    fl = np.flatnonzero(np.diff(g) != 0)
    win = max([int(((fl >= i) & (fl < i + 20)).sum()) for i in range(0, max(len(g) - 1, 1))] or [0])
    return dict(steps=len(g), gated_steps=int(g.sum()), flips=len(fl), max_flips_20=win, v_min=float(np.min(v)) if len(v) else np.nan), np.array(v), g


def hugsim(a):
    sc = {"base": {s: hs_runs([f"P2H10-F-s{s}_spec_plan_smooth-rr1", f"P2H10-F-s{s}_spec_plan_smooth-rr2"]) for s in SEEDS},
          "sg": {s: hs_runs([find_run(s, "sg")])[0] for s in SEEDS}, "dn": {s: hs_runs([find_run(s, "dn")])[0] for s in SEEDS}}
    idx = sc["sg"][0].index
    def hd(arm, s):
        if arm == "base":
            return np.mean([u.reindex(idx).hdscore for u in sc["base"][s]], 0)
        return sc[arm][s].reindex(idx).hdscore.to_numpy(float)
    def arm_mean(arm):
        return np.mean([hd(arm, s) for s in SEEDS], 0)
    rows = []
    for arm in ("sg", "dn"):
        d = paired(arm_mean(arm), arm_mean("base"))
        rows.append({"contrast": f"{arm.upper()} - BASE (seed mean, 64 scenarios)", "HD BASE": round(d["mean_b"], 3), f"HD {arm.upper()}": round(d["mean_a"], 3), "diff [95% CI]": f3(d, p=3),
                     "s0": f"{np.mean(hd(arm, 0) - hd('base', 0)):+.3f}", "s1": f"{np.mean(hd(arm, 1) - hd('base', 1)):+.3f}",
                     "wins / losses / ties (|d| < 0.02)": f"{int((arm_mean(arm) - arm_mean('base') > .02).sum())} / {int((arm_mean(arm) - arm_mean('base') < -.02).sum())} / {int((abs(arm_mean(arm) - arm_mean('base')) <= .02).sum())}"})
    md(pd.DataFrame(rows), "hugsim_hd", f"Noise reference: stored BASE rr1 vs rr2, mean per-scenario HD difference per seed: s0 {np.mean(sc['base'][0][0].reindex(idx).hdscore - sc['base'][0][1].reindex(idx).hdscore):+.4f} s1 {np.mean(sc['base'][1][0].reindex(idx).hdscore - sc['base'][1][1].reindex(idx).hdscore):+.4f}; mean |d| per scenario s0 {np.mean(abs(sc['base'][0][0].reindex(idx).hdscore - sc['base'][0][1].reindex(idx).hdscore)):.3f}.")
    # classes (per-arm means over the two seeds; BASE: mean over seeds and repeats)
    def cls(u):
        u = u.reindex(idx)
        return {"launch stall": int(u.launch_stall.astype(bool).sum()), "stuck": int(u.stuck.astype(bool).sum()), "spin": int(u.spin.astype(bool).sum()),
                "fg collision": int((u.cls == "fg_coll").sum()), "bg collision": int((u.cls == "bg_coll").sum()), "off route": int((u.cls == "off_route").sum()),
                "complete": int((u.cls == "complete").sum())}
    rows = []
    for arm in ("base", "sg", "dn"):
        cs = [cls(u) for s in SEEDS for u in (sc["base"][s] if arm == "base" else [sc[arm][s]])]
        r = {"arm": arm.upper()} | {k: round(float(np.mean([c[k] for c in cs])), 2) for k in cs[0]}
        r["per seed (stall / stuck / spin)"] = ", ".join(f"{cs[i]['launch stall']}/{cs[i]['stuck']}/{cs[i]['spin']}" for i in range(len(cs)))
        rows.append(r)
    md(pd.DataFrame(rows), "hugsim_classes", "Counts over 64 scenarios, mean over seeds (BASE: over seeds and the two repeats; per-run values in the last column, BASE runs in the order s0 rr1, s0 rr2, s1 rr1, s1 rr2).")
    # per-scenario sg flips
    rows, tr = [], {}
    for s in SEEDS:
        d = find_run(s, "sg")
        for scn in idx:
            row = sc["sg"][s].loc[scn]
            sd = _pl.Path(row.run_dir.replace("/root/autodl-tmp/ujs/runs", str(D)))
            st, v, g = flips(None, sd)
            b = np.mean([u.reindex(idx).loc[scn, "hdscore"] for u in sc["base"][s]])
            rows.append({"seed": s, "scenario": scn, **st, "HD SG": round(float(row.hdscore), 3), "HD BASE": round(float(b), 3), "dHD": round(float(row.hdscore - b), 3)})
            tr[(s, scn)] = (v, g)
    fl = pd.DataFrame(rows)
    fl["oscillates"] = (fl.flips >= 6) | (fl.max_flips_20 >= 3)
    fl.to_csv(OUT / "hugsim_gate_flips.csv", index=False)
    summ = pd.DataFrame([{"seed": s, "scenarios with any gated step": int((fl[fl.seed == s].gated_steps > 0).sum()), "gated steps (total)": int(fl[fl.seed == s].gated_steps.sum()),
                          "scenarios with >= 1 flip": int((fl[fl.seed == s].flips > 0).sum()), "max flips in a scenario": int(fl[fl.seed == s].flips.max()),
                          "oscillating scenarios (>= 6 flips or >= 3 in 20 steps)": int(fl[fl.seed == s].oscillates.sum())} for s in SEEDS])
    md(summ, "hugsim_gate_summary")
    top = fl.sort_values(["flips", "gated_steps"], ascending=False).head(12)
    md(top, "hugsim_gate_top", "Scenarios with the most gate flips (both seeds).")
    # figure: speed traces
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = top.head(6)[["seed", "scenario"]].values.tolist()
    ls = fl[(fl.gated_steps > 0)].sort_values("dHD").head(2)[["seed", "scenario"]].values.tolist()
    pick = [tuple(p) for p in pick]
    for p in ls:
        if tuple(p) not in pick:
            pick.append(tuple(p))
    pick = pick[:8]
    if pick:
        fig, ax = plt.subplots(len(pick), 1, figsize=(9, 1.9 * len(pick)), sharex=False)
        ax = np.atleast_1d(ax)
        for a_, (s, scn) in zip(ax, pick):
            v, g = tr[(int(s), scn)]
            a_.plot(v, lw=1.2, color="#1f6feb")
            a_.axhline(GATE, color="grey", lw=0.8, ls="--")
            a_.fill_between(range(len(g)), 0, 1, where=g > 0, transform=a_.get_xaxis_transform(), color="#e5534b", alpha=0.25, lw=0)
            r = fl[(fl.seed == s) & (fl.scenario == scn)].iloc[0]
            a_.set_ylabel("fed vx m/s", fontsize=8)
            a_.set_title(f"s{s} {scn}: gate flips {r.flips}, gated steps {r.gated_steps}, HD {r['HD SG']} (BASE {r['HD BASE']})", fontsize=8)
        ax[-1].set_xlabel("simulator step (0.25 s)")
        fig.tight_layout()
        (OUT / "figs").mkdir(exist_ok=True)
        fig.savefig(OUT / "figs/gate_speed_traces.png", dpi=110)


# ---------------------------------------------------------------- check
def check(a):
    tab = np.load(D / "op_parity/cache/lb_navtest/tab.npz", allow_pickle=True)
    touched = tab["ego"][:, 4] * 10.0 < GATE
    rows = []
    for s in SEEDS:
        bl = np.load(D / f"op_lb/lb_navtest/plans/warp@cinque_PPP2H10-F-s{s}.npz", allow_pickle=True)
        for opt in ("sg", "dn"):
            p = np.load(D / f"bench/ol/lb_navtest/plans/P2H10-F-s{s}@warp_{opt}.npz", allow_pickle=True)
            assert p["names"].tolist() == bl["names"].tolist()
            d = np.abs(p["plan_pos"] - bl["plan_pos"]).reshape(len(touched), -1).max(1)
            rows.append({"seed": s, "arm": opt, "tokens differing": int((d > 1e-4).sum()), "differing among untouched": int(((d > 1e-4) & ~touched).sum()),
                         "max |dxy| untouched (m)": float(d[~touched].max()), "mean |dxy| touched (m)": float(d[touched].mean()), "touched": int(touched.sum())})
    mb = {s: np.load(D / f"bench/sg/meanbias-P2H10-F-s{s}.npy") for s in SEEDS}
    md(pd.DataFrame(rows), "check", "mean-bias rms: " + ", ".join(f"s{s} {float(np.sqrt((mb[s] ** 2).mean())):.3f}" for s in SEEDS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "navtest", "hugsim"])
    a = ap.parse_args()
    {"check": check, "navtest": navtest, "hugsim": hugsim}[a.cmd](a)
