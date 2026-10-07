"""Agent-hinge lane (plans/2026-10-07-agent-hinge-prereg.md): gate and tables. Inputs come from jevdrive.bench (units of bench runs, or the lanes'
stored results through bench.tables.load). Paired cluster bootstrap over the navtest logs (jevdrive.stats, B 10 000), per scenario on HUGSIM.

  gate     --new HA-F-s0 --ref HP-F-s0          small read -> $DATA_DIR/runs/op_parity/agent/gate-<new>.json; exit 0 pass, 1 run lambda 3, 2 stop
  navtest  --arm L=spec[+spec] --ref L=spec[+spec] [--out stem]   arms / paired / D3-stratum / turn-bin tables (WA-JEPA added)
  navhard  --arm ... --ref ...                   combined / stage 1 / stage 2, paired by the stage-1 log
  hugsim   --arm ... --ref ... [--ref-repeats r1 r2]   spec_plan_smooth HD on all 64 / turning 23 / fg 31 / D3b 10, fg collision counts; WA-JEPA exam
NC + TTC failure = token with NC < 1 or TTC < 1 (four_dirs D3); D3 strata = the P2H10 D3 tokens of results/four_dirs/nav_tokens_navtest.csv.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive import stats  # noqa: E402
from jevdrive.bench import tables as T  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

RUN = data_dir() / "runs/op_parity/agent"
RES = _R / "experiments/op_parity/results/agent_hinge"
FD = _R / "experiments/op_parity/results/four_dirs"
G_NCTTC, G_EPDMS, G_EP = -0.3, 0.2, -0.2


def arm(s):
    lab, specs = T.parse_arm(s)
    return lab, specs


def nav_units(specs, bench="navtest"):
    us = [T.load(bench, s)[0] for s in specs]
    if any(u is None for u in us):
        raise SystemExit(f"{bench}: missing result for one of {specs}")
    return us


def nav_metrics(u):
    """per-token metric frame (x 100): EPDMS, NC+TTC fail, NC fail, TTC fail, EP, DAC fail."""
    return pd.DataFrame({"EPDMS": 100 * u.score, "NC+TTC fail %": 100.0 * ((u.NC < 1) | (u.TTC < 1)), "NC fail %": 100.0 * (u.NC < 1),
                         "TTC fail %": 100.0 * (u.TTC < 1), "EP": 100 * u.EP, "DAC fail %": 100.0 * (u.DAC < 1)}, index=u.index)


def seed_mean(specs, bench="navtest"):
    us = nav_units(specs, bench)
    common = sorted(set.intersection(*[set(u.index) for u in us]))
    m = sum(nav_metrics(u.loc[common]) for u in us) / len(us)
    return m, us[0].loc[common, "log"]


METRICS = ["EPDMS", "NC+TTC fail %", "NC fail %", "TTC fail %", "EP", "DAC fail %"]


def paired_rows(A, B, toks, logs, label, metrics=METRICS):
    r = {"pair": label, "n": len(toks)}
    for k in metrics:
        p = stats.paired(A.loc[toks, k].to_numpy(), B.loc[toks, k].to_numpy(), groups=logs.loc[toks].to_numpy())
        r[k] = f"{p['mean']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}]"
    return r


def cmd_gate(a):
    A, la = seed_mean([a.new])
    B, _ = seed_mean([a.ref])
    toks = sorted(set(A.index) & set(B.index))
    g = la.loc[toks].to_numpy()
    res = {"new": a.new, "ref": a.ref, "n": len(toks)}
    for k in ("EPDMS", "NC+TTC fail %", "EP", "NC fail %", "TTC fail %", "DAC fail %"):
        p = stats.paired(A.loc[toks, k].to_numpy(), B.loc[toks, k].to_numpy(), groups=g)
        res[k] = {"new": p["mean_a"], "ref": p["mean_b"], "diff": p["mean"], "lo": p["lo"], "hi": p["hi"]}
    d_f, d_s, d_ep = res["NC+TTC fail %"]["diff"], res["EPDMS"]["diff"], res["EP"]["diff"]
    moved = d_f <= G_NCTTC
    if moved and d_s >= G_EPDMS and d_ep >= G_EP:
        v = "pass"
    elif moved and d_ep < G_EP:
        v = "lambda3"
    else:
        v = "stop"
    res |= {"ncttc_moved": bool(moved), "epdms_ok": bool(d_s >= G_EPDMS), "ep_ok": bool(d_ep >= G_EP), "verdict": v,
            "rule": "pass: NC+TTC fail diff <= -0.3 pp and EPDMS >= +0.2 and EP >= -0.2; lambda3: NC+TTC moved but EP < -0.2; else stop (prereg + deviation 5)"}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / f"gate-{a.new}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit({"pass": 0, "lambda3": 1, "stop": 2}[v])


def d3_strata():
    t = pd.read_csv(FD / "nav_tokens_navtest.csv")
    d = t[t.bucket.str.startswith("D3") & (t.arm == "P2H")]
    out = [("P2H10 D3 tokens (any)", sorted(set(d.token)))]
    for c in ["stopped vehicle ahead", "static object", "lead vehicle", "cut-in", "oncoming", "crossing / turn conflict", "VRU", "sideswipe (ego across lanes)"]:
        out.append((f"D3 {c}", sorted(set(d.token[d.cls == c]))))
    out.append(("D3 plan > 1.1 x logged speed", sorted(set(d.token[d.spd > 1.1]))))
    return out


def turn_bins(toks):
    fz = np.load(data_dir() / "runs/navsim_zs/index/navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    dy = np.array([abs(np.degrees(np.unwrap(fut[k][:, 2])[-1])) if k in fut else np.nan for k in toks])
    s = pd.Series(dy, index=toks)
    return [(f"|heading change| {lo}-{hi} deg" if hi < 400 else f"|heading change| > {lo} deg", s.index[(s >= lo) & (s < hi)].tolist())
            for lo, hi in ((0, 5), (5, 20), (20, 45), (45, 999))]


def cmd_navtest(a):
    la, sa = arm(a.arm)
    lr, sr = arm(a.ref)
    A, logs = seed_mean(sa)
    B, _ = seed_mean(sr)
    W, _ = seed_mean(["WA-JEPA"])
    toks = sorted(set(A.index) & set(B.index) & set(W.index))
    rows = []
    for lab, X, sp in ((lr, B, sr), (la, A, sa), ("WA-JEPA", W, ["WA-JEPA"])):
        r = {"arm": lab, "seeds": "+".join(sp), "n": len(toks), **{k: X.loc[toks, k].mean() for k in METRICS}}
        if len(sp) > 1:
            per = [nav_metrics(u).reindex(toks) for u in nav_units(sp)]
            r["EPDMS per seed"] = " / ".join(f"{p.EPDMS.mean():.2f}" for p in per)
            r["NC+TTC per seed"] = " / ".join(f"{p['NC+TTC fail %'].mean():.2f}" for p in per)
        rows.append(r)
    stem = RES / a.out
    stats.write_table(rows, f"{stem}_arms", floatfmt=".2f", note="navtest, protocol W, devkit v2; seed means per token; fail % = share of tokens")
    pr = [paired_rows(A, B, toks, logs, f"{la} - {lr}"), paired_rows(A, W, toks, logs, f"{la} - WA-JEPA"), paired_rows(B, W, toks, logs, f"{lr} - WA-JEPA")]
    if len(sa) == len(sr) > 1:                                       # matched seeds (same row stream)
        for x, y in zip(sa, sr):
            X, _ = seed_mean([x])
            Y, _ = seed_mean([y])
            pr.append(paired_rows(X, Y, toks, logs, f"{x} - {y}"))
    stats.write_table(pr, f"{stem}_paired", floatfmt=".2f", note="paired cluster bootstrap over logs, B 10 000, 95% percentile CI")
    st = []
    for lab, ts in d3_strata() + turn_bins(toks):
        ts = [t for t in ts if t in set(toks)]
        if len(ts) < 5:
            continue
        r = {"stratum": lab, "n": len(ts)}
        for k in ("NC+TTC fail %", "EPDMS"):
            r[f"{k} {lr}"] = B.loc[ts, k].mean()
            r[f"{k} {la}"] = A.loc[ts, k].mean()
            p = stats.paired(A.loc[ts, k].to_numpy(), B.loc[ts, k].to_numpy(), groups=logs.loc[ts].to_numpy())
            r[f"{k} diff"] = f"{p['mean']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}]"
        r["NC+TTC fail % WA"] = W.loc[ts, "NC+TTC fail %"].mean()
        st.append(r)
    stats.write_table(st, f"{stem}_strata", floatfmt=".2f",
                      note="D3 strata: tokens where P2H10 (s0 or s1) failed NC or TTC, classes of results/four_dirs/nav_tokens_navtest.csv (fixed before scoring); "
                           "turn bins: logged 4 s heading change")
    print(open(f"{stem}_arms.md").read(), open(f"{stem}_paired.md").read(), open(f"{stem}_strata.md").read(), sep="\n")


def cmd_navhard(a):
    la, sa = arm(a.arm)
    lr, sr = arm(a.ref)

    def sm(specs):
        us = nav_units(specs, "navhard")
        common = sorted(set.intersection(*[set(u.index) for u in us]))
        return sum(u.loc[common, ["combined", "stage1", "stage2"]].astype(float) for u in us) * 100 / len(us), us[0].loc[common, "log"]
    A, logs = sm(sa)
    B, _ = sm(sr)
    W, _ = sm(["WA-JEPA"])
    g = sorted(set(A.index) & set(B.index) & set(W.index))
    rows = [{"arm": lab, "n": len(g), **{k: X.loc[g, k].mean() for k in ("combined", "stage1", "stage2")}} for lab, X in ((lr, B), (la, A), ("WA-JEPA", W))]
    stem = RES / a.out
    stats.write_table(rows, f"{stem}_arms", floatfmt=".2f", note="navhard two-stage, protocol G, mean over the scene-mapping groups, seed means")
    pr = []
    for lab, X, Y in ((f"{la} - {lr}", A, B), (f"{la} - WA-JEPA", A, W), (f"{lr} - WA-JEPA", B, W)):
        r = {"pair": lab, "n": len(g)}
        for k in ("combined", "stage1", "stage2"):
            p = stats.paired(X.loc[g, k].to_numpy(), Y.loc[g, k].to_numpy(), groups=logs.loc[g].to_numpy())
            r[k] = f"{p['mean']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}]"
        pr.append(r)
    stats.write_table(pr, f"{stem}_paired", floatfmt=".2f", note="paired cluster bootstrap over the stage-1 logs, B 10 000")
    print(open(f"{stem}_arms.md").read(), open(f"{stem}_paired.md").read(), sep="\n")


def hug_units(spec, preset, repeat=""):
    kw = {"repeat": repeat} if repeat else {}
    u, src = T.load("hugsim", spec, preset, **kw)
    return u, src


def cmd_hugsim(a):
    la, sa = arm(a.arm)
    lr, sr = arm(a.ref)
    P = "spec_plan_smooth"

    def hd(specs, repeats=("",)):
        cols, fg, srcs = [], [], []
        for s in specs:
            for rp in repeats:
                u, src = hug_units(s, P, rp)
                if u is None:
                    continue
                cols.append(u.hdscore.astype(float).rename(f"{s}{rp}"))
                fg.append((u.end == "fg_collision").rename(f"{s}{rp}"))
                srcs.append(src)
        H, F = pd.concat(cols, axis=1), pd.concat(fg, axis=1).astype(float)
        return H, F, srcs
    HA, FA, src_a = hd(sa)
    HB, FB, src_b = hd(sr, ("",) + tuple(a.ref_repeats))
    uw, src_w = hug_units("WA-JEPA", "exam")
    HW, FW = uw.hdscore.astype(float), (uw.end == "fg_collision").astype(float)
    ev = pd.read_csv(FD / "hugsim_fg_events.csv")
    ev = ev[ev.arm.str.startswith("P2H10")]
    fg31 = sorted(set(ev.scenario))
    d3b = sorted(set(ev.scenario[ev.fg_type.isin(["lead_stopped", "lead_moving"])]))
    from jevdrive.bench.sets import hugsim_scenarios
    t23 = {_pl.Path(x).stem for x in hugsim_scenarios("turn23")}
    sc = sorted(set(HA.dropna().index) & set(HB.index) & set(HW.index))
    turn = [s for s in sc if s in t23]
    sets = [("all 64", sc), ("turning 23", turn), ("fg 31 (P2H10)", [s for s in sc if s in fg31]), ("D3b 10 (P2H10)", [s for s in sc if s in d3b])]
    a_, b_ = HA.mean(1), HB.mean(1)
    rows = []
    for lab, ss in sets:
        if not ss:
            continue
        r = {"set": lab, "n": len(ss), f"HD {lr}": b_.loc[ss].mean(), f"HD {la}": a_.loc[ss].mean(), "HD WA-JEPA (exam)": HW.loc[ss].mean()}
        p = stats.paired(a_.loc[ss].to_numpy(), b_.loc[ss].to_numpy())
        r[f"{la} - {lr}"] = f"{p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"
        p = stats.paired(a_.loc[ss].to_numpy(), HW.loc[ss].to_numpy())
        r[f"{la} - WA-JEPA"] = f"{p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"
        r[f"fg runs {lr} (per run)"] = FB.loc[ss].sum(0).mean()
        r[f"fg runs {la} (per seed)"] = FA.loc[ss].sum(0).mean()
        r["fg WA-JEPA"] = FW.loc[ss].sum()
        rows.append(r)
    stem = RES / a.out
    stats.write_table(rows, stem, floatfmt=".3f",
                      note=f"HUGSIM {P}, HD per scenario = mean over seeds (and stored repeats for the reference: {', '.join(HB.columns)}); "
                           f"scenario bootstrap B 10 000; fg = runs ending in a foreground (vehicle) collision, mean count per seed / repeat. "
                           f"fg 31 / D3b 10 from results/four_dirs/hugsim_fg_events.csv. Sources: {sorted(set(src_a))}; ref {sorted(set(src_b))}; WA {src_w}")
    per = pd.DataFrame({"scenario": sc, f"HD {lr}": b_.loc[sc].values, f"HD {la}": a_.loc[sc].values, "HD WA": HW.loc[sc].values,
                        "fg31": [s in fg31 for s in sc], "D3b": [s in d3b for s in sc], f"fg {la}": FA.loc[sc].mean(1).values,
                        f"fg {lr}": FB.loc[sc].mean(1).values})
    per.to_csv(f"{stem}_scenarios.csv", index=False, float_format="%.4f")
    print(open(f"{stem}.md").read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--new", required=True)
    p.add_argument("--ref", default="HP-F-s0")
    for c in ("navtest", "navhard", "hugsim"):
        p = sp.add_parser(c)
        p.add_argument("--arm", required=True)
        p.add_argument("--ref", required=True)
        p.add_argument("--out", default=c)
        if c == "hugsim":
            p.add_argument("--ref-repeats", nargs="*", default=["r1", "r2"])
    a = ap.parse_args()
    {"gate": cmd_gate, "navtest": cmd_navtest, "navhard": cmd_navhard, "hugsim": cmd_hugsim}[a.cmd](a)
