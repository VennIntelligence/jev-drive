"""op_parity hinge lane (plans/2026-10-06-hinge-prereg.md): P2 fine-tuned with the footprint drivable-area SDF hinge vs P2 (decision 144 recipe).

  gate     --new P2H10-F-s0 --ref P2-F-s0   small read: navtest EPDMS and DAC-failure paired diff for one seed -> $DATA_DIR/runs/op_parity/hinge/gate-<new>.json
                                            (exit 0 = pass; 1 = EPDMS drop > 0.3 but DAC moved: try the lower lambda; 2 = stop)
  navtest  --hinge P2H10               seed means of P2 / hinge / WA-JEPA (protocol W, the devkit of full.md) -> results/hinge_navtest_{arms,paired}.{md,csv}
  navhard  --hinge P2H10               navhard two-stage under protocol G, P2 / hinge / WA-JEPA -> results/hinge_navhard_{arms,paired}.{md,csv}
HUGSIM goes through pp_hugsim_report.py full (env FULL_TAGS, FULL_OUT=results/hugsim_hinge).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

SUBS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
        "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
OUT = _R / "experiments/op_parity/results"
RUN = data_dir() / "runs/op_parity/hinge"
GATE_EPDMS, GATE_DAC = -0.3, -0.3        # stop if EPDMS diff < -0.3; DAC "moves" if the failure rate falls by >= 0.3 pp


def _navtest(models):
    from jevdrive import navsim_zs as Z
    import pp_eval as E
    E.DATA, E.FRAMES = "lb_navtest", "warp"
    log = {e["token"]: e["log_name"] for e in Z.load_index("navtest", slim=True)}
    got = {}
    for m in models:
        f = E.eval_csv(m)
        if f is None:
            raise SystemExit(f"no navtest score for {m}")
        got[m] = E.read_csv(f)[0]
    return E, log, got


def cmd_gate(a):
    from jevdrive import stats
    _, log, t = _navtest([a.new, a.ref])
    toks = sorted(set(t[a.new].index) & set(t[a.ref].index))
    g = np.array([log[k] for k in toks])
    col = lambda m, c: t[m].loc[toks, c].to_numpy(float)  # noqa: E731
    de = stats.paired(100 * col(a.new, "score"), 100 * col(a.ref, "score"), groups=g)
    fail = lambda m: 100.0 * (col(m, SUBS["DAC"]) < 1)  # noqa: E731
    dd = stats.paired(fail(a.new), fail(a.ref), groups=g)
    res = {"new": a.new, "ref": a.ref, "n": len(toks), "epdms_diff": de["mean"], "epdms_lo": de["lo"], "epdms_hi": de["hi"],
           "epdms_new": de["mean_a"], "epdms_ref": de["mean_b"], "dac_fail_new": dd["mean_a"], "dac_fail_ref": dd["mean_b"],
           "dac_fail_diff_pp": dd["mean"], "dac_lo": dd["lo"], "dac_hi": dd["hi"]}
    score_ok, dac_ok = de["mean"] >= GATE_EPDMS, dd["mean"] <= GATE_DAC
    res |= {"score_ok": bool(score_ok), "dac_moved": bool(dac_ok), "verdict": "pass" if score_ok and dac_ok else ("lower_lambda" if dac_ok else "stop")}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / f"gate-{a.new}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit({"pass": 0, "lower_lambda": 1, "stop": 2}[res["verdict"]])


def cmd_navtest(a):
    from jevdrive import stats
    from jevdrive import navsim_zs as Z
    groups = {"P2": ["P2-F-s0", "P2-F-s1"], f"{a.hinge}": [f"{a.hinge}-F-s0", f"{a.hinge}-F-s1"]}
    E, log, t = _navtest([m for v in groups.values() for m in v])
    import pp_eval  # noqa: F401
    wa = E.read_csv(data_dir() / E.WAJEPA_CSV)[0]
    toks = sorted(set.intersection(*[set(x.index) for x in t.values()], set(wa.index)))
    g = np.array([log[k] for k in toks])
    fz = np.load(Z.root("index") / "navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    F = np.stack([fut[k] for k in toks])
    plen = lambda P: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[:, :, :2]], 1), axis=-1), axis=-1).sum(1)  # noqa: E731
    Lg = plen(F)
    mv = Lg > 2.0

    def vec(arm, c="score", fail=False):
        ts = [wa] if arm == "WA-JEPA" else [t[m] for m in groups[arm]]
        v = [ts_.loc[toks, c].to_numpy(float) for ts_ in ts]
        return np.mean([100.0 * (x < 1) if fail else 100.0 * x for x in v], 0)
    arms = list(groups) + ["WA-JEPA"]
    rows = []
    for arm in arms:
        r = {"arm": arm, "EPDMS": vec(arm).mean()}
        if arm != "WA-JEPA":
            r["EPDMS s0 / s1"] = " / ".join(f"{100 * t[m].loc[toks, 'score'].mean():.2f}" for m in groups[arm])
        r |= {k: float(np.nanmean(vec(arm, c) if k != "DAC" else vec(arm, c))) for k, c in SUBS.items()}
        r["DAC fail %"] = float(vec(arm, SUBS["DAC"], fail=True).mean())
        if arm != "WA-JEPA":
            zs = [np.load(data_dir() / "runs/op_lb/lb_navtest/preds" / f"warp-cinque_PP{m}__base.npz") for m in groups[arm]]
            P = np.mean([np.stack([dict(zip(z["tokens"].tolist(), z["poses"]))[k] for k in toks]) for z in zs], 0)
            r |= {"ade_vs_log": float(np.linalg.norm(P[:, :, :2] - F[:, :, :2], axis=-1).mean()), "speed_ratio_med": float(np.median(plen(P)[mv] / Lg[mv]))}
        rows.append(r)
    stats.write_table(rows, OUT / "hinge_navtest_arms", floatfmt=".2f",
                      note=f"navtest {len(toks)} tokens, protocol W, devkit v2 EPDMS x 100, seed means; DAC fail % = share of tokens with DAC < 1 (seed mean)")
    pr = []

    def pair(x, y, vx=None, vy=None, label=None):
        vx = vec(x) if vx is None else vx
        vy = vec(y) if vy is None else vy
        r = stats.paired(vx, vy, groups=g)
        d = {"pair": label or f"{x} - {y}", **{k: r[k] for k in ("mean", "lo", "hi", "mean_a", "mean_b", "n", "units")}}
        rf = stats.paired(vec(x, SUBS["DAC"], True), vec(y, SUBS["DAC"], True), groups=g)
        d |= {"dDACfail pp": rf["mean"], "dDACfail lo": rf["lo"], "dDACfail hi": rf["hi"]}
        for k in ("EP", "DAC", "NC", "TTC", "EC"):
            d[f"d{k}"] = stats.paired(vec(x, SUBS[k]), vec(y, SUBS[k]), groups=g)["mean"]
        pr.append(d)
    H = a.hinge
    pair(H, "P2"), pair(H, "WA-JEPA"), pair("P2", "WA-JEPA")
    for s in (0, 1):                                               # matched seeds (same row stream)
        x, y = f"{H}-F-s{s}", f"P2-F-s{s}"
        r = lambda m, c="score", fail=False: (100.0 * (t[m].loc[toks, c].to_numpy(float) < 1) if fail else 100.0 * t[m].loc[toks, c].to_numpy(float))  # noqa: E731
        rr = stats.paired(r(x), r(y), groups=g)
        rf = stats.paired(r(x, SUBS["DAC"], True), r(y, SUBS["DAC"], True), groups=g)
        pr.append({"pair": f"{x} - {y}", **{k: rr[k] for k in ("mean", "lo", "hi", "mean_a", "mean_b", "n", "units")},
                   "dDACfail pp": rf["mean"], "dDACfail lo": rf["lo"], "dDACfail hi": rf["hi"]})
    stats.write_table(pr, OUT / "hinge_navtest_paired", floatfmt=".2f",
                      note="per-token EPDMS x 100 (seed means), cluster bootstrap over navtest logs, B 10000; dDACfail = DAC failure share diff in pp (negative = fewer failures); dX = subscore diffs x 100")
    import pandas as pd
    print(pd.DataFrame(rows).to_string()), print(pd.DataFrame(pr).to_string())


def cmd_navhard(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive import navsim_zs as Z
    D = data_dir() / "runs/op_parity"
    lg = {e["token"]: e["log_name"] for e in Z.load_index("navhard_two_stage", slim=True)}
    src = {"P2 (G)": [D / "navhard_gimm/harness/P2-F-s0", D / "navhard_gimm/harness/P2-F-s1"],
           f"{a.hinge} (G)": [RUN / f"harness/{a.hinge}-F-s0", RUN / f"harness/{a.hinge}-F-s1"],
           "P0 (G)": [D / "navhard_gimm/harness/P0"], "WA-JEPA": [D / "navhard/harness/wajepa"]}
    runs = {k: [(pd.read_csv(d / "harness_groups.csv").set_index("group"), json.loads((d / "harness_summary.json").read_text())) for d in v
                if (d / "harness_groups.csv").exists()] for k, v in src.items()}
    runs = {k: v for k, v in runs.items() if v}
    g0 = runs["WA-JEPA"][0][0]
    logs = np.array([lg.get(o, "?") for o in g0.orig])
    rows = []
    for k, v in runs.items():
        assert all((t.orig.values == g0.orig.values).all() for t, _ in v), "group order differs"
        rows.append({"arm": k, "seeds": len(v)} | {c: float(np.mean([s[c] for _, s in v])) for c in ("combined", "stage1", "stage2")} |
                    {"combined s0 / s1": " / ".join(f"{s['combined']:.2f}" for _, s in v)})
    stats.write_table(rows, OUT / "hinge_navhard_arms", floatfmt=".2f", note="navhard_two_stage, protocol G (GIMM frames), devkit v2 EPDMS two-stage aggregation; seed means")
    grp = lambda k, c: np.mean([t[c].to_numpy(float) for t, _ in runs[k]], 0)  # noqa: E731
    pr = []
    for x, y in [(f"{a.hinge} (G)", "P2 (G)"), (f"{a.hinge} (G)", "WA-JEPA"), ("P2 (G)", "WA-JEPA"), (f"{a.hinge} (G)", "P0 (G)")]:
        if x in runs and y in runs:
            for c in ("combined", "stage1", "stage2"):
                r = stats.paired(grp(x, c), grp(y, c), groups=logs)
                pr.append({"pair": f"{x} - {y}", "score": c, **{k: r[k] for k in ("mean", "lo", "hi", "mean_a", "mean_b", "n", "units")}})
    stats.write_table(pr, OUT / "hinge_navhard_paired", floatfmt=".2f", note="per-group two-stage EPDMS, seed means, cluster bootstrap over stage-1 logs, B 10000")
    print(pd.DataFrame(rows).to_string()), print(pd.DataFrame(pr).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--new", required=True)
    p.add_argument("--ref", required=True)
    for n in ("navtest", "navhard"):
        p = sp.add_parser(n)
        p.add_argument("--hinge", default="P2H10")
    a = ap.parse_args()
    {"gate": cmd_gate, "navtest": cmd_navtest, "navhard": cmd_navhard}[a.cmd](a)
