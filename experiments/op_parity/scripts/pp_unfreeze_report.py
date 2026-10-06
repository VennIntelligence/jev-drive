"""op_parity vision-unfreeze report (plans/2026-10-06-unfreeze-prereg.md).

  equiv   pixel-path navtest plans vs the cached-token plans (P0, full-run P2-F-s0): mean < 0.01 m or exit 1 (no scoring)
  pilot   navtest v2 EPDMS per variant (seed means), paired contrasts vs F, guards (speed ratio, EP / DAC trade, dev drift_off) and the
          pre-registered gate -> results/unfreeze_pilot_{arms,paired}.{md,csv}, results/unfreeze_pilot_gate.json
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

OUT = _R / "experiments/op_parity/results"
U = data_dir() / "runs/op_parity/unfreeze"
VARS = {"F": "warp", "U1": "warp", "U1L": "warp", "U2": "warp", "V": "vh140"}
SUBS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
        "EC": "two_frame_extended_comfort"}
GATE = 0.5
DRIFT = 0.10


def cmd_equiv(a):
    eq = json.loads((U / "equivalence_warp_lb_navtest.json").read_text())
    bad = {m: v for m, v in eq.items() if m in ("P0", "P2-F-s0") and v["vs_cached_plan_mean_m"] >= 0.01}
    print(json.dumps(eq, indent=1))
    if bad or not all(m in eq for m in ("P0", "P2-F-s0")):
        raise SystemExit(f"equivalence fails: {bad or 'missing'}")


def dev_summary(tag):
    fs = sorted((data_dir() / "runs/op_parity" / f"utrain-{tag}").glob("*/DONE"))
    return json.loads(fs[-1].read_text()) if fs else {}


def cmd_pilot(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive import navsim_zs as Z
    import pp_eval as E
    log = {e["token"]: e["log_name"] for e in Z.load_index("navtest", slim=True)}
    fz = np.load(Z.root("index") / "navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    runs = {}
    for v, fr in list(VARS.items()) + [("P0 (W)", "warp"), ("P0 (vh140)", "vh140"), ("P2 full (frozen)", "warp")]:
        E.FRAMES = fr
        tags = {"P0 (W)": ["P0"], "P0 (vh140)": ["P0"], "P2 full (frozen)": ["P2-F-s0", "P2-F-s1"]}.get(v, [f"UF-{v}-s{s}" for s in (0, 1)])
        got = []
        for m in tags:
            f = E.eval_csv(m)
            if f is None:
                continue
            t, _ = E.read_csv(f)
            z = np.load(data_dir() / "runs/op_lb/lb_navtest/preds" / f"{E.stem(m).replace('@', '-')}__base.npz")
            got.append((m, t, dict(zip(z["tokens"].tolist(), z["poses"]))))
        if got:
            runs[v] = got
    runs["WA-JEPA"] = [("WA-JEPA", E.read_csv(data_dir() / E.WAJEPA_CSV)[0], None)]
    toks = sorted(set.intersection(*[set(t.index) for v in runs.values() for _, t, _ in v]))
    g = np.array([log[t] for t in toks])
    F = np.stack([fut[t] for t in toks])
    plen = lambda P: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[:, :, :2]], 1), axis=-1), axis=-1).sum(1)  # noqa: E731
    Lg = plen(F)
    mv = Lg > 2.0

    def tok(v, col="score"):
        return np.mean([100 * t.loc[toks, col].to_numpy(float) for _, t, _ in runs[v]], 0)

    rows, guard = [], {}
    for v, got in runs.items():
        r = {"arm": v, "seeds": len(got), "EPDMS": tok(v).mean(),
             "EPDMS per seed": " / ".join(f"{100 * t.loc[toks, 'score'].mean():.2f}" for _, t, _ in got)}
        r |= {k: float(np.nanmean(tok(v, c))) for k, c in SUBS.items()}
        if got[0][2] is not None:
            P = np.mean([np.stack([p[t] for t in toks]) for _, _, p in got], 0)
            r |= {"ade_vs_log": float(np.linalg.norm(P[:, :, :2] - F[:, :, :2], axis=-1).mean()), "speed_ratio_med": float(np.median(plen(P)[mv] / Lg[mv]))}
        ds = [dev_summary(m) for m, _, _ in got if m.startswith("UF-")]
        if ds:
            r |= {"dev_ade": float(np.mean([d.get("dev_ade", np.nan) for d in ds])), "dev_drift_off": float(np.mean([d.get("dev_drift_off", np.nan) for d in ds]))}
        rows.append(r)
    stats.write_table(rows, OUT / "unfreeze_pilot_arms", floatfmt=".2f",
                      note=f"navtest {len(toks)} tokens, pixel path, devkit navsim main @0a380a9 v2 EPDMS x 100; seed means; V and P0 (vh140) on the 1.40 m virtual camera")
    pr, gate = [], {}
    for x in ("U1", "U1L", "U2", "V", "P2 full (frozen)", "F"):
        for y in (["F"] if x != "F" else []) + ["WA-JEPA"]:
            if x not in runs or y not in runs:
                continue
            r = stats.paired(tok(x), tok(y), groups=g)
            row = {"pair": f"{x} - {y}", **{k: r[k] for k in ("mean", "lo", "hi", "mean_a", "mean_b", "n", "units")}}
            sub = {k: stats.paired(tok(x, c), tok(y, c), groups=g) for k, c in (("EP", SUBS["EP"]), ("DAC", SUBS["DAC"]), ("NC", SUBS["NC"]), ("TTC", SUBS["TTC"]))}
            row |= {f"d{k}": s["mean"] for k, s in sub.items()}
            pr.append(row)
            if y == "F" and x in ("U1", "U1L", "U2", "V"):
                ax = next(q for q in rows if q["arm"] == x)
                trade = (sub["EP"]["mean"] * sub["DAC"]["mean"] < 0) and all(s["lo"] > 0 or s["hi"] < 0 for s in (sub["EP"], sub["DAC"]))
                c = {"diff": r["mean"] >= GATE, "speed": abs(ax.get("speed_ratio_med", np.nan) - 1) <= 0.05, "no_trade": not trade,
                     "drift_off": ax.get("dev_drift_off", np.inf) <= DRIFT}
                gate[x] = {"diff_vs_F": r["mean"], "ci": [r["lo"], r["hi"]], "speed_ratio": ax.get("speed_ratio_med"),
                           "dEP": sub["EP"]["mean"], "dDAC": sub["DAC"]["mean"], "drift_off": ax.get("dev_drift_off"), "checks": c, "pass": all(c.values())}
    stats.write_table(pr, OUT / "unfreeze_pilot_paired", floatfmt=".2f", note="per-token EPDMS x 100 (seed means), cluster bootstrap over navtest logs, B 10000")
    passing = {k: v for k, v in gate.items() if v["pass"] and k != "V"}
    gate["_decision"] = {"line": GATE, "drift_line_m": DRIFT, "best_U": max(passing, key=lambda k: passing[k]["diff_vs_F"]) if passing else None,
                         "V_pass": gate.get("V", {}).get("pass")}
    (OUT / "unfreeze_pilot_gate.json").write_text(json.dumps(gate, indent=1, default=float))
    print(pd.DataFrame(rows).to_string()), print(pd.DataFrame(pr).to_string()), print(json.dumps(gate, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["equiv", "pilot"])
    a = ap.parse_args()
    {"equiv": cmd_equiv, "pilot": cmd_pilot}[a.cmd](a)
