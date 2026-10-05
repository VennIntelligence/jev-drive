"""op_parity full run, NAVSIM part (results/full.md): navtest v2 EPDMS of P0 and P1 / P2 / P3 x 2 seeds under protocol W, seed means, paired
contrasts and the same-harness WA-JEPA comparison -> results/full_navtest_{arms,paired}.{md,csv}.

  python experiments/op_parity/scripts/pp_full_report.py        (box; reads the devkit CSVs and pose files written by pp_eval.py)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_pl.Path(__file__).parent)]

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

SUBS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
        "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
ARMS = {"P0": ["P0"], "P1": ["P1-F-s0", "P1-F-s1"], "P2": ["P2-F-s0", "P2-F-s1"], "P3": ["P3-F-s0", "P3-F-s1"],
        "P3 side masked": ["P3-F-s0:noside", "P3-F-s1:noside"]}


def main():
    from jevdrive import stats
    from jevdrive import navsim_zs as Z
    import pp_eval as E
    E.DATA, E.FRAMES = "lb_navtest", "warp"
    log = {e["token"]: e["log_name"] for e in Z.load_index("navtest", slim=True)}
    fz = np.load(Z.root("index") / "navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    runs = {}
    for arm, ms in ARMS.items():
        got = []
        for m in ms:
            f = E.eval_csv(m)
            if f is None:
                continue
            t, _ = E.read_csv(f)
            z = np.load(data_dir() / "runs" / "op_lb" / "lb_navtest" / "preds" / f"{E.stem(m).replace('@', '-')}__base.npz")
            got.append((m, t, dict(zip(z["tokens"].tolist(), z["poses"]))))
        if got:
            runs[arm] = got
    runs["WA-JEPA"] = [("WA-JEPA", E.read_csv(data_dir() / E.WAJEPA_CSV)[0], None)]
    toks = sorted(set.intersection(*[set(t.index) for v in runs.values() for _, t, _ in v]))
    g = np.array([log[t] for t in toks])
    F = np.stack([fut[t] for t in toks])
    plen = lambda P: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[:, :, :2]], 1), axis=-1), axis=-1).sum(1)  # noqa: E731
    Lg = plen(F)
    mv = Lg > 2.0

    def tok(arm, col="score"):
        return np.mean([100 * t.loc[toks, col].to_numpy(float) for _, t, _ in runs[arm]], 0)

    rows = []
    for arm, v in runs.items():
        r = {"arm": arm, "seeds": len(v), "EPDMS": tok(arm).mean()}
        r |= {"EPDMS s0 / s1": " / ".join(f"{100 * t.loc[toks, 'score'].mean():.2f}" for _, t, _ in v)}
        r |= {k: float(np.nanmean(tok(arm, c))) for k, c in SUBS.items()}
        if v[0][2] is not None:
            P = np.mean([np.stack([p[t] for t in toks]) for _, _, p in v], 0)
            r |= {"ade_vs_log": float(np.linalg.norm(P[:, :, :2] - F[:, :, :2], axis=-1).mean()),
                  "speed_ratio_med": float(np.median(plen(P)[mv] / Lg[mv]))}
        rows.append(r)
    out = _R / "experiments" / "op_parity" / "results"
    stats.write_table(rows, out / "full_navtest_arms", floatfmt=".2f",
                      note=f"navtest {len(toks)} tokens, protocol W, devkit navsim main @0a380a9 v2 EPDMS x 100; seed means (per-token mean of the seeds)")
    pr = []
    pairs = [("P1", "P0"), ("P2", "P1"), ("P3", "P1"), ("P3", "P2"), ("P3", "P3 side masked")] + [(a, "WA-JEPA") for a in runs if a != "WA-JEPA"]
    for a, b in pairs:
        if a in runs and b in runs:
            r = stats.paired(tok(a), tok(b), groups=g)
            pr.append({"pair": f"{a} - {b}", **{k: r[k] for k in ("mean", "lo", "hi", "mean_a", "mean_b", "n", "units")}})
            for k in ("EP", "DAC", "NC", "TTC", "EC"):
                pr[-1][f"d{k}"] = stats.paired(tok(a, SUBS[k]), tok(b, SUBS[k]), groups=g)["mean"]
    stats.write_table(pr, out / "full_navtest_paired", floatfmt=".2f", note="per-token EPDMS x 100 (seed means), cluster bootstrap over navtest logs, B 10000; dX = subscore diffs")
    import pandas as pd
    print(pd.DataFrame(rows).to_string()), print(pd.DataFrame(pr).to_string())


if __name__ == "__main__":
    main()
