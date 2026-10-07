"""Four-directions follow-up checks (CPU, stored outputs; results/four_dirs.md):

  speed     planned vs logged distance into sharp turns and braking approaches on navtest (P2H, WA-JEPA)   -> results/four_dirs/entry_speed_*.csv
  command   how early the navtest driving command switches from straight to left / right before a > 45 deg turn -> results/four_dirs/command_lead_navtest.csv
  turncoll  turn-train arms: change of NC / TTC failures vs H on turning tokens                              -> results/four_dirs/turn_train_collisions.csv

Inputs: experiments/op_probe/results/turn-gain/data/navtest_turn.csv.gz (per-token plan / log geometry), $DATA_DIR/runs/op_probe/joint/navtest_tokens.parquet,
$DATA_DIR/runs/bench/navtest/<tag>@warp/units.csv. Ratios are ratios of sums; CIs: cluster bootstrap over logs (B 4000, seed 0).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = _R / "experiments/op_parity/results/four_dirs"
TURN = _R / "experiments/op_probe/results/turn-gain/data/navtest_turn.csv.gz"


def ddir():
    from jevdrive.common import data_dir
    return data_dir()


def ratio_ci(s, num, den, rng, B=4000):
    g = s.groupby("log")[[num, den]].sum().values
    idx = rng.integers(0, len(g), (B, len(g)))
    r = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return f"{g[:, 0].sum() / g[:, 1].sum():.3f} [{np.percentile(r, 2.5):.3f}, {np.percentile(r, 97.5):.3f}]"


def cmd_speed(_):
    rng = np.random.default_rng(0)
    d = pd.read_csv(TURN)
    d = d[d.ok].copy()
    for c in ["s2", "s4", "apk"]:
        d[f"P2H_{c}"] = (d[f"P2Hs0_{c}"] + d[f"P2Hs1_{c}"]) / 2
    a, br = d.L_yaw4.abs(), d.L_s4 / (4 * d.v0)
    sets = {"entry_speed_navtest": ({">45 deg": a > 45, ">45 deg, v0 > 6": (a > 45) & (d.v0 > 6), ">45 deg, v0 3-6": (a > 45) & d.v0.between(3, 6),
                                     ">45 deg, v0 < 3": (a > 45) & (d.v0 < 3), "20-45 deg": a.between(20, 45), "<5 deg": a < 5}, ["s2", "s4", "apk"]),
            "entry_speed_braking_navtest": ({"braking (log s4 < 0.8 v0*4), v0 > 6, turn > 20 deg in 4 s": (br < 0.8) & (d.v0 > 6) & (a > 20),
                                             "braking, v0 > 6, straight < 5 deg": (br < 0.8) & (d.v0 > 6) & (a < 5),
                                             "braking, v0 > 6, all": (br < 0.8) & (d.v0 > 6)}, ["s2", "s4"])}
    for name, (strata, cols) in sets.items():
        rows = []
        for k, m in strata.items():
            s = d[m]
            r = {"stratum": k, "n": len(s), "median v0": round(s.v0.median(), 2)}
            for who in ["P2H", "WA"]:
                for c in cols:
                    r[f"{who} {c} / log"] = ratio_ci(s, f"{who}_{c}", f"L_{c}", rng)
            rows.append(r)
        o = pd.DataFrame(rows)
        o.to_csv(OUT / f"{name}.csv", index=False)
        print(o.to_markdown(index=False))


def cmd_command(_):
    """Per log, every straight -> left / right switch of the command (adjacent 0.5 s tokens) that is followed, inside the same command run, by a
    token whose logged 4 s heading change exceeds 45 deg: time and distance (v0 x dt) from the switch to that token."""
    d = pd.read_parquet(ddir() / "runs/op_probe/joint/navtest_tokens.parquet").sort_values(["log", "ts"])
    rows = []
    for lg, g in d.groupby("log"):
        t, c, a, v = g.ts.values / 1e6, g.cmd.values, g.dyaw.abs().values, g.v0.values
        i = 1
        while i < len(g):
            if c[i] in ("left", "right") and c[i - 1] == "straight" and t[i] - t[i - 1] < 0.75:
                j = i
                while j + 1 < len(g) and c[j + 1] == c[i] and t[j + 1] - t[j] < 0.75:
                    j += 1
                k = [m for m in range(i, j + 1) if a[m] > 45]
                if k:
                    m = k[0]
                    rows.append({"log": lg, "cmd": c[i], "dt_to_turn_in_horizon_s": t[m] - t[i], "dist_m": float(np.sum(v[i:m] * np.diff(t[i:m + 1]))),
                                 "v_at_switch": v[i], "dyaw_at_switch": g.dyaw.values[i]})
                i = j + 1
            else:
                i += 1
    o = pd.DataFrame(rows)
    o.to_csv(OUT / "command_lead_navtest.csv", index=False)
    print(len(o), "switch events")
    print(o[["dt_to_turn_in_horizon_s", "dist_m", "v_at_switch", "dyaw_at_switch"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
    x = pd.crosstab(pd.cut(d.dyaw, [-400, -45, -20, -5, 5, 20, 45, 400]), d.cmd)
    print(x.to_string())


def cmd_turncoll(_):
    rng = np.random.default_rng(0)
    t = pd.read_parquet(ddir() / "runs/op_probe/joint/navtest_tokens.parquet").set_index("token")
    turn = t.dyaw.abs() > 20

    def fail(m):
        x = [pd.read_csv(ddir() / f"runs/bench/navtest/{m}-F-s{s}@warp/units.csv").set_index("token") for s in (0, 1)]
        return sum(((u.NC < 1) | (u.TTC < 1)).astype(float) for u in x) / 2
    H, rows = fail("HP"), []
    for m in ["T1P", "T2P", "T3P"]:
        D = (fail(m) - H).reindex(t.index)
        for lab, mask in [("turn > 20 deg", turn), ("all", turn | ~turn)]:
            dd = D[mask].dropna()
            g = dd.groupby(t.log.reindex(dd.index)).agg(["sum", "count"]).values
            idx = rng.integers(0, len(g), (4000, len(g)))
            b = g[idx, 0].sum(1) / g[idx, 1].sum(1) * 100
            rows.append({"arm": m, "stratum": lab, "n": len(dd), "H NC|TTC fail %": round(100 * H.reindex(dd.index).mean(), 2),
                         "arm - H (pp)": f"{100 * dd.mean():+.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]"})
    o = pd.DataFrame(rows)
    o.to_csv(OUT / "turn_train_collisions.csv", index=False)
    print(o.to_markdown(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n, f in [("speed", cmd_speed), ("command", cmd_command), ("turncoll", cmd_turncoll)]:
        sp.add_parser(n).set_defaults(fn=f)
    a = ap.parse_args()
    a.fn(a)
