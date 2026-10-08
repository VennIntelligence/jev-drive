"""op_parity turn selector bench: identity gate (G-id) and the pooled report of the bench runs (plans/2026-10-08-turn-selector-bench-prereg.md).

  python turn_selbench_report.py idcheck --subset navsim/op-parity-tsbench-smoke
  python turn_selbench.py report                       (calls report(a) below)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

TERMS = ["score", "NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]


def units(bench, key):
    import pandas as pd
    from jevdrive.bench.runner import bench_root
    return pd.read_csv(bench_root(bench, key) / "units.csv").set_index("token")


def cmd_idcheck(a):
    """G-id: ts0 on the smoke logs equals the archived SH30-F-s0 / s1 bench scores token by token (max abs diff <= 1e-9, NaN pattern equal)."""
    from jevdrive.bench.models import data_dir
    tag = a.subset.replace("/", "-")
    res = {}
    for s in (0, 1):
        new, old = units("navtest", f"SH30-F-s{s}@warp_ts0_{tag}"), units("navtest", f"SH30-F-s{s}@warp")
        assert len(new) > 300, f"smoke run has {len(new)} tokens"
        old = old.loc[new.index]
        d = [float(np.nanmax(np.abs(new[c].to_numpy(float) - old[c].to_numpy(float)))) for c in TERMS]
        nan_ok = all((np.isnan(new[c].to_numpy(float)) == np.isnan(old[c].to_numpy(float))).all() for c in TERMS)
        res[f"s{s}"] = dict(n=int(len(new)), max_abs_diff=max(d), nan_pattern_equal=bool(nan_ok), per_term=dict(zip(TERMS, d)))
    res["ok"] = bool(all(v["max_abs_diff"] <= 1e-9 and v["nan_pattern_equal"] for k, v in res.items() if k != "ok"))
    out = data_dir() / "runs/op_parity/turn_selbench/gate_id.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if not res["ok"]:
        raise SystemExit("G-id failed")


def report(a):
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("idcheck")
    p.add_argument("--subset", required=True)
    a = ap.parse_args()
    {"idcheck": cmd_idcheck}[a.cmd](a)
