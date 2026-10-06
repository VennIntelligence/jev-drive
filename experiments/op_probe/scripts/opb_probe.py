"""op_probe analysis (plans/2026-10-06-dac-localize-prereg.md sections 3-5): token sets, probes per stage, decoders, report tables.

  select                      navtest token sets (F / PP / R / FF from op_parity's gap tables) + the small-read rows -> runs/op_probe/sets/
  probe  [--small]            ridge (and MLP) probes per stage, metrics M1 / M2 on navtest sets -> runs/op_probe/probe[-small]/
  decode [--small]            plan decoders per stage (D-imit / D-hinge) -> navtest poses npz for opb_score.py
  report                      results/dac-localize tables

op-train env (torch); probes / decoders on one GPU (pool job).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "runs" / "op_probe"
SETS = ROOT / "sets"
CACHE = data_dir() / "runs" / "op_parity" / "cache"
GAP = data_dir() / "runs" / "op_parity" / "gap" / "gap_navtest_shap.npz"
J_DAC = 1                                         # TERMS index of DAC in the gap tables


def cmd_select(a):
    from jevdrive.run import Run
    with Run("op_probe", "select", config=vars(a)) as run:
        tab = np.load(CACHE / "lb_navtest" / "tab.npz")
        names = tab["names"]
        z = np.load(GAP, allow_pickle=True)
        pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
        ix = np.array([pos[t] for t in names])
        p2 = z["X2"][0][ix, J_DAC] < 1                            # P2-F-s0 fails DAC (the model whose features are probed)
        p2s1 = z["X2"][1][ix, J_DAC] < 1
        wa = z["Xw"][ix, J_DAC] < 1
        sets = {"F": p2 & ~wa, "PP": ~p2 & ~wa, "R": wa & ~p2, "FF": p2 & wa, "F_both_seeds": p2 & p2s1 & ~wa}
        rng = np.random.default_rng(0)
        small = np.sort(np.r_[rng.choice(np.flatnonzero(sets["F"]), 200, replace=False), rng.choice(np.flatnonzero(sets["PP"]), 200, replace=False)])
        SETS.mkdir(parents=True, exist_ok=True)
        np.savez(SETS / "navtest_sets.npz", tokens=names, log=tab["log"], **sets)
        np.save(SETS / "small400_rows.npy", small)
        (SETS / "small400_tokens.txt").write_text("\n".join(names[small]) + "\n")
        run.summary.update({k: int(v.sum()) for k, v in sets.items()} | {"small": len(small)})
        run.info(json.dumps(run.summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    a = ap.parse_args()
    {"select": cmd_select}[a.cmd](a)
