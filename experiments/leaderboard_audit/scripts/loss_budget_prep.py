"""Loss budget, step 1 (CPU, navsim2 env): per-token rows and geometry for oracle substitution on navhard and navtest.

For every token and arm (native = shipped Cinque, best = it_dw3-s0 + selector sel-rot0-r0.6, decision 101) this stores the plan end
pose, the reference end pose (PDM-Closed reference and the human trajectory of the metric cache), the start offset to the route
centreline and, for DAC failures, the first-departure time of the simulated ego. navhard also stores the full devkit score rows
(arm plans, PDM reference, human trajectory) so that step 2 can substitute rows and aggregate with the devkit's two-stage code.

  DATA_DIR=... PYTHONPATH=$DATA_DIR/third_party/navsim:$DATA_DIR/third_party/nuplan-devkit:experiments/skill_pack/scripts \
    $DATA_DIR/envs/navsim2/bin/python experiments/leaderboard_audit/scripts/loss_budget_prep.py navhard|navtest --procs 40
Output: $DATA_DIR/runs/leaderboard_audit/loss_budget/prep_<board>.pkl
"""
import argparse
import glob
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/skill_pack/scripts")]
import offroad_lib as L  # noqa: E402
import offroad_replay_cf as R  # noqa: E402

OUTD = L.D / "runs/leaderboard_audit/loss_budget"
BEST = "gimm-cinque_Oit_dw3-s0_al-sel-rot0-r0.6__base.npz"
ARMS = {"navhard": {"native": L.NATIVE_POSES, "best": L.D / "runs/op_lb/lb_navhard/preds" / BEST},
        "navtest": {"native": L.D / "runs/op_lb/lb_navtest/preds/gimm-cinque__base.npz", "best": L.D / "runs/op_lb/lb_navtest/preds" / BEST}}
CACHE = {"navhard": L.MCACHE, "navtest": L.D / "runs/navsim/metric_cache/v1_navtest"}
_W = {}


def _init(board):
    sim, scorer, policy, mapping, samp = L.setup_scoring()
    cp = {Path(p).parent.name: p for p in glob.glob(str(CACHE[board] / "*/*/*/metric_cache.pkl"))}
    _W.update(board=board, sim=sim, scorer=scorer, policy=policy, samp=samp, cp=cp, P={k: L.poses_by_token(v) for k, v in ARMS[board].items()})


def end_info(p8):
    return dict(end_x=float(p8[-1, 0]), end_y=float(p8[-1, 1]), end_yaw=float(p8[-1, 2]))


def work(token):
    W = _W
    mc = L.load_cache(W["cp"][token])
    ref = L.pdm_ref_ego(mc, W["samp"])
    p_ref = L.poses_from_dense(ref)
    p_hum = None if getattr(mc, "human_trajectory", None) is None else np.asarray(mc.human_trajectory.poses, np.float64)   # None for synthetic stage-2 frames
    out = dict(token=token, ref=end_info(p_ref), human=None if p_hum is None else end_info(p_hum), start=R.start_features(mc), arms={}, rows={})
    g = R.geoms(mc)
    plans = {"ref": p_ref, **({} if p_hum is None else {"human": p_hum}), **{a: W["P"][a][token] for a in W["P"]}}
    for name, p8 in plans.items():
        if W["board"] == "navhard":
            row, st = R._score(mc, p8, token)          # R._score reads R._W
            out["rows"][name] = row
        else:
            st = L.simulate(W["sim"], mc, p8)
        if name in W["P"]:
            f = R.features(mc, g, st, p8, ref)
            out["arms"][name] = dict(**end_info(p8), first=f["first"], start_outside=f["start_outside"], junction=f["junction"], dac_sim=f["dac"])
    return out


def init2(board):
    _init(board)
    R._W.update(sim=_W["sim"], scorer=_W["scorer"], policy=_W["policy"], samp=_W["samp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board", choices=["navhard", "navtest"])
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    idx = L.index() if a.board == "navhard" else pickle.load(open(L.D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))
    tokens = [e["token"] for e in idx]
    cmd = {e["token"]: int(np.argmax(e["cmd"][-1])) for e in idx}
    if a.limit:
        tokens = tokens[::len(tokens) // a.limit][:a.limit]
    t0 = time.time()
    res = {}
    with mp.get_context("fork").Pool(a.procs, initializer=init2, initargs=(a.board,)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tokens, chunksize=4)):
            r["cmd"] = cmd[r["token"]]
            res[r["token"]] = r
            if i % 500 == 0:
                print(i, len(tokens), f"{time.time() - t0:.0f}s", flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUTD / f"prep_{a.board}{'_dbg' if a.limit else ''}.pkl", "wb"), protocol=4)
    print("done", len(res), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
