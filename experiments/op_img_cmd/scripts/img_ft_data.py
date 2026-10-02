"""Image-command fine-tune (Q2): the training token pool, its route geometry and its op_lb frames.
Plan: ../plans/2026-10-04-img-cmd-ft-prereg.md.

The pool is disjoint from the zero-shot eval: junction frames of the decision-93 pair inventory (status branch, >= 2 exit
classes, taken branch known) whose LOG holds no eval token (the 385 junction + 293 straight frames of geom/nav.pkl) and whose
token is in neither op_lb's lb_navtrain (the eval source) nor L3's lb_h1train; at most PER_SEG frames per approach segment
(log, branching lane). Plus lane-keeping frames (no branching lane within 30 m, command straight, v >= 0.5) from the same logs.
Train / dev by log (dev 10%, used only for the pre-registered iteration rule).

  geom   (envs/navsim2)    candidates -> route geometry (img_geom_nav.process_log) -> geom/ft_cand.pkl
  pick   (envs/navsim2)    lane-match filter, final draw, splits navsim/img-ft-{train,dev}, geom/ft.pkl (row = -1 until prep)
  prep   (envs/openpilot)  op_lb.cmd_prep for data lb_imgtrain (4 keyframes per token), then the op_lb rows into geom/ft.pkl
  synth  (envs/vfi)        GIMM context frames of lb_imgtrain (op_lb.cmd_synth; chunks claimed atomically, several workers per card)
"""
import argparse, json, os, pickle, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).resolve().parent)]

DATA = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data")))
GEOM = DATA / "runs" / "op_img_cmd" / "geom"
OPLB = "lb_imgtrain"
PER_SEG = 3
N_JUNCTION = 1500
N_STRAIGHT = 300
CAND_X = 1.35           # candidates drawn above the target (the lane-match filter drops ~15%)
DEV_FRAC = 0.10


def excluded():
    """(tokens, logs) that the pool must avoid."""
    ev = pickle.load(open(GEOM / "nav.pkl", "rb"))
    lb = json.loads((DATA / "runs" / "op_lb" / "lb_navtrain" / "meta.json").read_text())["names"]
    h1 = json.loads((DATA / "runs" / "op_lb" / "lb_h1train" / "meta.json").read_text())["names"]
    return set(lb) | set(h1), {s["log"] for s in ev}


def cmd_geom(a):
    import pandas as pd
    import img_geom_nav as G
    df = pd.read_parquet(DATA / "processed" / "op_common_cause" / "pair_inventory" / "navtrain_frames.parquet")
    tok_x, log_x = excluded()
    df = df[~df.token.isin(tok_x) & ~df.log.isin(log_x)]
    rng = np.random.default_rng(0)
    J = df[(df.status == "branch") & df.classes.str.contains(",") & (df.taken != "")].assign(kind="junction")
    J = J.iloc[rng.permutation(len(J))].groupby(["log", "node"]).head(PER_SEG)
    J = J.iloc[:int(N_JUNCTION * CAND_X)]
    S = df[(df.status == "no_junction_30m") & (df.cmd == "straight") & (df.v >= 0.5) & df.log.isin(set(J.log))]
    S = S.sample(min(int(N_STRAIGHT * CAND_X), len(S)), random_state=0).assign(kind="straight")
    D = pd.concat([J, S]).assign(row=-1)
    print(f"candidates: junction {len(J)} ({J.log.nunique()} logs, {J.groupby(['log', 'node']).ngroups} segments), straight {len(S)}")
    keep = ["token", "log", "kind", "v", "cmd", "dist", "classes", "taken", "row"]
    jobs = [(log, g[keep].to_dict("records")) for log, g in D.groupby("log")]
    res = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(G.process_log, jobs)):
            res += r
            if k % 50 == 0:
                print(k, len(jobs), len(res), flush=True)
    res.sort(key=lambda s: (s["kind"], s["token"]))
    pickle.dump(res, open(GEOM / "ft_cand.pkl", "wb"))
    print("geometry:", pd.Series([s["kind"] for s in res]).value_counts().to_dict())


def cmd_pick(a):
    import pandas as pd
    import img_overlay as O
    from jevdrive.data import splits
    C = [s for s in pickle.load(open(GEOM / "ft_cand.pkl", "rb")) if O.valid(s)]
    rng = np.random.default_rng(1)
    J = [s for s in C if s["kind"] == "junction"]
    J = [J[k] for k in rng.permutation(len(J))[:N_JUNCTION]]
    logs = {s["log"] for s in J}
    S = [s for s in C if s["kind"] == "straight" and s["log"] in logs]
    S = [S[k] for k in rng.permutation(len(S))[:N_STRAIGHT]]
    P = sorted(J + S, key=lambda s: (s["kind"], s["token"]))
    ul = np.array(sorted(logs))
    dev_logs = set(np.random.default_rng(20261004).permutation(ul)[: int(round(DEV_FRAC * len(ul)))])
    tok_x, log_x = excluded()
    assert not ({s["token"] for s in P} & tok_x) and not ({s["log"] for s in P} & log_x)
    origin = (f"decision-93 pair-inventory junction frames (status branch, >= 2 classes, taken known) + lane-keeping frames "
              f"(no_junction_30m, cmd straight, v >= 0.5) whose log holds no op_img_cmd eval token (geom/nav.pkl) and whose token "
              f"is not in op_lb lb_navtrain / lb_h1train; <= {PER_SEG} per (log, branching lane); lane-match <= 1.5 m; "
              f"{N_JUNCTION} junction (seed 0 / 1) + {N_STRAIGHT} straight; dev = {DEV_FRAC:.0%} of the logs (seed 20261004)")
    for nm, dv in (("img-ft-train", False), ("img-ft-dev", True)):
        mem = [s["token"] for s in P if (s["log"] in dev_logs) == dv]
        sp = splits.define("navsim", nm, mem, unit="token", origin=origin, status="frozen", used_by=["op_img_cmd"],
                           notes="op_img_cmd image-command fine-tune pool (train / dev by log); disjoint from the zero-shot eval logs")
        print(sp.id, len(mem))
    for s in P:
        s["split"] = "dev" if s["log"] in dev_logs else "train"
    pickle.dump(P, open(GEOM / "ft.pkl", "wb"))
    print(pd.DataFrame([dict(kind=s["kind"], split=s["split"]) for s in P]).value_counts().to_dict())


def cmd_prep(a):
    import op_lb as B
    from jevdrive import navsim_zs as Z
    B.SPLITS[OPLB] = "navtrain"
    P = pickle.load(open(GEOM / "ft.pkl", "rb"))
    sel = {s["token"] for s in P}
    Z.nonav_subset = lambda idx_, per_command=0, seed=0: sel            # op_lb.cmd_prep's navtrain draw -> this pool
    B.cmd_prep(argparse.Namespace(data=OPLB, per_cmd=0, seed=0, workers=a.workers))
    row = {n: k for k, n in enumerate(B.meta(OPLB)["names"])}
    assert set(row) == sel
    for s in P:
        s["row"] = row[s["token"]]
    pickle.dump(P, open(GEOM / "ft.pkl", "wb"))


def cmd_synth(a):
    import op_lb as B
    B.SPLITS[OPLB] = "navtrain"
    B.cmd_synth(argparse.Namespace(data=[OPLB], method="gimm", gpu=a.gpu, vram_gb=a.vram_gb, cap_gb=a.cap_gb, chunk=32,
                                   batch=8, workers=1, limit_chunks=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("geom", "pick", "prep"):
        p = sp.add_parser(n)
        p.add_argument("--workers", type=int, default=40)
    p = sp.add_parser("synth")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--vram-gb", type=float, default=14.0)
    p.add_argument("--cap-gb", type=float, default=62.0)
    a = ap.parse_args()
    {"geom": cmd_geom, "pick": cmd_pick, "prep": cmd_prep, "synth": cmd_synth}[a.cmd](a)
