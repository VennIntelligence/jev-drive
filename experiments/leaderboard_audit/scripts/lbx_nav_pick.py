"""Loss-budget examples (tmp/lbx), NAV lane step 1: per-token class membership and representative picks.

Recomputes the per-token class flags of loss_budget_nav.py (same tables / geometry functions, shipped = native, best = it_dw3 +
selector) and picks, per class, the tokens whose score loss (reference token score minus the shipped token score) is nearest the
class median, one per log. Writes $DATA_DIR/runs/leaderboard_audit/loss_budget/lbx_<board>.json (picks + class stats) and
lbx_<board>_tokens.pkl (all tokens).

  navsim2 env, CPU:  lbx_nav_pick.py navhard|navtest
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loss_budget_nav as N  # noqa: E402

L = N.L
CLASSES = ["road_edge_wide", "early_clip", "wrong_direction", "collisions", "stopped_slow", "no_command"]
K = {"navhard": 3, "navtest": 3}
K_SMALL = 2          # classes with fewer than 20 tokens


def main(board):
    prep = pickle.load(open(N.PREP / f"prep_{board}.pkl", "rb"))
    T, _ = N.tables(board, prep)
    G = N.geometry(board, prep, T)
    idx = {e["token"]: e for e in (L.index() if board == "navhard" else pickle.load(open(L.D / "runs/navsim_zs/index/navtest_slim.pkl", "rb")))}
    tok = list(prep)
    df = pd.DataFrame(index=tok)
    df["log"] = [idx[t]["log_name"] for t in tok]
    df["stage"] = [idx[t]["stage"] for t in tok]
    df["cmd"] = [prep[t]["cmd"] for t in tok]
    df["v0"] = [float(np.linalg.norm(idx[t]["vel"][-1])) for t in tok]
    df["ref_score"] = T["ref"].score
    for a in ("native", "best"):
        df[f"{a}_score"] = T[a].score
        for s in N.SUBS:
            df[f"{a}_{s}"] = T[a][s]
        for c in CLASSES + ["failing", "offroad_all"]:
            df[f"{a}_{c}"] = G[a][c].astype(float)
    for s in N.SUBS:
        df[f"ref_{s}"] = T["ref"][s]
    df["loss"] = (df.ref_score - df.native_score).clip(lower=0)
    if board == "navhard":
        re = pd.read_pickle(L.OUT / "roadedge_table.pkl").set_index("token")
        df["edge_err"] = re["native_edge_err"].reindex(tok)
    df.to_pickle(N.PREP / f"lbx_{board}_tokens.pkl")

    used_logs, picks, stats = set(), [], {}
    for c in CLASSES:
        m = df[f"native_{c}"] == 1
        if not m.any():
            stats[c] = dict(n=0)
            continue
        sub = df[m]
        med = float(sub.loss.median())
        stats[c] = dict(n=int(m.sum()), loss_median=med, loss_q25=float(sub.loss.quantile(.25)), loss_q75=float(sub.loss.quantile(.75)),
                        best_also=int((m & (df[f"best_{c}"] == 1)).sum()), best_failing=int((m & (df.best_failing == 1)).sum()),
                        stage2=float((sub.stage == "two").mean()))
        k = K[board] if m.sum() >= 20 else K_SMALL
        order = (sub.loss - med).abs().sort_values().index
        got = 0
        for t in order:
            if df.at[t, "log"] in used_logs or not Path(idx[t]["cams"][-1]["CAM_F0"]["path"]).exists():
                continue
            used_logs.add(df.at[t, "log"])
            r = df.loc[t]
            picks.append(dict(cls=c, token=t, log=r.log, stage=r.stage, cmd=L.CMDS[int(r.cmd)], v0=r.v0, loss=float(r.loss), class_median=med,
                              rank_in_class=int((sub.loss < r.loss).sum()), n_class=int(m.sum()),
                              other_classes=[o for o in CLASSES if o != c and r[f"native_{o}"] == 1],
                              best_same_class=bool(r[f"best_{c}"] == 1), best_failing=bool(r.best_failing == 1),
                              edge_err=None if board == "navtest" or pd.isna(r.get("edge_err")) else float(r.edge_err),
                              scores={a: {s: float(r[f"{a}_{s}"]) for s in ["score"] + N.SUBS} for a in ("native", "best", "ref")}))
            got += 1
            if got >= k:
                break
    out = dict(board=board, rule="per class: shipped (native) member tokens sorted by |loss - class median loss|, one token per log, "
                                 "no log reused across classes; loss = reference token score - shipped token score (clipped at 0)",
               stats=stats, picks=picks)
    (N.PREP / f"lbx_{board}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(stats, indent=1))
    for p in picks:
        print(p["cls"], p["token"], p["stage"], p["cmd"], f"loss {p['loss']:.3f} med {p['class_median']:.3f}", "best_fail", p["best_failing"], p["best_same_class"], p["other_classes"])


if __name__ == "__main__":
    main(sys.argv[1])
