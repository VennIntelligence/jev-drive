"""Loss budget, step 2 (CPU, navsim2 env): oracle substitution on navtest (v1 PDMS) and navhard (two-stage EPDMS).

Reads the step-1 pickle (loss_budget_prep.py) and the official per-token results. For each arm (shipped Cinque = native, best = it_dw3-s0 +
selector sel-rot0-r0.6, decision 101) and each failure class, the class's tokens are replaced by the reference trajectory's token
scores (navhard: PDM-Closed reference scored in-process; navtest: the human trajectory's official CSV) and the board score is
recomputed with the official aggregation. EP and comfort are term levers (term set to 1), the reference is not used for them.
Every number is an ORACLE CEILING: tokens are replaced, never predicted.

  navsim2 env:  loss_budget_nav.py navtest|navhard   -> $DATA_DIR/runs/leaderboard_audit/loss_budget/<board>.json
Token rule (clip): a replaced token keeps its own scores where the reference token would score lower.
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/skill_pack/scripts")]
import offroad_lib as L  # noqa: E402

PREP = L.D / "runs/leaderboard_audit/loss_budget"
OUT = PREP            # json next to the prep pickles; copied into experiments/leaderboard_audit/results/loss_budget/
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
GATE = SUBS[:4]
TRAJ = SUBS[:7]                       # taken from the reference row; the two comfort terms stay the arm's own
COMF = ["history_comfort", "two_frame_extended_comfort"]
WT = dict(ego_progress=5, time_to_collision_within_bound=5, lane_keeping=2, history_comfort=2, two_frame_extended_comfort=2)
ARMS = ["native", "best"]
CLASSES = ["wrong_direction", "road_edge_wide", "early_clip", "collisions", "ep_progress", "comfort", "stopped_slow", "no_command"]
EXTRA = ["offroad_all", "other_gate", "ep_at_ref"]       # context rows: every DAC failure, and DDC / TLC failures (not in the requested list)
LOSS_MIN = 0.2          # failing token: arm score below (1 - LOSS_MIN) x the reference token score


def rescore(d, board):
    if board == "navtest":                  # v1 PDMS: NC x DAC x (5 EP + 5 TTC + 2 C) / 12 (DDC is reported only)
        m = d.no_at_fault_collisions * d.drivable_area_compliance
        return m * (5 * d.ego_progress + 5 * d.time_to_collision_within_bound + 2 * d.history_comfort) / 12
    m = np.prod([d[c] for c in GATE], axis=0)
    return m * sum(WT[k] * d[k] for k in WT) / 16


def board_score(d, board, mapping=None):
    if board == "navtest":
        return 100 * float(d.score.mean())
    from navsim.planning.script.run_pdm_score import calculate_individual_mapping_scores
    comb, _, _ = calculate_individual_mapping_scores(d[SUBS + ["score", "token", "weight"]], mapping)
    return 100 * float(comb["score"])


def load_csv_v1(name):
    d = pd.read_csv(sorted(glob.glob(str(L.D / "runs/navsim/eval" / name / "*/*.csv")))[-1]).set_index("token")
    d = d[d.index != "average"]
    d = d.rename(columns={"comfort": "history_comfort"})
    d["traffic_light_compliance"] = d["lane_keeping"] = d["two_frame_extended_comfort"] = 1.0
    return d


def tables(board, prep):
    """Per-arm and reference token tables (index = token), columns SUBS + score (+ weight, token, stage for navhard) and a mapping."""
    tokens = list(prep)
    if board == "navtest":
        csv = {"native": "v1_navtest_opi_lb_navtest_gimm-cinque__base", "best": "v1_navtest_opi_lb_navtest_gimm-cinque_Oit_dw3-s0_al-sel-rot0-r0.6__base",
               "ref": "v1_navtest_human"}
        T = {k: load_csv_v1(v).loc[tokens] for k, v in csv.items()}
        for k in T:
            T[k]["token"] = T[k].index
        return T, None
    import offroad_replay_cf as R
    _, _, _, mapping, samp = L.setup_scoring()
    T = {}
    for name in ["native", "best", "ref"]:
        comb, s1, s2, df = R.aggregate([prep[t]["rows"][name] for t in tokens], mapping, samp)
        T[name] = df.set_index("token", drop=False).loc[tokens]
        if name != "ref":
            print(board, name, "combined", 100 * float(comb["score"]), "s1", 100 * float(s1["score"]), "s2", 100 * float(s2["score"]), flush=True)
    return T, mapping


def geometry(board, prep, T):
    """Per-arm class flags (index = token)."""
    tokens = list(prep)
    if board == "navtest":
        idx = {e["token"]: e for e in pickle.load(open(L.D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))}
        stage = pd.Series(["one"] * len(tokens), index=tokens)
    else:
        idx = {e["token"]: e for e in L.index()}
        stage = pd.Series([idx[t]["stage"] for t in tokens], index=tokens)
    v0 = pd.Series([float(np.linalg.norm(idx[t]["vel"][-1])) for t in tokens], index=tokens)
    edge = None
    if board == "navhard":
        re = pd.read_pickle(L.OUT / "roadedge_table.pkl").set_index("token")
        edge = {"native": re["native_edge_err"]}
    out = {}
    for a in ARMS:
        d = pd.DataFrame(index=tokens)
        g = lambda f: pd.Series([f(prep[t]) for t in tokens], index=tokens)  # noqa: E731
        end_y, end_x, end_yaw = (g(lambda p, k=k: p["arms"][a][k]) for k in ("end_y", "end_x", "end_yaw"))
        ry, rx, ryaw = (g(lambda p, k=k: p["ref"][k]) for k in ("end_y", "end_x", "end_yaw"))
        first = g(lambda p: -1 if p["arms"][a]["first"] is None else p["arms"][a]["first"])
        coff, chead = g(lambda p: p["start"]["center_off"]), g(lambda p: p["start"]["center_head_err"])
        cmd = g(lambda p: p["cmd"])
        t, r = T[a], T["ref"]
        gates_ok = (t[GATE[:2] if board == "navtest" else GATE] >= 1).all(axis=1)
        failing = (t.score < (1 - LOSS_MIN) * r.score)
        opp = (end_y * ry < 0) & (end_y.abs() > 1) & (ry.abs() > 1) & ((end_y - ry).abs() > 2)
        s = np.sign(ryaw)
        missed = (ryaw.abs() >= 0.3) & (s * end_yaw < 0.5 * ryaw.abs()) & cmd.isin([0, 2])
        d["failing"] = failing
        d["wrong_direction"] = failing & opp
        d["early_clip"] = failing & (t.drivable_area_compliance < 1) & (first >= 0) & (first <= 15) & ((coff.abs() >= 0.5) | (chead.abs() >= 0.1))
        d["road_edge_wide"] = failing & (t.drivable_area_compliance < 1) & (stage == "two") & (edge[a].reindex(tokens) > 1.0).fillna(False) if (edge and a in edge) else np.nan
        d["collisions"] = failing & ((t.no_at_fault_collisions < 1) | (t.time_to_collision_within_bound < 1))
        d["ep_progress"] = failing & (t.ego_progress < 0.8)
        d["stopped_slow"] = failing & (t.ego_progress < 0.5) & ((end_x < 0.5 * rx) | ((v0 < 1) & (end_x < 2)))
        d["comfort"] = failing & ((t.history_comfort < 1) | (t.two_frame_extended_comfort < 1))
        d["no_command"] = failing & missed
        d["ep_at_ref"] = failing & (t.ego_progress < 0.8)
        d["offroad_all"] = failing & (t.drivable_area_compliance < 1)
        d["other_gate"] = failing & ~gates_ok & ~d.offroad_all & (t.no_at_fault_collisions >= 1)
        # primary exclusive class of every failing token (for the shares that add up): order wrong_direction, no_command, offroad, collision, gates, stopped, progress
        pri = pd.Series("other", index=tokens)
        for c, m in (("ep_progress", d.ep_progress), ("stopped_slow", d.stopped_slow), ("other_gate", failing & ~gates_ok), ("collisions", d.collisions),
                     ("offroad_other", failing & (t.drivable_area_compliance < 1)), ("early_clip", d.early_clip), ("no_command", d.no_command),
                     ("wrong_direction", d.wrong_direction)):
            pri[m] = c
        d["primary"] = pri.where(failing, "")
        d["stage"] = stage
        out[a] = d
    return out


def apply(t, r, masks, board):
    """Token table t with `masks` applied: traj (replace by the reference token), ep (EP -> 1), comfort (HC, EC -> 1)."""
    x = t.copy()
    traj = masks.get("traj")
    if traj is not None and traj.any():
        cand = x.copy()
        cand.loc[traj, TRAJ] = r.loc[traj, TRAJ].to_numpy()
        cand["score"] = rescore(cand, board)
        better = traj & (cand.score > x.score)                 # clip: never worse than the arm's own token
        x.loc[better, TRAJ] = r.loc[better, TRAJ].to_numpy()
    if masks.get("ep") is not None:
        x.loc[masks["ep"], "ego_progress"] = 1.0
    if masks.get("epref") is not None:       # progress of the reference token where it is higher
        m = masks["epref"] & (r.ego_progress > x.ego_progress)
        x.loc[m, "ego_progress"] = r.loc[m, "ego_progress"]
    if masks.get("comf") is not None:
        x.loc[masks["comf"], COMF] = 1.0
    x["score"] = rescore(x, board)
    return x


def run(board):
    prep = pickle.load(open(PREP / f"prep_{board}.pkl", "rb"))
    T, mapping = tables(board, prep)
    G = geometry(board, prep, T)
    res = {}
    for a in ARMS:
        t, g, r = T[a], G[a], T["ref"]
        base_given = board_score(t, board, mapping)
        base = board_score(apply(t, r, {}, board), board, mapping)       # recomputed from the sub-scores: must equal the official score
        n_fail = int(g.failing.sum())
        res[a] = dict(score=base_given, recomputed=base, n=len(t), n_failing=n_fail, rows={})
        mm = {}
        for c in CLASSES + EXTRA:
            if c in ("comfort",):
                mm[c] = dict(comf=pd.Series(True, index=t.index))
            elif c == "ep_progress":
                mm[c] = dict(ep=(t.ego_progress < 0.8))
            elif c == "ep_at_ref":
                mm[c] = dict(epref=(t.ego_progress < 0.8))
            elif g[c].isna().all():
                mm[c] = None
            else:
                mm[c] = dict(traj=g[c].astype(bool))
        for c in CLASSES + EXTRA:
            if mm[c] is None:
                res[a]["rows"][c] = dict(n=None, share=None, ceiling=None)
                continue
            m = mm[c]
            res[a]["rows"][c] = dict(n=int(g[c].astype(bool).sum()), share=float(g[c].astype(bool).sum() / max(n_fail, 1)),
                                     ceiling=board_score(apply(t, r, m, board), board, mapping) - base)
        # all together: union of the trajectory classes + EP term + comfort term
        allm = dict(traj=pd.concat([g[c].astype(bool) for c in CLASSES if c not in ("comfort", "ep_progress") and not g[c].isna().all()], axis=1).any(axis=1),
                    ep=(t.ego_progress < 0.8), comf=pd.Series(True, index=t.index))
        res[a]["all"] = board_score(apply(t, r, allm, board), board, mapping) - base
        res[a]["all_traj_only"] = board_score(apply(t, r, dict(traj=allm["traj"]), board), board, mapping) - base
        # unclipped (plain replacement) ceiling of the union of trajectory classes, as a sensitivity
        x = t.copy()
        x.loc[allm["traj"], TRAJ] = r.loc[allm["traj"], TRAJ].to_numpy()
        x["score"] = rescore(x, board)
        res[a]["all_traj_unclipped"] = board_score(x, board, mapping) - base
        res[a]["primary"] = {k: int(v) for k, v in g.primary.value_counts().items() if k}
        if board == "navhard":
            res[a]["stage2_share_failing"] = float((g.failing & (g.stage == "two")).sum() / max(n_fail, 1))
    res["reference_score"] = board_score(T["ref"], board, mapping)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{board}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    run(sys.argv[1])
