"""navhard two-stage per-token devkit scoring of one pose file, for the guard's paired readouts (envs/navsim2, CPU).

Reuses the harness of experiments/skill_pack/scripts/hist_align_report.py: offroad_replay_cf._score (navsim's pdm_score with the
run_pdm_score.py simulator / scorer / reactive traffic) per token, offroad_replay_cf.aggregate (the devkit's
create_scene_aggregators + compute_final_scores + calculate_individual_mapping_scores). Per scene-mapping group (the unit the
devkit averages over) the devkit's calculate_individual_mapping_scores on that one group gives the group's combined / stage-1 /
stage-2 value: the mean over groups is the official number, so paired CIs resample groups.

  $DATA_DIR/envs/navsim2/bin/python experiments/op_guard/scripts/nav_harness.py --poses <preds.npz> --out <dir> --procs 24
  $DATA_DIR/envs/navsim2/bin/python experiments/op_guard/scripts/nav_harness.py --freeze-early   # decision 110's early-turn set
Out: <dir>/harness_tokens.csv (token, stage, group, score, weight, subscores), harness_groups.csv, harness_summary.json.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/skill_pack/scripts")]
import offroad_lib as L  # noqa: E402

_W = {}
EARLY = REPO / "experiments/op_guard/results/navhard_early_turn.csv"
REF = L.D / "runs/hugsim-spinattr/navhard_ref.csv"     # experiments/hugsim/scripts/spin_attr_navhard_ref.py (decision 110 Q4)


def _init(poses):
    import offroad_replay_cf as R
    R._init(argparse.Namespace(variants=[]))
    _W.update(R=R, P=L.poses_by_token(poses))


def work(token):
    R = _W["R"]
    return token, R._score(L.load_cache(R._W["cp"][token]), _W["P"][token], token)[0]


def freeze_early():
    """Tokens whose PDM reference heading change within 2 s is >= 5 deg (decision 110 point 6; spin_attr_q4.py's main definition)."""
    ref = pd.read_csv(REF)
    stage = {e["token"]: {"one": 1, "two": 2}[e["stage"]] for e in L.index()}
    ref["stage"] = ref.token.map(stage)
    ref = ref[ref.stage.isin([1, 2]) & (ref.pdm_h2.abs() >= 5)][["token", "stage", "pdm_h2"]].sort_values(["stage", "token"])
    EARLY.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(EARLY, index=False, float_format="%.4f")
    print(ref.stage.value_counts().sort_index().to_dict(), "of", pd.Series(stage).value_counts().sort_index().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses")
    ap.add_argument("--out")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--freeze-early", action="store_true")
    a = ap.parse_args()
    if a.freeze_early:
        return freeze_early()
    import offroad_replay_cf as R
    from dataclasses import fields
    from navsim.common.dataclasses import PDMResults
    from navsim.planning.script.run_pdm_score import calculate_individual_mapping_scores
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = L.index()
    tokens = [e["token"] for e in idx]
    stage = {e["token"]: {"one": 1, "two": 2}[e["stage"]] for e in idx}
    t0, rows = time.time(), {}
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.poses,)) as pool:
        for i, (t, r) in enumerate(pool.imap_unordered(work, tokens, chunksize=4)):
            rows[t] = r
            if i % 1000 == 0:
                print(i, f"{time.time() - t0:.0f} s", flush=True)
    t_score = time.time() - t0
    _, _, _, mapping, samp = L.setup_scoring()
    comb, s1, s2, df = R.aggregate([rows[t] for t in tokens], mapping, samp)
    cols = [c for c in df.columns if ((any(s.name in c for s in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score") and c != "pdm_score")]
    dfc = df[cols + ["token", "weight"]]
    grp, g_of = [], {}
    for g, (key, pairs) in enumerate(mapping.items()):
        c, a1, a2 = calculate_individual_mapping_scores(dfc, {key: pairs})
        grp.append(dict(group=g, orig=key[0], prev=key[1], combined=100 * float(c["score"]), stage1=100 * float(a1["score"]), stage2=100 * float(a2["score"])))
        for t in (key[0], key[1], *[x for p in pairs for x in p]):
            g_of.setdefault(t, g)
    G = pd.DataFrame(grp)
    assert abs(G.combined.mean() - 100 * float(comb["score"])) < 1e-6, (G.combined.mean(), comb["score"])
    tok = dfc.copy()
    tok["stage"] = tok.token.map(stage)
    tok["group"] = tok.token.map(g_of).fillna(-1).astype(int)
    tok[["token", "stage", "group", "score", "weight"] + [c for c in cols if c != "score"]].to_csv(out / "harness_tokens.csv", index=False)
    G.to_csv(out / "harness_groups.csv", index=False)
    summ = dict(poses=str(a.poses), combined=100 * float(comb["score"]), stage1=100 * float(s1["score"]), stage2=100 * float(s2["score"]),
                n_tokens=len(tokens), n_groups=len(G), n_ungrouped=int((tok.group < 0).sum()), score_s=round(t_score, 1), procs=a.procs)
    (out / "harness_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
