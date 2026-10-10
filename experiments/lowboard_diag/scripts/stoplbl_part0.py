"""stoplbl Part 0: how many EPDMS points the navtest / navhard recipes lose in the three situations (red light / stop line, yield / stop sign, sharp turn).

  .venv/bin/python experiments/lowboard_diag/scripts/stoplbl_part0.py IN_DIR OUT_DIR
IN_DIR holds copies of the box files: navtest.parquet (stoplbl_label.py), nt_<arm>.csv (bench navtest units.csv), nhu_<arm>.csv + nh_<arm>.csv (navhard
units.csv + harness/harness_tokens.csv), strata_navtest.csv (bench strata). Arms: seed means per recipe. Situation = hindsight-route label at the token
(see stoplbl_label.py); a token is in a situation when the target is within 50 m along the logged route.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.bench.tables import TERMS, score_of, shapley  # noqa: E402

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SUBS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
NT = {"SH30": ["SH30-F-s0", "SH30-F-s1"], "P2H10": ["P2H10-F-s0", "P2H10-F-s1"], "P2H10S": ["P2H10S-F-s0", "P2H10S-F-s1"]}
NH = {"SH30": ["SH30-F-s0", "SH30-F-s1"], "P2H10": ["P2H10-F-s2", "P2H10-F-s3"], "P2H10S": ["P2H10S-F-s0", "P2H10S-F-s1"]}
rng = np.random.default_rng(0)

L = pd.read_parquet(IN / "navtest.parquet").set_index("token")
L_all = L
L = L.loc[L.index.isin(pd.read_csv(IN / "nt_SH30-F-s0.csv").token)]      # the 12 146 navtest tokens
ok = (L.full | (L.s_end >= 50))
sit = pd.DataFrame(index=L.index)
sit["red50"] = L.tl_red_d <= 50
sit["red50_stopped"] = sit.red50 & (L.tl_red_vmin < 1.0)
sit["red50_passed_fast"] = sit.red50 & (L.tl_red_vmin >= 3.0)
sit["tlline50"] = L.tl_line_d <= 50
sit["stopsign50"] = L.ss_d <= 50
sit["yield50"] = (L.ts_d <= 50) | (L.yl_d <= 50)
sit["pedcw50"] = L.cwped_d <= 50
sit["turn45_map50"] = (L.turn_deg >= 45) & (L.turn_s <= 50)
st = pd.read_csv(IN / "strata_navtest.csv").set_index("token")
sit["turn_gt45_logged"] = st.turn.reindex(sit.index).eq(">45")
sit["stoptarget"] = sit.red50 | sit.stopsign50 | sit.yield50 | sit.pedcw50
sit["valid"] = ok
# exclusive, priority red > stop sign > yield > ped crossing > turn > rest
ex = np.select([sit.red50, sit.stopsign50, sit.yield50, sit.pedcw50, sit.turn_gt45_logged], ["red", "stop_sign", "yield", "ped_cw", "turn>45"], "rest")
sit["excl"] = ex


def load_nt(arms):
    ds = [pd.read_csv(IN / f"nt_{a}.csv").set_index("token")[SUBS] for a in arms]
    return sum(ds) / len(ds)


def boot_excess(x, mask, logs, B=1000):
    """cluster (log) bootstrap CI of  n_set/N * (mean_rest - mean_set)  in score points."""
    ul = np.unique(logs)
    idx = {u: np.where(logs == u)[0] for u in ul}
    out = []
    for _ in range(B):
        pick = np.concatenate([idx[u] for u in rng.choice(ul, len(ul))])
        xs, ms = x[pick], mask[pick]
        if ms.sum() == 0 or (~ms).sum() == 0:
            continue
        out.append(ms.mean() * (xs[~ms].mean() - xs[ms].mean()))
    return np.percentile(out, [2.5, 97.5])


rows, subrows, tlc_rows, term_rows = [], [], [], []
for rec, arms in NT.items():
    S = load_nt(arms).reindex(sit.index)
    X = S[SUBS].values
    sc = 100 * score_of(X)
    N = len(S)
    logs = L.log.values
    tot_loss = (100 - sc).sum()
    ph = shapley(X, np.ones_like(X)) * 100
    for j, t in enumerate(SUBS):
        subrows.append(dict(board="navtest", recipe=rec, term=t, mean_sub=np.nanmean(X[:, j]), loss_pts=ph[:, j].mean(), share_of_loss=ph[:, j].sum() / tot_loss,
                            fail_frac=float(np.nanmean(X[:, j] < 0.999))))
    # per-term split of the excess loss of each situation (Shapley from perfect, set mean minus rest mean, times the set fraction)
    for name in ["red50", "stopsign50", "yield50", "pedcw50", "stoptarget", "turn45_map50", "turn_gt45_logged"]:
        m = sit[name].values & sit.valid.values
        rest = ~sit[name].values & sit.valid.values
        for j, t in enumerate(SUBS):
            term_rows.append(dict(recipe=rec, situation=name, term=t, loss_in_set_pts=ph[m, j].sum() / N,
                                  excess_pts=m.mean() * (ph[m, j].mean() - ph[rest, j].mean())))
    # TLC detail
    tlc_fail = X[:, 3] < 0.5
    tlc_rows.append(dict(recipe=rec, n_tlc_fail=int(tlc_fail.sum()), frac=tlc_fail.mean(), score_on_fail=sc[tlc_fail].mean() if tlc_fail.any() else np.nan,
                         red50_among_fail=float(sit.red50.values[tlc_fail].mean()) if tlc_fail.any() else np.nan,
                         tlline50_among_fail=float(sit.tlline50.values[tlc_fail].mean()) if tlc_fail.any() else np.nan,
                         fail_in_red50=int((tlc_fail & sit.red50.values).sum()), n_red50=int(sit.red50.sum()),
                         shapley_tlc_pts=ph[:, 3].mean()))
    for name in ["red50", "red50_stopped", "red50_passed_fast", "tlline50", "stopsign50", "yield50", "pedcw50", "stoptarget", "turn45_map50", "turn_gt45_logged"]:
        m = sit[name].values & sit.valid.values
        rest = ~sit[name].values & sit.valid.values
        if m.sum() == 0:
            continue
        r = dict(board="navtest", recipe=rec, situation=name, n=int(m.sum()), frac=m.mean(), epdms_set=sc[m].mean(), epdms_rest=sc[rest].mean(),
                 loss_in_set_pts=(100 - sc[m]).sum() / N, share_of_board_loss=(100 - sc[m]).sum() / tot_loss,
                 excess_pts=m.mean() * (sc[rest].mean() - sc[m].mean()))
        lo, hi = boot_excess(sc, m, logs)
        r.update(excess_lo=lo, excess_hi=hi)
        for j, t in enumerate(SUBS[:7]):
            r[f"{t}_set"], r[f"{t}_rest"] = np.nanmean(X[m, j]), np.nanmean(X[rest, j])
        rows.append(r)
    # exclusive partition
    for g in ["red", "stop_sign", "yield", "ped_cw", "turn>45", "rest"]:
        m = (sit.excl.values == g) & sit.valid.values
        rows.append(dict(board="navtest", recipe=rec, situation="EXCL:" + g, n=int(m.sum()), frac=m.mean(), epdms_set=sc[m].mean(),
                         loss_in_set_pts=(100 - sc[m]).sum() / N, share_of_board_loss=(100 - sc[m]).sum() / tot_loss,
                         **{f"{t}_set": np.nanmean(X[m, j]) for j, t in enumerate(SUBS[:7])}))
    rows.append(dict(board="navtest", recipe=rec, situation="ALL", n=N, frac=1.0, epdms_set=sc.mean(), loss_in_set_pts=(100 - sc).sum() / N, share_of_board_loss=1.0))

# ---- navhard: group-level, situation of the group's stage-1 tokens (any), TLC rates per stage
for rec, arms in NH.items():
    us = [pd.read_csv(IN / f"nhu_{a}.csv").set_index("group") for a in arms]
    U = sum(u[["combined", "stage1", "stage2"]] for u in us) / len(us)
    h0 = pd.read_csv(IN / f"nh_{arms[0]}.csv")
    s1 = h0[h0.stage == 1][["group", "token"]]
    s1 = s1.join(sit[["red50", "stopsign50", "yield50", "pedcw50", "turn_gt45_logged", "stoptarget", "valid"]], on="token")
    G = s1.groupby("group")[["red50", "stopsign50", "yield50", "pedcw50", "turn_gt45_logged", "stoptarget"]].max().astype(bool).reindex(U.index)
    cm = U.combined.values
    tot_loss = (100 - cm).sum()
    lg = us[0].log.values
    for name in ["red50", "stopsign50", "yield50", "pedcw50", "stoptarget", "turn_gt45_logged"]:
        m = G[name].values
        if m.sum() == 0:
            continue
        rest = ~m
        lo, hi = boot_excess(cm, m, lg)
        rows.append(dict(board="navhard", recipe=rec, situation=name, n=int(m.sum()), frac=m.mean(), epdms_set=cm[m].mean(), epdms_rest=cm[rest].mean(),
                         loss_in_set_pts=(100 - cm[m]).sum() / len(cm), share_of_board_loss=(100 - cm[m]).sum() / tot_loss,
                         excess_pts=m.mean() * (cm[rest].mean() - cm[m].mean()), excess_lo=lo, excess_hi=hi,
                         stage1_set=U.stage1.values[m].mean(), stage1_rest=U.stage1.values[rest].mean(),
                         stage2_set=U.stage2.values[m].mean(), stage2_rest=U.stage2.values[rest].mean()))
    rows.append(dict(board="navhard", recipe=rec, situation="ALL", n=len(cm), frac=1.0, epdms_set=cm.mean(), loss_in_set_pts=(100 - cm).sum() / len(cm), share_of_board_loss=1.0))
    # TLC by stage over the seeds' token tables
    for stg in (1, 2):
        f = []
        for a in arms:
            h = pd.read_csv(IN / f"nh_{a}.csv")
            h = h[h.stage == stg]
            f.append((h.traffic_light_compliance < 0.5).mean())
        tlc_rows.append(dict(recipe=rec, board=f"navhard stage {stg}", frac=float(np.mean(f)),
                             n_tlc_fail=int(round(np.mean(f) * (h.shape[0])))))
# navhard stage-2 token-level Shapley of TLC (points of the stage-2 token mean)
cols = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
for rec, arms in NH.items():
    for stg in (1, 2):
        sh = []
        for a in arms:
            h = pd.read_csv(IN / f"nh_{a}.csv")
            h = h[h.stage == stg]
            X = h[cols].values.astype(float)
            ph = shapley(X, np.ones_like(X)) * 100
            sh.append((ph[:, 3].mean(), ph.sum(1).mean()))
        tlc_rows.append(dict(recipe=rec, board=f"navhard stage {stg} token-level", shapley_tlc_pts=np.mean([x[0] for x in sh]), n_tlc_fail=-1,
                             frac=np.mean([x[0] / x[1] for x in sh])))
pd.DataFrame(term_rows).to_csv(OUT / "part0_term_excess_navtest.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "part0_situations.csv", index=False)
pd.DataFrame(subrows).to_csv(OUT / "part0_subscore_loss_navtest.csv", index=False)
pd.DataFrame(tlc_rows).to_csv(OUT / "part0_tlc.csv", index=False)
sit.assign(valid=sit.valid).to_parquet(OUT.parent.parent.joinpath("results/stoplbl/_sit_tmp.parquet")) if False else None
sit.to_csv(IN / "navtest_situations.csv")
print("ok")
