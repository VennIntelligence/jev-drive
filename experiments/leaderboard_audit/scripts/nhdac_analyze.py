"""navhard DAC attribution, step 2 (any machine with pandas). Pre-registration: plans/2026-10-04-navhard-dac-prereg.md.

Reads feat.pkl and scores.pkl (nhdac_feat.py), classifies every official DAC failure of each arm into one primary cause
(pre-registered order) plus overlapping flags, and computes oracle ceilings per cause with the devkit two-stage aggregation
(re-implemented in numpy and checked against the devkit numbers): (a) the cause's DAC-failing tokens take the PDM reference's
trajectory terms (clipped: never worse than the arm's own token; comfort stays the arm's), (b) only DAC set to 1.
CIs: 2 000 bootstrap resamples of the two-stage mapping groups.

  python nhdac_analyze.py <dir with feat.pkl, scores.pkl>   -> <dir>/nhdac.json, <dir>/nhdac_tokens.csv
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GATE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"]
WT = dict(ego_progress=5, time_to_collision_within_bound=5, lane_keeping=2, history_comfort=2, two_frame_extended_comfort=2)
TRAJ = GATE + ["ego_progress", "time_to_collision_within_bound", "lane_keeping"]
D5, D10 = np.deg2rad(5), np.deg2rad(10)
ORDER = ["start_outside", "map_narrow", "wrong_direction", "tracker_lag", "early_start_offset", "under_turn", "over_turn", "too_fast",
         "lane_change", "other"]
FLAGS = ["infeasible", "ref_fails", "d110_set", "junction", "bend", "straight", "cmd_fixable", "synthetic_history", "shallow", "walkway"]
ARMS = ["native", "best"]
NB, SEED = 2000, 0


def rescore(d):
    return np.prod([d[c].to_numpy() for c in GATE], axis=0) * sum(w * d[k].to_numpy() for k, w in WT.items()) / 16


class Agg:
    """Two-stage aggregation of calculate_individual_mapping_scores, vectorised over groups."""

    def __init__(self, mapping, tokens, weight):
        ix = {t: i for i, t in enumerate(tokens)}
        self.w = weight
        self.g = []
        for (o, p), pairs in mapping.items():
            f = [ix[a[0]] for a in pairs if len(a) > 0]
            s = [ix[a[1]] for a in pairs if len(a) > 1]
            self.g.append((ix[o], np.array(f, int), ix[p], np.array(s, int)))
        n = len(self.g)
        self.member = np.zeros((len(tokens), n), np.float32)        # token x group incidence (for weighted token counts)
        for k, (o, f, p, s) in enumerate(self.g):
            self.member[[o, p, *f, *s], k] = 1

    def wavg(self, x, idx):
        return np.nan if len(idx) == 0 else float((x[idx] * self.w[idx]).sum() / self.w[idx].sum())

    def groups(self, score):
        return np.array([(score[o] * self.wavg(score, f) + score[p] * self.wavg(score, s)) / 2 for o, f, p, s in self.g])


def boot_idx(n):
    return np.random.default_rng(SEED).integers(0, n, (NB, n))


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def classify(feat, T, arm):
    rows = []
    for t, r in feat.items():
        f, rf, st = r["arms"][arm], r["ref"], r["start"]
        dac = float(T[arm].loc[t, "drivable_area_compliance"])
        x = dict(token=t, stage=r["stage"], cmd=r["cmd"], dac=dac, dac_geom=f["dac_geom"], ref_dac=float(T["ref"].loc[t, "drivable_area_compliance"]),
                 r_dpsi2=rf["dpsi2"], r_dpsi4=rf["dpsi4"], r_y4=rf["y4"], r_s4=rf["s4"], p_dpsi2=f["dpsi2"], p_dpsi4=f["dpsi4"], p_s4=f["s4"],
                 end_y=f["end_y"], ref_end_y=r["ref_end_y"], center_off=st["center_off"], head_err=st["center_head_err"])
        for k in ("first", "start_outside", "side", "d_worst", "junction", "map_narrow", "walk", "raw_ok", "retime_ok", "rot0_ok", "feasible", "n_feasible"):
            x[k] = f.get(k, np.nan)
        rows.append(x)
    d = pd.DataFrame(rows).set_index("token")
    fail = d.dac < 1
    ref_turn = (d.r_dpsi4.abs() >= D10) | (d.r_dpsi2.abs() >= D5)
    s = np.sign(np.where(d.r_dpsi4.abs() >= D10, d.r_dpsi4, d.r_dpsi2))
    ratio = s * d.p_dpsi4 / d.r_dpsi4.abs().clip(lower=1e-6)
    outside = d.side * s < 0
    c = {}
    c["start_outside"] = d.start_outside == True  # noqa: E712
    c["map_narrow"] = d.map_narrow == True  # noqa: E712
    c["wrong_direction"] = (d.end_y * d.ref_end_y < 0) & (d.end_y.abs() > 1) & (d.ref_end_y.abs() > 1) & ((d.end_y - d.ref_end_y).abs() > 2)
    c["tracker_lag"] = d.raw_ok == True  # noqa: E712
    c["early_start_offset"] = (d.stage == "two") & (d["first"] <= 15) & ((d.center_off.abs() >= 0.5) | (d.head_err.abs() >= 0.1))
    c["under_turn"] = ref_turn & outside & (ratio < 0.75)
    c["over_turn"] = (ref_turn & ~outside & (ratio > 1.25)) | ((d.r_dpsi4.abs() < D5) & (d.p_dpsi4.abs() >= D10))
    c["too_fast"] = (d.p_s4 >= 1.15 * d.r_s4) & (d.retime_ok == True)  # noqa: E712
    c["lane_change"] = (d.r_y4.abs() >= 2) & (d.r_dpsi4.abs() < D10)
    c["other"] = pd.Series(True, index=d.index)
    pri = pd.Series("", index=d.index)
    for k in reversed(ORDER):
        pri[c[k]] = k
    d["primary"] = pri.where(fail, "")
    for k in ORDER[:-1]:
        d["c_" + k] = c[k] & fail
    d["ratio"], d["ref_turn"], d["outside"] = ratio, ref_turn, outside
    fl = {"ref_fails": d.ref_dac < 1, "d110_set": d.r_dpsi2.abs() >= D5, "junction": d.junction == True,  # noqa: E712
          "bend": (d.junction != True) & (d.r_dpsi4.abs() >= D10), "straight": (d.junction != True) & (d.r_dpsi4.abs() < D10),  # noqa: E712
          "cmd_fixable": (c["under_turn"] | c["wrong_direction"]) & d.cmd.isin([0, 2]) & (np.where(d.cmd == 0, 1, -1) == s),
          "synthetic_history": (d.stage == "two") & (d.rot0_ok == True), "shallow": d.d_worst < 0.3, "walkway": d.walk == True,  # noqa: E712
          "infeasible": d.feasible == False}  # noqa: E712
    for k, v in fl.items():
        d["f_" + k] = v & fail
    return d


def main(dd):
    dd = Path(dd)
    feat = pickle.load(open(dd / "feat.pkl", "rb"))
    S = pickle.load(open(dd / "scores.pkl", "rb"))
    tokens = list(S["native"].index)
    T = {a: S[a].loc[tokens].copy() for a in ARMS + ["ref"]}
    for a in T:
        assert np.allclose(rescore(T[a]), T[a].score.to_numpy(), atol=1e-6), a
    AG = {a: Agg(S["mapping"], tokens, T[a].weight.to_numpy()) for a in ARMS + ["ref"]}      # stage-2 weights depend on the arm's stage-1 plan
    B = boot_idx(len(AG["native"].g))
    out = {"check": {}, "arms": {}}
    for a in ARMS + ["ref"]:
        agg = AG[a]
        mine = 100 * np.nanmean(agg.groups(T[a].score.to_numpy()))
        out["check"][a] = dict(fast=mine, devkit=100 * S[a + "_official"]["combined"])
        print(a, out["check"][a])
    allrows = []
    for a in ARMS:
        t, r, agg = T[a], T["ref"], AG[a]
        d = classify(feat, T, a).loc[tokens]
        fail = (d.dac < 1).to_numpy()
        base_g = agg.groups(t.score.to_numpy())
        base = 100 * np.nanmean(base_g)
        res = dict(score=base, n_fail=int(fail.sum()), n_fail_s1=int((fail & (d.stage == "one")).sum()), n_fail_s2=int((fail & (d.stage == "two")).sum()),
                   geom_agree=float((d.dac_geom.to_numpy() == (d.dac >= 1).to_numpy()).mean()),
                   geom_fail_official_pass=int(((~d.dac_geom) & (d.dac >= 1)).sum()), official_fail_geom_pass=int((d.dac_geom & (d.dac < 1)).sum()),
                   n_tokens=len(d), rows={}, flags={}, cross={})
        mult = np.stack([np.bincount(b, minlength=len(agg.g)) for b in B]).astype(np.float32)            # (NB, groups)
        deg = agg.member.sum(1).clip(min=1)
        tokw = (mult @ agg.member.T) / deg                                                                   # (NB, tokens)
        nf_b = tokw @ fail.astype(np.float32)

        def oracle(mask, mode):
            x = t.copy()
            m = mask.copy()
            if mode == "ref":
                cand = x.copy()
                cand.loc[m, TRAJ] = r.loc[m, TRAJ].to_numpy()
                better = m & (rescore(cand) > x.score.to_numpy())
                x.loc[better, TRAJ] = r.loc[better, TRAJ].to_numpy()
            else:
                x.loc[m, "drivable_area_compliance"] = 1.0
            g = agg.groups(rescore(x))
            dg = g - base_g
            ok = ~np.isnan(dg)
            boots = np.array([100 * np.nanmean(dg[b]) for b in B])
            return 100 * float(np.nanmean(dg[ok])), ci(boots)

        def row(mask):
            n = int(mask.sum())
            sh = tokw @ mask.astype(np.float32) / nf_b
            o_ref, o_ref_ci = oracle(mask, "ref")
            o_dac, o_dac_ci = oracle(mask, "dac")
            return dict(n=n, n_s1=int((mask & (d.stage == "one").to_numpy()).sum()), n_s2=int((mask & (d.stage == "two").to_numpy()).sum()),
                        share=n / max(fail.sum(), 1), share_ci=ci(sh), ref_fails=int((mask & (d.ref_dac < 1).to_numpy()).sum()),
                        oracle_ref=o_ref, oracle_ref_ci=o_ref_ci, oracle_dac1=o_dac, oracle_dac1_ci=o_dac_ci)

        for k in ORDER:
            res["rows"][k] = row((d.primary == k).to_numpy())
            print(a, k, {q: (round(v, 3) if isinstance(v, float) else v) for q, v in res["rows"][k].items()}, flush=True)
        res["rows"]["all_dac"] = row(fail)
        pr = d.primary.to_numpy()
        inf = d.f_infeasible.to_numpy()
        for name, m in (("grp_scorer_M_L", np.isin(pr, ["map_narrow", "tracker_lag"])), ("grp_start_E", pr == "early_start_offset"),
                        ("grp_plan_W_U_O_F_C_R", np.isin(pr, ["wrong_direction", "under_turn", "over_turn", "too_fast", "lane_change", "other"])),
                        ("E_infeasible", (pr == "early_start_offset") & inf), ("E_feasible", (pr == "early_start_offset") & ~inf),
                        ("scorer_or_infeasible", np.isin(pr, ["map_narrow", "tracker_lag"]) | (fail & inf))):
            res["rows"][name] = row(m)
        for k in ORDER[:-1]:
            res["flags"]["any_" + k] = int(d["c_" + k].sum())
        for k in FLAGS:
            res["flags"][k] = row(d["f_" + k].to_numpy()) if k in ("cmd_fixable", "synthetic_history", "ref_fails", "shallow", "infeasible") else int(d["f_" + k].sum())
        res["cross"]["primary_x_geometry"] = pd.crosstab(d.primary[fail], np.select([d.f_junction[fail], d.f_bend[fail]], ["junction", "bend"], "straight")).to_dict()
        res["cross"]["primary_x_d110"] = pd.crosstab(d.primary[fail], d.f_d110_set[fail]).to_dict()
        res["cross"]["primary_x_stage"] = pd.crosstab(d.primary[fail], d.stage[fail]).to_dict()
        res["cross"]["primary_x_reffail"] = pd.crosstab(d.primary[fail], d.f_ref_fails[fail]).to_dict()
        res["cross"]["primary_x_infeasible"] = pd.crosstab(d.primary[fail], d.f_infeasible[fail]).to_dict()
        res["cross"]["primary_x_walkway"] = pd.crosstab(d.primary[fail], d.f_walkway[fail]).to_dict()
        res["cross"]["primary_x_shallow"] = pd.crosstab(d.primary[fail], d.f_shallow[fail]).to_dict()
        # base rate of the d110 set among all tokens vs DAC failures
        res["d110_rate_all"] = float((d.r_dpsi2.abs() >= D5).mean())
        res["d110_rate_fail"] = float((d.r_dpsi2.abs() >= D5)[fail].mean())
        res["turn_ratio_fail_refturn_median"] = float(d.ratio[fail & d.ref_turn.to_numpy()].median())
        res["turn_ratio_pass_refturn_median"] = float(d.ratio[~fail & d.ref_turn.to_numpy()].median())
        out["arms"][a] = res
        d["arm"] = a
        allrows.append(d)
    # paired native -> best per primary class: tokens that leave / enter the class
    dn, db = allrows[0], allrows[1]
    out["paired"] = {k: dict(native=int((dn.primary == k).sum()), best=int((db.primary == k).sum()),
                             fixed=int(((dn.primary == k) & (db.dac >= 1)).sum()), moved=int(((dn.primary == k) & (db.dac < 1) & (db.primary != k)).sum()),
                             new=int(((db.primary == k) & (dn.dac >= 1)).sum())) for k in ORDER}
    pd.concat(allrows).to_csv(dd / "nhdac_tokens.csv")
    (dd / "nhdac.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["paired"], indent=0))


if __name__ == "__main__":
    main(sys.argv[1])
