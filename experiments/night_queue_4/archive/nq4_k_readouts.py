"""Night queue 4, K: the capability readouts of the recipe ladder K0-K3 on the stored open-loop exports
(fc65452:todos/2026-09-26-night-queue-4.md, K: "每级在 P5 v1 BA（开环，按同一交叉拟合出预测）、P6 bypass、I3 上各出一次", criterion 1).

Only the registered exam code, unchanged, on runs/nq4/k/openloop/{ba,i3,p6}_<level>.npz (no refit, no re-export):
  P5  p5_exam.exam + reactivity_mc.criteria on the P5 v1 BA pairs, cross-fitted `unseen` predictions; the level - K0
      paired pedestrian flip difference with criteria's paired recipe (same frames, route bootstrap)
  I3  elicit_i3.exam's judge (p5_exam.exam without the TFv6 columns) on the HUGSIM pairs, main readout R1 (R2 descriptive);
      the level - K0 paired pooled flip difference as elicit_i3's paired_vs_prior
  P6  nq3_p6.judge_one (night-queue-3 rule 7, as Q1) on the P6 v0 exam, the route's unseen readout; level - K0 paired
      bypass-flip / stop-substitution differences with nq3_p6.paired_diff (descriptive: P6 is not in criterion 1)
Criterion 1's capability half, mechanically: the paired CI inside [-10, +10] pp = "不动", otherwise "动了".

    python -m experiments.night_queue_4.archive.nq4_k_readouts            -> runs/nq4/k/readouts/<stamp>/*.csv|md|json
"""
import json

import numpy as np
import pandas as pd

from jevdrive.common import data_dir

LEVELS = ("K0", "K1", "K2", "K3")
REF = "K0"
BAND = 0.10                                   # criterion 1: +-10 pp


def _openloop(kind: str, level: str) -> dict:
    return dict(np.load(data_dir() / "runs" / "nq4" / "k" / "openloop" / f"{kind}_{level}.npz", allow_pickle=True))


def _flip(sub: pd.DataFrame, ex: str, tau: float) -> np.ndarray:
    """reactivity_mc.criteria / elicit_i3's per-frame directional flip."""
    from jevdrive import p5_exam as E
    return ((np.sign(sub[ex]) == np.sign(sub.d_expert)) & E._moved(sub[ex], tau)).astype(float).to_numpy()


def _band(lo: float, hi: float) -> str:
    return "不动" if lo >= -BAND and hi <= BAND else "动了"


def p5(rl) -> tuple[pd.DataFrame, pd.DataFrame]:
    from jevdrive import elicit_i3 as I, p5_exam as E, reactivity_mc as MC
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
    preds = {}
    for lv in LEVELS:
        z = _openloop("ba", lv)
        assert (z["frame_name"].astype(str) == t.frame_name.to_numpy()).all(), f"ba_{lv}: frame order"
        preds[lv] = z["unseen"].astype(np.float64)
    oo, nn = E.deltas(obs, null, t, preds)
    res = E.exam(oo, nn, pairs, list(LEVELS))
    crit = MC.criteria(res, LEVELS, REF)
    r = res["obs"][res["obs"].reactive]
    ped = r[r.family.isin(E.PED_FAMILIES)]
    rows = []
    for lv in LEVELS:
        d, lo, hi = E.boot_ratio(_flip(ped, lv, res["taus"][lv]) - _flip(ped, REF, res["taus"][REF]), np.ones(len(ped)),
                                 ped.base_id.to_numpy())
        rows.append({"level": lv, "ped_delta_vs_K0": d, "ped_delta_lo": lo, "ped_delta_hi": hi,
                     "ped_verdict": "reference" if lv == REF else _band(lo, hi)})
    crit = crit.rename(columns={"arm": "level"}).merge(pd.DataFrame(rows), on="level")
    crit.insert(1, "tau_model", crit.level.map(res["taus"]))
    rl.log.info("P5: tau_exp %.3f, pooled families %s, pedestrian reactive frames %d on %d routes", res["tau_exp"],
                res["pooled_families"], len(ped), ped.base_id.nunique())
    return crit, res["flips"]


def i3(rl) -> tuple[pd.DataFrame, pd.DataFrame]:
    from jevdrive import elicit_i3 as I, p5_exam as E
    with I.p5_set(I.I3):
        ta, pa, _, obs3, null3, pairs3 = E.load()
        t3 = ta[ta.frame_name.isin(set(I.needed())).to_numpy()].reset_index(drop=True)
    E.TFV6 = {}                                               # elicit_i3.exam: I3 has no TFv6 columns
    summ, flips = [], []
    for fold in ("R1", "R2"):
        preds = {}
        for lv in LEVELS:
            z = _openloop("i3", lv)
            assert (z["frame_name"].astype(str) == t3.frame_name.to_numpy()).all(), f"i3_{lv}: frame order"
            preds[lv] = z[fold].astype(np.float64)
        oo, nn = E.deltas(obs3, null3, t3, preds)
        res = E.exam(oo, nn, pairs3, list(LEVELS))
        fl = res["flips"].assign(readout=fold)
        flips.append(fl)
        r = res["obs"][res["obs"].reactive]
        pooled = r[r.family.isin(res["pooled_families"])]
        for lv in LEVELS:
            p = fl[(fl.examinee == lv) & (fl.scope == "pooled")].iloc[0]
            d, lo, hi = E.boot_ratio(_flip(pooled, lv, res["taus"][lv]) - _flip(pooled, REF, res["taus"][REF]),
                                     np.ones(len(pooled)), pooled.base_id.to_numpy())
            summ.append({"readout": fold, "level": lv, "tau_model": p.tau_model, "n_reactive": p.n_reactive,
                         "routes_reactive": p.routes_reactive, "flip": p.flip_rate, "flip_lo": p.flip_lo,
                         "flip_hi": p.flip_hi, "false_flip_nonreactive": p.false_flip_nonreactive,
                         "null_ff_oos": p.false_flip_null_oos, "delta_vs_K0": d, "delta_lo": lo, "delta_hi": hi,
                         "verdict": "reference" if lv == REF else _band(lo, hi) if fold == "R1" else "descriptive (R2)"})
        rl.log.info("I3 %s: tau_exp %.3f, pooled families %s, %d reactive frames", fold, res["tau_exp"],
                    res["pooled_families"], len(pooled))
    return pd.DataFrame(summ), pd.concat(flips, ignore_index=True)


def p6(rl) -> tuple[pd.DataFrame, pd.DataFrame]:
    from jevdrive import nq3_p6 as J
    t = pd.read_parquet(data_dir() / "processed" / "carla_p6" / "index.parquet", columns=["frame_name"])
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    p = J.pairs("carla_p6")
    rows, per, scored = [], [], {}
    for lv in LEVELS:
        z = _openloop("p6", lv)
        a = np.full((len(t), 20, 2), np.nan, np.float64)
        a[pos[z["frame_name"].astype(str)].to_numpy()] = z["unseen"]
        row, pc, scored[lv] = J.judge_one(lv, p, a)
        rows.append(row)
        per.append(pc)
    tab = pd.DataFrame(rows)
    for col in ("flip", "stop_sub"):
        d = [J.paired_diff(scored[lv], scored[REF], col) for lv in LEVELS]
        tab[f"{col}_delta_vs_K0"], tab[f"{col}_delta_lo"], tab[f"{col}_delta_hi"] = map(list, zip(*d))
    rl.log.info("P6: %d bypass frames (main classes)", int(tab.n_frames.iloc[0]))
    return tab, pd.concat(per, ignore_index=True)


def main():
    import time
    from jevdrive.runlog import RunLog
    rl = RunLog("nq4", "k", "readouts")
    out = rl.dir
    t0 = time.time()
    p5c, p5f = p5(rl)
    p5c.to_csv(out / "p5_criteria.csv", index=False)
    p5f.to_csv(out / "p5_flips.csv", index=False)
    rl.log.info("P5 done (%.0f s)", time.time() - t0)
    i3s, i3f = i3(rl)
    i3s.to_csv(out / "i3_summary.csv", index=False)
    i3f.to_csv(out / "i3_flips.csv", index=False)
    rl.log.info("I3 done (%.0f s)", time.time() - t0)
    p6s, p6p = p6(rl)
    p6s.to_csv(out / "p6_summary.csv", index=False)
    p6p.to_csv(out / "p6_per_scenario.csv", index=False)
    rl.log.info("P6 done (%.0f s)", time.time() - t0)
    main_i3 = i3s[i3s.readout == "R1"].set_index("level")
    p5i = p5c.set_index("level")
    verdict = {lv: {"p5_ped_delta": [p5i.ped_delta_vs_K0[lv], p5i.ped_delta_lo[lv], p5i.ped_delta_hi[lv]],
                    "p5_verdict": p5i.ped_verdict[lv],
                    "i3_delta": [main_i3.delta_vs_K0[lv], main_i3.delta_lo[lv], main_i3.delta_hi[lv]],
                    "i3_verdict": main_i3.verdict[lv]} for lv in LEVELS if lv != REF}
    both = all(v["p5_verdict"] == "不动" and v["i3_verdict"] == "不动" for v in verdict.values())
    verdict["capability_half"] = "不动 (every level, both exams)" if both else "动了 (at least one level / exam)"
    verdict["ds_half"] = "not judged here: needs the closed-loop unseen DS of K0-K3 (3 seeds, 220 routes)"
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1, ensure_ascii=False, default=float))
    md = ["## P5 v1 BA (cross-fitted unseen)", p5c.to_markdown(index=False, floatfmt=".4f"), "",
          "## I3 (R1 main, R2 descriptive)", i3s.to_markdown(index=False, floatfmt=".4f"), "",
          "## P6 v0 bypass (rule 7; descriptive for K)", p6s.drop(columns=["readout"], errors="ignore").to_markdown(index=False, floatfmt=".4f"), "",
          "## Criterion 1, capability half", "```", json.dumps(verdict, indent=1, ensure_ascii=False, default=float), "```"]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    rl.log.info("\n%s", "\n".join(md))
    rl.log.info("done (%.0f s) -> %s", time.time() - t0, out)
    rl.close()


if __name__ == "__main__":
    main()
