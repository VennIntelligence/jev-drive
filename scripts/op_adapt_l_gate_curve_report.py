"""Registered line verdicts and paired curve saturation; no new model selection.

Writes small CSV/JSON results to the run directory, for download to research/results/op-adapt-L/gate-curve.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from jevdrive import op_adapt_l as L  # noqa: E402
from op_adapt_l_gate_curve import ROOT, PREP, grant_check  # noqa: E402
from op_adapt_l_gate_curve_readout import Bootstrap, indicators, old_eval  # noqa: E402
from op_adapt_l_gate_curve_prep import dump  # noqa: E402
from op_adapt_l_train import Log  # noqa: E402

GAIN_LINES={"start":.05,"stop":.15,"turn_onset":.06}


def safety(df,seed):
    d=df[df.seed.astype(str)==str(seed)]
    def metric(s,k,scale=1.):
        row=d[(d["slice"]==s)&(d.metric==k)&(d.xscale==scale)]
        assert len(row)==1, f"Missing or duplicate {seed}/{s}/{k}/{scale}"
        return row.iloc[0]
    return {
        "stay_false_start":float(metric("stay","false_start").hi)<=.02,
        "control_false_stop":float(metric("control","false_stop").hi)<=.02,
        "control_false_turn":float(metric("control","false_turn").hi)<=.02,
        "straight_false_turn":float(metric("straight_int","false_turn").hi)<=.02,
        "drift_median":float(metric("other","drift_median").adapt)<=.10,
        "drift_p95":float(metric("other","drift_p95").adapt)<=.50,
        "slow":float(metric("other","slow").delta)<=.02,
        "fast":float(metric("other","fast").delta)<=.02,
        "rfs_raw_no_harm":float(metric("all","rfs").lo)>=-.10,
        "rfs_cal_no_harm":float(metric("all","rfs",1.06).lo)>=-.10,
        "rfs_raw_nonnegative":float(metric("all","rfs").delta)>=0,
        "rfs_cal_nonnegative":float(metric("all","rfs",1.06).delta)>=0,
    }


def main():
    out=ROOT/"report";out.mkdir(parents=True,exist_ok=True)
    if (out/"DONE").exists() or (out/"ERROR").exists():raise SystemExit("Existing report sentinel")
    log=Log(out)
    assert log.tb is not None
    try:
        log.event("start", resources=grant_check(gpu=False))
        with tqdm(total=1, desc="Waiting for complete curve") as bar:
            while not (ROOT / "chain/DONE").exists():
                if (ROOT / "chain/ERROR").exists():
                    raise RuntimeError("Curve chain failed; report is blocked")
                time.sleep(5)
            bar.update(1)
        baseline_config = json.loads((L.lroot("runs", "main-s0") / "config.json").read_text())
        original_cfg = {k: v for k, v in baseline_config["cfg"].items() if k not in ("name", "seed")}
        counts_table = pd.read_csv(PREP / "counts.csv").astype({"size": str})
        timings = []
        for size in ("25", "110", "300", "all"):
            for seed in (0, 1):
                directory = ROOT / f"curve-{size}-s{seed}"
                assert (directory / "DONE").exists() and (ROOT / f"eval-{size}-s{seed}/DONE").exists()
                config = json.loads((directory / "config.json").read_text())
                actual_cfg = {k: v for k, v in config["cfg"].items() if k not in ("name", "seed")}
                assert actual_cfg == original_cfg, f"Configuration differs from main: {size}/{seed}"
                assert config["cfg"]["seed"] == seed
                for key in ("wod:stay", "wod:control", "wod:straight_int", "wod:other", "nus"):
                    assert config["pools"][key] == baseline_config["pools"][key], f"Contrast pool changed: {key}"
                for sl in L.SLICE3:
                    row = counts_table[(counts_table["slice"] == sl) & (counts_table["size"] == size)].iloc[0]
                    assert config["pools"][f"wod:{sl}"] == int(row.frames)
                dev = json.loads((directory / "dev.json").read_text())
                assert dev["nonfinite"] == 0 and dev["steps"] == 4000
                timings.append({"size": size, "seed": seed, "steps": dev["steps"], "train_seconds_including_dev": dev["train_s"],
                                "seq_per_s_including_dev": dev["seq_per_s"], "data_wait_fraction": dev["data_wait_frac"],
                                "peak_reserved_gib": dev["peak_reserved_gb"], "nonfinite": dev["nonfinite"]})
        pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
        dump(out / "run_audit.json", {"runs": 8, "configuration_matches_main": True,
             "contrast_and_other_pools_match_main": True, "imitation_pools_match_manifest": True,
             "all_steps_4000": True, "nonfinite_losses": 0})
        events = [json.loads(line) for line in (ROOT / "curve-unit/events.jsonl").read_text().splitlines()]
        loss_bins = [e["value"] for e in events if e.get("tag") == "loss/imit"]
        assert len(loss_bins) == 16 and min(loss_bins) >= 0
        # Exact 20% = 160 steps. Nonnegative loss bounds the two unknown ten-step fragments.
        first160_lower = 50 * sum(loss_bins[:3]) / 160
        last160_upper = 50 * sum(loss_bins[-4:]) / 160
        assert last160_upper < first160_lower, "Exact 20% unit-loss decrease is not certified by the stored bins"
        dump(out / "unit_trend_bounds.json", {"first160_mean_lower_bound": first160_lower,
             "last160_mean_upper_bound": last160_upper, "registered_exact_20_percent_certified": True,
             "basis": "nonnegative imitation loss; stored disjoint 50-step means"})
        gate=pd.read_csv(ROOT/"gate-val/metrics.csv")
        auc=pd.read_csv(ROOT/"gate-val/auc.csv")
        front=json.loads((ROOT/"frontier/frontier.json").read_text())
        decisions=[]
        g0=bool(auc.loc[auc.population=="all_stationary","G0_pass"].iloc[0])
        for seed in (0,1,2):
            r=safety(gate,seed)
            g=gate[(gate.seed.astype(str)==str(seed))&(gate["slice"]=="start")&(gate.metric=="cap_start")].iloc[0]
            lines={"G0":g0,"G1":bool(g.delta>=.07 and g.lo>.05),"G2":r["stay_false_start"],
                   "G3":all(r[k] for k in ("control_false_stop","control_false_turn","straight_false_turn")),
                   "G4":all(r[k] for k in ("drift_median","drift_p95","slow","fast")),
                   "G5":r["rfs_raw_no_harm"] and r["rfs_cal_no_harm"],"G6":bool(front["G6_pass"])}
            for line,passed in lines.items(): decisions.append({"model":"gate","seed":seed,"line":line,"passed":bool(passed)})
        curve=[]
        for size in ("25","110","300","all"):
            d=pd.read_csv(ROOT/f"curve-read-{size}/metrics.csv")
            curve.append(d)
            for seed in (0,1):
                r=safety(d,seed)
                for s in L.SLICE3:
                    g=d[(d.seed.astype(str)==str(seed))&(d["slice"]==s)&(d.metric==f"cap_{s}")].iloc[0]
                    decisions.append({"model":f"curve-{size}","seed":seed,"line":f"C1-{s}",
                                      "passed":bool(g.delta>=GAIN_LINES[s] and g.lo>0)})
                lines={"C2":all(r[k] for k in ("stay_false_start","control_false_stop","control_false_turn","straight_false_turn")),
                       "C3":all(r[k] for k in ("drift_median","drift_p95","slow","fast")),
                       "C4":r["rfs_raw_no_harm"] and r["rfs_cal_no_harm"]}
                for line,passed in lines.items():decisions.append({"model":f"curve-{size}","seed":seed,"line":line,"passed":bool(passed)})
        metrics=pd.concat([gate,*curve],ignore_index=True)
        metrics.to_csv(out/"metrics.csv",index=False)
        verdicts=pd.DataFrame(decisions);verdicts.to_csv(out/"lines_by_seed.csv",index=False)
        grouped=verdicts.groupby(["model","line"]).passed.all().reset_index()
        grouped.to_csv(out/"lines_all_seeds.csv",index=False)
        # Paired saturation compares the two-seed mean indicators, using the same whole-segment bootstrap.
        o=old_eval("O","wodval");rows=o["rows"]
        tab=dict(np.load(L.lroot("prep")/"wodval.npz",allow_pickle=True))
        reference=indicators(tab,rows,o["plan"])
        per={}
        for size in tqdm(("25","110","300","all"),desc="Curve paired indicators",mininterval=5):
            per[size]=[indicators(tab,rows,np.load(ROOT/f"eval-{size}-s{seed}/wodval.npz")["plan"]) for seed in (0,1)]
        ratios=[];interpretations=[]
        counts=pd.read_csv(PREP/"counts.csv").astype({"size":str})
        for s in L.SLICE3:
            mask=tab[f"s_{s}"][rows];key=f"cap_{s}"
            boot=Bootstrap(tab["seq"][rows[mask]])
            full=np.mean([m[key][mask] for m in per["all"]],axis=0)
            baseline=reference[key][mask]
            full_gain=float((full-baseline).mean())
            for size in ("25","110","300","all"):
                small=np.mean([m[key][mask] for m in per[size]],axis=0)
                gain=float((small-baseline).mean())
                comparison=boot.paired(small,full)
                ratio=gain/full_gain if full_gain>0 else None
                seed_ratio=[]
                for seed in (0,1):
                    denominator=float((per["all"][seed][key][mask].astype(float)-baseline).mean())
                    numerator=float((per[size][seed][key][mask].astype(float)-baseline).mean())
                    seed_ratio.append(numerator/denominator if denominator>0 else None)
                rs=counts[(counts["slice"]==s)&(counts["size"]==size)].iloc[0]
                passed=bool(verdicts[(verdicts.model==f"curve-{size}")&(verdicts.line==f"C1-{s}")].passed.all())
                ratios.append({"slice":s,"size":size,"segments":int(rs.segments),"events":int(rs.events),
                               "gain":gain,"full_gain":full_gain,"fraction_of_full_gain":ratio,
                               "seed0_fraction":seed_ratio[0],"seed1_fraction":seed_ratio[1],
                               "delta_vs_full":comparison["delta"],"lo_vs_full":comparison["lo"],"hi_vs_full":comparison["hi"],
                               "behavior_learned":passed,"saturated":bool(ratio is not None and ratio>=.9 and comparison["lo"]>=-.02)})
            r=next(q for q in ratios if q["slice"]==s and q["size"]=="110")
            read=("training_value_at_100_segment_scale" if r["fraction_of_full_gain"] is not None and
                  r["fraction_of_full_gain"]>=.6 and r["behavior_learned"] else
                  "primarily_adjudication_or_exam_at_this_budget" if r["fraction_of_full_gain"] is not None and
                  r["fraction_of_full_gain"]<=.3 else "partial_or_unresolved")
            sat=next((q["size"] for q in ratios if q["slice"]==s and q["saturated"]),None)
            interpretations.append({"slice":s,"at_110_segments":read,"saturation_size":sat})
        pd.DataFrame(ratios).to_csv(out/"curve_ratios.csv",index=False)
        dump(out/"interpretations.json",{"curve":interpretations,
              "gate_all_lines_pass":bool(grouped[grouped.model=="gate"].passed.all()),
              "bootstrap_replicates":2000,"bootstrap_seed":0,"training_seed_uncertainty_in_CI":False})
        log.event("end",status="complete");(out/"DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    except BaseException as e:
        (out/"ERROR").write_text(traceback.format_exc());log.event("error",error=str(e));raise
    finally:log.tb.close()


if __name__=="__main__":main()
