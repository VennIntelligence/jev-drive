"""BODY1 arm 4.3, prereg Amendment 5 item 3: select the weight of the hinge-only rows on the validation part of the train logs.

For every candidate tag: relative fall of (agent rate + boundary rate) of the own plan on navsim/body1-val-logs against the switch-off pilot
(results/loss/g3_<val name>.json of bd4_g3.py --set val), subject to dev ADE <= switch-off pilot + 0.01 m and dev drift_off <= 0.30 (the
trainers' last dev evaluation). The largest fall among the candidates that meet the constraints is taken; none -> the arm ends.
Reads no hold-log file.

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/bd4_select.py --cand 3:P2H10B-Pw3-s0:a5_val_w3 10:P2H10B-Pw10-s0:a5_val_w10 --ref P2H10-P-s0
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402

import b1 as B  # noqa: E402

OUT = B.REPO / "experiments/body1/results/loss"


def dev(tag):
    from jevdrive.common import data_dir
    d = sorted((data_dir() / "runs/op_parity" / f"train-{tag}").iterdir())[-1]
    o = {}
    for ln in open(d / "events.jsonl"):
        e = json.loads(ln)
        if e.get("kind") == "scalar" and e["tag"].startswith("dev/"):
            o[e["tag"][4:]] = e["value"]                              # the last evaluation wins
    return o


def main(a):
    from jevdrive.run import Run
    with Run("body1", "a5-select", config=vars(a)) as run:
        ref = dev(a.ref)
        rows = []
        for c in a.cand:
            w, tag, name = c.split(":")
            g = json.loads((OUT / f"g3_{name}.json").read_text())
            assert g["set"] == "val" and g["new"] == tag and a.ref in g["refs"]
            p, d = g["pooled"][a.ref], dev(tag)
            ok_ade, ok_drift = d["ade"] <= ref["ade"] + 0.01, d["drift_off"] <= 0.30
            rows.append(dict(w=float(w), tag=tag, n=g["n"], agent=p["agent"]["new"], agent_ref=p["agent"]["base"], bnd=p["bnd"]["new"], bnd_ref=p["bnd"]["base"],
                             sum=p["sum"]["new"], sum_ref=p["sum"]["base"], sum_rel_fall=p["sum"]["rel_fall"], sum_diff=p["sum"]["diff"], sum_lo=p["sum"]["lo"],
                             sum_hi=p["sum"]["hi"], agent_rel_fall=p["agent"]["rel_fall"], bnd_rel_fall=p["bnd"]["rel_fall"], dev_ade=d["ade"], dev_ade_ref=ref["ade"],
                             dev_drift_off=d["drift_off"], ok_ade=bool(ok_ade), ok_drift=bool(ok_drift), eligible=bool(ok_ade and ok_drift)))
        el = [r for r in rows if r["eligible"]]
        sel = max(el, key=lambda r: r["sum_rel_fall"]) if el else None
        res = dict(rule="Amendment 5 item 3: larger relative fall of (agent + boundary rate) on navsim/body1-val-logs; dev ADE <= ref + 0.01 m; drift_off <= 0.30",
                   ref=a.ref, candidates=rows, selected_w=sel and sel["w"], selected_tag=sel and sel["tag"])
        (OUT / "a5_selection.json").write_text(json.dumps(res, indent=1) + "\n")
        run.info(json.dumps(res, indent=1))
        run.summary.update(selected_w=res["selected_w"], selected_tag=res["selected_tag"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True, help="w:tag:g3-name")
    ap.add_argument("--ref", default="P2H10-P-s0")
    main(ap.parse_args())
