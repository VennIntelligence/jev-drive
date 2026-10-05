#!/usr/bin/env python
"""op_wide_ft open-loop 2 x 2 on the CARLA dev junctions (plans/2026-10-05-wide-ft-prereg.md, gate (c)): model x input wide FOV, paired per row.

Rows = rft_eval's carla_exit set (dev poses not stopped, every exit, clean polyline command); per row: exit correct (class of the plan path at arc
d + 12 m = the command), command-signed lateral offset at d + 10 m, desired curvature towards the commanded side (turn rows). Input FOV = which bank
the trunks come from (w58 = wide 58.7 deg, w116 = 116 deg; same poses, road identical). Paired differences: jevdrive.stats.paired, clusters =
junctions. op-train env, one GPU, OP_RFT_ROOT=$DATA_DIR/runs/op_wide_ft.

  wide_ol.py --models O wf-w58-s0 wf-w116-s0 [--out experiments/op_wide_ft/results/ol2x2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "scripts"), str(REPO / "experiments/op_adapt_r2/lib"), str(REPO / "experiments/op_route_ft/scripts")]
import rft as F  # noqa: E402
import rft_eval as E  # noqa: E402
from jevdrive import stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["O", "wf-w58-s0", "wf-w116-s0"])
    ap.add_argument("--out", default=str(REPO / "experiments/op_wide_ft/results/ol2x2"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    C, _, S = E.build_sets("w58", 600)
    s = S["carla_exit"]
    T = {"58": C.T, "116": np.load(F.rroot("bank", "w116") / "trunk.npy", mmap_mode="r")}
    v0 = C.tab["v0"][s["pose"]].astype(float)
    turn = s["cmd"] != "straight"
    sg = np.where(s["cmd"] == "left", 1.0, -1.0)
    cl = s["cluster"]
    per = {}
    for name in a.models:
        m = F.load_rmodel(name, dev)
        fk = "fp" if (m.route is not None and m.route.enc == "poly") else "fb"
        for fov, Tb in T.items():
            pl, ac = E.run_rows(m, lambda r, Tb=Tb: Tb[r], s["pose"], s[fk], C.tab["slot_valid"][s["pose"]], C.tab["tc"][s["pose"]], dev)
            c = E.cls_at_arc(pl, C.cam, C.tab["d"][s["pose"]] + 12.0)
            per[f"{name}@{fov}"] = dict(ok=(c == s["cmd"]).astype(float), lat=sg * E.lat_at_arc(pl, C.cam, C.tab["d"][s["pose"]] + 10.0),
                                        k=sg * (-ac / np.maximum(1.0, v0) ** 2))
        del m
        torch.cuda.empty_cache()
    rows = []
    for k, d in per.items():
        rows.append(dict(contrast=k, metric="exit_ok", **stats.bootstrap(d["ok"], groups=cl)))
        rows.append(dict(contrast=k, metric="exit_ok_turn", **stats.bootstrap(d["ok"][turn], groups=cl[turn])))
        rows.append(dict(contrast=k, metric="lat_turn_m", **stats.bootstrap(d["lat"][turn], groups=cl[turn])))
        rows.append(dict(contrast=k, metric="k_turn", **stats.bootstrap(d["k"][turn], groups=cl[turn])))
    pairs = [("wf-w116-s0@116", "wf-w58-s0@58"), ("wf-w116-s0@116", "wf-w116-s0@58"), ("wf-w58-s0@116", "wf-w58-s0@58"), ("O@116", "O@58"),
             ("wf-w116-s0@58", "wf-w58-s0@58")]
    res = {}
    for x, y in pairs:
        if x in per and y in per:
            for met, fld, sel in (("exit_ok", "ok", np.ones(len(cl), bool)), ("exit_ok_turn", "ok", turn), ("lat_turn_m", "lat", turn), ("k_turn", "k", turn)):
                vx, vy = per[x][fld][sel], per[y][fld][sel]
                r = stats.paired(vx, vy, groups=cl[sel])
                rows.append(dict(contrast=f"{x} - {y}", metric=met, **r))
                res[f"{x} - {y}|{met}"] = r
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.write_table(rows, out, note="CARLA dev exit rows (n %d, %d turn rows), clusters = junctions" % (len(cl), int(turn.sum())))
    g = res.get("wf-w116-s0@116 - wf-w58-s0@58|exit_ok")
    own = res.get("wf-w116-s0@116 - wf-w116-s0@58|exit_ok")
    if g and own:
        json.dump(dict(diff_vs_w58=dict(mean=g["mean"], lo=g["lo"], hi=g["hi"]), own_116_minus_58=own["mean"], own=dict(lo=own["lo"], hi=own["hi"])),
                  open(str(out) + "_gate.json", "w"), indent=1)
    print(Path(str(out) + ".md").read_text())


if __name__ == "__main__":
    main()
