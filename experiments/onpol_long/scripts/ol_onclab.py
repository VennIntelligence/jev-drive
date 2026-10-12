#!/usr/bin/env python3
"""ONPOL-LONG Amendment 16: label study of point-based oncoming hinge variants on the zeros_mech-rule region (no training). Box, AlpaSim's venv, CPU.

  ol_onclab.py --onc DIR/onc.npz --rows L18YS --events CSV --held <held run dir> --out F.json
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R / "experiments/onpol_long/scripts"), str(R / "experiments/body1/lib"), str(R / "experiments/zeros_mech/scripts"), str(R / "experiments/anchor_offset/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CEN = 1.461
VAR = {"a rear axle": ("pt", 0.0), "b box centre": ("pt", CEN), "c footprint, 0.5 m depth": ("box", -0.5), "d footprint, 1.0 m depth": ("box", -1.0)}


def dense9(P9):
    """(n, 9, 3) poses at 0 .. 4 s -> (n, 1, 41, 3) linear at 0.1 s."""
    t = np.arange(41) * 0.1
    T = np.arange(9) * 0.5
    out = np.stack([np.stack([np.interp(t, T, P9[i, :, c]) for c in range(3)], -1) for i in range(len(P9))])
    return out[:, None].astype(np.float32)


def active(d, sdf):
    """d (n, 1, 41, 3) rear-axle poses, sdf (n, 128, 96) (+ outside the oncoming region) -> {variant: (n,) bool}."""
    import sweep as SW
    m, ins = SW.corner_margins(d, sdf)
    mb = np.where(ins.all(-1), m.min(-1), np.inf)[:, 0].min(1)
    out = {}
    for k, (kind, v) in VAR.items():
        if kind == "pt":
            x, y = d[..., 0] + v * np.cos(d[..., 2]), d[..., 1] + v * np.sin(d[..., 2])
            s, ins_ = SW.sdf_at(sdf, x, y)
            out[k] = (np.where(ins_, s, np.inf)[:, 0].min(1) < 0)
        else:
            out[k] = mb < v
    return out


def main(a):
    import sweep as SW
    sdf = np.load(a.onc)["sdf"].astype(np.float32)
    M = np.load(DATA / "runs/onpol_nuplan/rows" / a.rows / "meta.npz")
    n = len(sdf)
    served = active(SW.dense(M["poses"][:, None].astype(np.float32), np.zeros((n, 1, 2), np.float32)), sdf)
    lf = M["logfut"]
    ok = np.isfinite(lf).all((1, 2))
    human = active(dense9(np.nan_to_num(lf)), sdf)
    dpsi = np.degrees(np.abs((lf[:, -1, 2] - lf[:, 0, 2] + np.pi) % (2 * np.pi) - np.pi))
    st, tu = dpsi < 15, dpsi >= 45
    res = dict(rows=n, human_rows=int(ok.sum()), table={k: dict(served=float(served[k].mean()), human=float(human[k][ok].mean()),
                                                                 served_straight=float(served[k][st].mean()), served_turn=float(served[k][tu].mean()),
                                                                 human_straight=float(human[k][ok & st].mean()), human_turn=float(human[k][ok & tu].mean()))
                                                       for k in VAR})
    usable = [k for k, v in res["table"].items() if v["served"] <= 0.10 and v["human"] <= 0.03]
    res["usable"] = usable
    # d268's across-traffic early-inside events (YS-L18-s1), served plan 2 decisions before the event, region computed now on the held scene
    import ol_zmonc as ZO
    import zm_read as Z
    from ao_e1 import drive_records, loader
    ev = [r for r in csv.DictReader(open(a.events)) if r["cls"] == "early-inside" and np.sign(float(r["dpsi_log"])) * float(r["s_onc"]) > 0]
    recs = drive_records(a.held)
    cfg = str(Path(a.held) / "generated-user-config-0.yaml")
    E = []
    for r in ev:
        k = max(0, int(float(r["k_ev"])) - 2)
        rr = {x["k"]: x for x in recs.get(r["scene"], [])}.get(k)
        if rr is None:
            continue
        ds = loader(cfg).get_data_source(r["scene"])
        tr = ds.rig.trajectory
        z = float(np.interp(rr["t0"], np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, 2]))
        an = np.asarray(rr["anchor"], float)
        P = Z.to_local(an, np.c_[ZO.XC.ravel(), ZO.YC.ravel()])
        m = ZO.classify(Z.Lanes(ds.map), P, z, an[2], 60.0).reshape(ZO.NH, ZO.NW) == 2
        s = ZO.sdf_of(m)[None]
        act = active(SW.dense(np.asarray(rr["poses"], np.float32)[None, None], np.zeros((1, 1, 2), np.float32)), s)
        E.append(dict(scene=r["scene"][-16:-8], type=r["type"], k=k, **{q: bool(v[0]) for q, v in act.items()}))
    res["events"] = E
    res["events_active_share"] = {q: float(np.mean([e[q] for e in E])) if E else None for q in VAR}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onc", required=True), ap.add_argument("--rows", default="L18YS"), ap.add_argument("--events", required=True)
    ap.add_argument("--held", required=True), ap.add_argument("--out", required=True)
    main(ap.parse_args())
