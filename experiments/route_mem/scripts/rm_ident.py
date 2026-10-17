#!/usr/bin/env python3
"""Flags-off identity of the anchored layout (results/anchor_check.md; lib/rmem.py "Anchored layout", rm_build.py anchor, rm_train --rm-anchor):
with the new flag off, the old layouts' features, the served memory and training are bit for bit what the code before the change gives.

  feats  (box, AlpaSim venv)  [JEV_REPO=<checkout>] rm_ident.py feats --out F.npz [--logs 24] [--rollouts 24] [--sets rsg_held v2_held]
         with the code of JEV_REPO (default: this checkout), from the same inputs: the projected and ordered rows of a sample of navtrain
         logs (rm_build.nav_one: features, ok, fut_d, jdist, turn40, trav) and of on1_L18RB0 rollouts (rm_build.pool_one: features,
         route0_d, anchors, route labels), and the served Tracker() / Tracker(order=True) at every decision of the dumped read messages
         (rm_align extract --dump; pose_of = the decision anchors, as the gate served = offline). Run once per checkout.
  cmp    (any)  rm_ident.py cmp --a OLD.npz --b NEW.npz --out NAME --res DIR      every array byte-identical (dtype, shape, bytes)
         -> DIR/NAME.json; also the share of the sampled rows equal to the stored training files after the float16 cast (context only)
  bitcmp (op-train)  rm_ident.py bitcmp --a TAG --b TAG --out NAME --res DIR [--hash H]   two rm_train runs: every logged scalar (throughput
         aside), every weight, dev ADE and on_hash_60 equal (and = H) -> DIR/NAME.json; exit 2 otherwise (nx.py bitcmp's rule)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

REPO = Path(os.environ.get("JEV_REPO") or Path(__file__).resolve().parents[3])      # the checkout whose rmem / rm_build are tested
sys.path[:0] = [str(p) for p in (REPO, REPO / "lib", REPO / "experiments/route_mem/lib", REPO / "experiments/route_mem/scripts",
                                 REPO / "experiments/anchor_offset/scripts", REPO / "experiments/alpasim/scripts", REPO / "experiments/alpasim/lib",
                                 REPO / "experiments/eyes_vs_head/scripts")]


def cmd_feats(a):
    import rmem as RM
    import rm_build as RB
    import ev as EV
    from jevdrive.common import data_dir
    D = data_dir()
    out = dict(repo=np.array(str(REPO)))
    data = "navtrain_full.s0of12"                                      # NAVSIM rows: every n-th log of the shard (largest first, as nav builds)
    tab = np.load(D / "runs/op_parity/cache" / data / "tab.npz")
    by = {}
    for r, (t, lg) in enumerate(zip(tab["names"].tolist(), tab["log"].tolist())):
        by.setdefault(lg, []).append((r, t))
    logs = sorted(by, key=lambda x: (-len(by[x]), x))
    logs = logs[::max(1, len(logs) // a.logs)][:a.logs]
    rows, P, O = [], [], []
    for lg in logs:
        p_ = RB.nav_one(("trainval", lg, by[lg], set()))
        o_ = RB.nav_one(("trainval", lg, by[lg], set(), True))
        rows.append(p_[0])
        P.append(p_[1:])
        O.append(o_[1:])
    out["nav_rows"] = np.concatenate(rows)
    for i, k in enumerate(("f", "k", "ok", "nmsg", "fut_d", "jdist", "turn40")):
        out[f"nav_p_{k}"] = np.concatenate([x[i] for x in P])
    for i, k in enumerate(("f", "ok", "trav")):
        out[f"nav_o_{k}"] = np.concatenate([x[i] for x in O])
    m = np.load(D / "runs/onpol_nuplan/rows/L18RB0/meta.npz", allow_pickle=True)   # pool rows: the first rollouts of on1_L18RB0
    byp = {}
    for i, (rn, sc, k) in enumerate(zip(m["run"].tolist(), m["scene"].tolist(), m["k"].tolist())):
        byp.setdefault((rn, sc), []).append((i, int(k)))
    PI, PF, PD, PA, PL, PO = [], [], [], [], [], []
    for key in sorted(byp)[:a.rollouts]:
        ks = [k for _, k in byp[key]]
        v, vo = RB.pool_one((*key, ks, EV.CFG)), RB.pool_one((*key, ks, EV.CFG, True))
        for i, k in byp[key]:
            PI.append(i)
            PF.append(v[k][0]), PD.append(v[k][1]), PA.append(v[k][2]), PL.append(v[k][3]), PO.append(vo[k])
    out |= dict(pool_rows=np.array(PI), pool_p_f=np.stack(PF), pool_p_r0=np.array(PD), pool_p_anchor=np.stack(PA), pool_p_label=np.stack(PL),
                pool_o_f=np.stack(PO))
    for s in a.sets:                                                   # the served Tracker on the dumped read messages
        Mg = pickle.load(open(D / "runs/route_mem/anchor" / f"{s}.msg.pkl", "rb"))
        TP, TO, TD = [], [], []
        for rid in sorted(Mg):
            g = Mg[rid]
            ts = [int(t) for t in g["t0"]]
            pose = dict(zip(ts, (np.asarray(x, float) for x in g["A"])))
            for order, acc in ((False, TP), (True, TO), (None, TD)):
                tr = RM.Tracker() if order is None else RM.Tracker(order=order)
                for t, w in zip(ts, g["W"]):
                    tr.add(t, w)
                    acc.append(tr.features(t, pose.get, pose[t]))
        out |= {f"trk_{s}_p": np.stack(TP), f"trk_{s}_o": np.stack(TO), f"trk_{s}_default": np.stack(TD)}
    sc = [json.loads(Path(f).read_text()).get("head", {}) for f in glob.glob(str(D / "runs/op_parity/runs/*/route_mem.json"))]
    out |= dict(sidecars=np.array(len(sc)), sidecars_order=np.array(sum(bool(h.get("order")) for h in sc)),
                sidecars_anchor=np.array(sum(bool(h.get("anchor")) for h in sc)))
    np.savez(a.out, **out)
    print(json.dumps({k: list(v.shape) for k, v in out.items()}))


def cmd_cmp(a):
    from jevdrive.common import data_dir
    A, B = np.load(a.a), np.load(a.b)
    keys = sorted((set(A.files) | set(B.files)) - {"repo"})
    res = dict(a=str(A["repo"]), b=str(B["repo"]), arrays={})
    for k in keys:
        x, y = (A[k] if k in A.files else None), (B[k] if k in B.files else None)
        same = x is not None and y is not None and x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        res["arrays"][k] = dict(shape=list(x.shape) if x is not None else None, identical=bool(same))
    res["identical"] = all(v["identical"] for k, v in res["arrays"].items() if not k.startswith("sidecars"))
    D = data_dir() / "runs/route_mem/feat"                             # context: the sampled rows against the stored training files (f16)
    z, zo = np.load(D / "navtrain_full.s0of12.npz"), np.load(D / "navtrain_full.s0of12.o.npz")
    r = B["nav_rows"]
    ok = B["nav_p_ok"]
    res["stored_nav_p_equal"] = float((B["nav_p_f"][ok].astype(np.float16) == z["f"][r][ok]).all(1).mean())
    res["stored_nav_o_equal"] = float((B["nav_o_f"][ok].astype(np.float16) == zo["f"][r][ok]).all(1).mean())
    zp, zpo = np.load(D / "on1_L18RB0.npz"), np.load(D / "on1_L18RB0.o.npz")
    res["stored_pool_p_equal"] = float((B["pool_p_f"].astype(np.float16) == zp["f"][B["pool_rows"]]).all(1).mean())
    res["stored_pool_o_equal"] = float((B["pool_o_f"].astype(np.float16) == zpo["f"][B["pool_rows"]]).all(1).mean())
    res["sidecars"] = dict(n=int(B["sidecars"]), order=int(B["sidecars_order"]), anchor=int(B["sidecars_anchor"]))
    write(a, res)
    raise SystemExit(0 if res["identical"] else 2)


def run_of(tag):
    import torch
    from jevdrive.common import data_dir
    d = sorted(glob.glob(str(data_dir() / f"runs/route_mem/train-{tag}/*/")))[-1]
    sc = {}
    for line in open(Path(d) / "events.jsonl"):
        e = json.loads(line)
        if e["kind"] == "scalar":
            sc[(e["tag"], e["step"])] = e["value"]
    ck = torch.load(data_dir() / f"runs/op_parity/runs/{tag}/ckpt-final.pt", map_location="cpu", weights_only=False)
    return sc, json.loads((Path(d) / "DONE").read_text()), ck["model"]


def cmd_bitcmp(a):
    (sa, da, ma), (sb, db, mb) = run_of(a.a), run_of(a.b)
    keys = [k for k in sorted(set(sa) & set(sb), key=str) if not k[0].startswith("throughput/")]
    same = all(set(ma[g] or {}) == set(mb[g] or {}) for g in ("net", "parity"))
    dw = max(float((ma[g][k].float() - mb[g][k].float()).abs().max()) for g in ("net", "parity") for k in ma[g]) if same else float("inf")
    r = dict(a=a.a, b=a.b, note=a.note, scalars=len(keys), scalar_keys_equal={k[0] for k in sa} == {k[0] for k in sb},
             scalar_max_abs_diff=max(abs(sa[k] - sb[k]) for k in keys), weight_max_abs_diff=dw, weight_keys_equal=same,
             dev_ade=[da.get("dev_ade"), db.get("dev_ade")], on_hash_60=[da.get("on_hash_60"), db.get("on_hash_60")], want_hash=a.hash or None)
    r["identical"] = bool(r["scalar_max_abs_diff"] == 0 and dw == 0 and same and r["scalar_keys_equal"] and r["dev_ade"][0] == r["dev_ade"][1]
                          and r["on_hash_60"][0] == r["on_hash_60"][1] and (not a.hash or r["on_hash_60"][0] == a.hash))
    write(a, r)
    raise SystemExit(0 if r["identical"] else 2)


def write(a, res):
    from jevdrive.common import data_dir
    for d in (Path(a.res), data_dir() / "runs/route_mem/anchor"):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{a.out}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("feats")
    f.add_argument("--out", required=True), f.add_argument("--logs", type=int, default=24), f.add_argument("--rollouts", type=int, default=24)
    f.add_argument("--sets", nargs="*", default=["rsg_held", "v2_held"])
    c = sp.add_parser("cmp")
    c.add_argument("--a", required=True), c.add_argument("--b", required=True), c.add_argument("--out", required=True), c.add_argument("--res", required=True)
    b = sp.add_parser("bitcmp")
    b.add_argument("--a", required=True), b.add_argument("--b", required=True), b.add_argument("--out", required=True), b.add_argument("--res", required=True)
    b.add_argument("--hash", default=""), b.add_argument("--note", default="")
    a = p.parse_args()
    dict(feats=cmd_feats, cmp=cmd_cmp, bitcmp=cmd_bitcmp)[a.cmd](a)
