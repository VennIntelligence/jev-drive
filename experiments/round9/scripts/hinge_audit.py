#!/usr/bin/env python3
"""ROUND9 HINGE-AUDIT (plans/2026-10-22-hinge-audit.md): are the S consequence hinges satisfied by the human's own log future and by the
teacher labels; root causes of every human violation; expert-calibrated hinge candidates. CPU only, stored files + a CPU forward of
R6DAG-L18-s1. Chain: hinge_audit.sh. Box outputs: $DATA_DIR/runs/round9/hinge_audit/ -> experiments/round9/results/hinge_audit/.

  fwd    (op-train)  --parts S|ph|nav --shard i/n: the trainer's build with R6DAG's argv (rtc.py), eval-mode plans of R6DAG-L18-s1 on a
                     seeded --frac of the chosen pool rows (S = nuPlan S rows rb0 / rtb1 / y / ys, ph = PAI human rows; nav = navtrain
                     imitation rows) + the trainer's own terms on them (cross-check) -> $O/fwd_<part>_<shard>.npz
  steps  (op-train)  per pool (S | PAID | PAIH) and path (log / fut / plan): per-step raw clearances of every term (agent corridor / side
                     distance, road corner SDF (+ drivable SDF on nuPlan), onc / soft region SDF, side distance) plus the root-cause
                     counterfactuals of the log path -> $O/steps_<pool>_<path>.npz
  report (op-train)  every table of the result page from the steps files -> $O/audit.json, results dir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round9/scripts"), str(REPO / "experiments/defect_fix/lib"),
                str(REPO / "experiments/body1/lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/hinge_audit"
CR = DATA / "runs/op_parity/cache"
S_PARTS = ("rb0", "rtb1", "y", "ys")
CKPT = "R6DAG-L18-s1"


# ---------------------------------------------------------------- fwd (op-train, CPU)
def cmd_fwd(a):
    import torch
    import rtc
    from jevdrive.common import n_cpus
    torch.set_num_threads(a.threads or n_cpus())
    rt = REPO / "experiments/route_mem/scripts/rm_train.py"
    src = rt.read_text()
    tail = "dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)"
    assert src.count(tail) == 1
    g = {"__name__": "__main__", "__file__": str(rt)}
    sys.argv = [str(rt), "rely", *rtc.R6DAG_ARGV.format(tag=CKPT, data=DATA).split()]
    exec(compile(src.replace(tail, "_HA_ARGS = a"), str(rt), "exec"), g)
    ra = g["_HA_ARGS"]
    ra.rm_from = CKPT
    dev = torch.device("cpu")
    ns, *_ = g["build"](ra, dev)
    LS = ns.losses(ns.mode)
    import pp_train as T
    S, model = ns.S, ns.model.eval()
    names = S.tab["names"].astype(str)
    part = np.array([x.split(":")[1] if x.startswith("on:") else "" for x in names])
    i, n = (int(x) for x in a.shard.split("/"))
    rng = np.random.default_rng(a.seed)
    if a.parts == "nav":
        hz = LS.hinge
        cand = np.flatnonzero((np.arange(len(names)) < ns.on0) & S.has_fut.cpu().numpy() & hz.ok.cpu().numpy())
    else:
        want = S_PARTS if a.parts == "S" else ("ph",)
        cand = np.flatnonzero(np.isin(part, want) & (np.arange(len(names)) >= ns.on0))
    sel = np.sort(rng.choice(cand, int(round(a.frac * len(cand))), replace=False))
    sel = sel[i::n]
    print(f"{a.parts} shard {a.shard}: {len(sel)} of {len(cand)} rows", flush=True)
    pi_ = torch.as_tensor(S.pi, device=dev)
    P, TR = [], {k: [] for k in ("agent", "road", "lat_onc", "lat_soft", "lat_side", "nav", "nav_fut")}
    t0 = time.time()
    for j in range(0, len(sel), a.bs):
        r = torch.as_tensor(sel[j:j + a.bs], device=dev)
        with torch.no_grad():
            o = model(S.front[r.cpu().numpy()], S.ego[r], S.tc[r], None, None, nv=None if ns.nv is None else ns.nv[r]).float()
            x, y, psi = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[r], LS.W)
            if a.parts == "nav":
                TR["nav"].append(torch.relu(LS.hinge.margin - LS.hinge.margins(x, y, psi, r)).mean(1).numpy())
                F = S.fut[r]
                TR["nav_fut"].append(torch.relu(LS.hinge.margin - LS.hinge.margins(F[..., 0], F[..., 1], F[..., 2], r)).mean(1).numpy())
            else:
                TR["agent"].append((LS.agent2.per_step(x, y, psi, r).mean(1) * LS.on_aok[r]).numpy())
                if LS.road is not None:
                    TR["road"].append((torch.relu(LS.road.margin - LS.road.margins(x, y, psi, r)).mean(1) * LS.road.ok[r]).numpy())
                lt = LS.lon.loss(torch.stack([x, y, psi], -1), r - LS.on0, LS.pack)
                for k in ("lat_onc", "lat_soft", "lat_side"):
                    if k in lt:
                        TR[k].append(lt[k].numpy())
        P.append(torch.stack([x, y, psi], -1).numpy())
        if j == 0:
            print(f"first batch {time.time() - t0:.1f} s", flush=True)
    O.mkdir(parents=True, exist_ok=True)
    extra = {}
    if a.parts == "nav":                                                   # the navtrain human label on EVERY labelled row (no forward needed)
        allc = cand
        hv = []
        with torch.no_grad():
            for j in range(0, len(allc), 4096):
                r = torch.as_tensor(allc[j:j + 4096])
                F = S.fut[r]
                m = LS.hinge.margins(F[..., 0], F[..., 1], F[..., 2], r)
                hv.append(torch.stack([torch.relu(LS.hinge.margin - m).mean(1), m.amin(1)], 1).numpy())
        hv = np.concatenate(hv)
        extra = dict(all_rows=allc, all_fut_hinge=hv[:, 0], all_fut_minsdf=hv[:, 1], speed_all=S.tab["speed"][allc] if "speed" in S.tab else np.zeros(0),
                     fut_all=S.fut[torch.as_tensor(allc)].numpy(), margin=LS.hinge.margin)
    np.savez(O / f"fwd_{a.parts}_{i}.npz", rows=sel, names=names[sel], poses=np.concatenate(P), on0=ns.on0,
             **{f"tr_{k}": np.concatenate(v) for k, v in TR.items() if v}, **extra)
    print(json.dumps(dict(parts=a.parts, shard=a.shard, rows=len(sel), sec=round(time.time() - t0, 1))), flush=True)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fwd")
    f.add_argument("--parts", choices=["S", "ph", "nav"], required=True)
    f.add_argument("--shard", default="0/1")
    f.add_argument("--frac", type=float, default=0.3)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--bs", type=int, default=256)
    f.add_argument("--threads", type=int, default=0)
    s = sp.add_parser("steps")
    s.add_argument("--pool", choices=["S", "PAID", "PAIH"], required=True)
    s.add_argument("--threads", type=int, default=0)
    s.add_argument("--limit", type=int, default=0)
    sp.add_parser("report")
    a = p.parse_args()
    dict(fwd=cmd_fwd, steps=globals().get("cmd_steps"), report=globals().get("cmd_report"))[a.cmd](a)


if __name__ == "__main__":
    main()
