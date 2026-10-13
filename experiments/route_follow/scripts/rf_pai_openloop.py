#!/usr/bin/env python3
"""Turning school selection ladder step 3 (prereg experiments/route_follow/plans/2026-10-13-turning-school-prereg.md, Amendment 2): an OFFLINE
PAI turn read. Checkpoints are run open loop on log-following PAI rows (the human's states, frozen-encoder tokens, the driver's route memory)
around strict route turns and scored against the human. Box, envs/op-train, one GPU through the pool (CPU works, slowly).

Rows: a log-pass cache in the onp_cache layout (tok.npy, tab.npz: tix, ego_in, cam_t, lht, imit, log_fut, k, t0_us, usable, ...) and its
rf_pait.py `route` output (per scene: f (route memory), anchor, poly, turns, topo), the scenes of --scenes minus --exclude (default: the exam's
hard_exam list, never used for selection). Turns, windows and the distance to the entry d (route stations) are rf_pait pack's.
Per turn and checkpoint (sign s = the turn's direction, left +):
  speed_d      plan speed (chord arc of the 0.5 / 1.0 s poses over 1 s) at the row nearest d = 40, 20, 0 m before the entry (within 4 m),
               against the human's (the same on `imit`, the human-timed demo)
  lat2         rows with d in [-10, 20] m and |needed| >= 0.5 m: s x plan y at 2 s divided by s x human y at 2 s (needed); median per turn
  side4        rows with d in [-10, 20] m and |human y at 4 s| >= 1 m: plan y at 4 s on the turn's side (share)
Each turn's row of the read says whether its scene is in a training pool part (--train-scenes: the pool's scenes), so held-out and in-sample
turns are reported apart. -> <out>/<name>.json (per tag: per turn, medians; held-out / in-sample) and <out>/<name>.md.

  rf_pai_openloop.py --cache <log-pass cache> --route <rf_pait route dir> --scenes <list> --train-scenes <list> \
      --tags RSG-L18-s0 RSGPAIT-L18-s0 RPAIT-L18-s0 RPAITC-L18-s0 --out experiments/route_follow/results/pai_openloop --name p1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R / "experiments/route_follow/scripts"), str(R / "experiments/route_mem/lib"), str(R / "experiments/op_parity/scripts"),
                str(R / "experiments/onpol_pai/lib"), str(R / "lib"), str(R / "experiments/body1/lib"), str(R)]
import rf_pait as RP  # noqa: E402

DS = (40, 20, 0)


def rows_of(a):
    """-> per-row arrays of the selected turn rows: cache index, scene, turn id (global), d, sign, topo, route features."""
    T = np.load(Path(a.cache) / "tab.npz")
    scenes = T["scenes"][T["sid"]]
    excl = set(RP.scenes_of(a.exclude)) if a.exclude else set()
    want = [s for s in RP.scenes_of(a.scenes) if s not in excl]
    out = dict(ci=[], sc=[], tid=[], d=[], sgn=[], topo=[], f=[])
    turns = []
    for sc in want:
        f = Path(a.route) / f"{sc}.npz"
        if sc not in set(T["scenes"].tolist()) or not f.exists():
            continue
        Z = np.load(f)
        P = Z["poly"][:, :2]
        idx = np.flatnonzero(scenes == sc)
        idx = idx[np.argsort(T["k"][idx])]
        pos = {int(k): j for j, k in enumerate(Z["k"])}
        j = np.array([pos[int(k)] for k in T["k"][idx]])
        arc = RP.arc_on(P, Z["anchor"][j, :2])
        good = T["usable"][idx] & T["imit_ok"][idx].all(1) & T["log_fut_ok"][idx].all(1)
        for (s_in, s_out, deg), cl in zip(Z["turns"], Z["topo"]):
            if not (arc >= s_in).any() or arc[0] > s_out:
                continue
            d = s_in - arc
            w = good & (d <= 45) & (arc <= s_out)
            if not w.any():
                continue
            tid = len(turns)
            turns.append(dict(scene=sc, deg=float(deg), topo=str(cl), train=sc in a._train))
            n = int(w.sum())
            out["ci"].append(idx[w]), out["sc"].append(np.full(n, sc)), out["tid"].append(np.full(n, tid)), out["d"].append(d[w])
            out["sgn"].append(np.full(n, np.sign(deg))), out["topo"].append(np.full(n, str(cl))), out["f"].append(Z["f"][j[w]])
    return T, {k: np.concatenate(v) for k, v in out.items()}, turns


def plans(tag, T, X, dev, bs=256):
    """Rear-axle plan poses (n, 8, 3) of checkpoint `tag` on the rows (route features fed iff the checkpoint has the branch)."""
    import torch
    import onp_loss as OL
    import pp_train as T_
    import rmem as RM
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    m = RM.load_pmodel(tag, dev)
    br = RM.has_branch(tag)
    W = torch.as_tensor(R2.t_weights(T_.T8), device=dev)
    pi = torch.as_tensor(np.arange(m.net.slices["plan"].start, m.net.slices["plan"].start + 495), device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(X["ci"]), bs):
            c = X["ci"][i:i + bs]
            front = torch.as_tensor(np.stack([X["_tok"][T["tix"][r]] for r in c]), device=dev)
            ego = np.asarray(T["ego_in"][c], np.float32)
            if br:
                ego = np.concatenate([ego, X["f"][i:i + bs].astype(np.float32)], 1)
            lht = T["lht"][c].astype(bool)
            tc = torch.as_tensor(np.where(lht[:, None], [[0.0, 1.0]], [[1.0, 0.0]]), dtype=torch.float32, device=dev)
            o = m(front, torch.as_tensor(ego, device=dev), tc).float()
            x, y, psi = OL.rear_xy(o[:, pi].view(-1, 33, 15), torch.as_tensor(T["cam_t"][c][:, :2], dtype=torch.float32, device=dev), W)
            out.append(torch.stack([x, y, psi], -1).cpu().numpy())
    return np.concatenate(out)


def score(F, H, X, turns):
    """Plan poses F, human demo H (n, 8, 3) -> per turn metrics and summary."""
    v = lambda G: np.hypot(G[:, 1, 0], G[:, 1, 1])               # noqa: E731  chord from the origin to the 1.0 s pose over 1 s
    vp, vh = v(F), v(H)
    s = X["sgn"]
    per = []
    for t, tu in enumerate(turns):
        m = X["tid"] == t
        r = dict(tu)
        for d in DS:
            q = np.flatnonzero(m & (np.abs(X["d"] - d) <= 4))
            if len(q):
                i = q[np.argmin(np.abs(X["d"][q] - d))]
                r[f"v_plan_{d}"], r[f"v_human_{d}"] = float(vp[i]), float(vh[i])
        a = m & (X["d"] >= -10) & (X["d"] <= 20)
        need2, need4 = s * H[:, 3, 1], s * H[:, 7, 1]
        q = a & (np.abs(need2) >= 0.5)
        r["lat2"] = float(np.median(s[q] * F[q, 3, 1] / need2[q])) if q.any() else None
        q = a & (np.abs(need4) >= 1.0)
        r["side4"] = float(np.mean(s[q] * F[q, 7, 1] > 0)) if q.any() else None
        per.append(r)

    def summ(rs):
        o = dict(turns=len(rs))
        for d in DS:
            p = [(x[f"v_plan_{d}"], x[f"v_human_{d}"]) for x in rs if f"v_plan_{d}" in x]
            if p:
                o[f"v_plan_{d}"], o[f"v_human_{d}"] = float(np.median([u for u, _ in p])), float(np.median([w for _, w in p]))
                o[f"v_excess_{d}"] = float(np.median([u - w for u, w in p]))
        for k in ("lat2", "side4"):
            z = [x[k] for x in rs if x[k] is not None]
            o[k] = float(np.median(z)) if z else None
        return o
    return per, dict(all=summ(per), held_out=summ([x for x in per if not x["train"]]), in_sample=summ([x for x in per if x["train"]]),
                     straight_ahead=summ([x for x in per if x["topo"] in ("straight_exit", "straight_road")]))


def main(a):
    import torch
    dev = torch.device(a.device)
    a._train = set(RP.scenes_of(a.train_scenes)) if a.train_scenes else set()
    T, X, turns = rows_of(a)
    X["_tok"] = np.load(Path(a.cache) / "tok.npy", mmap_mode="r")
    H = T["imit"][X["ci"]].astype(np.float64)
    res = dict(cache=a.cache, rows=int(len(X["ci"])), turns=len(turns), held_out_turns=sum(not t["train"] for t in turns),
               excluded=a.exclude, tags={})
    for tag in a.tags:
        F = plans(tag, T, X, dev).astype(np.float64)
        per, sm = score(F, H, X, turns)
        res["tags"][tag] = dict(summary=sm, per_turn=per)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    hd = "| tag | part | turns | v plan / human @40 | @20 | @0 | v excess @0 | lat2 | side4 |\n|:--|:--|--:|--:|--:|--:|--:|--:|--:|\n"
    f = lambda x, k: "-" if x.get(k) is None else f"{x[k]:.2f}"  # noqa: E731
    md = "".join(f"| {t} | {p} | {s['turns']} | {f(s, 'v_plan_40')} / {f(s, 'v_human_40')} | {f(s, 'v_plan_20')} / {f(s, 'v_human_20')} | "
                 f"{f(s, 'v_plan_0')} / {f(s, 'v_human_0')} | {f(s, 'v_excess_0')} | {f(s, 'lat2')} | {f(s, 'side4')} |\n"
                 for t, v in res["tags"].items() for p, s in v["summary"].items())
    (o / f"{a.name}.md").write_text(f"# Offline PAI turn read `{a.name}` ({res['rows']} rows, {res['turns']} turns, {res['held_out_turns']} held out; "
                                    f"excluded {a.exclude or 'none'})\n\n" + hd + md)
    print(hd + md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True), ap.add_argument("--route", required=True), ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--exclude", default=str(R / "experiments/alpasim/results/pai_exam/hard_exam.txt"))
    ap.add_argument("--train-scenes", nargs="*", default=[]), ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=str(R / "experiments/route_follow/results/pai_openloop")), ap.add_argument("--name", default="read")
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
