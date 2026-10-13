#!/usr/bin/env python3
"""SIDE-TAG (fix 1 of research/route-follow/side-bias.md; plans/2026-10-13-rsgboth-prereg.md Amendment 2): per-row turn-side x path-side tags
on the on-policy pool, so rm_train --on-weight can rebalance the right-turn rows whose state sits outside vs inside the human path.

  pool  (box, repo .venv)  rf_sidetag.py pool --src on1_L18RB0 --out on1_L18RB0s [--weight rside=R_in:2,rside=R_out:0.48] [--dry-out JSON]
        Hard-linked copy of the pool (every file but tab.npz is the source's inode, warp dir symlinked, route_mem feature file copied, as
        rf_rsgbv.py pool) whose tab.npz gains three string keys and changes nothing else:
          tside  L / R / none      turn side on the human's logged path (rf_sidebias Poly.turn: >= 60 deg over a 30 m window, ego inside
                                   it or <= 20 m before; none = no such turn, straight rows included)
          pside  in / on / out / none   the state's offset e0 to that path, inside-signed (meta e0): in < -0.5 m, out > 0.5 m, on between;
                                   none on non-turn rows (rf_sidebias EBIN, the 'inside' / 'outside' bins of side-bias.md table 1)
          rside  <tside>_<pside> on turn rows (L_in, R_out, ...), else none: the AND of the two, since --on-weight multiplies one factor per
                                   KEY=VALUE term and has no conjunction.
        Gate R1: every other file same inode, every source tab array byte-equal (names, features, fut / fut_reg, served, ...), tags aligned with
        meta names; for --src on1_L18RB0 the counts must reproduce side-bias.md (R_out 321, R_in 77); any pool records r_out_in / l_out_in.
        Dry run of --weight: rm_train's own parsing (tab key, == value, product of factors; a missing key is a factor 1), then 1000 batches of
        --on-k 13 rows drawn with rm_train's generator (default_rng([seed, 0, 53]).choice(n, 13, replace=True, p=w / w.sum())): drawn rows per
        rside, against the uniform draw and against the expectation; plus the weighted mean imitation push at 2 s (fut - served lateral,
        inside -) per side, before and after, from the same classification (side_metrics of rf_sidebias).
  read  (Mac, .venv)  rf_sidetag.py read --held DIR --pairs A1,A2:B1,B2 ... --out MD
        DIR holds rf_sidebias.py held CSVs (held_<label>.csv.gz). Per side (L / R / S), per scene the label group's mean plan / driven lateral
        at 2 s (inside -), then A - B paired by scene, log-clustered bootstrap 95 % CI (rf_sidebias.ci). Also each label's own means.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_sidebias as SB  # noqa: E402


def classify(meta, tab):
    n = len(meta["names"])
    ts, ps = np.full(n, "none", "<U4"), np.full(n, "none", "<U4")
    push = np.full(n, np.nan)
    fut, srv = tab["fut"].astype(float), tab["served"].astype(float)
    for r in range(n):
        G = SB.Poly(meta["gpath"][r])
        se = float(G.project(np.zeros((1, 2)))[0][0])
        side, _, _, _, _ = G.turn(se)
        if not side:
            continue
        e = -side * float(meta["e0"][r])
        ts[r], ps[r] = ("L" if side > 0 else "R"), ("in" if e < -SB.EBIN else "out" if e > SB.EBIN else "on")
        fl = SB.side_metrics(G, fut[r, [3]], side, se - 5, se + 60)[0][0]
        sl = SB.side_metrics(G, srv[r, [3]], side, se - 5, se + 60)[0][0]
        push[r] = fl - sl
    rs = np.where(ts == "none", "none", np.char.add(np.char.add(ts, "_"), ps))
    return ts, ps, rs.astype("<U6"), push


def weights(tab, spec):
    """rm_train cmd_train's --on-weight parsing, verbatim in effect."""
    w = np.ones(len(tab["names"]))
    for term in spec.split(","):
        kv, f = term.rsplit(":", 1)
        k, v = kv.split("=", 1)
        if k in tab:
            w[np.asarray(tab[k]).astype(str) == v] *= float(f)
    return w


def cmd_pool(a):
    D = Path(os.environ["DATA_DIR"])
    CR, FE = D / "runs/op_parity/cache", D / "runs/route_mem/feat"
    src, dst = CR / a.src, CR / a.out
    assert not dst.exists(), f"{dst} exists"
    tab = dict(np.load(src / "tab.npz"))
    meta = np.load(D / "runs/onpol_nuplan/rows" / a.src.removeprefix("on1_") / "meta.npz", allow_pickle=True)
    assert np.array_equal(meta["names"], tab["names"]), "meta / tab rows"
    ts, ps, rs, push = classify(meta, tab)
    dst.mkdir()
    for f in src.iterdir():
        if f.name != "tab.npz":
            os.link(f, dst / f.name)
    np.savez(dst / "tab.npz", **(tab | dict(tside=ts, pside=ps, rside=rs)))
    (CR / f"{a.out}@warp").symlink_to(f"{a.src}@warp")
    shutil.copy(FE / f"{a.src}.npz", FE / f"{a.out}.npz")
    t2 = dict(np.load(dst / "tab.npz"))
    cnt = {v: int((rs == v).sum()) for v in np.unique(rs)}
    g = dict(rows=len(rs), rside=cnt, tside={v: int((ts == v).sum()) for v in np.unique(ts)},
             inode_same=all((dst / f.name).stat().st_ino == f.stat().st_ino for f in src.iterdir() if f.name != "tab.npz"),
             arrays_same=all(t2[k].dtype == v.dtype and t2[k].shape == v.shape and t2[k].tobytes() == v.tobytes() for k, v in tab.items()),
             new_keys=sorted(set(t2) - set(tab)), feat_same=(FE / f"{a.out}.npz").read_bytes() == (FE / f"{a.src}.npz").read_bytes(),
             reproduces_side_bias=(cnt.get("R_out") == 321 and cnt.get("R_in") == 77) if a.src == "on1_L18RB0" else None,
             r_out_in=[cnt.get("R_out", 0), cnt.get("R_in", 0)], l_out_in=[cnt.get("L_out", 0), cnt.get("L_in", 0)])
    g["pass"] = bool(g["inode_same"] and g["arrays_same"] and g["feat_same"] and g["new_keys"] == ["pside", "rside", "tside"]
                     and g["reproduces_side_bias"] is not False)
    # dry run of the weights, rm_train's draw
    w = weights(t2, a.weight)
    p = w / w.sum()
    rng = np.random.default_rng([a.seed, 0, 53])
    dr = np.concatenate([rng.choice(len(w), a.on_k, replace=True, p=p) for _ in range(a.draws)])
    ur = np.random.default_rng([a.seed, 0, 53]).choice(len(w), a.draws * a.on_k, replace=True)
    dry = {}
    for v in cnt:
        m = rs == v
        dry[v] = dict(rows=int(m.sum()), weight=float(w[m][0]), drawn=int(m[dr].sum()), drawn_uniform=int(m[ur].sum()),
                      expected=float(p[m].sum() * len(dr)))
    pu = {}
    for sd in "LR":
        m = (ts == sd) & np.isfinite(push)
        pu[sd] = dict(push2_uniform=float(push[m].mean()), push2_weighted=float((push[m] * w[m]).sum() / w[m].sum()),
                      push2_drawn=float(np.nanmean(push[dr][m[dr]])))
    g["dry"] = dict(weight=a.weight, draws=a.draws, on_k=a.on_k, rows_drawn=int(len(dr)), seed=a.seed, per_rside=dry, push=pu,
                    r_out_in_drawn=dry["R_out"]["drawn"] / max(dry["R_in"]["drawn"], 1),
                    r_out_in_uniform=dry["R_out"]["drawn_uniform"] / max(dry["R_in"]["drawn_uniform"], 1),
                    p_min_max_x_uniform=[float(p.min() * len(p)), float(p.max() * len(p))])
    (CR / f"{a.out}.r1.json").write_text(json.dumps(g, indent=1))
    if a.dry_out:
        Path(a.dry_out).write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))
    raise SystemExit(0 if g["pass"] else 2)


def cmd_read(a):
    import pandas as pd
    H = Path(a.held)
    md = ["# Held 460 @18: plan / driven lateral at 2 s against the human path, per turn side (rf_sidetag.py read; inside -, straight left +)", ""]
    labs = sorted({x for p in a.pairs for g in p.split(":") for x in g.split(",")})
    T = {lb: pd.read_csv(H / f"held_{lb}.csv.gz") for lb in labs}
    md += ["| label | side | decisions | plan lat 2 s | driven lat 2 s | driven - plan 2 s |", "|:--|:--|--:|:--|:--|:--|"]
    for lb in labs:
        X = T[lb]
        for sd in "LRS":
            Y = X[X.side == sd]
            g = SB.logof(Y.scene)
            md.append(f"| {lb} | {sd} | {len(Y)} | {SB.ci(Y['plan_lat2.0'], g)} | {SB.ci(Y['drv_lat2.0'], g)} | {SB.ci(Y['dlat2.0'], g)} |")
    md += ["", "Paired by scene (group mean over its seeds per scene, then A - B):", "",
           "| pair | side | scenes | plan lat 2 s | driven lat 2 s |", "|:--|:--|--:|:--|:--|"]
    out = []

    def per_scene(grp, sd, c):
        return pd.concat([T[x][T[x].side == sd].groupby("scene")[c].mean() for x in grp.split(",")], axis=1).mean(axis=1)
    for p in a.pairs:
        A, B = p.split(":")
        for sd in "LRS":
            cells = []
            for c in ("plan_lat2.0", "drv_lat2.0"):
                d = (per_scene(A, sd, c) - per_scene(B, sd, c)).dropna()
                m, lo, hi, n = SB.boot(d.values, SB.logof(d.index))
                out.append(dict(pair=p, side=sd, metric=c, scenes=len(d), diff=m, lo=lo, hi=hi))
                cells.append(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            md.append(f"| {A} - {B} | {sd} | {len(d)} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(md) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="cmd", required=True)
    q = s.add_parser("pool")
    q.add_argument("--src", default="on1_L18RB0"), q.add_argument("--out", required=True)
    q.add_argument("--weight", default="rside=R_in:2,rside=R_out:0.48"), q.add_argument("--draws", type=int, default=1000)
    q.add_argument("--on-k", type=int, default=13), q.add_argument("--seed", type=int, default=0), q.add_argument("--dry-out", default="")
    q = s.add_parser("read")
    q.add_argument("--held", required=True), q.add_argument("--pairs", nargs="+", required=True), q.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(pool=cmd_pool, read=cmd_read)[a.cmd](a)
