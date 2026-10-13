#!/usr/bin/env python3
"""RSGboth variants (plans/2026-10-13-rsgboth-prereg.md): where RSGboth's extra zeros come from, and the c2 pool (straight rows' imitation
target capped at the model's own speed). Box; `diag` / `pool` in AlpaSim's venv, CPU.

  diag --collect DIR --follow DIR --pairs A:B .. --out DIR
      DIR = zeros_mech collect (zm_read.py collect, labels <tag>~c<i>) of the held 460 @18 rollouts, follow = an rf_follow.py output of the same
      tags (@18 labels). Per zero rollout (score 0): event type (zm_read ev_type: offroad / collision / corridor), event decision k_ev, ego speed
      v_e there and the log's speed at the same time (v_log, from the logged station), road class at k_ev (junction = ego rear axle in an
      intersection lane (zm junc_rear) or in rf_follow's junction stretch; curve = |route heading change over the next 60 m| >= 15 deg;
      else straight), lead / box object in the ego corridor within the last 3 decisions (zm `lead` / `box`), turn scene (rf_follow).
      Extra zeros of A over B = scenes zero in A and scored in B. Speed: mean v_e and v_e - v_log over decisions both labels visited before
      either's first zero event, by road class.  -> diag.json, zeros.csv, diag.md
  pool --src on1_L18RB0 --out on1_L18RB0t | --out on1_L18RB0c2 --cap
      hard-linked copy of the pool (warp dir symlinked, feature file copied) whose tab gains the string key `rclass` (rm_train --on-weight):
      route label (route.npz poly) heading change over its first 40 m and the human target's yaw at 4 s -> turn (>= 45 deg either),
      curve (>= 15 / >= 10 deg), straight (else), none (no route label). --cap: tab `fut` / `fut_reg` changed on the NON-TURN rows (straight,
      curve) with v0 >= 2 m/s only: the target keeps its path, its arc length at t = 0.5 .. 4 s is capped at max(arc of the served plan at t,
      v0 t) (v0 = tab `speed`), i.e. the imitation target never asks for more progress than the row's own speed (or its served plan) holds;
      it still slows where the human slowed. Turn rows, rows without a label and launches (v0 < 2 m/s) keep the human's target.
      Gate R1: every other file is the source's inode, every tab array except fut / fut_reg byte-equal, the uncapped rows' fut / fut_reg
      byte-equal, capped poses on the source path (<= 2 cm), capped arcs <= the source arcs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

KW = 3                                     # decisions before the event searched for a lead / box


def cls_at(d):
    j = d.junc_rear.astype(bool) | d.junction.fillna(False).astype(bool)
    return np.where(j, "junction", np.where(d.route60.abs() >= 15, "curve", "straight"))


def diag(a):
    import pandas as pd
    R = pd.read_parquet(Path(a.collect) / "per_rollout.parquet")
    D = pd.read_parquet(Path(a.collect) / "per_decision.parquet")
    for X in (R, D):
        X["tag"] = X.label.str.split("~").str[0]
    Fd = pd.read_parquet(Path(a.follow) / "decisions.parquet")
    Fs = pd.read_csv(Path(a.follow) / "scenes.csv")
    for X in (Fd, Fs):
        X["tag"] = X.label.str.split("@").str[0]
    D = D.merge(Fd[["tag", "scene", "k", "junction"]], on=["tag", "scene", "k"], how="left")
    D = D.sort_values(["tag", "scene", "k"])
    D["v_log"] = D.groupby(["tag", "scene"]).s_log_t.diff().shift(-1) / 0.5
    D["v_log"] = D.v_log.fillna(D.groupby(["tag", "scene"]).s_log_t.diff() / 0.5)
    D["cls"] = cls_at(D)
    turn = Fs.assign(turn=Fs.turn.astype(str).isin(["True", "1", "1.0"]))[["tag", "scene", "turn", "outcome"]]
    R = R[R.has_rollout].merge(turn, on=["tag", "scene"], how="left")
    Z = R[R.score == 0].copy()
    rows = []
    for _, z in Z.iterrows():
        dd = D[(D.tag == z.tag) & (D.scene == z.scene)]
        ke = int(z.k_ev) if z.k_ev >= 0 else int(z.k_end)
        e = dd[dd.k == ke]
        w = dd[(dd.k >= ke - KW) & (dd.k <= ke)]
        e = e.iloc[0] if len(e) else (dd.iloc[-1] if len(dd) else None)
        rows.append(dict(tag=z.tag, scene=z.scene, type=z.type, kinds=z.kinds, k_ev=ke, turn_scene=bool(z.turn) if pd.notna(z.turn) else False,
                         cls=e.cls if e is not None else "", v_e=float(e.v_e) if e is not None else np.nan,
                         v_log=float(e.v_log) if e is not None else np.nan, route60=float(e.route60) if e is not None else np.nan,
                         lead=bool(w.lead.any()), box=bool(w.box.any()), box_gap=float(w.box_gap.min()) if w.box.any() else np.nan,
                         hit_static=z.get("hit_static"), hit_dpsi=z.get("hit_dpsi"), hit_x=z.get("hit_x"), hit_y=z.get("hit_y"), hit_v=z.get("hit_v"),
                         at_fault=float(z.collision_at_fault or 0) > 0))
    Zt = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(collect=str(a.collect), follow=str(a.follow), pairs={})
    md = ["# RSGboth extra zeros: diagnosis (held 460 @18, rf_rsgbv.py diag)", ""]
    for p in a.pairs:
        A, B = p.split(":")
        za, zb = set(Zt[Zt.tag == A].scene), set(Zt[Zt.tag == B].scene)
        both = set(R[R.tag == A].scene) & set(R[R.tag == B].scene)
        xa = Zt[(Zt.tag == A) & Zt.scene.isin(both - zb)].assign(set="extra_A")
        xb = Zt[(Zt.tag == B) & Zt.scene.isin(both - za)].assign(set="extra_B")
        sh = Zt[(Zt.tag == A) & Zt.scene.isin(za & zb)].assign(set="shared")
        Zt.loc[xa.index.intersection(Zt.index), "extra"] = p
        def summ(x):
            return dict(n=int(len(x)), type=x.type.value_counts().to_dict(), cls=x.cls.value_counts().to_dict(), turn_scene=int(x.turn_scene.sum()),
                        lead=int(x.lead.sum()), box=int(x.box.sum()), at_fault=int(x.at_fault.sum()),
                        v_e_med=float(x.v_e.median()) if len(x) else None, v_log_med=float(x.v_log.median()) if len(x) else None,
                        v_ratio_med=float((x.v_e / x.v_log.clip(lower=0.5)).median()) if len(x) else None,
                        by_type_cls=x.groupby(["type", "cls"]).size().rename("n").reset_index().to_dict("records"))
        # speed on visited decisions before either label's first event
        ka = R[R.tag == A].set_index("scene").k_end
        kb = R[R.tag == B].set_index("scene").k_end
        da, db = D[D.tag == A], D[D.tag == B]
        m = da.merge(db[["scene", "k", "v_e", "v_log"]], on=["scene", "k"], suffixes=("_a", "_b"))
        lim = np.minimum(m.scene.map(ka).fillna(99), m.scene.map(kb).fillna(99))
        m = m[(m.k < lim) & (m.k >= 2)]
        sp = {}
        for c, g in m.groupby("cls"):
            dv = (g.v_e_a - g.v_e_b).to_numpy()
            sc = g.scene.to_numpy()
            us = np.unique(sc)
            idx = np.searchsorted(us, sc)
            s1, n1 = np.bincount(idx, dv, len(us)), np.bincount(idx, minlength=len(us))
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(2000):
                cnt = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us))
                bs.append((cnt * s1).sum() / max((cnt * n1).sum(), 1))
            sp[c] = dict(decisions=int(len(g)), scenes=int(len(us)), v_a=float(g.v_e_a.mean()), v_b=float(g.v_e_b.mean()), v_log=float(g.v_log_a.mean()),
                         dv=float(dv.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                         over_log_a=float((g.v_e_a - g.v_log_a).mean()), over_log_b=float((g.v_e_b - g.v_log_b).mean()),
                         share_fast_a=float((g.v_e_a > g.v_log_a + 1).mean()), share_fast_b=float((g.v_e_b > g.v_log_b + 1).mean()))
        res["pairs"][p] = dict(zeros_A=len(za & both), zeros_B=len(zb & both), extra_A=summ(xa), extra_B=summ(xb), shared=summ(sh), speed=sp)
        md += [f"## {A} vs {B}", "", f"zeros {len(za & both)} vs {len(zb & both)}; extra in A {len(xa)}, extra in B {len(xb)}, shared {len(sh)}", "",
               "| set | n | offroad / collision / corridor | straight / curve / junction | turn scene | lead / box (last 3 dec) | v_e median (log) |",
               "|:--|--:|:--|:--|--:|:--|:--|"]
        for nm, x in (("extra A", xa), ("extra B", xb), ("shared (A)", sh)):
            t, c = x.type.value_counts(), x.cls.value_counts()
            md.append(f"| {nm} | {len(x)} | {t.get('offroad', 0)} / {t.get('collision', 0)} / {t.get('corridor', 0)} | {c.get('straight', 0)} / "
                      f"{c.get('curve', 0)} / {c.get('junction', 0)} | {int(x.turn_scene.sum())} | {int(x.lead.sum())} / {int(x.box.sum())} | "
                      f"{x.v_e.median():.1f} ({x.v_log.median():.1f}) |")
        md += ["", "| road class | decisions | v A / B / log (m/s) | A - B [95 % CI] | A - log / B - log | share > log + 1 m/s A / B |", "|:--|--:|:--|:--|:--|:--|"]
        for c, s in sp.items():
            md.append(f"| {c} | {s['decisions']} | {s['v_a']:.2f} / {s['v_b']:.2f} / {s['v_log']:.2f} | {s['dv']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] | "
                      f"{s['over_log_a']:+.2f} / {s['over_log_b']:+.2f} | {s['share_fast_a']:.2f} / {s['share_fast_b']:.2f} |")
        md += ["", "Extra zeros of A (scene, type, class, v_e / v_log, lead, box gap):", ""]
        for _, x in xa.sort_values(["type", "cls"]).iterrows():
            md.append(f"- {x.scene} {x.type} {x.cls} k {x.k_ev} v {x.v_e:.1f} / {x.v_log:.1f} lead {int(x.lead)} box {x.box_gap:.0f} turn {int(x.turn_scene)}"
                      + (f" hit static {x.hit_static} dpsi {x.hit_dpsi:.0f} x {x.hit_x:.1f} y {x.hit_y:.1f}" if x.type == "collision" and pd.notna(x.hit_x) else ""))
        md.append("")
    Zt.to_csv(out / "zeros.csv", index=False)
    (out / "diag.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "diag.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def arc(P):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def cap_row(fut, served, v0):
    """fut / served (8, 3) in the row frame -> capped fut (8, 3), changed flag. Path = origin + fut poses (piecewise linear)."""
    P = np.r_[[[0.0, 0.0, 0.0]], fut.astype(np.float64)]
    S = arc(P[:, :2])
    Ss = arc(np.r_[[[0.0, 0.0]], served[:, :2].astype(np.float64)])[1:]
    t = 0.5 * np.arange(1, 9)
    cap = np.maximum(Ss, v0 * t)
    s = np.minimum(S[1:], cap)
    s = np.maximum.accumulate(s)
    if np.all(s >= S[1:] - 1e-6):
        return fut, False
    x, y = np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1])
    yaw = np.interp(s, S, np.unwrap(P[:, 2]))
    return np.stack([x, y, yaw], 1).astype(fut.dtype), True


def pool(a):
    D = Path(os.environ["DATA_DIR"])
    CR, FE = D / "runs/op_parity/cache", D / "runs/route_mem/feat"
    src, dst = CR / a.src, CR / a.out
    assert not dst.exists(), f"{dst} exists"
    tab = dict(np.load(src / "tab.npz"))
    r = np.load(src / "route.npz")
    assert np.array_equal(r["tokens"], tab["names"])
    poly = r["poly"].astype(np.float64)
    seg = np.diff(poly, axis=1)
    hd = np.unwrap(np.arctan2(seg[..., 1], seg[..., 0]), axis=1)
    cum = np.cumsum(r["seglen"], 1)
    i40 = np.minimum((cum < 40).sum(1), seg.shape[1] - 1)
    rturn = np.degrees(np.abs(hd[np.arange(len(hd)), i40] - hd[:, 0]))
    fy = np.degrees(np.abs(tab["fut"][:, 7, 2]))
    turn = r["ok"] & ((rturn >= 45) | (fy >= 45))
    curve = r["ok"] & ~turn & ((rturn >= 15) | (fy >= 10))
    rcls = np.where(~r["ok"], "none", np.where(turn, "turn", np.where(curve, "curve", "straight")))
    capr = (rcls == "straight") | (rcls == "curve")
    capr &= (tab["speed"] >= 2.0) & a.cap
    fut, fr = tab["fut"].copy(), tab["fut_reg"].copy()
    ch = np.zeros(len(fut), bool)
    for i in np.flatnonzero(capr):
        fut[i], ch[i] = cap_row(tab["fut"][i], tab["served"][i], float(tab["speed"][i]))
        if ch[i]:
            fr[i], _ = cap_row(tab["fut_reg"][i], tab["served"][i], float(tab["speed"][i]))
    dst.mkdir()
    for f in src.iterdir():
        if f.name != "tab.npz":
            os.link(f, dst / f.name)
    np.savez(dst / "tab.npz", **(tab | dict(fut=fut, fut_reg=fr, rclass=rcls)))
    (CR / f"{a.out}@warp").symlink_to(f"{a.src}@warp")
    import shutil
    shutil.copy(FE / f"{a.src}.npz", FE / f"{a.out}.npz")
    # gate R1
    t2 = np.load(dst / "tab.npz")
    same = {k: t2[k].dtype == tab[k].dtype and t2[k].shape == tab[k].shape and t2[k].tobytes() == tab[k].tobytes() for k in tab if k not in ("fut", "fut_reg")}
    inode = {f.name: (dst / f.name).stat().st_ino == f.stat().st_ino for f in src.iterdir() if f.name != "tab.npz"}
    ns = ~ch
    dev = []
    for i in np.flatnonzero(ch):
        P = np.r_[[[0.0, 0.0]], tab["fut"][i, :, :2].astype(np.float64)]
        q = t2["fut"][i, :, :2].astype(np.float64)
        a_, b_ = P[:-1], P[1:]
        ab = b_ - a_
        u = np.clip((((q[:, None] - a_) * ab).sum(-1)) / np.maximum((ab ** 2).sum(-1), 1e-9), 0, 1)
        dev.append(float(np.hypot(*(a_ + u[..., None] * ab - q[:, None]).transpose(2, 0, 1)).min(1).max()))
    arc_new = np.array([arc(np.r_[[[0, 0]], t2["fut"][i, :, :2]])[-1] for i in np.flatnonzero(ch)])
    arc_old = np.array([arc(np.r_[[[0, 0]], tab["fut"][i, :, :2]])[-1] for i in np.flatnonzero(ch)])
    g = dict(rows=int(len(fut)), rclass={c: int((rcls == c).sum()) for c in np.unique(rcls)}, cap=bool(a.cap), capped_rows=int(capr.sum()), changed=int(ch.sum()), inode_same=all(inode.values()), arrays_same=all(same.values()),
             uncapped_fut_equal=bool(t2["fut"][ns].tobytes() == tab["fut"][ns].tobytes() and t2["fut_reg"][ns].tobytes() == tab["fut_reg"][ns].tobytes()),
             path_dev_max=float(max(dev) if dev else 0.0), arc_le=bool((arc_new <= arc_old + 1e-4).all()),
             arc4_cut_mean=float((arc_old - arc_new).mean()) if len(ch.nonzero()[0]) else 0.0,
             arc4_cut_median=float(np.median(arc_old - arc_new)) if ch.any() else 0.0)
    g["pass"] = bool(g["inode_same"] and g["arrays_same"] and g["uncapped_fut_equal"] and g["path_dev_max"] <= 0.02 and g["arc_le"])
    (dst.parent / f"{a.out}.r1.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))
    raise SystemExit(0 if g["pass"] else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("diag")
    p.add_argument("--collect", required=True), p.add_argument("--follow", required=True), p.add_argument("--out", required=True)
    p.add_argument("--pairs", nargs="+", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--src", default="on1_L18RB0"), p.add_argument("--out", required=True), p.add_argument("--cap", action="store_true")
    a = ap.parse_args()
    dict(diag=diag, pool=pool)[a.cmd](a)
