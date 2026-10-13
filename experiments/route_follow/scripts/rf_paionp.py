#!/usr/bin/env python3
"""PAIONP (plans/2026-10-13-paionp-prereg.md): imitation on the policy's OWN PAI states. The pool on1_PAIRB0 (rf_pai_pool.py: RB0-F-s0's own
rollouts on the 160 PAI train scenes, 22 695 rows) carries `fut` = onp_rows `imit`, the recovery target built from the human log at the row's
time (the log's own advance over t0 + 0.5 k s from the ego's projection on the logged path, the lateral offset e0 faded over max(8 m, 2 s v0)).
Here it becomes the `both` half of a mixed pool (rm_train --on-mode mixed). envs/op-train; `tag` / `pack` CPU, `r1` one GPU.

  tag    pool tab + the source cache tab (runs/route_night/cache/pol/tab.npz copied to the box) -> per row: the human turn ahead / under the ego
         and the correction statistics fut vs the policy's own plan (`poses_model`). Turn = rf_pai_follow's rule (30 m window, net heading
         change >= 60 deg, change over [a - 10, a + 40] m >= 45 deg, entry / exit at 10 / 90 %) on the row's logged path (cache `path`: the
         log polyline at arcs -20 .. +100 m around the ego's projection, valid points only); turn-approach row = some turn with
         entry - 40 m <= ego station <= exit (bins 40-20, 20-0 m before the entry, in the turn); straight = no turn on the visible path at all
         (rows with a turn further than 40 m ahead or behind the ego are `other`). Per row: dlat2 = (fut - plan) lateral at 2 s, signed + toward
         the turn's inside on turn rows; darc4 = chord arc of fut - plan to 4 s (+ = the human goes farther = faster); d2 = |fut - plan| at 2 s.
         Usable for training: imit valid at all 8 poses (rf_pai_pool fills NaN with 0) and fut crosses no road-edge segment (rf_pait.crosses on the
         row's edge boxes, rule of rf_pait's G-road). -> runs/route_follow/paionp/tags.npz, <out>/correction.{json,md}
  pack   the usable rows of train160 scenes (asserted: none in hard_exam / control / hard_train) -> cache/on1_<name> (tab + `turn` approach /
         straight / other, `side`, `d_entry`; agent, road ok False, route subsets; feat; @warp front / teacher subsets), then
         rf_pait.py merge on1_L18RB0:s:self + on1_<name>:p:both is the shell's step (rf_paionp.sh). Prints the --on-weight that puts the PAI rows
         at --share of the draws with turn-approach rows x --turn-w.
  r1     RB0-F-s0 (the driver) and RB0-F-s1 (wrong seed) on the pool tokens + ego ++ route features vs the logged poses_model, per coordinate
         tolerance max(0.03 m, 4 fp16 ulps) on the 8 rear-axle x, y (rf_pait r1's rule); pass: driver over-tolerance share <= 0.01, control >= 0.9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "experiments/route_follow/scripts"), str(R / "experiments/route_mem/lib"), str(R / "experiments/route_mem/scripts"),
                str(R / "experiments/op_parity/scripts"), str(R / "experiments/onpol_nuplan/scripts")]
D = Path(os.environ.get("DATA_DIR", "/data"))
CR, FEAT, OUT = D / "runs/op_parity/cache", D / "runs/route_mem/feat", D / "runs/route_follow/paionp"
EXAM = R / "experiments/alpasim/results/pai_exam"
TRAIN160 = R / "experiments/onpol_pai/results/split/train160.tsv"
APPROACH = 40.0


def scenes_of(f):
    return {l.split()[0] for l in Path(f).read_text().splitlines() if l.startswith("clipgt-")}


def row_turn(P, ok):
    """Logged path at arcs -20 .. +100 (121, 2), valid mask -> (class, side, d_entry): class approach | straight | other, side +1 left / -1
    right / 0, d_entry = entry station - ego station (m, + = entry ahead)."""
    import rf_pai_follow as PF
    idx = np.flatnonzero(ok)
    if len(idx) < 40 or not ok[20]:
        return "other", 0, np.nan
    X = PF.dedup(P[idx].astype(np.float64))
    S = PF.stations(X)
    se = PF.project(X, S, P[20].astype(np.float64))[0]
    s, h = PF.heading(X, S)
    TT = PF.all_turns(s, h, 0.0, S[-1])
    if not TT:
        return "straight", 0, np.nan
    for ent, ext, th in TT:
        if ent - APPROACH <= se <= ext:
            return "approach", int(np.sign(th)), ent - se
    return "other", 0, np.nan


def cmd_tag(a):
    import rf_pait as PT
    tb = np.load(CR / f"on1_{a.pool}" / "tab.npz")
    C = np.load(a.cache)
    names = tb["names"]
    key = {(s, int(k)): i for i, (s, k) in enumerate(zip(C["scenes"][C["sid"]].tolist(), C["k"].tolist()))}
    ci = np.array([key[(n.split(":")[2], int(n.split(":")[3]))] for n in names.tolist()])
    n = len(ci)
    assert np.array_equal(C["ego_in"][ci].astype(np.float32), tb["ego"]), "pool ego != cache ego_in"
    fut_c = C["imit"][ci]
    iok = C["imit_ok"][ci].all(1)
    assert np.allclose(np.nan_to_num(fut_c), tb["fut"], atol=1e-6), "pool fut != cache imit"
    plan = C["poses_model"][ci].astype(np.float64)
    fut = np.nan_to_num(fut_c).astype(np.float64)
    cls, side, de = np.empty(n, "U8"), np.zeros(n, np.int8), np.full(n, np.nan, np.float32)
    for i in range(n):
        cls[i], side[i], de[i] = row_turn(C["path"][ci[i]], C["path_ok"][ci[i]])
    cross = PT.crosses(fut.astype(np.float32), C["edge_box"][ci], C["edge_ok"][ci])
    use = iok & ~cross
    dlat2 = (fut[:, 3, 1] - plan[:, 3, 1]) * np.where(side != 0, side, 1)
    darc4 = PT.chord(fut) - PT.chord(plan)
    d2 = np.hypot(*(fut[:, 3, :2] - plan[:, 3, :2]).T)
    v0, e0, ds0 = C["v0"][ci], C["e0"][ci], C["ds0"][ci]
    sc = np.array([x.split(":")[2] for x in names.tolist()])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "tags.npz", names=names, turn=cls, side=side, d_entry=de, imit_ok=iok, cross=cross, use=use, dlat2=dlat2, darc4=darc4, d2=d2,
             v0=v0, e0=e0, ds0=ds0, scene=sc)

    def q(x):
        x = x[np.isfinite(x)]
        return dict(n=int(len(x)), median=float(np.median(x)), q25=float(np.quantile(x, 0.25)), q75=float(np.quantile(x, 0.75))) if len(x) else dict(n=0)

    def block(m):
        return dict(rows=int(m.sum()), scenes=int(len(np.unique(sc[m]))), dlat2=q(dlat2[m]), abs_dlat2=q(np.abs(dlat2[m])),
                    share_abs_dlat2_gt_05=float((np.abs(dlat2[m]) > 0.5).mean()) if m.any() else np.nan, darc4=q(darc4[m]), d2=q(d2[m]),
                    v0=q(v0[m]), abs_e0=q(np.abs(e0[m])), ds0=q(ds0[m]))
    ap = cls == "approach"
    groups = {"all usable": use, "turn approach (entry - 40 m .. exit)": use & ap, "  40 .. 20 m before entry": use & ap & (de > 20),
              "  20 .. 0 m before entry": use & ap & (de <= 20) & (de > 0), "  in the turn (entry .. exit)": use & ap & (de <= 0),
              "  left turns": use & ap & (side > 0), "  right turns": use & ap & (side < 0), "straight (no turn on the path)": use & (cls == "straight"),
              "other (turn > 40 m ahead / behind)": use & (cls == "other")}
    res = dict(pool=a.pool, rows=n, imit_ok=int(iok.sum()), fut_crosses_edge=int((iok & cross).sum()), usable=int(use.sum()),
               rows_by_turn={c: int((cls == c).sum()) for c in ("approach", "straight", "other")},
               usable_by_turn={c: int((use & (cls == c)).sum()) for c in ("approach", "straight", "other")},
               turns_scenes=int(len(np.unique(sc[use & ap]))), groups={k: block(m) for k, m in groups.items()})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "correction.json").write_text(json.dumps(res, indent=1))
    f = lambda d: f"{d['median']:+.2f} [{d['q25']:+.2f}, {d['q75']:+.2f}]" if d.get("n") else "-"  # noqa: E731
    L = [f"# on1_{a.pool}: fut (human-log recovery target) vs the policy's own plan (poses_model), rf_paionp.py tag", "",
         f"Rows {n}; imit valid at all 8 poses {iok.sum()}; of those fut crosses a road edge {int((iok & cross).sum())}; usable {use.sum()}.",
         f"Turn class of all rows: {res['rows_by_turn']}; usable: {res['usable_by_turn']}; scenes with usable turn-approach rows {res['turns_scenes']}.", "",
         "Median [IQR]. dlat2 = fut - plan lateral at 2 s (turn rows: + = toward the turn's inside); darc4 = fut - plan chord arc to 4 s (+ = human farther);"
         " d2 = |fut - plan| at 2 s.", "",
         "| rows | n | scenes | dlat2 m | abs dlat2 > 0.5 m | darc4 m | d2 m | v0 m/s | abs e0 m | ds0 m |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, b in res["groups"].items():
        L.append(f"| {k} | {b['rows']} | {b['scenes']} | {f(b['dlat2'])} | {b['share_abs_dlat2_gt_05']:.2f} | {f(b['darc4'])} | {f(b['d2'])} | "
                 f"{f(b['v0'])} | {f(b['abs_e0'])} | {f(b['ds0'])} |")
    (out / "correction.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_pack(a):
    Z = np.load(OUT / "tags.npz")
    src, dst, dw = CR / f"on1_{a.pool}", CR / f"on1_{a.name}", CR / f"on1_{a.name}@warp"
    tb = dict(np.load(src / "tab.npz"))
    assert np.array_equal(Z["names"], tb["names"])
    sc = Z["scene"]
    tr, bad = scenes_of(TRAIN160), set().union(*(scenes_of(EXAM / f) for f in ("hard_exam.txt", "control.txt", "hard_train.txt")))
    keep = np.flatnonzero(Z["use"] & np.isin(sc, list(tr)))
    assert not (set(sc[keep]) & bad), "exam / control scene in the pool"
    assert set(sc[keep]) <= tr
    n = len(keep)
    dst.mkdir(parents=True, exist_ok=True), dw.mkdir(parents=True, exist_ok=True)
    T = {k: v[keep] for k, v in tb.items()} | dict(turn=Z["turn"][keep].astype("U8"), side=np.array(["left", "right", ""])[np.where(Z["side"][keep] > 0, 0, np.where(Z["side"][keep] < 0, 1, 2))],
                                                    d_entry=Z["d_entry"][keep], src=np.full(n, "policy"))
    assert np.isfinite(T["fut"]).all()
    np.savez(dst / "tab.npz", **T)
    ag = np.load(src / "agent.npz")
    np.savez(dst / "agent.npz", tokens=T["names"], ok=ag["ok"][keep], box=ag["box"][keep], valid=ag["valid"][keep])
    np.savez(dst / "road.npz", tokens=T["names"], ok=np.zeros(n, bool), sdf=np.zeros((n, 128, 96), np.float16))
    rt = np.load(src / "route.npz")
    np.savez(dst / "route.npz", tokens=T["names"], **{k: rt[k][keep] for k in ("ok", "poly", "seglen", "dz", "cov20")})
    fz = np.load(FEAT / f"on1_{a.pool}.npz")
    assert np.array_equal(fz["names"], tb["names"])
    np.savez(FEAT / f"on1_{a.name}.npz", names=T["names"], f=fz["f"][keep], k=fz["k"][keep], ok=fz["ok"][keep])
    th = np.load(CR / f"on1_{a.pool}@warp/teacher.npz")
    np.savez(dw / "teacher.npz", out=th["out"][keep], plan=th["plan"][keep], di=th["di"], pi=th["pi"])
    fr = np.load(CR / f"on1_{a.pool}@warp/front.npy", mmap_mode="r")
    o = np.lib.format.open_memmap(dw / "front.npy", "w+", np.float16, (n,) + fr.shape[1:])
    for i in range(0, n, 1024):
        o[i:i + 1024] = fr[keep[i:i + 1024]]
    o.flush()
    rs = np.random.default_rng(0).choice(n, min(n, 256), replace=False)
    eq = all(np.array_equal(o[i], fr[keep[i]]) for i in rs)
    del o
    nt = int((T["turn"] == "approach").sum())
    ns = len(np.load(CR / f"on1_{a.base}" / "tab.npz")["names"])
    w = a.share / (1 - a.share) * ns / ((n - nt) + a.turn_w * nt)
    rec = dict(name=a.name, rows=n, scenes=int(len(np.unique(T["log"]))), rows_by_turn={c: int((T["turn"] == c).sum()) for c in ("approach", "straight", "other")},
               front_equal=bool(eq), base=a.base, base_rows=ns, share=a.share, turn_w=a.turn_w, pool_w=float(w),
               pai_draw_share=float(w * ((n - nt) + a.turn_w * nt) / (w * ((n - nt) + a.turn_w * nt) + ns)),
               approach_share_of_pai_draws=float(a.turn_w * nt / ((n - nt) + a.turn_w * nt)),
               on_weight=f"pool=p:{w:.4f},turn=approach:{a.turn_w:g}")
    (dst / "build.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


def cmd_r1(a):
    import torch
    import pp_train as T
    import rmem as RM
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive import op_adapt as A
    from rf_gates import plans
    dev = torch.device("cuda")
    tb = np.load(CR / f"on1_{a.pool}" / "tab.npz")
    C = np.load(a.cache)
    names = tb["names"]
    key = {(s, int(k)): i for i, (s, k) in enumerate(zip(C["scenes"][C["sid"]].tolist(), C["k"].tolist()))}
    ci = np.array([key[(x.split(":")[2], int(x.split(":")[3]))] for x in names.tolist()])
    rows = np.sort(np.random.default_rng(0).choice(len(names), min(a.n, len(names)), replace=False)) if a.n else np.arange(len(names))
    fz = np.load(FEAT / f"on1_{a.pool}.npz")
    assert np.array_equal(fz["names"], names)
    ego = np.concatenate([tb["ego"][rows], fz["f"][rows].astype(np.float32)], 1)
    tc = torch.as_tensor(np.where(tb["lht"][rows, None], [[0.0, 1.0]], [[1.0, 0.0]]), dtype=torch.float32, device=dev)
    cam = torch.as_tensor(tb["cam"][rows, 0], dtype=torch.float32, device=dev)
    fr = np.load(CR / f"on1_{a.pool}@warp/front.npy", mmap_mode="r")
    ref = C["poses_model"][ci[rows]][..., :2].astype(np.float64)
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(ref), 2.0 ** -14))) - 10)
    tol = np.maximum(4 * ulp, 0.03)
    W = torch.as_tensor(R2.t_weights(T.T8), device=dev, dtype=torch.float32)
    res = {}
    for tag in [a.tag] + a.control:
        m = RM.load_pmodel(tag, dev)
        pi = torch.as_tensor(A.plan_index(m.net.slices), device=dev)
        P = plans(m, fr, rows, ego, tc, cam, pi, W, dev)[..., :2]
        d = np.abs(P - ref)
        res[tag] = dict(rows=int(len(rows)), share_over=float((d > tol).reshape(len(rows), -1).any(1).mean()), max_m=float(d.max()),
                        median_m=float(np.median(d.reshape(len(rows), -1).max(1))), p99_m=float(np.quantile(d.reshape(len(rows), -1).max(1), 0.99)),
                        share_gt_005=float((d.reshape(len(rows), -1).max(1) > 0.05).mean()), share_gt_010=float((d.reshape(len(rows), -1).max(1) > 0.10).mean()))
        del m
        torch.cuda.empty_cache()
    g = dict(pool=a.pool, model=res[a.tag], control={c: res[c] for c in a.control},
             pass_=bool(res[a.tag]["share_over"] <= 0.01 and all(res[c]["share_over"] >= 0.9 for c in a.control)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"r1_{a.pool}.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))
    raise SystemExit(0 if g["pass_"] else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("tag", "pack", "r1"):
        p = sub.add_parser(c)
        p.add_argument("--pool", default="PAIRB0")
        p.add_argument("--cache", default=str(OUT / "src/cache_pol_tab.npz"))
        p.add_argument("--out", default=str(R / "experiments/route_follow/results/paionp"))
    p = sub.choices["pack"]
    p.add_argument("--name", default="PAIonp"), p.add_argument("--base", default="L18RB0")
    p.add_argument("--share", type=float, default=0.30), p.add_argument("--turn-w", type=float, default=2.0)
    p = sub.choices["r1"]
    p.add_argument("--tag", default="RB0-F-s0"), p.add_argument("--control", nargs="*", default=["RB0-F-s1"]), p.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    dict(tag=cmd_tag, pack=cmd_pack, r1=cmd_r1)[a.cmd](a)
