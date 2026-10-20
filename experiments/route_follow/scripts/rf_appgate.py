#!/usr/bin/env python3
"""APPROACH COURSE pilot gate (experiments/route_follow/plans/2026-10-15-approach-course-prereg.md, section 3): offline, one checkpoint
against its base on the held-out approach dev rows (dev logs of rf_apppool.py, never in the pool), plus the route v2 gates of the same
checkpoint (rf_v2gate.py's JSON: G-real, G-ST, G-over, G-rely). envs/op-train, one GPU (pool job --vram 20 --cpu 4 --ram 48).

  rf_appgate.py --tag APP1P-L18-s0 [--base RSG-L18-s0] [--v2 <rf_v2gate json>] [--B 10000] [--rows train --n 2000]  (train: diagnostic
      read on the pool's own rows, written as <tag>_trainrows)
      -> experiments/route_follow/results/approach/gate/<tag>.{json,md} (box repo) and $DATA_DIR/runs/route_follow/approach/gate/

Reads (on_approach.terms on each model's own plan, no dead zone): A-lat = mean |lateral offset to the human path| of the plan at the entry
line (end pose when the plan does not reach it) over the dev turn rows; A-speed = mean over the dev turn rows with the speed term (l3) and a
plan segment in the 26 m window of the mean speed excess over the human (m/s, 0 when not faster); also the heading error at the entry and
the negatives' end-pose |lateral| to the centreline (no drift: not above the base + 0.10 m). Drop = 1 - pilot / base, log-clustered
bootstrap CI (the same draw of logs for both models). Lines: A-lat drop >= 0.30 and A-speed drop >= 0.30; G-real dev ADE <= base + 0.01;
G-ST shift and straight-continuation lateral moves <= 0.10 m; G-over share <= the base's; G-rely (memory reliance on junction rows) >= 0.01.

APPROACH COURSE v2 (plans/2026-10-15-approach-v2-prereg.md): --guards-only --real-tol 0.015 --res-sub v2/gate: the gate is G-real (dev ADE <=
base + 0.015 m), G-ST, G-over, G-rely only; A-lat / A-speed / A-neg are reported. Also reported (on_approach's v2 reads, every dev row):
neg_dv = the plan's signed speed minus the human's on the straight negatives (26 m window of the virtual entry, human >= 2 m/s; a learnt
slow-down shows as a drop), over = the share of dev turn rows whose 2 s heading points > 10 deg further into the turn than the human path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/route_mem/lib"),
                str(REPO / "experiments/route_mem/scripts"), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/onpol_nuplan/lib")]
from jevdrive.tokens import open_front  # noqa: E402
SRC = (("nup", "L18RSG"), ("pai", "PAIonp"))
LINE_DROP, LINE_NEG = 0.30, 0.10


def main(a):
    import torch
    import on_approach as OA
    import pp_train as T
    import rf_gates as G
    import rmem as RM
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive import op_adapt as A
    from jevdrive.common import data_dir
    D = data_dir()
    CR, FEAT, AP = D / "runs/op_parity/cache", D / "runs/route_mem/feat", D / "runs/route_follow/approach"
    dev = torch.device("cuda")
    dz = np.load(AP / "dev.npz")
    rows = {}
    for s, pool in SRC:
        L = np.load(AP / f"lab_{s}.npz")
        if a.rows == "train":                                         # diagnostic: the pool's own rows (fit, not held out), <= n per source
            tr = np.flatnonzero(np.isin(L["cls"], ["turn", "neg"]) & ~L["dev"])
            r = np.sort(np.random.default_rng(0).choice(tr, min(a.n, len(tr)), replace=False))
        else:
            r = np.sort(dz[f"{s}_rows"])
        tb, fz = np.load(CR / f"on1_{pool}" / "tab.npz"), np.load(FEAT / f"on1_{pool}.npz")
        assert np.array_equal(tb["names"], L["names"]) and np.array_equal(fz["names"], L["names"])
        f = fz["f"][r].astype(np.float32)
        f[~fz["ok"][r]] = 0
        rows[s] = dict(pool=pool, r=r, ego=np.c_[tb["ego"][r], f].astype(np.float32), lht=tb["lht"][r], cam=tb["cam"][r, 0],
                       **{k: L[k][r] for k in ("cls", "H", "Hh", "Hv", "d_e", "l3", "log") + (("Hvh", "side") if a.guards_only else ())})
    cat = lambda k: np.concatenate([rows[s][k] for s, _ in SRC])  # noqa: E731
    cls, logs = cat("cls"), np.array([f"{s}:{x}" for s, _ in SRC for x in rows[s]["log"]])
    src = np.concatenate([np.full(len(rows[s]["r"]), s) for s, _ in SRC])
    pk = {k: torch.as_tensor(cat(v), device=dev) for k, v in (("H", "H"), ("Hh", "Hh"), ("Hv", "Hv"), ("d_e", "d_e"), ("l3", "l3"))}
    pk["ok"] = torch.ones(len(cls), dtype=torch.bool, device=dev)
    pv = None
    if a.guards_only:                                                 # v2 reads (on_approach's v2 branch; reads only, no target needed)
        Hvh, de_ = cat("Hvh"), cat("d_e")
        w = (np.arange(-10.0, 81.0)[None] >= de_[:, None] - 26.0) & (np.arange(-10.0, 81.0)[None] <= de_[:, None])
        l3n = (cls == "neg") & np.where(w, np.nan_to_num(Hvh, nan=-1.0) >= 2.0, True).all(1)
        pv = pk | {k: torch.as_tensor(v, device=dev) for k, v in dict(hand=np.ones(len(cls), bool), side=np.where(cls == "turn", cat("side"), 0).astype(np.int8),
                                                                        tgt=np.zeros(len(cls), np.float32), l3n=l3n, Hvh=Hvh, kind=(cls == "turn").astype(np.int8)).items()}
    W = torch.as_tensor(R2.t_weights(T.T8), device=dev, dtype=torch.float32)
    reads = {}
    for tg in (a.base, a.tag):
        m = RM.load_pmodel(tg, dev)
        assert not RM.head_order(tg), f"{tg}: the gate feeds the old memory layout"
        pi = torch.as_tensor(A.plan_index(m.net.slices), device=dev)
        P = []
        for s, _ in SRC:
            x = rows[s]
            fr = open_front(CR / f"on1_{x['pool']}@warp" / "front.npy")
            tc = torch.as_tensor(np.where(np.asarray(x["lht"], bool)[:, None], [[0.0, 1.0]], [[1.0, 0.0]]), dtype=torch.float32, device=dev)
            P.append(G.plans(m, fr, x["r"], x["ego"], tc, torch.as_tensor(x["cam"], dtype=torch.float32, device=dev), pi, W, dev))
        P = np.concatenate(P)
        Tm = OA.terms(torch.as_tensor(P, dtype=torch.float32, device=dev), torch.arange(len(cls), device=dev), pk)
        reads[tg] = {k: v.detach().cpu().numpy() for k, v in Tm.items()} | dict(plans=P)
        if pv is not None:
            T2 = OA.terms(torch.as_tensor(P, dtype=torch.float32, device=dev), torch.arange(len(cls), device=dev), pv)
            reads[tg] |= dict(dvn=T2["dvn"].cpu().numpy(), n_rows=T2["n_rows"].cpu().numpy(), over=T2["over"].cpu().numpy(),
                              over10=(T2["over"] > np.radians(10.0)).float().cpu().numpy())
        del m
        torch.cuda.empty_cache()
    turn, neg = cls == "turn", cls == "neg"
    vrow = turn & cat("l3") & reads[a.base]["v_rows"] & reads[a.tag]["v_rows"]
    ul = np.unique(logs)
    li = np.searchsorted(ul, logs)
    rng = np.random.default_rng(0)
    C = np.stack([np.bincount(rng.integers(0, len(ul), len(ul)), minlength=len(ul)) for _ in range(a.B)])     # (B, logs) draw counts

    def mean_ci(v, m):
        s, n = np.bincount(li[m], v[m], len(ul)), np.bincount(li[m], minlength=len(ul))
        b = (C * s).sum(1) / np.maximum((C * n).sum(1), 1)
        return float(v[m].mean()) if m.any() else np.nan, b

    def drop(key, m):
        vb, bb = mean_ci(reads[a.base][key], m)
        vp, bp = mean_ci(reads[a.tag][key], m)
        d = 1 - bp / np.where(bb > 0, bb, np.nan)
        return dict(base=vb, pilot=vp, drop=float(1 - vp / vb) if vb > 0 else np.nan, lo=float(np.nanpercentile(d, 2.5)), hi=float(np.nanpercentile(d, 97.5)),
                    rows=int(m.sum()), logs=int(len(np.unique(logs[m]))))
    res = dict(tag=a.tag, base=a.base, rows=dict(turn=int(turn.sum()), neg=int(neg.sum()), speed=int(vrow.sum()),
                                                 by_src={s: dict(turn=int((turn & (src == s)).sum()), neg=int((neg & (src == s)).sum())) for s, _ in SRC}),
               reads=dict(lat=drop("lat_e", turn), speed=drop("vex", vrow), head_deg=drop("head_e", turn & reads[a.base]["lat_cov"]),
                          lat_nup=drop("lat_e", turn & (src == "nup")), lat_pai=drop("lat_e", turn & (src == "pai")),
                          speed_nup=drop("vex", vrow & (src == "nup")), speed_pai=drop("vex", vrow & (src == "pai")),
                          loss1=drop("app1", turn), loss2=drop("app2", turn), loss3=drop("app3", vrow), neg_lat=drop("lat_e", neg)))
    if pv is not None:
        nrow = neg & reads[a.base]["n_rows"] & reads[a.tag]["n_rows"]
        for k, key, m in (("neg_dv", "dvn", nrow), ("over10_share", "over10", turn)):
            vb, _ = mean_ci(reads[a.base][key], m)
            vp, _ = mean_ci(reads[a.tag][key], m)
            dd_ = reads[a.tag][key] - reads[a.base][key]
            _, bd = mean_ci(dd_, m)
            res["reads"][k] = dict(base=vb, pilot=vp, drop=float(vp - vb), lo=float(np.nanpercentile(bd, 2.5)), hi=float(np.nanpercentile(bd, 97.5)),
                                   rows=int(m.sum()), logs=int(len(np.unique(logs[m]))), diff=True)
    for k in ("head_deg",):
        for q in ("base", "pilot"):
            res["reads"][k][q] = float(np.degrees(res["reads"][k][q]))
    nb, npl = res["reads"]["neg_lat"]["base"], res["reads"]["neg_lat"]["pilot"]
    gates = {"A-lat": dict(value=res["reads"]["lat"]["drop"], line=f">= {LINE_DROP}", pass_=bool(res["reads"]["lat"]["drop"] >= LINE_DROP)),
             "A-speed": dict(value=res["reads"]["speed"]["drop"], line=f">= {LINE_DROP}", pass_=bool(res["reads"]["speed"]["drop"] >= LINE_DROP)),
             "A-neg": dict(value=npl - nb, line=f"<= +{LINE_NEG} m", pass_=bool(npl - nb <= LINE_NEG))}
    v2 = Path(a.v2) if a.v2 else D / "runs/route_follow/v2gate" / f"{a.tag}.json"
    if v2.exists():
        g = json.loads(v2.read_text())["gates"]
        for k in ("G-real", "G-ST", "G-over", "G-rely"):
            gates[k] = dict(value=g[k]["value"], base=g[k].get("base"), line=g[k]["line"], pass_=bool(g[k]["pass_"]))
    else:
        gates["G-v2"] = dict(value=None, line=f"{v2} missing", pass_=False)
    if a.real_tol is not None and "G-real" in gates:                 # v2: the G-real line relaxed (prereg), from the same value / base
        g = gates["G-real"]
        g["line"], g["pass_"] = f"<= {g['base'] + a.real_tol:.4f} (base + {a.real_tol})", bool(g["value"] <= g["base"] + a.real_tol)
    if a.guards_only:
        for k in ("A-lat", "A-speed", "A-neg"):
            gates[k]["line"] += " (reported)"
    res["gates"] = gates
    res["all_pass"] = all(x["pass_"] for k, x in gates.items() if not (a.guards_only and k.startswith("A-")))
    out = [REPO / "experiments/route_follow/results/approach" / a.res_sub, D / "runs/route_follow/approach" / a.res_sub]
    if a.rows == "train":
        a.tag += "_trainrows"
    f3 = lambda x: ", ".join(f"{y:.3f}" for y in x) if isinstance(x, list) else ("-" if x is None else f"{x:.4f}")  # noqa: E731
    md = [f"# Approach gate: {a.tag} vs {a.base}", "", f"dev rows: {res['rows']['turn']} turn ({json.dumps(res['rows']['by_src'])}), {res['rows']['speed']} with the speed "
          f"read, {res['rows']['neg']} straight negatives; held-out logs, never in the pool.", "",
          "| read | base | pilot | drop [95 % CI] | rows / logs |", "|:--|--:|--:|:--|:--|"]
    for k, v in res["reads"].items():
        md.append(f"| {k} | {v['base']:.3f} | {v['pilot']:.3f} | {v['drop']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]{' (difference)' if v.get('diff') else ''} | "
                  f"{v['rows']} / {v['logs']} |")
    md += ["", "| gate | value | line | pass |", "|:--|--:|:--|:--|"]
    md += [f"| {k} | {f3(g['value'])} | {g['line']} | {'PASS' if g['pass_'] else 'FAIL'}{' (not gating)' if a.guards_only and k.startswith('A-') else ''} |"
           for k, g in gates.items()]
    md += ["", f"all pass: {res['all_pass']}"]
    for o in out:
        o.mkdir(parents=True, exist_ok=True)
        (o / f"{a.tag}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        (o / f"{a.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True), ap.add_argument("--base", default="RSG-L18-s0"), ap.add_argument("--v2", default="")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--rows", default="dev", choices=("dev", "train"), help="train: a diagnostic read on the pool's own rows (no gate)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--guards-only", action="store_true", help="v2: only G-real / G-ST / G-over / G-rely gate; the A-* reads are reported")
    ap.add_argument("--real-tol", type=float, default=None, help="v2: G-real line base + this (rf_v2gate's own line is base + 0.01)")
    ap.add_argument("--res-sub", default="gate", help="output subdir under results/approach and runs/route_follow/approach")
    main(ap.parse_args())
