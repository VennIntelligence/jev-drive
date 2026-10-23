#!/usr/bin/env python3
"""ROUND9 P1L helpers (plans/2026-10-24-p1l.md = the registration; lane scripts/p1l.sh). Box, repo root, repo venv. Reuses pcl.py.
  p1l.py lrcheck --tag T --on-start T0 [--full] --out J   every logged lr/g<i> vs the formula: before T0 P1's schedule (the run's own config),
                                       from T0 NTV0-L18-s0's per-group schedule (its stored meta) over the remaining steps; --full: the
                                       remaining steps must equal NTV0's 9 500
  p1l.py bs --out DIR                  behavior_stats (the box copy, as for NTV0 / P1) on the P1L read
  p1l.py verdict --dir O --out P       registered lines, paired CIs, class table, gap shares, gradient ratio, highway, cost -> P.{json,md}
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl as C  # noqa: E402

DD = C.DD
TAG = "PCL1L-F-s0"
P1, P0, NT, RB = "PCL1-F-s0", "PCL0-L18-s0", "NTV0-L18-s0", "RB0-F-s0"
LABS = [TAG, P1, P0, NT, RB]
G_HI, G_LO = 0.88, 0.75                                       # registered lines (plan section 5)
SAMPLE = [10025, 10100, 10500, 15000, 19500]
PP = DD / "runs/round9/pool_pilot"


def meta(tag):
    return json.load(open(sorted(glob.glob(f"{DD}/runs/route_mem/train-{tag}/*/meta.json"))[-1]))["config"]


def sched(base, wu, k, n, floor):
    """rm_train's lr of one group at 0-based step k of an n-step schedule (pp_lr off: np.pi * k / n)."""
    c = 0.5 * (1 + np.cos(np.pi * k / n))
    if floor:
        c = floor + (1.0 - floor) * c
    return base * min(1.0, (k + 1) / wu) * c


def cmd_lrcheck(a):
    d = C.run_dirs(a.tag)[-1]
    c, n = meta(a.tag), meta(NT)
    o, no = c["onpol"], n["onpol"]
    T, steps = a.on_start, c["steps"]
    assert o["lr_scale"] == 1.0 and not o["lr_floor"] and o["lr2_at"] == T, o
    flags = dict(scale=(o["lr2_scale"], no["lr_scale"]), warmup=(o["lr2_warmup"], no["warmup"]), floor=(o["lr2_floor"], no["lr_floor"]))
    pre = [(c["lr"], c["warmup"]), (c["lr_new"], c["warmup"]), (o["rm_lr"], o["rm_warmup"])]
    post = [(n["lr"] * no["lr_scale"], no["warmup"]), (n["lr_new"] * no["lr_scale"], no["warmup"]), (no["rm_lr"], no["rm_warmup"])]
    lr = {}
    for e in C.events(d):
        if e.get("kind") == "scalar" and e["tag"].startswith("lr/g"):
            lr.setdefault(e["step"], {})[int(e["tag"][4:])] = float(e["value"])
    rows, worst = {}, 0.0
    for s in sorted(lr):
        k = s - 1                                             # logged at s = the lr of 0-based step s - 1
        want = [sched(b, w, k, steps, 0.0) for b, w in pre] if k < T else \
               [sched(b, w, k - T, steps - T, no["lr_floor"]) for b, w in post]
        got = [lr[s].get(i) for i in range(3)]
        rel = max(abs(g - w) / w for g, w in zip(got, want)) if None not in got and len(lr[s]) == 3 else float("inf")
        worst = max(worst, rel)
        rows[s] = dict(got=got, want=want, rel=rel)
    keep = sorted({s for s in rows if s <= 100 or abs(s - T) <= 50} | {s for s in SAMPLE if s in rows} | {max(rows)} if rows else set())
    ok = bool(rows) and worst <= 1e-9 and all(a_ == b_ for a_, b_ in flags.values()) and (not a.full or steps - T == n["steps"])
    rec = dict(tag=a.tag, run=str(d), on_start=T, steps=steps, horizon_after_T=steps - T, ntv0_steps=n["steps"], flags_vs_ntv0=flags,
               groups_pre=pre, groups_post_ntv0=post, logged_points=len(rows), max_rel_err=worst, points={s: rows[s] for s in keep}, pass_=ok)
    C.dump(a.out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "points"}, indent=1, default=float))
    sys.exit(0 if ok else 2)


def cmd_bs(a):
    C.TAGS = {"P1L": TAG}
    C.cmd_bs(a)


def cmd_verdict(a):
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(C.REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    OLD = {t: pd.read_csv(DD / "runs/pai_fix/facts" / f"{t}-281ag.csv").set_index("scene") for t in LABS}
    NEW = C.new_scores(PP)                                    # NTV0 / RB0-F (scorer_v2 report), P1 / P0 (pool_pilot sv2)
    x = pd.read_csv(o / "sv2/new/rollouts.csv")
    NEW[TAG] = x[x.read == f"{TAG}-ag"].drop_duplicates("scene", keep="last").set_index("scene").score
    rpc = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    common = sorted(set.intersection(*[set(OLD[t].index[OLD[t].score.notna()]) for t in LABS])
                    & set.intersection(*[set(NEW[t].dropna().index) for t in LABS]) - rpc)
    n = len(common)
    S = dict(old={t: OLD[t].score.reindex(common).to_numpy(float) for t in LABS}, new={t: NEW[t].reindex(common).to_numpy(float) for t in LABS})
    res = dict(scenes=n, reads={}, pairs={}, lines={})
    for t in LABS:
        res["reads"][t] = dict(old=PF.metrics(OLD[t].reindex(common)), new_mean=float(S["new"][t].mean()), new_zeros=int((S["new"][t] == 0).sum()),
                               dev_ade=C.dev_ade(t))
    for y in (P1, P0, NT, RB):
        k = f"{TAG}:{y}"
        res["pairs"][k] = {sc: stats.paired(S[sc][TAG], S[sc][y]) for sc in ("new", "old")}
        cl = {}
        for c in PF.ZCLS + PF.PCLS + ["zeros"]:
            u, v = ((S["old"][q] == 0) if c == "zeros" else (OLD[q].reindex(common).fclass == c).to_numpy() for q in (TAG, y))
            q_ = stats.paired(u.astype(float), v.astype(float))
            cl[c] = dict(a=int(u.sum()), b=int(v.sum()), diff=n * q_["mean"], lo=n * q_["lo"], hi=n * q_["hi"])
        res["pairs"][k]["classes"] = cl
    for sc in ("new", "old"):
        b, nt = S[sc][RB].mean(), S[sc][NT].mean()
        g = {t: float((S[sc][t].mean() - b) / (nt - b)) for t in (TAG, P1, P0)}
        d1 = res["pairs"][f"{TAG}:{P1}"][sc]["mean"]
        line = ("lr at S entry was the cause; use it in NTV1" if g[TAG] >= G_HI and d1 > 0 else
                "lr is not the cause" if g[TAG] <= G_LO else "undecided")
        res["lines"][sc] = dict(rb0f=float(b), ntv0=float(nt), line_hi=float(b + G_HI * (nt - b)), line_lo=float(b + G_LO * (nt - b)), g=g,
                                p1l_minus_p1=d1, line=line, registered=sc == "new")
    C.TAGS = {"P1L": TAG}
    hw, per = C.highway(o, NEW)
    hp = json.loads((PP / "results/verdict.json").read_text())["highway"]
    res["highway"] = hw + [r for r in hp if r["read"] == P1]
    rat = json.loads((o / "results/ratio_P1L.json").read_text())
    full = C.grad_ratio(C.run_dirs(TAG)[-1], 10000, 19500)
    p1r = json.loads((PP / "results/ratio_P1.json").read_text())
    p1f = C.grad_ratio(C.run_dirs(P1)[-1], 10000, 19500)
    res["ratio"] = dict(P1L_500=rat["ratio"], P1L_pass=rat["pass_"], P1L_after_T=full["ratio"], P1_500=p1r["ratio"], P1_after_T=p1f["ratio"],
                        band=C.RATIO_BAND)
    res["post"] = json.loads((o / "results/post_P1L.json").read_text())
    lc = json.loads((o / "results/lr_full.json").read_text())
    res["lr"] = dict(max_rel_err=lc["max_rel_err"], points=lc["logged_points"], sample={s: lc["points"].get(str(s)) for s in SAMPLE})
    res["ade"] = {f"{t} @ {s}": C.dev_ade(t, s) for t in (TAG, P1) for s in (10000, 19500)} | {t: C.dev_ade(t) for t in (P0, NT, RB)}
    res["cost"] = C.cost(o)
    f3 = C.f3
    L = ["# P1L verdict (PAI 281, ACC + GAP, Harmonizer OFF)", "",
         f"{n} scenes scored by all five reads (both scorers) with no driver RPC error in any of them.", "",
         "## Reads", "", "| read | old mean | new mean | zeros old / new | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in res["reads"].items():
        m = r["old"]
        L.append(f"| {t} | {m['mean']:.3f} | {r['new_mean']:.3f} | {m['zeros']} / {r['new_zeros']} | {m['z_offroad']} | {m['z_zero-other']} + {m['z_side']} | "
                 f"{m['z_route-turn']} | {m['z_overshoot']} | {m['z_AF-front']} | {m['p_rear-hit']} | {m['p_slow']} | {m['launch']} | "
                 + (f"{r['dev_ade']:.4f} |" if r["dev_ade"] is not None else "- |"))
    L += ["", "## Paired differences (95 % bootstrap CI, scenes as units)", "", "| pair | new scorer | old scorer | zeros (old) |", "|:--|--:|--:|--:|"]
    for k, p in res["pairs"].items():
        z = p["classes"]["zeros"]
        L.append(f"| {k.replace(':', ' - ')} | {f3(p['new'])} | {f3(p['old'])} | {z['diff']:+.0f} [{z['lo']:+.0f}, {z['hi']:+.0f}] |")
    cls = PF.ZCLS + PF.PCLS
    L += ["", "Zero / partial classes, P1L minus each read (count difference, 95 % CI; old scorer classes):", "",
          "| P1L - ref | " + " | ".join(cls) + " |", "|:--|" + "--:|" * len(cls)]
    for y in (P1, P0, NT, RB):
        c = res["pairs"][f"{TAG}:{y}"]["classes"]
        L.append(f"| - {y} | " + " | ".join(f"{c[q]['diff']:+.0f} [{c[q]['lo']:+.0f}, {c[q]['hi']:+.0f}]" for q in cls) + " |")
    L += ["", "## Registered lines (g = (arm - RB0-F) / (NTV0-s0 - RB0-F); >= 88 % and P1L - P1 > 0 / <= 75 %; new scorer registered, old reported)", ""]
    for sc, q in res["lines"].items():
        L.append(f"- {sc} scorer: RB0-F {q['rb0f']:.3f}, NTV0-s0 {q['ntv0']:.3f} -> lines {q['line_hi']:.3f} / {q['line_lo']:.3f}; g P1L **{q['g'][TAG]:.0%}**, "
                 f"P1 {q['g'][P1]:.0%}, P0 {q['g'][P0]:.0%}; P1L - P1 {q['p1l_minus_p1']:+.3f} -> **{q['line']}**" + ("" if q["registered"] else " (reported only)"))
    L += ["", "## Highway (behavior_stats Q1 rules)", "", "| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | line |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in res["highway"]:
        L.append(f"| {r['read']} | {r['steps']} | {r['median']:.2f} | {r['p10']:.2f} | {r['lt85']:.0%} | {r['gt110']:.0%} | {r['dv']:+.2f} | {r['slow_rear_pts']:.2f} | "
                 f"{'highway slowness present' if r['slow'] else 'not slow on highway'} |")
    bad = per[TAG][per[TAG].med < 0.90] if TAG in per else []
    L.append(f"\nP1L highway-majority scenes with a free-road median < 0.90: {len(bad)}" + (": " + ", ".join(f"{s[7:15]} {r.med:.2f}" for s, r in bad.iterrows()) if len(bad) else ""))
    r = res["ratio"]
    L += ["", "## Gates and schedule", "",
          f"- S / imit gradient ratio, 500 steps after T: P1L {r['P1L_500']:.3f} (band {r['band'][0]:.2f}-{r['band'][1]:.2f}, {'pass' if r['P1L_pass'] else 'FAIL'}), "
          f"P1 {r['P1_500']:.3f}; whole post-T segment: P1L {r['P1L_after_T']:.3f}, P1 {r['P1_after_T']:.3f}",
          f"- post: final dev ADE {res['post']['dev_ade']:.4f}; pre-T segment vs {P1}: {res['post']['pre_T_vs_ref']['identical']} / "
          f"{res['post']['pre_T_vs_ref']['scalars_compared']} scalars bit-identical (steps <= 10 000)",
          f"- lr: {res['lr']['points']} logged points, max relative error vs the formula {res['lr']['max_rel_err']:.1e}", "",
          "| step | g0 net | g1 adapter | g2 route branch |", "|--:|--:|--:|--:|"]
    for s, v in res["lr"]["sample"].items():
        if v:
            L.append(f"| {s} | " + " | ".join(f"{x:.3e}" for x in v["got"]) + " |")
    L += ["", "| navtrain dev ADE | m |", "|:--|--:|"] + [f"| {k} | {v:.4f} |" if v is not None else f"| {k} | - |" for k, v in res["ade"].items()]
    c = res["cost"]
    L += ["", "## Cost", "", f"GPU jobs {c['gpu_job_h']:.1f} job-hours (training {c['train_h']:.2f}, reads {c['read_h']:.2f}, gates / smokes {c['gate_h']:.2f}); "
          f"wall {c['wall_h']:.1f} h from the first start to the last end."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", res)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("lrcheck")
    p.add_argument("--tag", required=True), p.add_argument("--on-start", type=int, required=True), p.add_argument("--full", action="store_true")
    p.add_argument("--out", required=True)
    p = s.add_parser("bs")
    p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=16)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(lrcheck=cmd_lrcheck, bs=cmd_bs, verdict=cmd_verdict)[a.sub](a)


if __name__ == "__main__":
    main()
