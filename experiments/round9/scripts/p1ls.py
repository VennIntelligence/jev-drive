#!/usr/bin/env python3
"""ROUND9 P1L-SPLIT helpers (plans/2026-10-24-p1l-split.md = the registration; lane scripts/p1ls.sh). Box, repo root, repo venv. Reuses pcl.py
(argv base, ident, tcheck, smoke, ratio, post, cost) and p1l.py (meta, sched); edits neither.
  p1ls.py argv ARM TAG [--steps N] [--on-start T] [--save-at S|none]   ARM: P1 / P1L (pcl.py's argv) or P1LA / P1LB = P1L's argv
                                       + --lr2-groups net,adapter (A) / route (B)
  p1ls.py lrcheck --tag T --on-start T0 [--full] --out J   every logged lr/g<i> vs the formula: groups named by the run's --lr2-groups
                                       ('' = all) follow NTV0-L18-s0's per-group schedule from T0, the others P1's own schedule over the whole run
  p1ls.py verdict --dir O --out P      registered lines, paired CIs, class table vs P1 / P1L, gradient ratio, lr, ADE, cost -> P.{json,md}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1l as L  # noqa: E402
import pcl as C  # noqa: E402

DD = C.DD
ARMS = dict(P1LA=("PCL1LA-F-s0", "net,adapter"), P1LB=("PCL1LB-F-s0", "route"))
P1L, P1, NT, RB = "PCL1L-F-s0", "PCL1-F-s0", "NTV0-L18-s0", "RB0-F-s0"
GN = ["net", "adapter", "route"]
NEAR = 0.01                                                   # registered: within 0.01 of P1L / of P1 (plan section 5)
SAMPLE = [10025, 10100, 10500, 15000, 19500]


def build(arm, tag, steps=None, on_start=None, save_at=None):
    if arm in ARMS:
        return C.build_argv("P1L", tag, steps, on_start, save_at) + ["--lr2-groups", ARMS[arm][1]]
    return C.build_argv(arm, tag, steps, on_start, save_at)


def cmd_argv(a):
    print("\n".join(build(a.arm, a.tag, a.steps, a.on_start, a.save_at)))


def cmd_lrcheck(a):
    d = C.run_dirs(a.tag)[-1]
    c, n = L.meta(a.tag), L.meta(NT)
    o, no = c["onpol"], n["onpol"]
    T, steps = a.on_start, c["steps"]
    assert o["lr_scale"] == 1.0 and not o["lr_floor"] and o["lr2_at"] == T, o
    sel = set((o.get("lr2_groups") or ",".join(GN)).split(","))
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
        want = [L.sched(*post[i], k - T, steps - T, no["lr_floor"]) if k >= T and GN[i] in sel else L.sched(*pre[i], k, steps, 0.0)
                for i in range(3)]
        got = [lr[s].get(i) for i in range(3)]
        rel = max(abs(g - w) / w for g, w in zip(got, want)) if None not in got and len(lr[s]) == 3 else float("inf")
        worst = max(worst, rel)
        rows[s] = dict(got=got, want=want, rel=rel)
    keep = sorted({s for s in rows if s <= 100 or abs(s - T) <= 50} | {s for s in SAMPLE if s in rows} | ({max(rows)} if rows else set()))
    ok = bool(rows) and worst <= 1e-9 and all(x == y for x, y in flags.values()) and (not a.full or steps - T == n["steps"])
    rec = dict(tag=a.tag, run=str(d), on_start=T, steps=steps, lr2_groups=sorted(sel), horizon_after_T=steps - T, ntv0_steps=n["steps"],
               flags_vs_ntv0=flags, groups_pre=pre, groups_post_ntv0=post, logged_points=len(rows), max_rel_err=worst,
               points={s: rows[s] for s in keep}, pass_=ok)
    C.dump(a.out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "points"}, indent=1, default=float))
    sys.exit(0 if ok else 2)


def line_of(x, p1l, p1):
    a_, b_ = abs(x - p1l) <= NEAR, abs(x - p1) <= NEAR
    if a_ and b_:
        return "both windows (P1L and P1 within 0.02 on this scorer)"
    return "carries P1L's gain" if a_ else "alone does nothing" if b_ else "partial"


def cmd_verdict(a):
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(C.REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    tags = [t for t, _ in ARMS.values()]
    labs = tags + [P1L, P1, NT, RB]
    OLD = {t: pd.read_csv(DD / "runs/pai_fix/facts" / f"{t}-281ag.csv").set_index("scene") for t in labs}
    NEW = C.new_scores(C.O_)                                  # NTV0 / RB0-F (scorer_v2 report), P1 (pool_pilot sv2)
    for t, f in [(P1L, DD / "runs/round9/p1l/sv2/new/rollouts.csv")] + [(t, o / "sv2/new/rollouts.csv") for t in tags]:
        x = pd.read_csv(f)
        NEW[t] = x[x.read == f"{t}-ag"].drop_duplicates("scene", keep="last").set_index("scene").score
    rpc = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    common = sorted(set.intersection(*[set(OLD[t].index[OLD[t].score.notna()]) for t in labs])
                    & set.intersection(*[set(NEW[t].dropna().index) for t in labs]) - rpc)
    n = len(common)
    S = dict(new={t: NEW[t].reindex(common).to_numpy(float) for t in labs}, old={t: OLD[t].score.reindex(common).to_numpy(float) for t in labs})
    res = dict(scenes=n, reads={}, pairs={}, lines={})
    for t in labs:
        res["reads"][t] = dict(old=PF.metrics(OLD[t].reindex(common)), new_mean=float(S["new"][t].mean()), new_zeros=int((S["new"][t] == 0).sum()),
                               dev_ade=C.dev_ade(t))
    cls = PF.ZCLS + PF.PCLS
    for x, y in [(t, r) for t in tags for r in (P1L, P1, NT, RB)] + [(tags[0], tags[1]), (P1L, P1)]:
        k = f"{x}:{y}"
        res["pairs"][k] = {sc: stats.paired(S[sc][x], S[sc][y]) for sc in ("new", "old")}
        cl = {}
        for c in cls + ["zeros"]:
            u, v = ((S["old"][q] == 0) if c == "zeros" else (OLD[q].reindex(common).fclass == c).to_numpy() for q in (x, y))
            q_ = stats.paired(u.astype(float), v.astype(float))
            cl[c] = dict(a=int(u.sum()), b=int(v.sum()), diff=n * q_["mean"], lo=n * q_["lo"], hi=n * q_["hi"])
        res["pairs"][k]["classes"] = cl
    for sc in ("new", "old"):
        m = {t: float(S[sc][t].mean()) for t in labs}
        b, nt = m[RB], m[NT]
        res["lines"][sc] = dict(p1l=m[P1L], p1=m[P1], rb0f=b, ntv0=nt, registered=sc == "new",
                                arms={arm: dict(tag=t, mean=m[t], minus_p1l=m[t] - m[P1L], minus_p1=m[t] - m[P1],
                                                frac_of_p1l_gain=(m[t] - m[P1]) / (m[P1L] - m[P1]) if m[P1L] != m[P1] else None,
                                                g=(m[t] - b) / (nt - b), line=line_of(m[t], m[P1L], m[P1])) for arm, (t, _) in ARMS.items()})
    res["ratio"], res["post"], res["lr"] = {}, {}, {}
    for arm, (t, _) in ARMS.items():
        r = json.loads((o / f"results/ratio_{arm}.json").read_text())
        res["ratio"][arm] = dict(r500=r["ratio"], pass_=r["pass_"], after_T=C.grad_ratio(C.run_dirs(t)[-1], 10000, 19500)["ratio"])
        res["post"][arm] = json.loads((o / f"results/post_{arm}.json").read_text())
        lc = json.loads((o / f"results/lr_full_{arm}.json").read_text())
        res["lr"][arm] = dict(max_rel_err=lc["max_rel_err"], points=lc["logged_points"], sample={s: lc["points"].get(str(s)) for s in SAMPLE})
    pr = json.loads((DD / "runs/round9/p1l/results/verdict.json").read_text())["ratio"]
    res["ratio"]["P1L"] = dict(r500=pr["P1L_500"], after_T=pr["P1L_after_T"])
    res["ratio"]["P1"] = dict(r500=pr["P1_500"], after_T=pr["P1_after_T"])
    res["ade"] = {f"{t} @ {s}": C.dev_ade(t, s) for t in tags + [P1L, P1] for s in (10000, 19500)}
    res["cost"] = C.cost(o)
    f3 = C.f3
    nm = {ARMS["P1LA"][0]: "P1La", ARMS["P1LB"][0]: "P1Lb", P1L: "P1L", P1: "P1", NT: "NTV0-s0", RB: "RB0-F"}
    Lm = ["# P1L-SPLIT verdict (PAI 281, ACC + GAP, Harmonizer OFF)", "",
          f"{n} scenes scored by all six reads (both scorers) with no driver RPC error in any of them.", "",
          "## Reads", "", "| read | new mean | old mean | zeros new / old | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in res["reads"].items():
        m = r["old"]
        Lm.append(f"| {nm[t]} ({t}) | {r['new_mean']:.3f} | {m['mean']:.3f} | {r['new_zeros']} / {m['zeros']} | {m['z_offroad']} | {m['z_zero-other']} + {m['z_side']} | "
                  f"{m['z_route-turn']} | {m['z_overshoot']} | {m['z_AF-front']} | {m['p_rear-hit']} | {m['p_slow']} | {m['launch']} | "
                  + (f"{r['dev_ade']:.4f} |" if r["dev_ade"] is not None else "- |"))
    Lm += ["", "## Registered lines (within 0.01 of P1L -> carries P1L's gain; within 0.01 of P1 -> alone does nothing; else partial)", ""]
    for sc, q in res["lines"].items():
        Lm.append(f"- {sc} scorer{'' if q['registered'] else ' (reported only)'}: P1L {q['p1l']:.3f}, P1 {q['p1']:.3f}")
        for arm, r in q["arms"].items():
            fr = f", {r['frac_of_p1l_gain']:.0%} of P1L - P1" if r["frac_of_p1l_gain"] is not None else ""
            Lm.append(f"  - {arm}: {r['mean']:.3f} (vs P1L {r['minus_p1l']:+.3f}, vs P1 {r['minus_p1']:+.3f}{fr}; g {r['g']:.0%}) -> **{r['line']}**")
    Lm += ["", "## Paired differences (95 % bootstrap CI, scenes as units)", "", "| pair | new scorer | old scorer | zeros (old) |", "|:--|--:|--:|--:|"]
    for k, p in res["pairs"].items():
        z = p["classes"]["zeros"]
        x, y = k.split(":")
        Lm.append(f"| {nm[x]} - {nm[y]} | {f3(p['new'])} | {f3(p['old'])} | {z['diff']:+.0f} [{z['lo']:+.0f}, {z['hi']:+.0f}] |")
    Lm += ["", "Zero / partial classes (old scorer classes; count difference, 95 % CI):", "",
           "| pair | " + " | ".join(cls) + " |", "|:--|" + "--:|" * len(cls)]
    for k, p in res["pairs"].items():
        x, y = k.split(":")
        if y in (P1L, P1):
            c = p["classes"]
            Lm.append(f"| {nm[x]} - {nm[y]} | " + " | ".join(f"{c[q]['diff']:+.0f} [{c[q]['lo']:+.0f}, {c[q]['hi']:+.0f}]" for q in cls) + " |")
    Lm += ["", "## Gates and schedule", ""]
    for arm in ARMS:
        r, p, lc = res["ratio"][arm], res["post"][arm], res["lr"][arm]
        Lm.append(f"- {arm}: S / imit ratio 500 steps after T {r['r500']:.3f} ({'pass' if r['pass_'] else 'FAIL'}, band {C.RATIO_BAND[0]:.2f}-{C.RATIO_BAND[1]:.2f}), "
                  f"whole post-T {r['after_T']:.3f}; final dev ADE {p['dev_ade']:.4f}; pre-T vs P1L {p['pre_T_vs_ref']['identical']} / "
                  f"{p['pre_T_vs_ref']['scalars_compared']} scalars bit-identical; lr {lc['points']} points, max rel. error {lc['max_rel_err']:.1e}")
    Lm.append(f"- reference ratios: P1L {res['ratio']['P1L']['r500']:.3f} / {res['ratio']['P1L']['after_T']:.3f}, "
              f"P1 {res['ratio']['P1']['r500']:.3f} / {res['ratio']['P1']['after_T']:.3f} (500 steps / whole post-T)")
    Lm += ["", "| step | P1La g0 / g1 / g2 | P1Lb g0 / g1 / g2 |", "|--:|:--|:--|"]
    for s in SAMPLE:
        v = [res["lr"][arm]["sample"].get(s) for arm in ARMS]
        Lm.append(f"| {s} | " + " | ".join(" / ".join(f"{x:.2e}" for x in w["got"]) if w else "-" for w in v) + " |")
    Lm += ["", "| navtrain dev ADE | m |", "|:--|--:|"] + [f"| {k} | {v:.4f} |" if v is not None else f"| {k} | - |" for k, v in res["ade"].items()]
    c = res["cost"]
    Lm += ["", "## Cost", "", f"GPU jobs {c['gpu_job_h']:.1f} job-hours (training {c['train_h']:.2f}, reads {c['read_h']:.2f}, gates / smokes {c['gate_h']:.2f}); "
           f"wall {c['wall_h']:.1f} h from the first start to the last end."]
    Path(f"{a.out}.md").write_text("\n".join(Lm) + "\n")
    C.dump(f"{a.out}.json", res)
    print("\n".join(Lm))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("argv")
    p.add_argument("arm", choices=("P1", "P1L", *ARMS)), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at")
    p = s.add_parser("lrcheck")
    p.add_argument("--tag", required=True), p.add_argument("--on-start", type=int, required=True), p.add_argument("--full", action="store_true")
    p.add_argument("--out", required=True)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(argv=cmd_argv, lrcheck=cmd_lrcheck, verdict=cmd_verdict)[a.sub](a)


if __name__ == "__main__":
    main()
