#!/usr/bin/env python3
"""torch.compile validation for rm_train (plans/2026-10-17-compile-prereg.md): the registered lines on stored reads (CPU). Chain: compile_val.sh.

  summary [--dry]   compiled RSGC-L18-s{0,1} vs eager RSG-L18-s{0,1} (same seed) against the eager seed spread (RSG-L18-s1 vs s0):
                    eager identity (RSGE-L18-s0 vs RSG-L18-s0, nx.py bitcmp), dev ADE, wait W / go G (wc), S_L (led F), desire 4 s (dc), lateral
                    2 s (wt.dev_read), held 460 @18 and PAI 281 (submission config) paired, wall time / card-hours, weight distances
                    -> $DATA_DIR/runs/cl_infra/compile/summary.{json,md}, experiments/cl_infra/results/compile_val.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/wait_cap/scripts"), str(REPO / "experiments/desire_cap/scripts")]
import wt  # noqa: E402
from jevdrive import stats  # noqa: E402

O = wt.DATA / "runs/cl_infra/compile"
RES = REPO / "experiments/cl_infra/results"
E, C, R = ("RSG-L18-s0", "RSG-L18-s1"), ("RSGC-L18-s0", "RSGC-L18-s1"), "RSGE-L18-s0"
CELLS = ("wait_W", "go_G", "S_L", "desire4")                   # registered ledger cells (+ lat2s, a pairwise read)


def ci0(p):
    return bool(p["lo"] <= 0 <= p["hi"])


def done(tag):
    d = sorted(glob.glob(f"{wt.DATA}/runs/route_mem/train-{tag}/*/"))[-1]
    D = json.load(open(d + "DONE"))
    sps = {e["step"]: e["value"] for e in map(json.loads, open(d + "events.jsonl"))
           if e["kind"] == "scalar" and e["tag"] == "throughput/steps_per_s"}
    el = {s: s / v for s, v in sps.items()}                           # elapsed loop time at each logged step
    n = max(el)
    m = dict(train_s=D["train_s"], wall_s=D["wall_s"], dev_ade=D["dev_ade"], on_hash_60=D.get("on_hash_60"), steps=n, loop_s=el[n])
    if 1500 in el:                                                    # steady state (dev evals at 2000 / 2500 in both arms)
        step_s = (el[n] - el[1500]) / (n - 1500)
        m |= dict(steady_ms=1e3 * step_s, warmup_s=el[25] - 25 * step_s)
    meta = json.load(open(d + "meta.json"))
    m |= dict(git=meta["git"]["sha"][:8], card=meta["env"].get("CUDA_VISIBLE_DEVICES"))
    return m


def cells(tag):
    """The registered open-loop cells of one tag: value, per-row vector (share cells) and logs; missing reads -> None."""
    out = {}
    try:
        c, x = wt.cap_read(tag)
        out["wait_W"] = dict(v=c["wait_W"], n=int(x["W"].sum()), rows=x["wait"][x["W"]], g=x["g"][x["W"]])
        out["go_G"] = dict(v=c["go_G"], n=int(x["G"].sum()), rows=1 - x["wait"][x["G"]], g=x["g"][x["G"]])
    except FileNotFoundError:
        pass
    try:
        s, g = wt.ledger_read(tag)["S_L"]
        out["S_L"] = dict(v=float(s.mean()), n=int(len(s)), rows=s, g=g)
    except FileNotFoundError:
        pass
    try:
        import dt
        d = dt.desire_read(tag)["arm"]
        out["desire4"] = dict(v=d["mean4"], n=d["rows"], lcl4=d["lcl4"], lcr4=d["lcr4"], pai_lca4=d["pai_lca4"])
    except FileNotFoundError:
        pass
    return out


def lat2s(a, b):
    try:
        return wt.dev_read(a, b)["lat2s_med_absdiff"]
    except FileNotFoundError:
        return None


def wdist(a, b):
    """Weight distance of two checkpoints: max |diff| and the L2 norm of the difference over every trainable tensor (net + parity)."""
    import torch
    ld = lambda t: torch.load(f"{wt.DATA}/runs/op_parity/runs/{t}/ckpt-final.pt", map_location="cpu", weights_only=False)["model"]  # noqa: E731
    A, B = ld(a), ld(b)
    mx, l2 = 0.0, 0.0
    for g in ("net", "parity"):
        for k in A[g] or {}:
            d = A[g][k].float() - B[g][k].float()
            mx, l2 = max(mx, float(d.abs().max())), l2 + float((d * d).sum())
    return dict(max_abs=mx, l2=l2 ** 0.5)


def cl_pair(MJ, MC):
    common = sorted(set(MJ) & set(MC))
    if not common:
        return None
    A, B = (np.array([M[s] for s in common]) for M in (MJ, MC))
    p = stats.paired(A[:, 0], B[:, 0], groups=np.array(["_".join(s.split("_")[:2]) for s in common]))
    r = wt.xpair(MJ, MC)
    r.pop("zero_flips", None)
    return r | dict(lo=float(p["lo"]), hi=float(p["hi"]), d=float(p["mean"]), ci0=ci0(p), dzeros=r["J"]["zeros"] - r["C"]["zeros"],
                    identical_scores=bool(np.array_equal(A[:, 0], B[:, 0])))


def pai(tag):
    return wt.metrics(glob.glob(f"{O}/pai/{tag}/runs/*/sim"))


def cmd_summary(a):
    res = dict(eager=list(E), compiled=list(C), rerun=R)
    f = wt.DATA / "runs/wait_cap/next/compile_eager_identity.json"
    res["identity"] = json.loads(f.read_text()) if f.exists() else None
    res["train"] = {t: done(t) for t in (R,) + E + C}
    tr = res["train"]
    res["train"]["saved"] = dict(per_training_s=[tr[R]["wall_s"] - tr[c]["wall_s"] for c in C],
                                 speedup_wall=[tr[R]["wall_s"] / tr[c]["wall_s"] for c in C],
                                 speedup_steady=[tr[R]["steady_ms"] / tr[c]["steady_ms"] for c in C] if "steady_ms" in tr[R] else None)
    res["weights"] = {f"{c} vs {e}": wdist(c, e) for c, e in zip(C, E)} | {"RSG-L18-s1 vs RSG-L18-s0 (seed)": wdist(E[1], E[0])}
    # 1 dev ADE
    ade = {t: tr[t]["dev_ade"] for t in E + C}
    sp = abs(ade[E[0]] - ade[E[1]])
    res["dev_ade"] = dict(values=ade, spread=sp, diff=[ade[c] - ade[e] for c, e in zip(C, E)])
    res["dev_ade"]["line"] = bool(all(abs(x) <= sp for x in res["dev_ade"]["diff"]))
    # 2 ledger cells
    CE = {t: cells(t) for t in E + C}
    res["ledger"], L2 = {}, []
    for k in CELLS:
        if not all(k in CE[t] for t in E + C):
            res["ledger"][k] = "missing"
            L2.append(None)
            continue
        v = {t: CE[t][k]["v"] for t in E + C}
        spr = abs(v[E[0]] - v[E[1]])
        floor = 1.0 / min(CE[t][k]["n"] for t in E + C) if "rows" in CE[E[0]][k] else 0.0
        spr_eff = max(spr, floor)
        dd = [v[c] - v[e] for c, e in zip(C, E)]
        r = dict(values=v, n={t: CE[t][k]["n"] for t in E + C}, spread=spr, spread_floor=floor, diff=dd, line=bool(all(abs(x) <= spr_eff for x in dd)))
        if "rows" in CE[E[0]][k]:
            r["paired"] = {f"{c} - {e}": stats.fmt(stats.paired(CE[c][k]["rows"], CE[e][k]["rows"], groups=CE[e][k]["g"])) for c, e in zip(C, E)}
            r["paired"]["seed s1 - s0"] = stats.fmt(stats.paired(CE[E[1]][k]["rows"], CE[E[0]][k]["rows"], groups=CE[E[0]][k]["g"]))
        if k == "desire4":
            r["lcl4_lcr4"] = {t: (CE[t][k]["lcl4"], CE[t][k]["lcr4"]) for t in E + C}
        res["ledger"][k] = r
        L2.append(r["line"])
    la = dict(compiled=[lat2s(c, e) for c, e in zip(C, E)], spread=lat2s(E[1], E[0]))
    la["line"] = None if None in la["compiled"] + [la["spread"]] else bool(all(x <= la["spread"] for x in la["compiled"]))
    res["ledger"]["lat2s"] = la
    L2.append(la["line"])
    res["ledger_line"] = None if None in L2 else bool(all(L2))
    # 3 held 460 @18
    H = {t: wt.metrics(glob.glob(wt.xdirs("held", t))) for t in E + C}
    res["held"] = {f"{c} vs {e}": cl_pair(H[c], H[e]) for c, e in zip(C, E)} | {"seed s1 vs s0": cl_pair(H[E[1]], H[E[0]])}
    hp = [res["held"][f"{c} vs {e}"] for c, e in zip(C, E)]
    res["held_line"] = None if None in hp or min(x["scenes"] for x in hp) < 460 else bool(all(x["ci0"] and abs(x["dzeros"]) <= 3 for x in hp))
    # 4 PAI 281 (submission config)
    ref, rr = wt.metrics(glob.glob(f"{wt.PAI_REF}/runs/*/sim")), pai(E[0])
    rep = cl_pair(rr, ref)
    e0 = ref if rep is not None and rep["identical_scores"] and rep["scenes"] == len(ref) else rr
    res["pai_e0_reread"] = dict(vs_stored=rep, e0_used="stored" if e0 is ref else "re-read")
    P = {E[0]: e0, E[1]: pai(E[1]), C[0]: pai(C[0]), C[1]: pai(C[1])}
    res["pai"] = {f"{c} vs {e}": cl_pair(P[c], P[e]) for c, e in zip(C, E)} | {"seed s1 vs s0": cl_pair(P[E[1]], P[E[0]])}
    pp = [res["pai"][f"{c} vs {e}"] for c, e in zip(C, E)]
    res["pai_line"] = None if None in pp or min(x["scenes"] for x in pp) < 281 else bool(all(x["ci0"] for x in pp))
    lines = dict(dev_ade=res["dev_ade"]["line"], ledger=res["ledger_line"], held=res["held_line"], pai=res["pai_line"])
    res["lines"] = lines
    res["verdict"] = ("SWITCH at the next lane boundary" if all(v is True for v in lines.values()) else
                      "INCOMPLETE" if any(v is None for v in lines.values()) else "KEEP eager default")
    write(res, a.dry)


def write(res, dry):
    f3 = lambda x, q=".4f": "-" if x is None else format(x, q)  # noqa: E731
    tr, idt = res["train"], res["identity"] or {}
    L = ["# rm_train torch.compile vs eager (RSG-L18 recipe, 2 500 steps from RB0-F-s0; prereg plans/2026-10-17-compile-prereg.md)", "",
         f"Eager identity {R} (today's code, flag off) vs RSG-L18-s0: identical {idt.get('identical')}, weight max |diff| {idt.get('weight_max_abs_diff')}, "
         f"on_hash_60 {idt.get('on_hash_60')}", "",
         "| tag | git | card | wall s | loop s | steady ms/step | warm-up s | dev ADE |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for t in (R,) + E + C:
        m = tr[t]
        L.append(f"| {t} | {m['git']} | {m['card']} | {m['wall_s']:.0f} | {m['loop_s']:.0f} | {f3(m.get('steady_ms'), '.1f')} | {f3(m.get('warmup_s'), '.0f')} | {m['dev_ade']:.4f} |")
    s = tr["saved"]
    L += ["", f"saved per training: {', '.join(f'{x:.0f} s' for x in s['per_training_s'])}; wall speed-up {', '.join(f'{x:.2f}x' for x in s['speedup_wall'])}; "
          f"steady {', '.join(f'{x:.2f}x' for x in s['speedup_steady'] or [])}", "",
          "weights: " + "; ".join(f"{k}: max {v['max_abs']:.2e}, L2 {v['l2']:.3f}" for k, v in res["weights"].items()), "",
          "| line | value E0 / E1 / C0 / C1 | spread (floor) | C - E (s0, s1) | paired C - E (s0; s1; seed) | pass |", "|---|---|--:|--:|---|:-:|"]
    d = res["dev_ade"]
    vv = " / ".join(f"{d['values'][t]:.4f}" for t in E + C)
    L.append(f"| dev ADE | {vv} | {d['spread']:.4f} | {', '.join(f'{x:+.4f}' for x in d['diff'])} | - | {d['line']} |")
    for k in CELLS:
        r = res["ledger"][k]
        if r == "missing":
            L.append(f"| {k} | missing | | | | - |")
            continue
        pr = "; ".join(r.get("paired", {}).values()) or "-"
        vv = " / ".join(f"{r['values'][t]:.4f}" for t in E + C)
        L.append(f"| {k} | {vv} | {r['spread']:.4f} ({r['spread_floor']:.4f}) | "
                 f"{', '.join(f'{x:+.4f}' for x in r['diff'])} | {pr} | {r['line']} |")
    la = res["ledger"]["lat2s"]
    L.append(f"| lateral 2 s med abs diff (m) | C vs E: {', '.join(f3(x, '.4f') for x in la['compiled'])} | E1 vs E0: {f3(la['spread'])} | - | - | {la['line']} |")
    L += ["", "| closed loop | scenes | zeros J / C | mean J / C | paired J - C [session CI] | at-fault J / C | offroad J / C | identical scores | pass |",
          "|---|--:|--:|--:|--:|--:|--:|:-:|:-:|"]
    for blk, ln in (("held", res["held_line"]), ("pai", res["pai_line"])):
        for k, r in res[blk].items():
            if r is None:
                L.append(f"| {blk} {k} | missing | | | | | | | |")
                continue
            ok = "-" if k.startswith("seed") else (r["ci0"] and (blk == "pai" or abs(r["dzeros"]) <= 3))
            L.append(f"| {blk} {k} | {r['scenes']} | {r['J']['zeros']} / {r['C']['zeros']} | {r['J']['mean']:.4f} / {r['C']['mean']:.4f} | {r['paired']} | "
                     f"{r['J']['at_fault']} / {r['C']['at_fault']} | {r['J']['offroad']} / {r['C']['offroad']} | {r['identical_scores']} | {ok} |")
    rp = res["pai_e0_reread"]
    L += ["", f"PAI E0: today's re-read vs the stored read: {'-' if rp['vs_stored'] is None else rp['vs_stored']['paired']}, identical scores "
          f"{None if rp['vs_stored'] is None else rp['vs_stored']['identical_scores']}; E0 used: {rp['e0_used']}", "",
          "lines: " + ", ".join(f"{k} {v}" for k, v in res["lines"].items()) + f"; **{res['verdict']}**"]
    if dry:
        print("\n".join(L))
        return
    O.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    for p in (O / "summary", RES / "compile_val"):
        p.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
        p.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("summary").add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"summary": cmd_summary}[a.cmd](a)
