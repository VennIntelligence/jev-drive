#!/usr/bin/env python3
"""Lane JOINT phase 1 read-out (plans/2026-10-11-joint1-prereg.md) of the closed-loop runs on the 1491 public scenes and the bench guardrails.

  .venv/bin/python experiments/alpasim/scripts/joint1_report.py --out DIR [--checks-only]       (box)
Writes DIR/{joint1_report.md, joint1_stats.json, joint1_per_scene.csv}. --checks-only: the reproducibility checks of the stage `ref` alone.

Sources per checkpoint (ot2_loop manifests, {label: [run dirs]}): stored compiled reads (TR1 / CF1 / BODY1) unless a reproducibility check of their
driver family failed, then the `refx` reruns; the reruns of the stage `ref` for the stored uncompiled reads; lane nuplan1491 for P2H10S-F on the 791;
the stage `arms` for the new checkpoints. Scene helpers (records, zero class, log, by-log bootstrap) come from c0b_report.py, the turn of a scene
(its navtest token's logged 4 s heading change) from experiments/body1/scripts/stop_report.py.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent), str(_H.parents[2] / "body1" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import c0b_report as R  # noqa: E402
import stop_report as SR  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RA = DATA / "runs/alpasim"
CLS = ("collision_at_fault", "offroad", "left_corridor_laterally")
SHORT = {"collision_at_fault": "collision", "offroad": "offroad", "left_corridor_laterally": "corridor"}
ARMS = {"P2H10": "P2H10-F", "P2H10S": "P2H10S-F", "APY": "APY10m10-AB", "APS": "JAPS", "APS-Y": "JAPSnoY", "APS-A": "JAPSnoA"}
DROP = {"yaw-rate rows": "APS-Y", "AP2 input standard": "APS-A", "body lesson S": "APY"}       # ingredient -> its drop-one arm
EVAL = ["APS", "APS-Y", "APS-A", "APY", "P2H10S"]                                                # arms read against the line
B, SEED = 10000, 0


def man(p):
    p = RA / p if not str(p).startswith("/") else Path(p)
    return json.loads(p.read_text()) if p.exists() else {}


def checks():
    """Stage `ref` reproducibility checks: rerun against stored compiled reads, scene by scene."""
    ref, out = man("joint1/ref/manifest.json"), {}
    pairs = {"sh30": [("P2H10-F-s0@chk0", man("tr1/a/manifest.json").get("P2H10-F-s0", []), "chunk0", "TR1"),
                      ("P2H10-F-s0@fresh", man("cf1/a/manifest.json").get("P2H10-F-s0@fresh", []), "cf1fresh", "CF1")],
             "ap2": [("APY10m10-AB-s0@fresh", man("cf1/a/manifest.json").get("APY10m10-AB-s0@fresh", []), "cf1fresh", "CF1")]}
    for fam, ps in pairs.items():
        for lab, stored, lst, src in ps:
            if lab not in ref:
                out[lab] = dict(family=fam, status="not run yet")
                continue
            a, b = R.load_driver(ref[lab])[0], R.load_driver(stored)[0]
            sc = sorted(set((RA / "c0b/lists" / f"{lst}.txt").read_text().split()))
            sa, sb = [s for s in sc if s in a], [s for s in sc if s in b]
            diff = [s for s in sc if s in a and s in b and (abs(a[s]["score"] - b[s]["score"]) > 1e-9 or R.zclass(a[s]) != R.zclass(b[s]))]
            out[lab] = dict(family=fam, against=src, scenes=len(sc), rerun=len(sa), stored=len(sb), differ=len(diff), differ_scenes=diff[:20],
                            max_abs=max((abs(a[s]["score"] - b[s]["score"]) for s in sc if s in a and s in b), default=float("nan")),
                            mean_rerun=float(np.mean([a[s]["score"] for s in sa])) if sa else None, mean_stored=float(np.mean([b[s]["score"] for s in sb])) if sb else None,
                            status="pass" if len(sa) == len(sb) == len(sc) and not diff else "FAIL")
    fam_ok = {f: all(v["status"] == "pass" for v in out.values() if v["family"] == f) for f in ("sh30", "ap2")}
    return out, fam_ok


def sources(fam_ok):
    """tag -> run dirs covering the 1491 scenes (module docstring)."""
    tr1, cf1, ref, refx, arms, n1491, body = (man(p) for p in ("tr1/a/manifest.json", "cf1/a/manifest.json", "joint1/ref/manifest.json",
                                                                "joint1/refx/manifest.json", "joint1/arms/manifest.json", "nuplan1491/a/manifest.json",
                                                                DATA / "runs/body1/cl/shape-cl/manifest.json"))
    src = {}
    for s in (0, 1):
        sh = fam_ok["sh30"]
        src[f"P2H10-F-s{s}"] = ((tr1 if sh else refx).get(f"P2H10-F-s{s}", []) + ref.get(f"P2H10-F-s{s}@new", []) +
                                (cf1.get(f"P2H10-F-s{s}@fresh", []) if sh else (ref if s == 0 else refx).get(f"P2H10-F-s{s}@fresh", [])))
        src[f"P2H10S-F-s{s}"] = ((body if sh else refx).get(f"P2H10S-F-s{s}", []) + n1491.get(f"P2H10S-F-s{s}@cf1fresh", []) +
                                 n1491.get(f"P2H10S-F-s{s}@ot3new", []))
        ap = fam_ok["ap2"]
        src[f"APY10m10-AB-s{s}"] = (ref.get(f"APY10m10-AB-s{s}", []) + (cf1 if ap else refx).get(f"APY10m10-AB-s{s}@new", []) +
                                    (cf1.get(f"APY10m10-AB-s{s}@fresh", []) if ap else (ref if s == 0 else refx).get(f"APY10m10-AB-s{s}@fresh", [])))
        for t in ("JAPS", "JAPSnoY", "JAPSnoA"):
            src[f"{t}-s{s}"] = arms.get(f"{t}-s{s}", []) + arms.get(f"{t}-s{s}@fresh", []) + arms.get(f"{t}-s{s}@new", [])
    for s in (2, 3):                                                 # descriptive only (lane nuplan1491 + BODY1)
        src[f"P2H10S-F-s{s}"] = body.get(f"P2H10S-F-s{s}", []) + n1491.get(f"P2H10S-F-s{s}@cf1fresh", []) + n1491.get(f"P2H10S-F-s{s}@ot3new", [])
    return src


def sboot(d, seed=SEED):
    """Sum of the per-scene values d with a 95 % bootstrap CI resampling scenes (the same draw for both sides of a paired difference)."""
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (B, len(d)))
    b = d[idx].sum(1)
    return float(d.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def f4(t, sign=True):
    return R.fmt(t, 4, sign)


def fz(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


class Arm:
    """Per-scene arrays of one tag (one seed) on the scene list."""

    def __init__(self, recs, nd, scenes):
        self.sc = np.array([recs[s]["score"] for s in scenes], float)
        self.cls = np.array([R.zclass(recs[s]) if recs[s]["score"] == 0 else "" for s in scenes])
        self.ev = np.array([float((recs[s].get("metrics") or {}).get("offroad_or_collision_at_fault", 0) or 0) for s in scenes])
        self.prog = np.array([float((recs[s].get("score_metrics") or {}).get("progress_clipped_rel") or 0.0) for s in scenes])
        self.fold = np.array([recs[s]["score"] == 0 and nd.get(s, 0) == 0 for s in scenes])

    def z(self, which=CLS):
        return np.isin(self.cls, which).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    CH, fam_ok = checks()
    L = ["# JOINT-1 read-out (generated by scripts/joint1_report.py; do not edit)", "", "## Reproducibility checks (stage `ref`)", "",
         "| rerun | against | scenes | rerun / stored rollouts | scenes that differ (score or zero class) | max abs score difference | mean rerun / stored | verdict |",
         "|:--|:--|--:|:--|--:|--:|:--|:--|"]
    for k, v in CH.items():
        if v.get("status") == "not run yet":
            L.append(f"| {k} | | | | | | | not run yet |")
            continue
        L.append(f"| {k} | {v['against']} | {v['scenes']} | {v['rerun']} / {v['stored']} | {v['differ']} | {v['max_abs']:.4f} | "
                 f"{v['mean_rerun']:.4f} / {v['mean_stored']:.4f} | {v['status']} |")
    L += ["", f"Driver families whose stored compiled reads are used: {', '.join(f for f, ok in fam_ok.items() if ok) or 'none'} "
          f"(the others come from the `refx` reruns).", ""]
    st = dict(checks=CH, family_ok=fam_ok)
    if a.checks_only:
        (out / "joint1_checks.md").write_text("\n".join(L) + "\n")
        (out / "joint1_checks.json").write_text(json.dumps(st, indent=1, default=float))
        print("\n".join(L))
        return
    src = sources(fam_ok)
    D = {t: R.load_driver(d) for t, d in src.items() if d}
    all_sc = set()
    for L5 in ("chunk0", "chunk1", "chunk2", "cf1fresh", "ot3new"):
        all_sc |= set((RA / "c0b/lists" / f"{L5}.txt").read_text().split())
    scenes = sorted(all_sc)
    assert len(scenes) == 1491, len(scenes)
    have = {t for t, d in D.items() if all(s in d[0] for s in scenes)}
    miss = {t: len([s for s in scenes if s not in D[t][0]]) if t in D else 1491 for t in src if t not in have}
    mism = {t: d[1] for t, d in D.items() if d[1]}
    logs = np.array([R.log_of(s) for s in scenes])
    T = SR.turns()
    turn = np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in scenes])
    g45 = np.abs(turn) > 45
    A = {t: Arm(D[t][0], D[t][3], scenes) for t in have}
    rec = {k: [f"{v}-s0", f"{v}-s1"] for k, v in ARMS.items()}
    ok = {k: all(t in A for t in v) for k, v in rec.items()}
    L += ["## Coverage", "", f"1491 scenes, {len(set(logs))} logs, {int(g45.sum())} turning more than 45 deg (navtest token's logged 4 s heading change; "
          f"{int(np.isnan(turn).sum())} scenes without a navtest token). Checkpoints with all 1491 scenes: {', '.join(sorted(have))}.",
          "" if not miss else f"Incomplete (scenes missing): {miss}.", "" if not mism else f"Scores that differ between two runs of one checkpoint: {mism}.", ""]
    st |= dict(scenes=len(scenes), logs=len(set(logs)), gt45=int(g45.sum()), missing=miss, sources={t: [str(x) for x in d] for t, d in src.items()})
    if not all(ok.values()):
        L.append(f"**Arms not complete: {[k for k, v in ok.items() if not v]}; the tables below cover the complete ones only.**\n")
    K = [k for k in ARMS if ok[k]]
    mean = lambda k, f: np.mean([f(A[t]) for t in rec[k]], 0)  # noqa: E731
    sc = {k: mean(k, lambda x: x.sc) for k in K}
    z3 = {k: mean(k, lambda x: x.z()) for k in K}
    zc = {k: {c: mean(k, lambda x, c=c: x.z((c,))) for c in CLS} for k in K}
    zco = {k: zc[k]["collision_at_fault"] + zc[k]["offroad"] for k in K}
    fold = {k: mean(k, lambda x: x.fold.astype(float)) for k in K}

    # ---- per arm
    L += ["## Per arm (recipe = per-scene mean of its two seeds; zero counts = seed means)", "",
          "| arm | tags | mean scene score [95 % CI, logs] | seeds | zeros (collision + offroad + corridor) | collision / offroad / corridor | "
          "route-fold zeros | at-fault events | slow | progress | > 45 deg: mean, zeros (collision / offroad / corridor) |",
          "|:--|:--|:--|:--|--:|:--|--:|--:|--:|--:|:--|"]
    st["arms"] = {}
    for k in K:
        m = R.ci(sc[k], logs)
        seeds = [float(A[t].sc.mean()) for t in rec[k]]
        ev = float(np.mean([A[t].ev.sum() for t in rec[k]]))
        slow = float(np.mean([((A[t].sc > 0) & (A[t].sc < 1)).sum() for t in rec[k]]))
        prog = float(np.mean([A[t].prog.mean() for t in rec[k]]))
        cz = {SHORT[c]: float(zc[k][c].sum()) for c in CLS}
        g = {SHORT[c]: float(zc[k][c][g45].sum()) for c in CLS}
        st["arms"][k] = dict(tags=rec[k], mean=m, seeds=seeds, zeros3=float(z3[k].sum()), classes=cz, fold=float(fold[k].sum()), events=ev, slow=slow,
                             progress=prog, gt45_mean=float(sc[k][g45].mean()), gt45_classes=g)
        L.append(f"| {k} | {' + '.join(rec[k])} | {R.fmt(m)} | {' / '.join(f'{x:.4f}' for x in seeds)} | {z3[k].sum():g} | "
                 f"{cz['collision']:g} / {cz['offroad']:g} / {cz['corridor']:g} | {fold[k].sum():g} | {ev:g} | {slow:g} | {prog:.4f} | "
                 f"{sc[k][g45].mean():.4f}, {sum(g.values()):g} ({g['collision']:g} / {g['offroad']:g} / {g['corridor']:g}) |")
    extra = [t for t in ("P2H10S-F-s2", "P2H10S-F-s3") if t in A]
    if extra:
        L += ["", "Descriptive only (not in any line): " + "; ".join(f"{t} {A[t].sc.mean():.4f}, zeros {A[t].z().sum():g} "
                                                                   f"({' / '.join(str(int(A[t].z((c,)).sum())) for c in CLS)})" for t in extra) + "."]

    # ---- registered line
    L += ["", "## Registered line (candidate for submission #2) against P2H10-F", "",
          "L1: zeros of the three classes, sum over scenes of (arm - P2H10-F) seed-mean indicators, 95 % CI by scene bootstrap (B 10 000, seed 0); "
          "met if < 0 with upper bound < 0. L2: mean scene score difference >= -0.002 (by-log CI shown). L3: guardrails (next section).", "",
          "| arm | L1 zeros difference [CI scenes] (CI logs) | L1 | L2 mean difference [CI logs] | L2 | collision + offroad zeros difference | corridor zeros difference |",
          "|:--|:--|:--|:--|:--|:--|:--|"]
    st["line"] = {}
    for k in [x for x in EVAL if x in K]:
        if "P2H10" not in K:
            break
        d1 = sboot(z3[k] - z3["P2H10"])
        d1l = R.ci(z3[k] - z3["P2H10"], logs)
        d2 = R.ci(sc[k] - sc["P2H10"], logs)
        dco, dcr = sboot(zco[k] - zco["P2H10"]), sboot(zc[k]["left_corridor_laterally"] - zc["P2H10"]["left_corridor_laterally"])
        l1, l2 = d1[0] < 0 and d1[2] < 0, d2[0] >= -0.002
        st["line"][k] = dict(L1=d1, L1_logs=[x * len(scenes) for x in d1l], L1_met=l1, L2=d2, L2_met=l2, co=dco, corridor=dcr)
        L.append(f"| {k} | {fz(d1)} ({d1l[1] * len(scenes):+.1f}, {d1l[2] * len(scenes):+.1f}) | {'met' if l1 else 'not met'} | {f4(d2)} | "
                 f"{'met' if l2 else 'not met'} | {fz(dco)} | {fz(dcr)} |")

    # ---- guardrails
    L += ["", "## Guardrails (jevdrive.bench; two-seed means; paired differences with by-log CIs, HUGSIM by scenario)", ""]
    try:
        G = guards(rec, K)
    except Exception as e:                                           # noqa: BLE001
        G = {"error": repr(e)}
    st["guards"] = G
    if "error" in G:
        L.append(f"Guardrail reads failed: {G['error']}\n")
    else:
        L += ["| arm | navtest EPDMS | - P2H10S-F [CI] (L3: >= -0.2) | navhard combined | - P2H10-F [CI] (L3: not below) | stage 1 / stage 2 - P2H10-F | "
              "HUGSIM HD | - P2H10-F [CI] (L3: >= -0.03) | L3 |", "|:--|--:|:--|--:|:--|:--|--:|:--|:--|"]
        for k in K:
            g = G.get(k, {})
            nt, nh, hu = g.get("navtest"), g.get("navhard"), g.get("hugsim")
            l3 = g.get("L3")
            L.append(f"| {k} | {nt['value']:.2f} | {fd(nt.get('vs_P2H10S'))} | " if nt else f"| {k} | n/a | n/a | ")
            L[-1] += (f"{nh['value']:.2f} | {fd(nh.get('vs_P2H10'))} | {fd(nh.get('s1_vs_P2H10'))} / {fd(nh.get('s2_vs_P2H10'))} | " if nh else "n/a | n/a | n/a | ")
            L[-1] += (f"{hu['value']:.3f} | {fd(hu.get('vs_P2H10'), '+.3f')} | " if hu else "not read | | ") + f"{l3} |"
    verdict = {}
    for k in [x for x in EVAL if x in K and x in st["line"]]:
        l3 = (G.get(k) or {}).get("L3") if "error" not in G else None
        verdict[k] = dict(L1=st["line"][k]["L1_met"], L2=st["line"][k]["L2_met"], L3=l3,
                          candidate=bool(st["line"][k]["L1_met"] and st["line"][k]["L2_met"] and l3 == "met"))
    cand = [k for k, v in verdict.items() if v["candidate"]]
    pick = min(cand, key=lambda k: (zco[k].sum(), -sc[k].mean())) if cand else None
    pend = [k for k, v in verdict.items() if v["L1"] and v["L2"] and v["L3"] in ("HUGSIM not read", None)]
    st["verdict"], st["recommended"], st["pending_hugsim"] = verdict, pick, pend
    L += ["", "### Verdict", ""] + [f"- {k}: L1 {'met' if v['L1'] else 'not met'}, L2 {'met' if v['L2'] else 'not met'}, L3 {v['L3']} -> "
                                    f"**{'candidate' if v['candidate'] else 'not a candidate'}**" for k, v in verdict.items()]
    L += ["", f"Recommended arm by the registered rule (fewest at-fault collision + offroad zeros among candidates): **{pick or 'none'}**"
          + (f"; arms meeting L1 and L2 that still need their HUGSIM read: {pend}" if pend else "") + "."]

    # ---- fixed vs broken
    L += ["", "## Fixed vs broken (per scene: zero seeds of the reference -> zero seeds of the arm; fixed 2 -> 0, broken 0 -> 2, partial = other changes)", ""]
    pairs = [(k, "P2H10") for k in K if k != "P2H10"] + [("APS", DROP[c]) for c in DROP if "APS" in K and DROP[c] in K]
    L += ["| arm vs reference | class | fixed | broken | partial (better / worse) | zeros reference -> arm |", "|:--|:--|--:|--:|:--|:--|"]
    st["flips"] = []
    for x, y in pairs:
        for name, cl in (("three classes", CLS), ("collision + offroad", CLS[:2]), ("corridor", CLS[2:]), ("collision", CLS[:1]), ("offroad", CLS[1:2])):
            nx = sum(A[t].z(cl) for t in rec[x])
            ny = sum(A[t].z(cl) for t in rec[y])
            fx, br = int(((ny == 2) & (nx == 0)).sum()), int(((ny == 0) & (nx == 2)).sum())
            pb, pw = int(((nx < ny) & ~((ny == 2) & (nx == 0))).sum()), int(((nx > ny) & ~((ny == 0) & (nx == 2))).sum())
            st["flips"].append(dict(arm=x, ref=y, cls=name, fixed=fx, broken=br, partial_better=pb, partial_worse=pw, ref_zeros=float(ny.sum() / 2), arm_zeros=float(nx.sum() / 2)))
            L.append(f"| {x} vs {y} | {name} | {fx} | {br} | {pb} / {pw} | {ny.sum() / 2:g} -> {nx.sum() / 2:g} |")

    # ---- zero composition and the non-additivity read
    if all(x in K for x in ("APS", "P2H10", *DROP.values())):
        L += ["", "## Zero composition (scenes that are zero in at least one seed; three classes)", ""]
        zs = {k: set(np.flatnonzero(sum(A[t].z() for t in rec[k]) > 0)) for k in K}
        L += ["| set | scenes |", "|:--|--:|"] + [f"| {k} | {len(zs[k])} |" for k in K]
        u = set().union(*(zs[DROP[c]] for c in DROP))
        L += [f"| APS and P2H10-F | {len(zs['APS'] & zs['P2H10'])} |", f"| APS only (zero in no drop-one arm) | {len(zs['APS'] - u)} |",
              f"| in every drop-one arm, not in APS | {len(set.intersection(*(zs[DROP[c]] for c in DROP)) - zs['APS'])} |"]
        L += ["", "## Non-additivity read (decision 244's wording; E = APS - P2H10-F)", "",
              "| quantity | E = APS - P2H10-F | ingredient | X - APS | X - P2H10-F | read |", "|:--|:--|:--|:--|:--|:--|"]
        Q = {"zeros (three classes)": (z3, True), "collision + offroad zeros": (zco, True),
             "corridor zeros": ({k: zc[k]["left_corridor_laterally"] for k in K}, True), "mean scene score": (sc, False)}
        st["nonadd"] = {}
        for qn, (v, is_z) in Q.items():
            ci = (lambda d: sboot(d)) if is_z else (lambda d: R.ci(d, logs))
            fmt = fz if is_z else f4
            E = ci(v["APS"] - v["P2H10"])
            e_sig = E[1] > 0 or E[2] < 0
            for c, x in DROP.items():
                xa, xb = ci(v[x] - v["APS"]), ci(v[x] - v["P2H10"])
                if not e_sig:
                    rd = "no effect of APS to attribute"
                else:
                    s = np.sign(E[0])
                    sig = lambda t: t[1] > 0 or t[2] < 0  # noqa: E731
                    if sig(xa) and np.sign(xa[0]) == -s and abs(xa[0]) >= abs(E[0]) / 2:
                        rd = "carries"
                    elif sig(xb) and np.sign(xb[0]) == s and abs(xb[0]) >= abs(E[0]) / 2:
                        rd = "is not needed"
                    elif sig(xa) and np.sign(xa[0]) == s:
                        rd = "hurts"
                    else:
                        rd = "unresolved"
                st["nonadd"].setdefault(qn, {})[c] = dict(E=E, x_minus_aps=xa, x_minus_base=xb, read=rd)
                L.append(f"| {qn} | {fmt(E)} | {c} | {fmt(xa)} | {fmt(xb)} | **{rd}** |")
    # ---- per scene table
    with (out / "joint1_per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        tags = sorted(A)
        w.writerow(["scene", "log", "turn4s_deg"] + [f"{t}|{c}" for t in tags for c in ("score", "zero_class", "atfault_events", "progress")])
        for i, s in enumerate(scenes):
            w.writerow([s, logs[i], "" if np.isnan(turn[i]) else round(float(turn[i]), 1)] +
                       [x for t in tags for x in (round(float(A[t].sc[i]), 6), A[t].cls[i], A[t].ev[i], round(float(A[t].prog[i]), 4))])
    (out / "joint1_stats.json").write_text(json.dumps(st, indent=1, default=float))
    (out / "joint1_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def fd(r, f="+.2f"):
    return "n/a" if not r else f"{r['mean']:{f}} [{r['lo']:{f}}, {r['hi']:{f}}]"


def guards(rec, K):
    """navtest EPDMS / navhard combined / HUGSIM HD (spec) per arm (two-seed means) and the L3 clauses."""
    from jevdrive import stats
    from jevdrive.bench import tables as BT

    def units(bench, spec, col, idx_log=True):
        kw = {"preset": "spec"} if bench == "hugsim" else {}
        u, _ = BT.load(bench, spec, **kw)
        return None if u is None else u
    out = {}
    U = {}
    for k in K:
        for bench, sfx, col in (("navtest", "", "score"), ("navhard", "@gimm", "combined"), ("hugsim", "", "hdscore")):
            us = [units(bench, t + sfx, col) for t in rec[k]]
            if any(u is None for u in us):
                continue
            ix = us[0].index
            for u in us[1:]:
                ix = ix.intersection(u.index)
            v = np.mean([u.loc[ix, col].to_numpy(float) for u in us], 0)
            extra = {}
            if bench == "navhard":
                extra = {c: np.mean([u.loc[ix, c].to_numpy(float) for u in us], 0) for c in ("stage1", "stage2")}
            U[(k, bench)] = (ix, v * (100.0 if bench == "navtest" else 1.0), us[0].loc[ix, "log"].astype(str).to_numpy() if "log" in us[0] else None, extra)
    for k in K:
        g = {}
        for bench, ref in (("navtest", "P2H10S"), ("navhard", "P2H10"), ("hugsim", "P2H10")):
            if (k, bench) not in U:
                continue
            ix, v, lg, ex = U[(k, bench)]
            g[bench] = dict(value=float(v.mean()))
            if (ref, bench) in U:
                rix, rv, _, rex = U[(ref, bench)]
                pos = {x: i for i, x in enumerate(rix)}
                common = [i for i, x in enumerate(ix) if x in pos]
                rj = [pos[ix[i]] for i in common]
                g[bench][f"vs_{ref}"] = stats.paired(v[common], rv[rj], None if bench == "hugsim" else lg[common])
                if bench == "navhard":
                    for c, cc in (("stage1", "s1"), ("stage2", "s2")):
                        g[bench][f"{cc}_vs_{ref}"] = stats.paired(ex[c][common], rex[c][rj], lg[common])
        nt, nh, hu = (g.get(b, {}).get(f"vs_{r}") for b, r in (("navtest", "P2H10S"), ("navhard", "P2H10"), ("hugsim", "P2H10")))
        if k == "P2H10":
            l3 = "reference (the line's base)"
        elif nt is None or nh is None:
            l3 = "not read"
        else:
            ok = nt["mean"] >= -0.2 and nh["hi"] >= 0
            l3 = ("not met" if not ok else "HUGSIM not read" if hu is None else "met" if hu["mean"] >= -0.03 else "not met")
        g["L3"] = l3
        out[k] = g
    return out


if __name__ == "__main__":
    main()
