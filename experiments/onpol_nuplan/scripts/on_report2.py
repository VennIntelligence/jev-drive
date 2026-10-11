#!/usr/bin/env python3
"""ONPOL-NUPLAN round-2 read-out (plans/2026-10-12-onpol2-prereg.md, sections B and C) of the `held` runs of on_read.sh, evaluated mechanically
with on_report.py's helpers (records, zero kinds, seed means, paired differences, fixed / broken and CIs are defined in its docstring).

  .venv/bin/python experiments/onpol_nuplan/scripts/on_report2.py [--out experiments/onpol_nuplan/results/read2]          (box)
  .venv/bin/python experiments/onpol_nuplan/scripts/on_report2.py --standin [--out experiments/onpol_nuplan/results/standin2]
Writes <out>/{on2_report.md, on2_stats.json, on2_per_scene.csv}. Source: $DATA_DIR/runs/onpol_nuplan/read/held/manifest.json on results/split/held.txt
(460 scenes; round 1's reads of P2H10S-F s0-s3 and SSELFnoL-ON s0 / s1 are in the same manifest: one driver path, this lane's runs). A tag without
all scenes is left out with a line saying so, and so is every row and sentence that needs it.

B  `YSELFnoL-ON` (S's two hinges on on-policy rows, YR10m10-F base) and `YNONE-ON` (the same continuation without the rows), s0 / s1, against the
   control `YR10m10-F` s0 / s1. Lines, seed means over the 460 scenes:
     (i)   YSELFnoL has fewer zeros in total and fewer at-fault zeros than the control
     (ii)  hard gate: its at-fault zeros are not above the control's
     (iii) fixed - broken (zeros of the three kinds) of YSELFnoL against the control exceeds the null = the maximum of |fixed - broken| of the
           control's s0 against its s1 (single seeds) and of round 1's four-seed null (the control P2H10S-F against itself, +2 on held)
   "Holds on the YR base" = (i) and (ii) and (iii). Beside it: YSELFnoL against YNONE (rows against steps), mean score, slow scenes, progress,
   GT distance (the scorer's dist_to_gt_trajectory, m), corridor zeros and the scenes turning more than 45 deg.
C  `SSELFnoL-ON` s0-s3 against `P2H10S-F` s0-s3: zeros, at-fault zeros (H as round 1: the arm's seed mean <= the control's), fixed - broken of arm
   s2, s3 against control s2, s3 and of all four against all four, the null as round 1 registered it (the three 2-against-2 splits of the
   control, both directions), and each new seed alone. "Decision 259 firms up" = the four-seed paired difference in zeros is negative with a
   scene CI excluding 0, and s2 and s3 each have fewer zeros than the control seed they start from; otherwise "259 is demoted in place".
   (The prereg's "seeds 2, 3 alone have fewer zeros than their control seeds" is read per seed; the two-seed mean is printed beside it.)

--standin is the test of this file and reads no held scene: stored reads of the TRAIN-split scenes (results/split/train.txt) play the read, with
stored recipes in the place of the round-2 tags (STANDIN; seeds 2 / 3 of JAPSnoY from lane seeds4's manifest). It ends with a self-check against
results/seeds4/seeds4_per_scene.csv (seeds4_report.py's table) restricted to the same scenes; exit 3 if anything differs.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent)]
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import on_report as O  # noqa: E402
from on_report import J, R, SR, stats  # noqa: E402

E, SPLIT, CLS, SHORT, KIND = O.E, O.SPLIT, O.CLS, O.SHORT, O.KIND
YC, YS, YN, S4, C4 = "control YR10m10-F", "YSELFnoL-ON", "YNONE-ON", "SSELFnoL-ON (4 seeds)", "control P2H10S-F (4 seeds)"
S23, C23, S01, C01 = "SSELFnoL-ON s2, s3", "control P2H10S-F s2, s3", "SSELFnoL-ON s0, s1", "control P2H10S-F s0, s1"
REC = {YC: [f"YR10m10-F-s{s}" for s in (0, 1)], YS: [f"YSELFnoL-ON-s{s}" for s in (0, 1)], YN: [f"YNONE-ON-s{s}" for s in (0, 1)],
       S4: [f"SSELFnoL-ON-s{s}" for s in range(4)], C4: [f"P2H10S-F-s{s}" for s in range(4)]}
REC |= {S23: REC[S4][2:], C23: REC[C4][2:], S01: REC[S4][:2], C01: REC[C4][:2]}
STANDIN = {"YR10m10-F": "P2H10-F", "YSELFnoL-ON": "JAPSnoY", "YNONE-ON": "APY10m10-AB", "SSELFnoL-ON": "JAPSnoY"}      # stored recipes that play the tags
NULL1 = 2                                                            # round 1's four-seed null on held (results/read/on_stats.json; the prereg's "+2")


def gt_dist(recs, scenes):
    return np.array([float((recs[s].get("metrics") or {}).get("dist_to_gt_trajectory", np.nan)) for s in scenes])


def null4(A, ctl):
    """Round 1's null: fixed - broken (zeros) of the control against itself over the three 2-against-2 splits, both directions."""
    return [O.flips(A, [ctl[i] for i in x], [ctl[i] for i in y])["net"] for p, q in O.SPLITS for x, y in ((p, q), (q, p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--standin", action="store_true", help="test on stored reads of train-split scenes; reads no held scene")
    ap.add_argument("--manifest", default=str(J.DATA / "runs/onpol_nuplan/read/held/manifest.json"))
    a = ap.parse_args()
    out = _pl.Path(a.out or E / "results" / ("standin2" if a.standin else "read2"))
    out.mkdir(parents=True, exist_ok=True)
    split = "train" if a.standin else "held"
    scenes = sorted((SPLIT / f"{split}.txt").read_text().split())
    keep = set(scenes)
    assert not keep & set((SPLIT / ("held.txt" if a.standin else "train.txt")).read_text().split()), "train and held overlap"
    tags = sorted({t for v in REC.values() for t in v})
    if a.standin:
        stored = J.sources(json.loads((_H.parents[2] / "alpasim/results/joint1/joint1_checks.json").read_text())["family_ok"])
        new = J.man(J.DATA / "runs/onpol_nuplan/seeds4/loop/manifest.json")
        for s in (2, 3):
            stored[f"JAPSnoY-s{s}"] = new.get(f"JAPSnoY-s{s}", []) + new.get(f"JAPSnoY-s{s}@fresh", []) + new.get(f"JAPSnoY-s{s}@new", [])
        name = lambda t: next((STANDIN[f] + t[len(f):] for f in STANDIN if t.startswith(f + "-s")), t)  # noqa: E731
        src = {t: stored.get(name(t), []) for t in tags}
    else:
        man = J.man(a.manifest)
        assert man, f"{a.manifest}: no manifest (run on_read.sh held first)"
        src = {t: man.get(t, []) for t in tags}
    D = {t: O.load(d, keep) for t, d in src.items() if d}
    have = sorted(t for t, d in D.items() if all(s in d[0] for s in scenes))
    miss = {t: (len(keep - set(D[t][0])) if t in D else len(scenes)) for t in tags if t not in have}
    A = {t: J.Arm(D[t][0], D[t][2], scenes) for t in have}
    GT = {t: gt_dist(D[t][0], scenes) for t in have}
    K = [k for k, v in REC.items() if all(t in A for t in v)]
    logs = np.array([R.log_of(s) for s in scenes])
    T = SR.turns()
    turn = np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in scenes])
    g45 = np.abs(turn) > 45
    G = {k: dict(sc=np.mean([A[t].sc for t in REC[k]], 0), slow=np.mean([(A[t].sc > 0) & (A[t].sc < 1) for t in REC[k]], 0), prog=np.mean([A[t].prog for t in REC[k]], 0),
                 gt=np.mean([GT[t] for t in REC[k]], 0), **{q: np.mean([A[t].z(c) for t in REC[k]], 0) for q, c in KIND.items()}) for k in K}
    n = len(scenes)
    L = [("# ONPOL-NUPLAN round 2 read-out: STAND-IN TEST on stored reads of train-split scenes (not a result)" if a.standin else
          "# ONPOL-NUPLAN round 2 read-out on the held scenes") + " (generated by scripts/on_report2.py; do not edit)", ""]
    if a.standin:
        L += ["Test of the reader only. Tags played by stored recipes: " + ", ".join(f"{k} <- {v}" for k, v in STANDIN.items()) + "; P2H10S-F is itself. "
              "These scenes were read by all of them before; the sentences below are mechanics, not findings.", ""]
    L += [f"{n} scenes of `alpasim_nuplan/onpol-{split}`, {len(set(logs))} logs, {int(g45.sum())} turning more than 45 deg. Tags with all scenes: {', '.join(have) or 'none'}.", ""]
    L += [f"**Skipped, not all scenes in the run dirs (scenes missing): {miss}.**", ""] if miss else []
    st = dict(standin=a.standin, split=split, scenes=n, logs=len(set(logs)), gt45=int(g45.sum()), tags={k: REC[k] for k in K}, missing=miss,
              sources={t: [str(x) for x in d] for t, d in src.items()}, arms={}, seeds={}, diff={}, flips=[], B={}, C={})

    def arm_rows(keys):
        rows = ["| arm | tags | per-seed mean score | seed-mean score [95 % CI, logs] | zeros | collision / offroad / corridor | at-fault zeros | "
                "per-seed zeros (collision / offroad / corridor) | at-fault events | slow | progress | GT distance (m) | > 45 deg: mean, zeros (collision / offroad / corridor) |",
                "|:--|:--|:--|:--|--:|:--|--:|:--|--:|--:|--:|--:|:--|"]
        for k in keys:
            if k not in K:
                rows.append(f"| {k} | {' + '.join(REC[k])} | not read | | | | | | | | | | |")
                continue
            m, ts = R.ci(G[k]["sc"], logs), REC[k]
            mean = lambda f: float(np.mean([f(A[t]) for t in ts]))  # noqa: E731
            r = st["arms"][k] = dict(tags=ts, mean=m, seeds=[float(A[t].sc.mean()) for t in ts], **{q: float(G[k][q].sum()) for q in KIND},
                                     per_seed={t: {SHORT[c]: int(A[t].z((c,)).sum()) for c in CLS} for t in ts}, events=mean(lambda x: x.ev.sum()),
                                     slow=mean(lambda x: ((x.sc > 0) & (x.sc < 1)).sum()), progress=mean(lambda x: x.prog.mean()), gt_dist=float(np.nanmean(G[k]["gt"])),
                                     gt45=dict(n=int(g45.sum()), mean=float(G[k]["sc"][g45].mean()), **{q: float(G[k][q][g45].sum()) for q in KIND}))
            g = r["gt45"]
            rows.append(f"| {k} | {' + '.join(ts)} | {' / '.join(f'{x:.4f}' for x in r['seeds'])} | {R.fmt(m)} | {r['zeros']:g} | {r['collision']:g} / {r['offroad']:g} / {r['corridor']:g} | "
                        f"{r['at-fault']:g} | {'; '.join('/'.join(str(v) for v in d.values()) for d in r['per_seed'].values())} | {r['events']:g} | {r['slow']:g} | {r['progress']:.4f} | "
                        f"{r['gt_dist']:.3f} | {g['mean']:.4f}, {g['zeros']:g} ({g['collision']:g} / {g['offroad']:g} / {g['corridor']:g}) |")
        return rows

    def pair_rows(pairs):
        rows = ["| A - B | mean score | zeros | at-fault zeros | collision | offroad | corridor | slow scenes | progress | GT distance (m) | > 45 deg: mean score | > 45 deg: zeros | > 45 deg: corridor |",
                "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
        for x, y in pairs:
            if x not in K or y not in K:
                rows.append(f"| {x} - {y} | not read | | | | | | | | | | | |")
                continue
            d, d45 = O.pair(G, x, y, logs), O.pair(G, x, y, logs, g45)
            g = G[x]["gt"] - G[y]["gt"]
            ok = np.isfinite(g)
            s, l = stats.bootstrap(g[ok]), R.ci(g[ok], logs[ok])
            d["gt_dist"] = dict(point=float(g[ok].mean()), scene=[s["lo"], s["hi"]], logs=[l[1], l[2]], n=int(ok.sum()))
            st["diff"][f"{x} - {y}"] = d | {"gt45": d45}
            rows.append(f"| {x} - {y} | {O.fs(d['score'])} | " + " | ".join(O.fq(d[q]) for q in KIND) + f" | {O.fq(d['slow'])} | {O.fs(d['progress'])} | {O.fs(d['gt_dist'])} | "
                        f"{O.fs(d45['score'])} | {O.fq(d45['zeros'])} | {O.fq(d45['corridor'])} |")
        return rows

    def flip_rows(pairs, kinds=("zeros", "at-fault", "corridor")):
        rows = ["| A vs B | kinds | fixed | broken | fixed - broken | partial (better / worse) | zeros B -> A |", "|:--|:--|--:|--:|--:|:--|:--|"]
        for x, y in pairs:
            if x not in K or y not in K:
                continue
            for q in kinds:
                f = O.flips(A, REC[x], REC[y], KIND[q])
                st["flips"].append(dict(arm=x, ref=y, kinds=q, **{k: v for k, v in f.items() if k not in ("fx", "br")}, fixed_scenes=[scenes[i] for i in f["fx"]],
                                        broken_scenes=[scenes[i] for i in f["br"]]))
                rows.append(f"| {x} vs {y} | {q} | {f['fixed']} | {f['broken']} | {f['net']:+d} | {f['partial_better']} / {f['partial_worse']} | {f['ref_zeros']:g} -> {f['arm_zeros']:g} |")
        for r in st["flips"]:
            if r["kinds"] == "zeros" and (r["arm"], r["ref"]) in pairs and (r["fixed_scenes"] or r["broken_scenes"]):
                ts = REC[r["ref"]] + REC[r["arm"]]
                rows += [""] if rows[-1].startswith("|") else []
                rows += [f"- {r['arm']} vs {r['ref']}, {w} ({len(r[w + '_scenes'])}): " + "; ".join(
                    f"`{s}` ({' '.join(SHORT.get(A[t].cls[scenes.index(s)], '-') for t in ts)})" for s in r[w + "_scenes"]) for w in ("fixed", "broken")]
        return rows
    net = lambda x, y: next(r["net"] for r in st["flips"] if (r["arm"], r["ref"], r["kinds"]) == (x, y, "zeros"))  # noqa: E731
    word = {True: "**holds**", False: "does not hold", None: "**not evaluated**"}
    n4 = null4(A, REC[C4]) if C4 in K else None

    # ---------------------------------------------------------------- B
    L += ["## B. On-policy S rows on the YR10m10-F base", ""]
    f = E / "results/pilot_y/pilot_gates.json"
    if f.exists():
        g = json.loads(f.read_text())
        st["B"]["pilot"] = {t: dict(passed=bool(r["pass"]), dev_ade=r.get("dev_ade"), drift=r.get("dev_drift_off"), fail=r.get("fail", [])) for t, r in g["arms"].items()}
        L += ["Pilot gates (results/pilot_y/pilot_gates.md): " + "; ".join(
            f"{t} {'pass' if r['passed'] else 'FAIL (' + '; '.join(r['fail']) + ')'} (dev ADE {r['dev_ade']:.3f}, dev_drift_off {r['drift']:.3f})" if r["dev_ade"] is not None else f"{t} FAIL"
            for t, r in st["B"]["pilot"].items()) + ".", ""]
    else:
        L += ["Pilot gates: results/pilot_y/pilot_gates.json not found.", ""]
    L += arm_rows([YC, YS, YN]) + [""]
    TR = st["B"]["trainings"] = O.trainings(REC[YS] + REC[YN])
    f3 = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    L += ["Trainings (trainer's dev set; newest run dir of each tag; step 0 = the archived YR10m10-F-s<seed>):", "",
          "| tag | dev ADE at step 0 | dev ADE at the last dev step | dev_drift_off | last dev step | ended | checkpoint |", "|:--|--:|--:|--:|--:|:--|:--|"]
    L += [f"| {t} | | | | | no run dir | |" if r is None else f"| {t} | {f3(r['dev0_ade'])} | {f3(r['dev_ade'])} | {f3(r['dev_drift_off'])} | {r['dev_step']} | "
          f"{'ok' if r['ended_ok'] else 'not ended' if r['ended_ok'] is None else 'FAILED'} | {'yes' if r['ckpt'] else 'no'} |" for t, r in TR.items()] + [""]
    pb = [(YS, YC), (YN, YC), (YS, YN)]
    L += ["Paired differences (A - B of per-scene seed means; point [95 % CI scenes] (CI logs), 10 000 draws; score, progress and GT distance = mean over scenes, zeros and slow = "
          "sum over scenes; slow = scenes with 0 < score < 1):", ""] + pair_rows(pb) + [""]
    L += ["Fixed and broken scenes (fixed = zero in every seed of B and in no seed of A, broken = the reverse):", ""] + flip_rows(pb) + [""]
    if YC in K:
        s01 = O.flips(A, REC[YC][:1], REC[YC][1:])
        r1 = max(n4) if a.standin and n4 is not None else NULL1          # held: the registered +2; the stand-in has its own control null
        nb = max(abs(s01["net"]), r1)
        st["B"]["null"] = dict(control_s0_vs_s1=dict(fixed=s01["fixed"], broken=s01["broken"], net=s01["net"]), round1_four_seed=n4, round1_registered=NULL1, value=nb)
        L += [f"Null of fixed - broken: the control's s0 against s1 (single seeds) fixed {s01['fixed']} / broken {s01['broken']} = {s01['net']:+d}, |.| = {abs(s01['net'])}; round 1's four-seed "
              "null " + (f"recomputed from this manifest: {', '.join(f'{v:+d}' for v in n4)}, maximum {max(n4):+d}" if n4 is not None else f"{NULL1:+d} (results/read/on_stats.json; the control's "
                                                                                                                                  "four seeds are not all in this source)")
              + (f" (the registered {NULL1:+d} is used)" if n4 is not None and not a.standin else "") + f". Null = {nb:+d}.", ""]
    if YC in K and YS in K:
        c, x, nx = st["arms"][YC], st["arms"][YS], net(YS, YC)
        i, ii, iii = bool(x["zeros"] < c["zeros"] and x["at-fault"] < c["at-fault"]), bool(x["at-fault"] <= c["at-fault"]), bool(nx > nb)
        st["B"]["lines"] = dict(i=i, ii=ii, iii=iii, holds=bool(i and ii and iii), net=nx, null=nb)
        L += [f"- (i) fewer seed-mean zeros in total and fewer at-fault zeros than the control: {word[i]}. Zeros {x['zeros']:g} against {c['zeros']:g}, at-fault zeros "
              f"{x['at-fault']:g} against {c['at-fault']:g}.",
              f"- (ii) hard gate, at-fault zeros not above the control's: **{'passes' if ii else 'NEGATIVE (fails)'}**. {x['at-fault']:g} against {c['at-fault']:g}.",
              f"- (iii) fixed - broken against the control exceeds the null: {word[iii]}. {nx:+d} against {nb:+d}.", "",
              f"**\"Holds on the YR base\": {'yes' if i and ii and iii else 'no'}** ((i) {'yes' if i else 'no'}, (ii) {'yes' if ii else 'no'}, (iii) {'yes' if iii else 'no'}).", ""]
    else:
        st["B"]["lines"] = None
        L += ["Lines (i) - (iii): **not evaluated** (YSELFnoL-ON or its control is not read).", ""]
    if YS in K and YN in K:
        d = st["diff"][f"{YS} - {YN}"]
        L += [f"Beside it, YSELFnoL-ON - YNONE-ON (rows against steps): mean score {O.fs(d['score'])}; zeros {O.fq(d['zeros'])}; slow scenes {O.fq(d['slow'])}; progress "
              f"{O.fs(d['progress'])}; GT distance {O.fs(d['gt_dist'])} m; corridor zeros {O.fq(d['corridor'])}; scenes turning more than 45 deg ({int(g45.sum())}): mean score "
              f"{O.fs(d['gt45']['score'])}, zeros {O.fq(d['gt45']['zeros'])}, corridor zeros {O.fq(d['gt45']['corridor'])}. Fixed - broken {net(YS, YN):+d}.", ""]
    L += ["Two seeds per arm, development scenes, no multiplicity correction.", ""]

    # ---------------------------------------------------------------- C
    L += ["## C. Seeds 2 and 3 of `SSELFnoL-ON` on P2H10S-F", ""] + arm_rows([C4, S4, C01, S01, C23, S23]) + [""]
    L += ["Each seed alone (the arm's seed against the control seed it starts from):", "",
          "| seed | mean score arm / control | zeros arm / control | collision / offroad / corridor arm | ... control | at-fault zeros arm / control | fewer zeros |", "|:--|:--|:--|:--|:--|:--|:--|"]
    each = {}
    for s in range(4):
        x, y = REC[S4][s], REC[C4][s]
        if x in A and y in A:
            zx, zy = int(A[x].z().sum()), int(A[y].z().sum())
            each[s] = st["seeds"][f"s{s}"] = dict(arm=x, control=y, mean=[float(A[x].sc.mean()), float(A[y].sc.mean())], zeros=[zx, zy], atfault=[int(A[x].z(CLS[:2]).sum()), int(A[y].z(CLS[:2]).sum())],
                                                  kinds={t: [int(A[t].z((c,)).sum()) for c in CLS] for t in (x, y)}, fewer=bool(zx < zy))
            e = each[s]
            L.append(f"| s{s}{' (new)' if s > 1 else ''} | {e['mean'][0]:.4f} / {e['mean'][1]:.4f} | {zx} / {zy} | {' / '.join(map(str, e['kinds'][x]))} | {' / '.join(map(str, e['kinds'][y]))} | "
                     f"{e['atfault'][0]} / {e['atfault'][1]} | {'yes' if e['fewer'] else 'no'} |")
        else:
            L.append(f"| s{s} | not read ({', '.join(t for t in (x, y) if t not in A)}) | | | | | |")
    pc = [(S4, C4), (S23, C23), (S01, C01)]
    L += ["", "Paired differences:", ""] + pair_rows(pc) + ["", "Fixed and broken scenes:", ""] + flip_rows(pc) + [""]
    if n4 is not None:
        st["C"]["null"] = dict(values=n4, max=max(n4))
        L += [f"Null of fixed - broken (the control against itself, three 2-against-2 splits, both directions): {', '.join(f'{v:+d}' for v in n4)}; maximum {max(n4):+d}.", ""]
    if S4 in K and C4 in K:
        c, x, d = st["arms"][C4], st["arms"][S4], st["diff"][f"{S4} - {C4}"]["zeros"]
        n23, n44 = net(S23, C23), net(S4, C4)
        sign, per, two = bool(d["point"] < 0 and d["scene"][1] < 0), bool(each[2]["fewer"] and each[3]["fewer"]), bool(st["arms"][S23]["zeros"] < st["arms"][C23]["zeros"])
        firm = bool(sign and per)
        st["C"]["lines"] = dict(zeros=[x["zeros"], c["zeros"]], atfault=[x["at-fault"], c["at-fault"]], H=bool(x["at-fault"] <= c["at-fault"]), net_s23=n23, net_all=n44, null=max(n4),
                                four_seed_zeros=d, sign_with_ci=sign, new_seeds_each_fewer=per, new_seeds_mean_fewer=two, firms_up=firm)
        L += [f"- Zeros, four seeds: {x['zeros']:g} against the control's {c['zeros']:g}; at-fault zeros {x['at-fault']:g} against {c['at-fault']:g} (H: "
              f"**{'passes' if x['at-fault'] <= c['at-fault'] else 'NEGATIVE (fails)'}**).",
              f"- Fixed - broken: arm s2, s3 against control s2, s3 {n23:+d}; all four against all four {n44:+d}; null maximum {max(n4):+d} "
              f"({'exceeded' if n23 > max(n4) else 'not exceeded'} by s2, s3; {'exceeded' if n44 > max(n4) else 'not exceeded'} by the four seeds).",
              f"- Four-seed paired difference in zeros: {O.fq(d)}: {'negative with a scene CI excluding 0' if sign else 'not negative with a scene CI excluding 0'}.",
              f"- Seeds 2 and 3 alone: s2 {each[2]['zeros'][0]} against {each[2]['zeros'][1]}, s3 {each[3]['zeros'][0]} against {each[3]['zeros'][1]}: "
              f"{'each has fewer zeros than its control seed' if per else 'not each has fewer zeros than its control seed'} (two-seed means {st['arms'][S23]['zeros']:g} against "
              f"{st['arms'][C23]['zeros']:g}: {'fewer' if two else 'not fewer'}" + ("; **the per-seed and the two-seed-mean readings of the prereg's clause disagree**" if per != two else "") + ").", "",
              "**Decision 259 firms up.**" if firm else "**Decision 259 is demoted in place** (the registered condition is not met).", ""]
    else:
        st["C"]["lines"] = None
        L += ["Lines of section C: **not evaluated** (SSELFnoL-ON s0-s3 or the control's four seeds are not all read).", ""]

    with (out / "on2_per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "turn4s_deg"] + [f"{t}|{c}" for t in have for c in ("score", "zero_class", "atfault_events", "progress", "gt_dist")])
        for i, s in enumerate(scenes):
            w.writerow([s, logs[i], "" if np.isnan(turn[i]) else round(float(turn[i]), 1)] +
                       [x for t in have for x in (round(float(A[t].sc[i]), 6), A[t].cls[i], A[t].ev[i], round(float(A[t].prog[i]), 4), round(float(GT[t][i]), 4))])
    rc = 0
    if a.standin:
        bad, rows = selfcheck(st, A, scenes, name)
        st["selfcheck"] = dict(rows=rows, mismatches=bad)
        L += ["## Stand-in self-check against results/seeds4/seeds4_per_scene.csv (seeds4_report.py's table, same scenes, direct expressions)", "",
              "| quantity | this reader | from the table | equal |", "|:--|:--|:--|:--|"] + [f"| {q} | {x} | {y} | {'yes' if e else '**NO**'} |" for q, x, y, e in rows]
        L += ["", f"Mismatches: {len(bad)}." + (f" {bad}" if bad else "")]
        rc = 3 if bad else 0
    (out / "on2_stats.json").write_text(json.dumps(st, indent=1, default=float))
    (out / "on2_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    _sys.exit(rc)


def selfcheck(st, A, scenes, name):
    """Recompute from seeds4_per_scene.csv (the stand-in's scenes only), without this file's helpers -> (mismatches, table rows)."""
    keep, T = set(scenes), {}
    with (E / "results/seeds4/seeds4_per_scene.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["scene"] in keep:
                T[r["scene"]] = r
    sc = {t: np.array([float(T[s][f"{name(t)}|score"]) for s in scenes]) for t in A}
    cl = {t: np.array([T[s][f"{name(t)}|zero_class"] for s in scenes]) for t in A}
    z = lambda t, c=CLS: np.isin(cl[t], c).astype(float)  # noqa: E731
    rows, bad = [], []

    def add(q, x, y, eq=None):
        e = (x == y) if eq is None else eq
        rows.append((q, x, y, bool(e)))
        if not e:
            bad.append(q)
    add("scenes in the table", len(T), len(scenes))
    add("per-scene scores, max abs difference over all tags (table rounds to 1e-6)", f"{max(float(np.abs(A[t].sc - sc[t]).max()) for t in A):.1e}", "<= 5.0e-07",
        all(np.abs(A[t].sc - sc[t]).max() <= 5e-7 for t in A))
    add("per-scene zero classes that differ, all tags", int(sum((A[t].cls != cl[t]).sum() for t in A)), 0)
    for k, r in st["arms"].items():
        add(f"{k}: seed-mean zeros / at-fault zeros", f"{r['zeros']:g} / {r['at-fault']:g}",
            f"{np.mean([z(t).sum() for t in REC[k]]):g} / {np.mean([z(t, CLS[:2]).sum() for t in REC[k]]):g}")
    for r in st["flips"]:
        if r["kinds"] == "zeros":
            nx, ny = sum(z(t) for t in REC[r["arm"]]), sum(z(t) for t in REC[r["ref"]])
            add(f"{r['arm']} vs {r['ref']}: fixed / broken", f"{r['fixed']} / {r['broken']}",
                f"{int(((ny == len(REC[r['ref']])) & (nx == 0)).sum())} / {int(((ny == 0) & (nx == len(REC[r['arm']]))).sum())}")
    for q, d in st["diff"].items():
        x, y = q.split(" - ")
        add(f"{q}: zeros difference, mean score difference", f"{d['zeros']['point']:+.2f}, {d['score']['point']:+.5f}",
            f"{float((np.mean([z(t) for t in REC[x]], 0) - np.mean([z(t) for t in REC[y]], 0)).sum()):+.2f}, "
            f"{float((np.mean([sc[t] for t in REC[x]], 0) - np.mean([sc[t] for t in REC[y]], 0)).mean()):+.5f}")
    if st["B"].get("null"):
        t0, t1 = REC[YC]
        add("B null: control s0 against s1, fixed - broken", f"{st['B']['null']['control_s0_vs_s1']['net']:+d}", f"{int(((z(t1) == 1) & (z(t0) == 0)).sum()) - int(((z(t1) == 0) & (z(t0) == 1)).sum()):+d}")
    return bad, rows


if __name__ == "__main__":
    main()
