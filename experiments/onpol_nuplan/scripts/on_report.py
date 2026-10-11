#!/usr/bin/env python3
"""ONPOL-NUPLAN read-out (plans/2026-10-11-onpol-prereg.md, "Read") of the `held` runs of on_read.sh: every table of the prereg and its hard gate
and verdict sentences, evaluated mechanically.

  .venv/bin/python experiments/onpol_nuplan/scripts/on_report.py [--out experiments/onpol_nuplan/results/read] [--stored-control 2 3]      (box)
  .venv/bin/python experiments/onpol_nuplan/scripts/on_report.py --standin [--out experiments/onpol_nuplan/results/standin]
Writes <out>/{on_report.md, on_stats.json, on_per_scene.csv}.

Sources: $DATA_DIR/runs/onpol_nuplan/read/held/manifest.json {tag: [run dirs]} on results/split/held.txt (460 scenes). A tag without all scenes is
left out with a line saying so, and so is every arm, pair and sentence that needs it. --stored-control takes the named control seeds from their
stored reads (joint1_report.sources(): BODY1 + lane nuplan1491) instead of a rerun: the prereg's last budget cut, to be used only if the reruns
of s0 / s1 reproduce their stored reads with no zero flip (the reproducibility section says whether they do).
--standin is the test of this file and reads no held scene: the stored reads of the TRAIN-split scenes (results/split/train.txt) play the
read, P2H10S-F s0-s3 as the control and STANDIN's stored recipes as the arms; records of other scenes are dropped as the run dirs are loaded.
It ends with a self-check against experiments/alpasim/results/joint1/joint1_per_scene.csv (the table joint1_report.py wrote) restricted to the
same scenes: per-scene scores and zero classes, fixed / broken counts by joint1_report's expressions and its CIs; exit 3 if anything differs.

Definitions (helpers of c0b_report.py / joint1_report.py, as the prereg says):
  zero kinds     collision (at fault) / offroad / corridor = the first set flag of a score-0 rollout (c0b_report.zclass); at-fault zeros =
                 collision + offroad; zeros = the three kinds. Route-fold zeros (score 0, no flag, no drive call) are in none of them.
  seed mean      of a count: the mean over the recipe's seeds of the per-seed count; of a score: the per-scene mean over the seeds
  paired diff    A - B of per-scene seed means. Mean score: mean over scenes; zero counts: sum over scenes. CIs 95 %, 10 000 draws, seed 0:
                 scene bootstrap (jevdrive.stats.bootstrap / joint1_report.sboot, the same draw) and log-clustered (c0b_report.ci)
  fixed, broken  of X against Y on the three kinds: fixed = zero in every seed of Y and in no seed of X, broken = the reverse
  null           fixed - broken of the control against itself: the three 2-against-2 splits of s0-s3, each in both directions (6 values)
  H              seed-mean at-fault zeros of the arm <= the control's four-seed mean (above = the arm is a negative)
  verdicts       module function verdicts(); every sentence is printed with the numbers that decide it
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[2] / "body1" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import c0b_report as R  # noqa: E402
import joint1_report as J  # noqa: E402
import stop_report as SR  # noqa: E402
from jevdrive import stats  # noqa: E402

E = _H.parents[1]
SPLIT = E / "results/split"
CLS, SHORT = J.CLS, J.SHORT
KIND = {"zeros": CLS, "at-fault": CLS[:2], "collision": CLS[:1], "offroad": CLS[1:2], "corridor": CLS[2:]}
FAM = {"ON-IMIT": "SIMIT-ON", "ON-SELF": "SSELF-ON", "ON-BOTH": "SBOTH-ON", "ON-NONE": "SNONE-ON", "ON-SELFnoL": "SSELFnoL-ON"}
CTRL, C4, C2 = "P2H10S-F", "control", "control (s0, s1)"
STANDIN = {"SIMIT-ON": "P2H10-F", "SSELF-ON": "JAPSnoA", "SBOTH-ON": "JAPS", "SNONE-ON": "JAPSnoY"}      # stored recipes that play the arms in --standin
SPLITS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def load(dirs, keep):
    """Run dirs -> (records, scenes whose runs disagree, drive calls) of the scenes in `keep` only; every other record is dropped here."""
    rec, mism, _, nd = R.load_driver(dirs)
    return {s: r for s, r in rec.items() if s in keep}, sorted(set(mism) & keep), {s: n for s, n in nd.items() if s in keep}


def flips(A, x, y, cl=CLS):
    """Tags x against tags y (the reference): fixed / broken / partial counts and the scene indices (module docstring)."""
    nx, ny = sum(A[t].z(cl) for t in x), sum(A[t].z(cl) for t in y)
    fx, br = (ny == len(y)) & (nx == 0), (ny == 0) & (nx == len(x))
    fa, fb = nx / len(x), ny / len(y)
    return dict(fixed=int(fx.sum()), broken=int(br.sum()), net=int(fx.sum()) - int(br.sum()), partial_better=int(((fa < fb) & ~fx).sum()),
                partial_worse=int(((fa > fb) & ~br).sum()), ref_zeros=float(fb.sum()), arm_zeros=float(fa.sum()), fx=np.flatnonzero(fx), br=np.flatnonzero(br))


def pair(G, a, b, logs, mask=None):
    """Paired difference a - b of two groups' per-scene seed means -> {quantity: point, scene CI, log CI}."""
    m = np.ones(len(logs), bool) if mask is None else mask
    d = G[a]["sc"][m] - G[b]["sc"][m]
    s, l = stats.bootstrap(d), R.ci(d, logs[m])
    out = dict(score=dict(point=float(d.mean()), scene=[s["lo"], s["hi"]], logs=[l[1], l[2]]))
    for k in KIND:
        d = G[a][k][m] - G[b][k][m]
        s, l = J.sboot(d), R.ci(d, logs[m])
        out[k] = dict(point=s[0], scene=[s[1], s[2]], logs=[l[1] * len(d), l[2] * len(d)])
    return out


def fs(q):
    return f"{q['point']:+.4f} [{q['scene'][0]:+.4f}, {q['scene'][1]:+.4f}] ({q['logs'][0]:+.4f}, {q['logs'][1]:+.4f})"


def fq(q):
    return f"{q['point']:+.1f} [{q['scene'][0]:+.1f}, {q['scene'][1]:+.1f}] ({q['logs'][0]:+.1f}, {q['logs'][1]:+.1f})"


def verdicts(st):
    """The hard gate and the four sentences of the prereg from the numbers in st -> ({arm: H}, [(sentence, holds or None, numbers)])."""
    ar, null = st["arms"], st["null"]
    cz = ar[C4]["at-fault"] if C4 in ar else None
    net = {k: v["net"] for k, v in st["net_vs_control01"].items()}
    nmax = max(null["values"]) if null else None
    H = {k: dict(atfault=v["at-fault"], control=cz, passes=bool(v["at-fault"] <= cz)) for k, v in ar.items() if k in FAM and cz is not None}
    out = []

    def beats(x, y):
        need = [k for k in (x, y) if k not in ar]
        if need or cz is None or nmax is None or x not in net:
            return None, f"cannot be evaluated: missing {need or 'the control (four seeds, or s0 / s1)'}"
        fewer = {k: bool(ar[x][k] < ar[y][k]) for k in ("collision", "offroad", "corridor", "at-fault")}
        ok = all(fewer.values()) and H[x]["passes"] and net[x] > nmax
        return ok, (f"seed-mean zeros {x} / {y}: " + ", ".join(f"{k} {ar[x][k]:g} / {ar[y][k]:g} ({'fewer' if v else 'not fewer'})" for k, v in fewer.items())
                    + f"; H: {x} at-fault {ar[x]['at-fault']:g} against the control's {cz:g} ({'passes' if H[x]['passes'] else 'fails'}); fixed - broken of {x} against "
                    f"{C2}: {net[x]:+d} against the null's maximum {nmax:+d} ({'exceeds' if net[x] > nmax else 'does not exceed'})")
    for x, y, name in (("ON-SELF", "ON-IMIT", "Self beats imitation"), ("ON-IMIT", "ON-SELF", "Imitation beats self")):
        out.append((name, *beats(x, y)))
    need = [k for k in ("ON-BOTH", "ON-IMIT", "ON-SELF") if k not in ar]
    if need or cz is None or nmax is None or "ON-BOTH" not in net:
        out.append(("Adds", None, f"cannot be evaluated: missing {need or 'the control (four seeds, or s0 / s1)'}"))
    else:
        b = ar["ON-BOTH"]
        f1 = {y: bool(b["at-fault"] < ar[y]["at-fault"]) for y in ("ON-IMIT", "ON-SELF")}
        f2 = {y: bool(b["zeros"] < ar[y]["zeros"]) for y in ("ON-IMIT", "ON-SELF")}
        ok = H["ON-BOTH"]["passes"] and all(f1.values()) and all(f2.values()) and net["ON-BOTH"] > nmax
        out.append(("Adds", ok, f"H: ON-BOTH at-fault {b['at-fault']:g} against the control's {cz:g} ({'passes' if H['ON-BOTH']['passes'] else 'fails'}); at-fault zeros ON-BOTH "
                    f"{b['at-fault']:g} against ON-IMIT {ar['ON-IMIT']['at-fault']:g} / ON-SELF {ar['ON-SELF']['at-fault']:g} ({'fewer than each' if all(f1.values()) else 'not fewer than each'}); "
                    f"zeros in total {b['zeros']:g} against {ar['ON-IMIT']['zeros']:g} / {ar['ON-SELF']['zeros']:g} ({'fewer than each' if all(f2.values()) else 'not fewer than each'}); "
                    f"fixed - broken against {C2}: {net['ON-BOTH']:+d} against the null's maximum {nmax:+d} ({'exceeds' if net['ON-BOTH'] > nmax else 'does not exceed'})"))
    if any(v is None for _, v, _ in out):
        out.append(("Not separated", None, "cannot be evaluated: a sentence above cannot"))
    else:
        d = st["diff"].get("ON-SELF - ON-IMIT")
        out.append(("Not separated", not any(v for _, v, _ in out), "none of the three sentences holds" if not any(v for _, v, _ in out) else "a sentence above holds"))
        if d:
            out[-1] = (*out[-1][:2], out[-1][2] + f"; ON-SELF - ON-IMIT [CI scenes] (CI logs): zeros in total {fq(d['zeros'])}, at-fault zeros {fq(d['at-fault'])}, mean score {fs(d['score'])}")
    return H, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--standin", action="store_true", help="test on stored reads of train-split scenes; reads no held scene")
    ap.add_argument("--stored-control", type=int, nargs="*", default=[], help="control seeds taken from their stored reads instead of a rerun")
    ap.add_argument("--manifest", default=str(J.DATA / "runs/onpol_nuplan/read/held/manifest.json"))
    a = ap.parse_args()
    out = _pl.Path(a.out or E / "results" / ("standin" if a.standin else "read"))
    out.mkdir(parents=True, exist_ok=True)
    split = "train" if a.standin else "held"
    scenes = sorted((SPLIT / f"{split}.txt").read_text().split())
    keep = set(scenes)
    assert not keep & set((SPLIT / ("held.txt" if a.standin else "train.txt")).read_text().split()), "train and held overlap"
    stored = J.sources(json.loads((_H.parents[2] / "alpasim/results/joint1/joint1_checks.json").read_text())["family_ok"])       # JOINT-1's recorded checks
    ctl = [f"{CTRL}-s{s}" for s in range(4)]
    rec = {k: [f"{v}-s0", f"{v}-s1"] for k, v in FAM.items()} | {C4: ctl, C2: ctl[:2]}
    if a.standin:
        src = {t: stored.get(t, []) for t in ctl} | {f"{f}-s{s}": stored.get(f"{STANDIN[f]}-s{s}", []) for f in STANDIN for s in (0, 1)}
    else:
        man = J.man(a.manifest)
        assert man, f"{a.manifest}: no manifest (run on_read.sh held first)"
        src = {t: man.get(t, []) for v in rec.values() for t in v} | {f"{CTRL}-s{s}": stored.get(f"{CTRL}-s{s}", []) for s in a.stored_control}
    D = {t: load(d, keep) for t, d in src.items() if d}
    have = sorted(t for t, d in D.items() if all(s in d[0] for s in scenes))
    miss = {t: (len(keep - set(D[t][0])) if t in D else len(scenes)) for t in src if t not in have}
    A = {t: J.Arm(D[t][0], D[t][2], scenes) for t in have}
    K = [k for k, v in rec.items() if all(t in A for t in v)]
    logs = np.array([R.log_of(s) for s in scenes])
    T = SR.turns()
    turn = np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in scenes])
    g45 = np.abs(turn) > 45
    G = {k: dict(sc=np.mean([A[t].sc for t in rec[k]], 0), **{q: np.mean([A[t].z(c) for t in rec[k]], 0) for q, c in KIND.items()}) for k in K}
    n = len(scenes)
    title = ("# ONPOL-NUPLAN read-out: STAND-IN TEST on stored reads of train-split scenes (not a result)" if a.standin else "# ONPOL-NUPLAN read-out on the held scenes")
    L = [title + " (generated by scripts/on_report.py; do not edit)", ""]
    if a.standin:
        L += ["Test of the reader only. Control = stored P2H10S-F s0-s3; arms played by stored recipes: " + ", ".join(f"{k} <- {STANDIN[v]}" for k, v in FAM.items() if v in STANDIN)
              + ". These scenes were read by all of them before; the verdict sentences below are mechanics, not findings.", ""]
    L += [f"{n} scenes of `alpasim_nuplan/onpol-{split}`, {len(set(logs))} logs, {int(g45.sum())} turning more than 45 deg (navtest token's logged 4 s heading change; "
          f"{int(np.isnan(turn).sum())} without one). Tags with all scenes: {', '.join(have) or 'none'}.", ""]
    L += [f"**Skipped, not all scenes in the run dirs (scenes missing): {miss}.**", ""] if miss else []
    gone = [k for k in rec if k not in K]
    L += [f"**Not read: {', '.join(gone)}; every table, pair and sentence below that needs one of them leaves it out or says so.**", ""] if gone else []
    if a.stored_control and not a.standin:
        L += [f"Control seeds {a.stored_control} are their stored reads (BODY1 / lane nuplan1491), not reruns of this lane.", ""]
    mism = {t: D[t][1] for t in have if D[t][1]}
    L += [f"Scores that differ between two runs of one tag: {mism}.", ""] if mism else []
    st = dict(standin=a.standin, split=split, scenes=n, logs=len(set(logs)), gt45=int(g45.sum()), tags={k: rec[k] for k in K}, missing=miss, not_read=gone,
              sources={t: [str(x) for x in d] for t, d in src.items()}, arms={}, diff={}, flips=[], net_vs_control01={}, null={})

    # ---- per arm
    L += ["## Per arm (score = per-scene mean of the seeds; counts = means over the seeds of the per-seed counts)", "",
          "| arm | tags | per-seed mean score | seed-mean score [95 % CI, logs] | zeros | collision / offroad / corridor | at-fault zeros | per-seed zeros (collision / offroad / corridor) | "
          "route-fold zeros | at-fault events | slow | progress | > 45 deg: mean, zeros (collision / offroad / corridor) |", "|:--|:--|:--|:--|--:|:--|--:|:--|--:|--:|--:|--:|:--|"]
    for k in K:
        m, ts = R.ci(G[k]["sc"], logs), rec[k]
        mean = lambda f: float(np.mean([f(A[t]) for t in ts]))  # noqa: E731
        r = dict(tags=ts, mean=m, seeds=[float(A[t].sc.mean()) for t in ts], **{q: float(G[k][q].sum()) for q in KIND}, fold=mean(lambda x: x.fold.sum()),
                 per_seed={t: {SHORT[c]: int(A[t].z((c,)).sum()) for c in CLS} for t in ts}, events=mean(lambda x: x.ev.sum()),
                 slow=mean(lambda x: ((x.sc > 0) & (x.sc < 1)).sum()), progress=mean(lambda x: x.prog.mean()),
                 gt45=dict(n=int(g45.sum()), mean=float(G[k]["sc"][g45].mean()), **{q: float(G[k][q][g45].sum()) for q in KIND}))
        st["arms"][k] = r
        g = r["gt45"]
        L.append(f"| {k} | {' + '.join(ts)} | {' / '.join(f'{x:.4f}' for x in r['seeds'])} | {R.fmt(m)} | {r['zeros']:g} | {r['collision']:g} / {r['offroad']:g} / {r['corridor']:g} | "
                 f"{r['at-fault']:g} | {'; '.join('/'.join(str(v) for v in d.values()) for d in r['per_seed'].values())} | {r['fold']:g} | {r['events']:g} | {r['slow']:g} | "
                 f"{r['progress']:.4f} | {g['mean']:.4f}, {g['zeros']:g} ({g['collision']:g} / {g['offroad']:g} / {g['corridor']:g}) |")

    # ---- paired differences
    arms = [k for k in FAM if k in K]
    pairs = ([(x, c) for c in (C4, C2) if c in K for x in arms] + [(x, "ON-NONE") for x in arms if x != "ON-NONE" and "ON-NONE" in K]
             + [(x, y) for x, y in (("ON-SELF", "ON-IMIT"), ("ON-BOTH", "ON-IMIT"), ("ON-BOTH", "ON-SELF"), ("ON-SELF", "ON-SELFnoL"), ("ON-BOTH", "ON-SELFnoL"), ("ON-IMIT", "ON-SELFnoL"))
                if x in K and y in K])
    L += ["", "## Paired differences (A - B of per-scene seed means; point [95 % CI scenes] (CI logs), 10 000 draws; score = mean over scenes, zeros = sum over scenes)", "",
          "| A - B | mean score | zeros | at-fault zeros | collision | offroad | corridor | > 45 deg: zeros | > 45 deg: corridor |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for x, y in pairs:
        d, d45 = pair(G, x, y, logs), pair(G, x, y, logs, g45)
        st["diff"][f"{x} - {y}"] = d | {"gt45": d45}
        L.append(f"| {x} - {y} | {fs(d['score'])} | " + " | ".join(fq(d[q]) for q in KIND) + f" | {fq(d45['zeros'])} | {fq(d45['corridor'])} |")

    # ---- fixed and broken
    fp = ([(x, c) for c in (C2, C4) if c in K for x in arms] + [(x, y) for x, y in pairs if y not in (C4, C2)])
    L += ["", "## Fixed and broken scenes (A against reference B: fixed = zero in every seed of B and in no seed of A, broken = the reverse; partial = other changes)", "",
          "| A vs B | kinds | fixed | broken | fixed - broken | partial (better / worse) | zeros B -> A |", "|:--|:--|--:|--:|--:|:--|:--|"]
    for x, y in fp:
        for q, cl in KIND.items():
            f = flips(A, rec[x], rec[y], cl)
            row = dict(arm=x, ref=y, kinds=q, **{k: v for k, v in f.items() if k not in ("fx", "br")}, fixed_scenes=[scenes[i] for i in f["fx"]], broken_scenes=[scenes[i] for i in f["br"]])
            st["flips"].append(row)
            if q == "zeros" and y == C2:
                st["net_vs_control01"][x] = row
            L.append(f"| {x} vs {y} | {q} | {f['fixed']} | {f['broken']} | {f['net']:+d} | {f['partial_better']} / {f['partial_worse']} | {f['ref_zeros']:g} -> {f['arm_zeros']:g} |")
    L += ["", "Scenes (zeros of the three kinds; per tag the kind of its zero, `-` = not a zero):", ""]
    for row in st["flips"]:
        if row["kinds"] == "zeros" and (row["fixed_scenes"] or row["broken_scenes"]):
            ts = rec[row["ref"]] + rec[row["arm"]]
            for w in ("fixed", "broken"):
                L += [f"- {row['arm']} vs {row['ref']}, {w} ({len(row[w + '_scenes'])}): " + "; ".join(
                    f"`{s}` ({' '.join(SHORT.get(A[t].cls[scenes.index(s)], '-') for t in ts)})" for s in row[w + "_scenes"])]
    if any(r["kinds"] == "zeros" for r in st["flips"]):
        L += ["", "Tag order inside the brackets: the reference's seeds, then A's."]

    # ---- null
    if C4 in K:
        vals, rows = [], []
        for p, q in SPLITS:
            for x, y in ((p, q), (q, p)):
                f = {k: flips(A, [ctl[i] for i in x], [ctl[i] for i in y], cl) for k, cl in KIND.items()}
                vals.append(f["zeros"]["net"])
                rows.append(dict(arm=[ctl[i] for i in x], ref=[ctl[i] for i in y], **{k: dict(fixed=v["fixed"], broken=v["broken"], net=v["net"]) for k, v in f.items()}))
        st["null"] = dict(values=vals, max=max(vals), rows=rows)
        L += ["", "## Null of fixed - broken (the control against itself: s{i,j} as A against the other two seeds as B)", "",
              "| A | B | zeros: fixed / broken / fixed - broken | at-fault: fixed / broken / fixed - broken | corridor: fixed / broken / fixed - broken |", "|:--|:--|:--|:--|:--|"]
        for r in rows:
            L.append(f"| {' + '.join(t[-2:] for t in r['arm'])} | {' + '.join(t[-2:] for t in r['ref'])} | " + " | ".join(
                f"{r[k]['fixed']} / {r[k]['broken']} / {r[k]['net']:+d}" for k in ("zeros", "at-fault", "corridor")) + " |")
        L += ["", f"Null values (zeros of the three kinds): {', '.join(f'{v:+d}' for v in vals)}; maximum {max(vals):+d}."]

    # ---- gate and verdicts
    H, V = verdicts(st)
    st["H"], st["verdicts"] = H, [dict(sentence=s, holds=v, numbers=t) for s, v, t in V]
    word = {True: "**holds**", False: "does not hold", None: "**not evaluated**"}
    L += ["", "## Hard gate H and the registered sentences", ""]
    L += [f"- H, {k}: seed-mean at-fault zeros {h['atfault']:g} against the control's four-seed mean {h['control']:g} -> **{'passes' if h['passes'] else 'NEGATIVE (fails H)'}**"
          for k, h in H.items()] or ["- H: not evaluated (the control's four seeds are not all read)"]
    L += [""] + [f"- \"{s}\": {word[v]}. {t}." for s, v, t in V]
    hold = [s for s, v, _ in V if v]
    st["verdict"] = None if any(v is None for _, v, _ in V) else hold
    L += ["", "Verdict: " + ("not evaluated (arms missing)" if st["verdict"] is None else " and ".join(f"\"{s}\"" for s in hold)) + "."]

    # ---- reproducibility of the control reruns
    if not a.standin:
        L += ["", "## Reproducibility: control reruns of this lane against the stored reads of the same scenes (BODY1 / lane nuplan1491 run dirs)", "",
              "| tag | scenes compared | scenes that differ (score or zero kind) | zero flips | max abs score difference | mean rerun / stored |", "|:--|--:|--:|--:|--:|:--|"]
        st["repro"] = {}
        for t in [x for x in ctl if x in A and x not in [f"{CTRL}-s{s}" for s in a.stored_control]]:
            b = load(stored.get(t, []), keep)[0]
            sc = [s for s in scenes if s in b]
            da = {s: D[t][0][s] for s in sc}
            dif = [s for s in sc if abs(da[s]["score"] - b[s]["score"]) > 1e-9 or R.zclass(da[s]) != R.zclass(b[s])]
            flip = [s for s in sc if (da[s]["score"] == 0) != (b[s]["score"] == 0)]
            mx = max((abs(da[s]["score"] - b[s]["score"]) for s in sc), default=float("nan"))
            st["repro"][t] = dict(compared=len(sc), differ=dif, zero_flips=flip, max_abs=mx, mean_rerun=float(np.mean([da[s]["score"] for s in sc])) if sc else None,
                                  mean_stored=float(np.mean([b[s]["score"] for s in sc])) if sc else None)
            L.append(f"| {t} | {len(sc)} of {n} | {len(dif)} | {len(flip)} | {mx:.4f} | " + (f"{st['repro'][t]['mean_rerun']:.4f} / {st['repro'][t]['mean_stored']:.4f} |" if sc else "n/a |"))
        for t, r in st["repro"].items():
            L += [f"\n{t}, scenes that differ: " + ", ".join(f"`{s}`" + (" (zero flip)" if s in r["zero_flips"] else "") for s in r["differ"])] if r["differ"] else []
        if not st["repro"]:
            L.append("| (no rerun of the control in the run dirs) | | | | | |")

    # ---- files
    with (out / "on_per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "turn4s_deg"] + [f"{t}|{c}" for t in have for c in ("score", "zero_class", "atfault_events", "progress")])
        for i, s in enumerate(scenes):
            w.writerow([s, logs[i], "" if np.isnan(turn[i]) else round(float(turn[i]), 1)] +
                       [x for t in have for x in (round(float(A[t].sc[i]), 6), A[t].cls[i], A[t].ev[i], round(float(A[t].prog[i]), 4))])
    rc = 0
    if a.standin:
        bad, rows = selfcheck(st, A, scenes, logs, rec, K)
        st["selfcheck"] = dict(rows=rows, mismatches=bad)
        L += ["", "## Stand-in self-check against joint1_per_scene.csv (joint1_report.py's table, same scenes, its expressions)", "",
              "| quantity | this reader | from joint1_report's table | equal |", "|:--|:--|:--|:--|"] + [f"| {q} | {x} | {y} | {'yes' if e else '**NO**'} |" for q, x, y, e in rows]
        L += ["", f"Mismatches: {len(bad)}." + (f" {bad}" if bad else "")]
        rc = 3 if bad else 0
    (out / "on_stats.json").write_text(json.dumps(st, indent=1, default=float))
    (out / "on_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    _sys.exit(rc)


def selfcheck(st, A, scenes, logs, rec, K):
    """Recompute from joint1_per_scene.csv (train scenes only) with joint1_report.py's own expressions -> (mismatches, table rows)."""
    keep, T = set(scenes), {}
    with (_H.parents[2] / "alpasim/results/joint1/joint1_per_scene.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["scene"] in keep:                                   # rows of other scenes are not parsed
                T[r["scene"]] = r
    name = lambda t: next((f"{STANDIN[f]}{t[len(f):]}" for f in STANDIN if t.startswith(f + "-s")), t)  # noqa: E731  stand-in tag -> stored tag
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
    f1 = lambda t: f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"  # noqa: E731
    for x in [k for k in FAM if k in K]:
        for y in (C2, "ON-NONE"):
            if y not in K or x == y:
                continue
            for q, c in (("zeros", CLS), ("at-fault", CLS[:2]), ("corridor", CLS[2:])):
                nx, ny = sum(z(t, c) for t in rec[x]), sum(z(t, c) for t in rec[y])         # joint1_report.py, "Fixed vs broken"
                ref = (int(((ny == 2) & (nx == 0)).sum()), int(((ny == 0) & (nx == 2)).sum()))
                mine = next(r for r in st["flips"] if (r["arm"], r["ref"], r["kinds"]) == (x, y, q))
                add(f"{x} vs {y}, {q}: fixed / broken", f"{mine['fixed']} / {mine['broken']}", f"{ref[0]} / {ref[1]}")
            zx, zy = (np.mean([z(t) for t in rec[k]], 0) for k in (x, y))                    # joint1_report.py, L1 and L2
            d1, d1l = J.sboot(zx - zy), R.ci(zx - zy, logs)
            d2 = R.ci(np.mean([sc[t] for t in rec[x]], 0) - np.mean([sc[t] for t in rec[y]], 0), logs)
            m = st["diff"][f"{x} - {y}"]
            add(f"{x} - {y}: zeros difference [CI scenes]", f1((m["zeros"]["point"], *m["zeros"]["scene"])), f1(d1))
            add(f"{x} - {y}: zeros difference, CI logs", f"{m['zeros']['logs'][0]:+.1f}, {m['zeros']['logs'][1]:+.1f}", f"{d1l[1] * len(scenes):+.1f}, {d1l[2] * len(scenes):+.1f}")
            add(f"{x} - {y}: mean score difference [CI logs]", f"{m['score']['point']:+.4f} [{m['score']['logs'][0]:+.4f}, {m['score']['logs'][1]:+.4f}]", J.f4(d2))
    return bad, rows


if __name__ == "__main__":
    main()
