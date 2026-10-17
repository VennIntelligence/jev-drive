#!/usr/bin/env python3
"""Regression table and aggregate lines of closed-loop arms on the stored AlpaSim reads (CPU, read-only on stored runs; box, repo root, op-train python).

  nx_regress.py table --tags [LABEL=]TAG ... --ref LABEL [--out NAME] [--dry]
        The standard regression set (REGSET below: the 28 scenes the user named after two rounds of zero-score videos, handoff 2026-10-17 section 3, plus the 12
        public no-launch scenes of explore_f.md; 34 distinct scenes) on every stored read a scene is in (held 460 @18, public 1491 @10). Per tag and row: the score,
        the failure type and progress_rel; per tag other than --ref: the status against --ref (fixed / still wrong / newly wrong / still right, rule below).
        Counts per read, per class. A tag without a stored read gets "-". -> results/next/<NAME>.{json,md}   (the reference columns of a later round are the stored
        RSG-L18-s0 / V2 reads: `--tags RSG=RSG-L18-s0 V2=RTB1ARO-L18-s0 NEW=<tag> --ref V2`)
  nx_regress.py agg --j [LABEL=]TAG --refs [LABEL=]TAG ... [--out NAME] [--dry]
        The aggregate table of one arm against the references (the first one is the line reference, V2 in prereg plans/2026-10-17-v2ac-prereg.md): held 460 @18 and
        public 1491 @10 zeros, mean, paired mean against every other arm with the session-clustered CI, at-fault, offroad, progress_rel, turns /24, static scene,
        stop17, front / side / rear collisions, creep counts; and the five lines of that prereg against the line reference. -> results/next/<NAME>.{json,md}

  nx_regress.py dist --tags [LABEL=]TAG ... --ref LABEL [--out NAME] [--dry]
        Descriptive creep measure (added after the first V2AC read; not a prereg line): per read, over the scenes where the log drives > 5 m, the distance the policy
        drives relative to the log (sum / sum and per-scene mean), the paired per-scene ratio and distance against --ref with the session-clustered CI, and the number of
        scenes under 50 % / 30 % of the log and under 1 m.

Definitions (fixed in code; prereg plans/2026-10-17-v2ac-prereg.md section 4):
  failure type  one rollout's results-summary.json row, checked in this order. score 0: at-fault collision (AF-front, else AF-side, else AF-rear, by which collision_*
                flag is set), offroad, corridor (left_corridor_laterally), joined with "+" when several; no flag: route-fail (map route generation, independent of the
                policy) or zero-other. Score >= 0.98: ok. Otherwise hit (a collision that is not at fault, including a rear-end contact), no-launch (the log drives > 5 m,
                the policy < 2 m), then by the share of the log's distance: barely (< 10 %), crawl (< 30 %), slow.
  status        right = score >= 0.9. Against --ref: fixed = ref wrong, arm right and >= 0.1 higher; newly wrong = ref right, arm wrong and >= 0.1 lower; still wrong = both
                wrong (or a flip across 0.9 smaller than 0.1 from the wrong side); still right otherwise. A flip across 0.9 by less than 0.1 is not a change.
  creep         the log drives > 5 m and the policy less than 30 % of it; no-launch: less than 1 m of it (both from dist_traveled_m / gt_dist_traveled_m).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]
import nx  # noqa: E402
import wt  # noqa: E402

DATA, NX = wt.DATA, nx.NX
PUB_REF = DATA / "runs/route_follow/guard/read10"        # stored public 1491 @10 reads <tag>/b0..b9 of the older arms (RSG-L18-s0 ...)
RIGHT, FLIP, OK_SCORE = 0.9, 0.1, 0.98
READS = ("held", "pub")
READ_NAME = dict(held="held18", pub="pub10")
STATIC_MAX = 1.0                                          # nx.py cont's stop rule: the static scene is stopped at <= 1.0 m without contact
# The user's 28 scenes (8-hex prefix of the clip id's hash), class letters as in the handoff (da53dac7 also belongs to E; it is kept in A), the read the user's
# video came from is irrelevant here: every read a scene is in gets a row.
USER = dict(A="d32f8367 a2839b1b c240593c de4ace24 da53dac7 517e1992", B="9a675656 53d37873 ddc1271e 1f544304 9d1d720d 81a72614 365c0937 a7fa8bcc 6d2b59f7 "
            "0854af02 4cf608d9 d33c6db3 a9609780", C="42c7526f", D="c159f0c3 4211afd0 d93e31df c2790319", E="94652fa0", F="c665621f 1b25edea", G="982b57b1")
# explore_f.md: public 1491 @10 scenes where RTB1O-L18-s0 drives < 1 m while the log drives > 5 m (12 scenes; RSG-L18-s0 0).
NOLAUNCH = "c159f0c3 e29b049a ed829447 4211afd0 982b57b1 1b25edea c665621f f3512ae1 79ecc8e6 d93e31df f4eec162 eeba0804".split()
CLASS_NAME = dict(A="A longitudinal", B="B lateral", C="C route memory", D="D red light", E="E rear-ended", F="F detour", G="G render", N="N public no-launch")


def regset():
    """short id -> sorted class letters (the user's class, then N)."""
    S = {}
    for c, ids in USER.items():
        for i in ids.split():
            S.setdefault(i, []).append(c)
    for i in NOLAUNCH:
        S.setdefault(i, []).append("N")
    assert sum(len(v.split()) for v in USER.values()) == 28 and len(NOLAUNCH) == 12 and len(S) == 34, (len(S), "regression set size")
    return S


def rollouts(dirs):
    """clipgt_id -> the first rollout row per scene over the run dirs (aggregate/results-summary.json)."""
    R = {}
    for d in dirs:
        f = Path(d) / "aggregate/results-summary.json"
        if f.exists():
            for r in json.loads(f.read_text())["rollouts"]:
                R.setdefault(r["clipgt_id"], r)
    return R


def read_dirs(tag):
    """The stored run dirs of a tag per read: held 460 @18 (rm_chain read) and public 1491 @10 (first location that has reads)."""
    pub = [glob.glob(g) for g in (f"{PUB_REF}/{tag}/b*", f"{NX}/cont/wide/pub/{tag}/b*", wt.xdirs("pub", tag), f"{wt.OUT}/read10/{tag}/b*")]
    return dict(held=glob.glob(wt.xdirs("held", tag)), pub=next((g for g in pub if g), []))


def failure_type(r):
    m, sc = r.get("metrics") or {}, float(r.get("score") or 0.0)
    pos = lambda k: (m.get(k) or 0) > 0  # noqa: E731
    if sc == 0:
        z = []
        if pos("collision_at_fault"):
            z.append("AF-front" if pos("collision_front") else "AF-side" if pos("collision_lateral") else "AF-rear")
        z += [n for k, n in (("offroad", "offroad"), ("left_corridor_laterally", "corridor")) if pos(k)]
        return "+".join(z) if z else "route-fail" if str(r.get("failure_reason") or "").startswith("Waypoint") else "zero-other"
    if sc >= OK_SCORE:
        return "ok"
    if pos("collision_any"):
        return "hit"
    dist, gt = float(m.get("dist_traveled_m") or 0.0), float(m.get("gt_dist_traveled_m") or 0.0)
    if gt > 5 and dist < 2:
        return "no-launch"
    ratio = dist / gt if gt > 0 else float("nan")
    return "barely" if ratio < 0.1 else "crawl" if ratio < 0.3 else "slow"


def status(ref, arm):
    """Status of an arm's score against the reference's: fixed / still wrong / newly wrong / still right (see the header)."""
    r0, r1 = ref >= RIGHT, arm >= RIGHT
    if r0 == r1:
        return "still right" if r0 else "still wrong"
    if r1:
        return "fixed" if arm - ref >= FLIP else "still wrong"
    return "newly wrong" if ref - arm >= FLIP else "still right"


def find(R, short):
    ks = [s for s in R if s.rsplit("-", 1)[-1].startswith(short)]
    assert len(ks) <= 1, (short, ks)
    return ks[0] if ks else None


def lab_tag(x):
    return tuple(x.split("=", 1)) if "=" in x else (x, x)


# ---------------------------------------------------------------- table
def cmd_table(a):
    arms = [lab_tag(x) for x in a.tags]
    labels = [l for l, _ in arms]
    assert a.ref in labels, f"--ref {a.ref} is not one of {labels}"
    S, R = regset(), {t: {rd: rollouts(ds) for rd, ds in read_dirs(t).items()} for _, t in arms}
    order = {c: i for i, c in enumerate("ABCDEFGN")}
    ids = sorted(S, key=lambda i: (order[S[i][0]], i))
    rows = []
    for i in ids:
        for rd in READS:
            ref_set = [R[t][rd] for _, t in arms if R[t][rd]]
            s = next((find(x, i) for x in ref_set if find(x, i)), None)
            if s is None:
                continue
            row = dict(short=i, set="+".join(S[i]), read=READ_NAME[rd], scene=s, arms={})
            for l, t in arms:
                r = R[t][rd].get(s)
                m = (r or {}).get("metrics") or {}
                row["arms"][l] = None if r is None else dict(
                    score=float(r.get("score") or 0.0), type=failure_type(r), progress_rel=m.get("progress_rel"),
                    dist=m.get("dist_traveled_m"), gt=m.get("gt_dist_traveled_m"))
            ref = row["arms"][a.ref]
            row["status"] = {l: (None if ref is None or x is None else status(ref["score"], x["score"])) for l, x in row["arms"].items() if l != a.ref}
            rows.append(row)
    others = [l for l in labels if l != a.ref]
    states = ("fixed", "still wrong", "newly wrong", "still right")
    res = dict(tags=dict(arms), ref=a.ref, right=RIGHT, flip=FLIP, rows=rows, counts={}, by_class={})
    for l in others:
        cnt = lambda sel: {s: sum(r["status"][l] == s for r in sel) for s in states}  # noqa: E731
        res["counts"][l] = {READ_NAME[rd]: cnt([r for r in rows if r["read"] == READ_NAME[rd]]) for rd in READS} | dict(total=cnt(rows))
        res["by_class"][l] = {c: cnt([r for r in rows if c in r["set"].split("+")]) for c in "ABCDEFGN"}
    mean = lambda v: float(np.mean(v)) if v else None  # noqa: E731
    res["class_mean"] = {c: dict(rows=sum(c in r["set"].split("+") for r in rows), **{l: mean([r["arms"][l]["score"] for r in rows if c in r["set"].split("+") and r["arms"][l]])
                                                                                        for l in labels}) for c in "ABCDEFGN"}
    res["class_mean"]["all"] = dict(rows=len(rows), **{l: mean([r["arms"][l]["score"] for r in rows if r["arms"][l]]) for l in labels})
    f2 = lambda x: "-" if x is None or x != x else f"{x:.2f}"  # noqa: E731
    L = [f"# Regression set: {', '.join(f'{l} = {t}' for l, t in arms)}; status against {a.ref}", "",
         f"{len(S)} scenes (the user's 28 + the 12 public no-launch scenes of explore_f.md, 6 overlap), one row per scene and stored read (held18 = held 460 @18, pub10 = public 1491 @10). "
         f"Failure type and status rules: header of `scripts/nx_regress.py` (right = score >= {RIGHT}; a change needs >= {FLIP}). progress_rel is the metric as stored.", "",
         "| scene | set | read | " + " | ".join(f"{l} score" for l in labels) + " | " + " | ".join(f"{l} type" for l in labels) + " | " + " | ".join(f"{l} prog" for l in labels)
         + " | " + " | ".join(f"{l} vs {a.ref}" for l in others) + " |", "|---|---|---|" + "--:|" * len(labels) + ":-:|" * len(labels) + "--:|" * len(labels) + ":-:|" * len(others)]
    for r in rows:
        g = lambda f, l: "-" if r["arms"][l] is None else f(r["arms"][l])  # noqa: E731
        L.append(f"| {r['short']} | {r['set']} | {r['read']} | " + " | ".join(g(lambda x: f2(x["score"]), l) for l in labels) + " | " + " | ".join(g(lambda x: x["type"], l) for l in labels)
                 + " | " + " | ".join(g(lambda x: f2(x["progress_rel"]), l) for l in labels) + " | " + " | ".join(r["status"][l] or "-" for l in others) + " |")
    for l in others:
        L += ["", f"## {l} against {a.ref}: counts of scene-reads", "", "| | fixed | still wrong | newly wrong | still right | rows |", "|---|--:|--:|--:|--:|--:|"]
        for k, c in res["counts"][l].items():
            L.append(f"| {k} | " + " | ".join(str(c[s]) for s in states) + f" | {sum(c.values())} |")
        L += ["", "| class | fixed | still wrong | newly wrong | still right | rows |", "|---|--:|--:|--:|--:|--:|"]
        for k, c in res["by_class"][l].items():
            L.append(f"| {CLASS_NAME[k]} | " + " | ".join(str(c[s]) for s in states) + f" | {sum(c.values())} |")
    L += ["", "## Mean score by class (scene-reads; a row counts in every class it belongs to; descriptive, added after the first read)", "",
          "| class | rows | " + " | ".join(labels) + " |", "|---|--:|" + "--:|" * len(labels)]
    for c, v in res["class_mean"].items():
        L.append(f"| {CLASS_NAME.get(c, 'all rows')} | {v['rows']} | " + " | ".join(f2(v[l]) for l in labels) + " |")
    nx.write(a.out, res, L, a.dry)


# ---------------------------------------------------------------- aggregate
def own(M):
    """wt.xpair's per-side summary of one read: zeros, mean, progress_rel, at-fault, offroad (scenes of M)."""
    X = np.array(list(M.values()))
    return dict(scenes=len(M), zeros=int((X[:, 0] == 0).sum()), mean=float(X[:, 0].mean()), progress_rel=float(np.nanmean(X[:, 1])),
                at_fault=int(np.nansum(X[:, 2] > 0)), offroad=int(np.nansum(X[:, 3] > 0)))


def creep(R):
    """Counts over the rollout rows of a read: creep (log > 5 m, policy < 30 % of it) and no-launch (< 1 m)."""
    c = n = 0
    for r in R.values():
        m = r.get("metrics") or {}
        gt, d = float(m.get("gt_dist_traveled_m") or 0.0), float(m.get("dist_traveled_m") or 0.0)
        if gt > 5:
            c += d < 0.3 * gt
            n += d < 1.0
    return dict(creep=int(c), no_launch=int(n))


def cont_of(tag):
    """The stored nx.py cont file of an arm (static scene, stop17, turns /24, arrival, dev ADE), or of a reference (its entries inside another arm's file)."""
    files = sorted(NX.glob("rtb_cont_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    docs = [json.loads(p.read_text()) for p in files]
    d = next((d for d in docs if d["arm"] == tag), None)
    src = d or next((d for d in docs if tag in d["cl"]), None)
    if src is None:
        return None
    h = next((h for h in src["held"].values() if (h.get("turns") or {}).get("completed", {}).get(tag) is not None), None)
    return dict(static=src["cl"].get(tag, {}).get("static"), stop17=src["cl"].get(tag, {}).get("stop17"),
                turns=None if h is None else (h["turns"]["completed"][tag], h["turns"]["turns"][tag]),
                arrival=src["arrival"].get(tag), dev_ade=src["dev_ade"].get(tag))


def cmd_agg(a):
    arms = [lab_tag(a.j)] + [lab_tag(x) for x in a.refs]
    ref = arms[1]
    D = {t: read_dirs(t) for _, t in arms}
    M = {t: {rd: wt.metrics(D[t][rd]) for rd in READS} for _, t in arms}
    Rr = {t: {rd: rollouts(D[t][rd]) for rd in READS} for _, t in arms}
    res = dict(arms={l: t for l, t in arms}, line_ref=ref[0], reads={})
    for rd in READS:
        out = res["reads"][rd] = {}
        for l, t in arms:
            assert M[t][rd], f"{l} = {t}: no stored {rd} read"
            r = out[l] = dict(tag=t, **own(M[t][rd]), coll=nx.coll_counts(D[t][rd]), **creep(Rr[t][rd]), pair={})
            for l2, t2 in arms:
                if t2 != t and M[t][rd] and M[t2][rd]:
                    r["pair"][l2] = dict(text=wt.xpair(M[t][rd], M[t2][rd])["paired"], **nx.pair_ci(M[t][rd], M[t2][rd]))
    cf = {l: cont_of(t) for l, t in arms}
    res["cont"] = cf
    J, V = arms[0], ref
    xp = wt.xpair(M[J[1]]["held"], M[V[1]]["held"])
    line = {}
    line["1 held zeros"] = dict(arm=xp["J"]["zeros"], ref=xp["C"]["zeros"], thr="<= ref + 2", met=bool(xp["J"]["zeros"] <= xp["C"]["zeros"] + 2))
    line["2 held progress_rel"] = dict(arm=xp["J"]["progress_rel"], ref=xp["C"]["progress_rel"], thr="> ref", met=bool(xp["J"]["progress_rel"] > xp["C"]["progress_rel"]))
    pj = res["reads"]["pub"][J[0]]["pair"].get(V[0])
    line["3 public paired mean vs ref, CI lower bound"] = dict(arm=None if pj is None else pj["lo"], ref=None, thr=">= -0.005", met=bool(pj is not None and pj["lo"] >= -0.005))
    rear = {l: res["reads"]["pub"][l]["coll"]["rear"] for l in (J[0], V[0])}
    line["4 public rear-ended scenes"] = dict(arm=rear[J[0]], ref=rear[V[0]], thr="<= ref", met=bool(rear[J[0]] <= rear[V[0]]))
    s = (cf[J[0]] or {}).get("static")
    line["5 static scene distance (m), no contact"] = dict(arm=None if s is None else s["dist"], ref=((cf[V[0]] or {}).get("static") or {}).get("dist"), thr="<= 1.0 and no contact",
                                                           met=bool(s is not None and not s["collision"] and s["dist"] <= STATIC_MAX))
    res["lines"], res["all_lines_met"] = line, all(x["met"] for x in line.values())
    f3 = lambda x: "-" if x is None else "%d" % x if float(x).is_integer() else "%.3f" % x  # noqa: E731

    def pair_cell(r, l, l2):
        q = r["pair"].get(l2)
        return "-" if l2 == l or q is None else "%+.3f [%+.3f, %+.3f]" % (q["mean"], q["lo"], q["hi"])

    def held_cells(c):
        tu, st, s17 = c.get("turns"), c.get("static"), c.get("stop17")
        return [("-" if tu is None else "%d / %d" % tu), ("-" if st is None else "%.2f m, %s" % (st["dist"], "contact" if st["collision"] else "no contact")),
                ("-" if s17 is None else "%d / %d" % (s17["zeros"], s17["n"]))]

    L = ["# %s = %s against %s (stored reads, one seed each)" % (J[0], J[1], ", ".join("%s = %s" % x for x in arms[1:])), ""]
    for rd, title in (("held", "Held 460 @18"), ("pub", "Public 1491 @10")):
        head = ["arm", "scenes", "zeros", "mean"] + ["paired vs " + l for l, _ in arms] + ["at-fault", "offroad", "progress_rel", "front / side / rear"]
        head += ["turns /24", "static scene", "stop17 zeros"] if rd == "held" else []
        head += ["creep", "no-launch"]
        L += ["## " + title, "", "| " + " | ".join(head) + " |", "|---|" + "--:|" * (len(head) - 1)]
        for l, t in arms:
            r, c = res["reads"][rd][l], cf[l] or {}
            cells = [l, r["scenes"], r["zeros"], "%.4f" % r["mean"]] + [pair_cell(r, l, l2) for l2, _ in arms]
            cells += [r["at_fault"], r["offroad"], "%.3f" % r["progress_rel"], "%d / %d / %d" % (r["coll"]["front"], r["coll"]["lateral"], r["coll"]["rear"])]
            cells += held_cells(c) if rd == "held" else []
            L.append("| " + " | ".join(str(x) for x in cells + [r["creep"], r["no_launch"]]) + " |")
        L.append("")
    L += ["## Lines of prereg plans/2026-10-17-v2ac-prereg.md (%s against %s)" % (J[0], V[0]), "", "| line | %s | %s | threshold | met |" % (J[0], V[0]), "|---|--:|--:|:-:|:-:|"]
    for k, x in line.items():
        L.append("| %s | %s | %s | %s | %s |" % (k, f3(x["arm"]), f3(x["ref"]), x["thr"], "yes" if x["met"] else "no"))
    L += ["", "All five lines met: %s." % ("yes" if res["all_lines_met"] else "no"), ""]
    for l, _ in arms:
        if cf[l]:
            L.append("%s: arrival 4 s arc @k=1 %s m, dev ADE %s." % (l, f3(cf[l]["arrival"]), f3(cf[l]["dev_ade"])))
    nx.write(a.out, res, L, a.dry)


# ---------------------------------------------------------------- distance relative to the log
def cmd_dist(a):
    from jevdrive import stats
    arms = [lab_tag(x) for x in a.tags]
    assert a.ref in [l for l, _ in arms], f"--ref {a.ref} is not one of {[l for l, _ in arms]}"
    R = {l: {rd: rollouts(read_dirs(t)[rd]) for rd in READS} for l, t in arms}
    res = dict(arms=dict(arms), ref=a.ref, reads={})
    L = [f"# Distance driven relative to the log (scenes where the log drives > 5 m): {', '.join(f'{l} = {t}' for l, t in arms)}", "",
         f"Paired columns are against {a.ref} (arm minus {a.ref}, session-clustered CI over scenes; one training seed per arm, so the CI does not cover seed-to-seed spread). "
         "Descriptive; not a line of the prereg.", ""]
    for rd in READS:
        common = sorted(set.intersection(*[set(R[l][rd]) for l, _ in arms]))
        dg = lambda l, s: tuple(float((R[l][rd][s].get("metrics") or {}).get(k) or 0.0) for k in ("dist_traveled_m", "gt_dist_traveled_m"))  # noqa: E731
        ids = [s for s in common if dg(arms[0][0], s)[1] > 5]
        grp = np.array(["_".join(s.split("_")[:2]) for s in ids])
        X = {l: np.array([dg(l, s) for s in ids]) for l, _ in arms}
        out = res["reads"][READ_NAME[rd]] = dict(scenes=len(ids), arms={})
        L += [f"## {READ_NAME[rd]}: {len(ids)} scenes", "", f"| arm | distance / log (sum) | per-scene ratio (mean) | paired ratio vs {a.ref} [CI] | paired distance vs {a.ref} (m) [CI] | < 50 % of log | < 30 % | < 1 m |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for l, _ in arms:
            x, y = X[l], X[a.ref]
            r = out["arms"][l] = dict(sum_ratio=float(x[:, 0].sum() / x[:, 1].sum()), mean_ratio=float(np.mean(x[:, 0] / x[:, 1])), lt50=int((x[:, 0] < 0.5 * x[:, 1]).sum()),
                                      lt30=int((x[:, 0] < 0.3 * x[:, 1]).sum()), lt1m=int((x[:, 0] < 1).sum()))
            if l != a.ref:
                pr, pd_ = (stats.paired(u, v, groups=grp) for u, v in ((x[:, 0] / x[:, 1], y[:, 0] / y[:, 1]), (x[:, 0], y[:, 0])))
                r["paired_ratio"], r["paired_dist"] = stats.fmt(pr), stats.fmt(pd_)
            L.append(f"| {l} | {r['sum_ratio']:.3f} | {r['mean_ratio']:.3f} | {r.get('paired_ratio', '-')} | {r.get('paired_dist', '-')} | {r['lt50']} | {r['lt30']} | {r['lt1m']} |")
        L.append("")
    nx.write(a.out, res, L, a.dry)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("table")
    p.add_argument("--tags", nargs="+", required=True, help="[LABEL=]TAG, in column order"), p.add_argument("--ref", required=True, help="label of the reference column")
    p.add_argument("--out", default="regress"), p.add_argument("--dry", action="store_true", help="print only")
    p = sp.add_parser("agg")
    p.add_argument("--j", required=True, help="[LABEL=]TAG of the arm"), p.add_argument("--refs", nargs="+", required=True, help="[LABEL=]TAG; the first is the line reference")
    p.add_argument("--out", default="agg"), p.add_argument("--dry", action="store_true", help="print only")
    p = sp.add_parser("dist")
    p.add_argument("--tags", nargs="+", required=True, help="[LABEL=]TAG"), p.add_argument("--ref", required=True, help="label of the reference arm")
    p.add_argument("--out", default="dist"), p.add_argument("--dry", action="store_true", help="print only")
    a = ap.parse_args()
    dict(table=cmd_table, agg=cmd_agg, dist=cmd_dist)[a.cmd](a)
