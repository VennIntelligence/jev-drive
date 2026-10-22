#!/usr/bin/env python3
"""NUPLAN-DAG (plans/2026-10-22-nuplan-dag.md): tables of the nuPlan DAgger map-teacher lane. nd.sh runs the chain.

  $TR nd.py held --tags T.. --refs R.. --out F        held 460 @18: own summary, paired vs refs (df.cl_read), zero classes (nx_regress.failure_type)
  $AP nd.py latsplit --ext X --tags T.. --out F      arena held lateral zeros (offroad / corridor) of each tag split by d395's pre-registered rule
                                                     (lat_zero.classify: a over-speed at curvature / slow-down, b path error at normal speed,
                                                     c other) on a free_m / native_gap extract; a is split a2 turn (human turns >= 28 deg) / a1 other
  $TR nd.py quick --tags T.. --refs R.. --out M   score-only headline (arena + held 460 pairs, zero classes) before any extract
  $V nd.py p1md --root D --tags T.. --b2 B --ntv0 N --out M    Part 1 report (arena pairs, held 460, zero classes, lateral split)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/defect_fix/scripts"), str(REPO / "experiments/defect_fix/lib"),
                str(REPO / "experiments/alpasim_proxy/scripts"), str(REPO / "experiments/wait_cap/scripts"), str(REPO / "experiments/zeros_mech/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
LATERAL = ("offroad", "corridor", "AF-side")


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float))


def cmd_held(a):
    import df
    import nx_regress as NR
    out = {}
    for t in a.tags:
        r = df.cl_read(t, [x for x in a.refs if x != t])["held"]
        R = NR.rollouts(df.read_dirs(t)["held"])
        z = Counter(NR.failure_type(x) for x in R.values() if float(x.get("score") or 0) == 0)
        r["zero_types"] = dict(z.most_common())
        r["scores"] = {k: float(x.get("score") or 0) for k, x in R.items()}
        out[t] = r
        print(t, r["scenes"], r["zeros"], f"{r['mean']:.4f}", {c: f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]" for c, p in r["pair"].items()}, flush=True)
    for t in a.refs:                                                    # the references' own zero classes
        if t not in out:
            R = NR.rollouts(df.read_dirs(t)["held"])
            out[t] = dict(scenes=len(R), zeros=sum(float(x.get("score") or 0) == 0 for x in R.values()),
                          mean=float(np.mean([float(x.get("score") or 0) for x in R.values()])),
                          zero_types=dict(Counter(NR.failure_type(x) for x in R.values() if float(x.get("score") or 0) == 0).most_common()),
                          scores={k: float(x.get("score") or 0) for k, x in R.items()})
    dump(a.out, out)


def cmd_latsplit(a):
    import pickle
    import pandas as pd
    import lat_zero as LZ
    X = Path(a.ext)
    D, St, Wn = pd.read_parquet(X / "dec.parquet"), pd.read_parquet(X / "steps.parquet"), pd.read_parquet(X / "win.parquet")
    H = pickle.load(open(X / "hum.pkl", "rb"))
    S = pd.read_csv(DATA / "runs/alpasim_proxy/arena/set/set.csv.gz").set_index("id")             # free_m arena's window set
    held = set(S.index[S.split == "held"])
    rows = []
    for t in a.tags:
        z = Wn[(Wn.tag == t) & (Wn.score == 0) & Wn.id.isin(held)]
        z = z[z.type.fillna("").str.contains("offroad|corridor")]
        for _, m in z.iterrows():
            if m.id not in H:
                rows.append(dict(tag=t, id=m.id, type=m.type, cls="c: no human track"))
                continue
            w = LZ.window(m.id, t, D, St, LZ.Track(H[m.id]))
            c = LZ.classify(w, "offroad+corridor" if "offroad" in m.type and "corridor" in m.type else "offroad" if "offroad" in m.type
                            else "corridor" if "corridor" in m.type else m.type)
            sub = ("a2: turn" if w.get("turn_deg", 0) >= 28 else "a1: stop / slow-down") if c.startswith("a") else ""
            rows.append(dict(tag=t, id=m.id, type=m.type, stratum=S.stratum.get(m.id), cls=c, asub=sub,
                             **{k: w.get(k) for k in ("r3", "slow", "kappa", "turn_deg", "outward", "exit_off", "v_pre", "vh_pre")}))
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    print(R.groupby(["tag", "cls"]).size().unstack(fill_value=0).to_string())


def arena_scores(tag):
    """Score-only arena read of a tag (no extract): window id -> rollout row, from the 18 chunk run dirs (jt.sh aread)."""
    import glob
    R = {}
    for f in sorted(glob.glob(str(DATA / f"runs/defect_fix/jt/arena/{tag}@off/c*/run/aggregate/results-summary.json"))):
        for r in json.load(open(f))["rollouts"]:
            R.setdefault(r["clipgt_id"], r)
    return R


def cmd_quick(a):
    """Headline tables from the scorer outputs only (no native_gap extract): arena held windows (score, failure_type, route-fail excluded as
    free_m arena does) and held 460 @18, each tag paired against the refs (log-clustered bootstrap), zero classes."""
    import pandas as pd
    import df
    import nx_regress as NR
    from jevdrive import stats
    S = pd.read_csv(DATA / "runs/alpasim_proxy/arena/set/set.csv.gz").drop_duplicates("id").set_index("id")
    held = S.index[S.split == "held"]
    tags = list(dict.fromkeys([*a.tags, *a.refs]))
    A = {t: arena_scores(t) for t in tags}
    H = {t: NR.rollouts(df.read_dirs(t)["held"]) for t in tags}
    sc = lambda r: float(r.get("score") or 0)  # noqa: E731
    L = [f"# {a.title}: score-only headline (scorer outputs, no extract)", "",
         "Arena: held windows read by both tags, route-fail windows dropped (free_m arena's rule); held 460 @18: scenes read by both. "
         "Paired = arm - ref, log-clustered bootstrap 95 % CI (jevdrive.stats.paired).", "",
         "| arm | ref | read | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |", "|:--|:--|:--|--:|--:|--:|--:|--:|"]
    lat = lambda r: sc(r) == 0 and any(k in NR.failure_type(r) for k in LATERAL)  # noqa: E731
    out = {}
    for t in a.tags:
        for ref in [x for x in a.refs if x != t]:
            for rd, D in (("arena", A), ("held 460", H)):
                ids = sorted(set(D[t]) & set(D[ref]) & (set(held) if rd == "arena" else set(D[t])))
                if rd == "arena":
                    ids = [i for i in ids if NR.failure_type(D[t][i]) != "route-fail" and NR.failure_type(D[ref][i]) != "route-fail"]
                if not ids:
                    continue
                x, y = np.array([sc(D[t][i]) for i in ids]), np.array([sc(D[ref][i]) for i in ids])
                g = S.log.reindex(ids).values if rd == "arena" else np.array([i.rsplit("-", 1)[0] for i in ids])
                p = stats.paired(x, y, groups=g)
                la, lb = sum(lat(D[t][i]) for i in ids), sum(lat(D[ref][i]) for i in ids)
                out[f"{t}|{ref}|{rd}"] = dict(n=len(ids), a=float(x.mean()), b=float(y.mean()), **{k: float(p[k]) for k in ("mean", "lo", "hi")},
                                              zeros_a=int((x == 0).sum()), zeros_b=int((y == 0).sum()), lat_a=la, lat_b=lb)
                L.append(f"| {t} | {ref} | {rd} | {len(ids)} | {x.mean():.4f} / {y.mean():.4f} | {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] | "
                         f"{(x == 0).sum()} / {(y == 0).sum()} | {la} / {lb} |")
    for rd, D in (("arena held", A), ("held 460", H)):
        L += ["", f"Zero classes, {rd} (nx_regress.failure_type):", "", "| class | " + " | ".join(tags) + " |", "|:--|" + "--:|" * len(tags)]
        C = {t: Counter(NR.failure_type(r) for i, r in D[t].items() if sc(r) == 0 and (rd != "arena held" or i in set(held))) for t in tags}
        for k in sorted({k for c in C.values() for k in c}, key=lambda k: -sum(c.get(k, 0) for c in C.values())):
            L.append(f"| {k} | " + " | ".join(str(C[t].get(k, 0)) for t in tags) + " |")
        out[f"zeros|{rd}"] = {t: dict(C[t]) for t in tags}
    Path(a.out).write_text("\n".join(L) + "\n")
    dump(Path(a.out).with_suffix(".json"), out)
    print("\n".join(L))


def ci(p):
    return f"{p['mean' if 'mean' in p else 'diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"


def cmd_p1md(a):
    import pandas as pd
    D = Path(a.root)
    H = json.load(open(D / "held.json"))
    LS = pd.read_csv(D / "latsplit.csv") if (D / "latsplit.csv").exists() else None
    L = [f"# Part 1: PAI checkpoints on nuPlan (native serve, B2's adoption reads)", "",
         f"Reads: arena held split (18 chunks, `jt.sh aread <tag> off`) and held 460 @18 (`rm_chain.sh read`), paired per window / scene with the "
         f"stored {a.b2} (B2, nuPlan candidate) and {a.ntv0} reads; log-clustered bootstrap (`jevdrive.stats.paired` / `nx.pair_ci`).", "",
         "## Arena held, native", "", "| arm | ref | windows | mean arm / ref | paired [CI] | zeros arm / ref | AF-front arm / ref | k9 speed ratio arm / ref |",
         "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    Ar = {t: json.load(open(D / f"arena_{t}.json")) for t in [*a.tags, a.ntv0]}
    for t, A in Ar.items():
        for r in [A["b"], *[k for k in A if k not in ("a", "b", A["b"])]]:
            x = A[r]
            L.append(f"| {t} | {r} | {x['windows']} | {x['mean_a']:.4f} / {x['mean_b']:.4f} | {x['diff']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] | "
                     f"{x['zeros_a']} / {x['zeros_b']} | {x['af_front_a']} / {x['af_front_b']} | {x['k9']['a']:.3f} / {x['k9']['b']:.3f} |")
    L += ["", "Arena zero classes (held windows, vs B2):", "", "| class | " + " | ".join([*a.tags, a.ntv0, a.b2]) + " |", "|:--|" + "--:|" * (len(a.tags) + 2)]
    ty = {t: {k: v for k, v in Ar[t][a.b2]["types"]["a"].items()} for t in [*a.tags, a.ntv0]}
    ty[a.b2] = Ar[a.tags[0]][a.b2]["types"]["b"]
    keys = sorted({k for v in ty.values() for k in v if k != "ok" and not k.startswith(("hit", "slow", "crawl", "barely", "partial"))},
                  key=lambda k: -sum(v.get(k, 0) for v in ty.values()))
    for k in keys:
        L.append(f"| {k} | " + " | ".join(str(ty[t].get(k, 0)) for t in [*a.tags, a.ntv0, a.b2]) + " |")
    L += ["", "Arena strata (held windows; mean arm / B2, zeros arm / B2):", "", "| stratum | " + " | ".join(a.tags) + " |", "|:--|" + "--:|" * len(a.tags)]
    for s in Ar[a.tags[0]][a.b2]["strata"]:
        L.append(f"| {s} ({Ar[a.tags[0]][a.b2]['strata'][s]['windows']}) | " + " | ".join(
            f"{Ar[t][a.b2]['strata'][s]['a']:.3f} / {Ar[t][a.b2]['strata'][s]['b']:.3f}, {Ar[t][a.b2]['strata'][s]['zeros_a']} / {Ar[t][a.b2]['strata'][s]['zeros_b']}"
            for t in a.tags) + " |")
    L += ["", "## Held 460 @18", "", "| arm | scenes | mean | zeros | lateral zeros | front / side / rear | paired vs " + " | paired vs ".join([a.b2, a.ntv0]) + " |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for t in [*a.tags, a.ntv0]:
        x = H[t]
        pr = [f"{ci(x['pair'][r])} ({x['pair'][r]['zeros']} / {x['pair'][r]['zeros_ref']})" if r in x["pair"] else "-" for r in (a.b2, a.ntv0)]
        L.append(f"| {t} | {x['scenes']} | {x['mean']:.4f} | {x['zeros']} | {x['lat_zeros']} | {x['coll']['front']} / {x['coll']['lateral']} / {x['coll']['rear']} | "
                 + " | ".join(pr) + " |")
    L += [f"| {a.b2} | {H[a.b2]['scenes']} | {H[a.b2]['mean']:.4f} | {H[a.b2]['zeros']} | | | | |", "",
          "Held 460 zero classes:", "", "| class | " + " | ".join([*a.tags, a.ntv0, a.b2]) + " |", "|:--|" + "--:|" * (len(a.tags) + 2)]
    keys = sorted({k for t in [*a.tags, a.ntv0, a.b2] for k in H[t]["zero_types"]}, key=lambda k: -sum(H[t]["zero_types"].get(k, 0) for t in H))
    for k in keys:
        L.append(f"| {k} | " + " | ".join(str(H[t]["zero_types"].get(k, 0)) for t in [*a.tags, a.ntv0, a.b2]) + " |")
    if LS is not None:
        L += ["", "## Arena lateral zeros split (d395 rule, `lat_zero.classify`; a2 = human turns >= 28 deg in the pre-exit span)", "",
              "| class | " + " | ".join([*a.tags, a.b2, a.ntv0]) + " |", "|:--|" + "--:|" * (len(a.tags) + 2)]
        LS["k"] = LS.cls.str[:1] + LS.asub.fillna("").str[:2].replace({"": ""})
        for k, lab in (("a", "a over-speed (all)"), ("aa2", "  a2 in a turn"), ("aa1", "  a1 stop / slow-down"), ("b", "b path error, normal speed"), ("c", "c other")):
            L.append(f"| {lab} | " + " | ".join(str(int((LS[(LS.tag == t)].k.str.startswith(k) if k == "a" else LS[(LS.tag == t)].k == k).sum())) for t in [*a.tags, a.b2, a.ntv0]) + " |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("held")
    q.add_argument("--tags", nargs="+", required=True), q.add_argument("--refs", nargs="+", required=True), q.add_argument("--out", required=True)
    q = sp.add_parser("latsplit")
    q.add_argument("--ext", required=True), q.add_argument("--tags", nargs="+", required=True), q.add_argument("--out", required=True)
    q = sp.add_parser("quick")
    q.add_argument("--tags", nargs="+", required=True), q.add_argument("--refs", nargs="+", required=True), q.add_argument("--out", required=True)
    q.add_argument("--title", default="NUPLAN-DAG")
    q = sp.add_parser("p1md")
    for k in ("root", "b2", "ntv0", "out"):
        q.add_argument(f"--{k}", required=True)
    q.add_argument("--tags", nargs="+", required=True)
    a = p.parse_args()
    dict(held=cmd_held, latsplit=cmd_latsplit, p1md=cmd_p1md, quick=cmd_quick)[a.cmd](a)


if __name__ == "__main__":
    main()
