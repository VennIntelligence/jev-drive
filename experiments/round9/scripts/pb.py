#!/usr/bin/env python3
"""ROUND9 PAI-BASE (plans/2026-10-22-pai-base.md): a base that sees cleaned PAI human-state rows from the start (RB0-P), against the
navtrain-only base RB0-F. Checks, gates and reports; lane script pb.sh. Box, repo root.

  pb.py wfile --out W.npz --json J      draw file of the base arms on pool on1_JTPH: R6FIX's weights (MV-LC d430 + F1 / F2 d433) with the
                                        S rows (grp S) at 0, the PAI parts (ph / ph05 / ph15) kept at their relative weights; data check:
                                        names = the pool's, w > 0 only on PAI human rows, their scenes in physical_ai_av/pai-rows@v1 and in
                                        none of test281 / curated-val / drives_val160 / train_exclude / exam_scenes / the N batch's
                                        same-drive scenes, none in the trainer's held logs (crc32 % 100 < 15); exit 2 on a failure
  pb.py draws --tag T --k K --steps N --w W.npz --out J
                                        the trainer's pool draw replayed on CPU (stream [seed, 0, 53], numpy weighted choice per step, the
                                        running sha1 of the indices): on_hash_60 / _1000 / final must equal the run's DONE; drawn rows by
                                        part, distinct scenes, any drawn row with w = 0 or in a held scene -> exit 2
  pb.py smoke --tag T --k K --out J     300-step smoke: DONE, finite scalars, batch line (128 - 13 - K) / 13 / K, dev ADE below step 0
  pb.py basegate --r281 J --draws J.. --ident J --out P    the registered base gate (plan section 4) -> P.{json,md}, CARRY (the carried tag)
                                        or exit 3 (no share carried: the chain stops)
  pb.py final --r281 J --read K=F.. --pcs J --out P        the registered adoption rule + per-class losses (loss_map classes) -> P.{json,md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
DD = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = DD / "runs/op_parity/cache"
POOL = "on1_JTPH"
PAI = ("ph", "ph05", "ph15")
EPS = 1e-4


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float))


def scene_of(log):
    return str(log)


def cmd_wfile(a):
    import pandas as pd
    from jevdrive.data import splits
    src = np.load(DD / "runs/round6/fix/w/R6FIX.npz")
    tb = np.load(CR / POOL / "tab.npz")
    names, grp, w0 = src["names"].astype(str), src["grp"].astype(str), src["w"].astype(np.float64)
    ck = dict(names_equal=bool(np.array_equal(names, tb["names"].astype(str))))
    part = tb["pool"].astype(str)
    w = np.where(np.isin(grp, PAI), w0, 0.0)
    w = w / w.sum()
    on = w > 0
    ck["only_pai_parts"] = bool(np.isin(part[on], PAI).all())
    sc = tb["log"].astype(str)
    used = sorted(set(sc[on].tolist()))
    tr = splits.load("physical_ai_av/pai-rows")
    bad = {}
    for nm in ("physical_ai_av/test281", "physical_ai_av/curated-val"):
        s = splits.load(nm)
        bad[nm] = sorted(set(used) & set(s.members))
    P = REPO / "experiments/pai_fix/scenes"
    for f in ("drives_val160.csv", "train_exclude.csv"):
        bad[f] = sorted(set(used) & set(pd.read_csv(P / f).scene_id))
    bad["exam_scenes.txt"] = sorted(set(used) & set((P / "exam_scenes.txt").read_text().split()))
    bad["n_t281_partner"] = sorted(set(used) & set(pd.read_csv(REPO / "experiments/pai_rows/results/n_t281_partner.tsv", sep="\t").scene_id))
    held = [s for s in used if zlib.crc32(s.encode()) % 100 < 15]
    ck |= dict(in_pai_rows=bool(set(used) <= set(tr.members)), excluded_hits={k: len(v) for k, v in bad.items()}, held_hits=len(held))
    share = {g: float(w[part == g].sum()) for g in PAI}
    rec = dict(src="runs/round6/fix/w/R6FIX.npz", pool=POOL, rows=len(w), drawable=int(on.sum()), scenes=len(used),
               drawable_by_part={g: int((on & (part == g)).sum()) for g in PAI}, share=share,
               split_ids={n: splits.load(n).id for n in ("physical_ai_av/pai-rows", "physical_ai_av/test281", "physical_ai_av/curated-val")},
               checks=ck, bad_examples={k: v[:5] for k, v in bad.items() if v})
    ok = ck["names_equal"] and ck["only_pai_parts"] and ck["in_pai_rows"] and not held and not any(bad.values())
    rec["pass_"] = bool(ok)
    np.savez(a.out, names=names, w=w, grp=grp)
    dump(a.json, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def run_dir(tag):
    ds = sorted((DD / "runs/route_mem").glob(f"train-{tag}/*/"))
    assert ds, f"no run dir of {tag}"
    return ds[-1]


def cmd_draws(a):
    z = np.load(a.w)
    w = z["w"].astype(np.float64)
    pw = w / w.sum()
    tb = np.load(CR / POOL / "tab.npz")
    part, sc = tb["pool"].astype(str), tb["log"].astype(str)
    done = json.loads((run_dir(a.tag) / "DONE").read_text())
    orng = np.random.default_rng([a.seed, 0, 53])
    sha = hashlib.sha1()
    cnt = np.zeros(len(w), np.int64)
    hs = {}
    for step in range(a.steps):
        i = orng.choice(len(w), a.k, replace=True, p=pw)
        sha.update(i.astype("<i8").tobytes())
        np.add.at(cnt, i, 1)
        if step + 1 in (60, 1000, a.steps):
            hs[step + 1] = sha.hexdigest()[:16]
    want = {60: done.get("on_hash_60"), 1000: done.get("on_hash_1000"), a.steps: done.get("on_hash")}
    d = cnt > 0
    held = np.array([zlib.crc32(s.encode()) % 100 < 15 for s in sc[d]])
    rec = dict(tag=a.tag, k=a.k, steps=a.steps, hash_replayed=hs, hash_logged={str(k): v for k, v in want.items()},
               hash_equal=all(hs[k] == v for k, v in want.items()), draws=int(cnt.sum()),
               draws_by_part={g: int(cnt[part == g].sum()) for g in sorted(set(part[d].tolist()))},
               share_by_part={g: float(cnt[part == g].sum() / cnt.sum()) for g in PAI}, distinct_rows=int(d.sum()),
               distinct_scenes=len(set(sc[d].tolist())), drawn_w0=int((d & (w == 0)).sum()), drawn_held_rows=int(held.sum()),
               passes_per_drawable_row=float(cnt.sum() / (w > 0).sum()))
    rec["pass_"] = bool(rec["hash_equal"] and rec["drawn_w0"] == 0 and rec["drawn_held_rows"] == 0 and set(rec["draws_by_part"]) <= set(PAI))
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["pass_"] else 2)


def cmd_smoke(a):
    d = run_dir(a.tag)
    log = (d / "log.txt").read_text()
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy \(mode (\w+)", log)
    ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]
    vals = [e["value"] for e in ev if e.get("kind") == "scalar" and not e["tag"].startswith("throughput/")]
    dev = {e["step"]: e["value"] for e in ev if e.get("kind") == "scalar" and e["tag"] == "dev/ade"}
    m0 = re.search(r"dev @ 0: ade ([\d.]+)", log)
    m1 = re.findall(r"dev @ (\d+): ade ([\d.]+)", log)
    ck = dict(done=(d / "DONE").exists(), finite=bool(np.isfinite(vals).all()) if vals else True,
              batch=b.groups() if b else None, batch_want=[str(128 - 13 - a.k), "13", str(a.k), "imit"],
              dev0=float(m0.group(1)) if m0 else None, dev_last=float(m1[-1][1]) if m1 else None, dev_events=len(dev))
    ok = ck["done"] and ck["finite"] and b is not None and list(b.groups()) == ck["batch_want"] and ck["dev_last"] is not None \
        and ck["dev_last"] < ck["dev0"]
    rec = dict(tag=a.tag, checks=ck, pass_=bool(ok))
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def f3(d):
    return "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"]) if d else "-"


def nl(r):
    return int(str(r.get("launch", "0/37")).split("/")[0])


def cmd_basegate(a):
    J = json.loads(Path(a.r281).read_text())
    R, P = J["reads"], J["pairs"]
    ident = json.loads(Path(a.ident).read_text())
    dr = {json.loads(Path(x).read_text())["tag"]: json.loads(Path(x).read_text()) for x in a.draws}
    ref = R["RB0F-ag"]
    rows, cand = [], []
    for s in ("30", "50", "70"):
        k = f"RB0P{s}-ag"
        p = (P.get(f"{k}:RB0F-ag") or {}).get("mean")             # r2.py r281: per pair {"mean": stats.paired(..), <class>: counts}
        ok = dict(ci_lo_gt0=bool(p) and p["lo"] > 0, launch=nl(R[k]) >= nl(ref) - 2, draws=dr.get(f"RB0-P{s}-s0", {}).get("pass_", False))
        if all(ok.values()):
            cand.append((p["mean"], -int(s), s))
        rows.append(f"| {s} % | {R[k]['mean']:.3f} | {f3(p)} | {R[k]['zeros']} | {R[k]['z_offroad']} + {R[k]['z_zero-other']} | {R[k]['launch']} | "
                    f"{' '.join(x for x, v in ok.items() if not v) or 'pass'} |")
    id_ok = bool(ident.get("identical", ident.get("pass_", False)))
    carry = None
    if cand and id_ok:
        carry = f"RB0-P{max(cand)[2]}-s0"
    L = ["# PAI-BASE base gate (plan section 4): RB0-P (PAI share 30 / 50 / 70 % of the imitation rows) vs RB0-F-s0, PAI 281 ACC + GAP, "
         "Harmonizer OFF", "",
         f"Identity (RB0-F-s0's argv rerun on HEAD, eager, 10 000 steps vs the stored RB0-F-s0): {'identical' if id_ok else 'NOT identical'}.", "",
         f"RB0-F-s0: mean {ref['mean']:.3f}, zeros {ref['zeros']}, offroad + zero-other {ref['z_offroad']} + {ref['z_zero-other']}, launched {ref['launch']}.", "",
         "| PAI share | mean | - RB0-F (paired, 95 % CI) | zeros | offroad + zero-other | launched | gate |", "|:--|--:|--:|--:|--:|--:|:--|", *rows, "",
         "Rule: paired 95 % CI lower bound > 0, launched >= RB0-F's - 2, the replayed draws pass; among passing shares the highest paired mean "
         "(tie: the lower share). Identity must hold.", "",
         f"**Carried: {carry or 'none (the line stops)'}.**"]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    dump(f"{a.out}.json", dict(carry=carry, identity=id_ok, candidates=cand, rows=rows))
    print("\n".join(L))
    if carry:
        (Path(a.out).parent.parent / "CARRY").write_text(carry + "\n")
        sys.exit(0)
    sys.exit(3)


def cmd_final(a):
    import pandas as pd
    sys.path.insert(0, str(REPO / "experiments/round7/scripts"))
    import loss_map as LM
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in a.read)}
    com = sorted(set.intersection(*[set(X.index[X.score.notna() & ~X.rpc.fillna(False).astype(bool)]) for X in F.values()]))
    F = {k: X.reindex(com) for k, X in F.items()}
    S = {k: X.score.to_numpy(float) for k, X in F.items()}
    from jevdrive import stats
    avg = lambda m, s: (S[f"{m}-{s}-ag"] + S[f"{m}-{s}-agH"]) / 2  # noqa: E731
    dag3 = np.mean([avg("R6DAG", s) for s in ("s0", "s1", "s2")], 0)
    line = float(dag3.mean())
    cls = [k for k, _, _ in LM.CLASSES]

    def zc(k):
        X = F[k]
        c = {q: 0 for q in cls}
        lost = {q: 0.0 for q in cls + ["near1"]}
        for r in X.itertuples():
            q = LM.classify(r)
            if q == "ok":
                continue
            lost[q] += 1 - r.score
            if r.score <= EPS:
                c[q] += 1
        return c, lost
    Z = {k: zc(k) for k in F}
    zavg = lambda m, s: {q: (Z[f"{m}-{s}-ag"][0][q] + Z[f"{m}-{s}-agH"][0][q]) / 2 for q in cls}  # noqa: E731
    zdag = {q: np.mean([zavg("R6DAG", s)[q] for s in ("s0", "s1", "s2")]) for q in cls}
    ver, rows = {}, []
    Q = json.loads(Path(a.pcs).read_text()) if a.pcs and Path(a.pcs).exists() else {}
    for s in ("s0", "s1"):
        m = float(avg("PBDAG", s).mean())
        za = zavg("PBDAG", s)
        worse = {q: za[q] - zdag[q] for q in cls if za[q] > zdag[q] + 2}
        ok = dict(avg_ge_line=m >= line, no_class_worse=not worse)
        ver[s] = dict(ok, avg=m, worse=worse)
        pd3 = stats.paired(avg("PBDAG", s), dag3)
        pn = stats.paired(S[f"PBDAG-{s}-ag"], S["NTV0-ag"])
        q = (Q.get(f"PBDAG-{s}+AG") or {})
        rows.append(f"| {s} | {S[f'PBDAG-{s}-ag'].mean():.3f} | {S[f'PBDAG-{s}-agH'].mean():.3f} | {m:.3f} vs {line:.3f} | {f3(pd3)} | {f3(pn)} | "
                    + ("-" if q.get("diff_point") is None else "%+.0f [%+.0f, %+.0f]" % (q["diff_point"], *q["diff_ci"]))
                    + f" | {', '.join(f'{k} +{v:.1f}' for k, v in worse.items()) or 'none'} | {' '.join(x for x, v in ok.items() if not v) or 'pass'} |")
    adopt = all(all(v for k, v in ver[s].items() if k in ("avg_ge_line", "no_class_worse")) for s in ver)
    L = ["", "## Registered adoption rule (plan section 6)", "",
         f"Both seeds: (OFF + ON) / 2 >= R6DAG's 3-seed mean on the same {len(com)} scenes ({line:.3f}; the record's 0.792 on its own scene set) "
         "and no loss_map class with more zeros (mean of OFF and ON) than R6DAG's 3-seed mean + 2.", "",
         "| seed | OFF | ON | (OFF + ON) / 2 vs line | - R6DAG 3-seed (paired) | OFF - NTV0 (paired) | ruler - NTV0 | classes worse > +2 | failing |",
         "|:--|--:|--:|--:|--:|--:|--:|:--|:--|", *rows, "", f"**Adopt: {'yes' if adopt else 'no'}.**", "",
         "## Per-class zeros / lost points (loss_map classes; zeros, points lost = sum of 1 - score)", "",
         "| class | " + " | ".join(F) + " |", "|:--|" + "--:|" * len(F)]
    for q in cls + ["near1"]:
        L.append(f"| {q} | " + " | ".join((f"{Z[k][0][q]} / " if q in Z[k][0] else "- / ") + f"{Z[k][1][q]:.1f}" for k in F) + " |")
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    dump(f"{a.out}.json", dict(scenes=len(com), line=line, verdict=ver, adopt=adopt, zeros={k: v[0] for k, v in Z.items()},
                                lost={k: v[1] for k, v in Z.items()}, zeros_dag3=zdag))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("wfile")
    p.add_argument("--out", required=True), p.add_argument("--json", required=True)
    p = sp.add_parser("draws")
    p.add_argument("--tag", required=True), p.add_argument("--k", type=int, required=True), p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0), p.add_argument("--w", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--k", type=int, required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("basegate")
    p.add_argument("--r281", required=True), p.add_argument("--draws", nargs="+", required=True), p.add_argument("--ident", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("final")
    p.add_argument("--read", nargs="+", required=True), p.add_argument("--md", required=True), p.add_argument("--pcs", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(wfile=cmd_wfile, draws=cmd_draws, smoke=cmd_smoke, basegate=cmd_basegate, final=cmd_final)[a.cmd](a)


if __name__ == "__main__":
    main()
