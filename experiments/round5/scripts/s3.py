#!/usr/bin/env python3
"""S3 outcome rows (round 5; plans/2026-10-21-s3-rows.md): D3's 4 serve perturbations of NTV0-L18-s0 + ACC + GAP on every non-held eligible curated
PAI train scene, the per-scene winner, the winners' rollouts re-rolled with frame dumps, rows labelled with the winner's SERVED plan (pool on1_PAIO).
s3.sh runs the chain; this file holds the selection, the checks and the tags.

  s3.py scenes --out results/s3_scenes.tsv          Mac, before any roll: the 719 non-held scenes of pai_rows/results/scenes.tsv, in_d3 = D3 scored all 4 arms
  s3.py lists --scenes TSV --root O                 box: chunk lists O/lists/cNNN.txt (c000 = 4 new scenes, one per class, + 2 D3 check scenes; then 25 each)
  s3.py check --root O --d3 D3                      box: stage-1 check on c000 -> O/checks/c000.json; exit 1 = stop
  s3.py select --root O --d3 D3 --scenes TSV        box: scores of all 719 (D3's roll for in_d3 scenes), winner, outcome type -> O/sel.json,
                                                    O/report/{s3_scores.csv, summary.json, tables.md}, re-roll lists O/reroll/<arm>/lists/rNNN.txt
  s3.py r1 --root O [--chunk r000]                  box: the re-roll against the stage-1 roll (score, served poses, anchors per decision) -> O/checks/r1_*.json;
                                                    without --chunk every re-rolled scene, and O/keep.tsv (the scenes whose re-roll reproduces)
  s3.py tag --root O --pool on1_PAIO                box: the pool's outcome tags (tab otype / ocls / osub / oarm) and outcome.npz (k, divergence, A's served plan)
  s3.py r2 --root O --pool on1_PAIO                 box: R2, every row's fut against the stage-1 drive.jsonl `poses` of the same decision -> O/checks/r2.json
  s3.py r1in --root O --pool on1_PAIO --fwd DIR --tag T --control W   box: R1 input path (trainer forward vs the model's own plan) -> O/checks/r1in.json
  s3.py mfix --pool on1_JTPHO --ref on1_JTPH        box: after the zero-copy merge: the prefix equals --ref (every file), agent ok of the prefix from --ref
  s3.py weights --pool on1_JTPHO --base R2JF --out R2JFO   box: --base's draw + weight 0 on the PAIO rows (R3)
  s3.py report --root O --pool on1_PAIO             box: rows by outcome type x class, disk, card-hours -> O/report/rows.{md,json}
  s3.py relabel --root O --pool on1_PAIO --out PAIOR   box (plan addendum 1): hindsight relabel, fut = the winner rollout's realised 4 s rig poses
                                                    (NaN where incomplete), gates L0 + self-checks -> results/s3/relabel.{md,json}; then pool on1_PAIOR
                                                    (same rows / files, tokens zero-copy, relabel.npz); exit 1 = stop (O/VERDICT.relabel)
  s3.py rcheck --pool on1_JTPHOR --ref on1_JTPHO --src on1_PAIOR   box: the relabelled merge equals the original except tab.fut's tail
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/wait_cap/scripts")]
import d3 as D3  # noqa: E402  (load; d3.py puts wait_cap/scripts on the path)
from nx_regress import failure_type  # noqa: E402

ARMS, CLS = D3.ARMS, D3.CLS
INFO, ADV, FULL, T_HAND, TOL = 0.2, 0.1, 0.98, 1.6, 0.05
CARD_STACKS = 3
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = DATA / "runs/op_parity/cache"
OTYPES = ("partial-fixed", "zero-rescued", "A-wins")


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))


def read_tsv(f):
    return list(csv.DictReader(open(f), delimiter="\t"))


# ---------------------------------------------------------------- stage 1: scenes, lists, check
def scenes(a):
    sel = {r["scene_id"]: r for r in csv.DictReader(open(REPO / "experiments/pai_av/results/nurec_select.csv"))}
    d3 = {r["scene"] for r in csv.DictReader(open(REPO / "experiments/round5/results/d3/d3_scores.csv"))}
    rows = [r for r in read_tsv(REPO / "experiments/pai_rows/results/scenes.tsv") if r["held"] == "False"]
    out = [dict(scene_id=r["scene_id"], cls=r["cls"] if r["cls"] in CLS else "other", sub=r["cls"], drive_g=sel[r["scene_id"]]["drive_g"],
                d402=r["d402"], in_d3=str(r["scene_id"] in d3)) for r in rows]
    with open(a.out, "w") as f:
        w = csv.DictWriter(f, list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)
    print(len(out), "scenes;", Counter(r["sub"] for r in out), "; in D3", sum(r["in_d3"] == "True" for r in out))


def lists(a):
    S = read_tsv(a.scenes)
    new = [r for r in S if r["in_d3"] == "False"]
    by = defaultdict(list)
    for r in new:
        by[r["cls"]].append(r["scene_id"])
    first = [by[c].pop(0) for c in CLS]                                          # c000: one new scene per class
    d3 = list(csv.DictReader(open(REPO / "experiments/round5/results/d3/d3_scores.csv")))
    chk = [next(r["scene"] for r in d3 if r["info"] == "1" and r["cls"] == "launch" and r["winners"] == "F"),
           next(r["scene"] for r in d3 if r["info"] == "1" and r["cls"] == "turn_sharp" and "A" not in r["winners"])]
    inter = []
    for i in range(max(map(len, by.values()))):
        inter += [by[c][i] for c in CLS if i < len(by[c])]
    L = Path(a.root) / "lists"
    L.mkdir(parents=True, exist_ok=True)
    (L / "c000.txt").write_text("\n".join(first + chk) + "\n")
    for i in range(0, len(inter), 25):
        (L / f"c{i // 25 + 1:03d}.txt").write_text("\n".join(inter[i:i + 25]) + "\n")
    dump(L / "d3_check.json", dict(scenes=chk))
    print(len(first) + len(inter), "new scenes in", 1 + (len(inter) + 24) // 25, "chunks; D3 check scenes", chk)


def runtimes(root):
    """-> [(runtime s, scored scenes)] of every finished chunk of a pai_box root."""
    out = []
    for sim in sorted(glob.glob(f"{root}/runs/*/sim")):
        f, log = Path(sim) / "aggregate/results-summary.json", Path(sim).parent / "pool/log.txt"
        if f.exists() and log.exists():
            m = re.findall(r"'runtime_s': ([0-9.]+)", log.read_text())
            if m:
                out.append((float(m[-1]), len(json.loads(f.read_text())["rollouts"])))
    return out


def check(a):
    O, D = Path(a.root), Path(a.d3)
    want = (O / "lists/c000.txt").read_text().split()
    chk = json.loads((O / "lists/d3_check.json").read_text())["scenes"]
    out, ok, why = {}, True, []
    per = []
    for arm in ARMS:
        R, bad, _ = D3.load(O / "roll" / arm)
        R3, _, _ = D3.load(D / "roll" / arm)
        n = sum(s in R for s in want)
        same = {s: (float(R[s].get("score") or 0), float(R3[s].get("score") or 0)) for s in chk if s in R and s in R3}
        out[arm] = dict(scored=n, listed=len(want), rpc=bad, d3_check=same)
        if n < len(want) - 1:
            ok, why = False, why + [f"{arm}: {n} of {len(want)} scored"]
        if len(same) < len(chk) or any(abs(x - y) >= 1e-3 for x, y in same.values()):
            ok, why = False, why + [f"{arm}: D3 check scenes {same} (want equal to D3's)"]
        per += [t / max(k, 1) for t, k in runtimes(O / "roll" / arm)]
    n_all = sum(len(f.read_text().split()) for f in (O / "lists").glob("c*.txt")) * len(ARMS)
    sps = float(np.mean(per)) if per else float("nan")
    out |= dict(s_per_scene=sps, card_h_projected=sps * n_all / 3600 / CARD_STACKS, rollouts=n_all)
    if not sps <= 120.0:                                                         # D3: about 40 stack-s a scene in 25-scene chunks; c000's 6 carry the start-up
        ok, why = False, why + [f"{sps:.0f} stack-s a scene on c000 (> 120: about 3 x D3's)"]
    out["ok"], out["why"] = ok, why
    dump(O / "checks/c000.json", out)
    print(json.dumps(out, default=str)[:2000])
    if not ok:
        (O / "VERDICT").write_text(f"STOP c000 check: {'; '.join(why)}\n")
        sys.exit(1)


# ---------------------------------------------------------------- selection
def otype_of(sc, win, info):
    if win != "A":
        return "zero-rescued" if sc["A"] == 0 else "partial-fixed"
    return "A-wins" if info else ""


def winner(sc):
    best = max(sc.values())
    if best - sc["A"] >= ADV - 1e-9:
        return next(k for k in ARMS[1:] if sc[k] >= best - 1e-9)
    return "A"


def select(a):
    from jevdrive import stats
    O, D = Path(a.root), Path(a.d3)
    meta = {r["scene_id"]: r for r in read_tsv(a.scenes)}
    R = {}
    for arm in ARMS:
        new, _, _ = D3.load(O / "roll" / arm)
        old, _, _ = D3.load(D / "roll" / arm)
        R[arm] = {s: (old[s] if meta[s]["in_d3"] == "True" else new.get(s)) for s in meta}
    sc_ok = [s for s in meta if all(R[k][s] is not None for k in ARMS)]
    missing = {k: sorted(s for s in meta if R[k][s] is None) for k in ARMS}
    rows, sel = [], {}
    for s in sc_ok:
        m = meta[s]
        sc = {k: float(R[k][s].get("score") or 0.0) for k in ARMS}
        hi, lo = max(sc.values()), min(sc.values())
        w = winner(sc)
        info = hi - lo >= INFO - 1e-9
        ot = otype_of(sc, w, info)
        worst = min(ARMS, key=lambda k: (sc[k], k != "A"))
        row = dict(scene=s, cls=m["cls"], sub=m["sub"], drive_g=m["drive_g"], in_d3=m["in_d3"], **{f"s_{k}": round(v, 4) for k, v in sc.items()},
                   **{f"t_{k}": failure_type(R[k][s]) for k in ARMS}, spread=round(hi - lo, 4), info=int(info), adv=round(hi - sc["A"], 4),
                   winner=w, worst=worst, otype=ot)
        rows.append(row)
        sel[s] = dict(row, drv={k: R[k][s]["_drv"] for k in ARMS})
    out = O / "report"
    out.mkdir(parents=True, exist_ok=True)
    stats.write_table(rows, out / "s3_scores")
    dump(O / "sel.json", sel)
    # ---- tables
    G = {r["scene"]: r["drive_g"] for r in rows}
    T = {}
    for c in CLS + ("all",):
        Rc = [r for r in rows if c in ("all", r["cls"])]
        sc = {k: np.array([r[f"s_{k}"] for r in Rc]) for k in ARMS}
        g = [G[r["scene"]] for r in Rc]
        inf = [r for r in Rc if r["info"]]
        T[c] = dict(n=len(Rc), info=len(inf), info_share=len(inf) / max(len(Rc), 1), mean={k: float(sc[k].mean()) for k in ARMS},
                    oracle=float(np.max(np.stack(list(sc.values())), 0).mean()), zeros={k: int((sc[k] == 0).sum()) for k in ARMS},
                    vs_A={k: stats.paired(sc[k], sc["A"], groups=g) for k in ARMS[1:]},
                    flip={k: dict(up=int((sc[k] - sc["A"] >= INFO).sum()), down=int((sc["A"] - sc[k] >= INFO).sum())) for k in ARMS[1:]},
                    winner={k: sum(r["winner"] == k and r["otype"] != "" for r in Rc) for k in ARMS},
                    otype={t: sum(r["otype"] == t for r in Rc) for t in OTYPES},
                    otype_arm={t: dict(Counter(r["winner"] for r in Rc if r["otype"] == t)) for t in OTYPES})
    S = dict(n=len(rows), missing=missing, d3_scenes=sum(r["in_d3"] == "True" for r in rows), classes=T,
             a_outcome={o: dict(n=len(x), winner=dict(Counter(r["winner"] for r in x))) for o, x in (
                 ("full", [r for r in rows if r["s_A"] >= FULL]), ("partial", [r for r in rows if 0 < r["s_A"] < FULL]), ("zero", [r for r in rows if r["s_A"] == 0]))})
    rt = {arm: runtimes(O / "roll" / arm) for arm in ARMS}
    stack_h = sum(t for v in rt.values() for t, _ in v) / 3600
    S["cost_stage1"] = dict(stack_h=stack_h, card_h=stack_h / CARD_STACKS, chunks=sum(len(v) for v in rt.values()))
    dump(out / "summary.json", S)
    f = lambda r: f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"  # noqa: E731
    L = [f"# S3 stage 1: {S['n']} scenes x 4 arms scored ({S['d3_scenes']} from D3's roll); generated by s3.py select", "",
         f"Winner: the best non-A arm when best - A >= {ADV} (ties F, S, L), else A. Informative: best - worst >= {INFO}.", "",
         "| class | n | informative | A mean | oracle | F - A | S - A | L - A | F up/down | S up/down | L up/down | zeros A/F/S/L |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for c, e in T.items():
        L.append(f"| {c} | {e['n']} | {e['info']} ({e['info_share']:.0%}) | {e['mean']['A']:.3f} | {e['oracle']:.3f} | "
                 + " | ".join(f(e["vs_A"][k]) for k in ARMS[1:]) + " | " + " | ".join(f"{e['flip'][k]['up']}/{e['flip'][k]['down']}" for k in ARMS[1:])
                 + " | " + "/".join(str(e["zeros"][k]) for k in ARMS) + " |")
    L += ["", "Row scenes by outcome type (winner arm counts in brackets):", "", "| class | " + " | ".join(OTYPES) + " | none |", "|---|--:|--:|--:|--:|"]
    for c, e in T.items():
        L.append(f"| {c} | " + " | ".join(f"{e['otype'][t]} ({', '.join(f'{k} {v}' for k, v in sorted(e['otype_arm'][t].items()))})" for t in OTYPES)
                 + f" | {e['n'] - sum(e['otype'].values())} |")
    L += ["", "Winner by A's own outcome: " + "; ".join(f"{o} {v['n']}: {v['winner']}" for o, v in S["a_outcome"].items()),
          f"Stage-1 cost (new rolls only): {S['cost_stage1']['card_h']:.2f} card-h ({S['cost_stage1']['chunks']} chunks). Missing: { {k: len(v) for k, v in missing.items()} }."]
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    # ---- re-roll lists: r000 = 3 check scenes (an F winner, a S / L winner, an A-wins scene), then 25 a chunk per arm
    take = [r for r in rows if r["otype"]]
    chk = []
    for want in (("F",), ("S", "L"), ("A",)):
        x = next((r for r in take if r["winner"] in want and r not in chk), None)
        if x is not None:
            chk.append(x)
    for arm in ARMS:
        d = O / "reroll" / arm / "lists"
        d.mkdir(parents=True, exist_ok=True)
        c0 = [r["scene"] for r in chk if r["winner"] == arm]
        rest = [r["scene"] for r in take if r["winner"] == arm and r["scene"] not in c0]
        if c0:
            (d / "r000.txt").write_text("\n".join(c0) + "\n")
        for i in range(0, len(rest), 25):
            (d / f"r{i // 25 + 1:03d}.txt").write_text("\n".join(rest[i:i + 25]) + "\n")
    dump(O / "reroll/check.json", dict(scenes=[r["scene"] for r in chk], arms=[r["winner"] for r in chk]))
    print("re-roll:", dict(Counter(r["winner"] for r in take)), "check", [(r["scene"][-8:], r["winner"]) for r in chk])


# ---------------------------------------------------------------- drive.jsonl records (served poses per decision)
_J = {}


def served(path, scene):
    """drive.jsonl of a roll chunk -> {k: (now, anchor (3,), poses (8, 3) served, the model's own plan (8, 3))} of the scene's inference
    decisions (raw 4-decimal records; the model's plan = poses_own when TTS switched, else poses_model); raises when the scene has more than
    one session in the file."""
    if path not in _J:
        D, ses = defaultdict(dict), defaultdict(set)
        for line in open(path):
            if '"infer": true' not in line:
                continue
            r = json.loads(line)
            if r.get("kind") == "drive" and r.get("infer"):
                D[r["scene"]][r["k"]] = (r["now"], np.array(r["anchor"], np.float64), np.array(r["poses"], np.float64),
                                         np.array(r.get("poses_own") or r.get("poses_model") or r["poses"], np.float64))
                ses[r["scene"]].add(r["session"])
        _J[path] = (D, ses)
    D, ses = _J[path]
    assert len(ses[scene]) == 1, f"{path}: {scene} has {len(ses[scene])} sessions"
    return D[scene]


def r1(a):
    O = Path(a.root)
    sel = json.loads((O / "sel.json").read_text())
    res, bad = {}, []
    for arm in ARMS:
        R, rpc, _ = D3.load(O / "reroll" / arm)
        for s, r in R.items():
            if a.chunk and Path(r["_drv"]).parts[-4] != f"ntv0-l18-s0_{a.chunk}":
                continue
            e = sel[s]
            assert e["winner"] == arm, (s, arm)
            x, y = served(e["drv"][arm], s), served(r["_drv"], s)
            ks = sorted(set(x) | set(y))
            dp = max((float(np.abs(x[k][2] - y[k][2]).max()) for k in ks if k in x and k in y), default=float("inf"))
            da = max((float(np.abs(x[k][1] - y[k][1]).max()) for k in ks if k in x and k in y), default=float("inf"))
            same = dict(score=[e[f"s_{arm}"], float(r.get("score") or 0)], decisions=[len(x), len(y)], keys_equal=set(x) == set(y),
                        poses_max_abs=dp, anchor_max_abs=da)
            same["ok"] = bool(abs(same["score"][0] - same["score"][1]) < 1e-3 and same["keys_equal"] and dp == 0 and da == 0)
            res[s] = dict(arm=arm, **same)
            if not same["ok"]:
                bad.append(s)
    want = json.loads((O / "reroll/check.json").read_text())["scenes"] if a.chunk == "r000" else [s for s, e in sel.items() if e["otype"]]
    lost = sorted(set(want) - set(res))
    ok = not bad and not lost if a.chunk == "r000" else True
    out = dict(chunk=a.chunk or "all", scenes=len(res), reproduced=len(res) - len(bad), mismatched=bad, not_rerolled=lost, ok=ok, per_scene=res)
    dump(O / "checks" / f"r1_{a.chunk or 'all'}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_scene"}, default=str))
    if not a.chunk:                                                              # the pool's scenes: re-rolled, reproduced
        keep = sorted(s for s, v in res.items() if v["ok"])
        (O / "keep.tsv").write_text("scene_id\n" + "".join(f"{s}\n" for s in keep))
        print(len(keep), "scenes kept")
    if not ok:
        (O / "VERDICT").write_text(f"STOP R1 ({a.chunk}): mismatched {bad}, not re-rolled {lost}\n")
        sys.exit(1)


# ---------------------------------------------------------------- pool tags, R2
def pool_rows(pool):
    tb = dict(np.load(CR / pool / "tab.npz"))
    nm = tb["names"].astype(str)
    return tb, np.array([n.split(":")[2] for n in nm]), np.array([int(n.split(":")[3]) for n in nm])


def tag(a):
    O = Path(a.root)
    sel = json.loads((O / "sel.json").read_text())
    tb, sc, kk = pool_rows(a.pool)
    n = len(sc)
    ot, cl, sb, am = (np.empty(n, "U16") for _ in range(4))             # tab keys otype / ocls / osub / oarm (U strings survive rf_pait merges)
    X = dict(names=tb["names"], k=kk.astype(np.int32), post_hand=np.zeros(n, bool), slot2=(kk % 5 == 0), k_div_worst=np.full(n, -1, np.int32),
             k_div_A=np.full(n, -1, np.int32), s_win=np.zeros(n, np.float32), s_A=np.zeros(n, np.float32), adv=np.zeros(n, np.float32),
             served_A=np.full((n, 8, 3), np.nan, np.float32), anchor_A=np.full((n, 3), np.nan, np.float32), anchor=np.zeros((n, 3), np.float32),
             has_A=np.zeros(n, bool), same_state_A=np.zeros(n, bool))
    for s in np.unique(sc):
        e, i = sel[s], np.flatnonzero(sc == s)
        w = e["winner"]
        ot[i], cl[i], sb[i], am[i] = e["otype"], e["cls"], e["sub"], w
        W, A, Wo = served(e["drv"][w], s), served(e["drv"]["A"], s), served(e["drv"][e["worst"]], s)
        t_first = min(v[0] for v in W.values())
        div = lambda P, Q: next((k for k in sorted(P) if k in Q and np.hypot(*(P[k][1][:2] - Q[k][1][:2])) > TOL), -1)  # noqa: E731
        X["k_div_worst"][i], X["k_div_A"][i] = div(W, Wo) if w != e["worst"] else -1, div(W, A) if w != "A" else -1
        X["s_win"][i], X["s_A"][i], X["adv"][i] = e[f"s_{w}"], e["s_A"], e["adv"]
        for j in i:
            k = int(kk[j])
            X["post_hand"][j] = (W[k][0] - t_first) * 1e-6 >= T_HAND
            X["anchor"][j] = W[k][1]
            if k in A:
                X["has_A"][j], X["served_A"][j], X["anchor_A"][j] = True, A[k][2], A[k][1]
                X["same_state_A"][j] = np.hypot(*(A[k][1][:2] - W[k][1][:2])) <= TOL
    assert (ot != "").all(), "a pool row of a scene without an outcome type"
    tb |= dict(otype=ot, ocls=cl, osub=sb, oarm=am)
    np.savez(CR / a.pool / "tab.npz", **tb)
    np.savez(CR / a.pool / "outcome.npz", **X)
    print(json.dumps(dict(rows=n, scenes=int(len(np.unique(sc))), otype=dict(Counter(ot.tolist())), arm=dict(Counter(am.tolist())))))


def r2(a):
    """R2: the stored label of every row (fut) against the served poses of the same decision in the stage-1 roll that set the outcome."""
    O = Path(a.root)
    sel = json.loads((O / "sel.json").read_text())
    tb, sc, kk = pool_rows(a.pool)
    fut = tb["fut"].astype(np.float64)
    d, miss = np.full(len(sc), np.inf), 0
    for s in np.unique(sc):
        e = sel[s]
        W = served(e["drv"][e["winner"]], s)
        for j in np.flatnonzero(sc == s):
            k = int(kk[j])
            if k in W:
                d[j] = float(np.abs(fut[j] - W[k][2]).max())
            else:
                miss += 1
    fin = np.isfinite(d)
    out = dict(pool=a.pool, rows=len(d), missing_decision=miss, max_abs=float(d[fin].max()) if fin.any() else None, bitwise_share=float((d == 0).mean()),
               over_1mm=int((d >= 1e-3).sum()), fut_eq_served_tab=bool(np.array_equal(tb["fut"], tb["served"])))
    out["ok"] = bool(miss == 0 and out["over_1mm"] == 0 and out["fut_eq_served_tab"])
    dump(O / "checks/r2.json", out)
    print(json.dumps(out))
    if not out["ok"]:
        (O / "VERDICT").write_text(f"STOP R2: {out}\n")
        sys.exit(1)


def r1in(a):
    """R1 input path (pp.py r1's line): the trainer-path forward of --tag on the pool's tokens + ego (rf_recov fwd plans) against the model's own
    plan of the decision in drive.jsonl (poses_own where TTS switched: there the driver's poses_model is the TTS candidate), and a wrong checkpoint."""
    O = Path(a.root)
    sel = json.loads((O / "sel.json").read_text())
    tb, sc, kk = pool_rows(a.pool)
    ref = np.full((len(sc), 8, 2), np.nan)
    for s in np.unique(sc):
        e = sel[s]
        W = served(e["drv"][e["winner"]], s)
        for j in np.flatnonzero(sc == s):
            ref[j] = W[int(kk[j])][3][:, :2]
    tol = np.maximum(4 * 2.0 ** (np.floor(np.log2(np.maximum(np.abs(ref), 2.0 ** -14))) - 10), 0.03)
    res = {}
    for t in (a.tag, a.control):
        z = np.load(Path(a.fwd) / f"plans_{t}.npz")
        P = np.full(ref.shape, np.nan)
        P[z["rows"]] = z["poses"][:, :8, :2]
        d = np.abs(P - ref).reshape(len(ref), -1).max(1)
        over = (np.abs(P - ref) > tol).reshape(len(ref), -1).any(1)
        res[t] = dict(share_over=float(over.mean()), median_m=float(np.nanmedian(d)), p99_m=float(np.nanpercentile(d, 99)), rows=int(len(d)),
                      missing=int(np.isnan(d).sum()), by_arm={x: float(over[tb["oarm"] == x].mean()) for x in np.unique(tb["oarm"])})
    out = dict(pool=a.pool, tag=a.tag, control=a.control, model=res[a.tag], wrong=res[a.control])
    out["ok"] = bool(res[a.tag]["missing"] == 0 and res[a.tag]["median_m"] <= 0.02 and res[a.tag]["share_over"] <= 0.10 and res[a.control]["share_over"] >= 0.9)
    dump(O / "checks/r1in.json", out)
    print(json.dumps(out))
    if not out["ok"]:
        (O / "VERDICT").write_text(f"STOP R1 input path: {out}\n")
        sys.exit(1)


# ---------------------------------------------------------------- R3: the merged pool and its draw file
def same(x, y):
    """Bit-equal arrays, NaN equal to NaN where the dtype has one."""
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")


def mfix(a):
    from jevdrive.tokens import open_front
    P, R = CR / a.pool, CR / a.ref
    tp, tr = dict(np.load(P / "tab.npz")), np.load(R / "tab.npz")
    n = len(tr["names"])
    bad = [k for k in tr.files if k in tp and not same(tp[k][:n], tr[k])]
    bad += [f"tab.{k} missing" for k in tr.files if k not in tp]
    for f in ("agent.npz", "road.npz", "drivable.npz", "route.npz"):
        if not (R / f).exists():
            continue
        zp, zr = np.load(P / f), np.load(R / f)
        for k in zr.files:
            if k == "ok" and f == "agent.npz":
                continue
            if zr[k].ndim and len(zr[k]) == n and not same(zp[k][:n], zr[k]):
                bad.append(f"{f}.{k}")
    feat = DATA / "runs/route_mem/feat"
    fp, fr = np.load(feat / f"{a.pool}.npz"), np.load(feat / f"{a.ref}.npz")
    bad += [f"feat.{k}" for k in fr.files if not same(fp[k][:n], fr[k])]
    zp, zr = np.load(CR / f"{a.pool}@warp/teacher.npz"), np.load(CR / f"{a.ref}@warp/teacher.npz")
    bad += [f"teacher.{k}" for k in zr.files if not same(zp[k][:n] if zr[k].ndim and len(zr[k]) == n else zp[k], zr[k])]
    i = np.sort(np.random.default_rng(0).choice(n, 4000, replace=False))
    Fp, Fr = open_front(CR / f"{a.pool}@warp/front.npy"), open_front(CR / f"{a.ref}@warp/front.npy")
    if not same(Fp[i], Fr[i]):
        bad.append("front sample")
    ag, agr = dict(np.load(P / "agent.npz")), np.load(R / "agent.npz")
    before = int((ag["ok"][:n] != agr["ok"]).sum())
    ag["ok"] = np.concatenate([agr["ok"], ag["ok"][n:]])                         # r2.py fix's PAI-human ok False, as the reference holds it
    np.savez(P / "agent.npz", **ag)
    part = tp["pool"].astype(str)
    rec = dict(pool=a.pool, ref=a.ref, ref_rows=n, rows=len(part), parts={p: int((part == p).sum()) for p in np.unique(part)}, agent_ok_fixed=before,
               new_rows_mode=sorted(set(tp["mode"][n:].astype(str).tolist())), new_agent_ok=int(ag["ok"][n:].sum()), mismatch=bad)
    dump(DATA / "runs/round5/s3/checks" / f"mfix_{a.pool}.json", rec)
    print(json.dumps(rec))
    if bad:
        sys.exit(2)


def weights(a):
    W = DATA / "runs/defect_fix/w"
    names = np.load(CR / a.pool / "tab.npz")["names"].astype(str)
    B = np.load(W / f"{a.base}.npz")
    n = len(B["names"])
    assert np.array_equal(B["names"].astype(str), names[:n]), f"{a.base}: rows != the pool's first rows"
    w = np.concatenate([B["w"], np.zeros(len(names) - n, B["w"].dtype)])
    grp = np.concatenate([B["grp"].astype(str), np.full(len(names) - n, "paio")]) if "grp" in B.files else None
    np.savez(W / f"{a.out}.npz", names=names, w=w, **({"grp": grp} if grp is not None else {}),
             **{k: B[k] for k in B.files if k not in ("names", "w", "grp")})
    print(json.dumps(dict(out=a.out, rows=len(names), base_rows=n, w0_new=len(names) - n)))


# ---------------------------------------------------------------- hindsight relabel (plan addendum 1): on1_PAIOR, label = the realised future
T_FUT = np.arange(1, 9) * 500_000                                                # onp_geom.T_FUT
L_LAUNCH, V_LAUNCH, L0_LINE = 0.5, 0.5, 4.2


def rig_n(p, A):
    """onp_rows.rig_n: local poses (N, m, 3) -> the rig frames at A (N, 3)."""
    c, s = np.cos(A[:, 2])[:, None], np.sin(A[:, 2])[:, None]
    dx, dy = p[..., 0] - A[:, None, 0], p[..., 1] - A[:, None, 1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, p[..., 2] - A[:, None, 2]], -1)


def path_len(fut):
    """r3.py path_len: fut (n, 8, 3) -> (n,) 4 s path length from the origin."""
    P = np.concatenate([np.zeros((len(fut), 1, 2)), fut[:, :, :2]], 1)
    return np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)


def realised(z, ks):
    """Row file z (the winner's rollout), decisions ks -> (fut (n, 8, 3) the realised rig poses at t0 + 0.5 .. 4 s in the t0 rig frame, NaN when
    incomplete; why: '' / 'end' / 'event'; max |target time - nearest decision| in us; the row file's anchors at ks)."""
    t0 = z["t0_us"].astype(np.int64)
    A = z["anchor"].astype(np.float64).copy()
    A[:, 2] = np.unwrap(A[:, 2])
    ev = int(np.asarray(z["t_event_us"]).min())
    t_end = min(int(t0[-1]), int(z["t_end_us"]))
    pos = {int(k): i for i, k in enumerate(z["k"])}
    i = np.array([pos[int(k)] for k in ks])
    tq = t0[i][:, None] + T_FUT[None]
    why = np.where(tq[:, -1] >= ev, "event", np.where(tq[:, -1] > t_end, "end", ""))
    P = np.stack([np.interp(tq, t0, A[:, c]) for c in range(3)], -1)
    fut = rig_n(P, A[i])
    fut[why != ""] = np.nan
    near = np.abs(t0[np.searchsorted(t0, tq).clip(0, len(t0) - 1)] - tq)
    return fut.astype(np.float32), why, int(near[why == ""].max()) if (why == "").any() else 0, A[i]


def relabel(a):
    from jevdrive.tokens import link_front
    O, src, dst = Path(a.root), CR / a.pool, CR / f"on1_{a.out}"                  # --out -: the gates and tables only (pre-flight), no pool
    assert a.out == "-" or not (dst.exists() or (CR / f"on1_{a.out}@warp").exists()), f"{dst} exists: never overwritten"
    tb, sc, kk = pool_rows(a.pool)
    X = np.load(src / "outcome.npz")
    assert np.array_equal(X["names"], tb["names"])
    n = len(sc)
    fut, why, d_anchor, gap, bad_link = np.full((n, 8, 3), np.nan, np.float32), np.full(n, "", "U5"), 0.0, 0, []
    arm = tb["oarm"].astype(str)
    for s in np.unique(sc):
        f = O / "rows/PAIO" / f"{s}.npz"
        if not Path(os.path.realpath(f)).parent.name.startswith(f"{arm[sc == s][0]}_r"):
            bad_link.append(s)
        j = np.flatnonzero(sc == s)
        with np.load(f) as z:
            fu, wh, g, An = realised(z, kk[j])
        fut[j], why[j], gap = fu, wh, max(gap, g)
        d_anchor = max(d_anchor, float(np.abs(An[:, :2] - X["anchor"][j, :2].astype(np.float64)).max()))
    ok = why == ""
    sv = tb["served"].astype(np.float64)
    v0, ot = tb["speed"].astype(np.float64), tb["otype"].astype(str)
    Lr, Ls = np.where(ok, path_len(np.nan_to_num(fut.astype(np.float64))), np.nan), path_len(sv)
    md = lambda x: [float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))] if len(x) else [float("nan")] * 3  # noqa: E731
    groups = {t: ot == t for t in OTYPES} | {"all but A-wins": ot != "A-wins", "all": np.ones(n, bool)}
    L0 = {}
    for g, m in groups.items():
        st = m & (v0 < V_LAUNCH)
        k_ = st & ok
        L0[g] = dict(rows_fed_lt_0p5=int(st.sum()), kept=int(k_.sum()),
                     realised=dict(stand_share=float((Lr[k_] < L_LAUNCH).mean()) if k_.any() else None, launch_rows=int((k_ & (Lr >= L_LAUNCH)).sum()),
                                   launch_path_med_p25_p75=md(Lr[k_ & (Lr >= L_LAUNCH)])),
                     served_same_rows=dict(stand_share=float((Ls[k_] < L_LAUNCH).mean()) if k_.any() else None, launch_rows=int((k_ & (Ls >= L_LAUNCH)).sum()),
                                           launch_path_med_p25_p75=md(Ls[k_ & (Ls >= L_LAUNCH)])),
                     served_all_rows=dict(stand_share=float((Ls[st] < L_LAUNCH).mean()) if st.any() else None, launch_path_med_p25_p75=md(Ls[st & (Ls >= L_LAUNCH)])))
    L1 = {}
    for g, m in groups.items():
        for sp, ms in (("fed < 0.5", v0 < V_LAUNCH), ("fed >= 0.5", v0 >= V_LAUNCH), ("any", np.ones(n, bool))):
            k_ = m & ms & ok
            d1, d4 = (np.hypot(*(fut[k_, i, :2] - sv[k_, i, :2]).T) for i in (1, 7))
            r = Lr[k_] / np.maximum(Ls[k_], 0.1)
            L1[f"{g} | {sp}"] = dict(rows=int(k_.sum()), d1s_med_p90=[float(np.median(d1)), float(np.percentile(d1, 90))] if k_.any() else None,
                                     d4s_med_p90=[float(np.median(d4)), float(np.percentile(d4, 90))] if k_.any() else None,
                                     path_ratio_med=float(np.median(r)) if k_.any() else None,
                                     path_med_realised_served=[float(np.median(Lr[k_])), float(np.median(Ls[k_]))] if k_.any() else None)
    drop = {t: dict(rows=int((ot == t).sum()), kept=int(((ot == t) & ok).sum()), end=int(((ot == t) & (why == "end")).sum()),
                    event=int(((ot == t) & (why == "event")).sum())) for t in OTYPES}
    l0 = L0["partial-fixed"]["realised"]["launch_path_med_p25_p75"][0]
    chk = dict(anchor_vs_drive_jsonl_m=d_anchor, max_target_time_gap_us=gap, row_file_not_winner=bad_link, kept_finite=bool(np.isfinite(fut[ok]).all()))
    passed = dict(L0=bool(l0 >= L0_LINE), anchor=d_anchor <= 1e-4, link=not bad_link, finite=chk["kept_finite"], kept_half=bool(ok.mean() >= 0.5))
    rec = dict(pool=a.pool, out=f"on1_{a.out}", rows=n, kept=int(ok.sum()), dropped=int((~ok).sum()), drop=drop, L0=L0, L1=L1, checks=chk,
               lines=dict(L0=f"partial-fixed launch rows' realised 4 s path median >= {L0_LINE} m"), passed=passed, ok=all(passed.values()),
               refs=dict(served_moving_median_d422=1.95, navtrain_launch=8.28, pai_human_launch_R3NS=4.16))
    res = REPO / "experiments/round5/results/s3"
    dump(res / "relabel.json", rec)
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    q = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
    Lm = [f"# S3 hindsight relabel: on1_PAIO -> on1_PAIOR (generated by s3.py relabel)", "",
          "Label = the winner rollout's realised rig poses at t0 + 0.5 .. 4 s in the t0 rig frame (onp_rows log_fut's transform on the rollout's own "
          "10 Hz anchors). Rows whose 4 s run past the rollout's end / scorer end / first scorer event keep fut = NaN (never drawn, never imitated).", "",
          f"Rows {n}, kept {rec['kept']}, dropped {rec['dropped']}: " + "; ".join(f"{t} {v['kept']} / {v['rows']} kept (end {v['end']}, event {v['event']})" for t, v in drop.items()), "",
          f"## L0: launch rows (fed < 0.5 m/s, 4 s path >= 0.5 m); line: partial-fixed realised median >= {L0_LINE} m -> **{'PASS' if passed['L0'] else 'FAIL'}** ({l0:.2f} m)", "",
          "Refs: served-plan movers 1.95 m (d422), navtrain launch 8.28 m, PAI human 4.16 m (R3NS).", "",
          "| rows | fed < 0.5 | kept | realised: stand share / launch rows / 4 s path median [p25, p75] | served, same rows | served, all fed < 0.5 rows |",
          "|:--|--:|--:|:--|:--|:--|"]
    for g, v in L0.items():
        r_, s_, s2 = v["realised"], v["served_same_rows"], v["served_all_rows"]
        Lm.append(f"| {g} | {v['rows_fed_lt_0p5']} | {v['kept']} | {f2(r_['stand_share'])} / {r_['launch_rows']} / {q(r_['launch_path_med_p25_p75'])} | "
                  f"{f2(s_['stand_share'])} / {s_['launch_rows']} / {q(s_['launch_path_med_p25_p75'])} | {f2(s2['stand_share'])} / {q(s2['launch_path_med_p25_p75'])} |")
    Lm += ["", "## L1: realised vs served plan (kept rows; position error in m)", "",
           "| rows | n | 1 s median / p90 | 4 s median / p90 | 4 s path realised / served (median ratio) | 4 s path median realised / served |", "|:--|--:|:--|:--|--:|:--|"]
    for g, v in L1.items():
        if v["rows"]:
            Lm.append(f"| {g} | {v['rows']} | {v['d1s_med_p90'][0]:.2f} / {v['d1s_med_p90'][1]:.2f} | {v['d4s_med_p90'][0]:.2f} / {v['d4s_med_p90'][1]:.2f} | "
                      f"{v['path_ratio_med']:.2f} | {v['path_med_realised_served'][0]:.2f} / {v['path_med_realised_served'][1]:.2f} |")
    Lm += ["", f"Checks: row-file anchor vs drive.jsonl anchor max {d_anchor:.2e} m; max target-time gap {gap} us; row files not the winner's: {len(bad_link)}; "
           f"kept labels finite {chk['kept_finite']}. Passed: {passed}"]
    (res / "relabel.md").write_text("\n".join(Lm) + "\n")
    print("\n".join(Lm))
    if a.out == "-":
        return
    if not rec["ok"]:
        (O / "VERDICT.relabel").write_text(f"STOP relabel: {passed}\n")
        sys.exit(1)
    # ---- the pool: same rows, names and files as --pool; tab.fut = the realised future (NaN = dropped); tokens zero-copy
    import shutil
    dst.mkdir(parents=True)
    w = Path(f"{dst}@warp")
    w.mkdir()
    for f in ("agent.npz", "road.npz", "route.npz", "outcome.npz", "build.json"):
        shutil.copy2(src / f, dst / f)
    shutil.copy2(Path(f"{src}@warp") / "teacher.npz", w / "teacher.npz")
    feat = DATA / "runs/route_mem/feat"
    shutil.copy2(feat / f"{a.pool}.npz", feat / f"on1_{a.out}.npz")
    link_front(w, [Path(f"{src}@warp") / "front.npy"])
    np.savez(dst / "tab.npz", **(tb | dict(fut=fut)))
    np.savez(dst / "relabel.npz", names=tb["names"], ok=ok, why=why, fut_served=tb["fut"], path_realised=Lr.astype(np.float32), path_served=Ls.astype(np.float32))
    dump(dst / "relabel.json", rec)
    print(json.dumps(dict(pool=str(dst), rows=n, kept=int(ok.sum()))))


def rcheck(a):
    """The relabelled merge against the original merge: every file bit-equal except tab.fut on the --src tail rows (= --src's fut)."""
    P, R, S = CR / a.pool, CR / a.ref, CR / a.src
    tp, tr, ts = np.load(P / "tab.npz"), np.load(R / "tab.npz"), np.load(S / "tab.npz")
    n, m = len(tr["names"]), len(ts["names"])
    bad = [k for k in tr.files if k != "fut" and not same(tp[k], tr[k])] + [f"tab.{k} only in one" for k in set(tp.files) ^ set(tr.files)]
    if not same(tp["fut"][:n - m], tr["fut"][:n - m]):
        bad.append("tab.fut prefix")
    if not same(tp["fut"][n - m:], ts["fut"]):
        bad.append("tab.fut tail != src fut")
    changed = int((~np.isclose(tp["fut"][n - m:], tr["fut"][n - m:], equal_nan=True)).any((1, 2)).sum())
    for f in ("agent.npz", "road.npz", "drivable.npz", "route.npz"):
        if (R / f).exists() or (P / f).exists():
            zp, zr = np.load(P / f), np.load(R / f)
            bad += [f"{f}.{k}" for k in set(zp.files) | set(zr.files) if k not in zp.files or k not in zr.files or not same(zp[k], zr[k])]
            del zp, zr
    feat = DATA / "runs/route_mem/feat"
    zp, zr = np.load(feat / f"{a.pool}.npz"), np.load(feat / f"{a.ref}.npz")
    bad += [f"feat.{k}" for k in zr.files if not same(zp[k], zr[k])]
    zp, zr = np.load(CR / f"{a.pool}@warp/teacher.npz"), np.load(CR / f"{a.ref}@warp/teacher.npz")
    bad += [f"teacher.{k}" for k in zr.files if not same(zp[k], zr[k])]
    jp, jr = (json.loads((CR / f"{x}@warp/front.parts.json").read_text()) for x in (a.pool, a.ref))
    if jp != jr:
        bad.append("front.parts.json")
    rec = dict(pool=a.pool, ref=a.ref, src=a.src, rows=n, tail=m, tail_fut_changed=changed, mismatch=bad, ok=not bad)
    dump(DATA / "runs/round5/s3/checks" / f"rcheck_{a.pool}.json", rec)
    dump(REPO / "experiments/round5/results/s3" / f"rcheck_{a.pool}.json", rec)
    print(json.dumps(rec))
    if bad:
        sys.exit(2)


# ---------------------------------------------------------------- report
def du(p):
    p = Path(p)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def report(a):
    O = Path(a.root)
    tb = np.load(CR / a.pool / "tab.npz")
    X = np.load(CR / a.pool / "outcome.npz")
    ot, cl = tb["otype"].astype(str), tb["ocls"].astype(str)
    sc = np.array([n.split(":")[2] for n in tb["names"].astype(str)])
    div = np.where(X["k_div_worst"] >= 0, X["k"] >= X["k_div_worst"], False)
    rows = {}
    for t in OTYPES + ("all",):
        for c in CLS + ("all",):
            m = (ot == t if t != "all" else np.ones(len(ot), bool)) & (cl == c if c != "all" else np.ones(len(cl), bool))
            rows[f"{t}|{c}"] = dict(scenes=int(len(np.unique(sc[m]))), rows=int(m.sum()), post_hand=int((m & X["post_hand"]).sum()),
                                    from_div=int((m & div).sum()), slot2=int((m & X["slot2"]).sum()), slot2_post_div=int((m & X["slot2"] & div).sum()),
                                    with_A=int((m & X["has_A"]).sum()), same_state_A=int((m & X["same_state_A"]).sum()))
    disk = {k: du(v) / 2 ** 30 for k, v in dict(stage1=O / "roll", reroll=O / "reroll", rows=O / "rows", cache=O / "cache", pool=CR / a.pool,
                                                 pool_warp=CR / f"{a.pool}@warp", merged=CR / "on1_JTPHO", merged_warp=CR / "on1_JTPHO@warp").items()}
    cost = {}
    for st in ("roll", "reroll"):
        h = sum(t for arm in ARMS for t, _ in runtimes(O / st / arm)) / 3600
        cost[st] = dict(stack_h=h, card_h=h / CARD_STACKS)
    ok = lambda d: next((d[k] for k in ("ok", "pass_", "pass", "identical") if k in d), None)  # noqa: E731
    rec = dict(pool=a.pool, rows=rows, disk_gb=disk, cost=cost, checks={f.stem: ok(json.loads(f.read_text())) for f in sorted((O / "checks").glob("*.json"))})
    dump(O / "report/rows.json", rec)
    L = [f"# S3 rows: pool {a.pool} (generated by s3.py report)", "",
         "Rows = the winner rollout's usable 10 Hz decisions before its first scorer event; label = the served plan. Each cell: rows (scenes); "
         "post-hand = after the 1.6 s ACC handoff; div = from the winner's divergence from the worst rollout; 2 Hz = k % 5 == 0.", "",
         "| outcome type | " + " | ".join(CLS) + " | all | post-hand | from div | 2 Hz | 2 Hz from div | with A at k |", "|---|" + "--:|" * (len(CLS) + 6)]
    for t in OTYPES + ("all",):
        L.append(f"| {t} | " + " | ".join(f"{rows[f'{t}|{c}']['rows']} ({rows[f'{t}|{c}']['scenes']})" for c in CLS + ("all",)) + " | "
                 + " | ".join(str(rows[f"{t}|all"][k]) for k in ("post_hand", "from_div", "slot2", "slot2_post_div", "with_A")) + " |")
    L += ["", "Disk (GiB): " + ", ".join(f"{k} {v:.1f}" for k, v in disk.items()),
          "Card-hours (rolls, 3 stacks a card): " + ", ".join(f"{k} {v['card_h']:.2f}" for k, v in cost.items()),
          "Checks: " + ", ".join(f"{k} {v}" for k, v in rec["checks"].items())]
    (O / "report/rows.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, args in dict(scenes=["out"], lists=["scenes", "root"], check=["root", "d3"], select=["root", "d3", "scenes"], r1=["root", "chunk?"],
                           tag=["root", "pool"], r2=["root", "pool"], r1in=["root", "pool", "fwd", "tag", "control"], mfix=["pool", "ref"], weights=["pool", "base", "out"], report=["root", "pool"],
                           relabel=["root", "pool", "out"], rcheck=["pool", "ref", "src"]).items():
        p = sp.add_parser(name)
        for x in args:
            p.add_argument(f"--{x.rstrip('?')}", required=not x.endswith("?"), default="")
    a = ap.parse_args()
    globals()[a.cmd](a)
