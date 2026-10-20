#!/usr/bin/env python3
"""D3 (round 5 diagnostic): is there outcome signal to learn from on PAI? NTV0-L18-s0 + ACC + GAP (g10, FAST=1) rolled K = 4 ways on
curated PAI train scenes; each rollout scored by the exam's scorer (results-summary.json); d3.sh runs the rolls.

  d3.py select --tsv experiments/pai_rows/results/scenes.tsv --sel experiments/pai_av/results/nurec_select.csv --out results/d3_scenes.tsv
                        Mac: the stratified draw (before any roll): non-held eligible scenes, one per drive group, order sha1(scene id)
  d3.py lists --scenes TSV --root O        box: the chunk lists (O/lists/cNNN.txt, 25 scenes, classes interleaved; O/lists/s000.txt = smoke)
  d3.py check --root O --stage smoke|c000  box: stage check, writes O/checks/<stage>.json; exit 1 = stop the chain
  d3.py report --root O --scenes TSV --out DIR   box: per-scene table (d3_scores.csv), summary.json, tables.md

Arms (d3.sh SV): A = the served config; F / S = the ACC targets scaled by s = 1.15 / 0.85 (set-speed gain JEV_ACC_VSET = s and curve limit
JEV_ACC_ALAT = 2.4 s^2, so the curve speed scales by s as well); L = route-seeking plan choice (JEV_TTS with weights w_route 1, all else 0,
margin 0.25 m: serve the TTS candidate of the model's own plan (lateral +-0.5 / 1 m, curvature gain 0.8-1.4, arc 0.9 / 1.1) whose mean distance
to the fed route is >= 0.25 m smaller than the plan's). Every knob exists in the driver; none is new serve logic.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/wait_cap/scripts")]

ARMS = ("A", "F", "S", "L")
NAME = {"A": "unperturbed", "F": "faster x1.15", "S": "slower x0.85", "L": "route-seeking"}
CLS = ("stop_nolead", "turn_sharp", "launch", "other")
N_CLS = {"stop_nolead": 38, "turn_sharp": 38, "launch": 38, "other": 36}   # other = bend 18 + rest 18
INFO = 0.2                                                                   # informative: best - worst >= 0.2
T_HAND = 1.6                                                                 # s since the session's first call (serve_fix.ACC_T_HAND)
CARD_STACKS = 3                                                              # stacks per card (the PAI-PARTIALS convention)


# ---------------------------------------------------------------- selection (Mac)
def select(a):
    sel = {r["scene_id"]: r for r in csv.DictReader(open(a.sel))}
    rows = [r for r in csv.DictReader(open(a.tsv), delimiter="\t") if r["held"] == "False"]
    h = lambda s: hashlib.sha1(s.encode()).hexdigest()  # noqa: E731
    rows.sort(key=lambda r: h(r["scene_id"]))
    want = dict(N_CLS)
    want.pop("other")
    want.update(bend=18, rest=18)
    used_g, out = set(), []
    for c in ("launch", "stop_nolead", "turn_sharp", "bend", "rest"):            # the smallest class first: it gets first pick of drive groups
        n = 0
        for r in rows:
            g = sel[r["scene_id"]]["drive_g"]
            if r["cls"] != c or g in used_g or n >= want[c]:
                continue
            used_g.add(g)
            n += 1
            out.append(dict(scene_id=r["scene_id"], cls=c if c in CLS else "other", sub=c, drive_g=g, d402=r["d402"],
                            country=sel[r["scene_id"]]["country"], night=sel[r["scene_id"]]["night"]))
        assert n == want[c], f"{c}: {n} of {want[c]}"
    with open(a.out, "w") as f:
        w = csv.DictWriter(f, list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)
    print(Counter(r["sub"] for r in out), len(out), "scenes,", len(used_g), "drive groups")


def lists(a):
    S = [r for r in csv.DictReader(open(a.scenes), delimiter="\t")]
    by = defaultdict(list)
    for r in S:
        by[r["cls"]].append(r["scene_id"])
    inter = []                                                                   # classes interleaved, so every chunk is a mix
    for i in range(max(map(len, by.values()))):
        inter += [by[c][i] for c in CLS if i < len(by[c])]
    L = Path(a.root) / "lists"
    L.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(inter), 25):
        (L / f"c{i // 25:03d}.txt").write_text("\n".join(inter[i:i + 25]) + "\n")
    smoke = next(r["scene_id"] for r in S if r["cls"] == "launch" and r["d402"] == "True")
    (L / "s000.txt").write_text(smoke + "\n")
    print(len(inter), "scenes in", (len(inter) + 24) // 25, "chunks; smoke", smoke)


# ---------------------------------------------------------------- reading a roll root
def load(root):
    """root = <arm root> (pai_box.sh layout) -> {scene: rollout row}, {scene: rpc reason}, [chunk runtime s]."""
    R, bad, rt = {}, {}, []
    for sim in sorted(glob.glob(f"{root}/runs/*/sim")):
        f = Path(sim) / "aggregate/results-summary.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text())["rollouts"]:
            s = r["clipgt_id"]
            if "RpcError" in str(r.get("failure_reason") or ""):
                bad[s] = str(r["failure_reason"]).splitlines()[0][:60]
                continue
            r["_drv"] = str(Path(sim) / "driver-logs/drive.jsonl")
            R.setdefault(s, r)
        log = Path(sim).parent / "pool/log.txt"
        if log.exists():
            m = re.findall(r"'runtime_s': ([0-9.]+)", log.read_text())
            if m:
                rt.append(float(m[-1]))
    return R, bad, rt


def drive_recs(path):
    """drive.jsonl -> {scene: [slim inference records sorted by k]}: k, now, anchor, the model plan's 2-4 s speed (vp), fix.acc, tts choice."""
    D = defaultdict(list)
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") == "drive" and r.get("infer") and "anchor" in r:
            p = np.r_[np.zeros((1, 2)), np.asarray(r.get("poses_model") or r["poses"], float)[:, :2]]
            D[r["scene"]].append(dict(k=r["k"], now=r["now"], anchor=r["anchor"], vp=float((np.hypot(*np.diff(p, axis=0).T) / 0.5)[4:].mean()),
                                      acc=(r.get("fix") or {}).get("acc") or {}, tts=(r["tts"].get("k") if isinstance(r.get("tts"), dict) else None),
                                      has_tts="tts" in r))
    for v in D.values():
        v.sort(key=lambda r: r["k"])
    return D


_DRV = {}


def recs(r):
    p = r["_drv"]
    if p not in _DRV:
        _DRV[p] = drive_recs(p)
    return _DRV[p].get(r["clipgt_id"], [])


def post_hand(rs):
    if not rs:
        return []
    t0 = rs[0]["now"]
    return [x for x in rs if (x["now"] - t0) * 1e-6 >= T_HAND]


def acc(x):
    return x["acc"]


def plan_speed(x):
    return x["vp"]


def activity(arm, rs):
    """Share of post-handoff decisions where the arm's knob acts on the served plan (A: none)."""
    rs = post_hand(rs)
    if not rs or arm == "A":
        return float("nan")
    if arm == "L":
        return float(np.mean([bool(x["tts"]) for x in rs]))
    s = 1.15 if arm == "F" else 0.85
    on = []
    for x in rs:
        a = acc(x)
        vh = a.get("v_hand")
        on.append(vh is not None and (s * plan_speed(x) > vh + 0.05 if arm == "F" else plan_speed(x) > vh + 0.05) or a.get("won") == "curve")
    return float(np.mean(on))


def diverge(ra, rb, tol=0.05):
    """First decision index (k) where two rollouts' ego anchors differ by > tol m, None if never."""
    A = {x["k"]: x["anchor"] for x in ra}
    for x in rb:
        y = A.get(x["k"])
        if y is not None and np.hypot(x["anchor"][0] - y[0], x["anchor"][1] - y[1]) > tol:
            return x["k"]
    return None


# ---------------------------------------------------------------- checks
def check(a):
    O = Path(a.root)
    st = a.stage
    base = O / ("smoke" if st == "smoke" else "roll")
    out, ok, why = {"stage": st}, True, []
    data = {}
    for arm in ARMS:
        R, bad, rt = load(base / arm)
        data[arm] = R
        n_list = len((O / "lists" / ("s000.txt" if st == "smoke" else f"{st}.txt")).read_text().split())
        out[arm] = dict(scored=len(R), rpc=len(bad), listed=n_list, runtime_s=rt, mean=float(np.mean([float(r.get("score") or 0) for r in R.values()])) if R else None)
        need = n_list if st == "smoke" else n_list - 2
        if len(R) < need:
            ok = False
            why.append(f"{arm}: {len(R)} scored of {n_list}")
    if ok:
        acts = {arm: [activity(arm, recs(r)) for r in data[arm].values()] for arm in ARMS if arm != "A"}
        out["activity"] = {k: float(np.nanmean(v)) for k, v in acts.items()}
        if st == "smoke":                                                        # the knob reached the driver: at the first post-handoff
            s = next(iter(data["A"]))                                            # decision with a plan above the handoff speed the states
            ra = post_hand(recs(data["A"][s]))                                   # are still identical, so v_set(F) / v_set(A) = 1.15 there
            ratio = {}
            for arm, g in (("F", 1.15), ("S", 0.85)):
                rb = {x["k"]: x for x in post_hand(recs(data[arm][s]))}
                for x in ra:
                    vh, vs = acc(x).get("v_hand"), acc(x).get("v_set")
                    if vh is not None and vs is not None and vs > vh + 0.3 and x["k"] in rb and \
                            np.allclose(rb[x["k"]]["anchor"], x["anchor"], atol=1e-3) and acc(rb[x["k"]]).get("v_set") is not None:
                        ratio[arm] = (round(acc(rb[x["k"]])["v_set"] / vs, 4), g)
                        break
            out["vset_ratio"] = ratio
            for arm, (q, g) in ratio.items():
                if abs(q - g) > 0.02 and abs(q - 1) > 1e-3:                      # 1 = v_hand binds in the arm (S below the handoff speed)
                    ok = False
                    why.append(f"{arm}: v_set ratio {q} (want {g})")
            if "F" not in ratio:
                why.append("note: no decision with v_set above v_hand: the F ratio was not testable")
            nt = sum(x["has_tts"] for x in recs(data["L"][s]))
            out["L_tts_records"] = nt
            if nt == 0:
                ok = False
                why.append("L: no tts record in drive.jsonl (JEV_TTS not active)")
            ref = stored_A()
            out["A_vs_stored_d402"] = {k: (float(data["A"][k].get("score") or 0), ref.get(k)) for k in data["A"]}
        else:
            per = [t / max(out[arm]["scored"], 1) for arm in ARMS for t in out[arm]["runtime_s"]]
            sc = float(np.mean(per)) if per else 60.0
            n_all = sum(len(f.read_text().split()) for f in (O / "lists").glob("c*.txt")) * len(ARMS)
            out["s_per_scene"] = sc
            out["card_h_projected"] = sc * n_all / 3600 / CARD_STACKS
            if out["card_h_projected"] > 6.0:
                ok = False
                why.append(f"projected {out['card_h_projected']:.2f} card-h > 6")
            if out["activity"].get("L", 0) == 0:
                why.append("note: L never switched on c000")
    out["ok"], out["why"] = ok, why
    (O / "checks").mkdir(exist_ok=True)
    (O / "checks" / f"{st}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("ok", "why")}, default=str))
    if not ok:
        (O / "VERDICT").write_text(f"STOP {st} check: {'; '.join(why)}\n")
        sys.exit(1)


def stored_A():
    R = {}
    for root in glob.glob("/root/autodl-tmp/ujs/runs/practice_pai/curated/roll/c*"):
        r, _, _ = load(root)
        R.update({k: float(v.get("score") or 0) for k, v in r.items()})
    return R


# ---------------------------------------------------------------- report
def report(a):
    from jevdrive import stats
    from nx_regress import failure_type

    O, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {r["scene_id"]: r for r in csv.DictReader(open(a.scenes), delimiter="\t")}
    D, BAD, RT = {}, {}, {}
    for arm in ARMS:
        D[arm], BAD[arm], RT[arm] = load(O / "roll" / arm)
    scenes = [s for s in meta if all(s in D[arm] for arm in ARMS)]
    ref = stored_A()
    rows = []
    for s in scenes:                                                             # chunk by chunk: drive logs are read once per chunk
        m = meta[s]
        sc = {arm: float(D[arm][s].get("score") or 0.0) for arm in ARMS}
        hi, lo = max(sc.values()), min(sc.values())
        win = [arm for arm in ARMS if sc[arm] >= hi - 1e-9]
        rr = {}
        for arm in ARMS:
            rr[arm] = recs(D[arm][s])
        n_post = {arm: len(post_hand(rr[arm])) for arm in ARMS}
        worst = min(ARMS, key=lambda k: (sc[k], k != "A"))
        w0 = win[0] if "A" not in win else "A"
        kd = diverge(rr[w0], rr[worst]) if w0 != worst else None
        n_div = sum(x["k"] >= kd for x in post_hand(rr[w0])) if kd is not None else 0
        row = dict(scene=s, cls=m["cls"], sub=m["sub"], drive_g=m["drive_g"], **{f"s_{k}": round(v, 4) for k, v in sc.items()},
                   **{f"t_{k}": failure_type(D[k][s]) for k in ARMS}, spread=round(hi - lo, 4), info=int(hi - lo >= INFO),
                   winners="".join(win), a_best=int("A" in win), n_win=len(win),
                   **{f"act_{k}": round(activity(k, rr[k]), 3) for k in ARMS if k != "A"},
                   n_dec_win=n_post[w0], k_div=kd, n_dec_div=n_div, A_stored=ref.get(s))
        rows.append(row)
    stats.write_table(rows, out / "d3_scores")
    S = {"n": len(rows), "missing": {arm: sorted(set(meta) - set(D[arm])) for arm in ARMS}, "rpc": {k: v for k, v in BAD.items()}}
    # cost
    rt = {arm: RT[arm] for arm in ARMS}
    stack_h = sum(sum(v) for v in rt.values()) / 3600
    S["cost"] = dict(stack_h=stack_h, card_h=stack_h / CARD_STACKS, chunks=sum(len(v) for v in rt.values()),
                     card_h_per_100_scenes=stack_h / CARD_STACKS / max(len(rows), 1) * 100)
    # determinism: A against the stored d402 roll of the same scene (same serve, frame dumps on, other chunk lists)
    pr = [(r["s_A"], r["A_stored"]) for r in rows if r["A_stored"] is not None]
    S["A_vs_d402"] = dict(n=len(pr), identical=sum(abs(x - y) < 1e-6 for x, y in pr), ge02=sum(abs(x - y) >= INFO for x, y in pr),
                          mean_abs=float(np.mean([abs(x - y) for x, y in pr])) if pr else None)
    # per class
    G = {r["scene"]: r["drive_g"] for r in rows}
    T = {}
    for c in CLS + ("all",):
        R = [r for r in rows if c in ("all", r["cls"])]
        if not R:
            continue
        sc = {k: np.array([r[f"s_{k}"] for r in R]) for k in ARMS}
        inf = [r for r in R if r["info"]]
        wins = Counter()
        for r in inf:
            for k in r["winners"]:
                wins[k] += 1 / r["n_win"]
        sole = Counter(r["winners"] for r in inf if r["n_win"] == 1)
        e = dict(n=len(R), info=len(inf), info_share=len(inf) / len(R), spread_mean=float(np.mean([r["spread"] for r in R])),
                 a_best=sum(r["a_best"] for r in R), a_best_info=sum(r["a_best"] for r in inf),
                 a_sole_best_info=sum(r["winners"] == "A" for r in inf),
                 wins={k: round(wins[k], 2) for k in ARMS}, sole_wins={k: sole.get(k, 0) for k in ARMS},
                 mean={k: float(sc[k].mean()) for k in ARMS}, zeros={k: int((sc[k] == 0).sum()) for k in ARMS},
                 oracle=float(np.max(np.stack(list(sc.values())), 0).mean()))
        g = [G[r["scene"]] for r in R]
        e["vs_A"] = {k: stats.paired(sc[k], sc["A"], groups=g) for k in ARMS if k != "A"}
        e["oracle_vs_A"] = stats.paired(np.max(np.stack(list(sc.values())), 0), sc["A"], groups=g)
        # sign table: in how many scenes each arm beats / loses to A by >= 0.2
        e["vs_A_flip"] = {k: dict(up=int((sc[k] - sc["A"] >= INFO).sum()), down=int((sc["A"] - sc[k] >= INFO).sum())) for k in ARMS if k != "A"}
        e["types"] = {k: dict(Counter(r[f"t_{k}"] for r in R).most_common()) for k in ARMS}
        e["rows"] = dict(dec_win=sum(r["n_dec_win"] for r in inf), dec_div=sum(r["n_dec_div"] for r in inf),
                         dec_win_nonA=sum(r["n_dec_win"] for r in inf if not r["a_best"]),
                         dec_div_nonA=sum(r["n_dec_div"] for r in inf if not r["a_best"]))
        e["activity"] = {k: float(np.nanmean([r[f"act_{k}"] for r in R])) for k in ARMS if k != "A"}
        T[c] = e
    # class rule: the arm with the best class mean, chosen leave-one-scene-out (no scene picks its own arm)
    rule = []
    for c in CLS:
        R = [r for r in rows if r["cls"] == c]
        M = np.array([[r[f"s_{k}"] for k in ARMS] for r in R])
        for i, r in enumerate(R):
            mu = (M.sum(0) - M[i]) / max(len(R) - 1, 1)
            rule.append((r, M[i, int(np.argmax(mu))], ARMS[int(np.argmax(mu))]))
    S["class_rule"] = {c: dict(Counter(k for r, _, k in rule if r["cls"] == c)) for c in CLS}
    a_ = np.array([v for _, v, _ in rule])
    b_ = np.array([r["s_A"] for r, _, _ in rule])
    S["class_rule_vs_A"] = stats.paired(a_, b_, groups=[r["drive_g"] for r, _, _ in rule])
    S["class_rule_vs_A_by_cls"] = {c: stats.paired(a_[[r["cls"] == c for r, _, _ in rule]], b_[[r["cls"] == c for r, _, _ in rule]])
                                   for c in CLS}
    S["classes"] = T
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    md(S, out / "tables.md")
    print((out / "tables.md").read_text())


def md(S, path):
    f = stats_fmt
    T = S["classes"]
    L = [f"# D3 tables ({S['n']} scenes x 4 arms scored; generated by d3.py report)", "",
         "| class | n | informative (best - worst >= 0.2) | mean spread | A best (tie incl.) | A best among informative | A sole best among informative | oracle mean | A mean |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for c, e in T.items():
        L.append(f"| {c} | {e['n']} | {e['info']} ({e['info_share']:.0%}) | {e['spread_mean']:.3f} | {e['a_best']} | {e['a_best_info']} | {e['a_sole_best_info']} | "
                 f"{e['oracle']:.3f} | {e['mean']['A']:.3f} |")
    L += ["", "Wins among informative scenes (ties split; sole wins in brackets):", "",
          "| class | " + " | ".join(f"{k} {NAME[k]}" for k in ARMS) + " |", "|---|" + "--:|" * len(ARMS)]
    for c, e in T.items():
        L.append(f"| {c} | " + " | ".join(f"{e['wins'][k]:.1f} ({e['sole_wins'][k]})" for k in ARMS) + " |")
    L += ["", "Arm mean and paired difference to A (bootstrap over drive groups); scenes up / down by >= 0.2 against A; zeros:", "",
          "| class | " + " | ".join(f"{k} - A" for k in ARMS[1:]) + " | oracle - A | " + " | ".join(f"{k} up/down" for k in ARMS[1:]) + " | zeros A/F/S/L |",
          "|---|" + "--:|" * (2 * len(ARMS))]
    for c, e in T.items():
        L.append(f"| {c} | " + " | ".join(f(e["vs_A"][k]) for k in ARMS[1:]) + f" | {f(e['oracle_vs_A'])} | "
                 + " | ".join(f"{e['vs_A_flip'][k]['up']}/{e['vs_A_flip'][k]['down']}" for k in ARMS[1:]) + " | "
                 + "/".join(str(e["zeros"][k]) for k in ARMS) + " |")
    L += ["", "Class rule (best class-mean arm, leave-one-scene-out) against A: " + f(S["class_rule_vs_A"]) + "; picks "
          + ", ".join(f"{c} {v}" for c, v in S["class_rule"].items()), "",
          "| class | rule - A |", "|---|--:|"] + [f"| {c} | {f(v)} |" for c, v in S["class_rule_vs_A_by_cls"].items()]
    L += ["", "Decision rows of the informative scenes (10 Hz inference decisions after the 1.6 s handoff; div = from the first decision where the "
          "winner's ego leaves the worst rollout's by > 5 cm):", "",
          "| class | winner decisions | from divergence | winner not A: decisions | winner not A: from divergence | knob activity F / S / L |",
          "|---|--:|--:|--:|--:|--:|"]
    for c, e in T.items():
        r = e["rows"]
        L.append(f"| {c} | {r['dec_win']} | {r['dec_div']} | {r['dec_win_nonA']} | {r['dec_div_nonA']} | "
                 + " / ".join(f"{e['activity'][k]:.2f}" for k in ARMS[1:]) + " |")
    c = S["cost"]
    d = S["A_vs_d402"]
    L += ["", f"Cost: {c['chunks']} chunks, {c['stack_h']:.2f} stack-h = {c['card_h']:.2f} card-h at {CARD_STACKS} stacks / card; "
          f"{c['card_h_per_100_scenes']:.2f} card-h per 100 scenes x 4 arms.",
          f"A against the stored d402 roll (same serve + frame dumps, other chunk lists): {d['n']} scenes, identical {d['identical']}, "
          f"|diff| >= 0.2 {d['ge02']}, mean |diff| {d['mean_abs']}.",
          f"RPC errors: { {k: len(v) for k, v in S['rpc'].items()} }; missing: { {k: len(v) for k, v in S['missing'].items()} }."]
    path.write_text("\n".join(L) + "\n")


def stats_fmt(r):
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--tsv", required=True)
    p.add_argument("--sel", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("lists")
    p.add_argument("--scenes", required=True)
    p.add_argument("--root", required=True)
    p = sp.add_parser("check")
    p.add_argument("--root", required=True)
    p.add_argument("--stage", required=True)
    p = sp.add_parser("report")
    p.add_argument("--root", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": select, "lists": lists, "check": check, "report": report}[a.cmd](a)
