#!/usr/bin/env python3
"""ROUND8 (plans/2026-10-22-r8k.md): checks and reports of the R8K arm and the Harmonizer render-noise check. Lane script: r8k.sh.

  r8k.py smoke --tag X --ctrl Y --wjson W --out J     300-step smoke (X = R8K, Y = R6DAG's argv): finite, batch 51 / 13 / 72, group shares = the
                                                      R7JNS draw file's, S-hinge / imitation gradient ratio vs Y in x[0.7, 1.3], dev ADE line,
                                                      twin swap share >= 0.99 with the drawn twin rows near the trainer's expectation, no --cmd-rule
  r8k.py noise --stored D --new D --facts-stored F --facts-new F --facts-off F --out P
                                                      ON re-read vs the stored ON read: per-scene score diffs, zero flips, served-plan identity,
                                                      the implied per-read noise of a 279-scene mean, and the stored ON - OFF on the same scenes
  r8k.py seeds --read K=F .. --out P                 (amendment 1, descriptive) 3-seed recipe means R8K vs R6DAG: per scene the seed mean of OFF, ON,
                                                      (OFF + ON) / 2 on the scenes every read scored without RPC error, paired scene bootstrap CIs
  r8k.py verdict --json J --md M --pcs P --read K=F ..  the registered adoption rule on r2.py r281's json -> appended to M, verdict into J
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "experiments/round7/scripts")]
import joint as J  # noqa: E402  (events, drive_recs, SIX, CUTIN; sets the round3 / round6 paths)

EPS = 1e-4


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float))


# ---------------------------------------------------------------- smoke
def cmd_smoke(a):
    dj, SJ = J.events(a.tag)
    dc, SC = J.events(a.ctrl)
    ratio = lambda S: sum(S["loss/grad/pool_hinge"].values()) / sum(S["loss/grad/imit"].values())  # noqa: E731
    rj, rc = ratio(SJ), ratio(SC)
    log = (dj / "log.txt").read_text()
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", log)
    done = json.loads((dj / "DONE").read_text()) if (dj / "DONE").exists() else {}
    gs = (done.get("draw_shares") or {}).get("group_share", {})
    ex = json.loads(Path(a.wjson).read_text())
    ade = SJ["dev/ade"]
    last = max(ade)
    tw0 = re.search(r"--twin \(round7 HARM-MIX\): (\{.*\})", log)
    tw1 = done.get("twin") or (json.loads(m.group(1)) if (m := re.search(r"--twin realised: (\{.*\})", log)) else {})
    steps = max(max(v) for v in SJ.values() if v)
    exp = json.loads(tw0.group(1))["twin_rows_per_batch"] * steps if tw0 else float("nan")
    ck = dict(done=bool(done), finite=all(np.isfinite(list(v.values())).all() for v in SJ.values()),
              batch=(b.groups() if b else None, ("51", "13", str(ex["k_on"]))),
              shares=all(abs(gs.get(k, -1) - v) < 1e-6 for k, v in ex["group_share"].items()),
              grad_ratio=dict(arm=rj, control=rc, rel=rj / rc, band=[0.7, 1.3]),
              dev_ade=dict(step=last, arm=SJ["dev/ade"][last], control=SC["dev/ade"].get(last), line=0.5362 + 0.03),
              twin=dict(setup=json.loads(tw0.group(1)) if tw0 else None, realised=tw1, steps=steps, expected_drawn=exp),
              cmd_rule_logged="--cmd-rule (round7 joint arm)" in log)
    drawn = tw1.get("drawn_with_twin", 0)
    ok = dict(done=ck["done"], finite=ck["finite"], batch=bool(b) and b.groups() == ck["batch"][1], shares=ck["shares"],
              grad=0.7 <= rj / rc <= 1.3, dev=ade[last] <= 0.5362 + 0.03, twin_swap=tw1.get("swap_share", 0) >= 0.99,
              twin_drawn=bool(exp == exp and 0.75 * exp <= drawn <= 1.25 * exp), no_cmd=not ck["cmd_rule_logged"])
    res = dict(tag=a.tag, ctrl=a.ctrl, checks=ck, ok=ok, pass_=bool(all(ok.values())))
    dump(a.out, res)
    print(json.dumps(res, indent=1, default=float))
    sys.exit(0 if res["pass_"] else 2)


# ---------------------------------------------------------------- noise
def boot(d, n=10000, seed=0):
    g = np.random.default_rng(seed)
    m = d[g.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def cmd_noise(a):
    import pandas as pd
    rd = lambda f: pd.read_csv(f).set_index("scene")  # noqa: E731
    A, B, O = rd(a.facts_stored), rd(a.facts_new), rd(a.facts_off)
    sc = [s for s in B.index if s in A.index and B.rpc.get(s) is not True and A.rpc.get(s) is not True and np.isfinite(B.score[s]) and np.isfinite(A.score[s])]
    sa, sb, so = A.score.reindex(sc).to_numpy(), B.score.reindex(sc).to_numpy(), O.score.reindex(sc).to_numpy()
    d = sb - sa
    za, zb = sa <= EPS, sb <= EPS
    # served plans: first inference where the two reads differ, per scene
    ra, rb = J.drive_recs(a.stored), J.drive_recs(a.new)
    per = {}
    for (s, k) in sorted(set(ra) & set(rb), key=lambda x: (x[0], x[1])):
        p = per.setdefault(s, dict(n=0, same=0, first_diff=None, max_abs=0.0))
        x = float(np.abs(np.asarray(ra[(s, k)]["poses"], float) - np.asarray(rb[(s, k)]["poses"], float)).max())
        p["n"] += 1
        p["same"] += x == 0.0
        p["max_abs"] = max(p["max_abs"], x)
        if x > 0 and p["first_diff"] is None:
            p["first_diff"] = k
    ident = [s for s, p in per.items() if p["first_diff"] is None]
    fd = [p["first_diff"] for p in per.values() if p["first_diff"] is not None]
    n_all = 279
    sd_pair = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    res = dict(
        scenes=len(sc), scenes_with_plans=len(per), plan_identical_scenes=len(ident),
        first_plan_diff_k=dict(median=float(np.median(fd)) if fd else None, min=int(min(fd)) if fd else None),
        inferences_compared=int(sum(p["n"] for p in per.values())), inferences_identical=int(sum(p["same"] for p in per.values())),
        score_identical=int((np.abs(d) <= 1e-9).sum()), score_diff_scenes=int((np.abs(d) > 1e-9).sum()),
        mean_stored=float(sa.mean()), mean_new=float(sb.mean()), new_minus_stored=boot(d), mean_abs_diff=float(np.abs(d).mean()),
        sd_scene_diff=sd_pair, zero_flips=dict(total=int((za != zb).sum()), stored_zero_new_not=int((za & ~zb).sum()), new_zero_stored_not=int((zb & ~za).sum())),
        big_moves=int((np.abs(d) > 0.5).sum()),
        # one read's noise on a mean over n scenes (independent scenes): sd(new - stored) / sqrt(2) / sqrt(n)
        per_read_mean_sd_279=sd_pair / np.sqrt(2) / np.sqrt(n_all), per_read_mean_sd_these=sd_pair / np.sqrt(2) / np.sqrt(len(sc)),
        on_minus_off_stored=boot(sa - so), on_minus_off_new=boot(sb - so),
        var_share_noise=float((sd_pair ** 2 / 2) / np.var(sa - so, ddof=1)) if len(sc) > 1 and np.var(sa - so) > 0 else None,
        classes=dict(stored=A.fclass.reindex(sc).value_counts().to_dict(), new=B.fclass.reindex(sc).value_counts().to_dict()),
        moved=[dict(scene=s, stored=float(x), new=float(y), off=float(z), c_stored=str(A.fclass[s]), c_new=str(B.fclass[s]),
                    first_plan_diff=per.get(s, {}).get("first_diff")) for s, x, y, z in zip(sc, sa, sb, so) if abs(y - x) > 1e-9])
    dump(f"{a.out}.json", res)
    f3 = lambda t: "%+.3f [%+.3f, %+.3f]" % t  # noqa: E731
    L = ["# Harmonizer render-noise check: R6DAG-L18-s1 ON re-read of c000 + c001 vs the stored ON read", "",
         f"Scenes compared {res['scenes']}; served plans compared on {res['scenes_with_plans']} scenes, {res['inferences_compared']} inferences "
         f"({res['inferences_identical']} bit-identical); scenes whose every served plan is identical: {res['plan_identical_scenes']}; "
         f"first differing inference k: median {res['first_plan_diff_k']['median']}, min {res['first_plan_diff_k']['min']}.", "",
         "| quantity | value |", "|:--|--:|",
         f"| mean stored ON / re-read ON | {res['mean_stored']:.3f} / {res['mean_new']:.3f} |",
         f"| re-read - stored (paired, 95 % CI) | {f3(res['new_minus_stored'])} |",
         f"| scenes with a different score / identical | {res['score_diff_scenes']} / {res['score_identical']} |",
         f"| mean abs scene diff | {res['mean_abs_diff']:.4f} |", f"| sd of scene diff | {sd_pair:.4f} |",
         f"| zero flips (stored 0 -> not / new 0) | {res['zero_flips']['total']} ({res['zero_flips']['stored_zero_new_not']} / {res['zero_flips']['new_zero_stored_not']}) |",
         f"| scenes moving > 0.5 | {res['big_moves']} |",
         f"| implied per-read noise sd of a 279-scene mean | {res['per_read_mean_sd_279']:.4f} |",
         f"| stored ON - OFF on these scenes | {f3(res['on_minus_off_stored'])} |", f"| re-read ON - OFF on these scenes | {f3(res['on_minus_off_new'])} |",
         f"| share of the stored ON - OFF scene variance that is render noise | {res['var_share_noise'] if res['var_share_noise'] is None else round(res['var_share_noise'], 3)} |",
         "", "## Scenes whose score moved", "", "| scene | OFF | stored ON | re-read ON | class stored / re-read | first plan diff k |", "|:--|--:|--:|--:|:--|--:|"]
    L += [f"| {m['scene'][7:15]} | {m['off']:.2f} | {m['stored']:.2f} | {m['new']:.2f} | {m['c_stored']} / {m['c_new']} | {m['first_plan_diff']} |"
          for m in sorted(res["moved"], key=lambda m: -abs(m["new"] - m["stored"]))]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- seeds (amendment 1)
def cmd_seeds(a):
    import pandas as pd
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in a.read)}
    com = sorted(set.intersection(*[set(X.index[X.score.notna() & ~X.rpc.fillna(False).astype(bool)]) for X in F.values()]))
    S = {k: X.score.reindex(com).to_numpy(float) for k, X in F.items()}
    seeds = sorted({k.split("-")[1] for k in F})
    rec = dict(scenes=len(com), seeds=seeds, per_read={k: float(v.mean()) for k, v in S.items()}, recipes={}, diff={}, seed_diff={})
    M = {}
    for r in ("R6DAG", "R8K"):
        off = np.mean([S[f"{r}-{sd}-ag"] for sd in seeds], 0)
        on = np.mean([S[f"{r}-{sd}-agH"] for sd in seeds], 0)
        M[r] = dict(OFF=off, ON=on, AVG=(off + on) / 2)
        rec["recipes"][r] = {q: float(v.mean()) for q, v in M[r].items()} | dict(
            seed_sd_OFF=float(np.std([S[f"{r}-{sd}-ag"].mean() for sd in seeds], ddof=1)), seed_sd_ON=float(np.std([S[f"{r}-{sd}-agH"].mean() for sd in seeds], ddof=1)))
    for q in ("OFF", "ON", "AVG"):
        rec["diff"][q] = boot(M["R8K"][q] - M["R6DAG"][q])
    for sd in seeds:                                                      # same-seed pairs (descriptive)
        rec["seed_diff"][sd] = {m: boot(S[f"R8K-{sd}-{m}"] - S[f"R6DAG-{sd}-{m}"]) for m in ("ag", "agH")}
    dump(f"{a.out}.json", rec)
    f3 = lambda t: "%+.3f [%+.3f, %+.3f]" % t  # noqa: E731
    L = [f"# 3-seed recipe means, R8K vs R6DAG (PAI 281, ACC + GAP; {len(com)} common scenes; amendment 1, descriptive, no adoption rule)", "",
         "| | " + " | ".join(f"{r}-{sd}" for r in ("R6DAG", "R8K") for sd in seeds) + " |", "|:--|" + "--:|" * (2 * len(seeds))]
    for m, lab in (("ag", "OFF"), ("agH", "ON")):
        L.append(f"| {lab} | " + " | ".join(f"{S[f'{r}-{sd}-{m}'].mean():.3f}" for r in ("R6DAG", "R8K") for sd in seeds) + " |")
    L.append("| (OFF + ON) / 2 | " + " | ".join(f"{(S[f'{r}-{sd}-ag'].mean() + S[f'{r}-{sd}-agH'].mean()) / 2:.3f}" for r in ("R6DAG", "R8K") for sd in seeds) + " |")
    L += ["", "| recipe mean (3 seeds) | R6DAG | R8K | R8K - R6DAG (paired scene bootstrap of the seed means) |", "|:--|--:|--:|--:|"]
    for q, lab in (("OFF", "OFF"), ("ON", "ON"), ("AVG", "(OFF + ON) / 2")):
        L.append(f"| {lab} | {rec['recipes']['R6DAG'][q]:.3f} | {rec['recipes']['R8K'][q]:.3f} | {f3(rec['diff'][q])} |")
    L.append(f"| seed sd of the read mean, OFF / ON | {rec['recipes']['R6DAG']['seed_sd_OFF']:.3f} / {rec['recipes']['R6DAG']['seed_sd_ON']:.3f} | "
             f"{rec['recipes']['R8K']['seed_sd_OFF']:.3f} / {rec['recipes']['R8K']['seed_sd_ON']:.3f} | |")
    L += ["", "| same-seed pair | OFF R8K - R6DAG | ON R8K - R6DAG |", "|:--|--:|--:|"]
    L += [f"| {sd} | {f3(rec['seed_diff'][sd]['ag'])} | {f3(rec['seed_diff'][sd]['agH'])} |" for sd in seeds]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- verdict
def cmd_verdict(a):
    import pandas as pd
    Jx = json.loads(Path(a.json).read_text())
    R, P = Jx["reads"], Jx["pairs"]
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in a.read)}
    lat = lambda k: R[k]["z_offroad"] + R[k]["z_zero-other"]  # noqa: E731
    nl = lambda k: int(str(R[k].get("launch", "0/37")).split("/")[0])  # noqa: E731
    Q = json.loads(Path(a.pcs).read_text()) if a.pcs and Path(a.pcs).exists() else {}
    q = lambda k: (Q.get(k) or {}).get("diff_point")  # noqa: E731
    qs = lambda k: "-" if q(k) is None else "%+.0f [%+.0f, %+.0f]" % (q(k), *Q[k]["diff_ci"])  # noqa: E731
    f3 = lambda d: "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"]) if d else "-"  # noqa: E731
    pm = lambda x, y: P.get(f"{x}:{y}", {}).get("mean")  # noqa: E731
    ver, ok_all = {}, True
    side = ["", "## Side by side (common scenes; ruler = local PCS - NTV0-s0+AG, same bootstrap draws)", "",
            "| | " + " | ".join(f"{m}-{s}" for s in ("s0", "s1") for m in ("R8K", "R7J", "R6DAG")) + " |", "|:--|" + "--:|" * 6]
    cols = [(m, s) for s in ("s0", "s1") for m in ("R8K", "R7J", "R6DAG")]
    rq = {"R8K": "R8K", "R7J": "R7J", "R6DAG": "DAG"}
    side += [
        "| OFF mean | " + " | ".join(f"{R[f'{m}-{s}-ag']['mean']:.3f}" for m, s in cols) + " |",
        "| ON mean | " + " | ".join(f"{R[f'{m}-{s}-agH']['mean']:.3f}" for m, s in cols) + " |",
        "| (OFF + ON) / 2 | " + " | ".join(f"{(R[f'{m}-{s}-ag']['mean'] + R[f'{m}-{s}-agH']['mean']) / 2:.3f}" for m, s in cols) + " |",
        "| ruler - NTV0 | " + " | ".join(qs(f"{rq[m]}-{s}+AG") for m, s in cols) + " |",
        "| offroad + zero-other OFF / ON | " + " | ".join(f"{lat(f'{m}-{s}-ag')} / {lat(f'{m}-{s}-agH')}" for m, s in cols) + " |",
        "| rear-hit / slow partials OFF | " + " | ".join(f"{R[f'{m}-{s}-ag']['p_rear-hit']} / {R[f'{m}-{s}-ag']['p_slow']}" for m, s in cols) + " |",
        "| rear-hit / slow partials ON | " + " | ".join(f"{R[f'{m}-{s}-agH']['p_rear-hit']} / {R[f'{m}-{s}-agH']['p_slow']}" for m, s in cols) + " |",
        "| launched OFF / ON | " + " | ".join(f"{nl(f'{m}-{s}-ag')} / {nl(f'{m}-{s}-agH')}" for m, s in cols) + " |",
        "| zeros OFF / ON | " + " | ".join(f"{R[f'{m}-{s}-ag']['zeros']} / {R[f'{m}-{s}-agH']['zeros']}" for m, s in cols) + " |"]
    rows = []
    for sd in ("s0", "s1"):
        k, c, kh, ch = f"R8K-{sd}-ag", f"R6DAG-{sd}-ag", f"R8K-{sd}-agH", f"R6DAG-{sd}-agH"
        if k not in R or kh not in R:
            continue
        doff, don = pm(k, c), pm(kh, ch)
        avg = (doff["mean"] + don["mean"]) / 2 if doff and don else float("nan")
        pk, p6 = q(f"R8K-{sd}+AG"), q(f"DAG-{sd}+AG")
        ok = dict(mean_vs_r6=bool(doff) and doff["mean"] >= 0, lat=lat(k) <= lat(c) + 2, launch=nl(k) >= 34,
                  ruler=pk is not None and p6 is not None and pk >= p6, offon_avg=avg == avg and avg >= 0)
        ver[sd] = ok
        ok_all &= all(ok.values())
        rows.append(f"| {sd} | {R[k]['mean']:.3f} | {f3(doff)} | {lat(k)} vs {lat(c)} + 2 | {R[k].get('launch', '')} | "
                    f"{qs(f'R8K-{sd}+AG')} vs {'-' if p6 is None else '%+.0f' % p6} | {R[kh]['mean']:.3f}, {f3(don)} | {avg:+.4f} | "
                    f"{f3(pm(k, f'R7J-{sd}-ag'))} / {f3(pm(kh, f'R7J-{sd}-agH'))} | {' '.join(x for x, v in ok.items() if not v) or 'all pass'} |")
    L = ["", "## Registered adoption rule (plan section 5)", "",
         "Both seeds: OFF 281 - same-seed R6DAG (paired point) >= 0; offroad + zero-other <= same-seed R6DAG + 2; launches >= 34 / 37; ruler - NTV0 "
         ">= same-seed R6DAG's in the same report; (OFF + ON) / 2 - same-seed R6DAG's (OFF + ON) / 2 (paired points) >= 0.", "",
         "| seed | OFF mean | OFF - R6DAG same seed | offroad + zero-other | launched | ruler - NTV0 | ON mean, - R6DAG same seed ON | (OFF + ON) / 2 - R6DAG | "
         "- R7J same seed OFF / ON | failing |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|", *rows, "",
         f"**Adopt: {'yes (both seeds pass)' if ok_all and len(ver) == 2 else 'no'}.**", *side]
    com = sorted(set.intersection(*[set(X.index[X.score.notna()]) for X in F.values()]))
    solved = (F["NTV0-ag"].score.reindex(com) >= 1 - EPS).to_numpy()
    L += ["", f"### New zeros / partials on the {int(solved.sum())} scenes NTV0 (OFF) scores 1.0", "", "| read | new zeros | new partials | classes |", "|:--|--:|--:|:--|"]
    for k, X in F.items():
        if k == "NTV0-ag":
            continue
        y, c = X.score.reindex(com)[solved], X.fclass.reindex(com)[solved]
        L.append(f"| {k} | {int((y <= EPS).sum())} | {int(((y > EPS) & (y < 1 - EPS)).sum())} | "
                 + ", ".join(f"{q_} {n}" for q_, n in c[y < 1 - EPS].value_counts().items()) + " |")
    pick = lambda X, h: next(((float(X.score[i]), str(X.fclass[i])) for i in X.index if h in i), (float("nan"), "?"))  # noqa: E731
    for title, hs in (("The six ruler-gap partial scenes (d436)", J.SIX), ("The two stuck curve cut-ins", J.CUTIN)):
        L += ["", f"### {title}: score / class", "", "| scene | " + " | ".join(F) + " |", "|:--|" + "--:|" * len(F)]
        L += [f"| {h} | " + " | ".join(f"{pick(X, h)[0]:.2f} {pick(X, h)[1]}" for X in F.values()) + " |" for h in hs]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    Jx["verdict"] = dict(adopt=bool(ok_all and len(ver) == 2), seeds=ver)
    Path(a.json).write_text(json.dumps(Jx, indent=1, default=float))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("smoke")
    for k in ("--tag", "--ctrl", "--wjson", "--out"):
        p.add_argument(k, required=True)
    p = sp.add_parser("noise")
    for k in ("--stored", "--new", "--facts-stored", "--facts-new", "--facts-off", "--out"):
        p.add_argument(k, required=True)
    p = sp.add_parser("seeds")
    p.add_argument("--read", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("verdict")
    p.add_argument("--json", required=True)
    p.add_argument("--md", required=True)
    p.add_argument("--pcs", default="")
    p.add_argument("--read", nargs="+", required=True)
    a = ap.parse_args()
    dict(smoke=cmd_smoke, noise=cmd_noise, seeds=cmd_seeds, verdict=cmd_verdict)[a.cmd](a)


if __name__ == "__main__":
    main()
