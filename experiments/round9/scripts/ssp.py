#!/usr/bin/env python3
"""ROUND9 S-SPLIT helpers (plans/2026-10-24-s-split.md = the registration; lane scripts/ssp.sh). Box, repo root, repo venv. Reuses pcl.py.
  ssp.py argv ARM TAG [--steps N] [--diag-every K]   the arm's trainer argv, one token a line (ARM: P0 NH ND)
                                                     P0 = PCL0-L18-s0's stored argv (pcl.py argv P0); NH = P0 + --ho-w 0 (every S-row hinge x 0,
                                                     distill_ho kept); ND = P0 + --ho-distill 0 (distill_ho x 0 on the S rows, hinges kept)
  ssp.py diff ARM                                    token diff against PCL0-L18-s0's stored argv
  ssp.py offcmp --a A --b B --upto N --out J         flag-off identity: every scalar of A and B at steps <= N bit-identical, same key sets
  ssp.py gcheck --arm ARM --tag T --out J            smoke check of the switch from the --diag steps: NH grad/pool == 0 exactly (hinges carry no
                                                     gradient), ND grad/pool > 0; both: distill_ho and the S hinge values logged, finite, nonzero
  ssp.py verdict --dir O --out P                     registered lines, paired CIs, class table vs P0 / NTV0NS, training-log terms, cost -> P.{json,md}
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl as C  # noqa: E402

DD = C.DD
PP = DD / "runs/round9/pool_pilot"
TAGS = dict(NH="SSNH-L18-s0", ND="SSND-L18-s0")
P0, NS, N0, RB = "PCL0-L18-s0", "NTV0NS-L18-s0", "NTV0-L18-s0", "RB0-F-s0"
LABS = list(TAGS.values()) + [P0, NS, N0, RB]
# registered (plan section 4): arm >= P0 - 0.02 -> the removed part is not needed; arm <= NTV0NS + 0.03 -> the removed part carries the gain
TOL_P0, TOL_NS = 0.02, 0.03
REMOVED = dict(NH="the S-row hinges (agent / road / lateral)", ND="distill_ho on the S rows")
HINGE_KEYS = ["loss/agent_ho", "loss/road_ho", "loss/on/lat_onc_ho", "loss/on/lat_soft_ho", "loss/on/lat_side_ho"]


def build(arm, tag, steps=None, diag_every=None):
    a = C.build_argv("P0", tag, steps)
    if arm == "NH":
        C.put(a, "--ho-w", "0")
    elif arm == "ND":
        assert "--ho-distill" not in a
        a += ["--ho-distill", "0"]
    if diag_every is not None:
        assert "--diag-every" not in a
        a += ["--diag-every", str(diag_every)]
    return a


def cmd_argv(a):
    print("\n".join(build(a.arm, a.tag, a.steps, a.diag_every)))


def cmd_diff(a):
    b, x = C.stored_argv(P0), build(a.arm, TAGS.get(a.arm, P0))
    print(" ".join(x))
    d = [(op, b[i1:i2], x[j1:j2]) for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, x, autojunk=False).get_opcodes() if op != "equal"]
    for q in d:
        print(f"{q[0]}: {P0} {q[1]} -> {a.arm} {q[2]}")
    want = dict(P0=[], NH=[["PCL0-L18-s0"], ["SSNH-L18-s0"], ["3"], ["0"]], ND=[["PCL0-L18-s0"], ["SSND-L18-s0"], [], ["--ho-distill", "0"]])[a.arm]
    got = [t for q in d for t in q[1:]]
    ok = got == want
    print("TOKEN DIFF AS REGISTERED" if ok else f"UNEXPECTED DIFF (want {want})")
    sys.exit(0 if ok else 1)


def cmd_offcmp(a):
    rec = C.compare(C.run_dirs(a.a)[-1], C.run_dirs(a.b)[-1], a.upto, False) | dict(a=a.a, b=a.b)
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["bit_identical"] else 2)


def series(d, tag):
    return {e["step"]: float(e["value"]) for e in C.events(d) if e.get("kind") == "scalar" and e["tag"] == tag}


def cmd_gcheck(a):
    d = C.run_dirs(a.tag)[-1]
    gp, dh = series(d, "loss/grad/pool"), series(d, "loss/distill_ho")
    hv = {k: series(d, k) for k in HINGE_KEYS}
    ok = bool(gp) and bool(dh) and all(np.isfinite(v) and v > 0 for v in dh.values())
    ok = ok and all(s and all(np.isfinite(v) for v in s.values()) and any(v != 0 for v in s.values()) for s in hv.values())
    if a.arm == "NH":
        sub = {k: series(d, k) for k in ("loss/grad/pool_hinge", "loss/grad/lat")}
        ok = ok and all(v == 0 for v in gp.values()) and all(v == 0 for s in sub.values() for v in s.values())
    else:
        sub = {}
        ok = ok and all(v > 0 for v in gp.values())
    rec = dict(arm=a.arm, tag=a.tag, run=str(d), grad_pool=gp, grad_sub=sub, distill_ho=dh, hinge_values={k: list(s.values()) for k, s in hv.items()},
               rule="NH: grad/pool, grad/pool_hinge, grad/lat == 0 at every diag step; ND: grad/pool > 0; both: distill_ho > 0, hinge values finite, nonzero",
               pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def log_terms(tag, last=1000):
    """Training-log means over the last `last` steps: distill_ho, the S hinge values, grad/pool, grad/imit."""
    ds = C.run_dirs(tag)
    if not ds:
        return None
    d = ds[-1]
    out = {}
    for k in ["loss/distill_ho", "loss/distill"] + HINGE_KEYS + ["loss/grad/pool", "loss/grad/imit"]:
        s = series(d, k)
        if s:
            hi = max(s)
            v = [x for st, x in s.items() if st > hi - last]
            out[k] = float(np.mean(v))
    return out


def line(arm_m, p0_m, ns_m):
    if arm_m >= p0_m - TOL_P0:
        return "not needed"
    if arm_m <= ns_m + TOL_NS:
        return "carries the gain"
    return "partial"


def combined(nh, nd):
    k = (nh, nd)
    if k == ("carries the gain", "not needed"):
        return "the hinges carry the S gain; distill_ho is not needed (body schema = consequence hinge holds)"
    if k == ("not needed", "carries the gain"):
        return "distill_ho carries the S gain; the hinges are not needed (the consequence-hinge claim does not hold for this gain)"
    if k == ("not needed", "not needed"):
        return "either part alone keeps the gain (redundant)"
    if k == ("carries the gain", "carries the gain"):
        return "both parts are needed together"
    return f"mixed (NH {nh}, ND {nd})"


def cmd_verdict(a):
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(C.REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    OLD = {t: pd.read_csv(DD / "runs/pai_fix/facts" / f"{t}-281ag.csv").set_index("scene") for t in LABS}
    NEW = C.new_scores(PP)                                    # NTV0-s0, RB0-F, RB0L, NTV0NS, and the pool pilot's reads (P0)
    x = pd.read_csv(o / "sv2/new/rollouts.csv")
    for t in TAGS.values():
        NEW[t] = x[x.read == f"{t}-ag"].drop_duplicates("scene", keep="last").set_index("scene").score
    rpc = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    common = sorted(set.intersection(*[set(OLD[t].index[OLD[t].score.notna()]) for t in LABS])
                    & set.intersection(*[set(NEW[t].dropna().index) for t in LABS]) - rpc)
    n = len(common)
    Sc = dict(new={t: NEW[t].reindex(common).to_numpy(float) for t in LABS}, old={t: OLD[t].score.reindex(common).to_numpy(float) for t in LABS})
    res = dict(scenes=n, reads={}, pairs={}, classes={}, lines={}, logs={})
    for t in LABS:
        res["reads"][t] = dict(old=PF.metrics(OLD[t].reindex(common)), new_mean=float(Sc["new"][t].mean()), new_zeros=int((Sc["new"][t] == 0).sum()),
                               dev_ade=C.dev_ade(t))
    pairs = [(t, y) for t in TAGS.values() for y in (P0, NS, N0, RB)] + [(TAGS["NH"], TAGS["ND"])]
    cls = PF.ZCLS + PF.PCLS
    for x_, y in pairs:
        k = f"{x_}:{y}"
        res["pairs"][k] = {sc: stats.paired(Sc[sc][x_], Sc[sc][y]) for sc in ("new", "old")}
        cl = {}
        for c in cls + ["zeros"]:
            if c == "zeros":
                u, v = (Sc["old"][x_] == 0).astype(float), (Sc["old"][y] == 0).astype(float)
            else:
                u, v = (OLD[x_].reindex(common).fclass == c).to_numpy(float), (OLD[y].reindex(common).fclass == c).to_numpy(float)
            q = stats.paired(u, v)
            cl[c] = dict(a=int(u.sum()), b=int(v.sum()), diff=n * q["mean"], lo=n * q["lo"], hi=n * q["hi"])
        res["classes"][k] = cl
    for sc in ("new", "old"):
        p0, ns = float(Sc[sc][P0].mean()), float(Sc[sc][NS].mean())
        q = dict(P0=p0, NTV0NS=ns, line_not_needed=p0 - TOL_P0, line_carries=ns + TOL_NS, arms={})
        for arm, t in TAGS.items():
            m = float(Sc[sc][t].mean())
            q["arms"][arm] = dict(tag=t, mean=m, share_of_S_gain=(m - ns) / (p0 - ns), line=line(m, p0, ns))
        q["combined"] = combined(q["arms"]["NH"]["line"], q["arms"]["ND"]["line"])
        res["lines"][sc] = q
    for t in (P0, *TAGS.values()):
        res["logs"][t] = log_terms(t)
    res["cost"] = C.cost(o)
    f3 = C.f3
    L = ["# S-SPLIT verdict (PAI 281, ACC + GAP, Harmonizer OFF)", "",
         f"{n} scenes scored by all six reads (both scorers) with no driver RPC error in any of them.", "",
         "## Registered lines (new scorer primary; arm >= P0 - 0.02: the removed part is not needed; arm <= NTV0NS + 0.03: it carries the gain)", ""]
    for sc, q in res["lines"].items():
        L.append(f"- {sc} scorer: P0 {q['P0']:.3f}, NTV0NS {q['NTV0NS']:.3f} -> lines {q['line_not_needed']:.3f} / {q['line_carries']:.3f}")
        for arm, r in q["arms"].items():
            L.append(f"  - {arm} {r['tag']} (removed: {REMOVED[arm]}): {r['mean']:.3f}, {r['share_of_S_gain']:.0%} of the NTV0NS -> P0 gap kept "
                     f"-> **{r['line']}**")
        L.append(f"  - together: **{q['combined']}**")
    L += ["", "## Reads", "", "| read | new mean | old mean | zeros new / old | offroad | drift (zero-other) + side | route-turn | overshoot | AF-front | "
          "rear-hit p | slow p | dev ADE |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in res["reads"].items():
        m = r["old"]
        L.append(f"| {t} | {r['new_mean']:.3f} | {m['mean']:.3f} | {r['new_zeros']} / {m['zeros']} | {m['z_offroad']} | {m['z_zero-other']} + {m['z_side']} | "
                 f"{m['z_route-turn']} | {m['z_overshoot']} | {m['z_AF-front']} | {m['p_rear-hit']} | {m['p_slow']} | "
                 + (f"{r['dev_ade']:.4f} |" if r["dev_ade"] is not None else "- |"))
    L += ["", "## Paired differences (95 % bootstrap CI, scenes as units)", "", "| pair | new scorer | old scorer |", "|:--|--:|--:|"]
    for k, p in res["pairs"].items():
        L.append(f"| {k.replace(':', ' - ')} | {f3(p['new'])} | {f3(p['old'])} |")
    L += ["", "## Zero / partial classes vs P0 and NTV0NS (old-scorer classes; count difference, 95 % CI)", "",
          "| arm - ref | zeros | " + " | ".join(cls) + " |", "|:--|--:|" + "--:|" * len(cls)]
    for x_ in TAGS.values():
        for y in (P0, NS):
            c = res["classes"][f"{x_}:{y}"]
            L.append(f"| {x_} - {y} | " + " | ".join(f"{c[q]['diff']:+.0f} [{c[q]['lo']:+.0f}, {c[q]['hi']:+.0f}]" for q in ["zeros"] + cls) + " |")
    L += ["", "## Training logs (means over the last 1 000 steps)", "", "| run | " + " | ".join(k.replace("loss/", "") for k in
          ["loss/distill_ho", "loss/distill"] + HINGE_KEYS + ["loss/grad/pool", "loss/grad/imit"]) + " |", "|:--|" + "--:|" * 9]
    for t, r in res["logs"].items():
        if r:
            L.append(f"| {t} | " + " | ".join(f"{r[k]:.4g}" if k in r else "-" for k in ["loss/distill_ho", "loss/distill"] + HINGE_KEYS
                                               + ["loss/grad/pool", "loss/grad/imit"]) + " |")
    c = res["cost"]
    L += ["", "## Cost", "", f"GPU jobs {c['gpu_job_h']:.1f} job-hours (training {c['train_h']:.2f}, reads {c['read_h']:.2f}, gates / smokes {c['gate_h']:.2f}); "
          f"wall {c['wall_h']:.1f} h."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", res)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("argv")
    p.add_argument("arm", choices=("P0", "NH", "ND")), p.add_argument("tag"), p.add_argument("--steps", type=int), p.add_argument("--diag-every", type=int)
    p = s.add_parser("diff")
    p.add_argument("arm", choices=("P0", "NH", "ND"))
    p = s.add_parser("offcmp")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--upto", type=int, required=True)
    p.add_argument("--out", required=True)
    p = s.add_parser("gcheck")
    p.add_argument("--arm", required=True, choices=("NH", "ND")), p.add_argument("--tag", required=True), p.add_argument("--out", required=True)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(argv=cmd_argv, diff=cmd_diff, offcmp=cmd_offcmp, gcheck=cmd_gcheck, verdict=cmd_verdict)[a.sub](a)


if __name__ == "__main__":
    main()
