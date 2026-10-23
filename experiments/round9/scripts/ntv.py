#!/usr/bin/env python3
"""ROUND9 NTV1 step 1 helpers (plans/2026-10-24-ntv1-step1.md = the registration; lane scripts/ntv.sh). Box, repo root, repo venv.
Reuses pcl.py (argv P1L, ident, ratio, post), p1l.py (lrcheck), cpl.py (command gate).
  ntv.py wfile                       NTV1M.npz on on1_JTPH: rb0 S rows (JT00S-rb0 weights) 0.5 + PAI human rows (RB0P weights) 0.5; gate G-W
  ntv.py argv ARM TAG [--steps N] [--on-start T] [--save-at S|none]   trainer argv, one token a line (ARM: N1 N2A N2B, or P1L / RB0F via pcl.py)
  ntv.py diff ARM                    token diff of the arm's argv against P1L's stored argv
  ntv.py smoke --arm A --tag T --on-start T0 --out J   composition gate of a 60-step smoke (batch line, pool / S rows per window, draw shares,
                                     S terms, snapshot)
  ntv.py same --a A --b B --upto N --out J   informational: scalars <= N of two runs (N2b vs N1 before T)
  ntv.py verdict --dir O --out P     registered lines, paired CIs, class tables, lateral zeros, NTV0-full-score new zeros, gates, cost -> P.{md,json}
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl as C  # noqa: E402

DD = C.DD
O_ = DD / "runs/round9/ntv1_step1"
WM = "runs/round9/ntv1_step1/w/NTV1M.npz"
RB0P = "runs/round9/pai_base/w/RB0P.npz"
TAGS = dict(N1="NT1N1-F-s0", N2A="NT1N2A-F-s0", N2B="NT1N2B-F-s0")
P1L, NT, R8 = "PCL1L-F-s0", "NTV0-L18-s0", "R8K-L18-s2"
REFS = [P1L, NT, R8]
K_ON = dict(N1=32, N2A=64, N2B=64)
K_PRE = dict(N1=0, N2A=58, N2B=0)
NORMAL = dict(N1="83", N2A="51", N2B="51")
SHARE_W = {"S": 0.5, "ph": 0.25, "ph05": 0.125, "ph15": 0.125}
SHARE_PRE = {"ph": 0.5, "ph05": 0.25, "ph15": 0.25}
SELF_BAND = (0.42, 0.58)
LINE_N1 = -0.015
LINE_N2 = 0.01
LAT = ("offroad", "zero-other", "side")


def build_argv(arm, tag, steps=None, on_start=None, save_at=None):
    if arm in ("P1L", "RB0F"):
        return C.build_argv(arm, tag, steps, on_start, save_at)
    a = C.build_argv("P1L", tag, steps, on_start, save_at) + ["--cmd-rule", "heading-all", "--on-mix-log"]
    if arm in ("N2A", "N2B"):
        C.put(a, "--on", "on1_JTPH")
        C.put(a, "--on-k", str(K_ON[arm]))
        C.put(a, "--on-wfile", WM)
    if arm == "N2A":
        a += ["--on-pre-k", str(K_PRE[arm]), "--on-pre-wfile", RB0P]
    return a


def cmd_argv(a):
    print("\n".join(build_argv(a.arm, a.tag, a.steps, a.on_start, a.save_at)))


def cmd_diff(a):
    b = C.stored_argv(P1L)
    x = build_argv(a.arm, TAGS[a.arm])
    print(" ".join(x))
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, x, autojunk=False).get_opcodes():
        if op != "equal":
            print(f"{op}: {P1L} {b[i1:i2]} -> {a.arm} {x[j1:j2]}")


def cmd_wfile(a):
    tb = np.load(DD / "runs/op_parity/cache/on1_JTPH/tab.npz")
    tj = np.load(DD / "runs/op_parity/cache/on1_JT/tab.npz")
    w1, w2 = np.load(DD / C.WFILE), np.load(DD / RB0P)
    nm, nj = tb["names"].astype(str), tj["names"].astype(str)
    n = len(nj)
    ck = dict(jt_prefix=bool(np.array_equal(nm[:n], nj)), rb0p_names=bool(np.array_equal(w2["names"].astype(str), nm)),
              rb0_names=bool(np.array_equal(w1["names"].astype(str), nj)))
    assert all(ck.values()), ck
    ws = np.zeros(len(nm))
    ws[:n] = w1["w"]
    wp = w2["w"].astype(np.float64)
    W = 0.5 * ws / ws.sum() + 0.5 * wp / wp.sum()
    part, mode, grp = tb["pool"].astype(str), tb["mode"].astype(str), w2["grp"].astype(str)
    nz_s, nz_p = ws > 0, wp > 0
    sh = C_shares(W, grp)
    ck |= dict(rb0_rows=int(nz_s.sum()), pai_rows=int(nz_p.sum()), overlap=int((nz_s & nz_p).sum()),
               rb0_all_self=bool((part[nz_s] == "rb0").all() and (mode[nz_s] == "self").all()),
               pai_all_ph_both=bool(np.isin(part[nz_p], ["ph", "ph05", "ph15"]).all() and (mode[nz_p] == "both").all()),
               rb0_uniform=bool(np.unique(W[nz_s]).size == 1), pai_prop=float(np.abs(W[nz_p] / wp[nz_p] - W[nz_p][0] / wp[nz_p][0]).max() / (W[nz_p][0] / wp[nz_p][0])),
               grp_jt_equals_jt00s=bool(np.array_equal(grp[:n], w1["grp"].astype(str))), shares=sh,
               mode_share=C_shares(W, mode))
    ok = (ck["rb0_rows"] == 13793 and ck["pai_rows"] == 193850 and not ck["overlap"] and ck["rb0_all_self"] and ck["pai_all_ph_both"]
          and ck["rb0_uniform"] and ck["pai_prop"] < 1e-9 and ck["grp_jt_equals_jt00s"]
          and all(abs(sh.get(k, 0) - v) < 1e-9 for k, v in SHARE_W.items()) and abs(sum(sh.values()) - 1) < 1e-9
          and abs(sum(v for k, v in sh.items() if k in SHARE_W) - 1) < 1e-9)
    out = DD / WM
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=tb["names"], w=W, grp=w2["grp"])
    ck["pass_"] = bool(ok)
    C.dump(O_ / "results/wfile.json", ck)
    print(json.dumps(ck, indent=1))
    sys.exit(0 if ok else 2)


def C_shares(w, g):
    u, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    s = np.bincount(inv, w, len(u)) / np.sum(w)
    return {k: float(v) for k, v in zip(u, s) if v > 0}


def scal(d, tag):
    return {e["step"]: float(e["value"]) for e in C.events(d) if e.get("kind") == "scalar" and e["tag"] == tag}


def cmd_smoke(a):
    d = C.run_dirs(a.tag)[-1]
    log = (d / "log.txt").read_text()
    T, kon, kpre = a.on_start, K_ON[a.arm], K_PRE[a.arm]
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy \(mode (\w+)", log)
    m = re.search(r"draw shares: (\{.*\})", log)
    dr = json.loads(m.group(1)) if m else {}
    vals = np.array([e["value"] for e in C.events(d) if e.get("kind") == "scalar"], float)
    steps = [int(x) for x in re.findall(r"step (\d+): ", log)]
    ck = dict(done=(d / "DONE").exists(), finite=bool(len(vals) and np.isfinite(vals).all()), last_logged_step=max(steps) if steps else None,
              final=(C.RUNS / a.tag / "ckpt-final.pt").exists(), snapshot=(C.RUNS / f"{a.tag}x{T}" / "ckpt-final.pt").exists(),
              batch=list(b.groups()) if b else None, batch_want=[NORMAL[a.arm], "13", str(kon), "mixed"])
    pr, sr = scal(d, "loss/on/batch_pool_rows"), scal(d, "loss/on/batch_self_rows")
    win, prev, post_n, self_post, self_pre_bad = {}, 0, 0, 0.0, 0.0
    for s in sorted(pr):
        npost = sum(1 for k in range(prev, s) if k >= T)
        want = (kon * npost + kpre * (s - prev - npost)) / (s - prev)
        win[s] = dict(pool=pr[s], want=want, self=sr.get(s))
        if npost == 0:
            self_pre_bad += abs(sr.get(s, np.nan))
        else:                                                        # mixed window: every pre-T row is a PAI row (no S row before T)
            self_post += sr.get(s, np.nan) * (s - prev)
            post_n += npost * kon
        prev = s
    ck["windows"] = win
    ck["self_share_after_T"] = self_post / post_n if post_n else None
    ck["self_rows_before_T"] = self_pre_bad
    want_sh = {"S": 1.0} if a.arm == "N1" else SHARE_W
    gs = {k: v for k, v in (dr.get("group_share") or {}).items() if v > 0}
    ck["draw_group_share"], ck["draw_group_share_want"] = gs, want_sh
    ok_sh = set(gs) == set(want_sh) and all(abs(gs[k] - v) < 1e-9 for k, v in want_sh.items())
    if a.arm == "N2A":
        pre = dr.get("pre") or {}
        pg = {k: v for k, v in (pre.get("group_share") or {}).items() if v > 0}
        ck["pre"] = pre
        ok_sh = ok_sh and pre.get("k") == kpre and set(pg) == set(SHARE_PRE) and all(abs(pg[k] - v) < 1e-9 for k, v in SHARE_PRE.items()) \
            and abs((pre.get("mode_share") or {}).get("both", 0) - 1) < 1e-9
    ck["draw_shares_ok"] = bool(ok_sh)
    st = C.s_terms(d, T)
    pre_s = sorted(f"{k[0]}@{k[1]}" for k, v in C.scalars(d, T).items() if C.S_KEY.match(k[0]) and v != 0)
    ck["s_terms_after"], ck["s_terms_before_nonzero"] = st, pre_s
    lo, hi = (1.0, 1.0) if a.arm == "N1" else SELF_BAND
    ok = (ck["done"] and ck["finite"] and ck["last_logged_step"] == 60 and ck["final"] and ck["snapshot"] and ck["batch"] == ck["batch_want"]
          and bool(win) and all(abs(v["pool"] - v["want"]) < 1e-9 for v in win.values()) and self_pre_bad == 0
          and ck["self_share_after_T"] is not None and lo - 1e-9 <= ck["self_share_after_T"] <= hi + 1e-9 and ok_sh
          and st["finite"] and st["nonzero"] and not pre_s)
    rec = dict(arm=a.arm, tag=a.tag, run=str(d), checks=ck, pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1, default=float))
    sys.exit(0 if ok else 2)


def cmd_same(a):
    da, db = C.run_dirs(a.a)[-1], C.run_dirs(a.b)[-1]
    rec = C.compare(da, db, a.upto, True) | dict(a=a.a, b=a.b, note="informational (registered: reported, not a gate)")
    C.dump(a.out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "steps"}, indent=1, default=float))


# ---------------------------------------------------------------- verdict
def cmd_verdict(a):
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(C.REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    arms = list(TAGS.values())
    labs = arms + REFS
    OLD = {t: pd.read_csv(DD / "runs/pai_fix/facts" / f"{t}-281ag.csv").set_index("scene") for t in labs}
    PS = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    NEW = {NT: PS[f"{NT}-ag"], R8: PS[f"{R8}-ag"]}
    x = pd.read_csv(DD / "runs/round9/p1l/sv2/new/rollouts.csv")
    NEW[P1L] = x[x.read == f"{P1L}-ag"].drop_duplicates("scene", keep="last").set_index("scene").score
    x = pd.read_csv(o / "sv2/new/rollouts.csv")
    for t in arms:
        NEW[t] = x[x.read == f"{t}-ag"].drop_duplicates("scene", keep="last").set_index("scene").score
    rpc = {t: int(F.rpc.fillna(False).astype(bool).sum()) for t, F in OLD.items()}
    bad = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    common = sorted(set.intersection(*[set(OLD[t].index[OLD[t].score.notna()]) for t in labs])
                    & set.intersection(*[set(NEW[t].dropna().index) for t in labs]) - bad)
    n = len(common)
    S = dict(new={t: NEW[t].reindex(common).to_numpy(float) for t in labs}, old={t: OLD[t].score.reindex(common).to_numpy(float) for t in labs})
    full = [s for s in common if OLD[NT].score[s] >= 0.98]
    res = dict(scenes=n, rpc=rpc, reads={}, pairs={}, lines={}, ntv0_full=len(full))
    for t in labs:
        F = OLD[t].reindex(common)
        m = PF.metrics(F)
        res["reads"][t] = dict(old=m, new_mean=float(S["new"][t].mean()), new_zeros=int((S["new"][t] == 0).sum()),
                               lateral=int(sum(m[f"z_{c}"] for c in LAT)), route_turn=int(m["z_route-turn"]),
                               ntv0_full_new_zeros_old=int((OLD[t].score.reindex(full) == 0).sum()),
                               ntv0_full_new_zeros_new=int((NEW[t].reindex(full) == 0).sum()),
                               dev_ade=C.dev_ade(t), dev_ade_10000=C.dev_ade(t, 10000) if t in arms + [P1L] else None)
    pairs = [(t, y) for t in arms for y in REFS] + [(TAGS["N2A"], TAGS["N1"]), (TAGS["N2B"], TAGS["N1"]), (TAGS["N2A"], TAGS["N2B"])]
    cls = PF.ZCLS + PF.PCLS + ["zeros", "lateral"]
    for x_, y in pairs:
        k = f"{x_}:{y}"
        res["pairs"][k] = {sc: stats.paired(S[sc][x_], S[sc][y]) for sc in ("new", "old")}
        cl = {}
        for c in cls:
            def ind(q):
                F = OLD[q].reindex(common)
                if c == "zeros":
                    return (S["old"][q] == 0)
                if c == "lateral":
                    return ((F.score == 0) & F.fclass.isin(LAT)).to_numpy()
                return (F.fclass == c).to_numpy()
            u, v = ind(x_).astype(float), ind(y).astype(float)
            q_ = stats.paired(u, v)
            cl[c] = dict(a=int(u.sum()), b=int(v.sum()), diff=n * q_["mean"], lo=n * q_["lo"], hi=n * q_["hi"])
        res["pairs"][k]["classes"] = cl
    n1 = TAGS["N1"]
    for sc in ("new", "old"):
        d1 = res["pairs"][f"{n1}:{P1L}"][sc]
        L = dict(n1_minus_p1l=d1, n1=("the heading command costs nothing in the S recipe (adopt)" if d1["mean"] >= LINE_N1
                                     else "below the line: the heading command costs in the S recipe (report)"))
        rt1 = res["reads"][n1]["route_turn"]
        for arm in ("N2A", "N2B"):
            t = TAGS[arm]
            dm = res["pairs"][f"{t}:{n1}"][sc]["mean"]
            rt = res["reads"][t]["route_turn"]
            L[arm] = dict(diff=dm, route_turn=rt, route_turn_n1=rt1,
                          line=("PAI human rows help in this placement" if rt < rt1 and dm >= LINE_N2 else
                                "PAI human rows hurt in this placement" if dm <= -LINE_N2 else "neutral"))
        L["registered"] = sc == "new"
        res["lines"][sc] = L
    res["gates"] = {}
    for f in sorted((o / "results").glob("*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        res["gates"][f.stem] = j.get("pass_", j.get("passed", j.get("bit_identical")))
    res["ratio"] = {}
    for arm, t in TAGS.items():
        f = o / f"results/ratio_{arm}.json"
        if f.exists():
            r = json.loads(f.read_text())
            full_ = C.grad_ratio(C.run_dirs(t)[-1], 10000, 19500)
            res["ratio"][arm] = dict(r500=r["ratio"], pass_=r["pass_"], after_T=full_["ratio"])
    res["cmdagree"] = {arm: json.loads((o / f"results/cmdagree_{arm}.json").read_text()) for arm in TAGS if (o / f"results/cmdagree_{arm}.json").exists()}
    res["cost"] = C.cost(o)
    f3 = C.f3
    nm = {TAGS["N1"]: "N1", TAGS["N2A"]: "N2a", TAGS["N2B"]: "N2b", P1L: "P1L", NT: "NTV0-s0", R8: "R8K-s2"}
    Lm = ["# NTV1 step 1 verdict (PAI 281, ACC + GAP, Harmonizer OFF; arms served with JEV_CMD=heading)", "",
          f"{n} scenes scored by all six reads (both scorers) with no driver RPC error in any of them; RPC scenes per read {rpc}.", "",
          "## Reads", "", "| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | "
          "rear-hit p | slow p | launched | new zeros on NTV0 full-score (old / new) | dev ADE 10k / end |", "|:--|" + "--:|" * 15]
    for t, r in res["reads"].items():
        m = r["old"]
        de = (f"{r['dev_ade_10000']:.4f} / " if r["dev_ade_10000"] else "- / ") + (f"{r['dev_ade']:.4f}" if r["dev_ade"] else "-")
        Lm.append(f"| {nm[t]} ({t}) | {r['new_mean']:.3f} | {m['mean']:.3f} | {r['new_zeros']} / {m['zeros']} | {r['lateral']} | {m['z_offroad']} | "
                  f"{m['z_zero-other']} | {m['z_side']} | {m['z_route-turn']} | {m['z_overshoot']} | {m['z_AF-front']} | {m['p_rear-hit']} | {m['p_slow']} | "
                  f"{m['launch']} | {r['ntv0_full_new_zeros_old']} / {r['ntv0_full_new_zeros_new']} | {de} |")
    Lm += ["", f"NTV0-s0 full-score scenes (old score >= 0.98) among the common set: {len(full)}.", "",
           "## Paired differences (95 % bootstrap CI, scenes as units)", "", "| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn |",
           "|:--|--:|--:|--:|--:|--:|"]
    for k, p in res["pairs"].items():
        c = p["classes"]
        x_, y = k.split(":")
        Lm.append(f"| {nm[x_]} - {nm[y]} | {f3(p['new'])} | {f3(p['old'])} | " + " | ".join(
            f"{c[q]['diff']:+.0f} [{c[q]['lo']:+.0f}, {c[q]['hi']:+.0f}]" for q in ("zeros", "lateral", "route-turn")) + " |")
    cz = PF.ZCLS + PF.PCLS
    Lm += ["", "Zero / partial classes (count difference, 95 % CI; old-scorer classes):", "", "| pair | " + " | ".join(cz) + " |", "|:--|" + "--:|" * len(cz)]
    for k, p in res["pairs"].items():
        x_, y = k.split(":")
        Lm.append(f"| {nm[x_]} - {nm[y]} | " + " | ".join(f"{p['classes'][q]['diff']:+.0f} [{p['classes'][q]['lo']:+.0f}, {p['classes'][q]['hi']:+.0f}]" for q in cz) + " |")
    Lm += ["", "## Registered lines (new scorer registered, old reported)", ""]
    for sc, q in res["lines"].items():
        Lm.append(f"- {sc} scorer: N1 - P1L {f3(q['n1_minus_p1l'])} (line >= {LINE_N1}) -> **{q['n1']}**" + ("" if q["registered"] else " (reported)"))
        for arm in ("N2A", "N2B"):
            r = q[arm]
            Lm.append(f"  - {arm} - N1 {r['diff']:+.3f}, route-turn zeros {r['route_turn']} vs N1 {r['route_turn_n1']} -> **{r['line']}**")
    Lm += ["", "## Gates", ""] + [f"- {k}: {v}" for k, v in res["gates"].items()]
    for arm, r in res["ratio"].items():
        Lm.append(f"- S / imit gradient ratio {arm}: 500 steps after T {r['r500']:.3f} (band {C.RATIO_BAND[0]:.2f}-{C.RATIO_BAND[1]:.2f}, "
                  f"{'pass' if r['pass_'] else 'FAIL'}); whole post-T segment {r['after_T']:.3f}")
    for arm, c in res["cmdagree"].items():
        Lm.append(f"- served command vs rule {arm}: " + "; ".join(f"{v['inferences']} inferences, agree {v['agree']:.5f}" for v in c["reads"].values())
                  + f" -> {'pass' if c['passed'] else 'FAIL'}")
    c = res["cost"]
    Lm += ["", "## Cost", "", f"GPU jobs {c['gpu_job_h']:.1f} job-hours (training {c['train_h']:.2f}, reads {c['read_h']:.2f}, gates / smokes {c['gate_h']:.2f}); "
           f"wall {c['wall_h']:.1f} h from the first start to the last end."]
    Path(f"{a.out}.md").write_text("\n".join(Lm) + "\n")
    C.dump(f"{a.out}.json", res)
    print("\n".join(Lm))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    s.add_parser("wfile")
    p = s.add_parser("argv")
    p.add_argument("arm", choices=("N1", "N2A", "N2B", "P1L", "RB0F")), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at")
    p = s.add_parser("diff")
    p.add_argument("arm", choices=("N1", "N2A", "N2B"))
    p = s.add_parser("smoke")
    p.add_argument("--arm", required=True, choices=list(TAGS)), p.add_argument("--tag", required=True)
    p.add_argument("--on-start", type=int, required=True), p.add_argument("--out", required=True)
    p = s.add_parser("same")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--upto", type=int, required=True)
    p.add_argument("--out", required=True)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(wfile=cmd_wfile, argv=cmd_argv, diff=cmd_diff, smoke=cmd_smoke, same=cmd_same, verdict=cmd_verdict)[a.sub](a)


if __name__ == "__main__":
    main()
