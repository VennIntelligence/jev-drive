#!/usr/bin/env python3
"""ROUND9 NTV1 step 4 (plans/2026-10-24-ntv1-step4.md = the registration; lane scripts/dg4.sh): DAgger iteration 2. D1 (NT3D1-F-s0) rolled on the 719
curated train scenes, rows with the same teacher b3 -> pool on1_PAIDC2 (part pd2); aggregated pool on1_JTDC2 = on1_JTDC + pd2; D3 = D1's argv on it.
Box, repo root (.venv unless noted). Reuses dgt.py (D1's argv, score / flip / highway blocks), dgc.py (same_u), pcl.py (helpers), round6 dag.py (mfix).
  argv ARM TAG [--steps N] [--on-start T] [--save-at S|none] [--seed K] [--mix]   trainer argv, one token a line
        D1  = dgt.py argv D1 (on1_JTDC / NTV1D)        D3 = D1 + --on on1_JTDC2 --on-wfile NTV1D2        D1Z = D1 on on1_JTDC2 / NTV1D2Z (weight 0)
  diff               D1's argv == its stored argv; D3's tokens differ from it only in --tag / --on / --on-wfile (exit 1 otherwise)
  mfix               (op-train) dag.py mfix on1_JTDC2 against on1_JTDC, unicode keys by value (step 2a amendment 2); part counts = on1_JTDC's + pd2
  weights            NTV1D2Z (NTV1D + pd2 at 0) and NTV1D2 (S part = NTV1D's, 8 / 40 uniform over pd + pd2 drawable rows) -> $O/w, results w_*.json
  wcheck --out J     both draw files against NTV1D, on1_PAIDC / on1_PAIDC2 drawable rows, shares
  t12                T1 (lane centre) and T2 (in canvas) passed
  build              step 4a summary: pool, rows vs on1_PAIDC, train-scene roll scores vs P1L's roll, gates, disk, cost -> results build.{json,md}
  bs --out DIR --tags T,..         behavior_stats on these reads (dgt.py bs)
  verdict --dir O --out P          D3 vs D1-s0, the D1 seed mean, P1L, NTV0-s0, R8K-s2, INTP75 (new + old), classes, flips, highway, lines -> O/CAND
  vverdict --dir O --out P         candidate tail: D3-s0 / s1 on val160 vs the stored R8K-s2 (and D1-s0, P1L) reads
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dgc as G  # noqa: E402
import dgt as T_  # noqa: E402
import pcl as C  # noqa: E402

DD, REPO = C.DD, C.REPO
O = DD / "runs/round9/ntv1_step4"
O2, O3, OV = DD / "runs/round9/ntv1_step2", DD / "runs/round9/ntv1_step3", DD / "runs/round9/val160"
RES = REPO / "experiments/round9/results/ntv1_step4"
CR, F_ = DD / "runs/op_parity/cache", DD / "runs/pai_fix/facts"
D1, D1S1, D3, D3S1 = "NT3D1-F-s0", "NT3D1-F-s1", "NT4D3-F-s0", "NT4D3-F-s1"
P1L, NT, R8, IP = T_.P1L, T_.NT, T_.R8, T_.IP
POOL1, POOL2, NP1, NP2 = "on1_JTDC", "on1_JTDC2", "on1_PAIDC", "on1_PAIDC2"
W1, WREL = "runs/round9/ntv1_step2/w/NTV1D.npz", "runs/round9/ntv1_step4/w"
K_ON, K_DAG = 32, 8
LINE = 0.015                                                   # registered (plan section 4)
CAND = 0.875                                                   # candidate line (plan section 6)
CLS = ("stop_nolead", "turn_sharp", "launch", "other")


def build(arm, tag, steps=None, on_start=None, save_at=None, seed=None, mix=False):
    a = T_.build("D1", tag, steps, on_start, save_at, seed, mix)
    if arm in ("D3", "D1Z"):
        C.put(a, "--on", POOL2)
        C.put(a, "--on-wfile", f"{WREL}/{'NTV1D2' if arm == 'D3' else 'NTV1D2Z'}.npz")
    return a


def cmd_argv(a):
    print("\n".join(build(a.arm, a.tag, a.steps, a.on_start, a.save_at, a.seed, a.mix)))


def cmd_diff(a):
    b = C.stored_argv(D1)
    d1, x = build("D1", D1), build("D3", D3)
    d = [(b[i - 1], b[i], x[i]) for i in range(len(b)) if b[i] != x[i]] if len(x) == len(b) else None
    want = {"--tag": D3, "--on": POOL2, "--on-wfile": f"{WREL}/NTV1D2.npz"}
    ok_d1 = d1 == b
    ok_d3 = d is not None and {q[0]: q[2] for q in d} == want and len(d) == len(want)
    rec = dict(d1_equals_stored=ok_d1, d3_diff=[list(q) for q in d] if d is not None else "length differs", want=want, d3_argv=" ".join(x),
               pass_=bool(ok_d1 and ok_d3))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["pass_"] else 1)


def cmd_mfix(a):
    sys.path.insert(0, str(REPO / "experiments/round6/scripts"))
    import dag as DG
    orig = DG.same
    DG.same = lambda x, y: G.same_u(np.asarray(x), np.asarray(y), orig)
    DG.cmd_mfix(argparse.Namespace(root=str(O), pool=POOL2, ref=POOL1))   # exits 2 on a mismatch
    rec = json.loads((O / f"checks/mfix_{POOL2}.json").read_text())
    p1 = np.load(CR / POOL1 / "tab.npz")["pool"].astype(str)
    want = {k: int(v) for k, v in zip(*np.unique(p1, return_counts=True))} | {"pd2": len(np.load(CR / NP2 / "tab.npz")["names"])}
    rec["parts_want"], rec["parts_ok"] = want, rec["parts"] == want
    C.dump(O / f"checks/mfix_{POOL2}.json", rec)
    print(json.dumps(rec))
    sys.exit(0 if rec["parts_ok"] and not rec["mismatch"] else 2)


def _rows():
    W = np.load(DD / W1)
    B = np.load(DD / C.WFILE)
    tb = np.load(CR / POOL2 / "tab.npz")
    X1, X2 = np.load(CR / NP1 / "dagger.npz"), np.load(CR / NP2 / "dagger.npz")
    n0, n1 = len(B["names"]), len(W["names"])
    return W, B, tb, X1, X2, n0, n1


def gshare(w, grp):
    return {str(g): float(w[grp == g].sum() / w.sum()) for g in sorted(set(grp.tolist())) if w[grp == g].sum() > 0}


def cmd_weights(a):
    W, B, tb, X1, X2, n0, n1 = _rows()
    names = tb["names"].astype(str)
    x2n = X2["names"].astype(str)
    assert np.array_equal(names[:n1], W["names"].astype(str)), f"{POOL2}: prefix != NTV1D's rows (= {POOL1})"
    assert len(names) == n1 + len(x2n) and all(p.endswith(q[2:]) for p, q in zip(names[n1:], x2n)), f"tail != {NP2}'s rows"
    w1 = W["w"]
    grp = np.concatenate([W["grp"].astype(str), np.full(len(x2n), "pd2")])
    wz = np.concatenate([w1, np.zeros(len(x2n), w1.dtype)])
    d1, d2 = n0 + np.flatnonzero(X1["draw"]), n1 + np.flatnonzero(X2["draw"])
    w = np.zeros(len(names), w1.dtype)
    w[:n0] = w1[:n0]                                             # the S part exactly NTV1D's (JT00S-rb0 normalised x 32 / 40)
    nd = len(d1) + len(d2)
    w[d1] = w[d2] = K_DAG / (K_ON + K_DAG) / nd                  # 8 / 40 uniform over both iterations' drawable rows (split by count)
    assert abs(w.sum() - 1) < 1e-9 and abs(wz.sum() - 1) < 1e-9, (w.sum(), wz.sum())
    (O / "w").mkdir(parents=True, exist_ok=True)
    cls2 = X2["cls"].astype(str)[X2["draw"].astype(bool)]
    for nm, ww in (("NTV1D2Z", wz), ("NTV1D2", w)):
        np.savez(O / "w" / f"{nm}.npz", names=names, w=ww, grp=grp)
        rec = dict(base=W1, pool=POOL2, rows=len(names), prefix_rows=n1, s_rows=int((ww[:n0] > 0).sum()), pd_drawable=int(len(d1)),
                   pd2_rows=int(len(x2n)), pd2_drawable=int(len(d2)), pd2_by_cls={c: int((cls2 == c).sum()) for c in CLS},
                   group_share=gshare(ww, grp), out=f"{WREL}/{nm}.npz")
        C.dump(O / "results" / f"w_{nm}.json", rec)
        print(json.dumps(rec, indent=1))


def cmd_wcheck(a):
    W, B, tb, X1, X2, n0, n1 = _rows()
    Z, N = np.load(O / "w/NTV1D2Z.npz"), np.load(O / "w/NTV1D2.npz")
    names, part, mode = tb["names"].astype(str), tb["pool"].astype(str), tb["mode"].astype(str)
    w, wz, w1 = N["w"], Z["w"], W["w"]
    dw = np.flatnonzero(w[n0:] > 0) + n0
    want_d = np.concatenate([n0 + np.flatnonzero(X1["draw"]), n1 + np.flatnonzero(X2["draw"])])
    n1d, n2d = int(X1["draw"].sum()), int(X2["draw"].sum())
    fut = tb["fut"][dw]
    i2 = np.flatnonzero(part == "pd2")
    ag, rd = np.load(CR / POOL2 / "agent.npz")["ok"].astype(bool), np.load(CR / POOL2 / "road.npz")["ok"].astype(bool)
    sh = lambda m: float(w[m].sum() / w.sum())  # noqa: E731
    ck = dict(names_equal=bool(np.array_equal(N["names"].astype(str), names) and np.array_equal(Z["names"].astype(str), names)),
              prefix_names_equal_ntv1d=bool(np.array_equal(names[:n1], W["names"].astype(str))),
              z_prefix_bitwise=bool(wz.dtype == w1.dtype and np.array_equal(wz[:n1], w1)), z_tail_zero=bool(not wz[n1:].any()),
              z_grp_prefix=bool(np.array_equal(Z["grp"].astype(str)[:n1], W["grp"].astype(str))),
              s_bitwise_ntv1d=bool(w.dtype == w1.dtype and np.array_equal(w[:n0], w1[:n0])), s_rows=int((w[:n0] > 0).sum()),
              s_parts=sorted(set(part[:n0][w[:n0] > 0])), s_modes=sorted(set(mode[:n0][w[:n0] > 0])),
              d_ids_equal_drawable=bool(np.array_equal(dw, want_d)), d_parts=sorted(set(part[dw])), d_modes=sorted(set(mode[dw])),
              d_uniform=bool(np.unique(w[dw]).size == 1), d_fut_finite=bool(np.isfinite(fut).all()),
              pd_drawable=n1d, pd2_drawable=n2d, share_s=sh(np.arange(len(w)) < n0), share_pd=sh(part == "pd"), share_pd2=sh(part == "pd2"),
              share_pd_want=0.2 * n1d / (n1d + n2d), share_pd2_want=0.2 * n2d / (n1d + n2d),
              grp_tail=dict(pd=sorted(set(N["grp"].astype(str)[n0:n1])), pd2=sorted(set(N["grp"].astype(str)[n1:]))),
              pd2_rows=int(len(i2)), pd2_modes={k: int(v) for k, v in zip(*np.unique(mode[i2], return_counts=True))},
              pd2_agent_ok=float(ag[i2].mean()), pd2_road_ok=float(rd[i2].mean()))
    ok = (ck["names_equal"] and ck["prefix_names_equal_ntv1d"] and ck["z_prefix_bitwise"] and ck["z_tail_zero"] and ck["z_grp_prefix"]
          and ck["s_bitwise_ntv1d"] and ck["s_rows"] == 13793 and ck["s_parts"] == ["rb0"] and ck["s_modes"] == ["self"] and ck["d_ids_equal_drawable"]
          and ck["d_parts"] == ["pd", "pd2"] and ck["d_modes"] == ["both"] and ck["d_uniform"] and ck["d_fut_finite"]
          and abs(ck["share_s"] - 0.8) < 1e-9 and abs(ck["share_pd"] - ck["share_pd_want"]) < 1e-9 and abs(ck["share_pd2"] - ck["share_pd2_want"]) < 1e-9
          and ck["grp_tail"] == dict(pd=["pd"], pd2=["pd2"]) and ck["pd2_modes"] == {"both": ck["pd2_rows"]})
    rec = dict(files=[f"{WREL}/NTV1D2Z.npz", f"{WREL}/NTV1D2.npz"], ref=W1, checks=ck, pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def cmd_t12(a):
    t1, t2 = json.loads((RES / "t1.json").read_text()), json.loads((RES / "t2.json").read_text())
    ok = bool(t1["passed"]["all"] and t2["in_canvas"]["passed"])
    rec = dict(t1_passed=t1["passed"], t2_in_canvas=t2["in_canvas"], pass_=ok)
    C.dump(O / "results/t12.json", rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def du(p):
    p = Path(p)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / 2 ** 30
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / 2 ** 30


def cost(o=O):
    from jevdrive.cl import pool as P
    J = P.snapshot()[0]["jobs"]
    h_, t0s, t1s = Counter(), [], []
    for f, grp in ((o / "jobs.txt", None), (o / "v160/jobs.txt", "v160")):
        if not f.exists():
            continue
        for ln in f.read_text().splitlines():
            if len(ln.split()) < 2:
                continue
            i, nm = ln.split()[:2]
            j = J.get(i)
            if not j or not j.get("t0") or j["spec"]["vram_gb"] <= 1:
                continue
            t1 = j.get("t1") or time.time()
            k = grp or ("roll" if nm.startswith("roll-") else "read281" if nm.startswith("pai-") else "train" if "-train-" in nm
                        else "build" if nm in ("dg4-cache", "dg4-fwd") else "gates")
            h_[k] += (t1 - j["t0"]) / 3600
            t0s.append(j["t0"]), t1s.append(t1)
    return {k: float(v) for k, v in h_.items()} | dict(gpu_job_h=float(sum(h_.values())), wall_h=(max(t1s) - min(t0s)) / 3600 if t0s else 0.0)


def facts_ok(f):
    import pandas as pd
    T = pd.read_csv(f).set_index("scene")
    return T[~T.rpc.fillna(False).astype(bool)]


def cmd_build(a):
    from jevdrive import stats
    sys.path.insert(0, str(REPO / "experiments/round6/scripts"))
    import dag as DG
    ld = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None  # noqa: E731
    A, R = facts_ok(O / "tax/roll.csv"), facts_ok(O2 / "tax/roll.csv")
    roll = {}
    for nm, X in (("D1 (NT3D1-F-s0)", A), ("P1L (step 2a roll)", R)):
        z = X[X.score == 0]
        roll[nm] = dict(scenes=len(X), mean=float(X.score.mean()), zeros=len(z), zero_fclass=dict(Counter(z.fclass)), full=int((X.score >= 0.98).sum()))
    cm = A.index.intersection(R.index)
    roll["paired D1 - P1L"] = stats.paired(A.score[cm].values, R.score[cm].values) | dict(scenes=len(cm))
    r2, r1 = ld(RES / f"rows_{NP2}.json"), ld(REPO / f"experiments/round9/results/ntv1_step2/rows_{NP1}.json")
    keys = ("rows", "drawable", "scenes", "slow", "no_label", "d0_sign", "T4")
    pools = {NP2: {k: r2[k] for k in keys} | dict(classes={c: r2["classes"][c]["drawable"] for c in CLS}),
             NP1: {k: r1[k] for k in keys} | dict(classes={c: r1["classes"][c]["drawable"] for c in CLS})}
    res = O / "results"
    g = dict(chk1=ld(O / "checks/c000.json"), t12=ld(res / "t12.json"), T4=r2["T4"], r1in=ld(O / "checks/r1in.json"), mfix=ld(O / f"checks/mfix_{POOL2}.json"),
             wcheck=ld(res / "gw_draw_files.json"), diff=ld(res / "diff_D3.json"), t3a_D1=ld(res / "t3a_D1_preT.json"), t3b_w0=ld(res / "t3b_w0.json"),
             g0b_D3=ld(res / "g0b_D3_preT.json"), smoke=ld(res / "smoke_D3.json"), smoke_lr=ld(res / "smoke_lr_D3.json"))
    short = {k: (v.get("pass_", v.get("ok", v.get("bit_identical"))) if isinstance(v, dict) else v) for k, v in g.items()}
    short["mfix"] = bool(g["mfix"] and g["mfix"].get("parts_ok") and not g["mfix"]["mismatch"])
    short["T4"] = dict(p99=r2["T4"]["alat_p99"], passed=r2["T4"]["passed"], fallback_rows=r2["T4"]["fallback_rows_dropped"], go_on=r2["T4"]["go_on"])
    disk = {k: du(v) for k, v in dict(roll=O / "roll", rows=O / "rows", teach=O / "teach", cache=O / "cache", pool=CR / NP2, pool_warp=CR / f"{NP2}@warp",
                                         merged=CR / POOL2, merged_warp=CR / f"{POOL2}@warp", feat=DD / f"runs/route_mem/feat/{NP2}.npz",
                                         feat_merged=DD / f"runs/route_mem/feat/{POOL2}.npz", fwd=O / "fwd", tax=O / "tax", w=O / "w").items()}
    disk["total"] = sum(disk.values())
    rt = DG.runtimes(O / "roll")
    cs = cost() | dict(roll_stack_h=sum(t for t, _ in rt) / 3600, roll_card_h=sum(t for t, _ in rt) / 3600 / DG.CARD_STACKS, roll_chunks=len(rt))
    rec = dict(student=D1, teacher="b3", pools=pools, w=dict(NTV1D2=ld(res / "w_NTV1D2.json"), NTV1D2Z=ld(res / "w_NTV1D2Z.json")), roll=roll,
               gates=short, gates_full=g, disk_gib=disk, cost=cs, t=time.strftime("%F %T"))
    C.dump(res / "build.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "gates_full"}, indent=1, default=float))
    bad = [k for k, v in short.items() if not (v["go_on"] if isinstance(v, dict) else bool(v))]
    print("gates not passed:", bad)
    sys.exit(2 if bad else 0)


def cmd_bs(a):
    T_.cmd_bs(a)


# ---------------------------------------------------------------- verdicts
def cmd_verdict(a):
    from jevdrive import stats
    o = Path(a.dir)
    labs = [D3, D1, D1S1, P1L, NT, R8, IP]
    OLD = {t: T_.facts(f"{t}-281ag") for t in labs}
    NEW = T_.ref_new()
    NEW[D1] = T_.new_series(O3 / "sv2/new/rollouts.csv", f"{D1}-ag")
    NEW[D1S1] = T_.new_series(O3 / "sv2c/new/rollouts.csv", f"{D1S1}-ag")
    NEW[D3] = T_.new_series(o / "sv2/new/rollouts.csv", f"{D3}-ag")
    nm = {D3: "D3", D1: "D1-s0", D1S1: "D1-s1", P1L: "P1L", NT: "NTV0-s0", R8: "R8K-s2", IP: "INTP75-NR (ref.)"}
    pairs = [(D3, y) for y in (D1, D1S1, P1L, NT, R8, IP)] + [(D1, P1L), (D1S1, D1)]
    res = T_.score_block(labs, OLD, NEW, pairs, nm)
    S = res["_S"]
    res["d1_seed_mean"] = {sc: float(((S[sc][D1] + S[sc][D1S1]) / 2).mean()) for sc in ("new", "old")}
    res["d3_vs_d1_mean"] = {sc: stats.paired(S[sc][D3], (S[sc][D1] + S[sc][D1S1]) / 2) for sc in ("new", "old")}
    lines = {}
    for sc in ("new", "old"):
        x = res["pairs"][f"{D3}:{D1}"][sc]
        lines[sc] = dict(d3_d1=x, line=("iteration 2 helps" if x["mean"] >= LINE else "iteration 2 hurts" if x["mean"] <= -LINE else "neutral"),
                         registered=sc == "new")
    res["lines"] = lines
    m = res["reads"][D3]["new_mean"]
    res["candidate"] = dict(line=CAND, d3_new=m, triggered=bool(m >= CAND))
    (o / "CAND").write_text(("D3" if m >= CAND else "none") + "\n")
    hw = T_.highway([(D3, o / "bs", f"{D3}-281ag"), (D1, O3 / "bs", f"{D1}-281ag"), (P1L, DD / "runs/round9/p1l/bs", f"{P1L}-281ag")]
                    + [(t, DD / "runs/pai_av/behavior_stats/out", f"{t}-281ag") for t in (NT, R8)], NEW)
    res["highway"] = hw
    res["gates"] = T_.gates(o)
    rf = o / "results/ratio_D3.json"
    if rf.exists():
        r = json.loads(rf.read_text())
        res["ratio"] = dict(r500=r["ratio"], pass_=r["pass_"], after_T=C.grad_ratio(C.run_dirs(D3)[-1], 10000, 19500)["ratio"],
                            d1_after_T=C.grad_ratio(C.run_dirs(D1)[-1], 10000, 19500)["ratio"])
    res["ade"] = {t: dict(at_10000=C.dev_ade(t, 10000), end=C.dev_ade(t)) for t in (D3, D1, P1L)}
    res["cost"] = cost(o)
    L = ["# NTV1 step 4 verdict: D3 = D1's recipe on the aggregated DAgger pool (PAI 281, ACC + GAP, Harmonizer OFF; P1L's serve)", ""]
    L += T_.md_block(res, nm, "Reads")
    dm = res["d3_vs_d1_mean"]
    L += ["", f"D1 seed mean (s0 + s1) / 2 on these scenes: new {res['d1_seed_mean']['new']:.3f}, old {res['d1_seed_mean']['old']:.3f}; "
          f"D3 - D1 seed mean: new {C.f3(dm['new'])}, old {C.f3(dm['old'])}.", "", "## Registered line (new scorer registered, old reported)", ""]
    for sc, q in lines.items():
        L.append(f"- {sc} scorer: D3 - D1-s0 {C.f3(q['d3_d1'])} (helps >= +{LINE}, hurts <= -{LINE}) -> **{q['line']}**" + ("" if q["registered"] else " (reported)"))
    c = res["candidate"]
    L += ["", f"Candidate rule (D3 new mean >= {CAND}): D3 {m:.3f} -> " + ("**triggered**: seed 1 + val160 reads of both seeds submitted" if c["triggered"]
                                                                          else "not triggered; the lane ends here")]
    L += T_.hw_md(hw, nm)
    L += ["", "## Gates", ""] + [f"- {k}: {v}" for k, v in res["gates"].items()]
    if "ratio" in res:
        r = res["ratio"]
        L.append(f"- S / imit gradient ratio D3: 500 steps after T {r['r500']:.3f} (band {C.RATIO_BAND[0]:.2f}-{C.RATIO_BAND[1]:.2f}, "
                 f"{'pass' if r['pass_'] else 'FAIL'}); whole post-T segment {r['after_T']:.3f} (D1 {r['d1_after_T']:.3f})")
    L += ["", "| navtrain dev ADE | step 10 000 | end |", "|:--|--:|--:|"] + [
        f"| {nm.get(t, t)} | {v['at_10000'] or float('nan'):.4f} | {v['end'] or float('nan'):.4f} |" for t, v in res["ade"].items()]
    cs = res["cost"]
    L += ["", "## Cost", "", "GPU job-hours: " + ", ".join(f"{k} {v:.2f}" for k, v in cs.items() if k not in ("gpu_job_h", "wall_h"))
          + f"; total {cs['gpu_job_h']:.1f}; wall {cs['wall_h']:.1f} h."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", T_.strip(res))
    print("\n".join(L))


def cmd_vverdict(a):
    from jevdrive import stats
    o = Path(a.dir)
    src = {D3: o / "v160/sv2/new/rollouts.csv", D3S1: o / "v160/sv2/new/rollouts.csv", R8: OV / "sv2/new/rollouts.csv",
           D1: OV / "sv2/new/rollouts.csv", P1L: OV / "sv2/new/rollouts.csv"}
    labs = list(src)
    OLD = {t: T_.facts(f"{t}-v160ag") for t in labs}
    NEW = {t: T_.new_series(f, f"{t}-v160") for t, f in src.items()}
    nm = {D3: "D3-s0", D3S1: "D3-s1", R8: "R8K-s2", D1: "D1-s0", P1L: "P1L"}
    pairs = [(t, y) for t in (D3, D3S1) for y in (R8, D1, P1L)] + [(D3S1, D3)]
    res = T_.score_block(labs, OLD, NEW, pairs, nm)
    S = res["_S"]
    res["seed_mean"] = {sc: float(((S[sc][D3] + S[sc][D3S1]) / 2).mean()) for sc in ("new", "old")}
    res["seed_mean_vs"] = {y: {sc: stats.paired((S[sc][D3] + S[sc][D3S1]) / 2, S[sc][y]) for sc in ("new", "old")} for y in (R8, D1, P1L)}
    res["ratio_s1"] = json.loads((o / "results/ratio_D3_s1.json").read_text()) if (o / "results/ratio_D3_s1.json").exists() else None
    res["ade_s1"] = C.dev_ade(D3S1)
    res["cost"] = cost(o)
    L = ["# NTV1 step 4 candidate tail: D3 seeds 0 / 1 on PAI val160 (152 scenes, ACC + GAP, Harmonizer off) vs the stored R8K-s2 read", ""]
    L += T_.md_block(res, nm, "val160")
    L += ["", f"Seed mean (D3-s0 + s1) / 2: new {res['seed_mean']['new']:.3f}, old {res['seed_mean']['old']:.3f}; vs " + "; ".join(
        f"{nm[y]} new {C.f3(v['new'])} / old {C.f3(v['old'])}" for y, v in res["seed_mean_vs"].items()), "",
        f"Seed 1 gates: S / imit ratio 500 steps after T {res['ratio_s1'] and res['ratio_s1']['ratio']}; dev ADE end {res['ade_s1']}.", "",
        f"Lane cost: {res['cost']['gpu_job_h']:.1f} GPU job-hours; wall {res['cost']['wall_h']:.1f} h."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", T_.strip(res))
    print("\n".join(L))


def cmd_s1verdict(a):
    """D3-s1 on PAI 281 (new scorer primary, old reported): s1, the D3 2-seed mean, paired vs R8K-s2 / D1 2-seed mean / P1L / INTP75, s1 - s0, classes, flips."""
    from jevdrive import stats
    o = Path(a.dir)
    labs = [D3S1, D3, D1, D1S1, P1L, NT, R8, IP]
    OLD = {t: T_.facts(f"{t}-281ag") for t in labs}
    NEW = T_.ref_new()
    NEW[D1] = T_.new_series(O3 / "sv2/new/rollouts.csv", f"{D1}-ag")
    NEW[D1S1] = T_.new_series(O3 / "sv2c/new/rollouts.csv", f"{D1S1}-ag")
    NEW[D3] = T_.new_series(o / "sv2/new/rollouts.csv", f"{D3}-ag")
    NEW[D3S1] = T_.new_series(o / "sv2s1/new/rollouts.csv", f"{D3S1}-ag")
    nm = {D3S1: "D3-s1", D3: "D3-s0", D1: "D1-s0", D1S1: "D1-s1", P1L: "P1L", NT: "NTV0-s0", R8: "R8K-s2", IP: "INTP75-NR (ref.)"}
    pairs = [(D3S1, D3), (D3S1, D1), (D3S1, R8), (D3S1, P1L), (D3S1, IP)]
    res = T_.score_block(labs, OLD, NEW, pairs, nm)
    S = res["_S"]
    m2 = {sc: (S[sc][D3] + S[sc][D3S1]) / 2 for sc in ("new", "old")}
    d1m = {sc: (S[sc][D1] + S[sc][D1S1]) / 2 for sc in ("new", "old")}
    res["d3_2seed_mean"] = {sc: float(m2[sc].mean()) for sc in m2}
    ref = {"R8K-s2": {sc: S[sc][R8] for sc in m2}, "D1 2-seed mean": d1m, "P1L": {sc: S[sc][P1L] for sc in m2}, "INTP75-NR": {sc: S[sc][IP] for sc in m2}}
    res["d3_2seed_vs"] = {k: {sc: stats.paired(m2[sc], y[sc]) for sc in m2} for k, y in ref.items()}
    res["cost"] = cost(o)
    L = ["# NTV1 step 4: D3 seed 1 on PAI 281 (ACC + GAP, Harmonizer OFF, D3-s0's serve and chunk lists)", ""]
    L += T_.md_block(res, nm, "Reads")
    L += ["", f"D3 2-seed mean (s0 + s1) / 2: new {m2['new'].mean():.3f}, old {m2['old'].mean():.3f}; D1 2-seed mean: new {d1m['new'].mean():.3f}, old {d1m['old'].mean():.3f}.",
          "", "| D3 2-seed mean minus | new scorer | old scorer |", "|:--|--:|--:|"]
    L += [f"| {k} | {C.f3(v['new'])} | {C.f3(v['old'])} |" for k, v in res["d3_2seed_vs"].items()]
    L += ["", "Zeros: " + ", ".join(f"{nm[t]} {res['reads'][t]['new_zeros']} new / {res['reads'][t]['old']['zeros']} old" for t in (D3S1, D3)), "",
          f"GPU job-hours (lane total): {res['cost']['gpu_job_h']:.1f}."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", T_.strip(res))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("argv")
    p.add_argument("arm", choices=("D1", "D3", "D1Z")), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at"), p.add_argument("--seed", type=int)
    p.add_argument("--mix", action="store_true")
    for k in ("diff", "wcheck"):
        s.add_parser(k).add_argument("--out", required=True)
    for k in ("mfix", "weights", "t12", "build"):
        s.add_parser(k)
    p = s.add_parser("bs")
    p.add_argument("--out", required=True), p.add_argument("--tags", required=True), p.add_argument("--workers", type=int, default=16)
    for k in ("verdict", "vverdict", "s1verdict"):
        p = s.add_parser(k)
        p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(argv=cmd_argv, diff=cmd_diff, mfix=cmd_mfix, weights=cmd_weights, wcheck=cmd_wcheck, t12=cmd_t12, build=cmd_build, bs=cmd_bs,
         verdict=cmd_verdict, vverdict=cmd_vverdict, s1verdict=cmd_s1verdict)[a.sub](a)


if __name__ == "__main__":
    main()
