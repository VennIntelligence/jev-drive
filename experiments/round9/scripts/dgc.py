#!/usr/bin/env python3
"""ROUND9 NTV1 step 2a (plans/2026-10-24-ntv1-step2.md = registration): P1L's (PCL1L-F-s0) own DAgger pool on1_PAIDC, round-1 teacher b3.
Lane helpers for dgc.sh (box, repo root, .venv unless noted; the roll -> rows -> teacher -> pool -> label chain is round6 dag.sh / dag.py):
  argv VAR TAG [--steps N] [--on-start T] [--save-at S] [--mix]   P1L's argv (pcl.py build_argv P1L) with
        p1l  unchanged                           t3  --on on1_JTDC --on-wfile NTV1D0 (DAgger rows at weight 0)
        a    on1_JTDC + NTV1D --on-k 40 --batch 136 --yr-mass 0.095 --cmd-rule heading-all (8 DAgger rows added: R6DAG's way)
        b    on1_JTDC + NTV1D --on-k 40 (8 DAgger rows replace 8 navtrain rows after T)            --mix: + --on-mix-log
  weights --k-dag K --out NAME     JT00S-rb0 on the merged pool's on1_JT prefix (K = 0: unchanged, the identity file) + K / (32 + K) uniform
                                   over the drawable DAgger rows -> $O/w/NAME.npz, results w_NAME.json
  pair --a TAG --b TAG --upto N --out F    every scalar <= N of two runs bit for bit (T3 b)
  smoke --var a|b --tag TAG --on-start T --out F    composition of a 60-step T = 30 run
  cmdf                             the command-rule fields the DAgger rows carry and rmem.cmd_heading on them -> results cmd_fields.json
  report                           rollout scores (facts), pool size, gates, disk, cost -> results build.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO), str(REPO / "experiments/route_mem/lib")]
import pcl as C  # noqa: E402

DD = C.DD
O = DD / "runs/round9/ntv1_step2"
RES = REPO / "experiments/round9/results/ntv1_step2"
CR, FEAT = DD / "runs/op_parity/cache", DD / "runs/route_mem/feat"
STUDENT, NP, POOL, BASE_W = "PCL1L-F-s0", "on1_PAIDC", "on1_JTDC", "runs/round9/pool_pilot/w/JT00S-rb0.npz"
WREL = "runs/round9/ntv1_step2/w"
K_ON, K_DAG = 32, 8
NORMAL = dict(a="83", b="75")
SELF_BAND = (0.72, 0.88)
CLS = ("stop_nolead", "turn_sharp", "launch", "other")


def build_argv(var, tag, steps=None, on_start=None, save_at=None, mix=False):
    a = C.build_argv("P1L", tag, steps, on_start, save_at)
    if var != "p1l":
        C.put(a, "--on", POOL)
        C.put(a, "--on-wfile", f"{WREL}/{'NTV1D0' if var == 't3' else 'NTV1D'}.npz")
    if var in ("a", "b"):
        C.put(a, "--on-k", str(K_ON + K_DAG))
    if var == "a":
        assert "--batch" not in a
        C.put(a, "--yr-mass", "0.095")
        a += ["--batch", "136", "--cmd-rule", "heading-all"]
    if mix:
        a.append("--on-mix-log")
    return a


def cmd_argv(a):
    print("\n".join(build_argv(a.var, a.tag, a.steps, a.on_start, a.save_at, a.mix)))


def cmd_weights(a):
    B = np.load(DD / BASE_W)
    names = np.load(CR / POOL / "tab.npz")["names"].astype(str)
    n0 = len(B["names"])
    assert np.array_equal(names[:n0], B["names"].astype(str)), f"{POOL}: prefix != JT00S-rb0's rows"
    X = np.load(CR / NP / "dagger.npz")
    xn = X["names"].astype(str)
    assert len(names) == n0 + len(xn) and all(x.endswith(y[2:]) for x, y in zip(names[n0:], xn)), "tail != on1_PAIDC's rows"
    bw = B["w"].astype(np.float64)
    w = np.zeros(len(names))
    dr = np.flatnonzero(X["draw"])
    if a.k_dag:
        k_all = K_ON + a.k_dag
        w[:n0] = bw / bw.sum() * (K_ON / k_all)
        w[n0 + dr] = a.k_dag / k_all / len(dr)
        assert abs(w.sum() - 1) < 1e-9, w.sum()
    else:
        w[:n0] = bw
        assert np.array_equal(w[:n0], B["w"]) and not w[n0:].any()
    grp = np.concatenate([B["grp"].astype(str), np.full(len(names) - n0, "pd")])
    gs = Counter()
    for g_, x in zip(grp, w / w.sum()):
        gs[str(g_)] += x
    cls = X["cls"].astype(str)[dr]
    rec = dict(base=BASE_W, pool=POOL, k_on=K_ON, k_dag=a.k_dag, rows=len(names), prefix_rows=n0, dag_rows=len(xn), dag_drawable=int(len(dr)),
               dag_scenes=int(len({n.split(":")[2] for n in xn[dr]})), group_share={k: float(v) for k, v in sorted(gs.items()) if v > 0},
               by_cls={c: int((cls == c).sum()) for c in CLS}, out=f"{WREL}/{a.out}.npz")
    (O / "w").mkdir(parents=True, exist_ok=True)
    np.savez(O / "w" / f"{a.out}.npz", names=names, w=w, grp=grp)
    C.dump(RES / f"w_{a.out}.json", rec)
    print(json.dumps(rec, indent=1))


def cmd_pair(a):
    da, db = C.run_dirs(a.a)[-1], C.run_dirs(a.b)[-1]
    rec = C.compare(db, da, a.upto, False) | dict(a=a.a, b=a.b, run_a=str(da), run_b=str(db))
    rec["pool_rows_b"] = C.pool_rows(db)
    rec["pool_drawn"] = bool(any(v > 0 for v in rec["pool_rows_b"].values()))
    rec["pass_"] = bool(rec["bit_identical"] and rec["pool_drawn"])
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["pass_"] else 2)


def cmd_smoke(a):
    d = C.run_dirs(a.tag)[-1]
    log = (d / "log.txt").read_text()
    T, kon = a.on_start, K_ON + K_DAG
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy \(mode (\w+)", log)
    m = re.search(r"draw shares: (\{.*\})", log)
    dr = json.loads(m.group(1)) if m else {}
    vals = np.array([e["value"] for e in C.events(d) if e.get("kind") == "scalar"], float)
    steps = [int(x) for x in re.findall(r"step (\d+): ", log)]
    sc = lambda k: {e["step"]: float(e["value"]) for e in C.events(d) if e.get("kind") == "scalar" and e["tag"] == k}  # noqa: E731
    pr, sr = sc("loss/on/batch_pool_rows"), sc("loss/on/batch_self_rows")
    ck = dict(done=(d / "DONE").exists(), finite=bool(len(vals) and np.isfinite(vals).all()), last_logged_step=max(steps) if steps else None,
              batch=list(b.groups()) if b else None, batch_want=[NORMAL[a.var], "13", str(kon), "mixed"])
    win, prev, post_n, self_post, self_pre = {}, 0, 0, 0.0, 0.0
    for s in sorted(pr):
        npost = sum(1 for k in range(prev, s) if k >= T)
        win[s] = dict(pool=pr[s], want=kon * npost / (s - prev), self=sr.get(s))
        if npost == 0:
            self_pre += abs(sr.get(s, np.nan))
        else:
            self_post += sr.get(s, np.nan) * (s - prev)
            post_n += npost * kon
        prev = s
    ck |= dict(windows=win, self_share_after_T=self_post / post_n if post_n else None, self_rows_before_T=self_pre)
    want = json.loads((RES / "w_NTV1D.json").read_text())["group_share"]
    gs = {k: v for k, v in (dr.get("group_share") or {}).items() if v > 0}
    ck |= dict(draw_group_share=gs, draw_group_share_want=want, draw_shares_ok=set(gs) == set(want) and all(abs(gs[k] - v) < 1e-9 for k, v in want.items()))
    st = C.s_terms(d, T)
    pre_s = sorted(f"{k[0]}@{k[1]}" for k, v in C.scalars(d, T).items() if C.S_KEY.match(k[0]) and v != 0)
    ck |= dict(s_terms_after=st, s_terms_before_nonzero=pre_s)
    cr = re.search(r"--cmd-rule \(round7 joint arm\): (\{.*\})", log)
    ck["cmd_rule"] = json.loads(cr.group(1)) if cr else None
    ok = (ck["done"] and ck["finite"] and ck["last_logged_step"] == 60 and ck["batch"] == ck["batch_want"] and bool(win)
          and all(abs(v["pool"] - v["want"]) < 1e-9 for v in win.values()) and self_pre == 0 and ck["self_share_after_T"] is not None
          and SELF_BAND[0] <= ck["self_share_after_T"] <= SELF_BAND[1] and ck["draw_shares_ok"] and st["finite"] and st["nonzero"] and not pre_s)
    if a.var == "a":                                                 # heading-all ran over the DAgger part too
        ck["cmd_rule_pd"] = (ck["cmd_rule"] or {}).get("parts", {}).get("pd")
        ok = ok and bool(ck["cmd_rule_pd"])
    rec = dict(var=a.var, tag=a.tag, run=str(d), checks=ck, pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1, default=float))
    sys.exit(0 if ok else 2)


def cmd_mfix(a):
    """dag.py mfix against on1_JT, tolerant of on1_JT's extra metadata keys (agent cls / log, road log / pose_global, tab acc / cmd / pose / vel:
    on1_JT's older build; rf_pait merge never writes them, on1_JTPH lacks the same keys): every COMMON key of the prefix bit for bit, ref-only
    keys listed; then dag.py's agent-ok fix of the prefix. (Attempt 1: dag.py mfix raised KeyError on agent.cls.)"""
    sys.path.insert(0, str(REPO / "experiments/round6/scripts"))
    import dag as DG
    from jevdrive.tokens import open_front
    P, R = CR / POOL, CR / "on1_JT"
    bad, ref_only = [], []
    for f in ("tab.npz", "agent.npz", "road.npz", "drivable.npz", "route.npz"):
        zp, zr = np.load(P / f), np.load(R / f)
        n = len(np.load(R / "tab.npz", mmap_mode="r")["names"]) if f != "tab.npz" else len(zr["names"])
        ref_only += [f"{f}.{k}" for k in zr.files if k not in zp.files]
        bad += [f"{f}.{k}" for k in zr.files if k in zp.files and not (k == "ok" and f == "agent.npz") and zr[k].ndim and len(zr[k]) == n
                and not DG.same(zp[k][:n], zr[k])]
    n = len(np.load(R / "tab.npz")["names"])
    fp, fr = np.load(FEAT / f"{POOL}.npz"), np.load(FEAT / "on1_JT.npz")
    bad += [f"feat.{k}" for k in fr.files if not DG.same(fp[k][:n], fr[k])]
    zp, zr = np.load(CR / f"{POOL}@warp/teacher.npz"), np.load(CR / "on1_JT@warp/teacher.npz")
    bad += [f"teacher.{k}" for k in zr.files if not DG.same(zp[k][:n] if zr[k].ndim and len(zr[k]) == n else zp[k], zr[k])]
    i = np.sort(np.random.default_rng(0).choice(n, 4000, replace=False))
    if not DG.same(open_front(CR / f"{POOL}@warp/front.npy")[i], open_front(R.parent / "on1_JT@warp/front.npy")[i]):
        bad.append("front sample")
    ag, agr = dict(np.load(P / "agent.npz")), np.load(R / "agent.npz")
    before = int((ag["ok"][:n] != agr["ok"]).sum())
    ag["ok"] = np.concatenate([agr["ok"], ag["ok"][n:]])
    np.savez(P / "agent.npz", **ag)
    part = np.load(P / "tab.npz")["pool"].astype(str)
    rec = dict(pool=POOL, ref="on1_JT", ref_rows=n, rows=len(part), parts={p: int((part == p).sum()) for p in np.unique(part)}, agent_ok_fixed=before,
               ref_only_keys=ref_only, mismatch=bad)
    C.dump(O / "checks" / f"mfix_{POOL}.json", rec)
    print(json.dumps(rec))
    sys.exit(2 if bad else 0)


def lab(c):
    return np.where(c.max(1) > 0.5, c.argmax(1), 3).astype(np.int8)


def cmd_cmdf(a):
    import rmem as RM
    F, tb, X = np.load(FEAT / f"{NP}.npz"), np.load(CR / NP / "tab.npz"), np.load(CR / NP / "dagger.npz")
    names = tb["names"].astype(str)
    assert np.array_equal(F["names"].astype(str), names) and np.array_equal(X["names"].astype(str), names), "feat / tab / dagger rows differ"
    f = F["f"]
    assert f.dtype == np.float16 and f.shape[1] == RM.DIM, (f.dtype, f.shape)
    has = (f.reshape(len(f), RM.N_PT, RM.NF)[..., 0] > 0.5).any(1)
    old = tb["ego"][:, 1:4].astype(np.float32)
    new = np.where(has[:, None], RM.cmd_heading(f)[:, :3], old)
    lo, ln = lab(old), lab(new)
    nm = ("L", "S", "R", "none")                                     # ego cols 1:4 = left / straight / right (rm_train cmd_all: 0 / 2 = L / R)
    share = lambda x, m: {nm[i]: round(float((x[m] == i).mean()), 4) for i in range(4)}  # noqa: E731
    out = {}
    for k, m in (("all", np.ones(len(f), bool)), ("drawable", X["draw"].astype(bool))):
        out[k] = dict(rows=int(m.sum()), memory_valid=round(float(has[m].mean()), 4), feat_ok=round(float(F["ok"][m].mean()), 4),
                      stored=share(lo, m), heading=share(ln, m), changed=round(float((lo[m] != ln[m]).mean()), 4),
                      lr_stored=round(float(np.isin(lo[m], [0, 2]).mean()), 4), lr_heading=round(float(np.isin(ln[m], [0, 2]).mean()), 4))
    fm = np.load(FEAT / f"{POOL}.npz")
    n0 = len(fm["names"]) - len(names)
    tail_same = bool(np.array_equal(fm["f"][n0:], f) and all(x.endswith(y[2:]) for x, y in zip(fm["names"].astype(str)[n0:], names)))
    rec = dict(pool=NP, fields=dict(feat=f"runs/route_mem/feat/{NP}.npz: names, f ({RM.N_PT} slots x {RM.NF}, float16 = the fed route memory, "
                                         "old projected layout), k, ok", stored_command="tab.ego[:, 1:4] (the served shipped-rule one-hot)",
                                    speed="tab.speed"),
               merged_tail_feat_identical=tail_same, **out)
    C.dump(RES / "cmd_fields.json", rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if tail_same else 2)


def du(p):
    p = Path(p)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / 2 ** 30
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / 2 ** 30


def facts(f):
    import pandas as pd
    T = pd.read_csv(f).set_index("scene")
    return T[~T.rpc.fillna(False).astype(bool)]


def cmd_report(a):
    from jevdrive import stats
    sys.path.insert(0, str(REPO / "experiments/round6/scripts"))
    os.environ.setdefault("DAG_O", str(O))
    import dag as DG
    A, R = facts(O / "tax/roll.csv"), facts(O / "tax/ref_r5lc.csv")
    roll = {}
    for nm, X in (("P1L", A), ("R5LC-L18-s0 (round-6 student, stored)", R)):
        z = X[X.score == 0]
        roll[nm] = dict(scenes=len(X), mean=float(X.score.mean()), zeros=len(z), zero_fclass=dict(Counter(z.fclass)), full=int((X.score >= 0.98).sum()))
    cm = A.index.intersection(R.index)
    roll["paired P1L - R5LC"] = stats.paired(A.score[cm].values, R.score[cm].values) | dict(scenes=len(cm))
    rows = json.loads((RES / f"rows_{NP}.json").read_text())
    ld = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None  # noqa: E731
    g = dict(chk1=ld(O / "checks/c000.json"), T1=ld(RES / "t1.json"), T2=ld(RES / "t2.json"), T3a=ld(RES / "t3a.json"), T3b=ld(RES / "t3b.json"),
             r1in=ld(O / "checks/r1in.json"), mfix=ld(O / f"checks/mfix_{POOL}.json"), smoke_a=ld(RES / "smoke_a.json"), smoke_b=ld(RES / "smoke_b.json"))
    gates = dict(chk1=g["chk1"] and g["chk1"]["ok"], T1=g["T1"] and g["T1"]["passed"], T2_in_canvas=g["T2"] and g["T2"]["in_canvas"],
                 T3a=g["T3a"] and {k: g["T3a"][k] for k in ("bit_identical", "scalars_compared", "identical", "steps")},
                 T3b=g["T3b"] and {k: g["T3b"][k] for k in ("pass_", "scalars_compared", "identical", "pool_drawn")},
                 T4=rows["T4"], r1in=g["r1in"], mfix=g["mfix"] and dict(mismatch=g["mfix"]["mismatch"], parts=g["mfix"]["parts"]),
                 smoke_a=g["smoke_a"] and g["smoke_a"]["pass_"], smoke_b=g["smoke_b"] and g["smoke_b"]["pass_"])
    disk = {k: du(v) for k, v in dict(roll=O / "roll", rows=O / "rows", teach=O / "teach", cache=O / "cache", pool=CR / NP, pool_warp=CR / f"{NP}@warp",
                                         merged=CR / POOL, merged_warp=CR / f"{POOL}@warp", feat=FEAT / f"{NP}.npz", feat_merged=FEAT / f"{POOL}.npz",
                                         fwd=O / "fwd", tax=O / "tax").items()}
    rt = DG.runtimes(O / "roll")
    cost = C.cost(O) | dict(roll_stack_h=sum(t for t, _ in rt) / 3600, roll_card_h=sum(t for t, _ in rt) / 3600 / DG.CARD_STACKS, roll_chunks=len(rt))
    rec = dict(student=STUDENT, teacher="b3", pool=dict(name=NP, rows=rows["rows"], drawable=rows["drawable"], scenes=rows["scenes"], slow=rows["slow"],
                                                         no_label=rows["no_label"], classes=rows["classes"], d0_sign=rows["d0_sign"]),
               w=dict(NTV1D=ld(RES / "w_NTV1D.json"), NTV1D0=ld(RES / "w_NTV1D0.json")), roll=roll, gates=gates, cmd_fields=ld(RES / "cmd_fields.json"),
               disk_gib=disk, cost=cost, t=time.strftime("%F %T"))
    C.dump(RES / "build.json", rec)
    print(json.dumps(rec, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("argv")
    p.add_argument("var", choices=("p1l", "t3", "a", "b")), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at"), p.add_argument("--mix", action="store_true")
    p = s.add_parser("weights")
    p.add_argument("--k-dag", type=int, required=True), p.add_argument("--out", required=True)
    p = s.add_parser("pair")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--upto", type=int, required=True), p.add_argument("--out", required=True)
    p = s.add_parser("smoke")
    p.add_argument("--var", choices=("a", "b"), required=True), p.add_argument("--tag", required=True), p.add_argument("--on-start", type=int, required=True)
    p.add_argument("--out", required=True)
    s.add_parser("cmdf")
    s.add_parser("mfix")
    s.add_parser("report")
    a = ap.parse_args()
    dict(argv=cmd_argv, weights=cmd_weights, pair=cmd_pair, smoke=cmd_smoke, cmdf=cmd_cmdf, mfix=cmd_mfix, report=cmd_report)[a.sub](a)


if __name__ == "__main__":
    main()
