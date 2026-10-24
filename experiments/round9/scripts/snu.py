#!/usr/bin/env python3
"""ROUND9 NTV1 S-NU (plans/2026-10-24-ntv1-snu.md = the registration; lane scripts/snu.sh): the student D3 (NT4D3-F-s0) rolls nuPlan onpol-train
exactly as RB0-F-s0 did for on1_L18RB0 -> pool on1_L18D3 (part d3s); arm pool on1_JTSNU = d3s + on1_JTDC2's parts without rb0; arm NT5SNU-F-s0 =
D3's argv with only the S part (pool, draw file, lateral raster) changed. Box, repo root (.venv unless noted). Reuses dg4.py (D3's argv, cost
pattern), dgt.py (score / flip / highway blocks), dgc.py (same_u, smoke), pcl.py (helpers, ident), round6 dag.py (same, mfix), df_lat.py (gate).
  argv ARM TAG [--steps N] [--on-start T] [--save-at S|none] [--seed K] [--mix]   D3 = dg4.py's D3; S = the arm; Z = D3 on on1_JTSNUZ / NSNUZ
  diff               D3's rebuilt argv == its stored argv; the arm differs from it only in --tag / --on / --on-wfile / --lat-file
  c0chk              roll c0: configs vs RB0-F-s0's c0 (only throughput / port / dir lines may differ), 172 / 172 rollouts scored
  repro              (op-train) L18RB0X (today's on_rows geo + rm_build pool on RB0-F-s0's own roll) vs the stored L18RB0 files, then the scratch removed
  pgate              (AlpaSim venv) poolgate.py on L18D3 (unchanged) + R1 / R2 from its gates.json
  afix | zfix        (op-train) the arm pool's rows vs on1_JTDC2 / on1_L18D3; the identity pool's prefix vs on1_JTDC2 (dag.py mfix)
  latchk             SNUv2: build errors, the rtb1 segment bit for bit = L18NR1v2's, d3s vs rb0 raster stats
  latgate            (op-train) df_lat.py gate on SNUv2 (d3s + rtb1); human-path activations vs L18NR1v2's gate <= 1.5x
  weights | wcheck --out J      NSNU (arm) / NSNUZ (identity) -> $O/w; the rb0 rule reproduces NTV1D2's S rows
  ident --tag T --ref R --upto N --stop M --out J -- cmd     pcl.py ident without the held-out pool diagnostics (pool_held/*, pool_fit/*)
  pair --a A --b B --upto N --out J                          every scalar <= N bit for bit, the same exclusion
  rollstats | rowstats       roll scores vs RB0-F-s0 on the same scenes; rows / drawable rows of d3s vs rb0
  verdict | vverdict | build     PAI 281 / val160 verdicts, the lane summary
"""
import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dg4 as D  # noqa: E402
import dgc as G  # noqa: E402
import dgt as T_  # noqa: E402
import pcl as C  # noqa: E402

DD, REPO = C.DD, C.REPO
O, O4, O3, OV = DD / "runs/round9/ntv1_snu", D.O, D.O3, D.OV
RES = O / "results"
CR, FEAT = D.CR, DD / "runs/route_mem/feat"
W, ROWS = DD / "runs/onpol_long", DD / "runs/onpol_nuplan/rows"
STU, STU1, D1, D1S1 = D.D3, D.D3S1, D.D1, D.D1S1
ARM = "NT5SNU-F-s0"
P1L, NT, R8, IP = T_.P1L, T_.NT, T_.R8, T_.IP
N, PRE, NREF, NX, RB0T = "L18D3", "d3s", "L18RB0", "L18RB0X", "RB0-F-s0"
POOL2, POOLA, POOLZ = D.POOL2, "on1_JTSNU", "on1_JTSNUZ"
LAT_OLD, LATA = DD / "runs/defect_fix/lat/L18NR1v2", O / "lat/SNUv2"
WREL, W2 = "runs/round9/ntv1_snu/w", D.WREL + "/NTV1D2.npz"
HOLD, S_SHARE = 15, 0.8
LINE = 0.015
IGN = ("pool_held/", "pool_fit/")
AP = DD / "third_party/alpasim/.venv/bin/python"
CFG_SAME = ("driver-config.yaml", "controller-config.yaml", "eval-config.yaml", "trafficsim-config.yaml")
CFG_LOOSE = ("generated-user-config-0.yaml", "wizard-config.yaml")
CFG_OK = re.compile(r"^\s*(-\s*)?(n_concurrent_rollouts|sensorsim_cache_size|nre_cache_size|log_dir|baseport):|^\s*-\s*localhost:\d+\s*$|^\s*-\s*--cache-size=\d+\s*$")


def ld(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def out_json(name, rec):
    RES.mkdir(parents=True, exist_ok=True)
    C.dump(RES / f"{name}.json", rec)
    print(json.dumps(rec, indent=1, default=float))


# ---------------------------------------------------------------- argv
def build(arm, tag, steps=None, on_start=None, save_at=None, seed=None, mix=False):
    a = D.build("D3", tag, steps, on_start, save_at, seed, mix)
    if arm == "S":
        C.put(a, "--on", POOLA)
        C.put(a, "--on-wfile", f"{WREL}/NSNU.npz")
        C.put(a, "--lat-file", str(LATA))
    elif arm == "Z":
        C.put(a, "--on", POOLZ)
        C.put(a, "--on-wfile", f"{WREL}/NSNUZ.npz")
    return a


def cmd_argv(a):
    print("\n".join(build(a.arm, a.tag, a.steps, a.on_start, a.save_at, a.seed, a.mix)))


def cmd_diff(a):
    b = C.stored_argv(STU)
    d3, x = build("D3", STU), build("S", ARM)
    d = [(b[i - 1], b[i], x[i]) for i in range(len(b)) if b[i] != x[i]] if len(x) == len(b) else None
    want = {"--tag": ARM, "--on": POOLA, "--on-wfile": f"{WREL}/NSNU.npz", "--lat-file": str(LATA)}
    ok = d3 == b and d is not None and {q[0]: q[2] for q in d} == want and len(d) == len(want)
    rec = dict(d3_equals_stored=d3 == b, arm_diff=[list(q) for q in d] if d is not None else "length differs", want=want, arm_argv=" ".join(x), pass_=bool(ok))
    out_json("diff_arm", rec)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- roll c0
def cmd_c0chk(a):
    new, ref = W / f"roll/{STU}/c0/run", W / f"roll/{RB0T}/c0/run"
    rec = dict(new=str(new), ref=str(ref), same={}, loose={})
    for f in CFG_SAME:
        rec["same"][f] = (new / f).read_bytes() == (ref / f).read_bytes()
    for f in CFG_LOOSE:
        x, y = ((d / f).read_text().replace(str(d), "<RUN>").splitlines() for d in (ref, new))      # the run's own dir (mounts, log dir) normalised
        ch = [ln for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes() if op != "equal" for ln in x[i1:i2] + y[j1:j2]]
        bad = [ln for ln in ch if not CFG_OK.match(ln)]
        rec["loose"][f] = dict(changed_lines=len(ch), allowed=sorted({CFG_OK.match(ln).group(0).strip() for ln in ch if CFG_OK.match(ln)}), bad=bad[:20])
    lst = (W / "lists/c0.txt").read_text().split()
    R = json.loads((new / "aggregate/results-summary.json").read_text())["rollouts"]
    Rr = json.loads((ref / "aggregate/results-summary.json").read_text())["rollouts"]
    sc = {r["clipgt_id"]: r.get("score") for r in R}
    rec |= dict(scenes_listed=len(lst), rollouts=len(R), scored=sum(v is not None for v in sc.values()), listed_scored=sum(sc.get(s) is not None for s in lst),
                sim_pkl=(new / "onpol_sim.pkl").exists(), mean_new=float(np.mean([v for v in sc.values() if v is not None])),
                mean_rb0=float(np.mean([r["score"] for r in Rr if r.get("score") is not None])),
                fail_new=dict(Counter(str(r.get("failure_reason")) for r in R)), fail_rb0=dict(Counter(str(r.get("failure_reason")) for r in Rr)))
    ok = all(rec["same"].values()) and not any(v["bad"] for v in rec["loose"].values()) and rec["listed_scored"] == len(lst) == rec["rollouts"] and rec["sim_pkl"]
    rec["pass_"] = bool(ok)
    out_json("c0chk", rec)
    sys.exit(0 if ok else 2)


# ---------------------------------------------------------------- R0 reproduction
def share_eq(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return None
    if x.ndim == 0:
        return float(np.array_equal(x, y))
    if x.dtype.kind in "fc" and y.dtype.kind in "fc":
        e = (x == y) | (np.isnan(x) & np.isnan(y))
    else:
        e = x == y
    return float(e.reshape(len(x), -1).all(1).mean()) if e.ndim else float(e)


def cmp_npz(fa, fb, skip=()):
    za, zb = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    out = {k: share_eq(za[k], zb[k]) for k in sorted(set(za.files) & set(zb.files)) if k not in skip}
    out["_only_a"], out["_only_b"] = sorted(set(za.files) - set(zb.files)), sorted(set(zb.files) - set(za.files))
    return out


def cmd_repro(a):
    pairs = {f"rows/{f}": (ROWS / NX / f, ROWS / NREF / f) for f in ("meta.npz", "all.npz", "active.npz", "lon.npz")}
    pairs |= {f"pool/{f}": (CR / f"on1_{NX}" / f, CR / f"on1_{NREF}" / f) for f in ("tab.npz", "agent.npz", "road.npz", "drivable.npz", "route.npz")}
    pairs["feat"] = (FEAT / f"on1_{NX}.npz", FEAT / f"on1_{NREF}.npz")
    res, bad = {}, []
    for k, (x, y) in pairs.items():
        if not (x.exists() and y.exists()):
            res[k] = dict(missing=[str(p) for p in (x, y) if not p.exists()])
            if k in ("rows/meta.npz", "pool/tab.npz", "feat", "pool/route.npz"):
                bad.append(f"{k} missing")
            continue
        r = cmp_npz(x, y)
        res[k] = r
        bad += [f"{k}.{q} {v}" for q, v in r.items() if not q.startswith("_") and (v is None or v < 0.99)]
    rec = dict(scratch=NX, ref=NREF, compared=res, below_099=bad, pass_=not bad)
    out_json("repro_rb0", rec)
    if not bad:                                                  # the scratch copy of RB0's rows is this lane's own: removed once compared
        for p in (ROWS / NX, CR / f"on1_{NX}", CR / f"on1_{NX}@warp"):
            if p.is_dir() and p.name in (NX, f"on1_{NX}", f"on1_{NX}@warp"):
                shutil.rmtree(p)
        (FEAT / f"on1_{NX}.npz").unlink(missing_ok=True)
    sys.exit(0 if not bad else 2)


# ---------------------------------------------------------------- pool gates
def cmd_pgate(a):
    p = subprocess.run([str(AP), str(DD / "runs/route_follow/poolgate.py"), N], capture_output=True, text=True)
    pg = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else None
    g = ld(W / f"rows/{N}/gates.json") or {}
    gr = ld(W / f"rows/{NREF}/gates.json") or {}
    f = g.get("fail", ["no gates.json"])
    r12 = [x for x in f if x.startswith(("R1", "R2"))] + (["no gates.json"] if not g else [])
    rec = dict(poolgate=pg, poolgate_rc=p.returncode, poolgate_rb0=ld(DD / f"runs/route_follow/report/poolgate-{NREF}.json"),
               R1=g.get("R1"), R2=g.get("R2"), R1_rb0=gr.get("R1"), R2_rb0=gr.get("R2"), K0_items=g.get("items"), fail=f, r1_r2_fail=r12)
    rec["pass_"] = bool(p.returncode == 0 and pg and pg.get("pass") and not r12)
    out_json("pgate", rec)
    sys.exit(0 if rec["pass_"] else 2)


# ---------------------------------------------------------------- merges
def _dag():
    sys.path.insert(0, str(REPO / "experiments/round6/scripts"))
    import dag as DG
    return DG


def cmd_afix(a):
    from jevdrive.tokens import open_front
    DG = _dag()
    same = lambda x, y: G.same_u(np.asarray(x), np.asarray(y), DG.same)  # noqa: E731
    A, R, P = CR / POOLA, CR / POOL2, CR / f"on1_{N}"
    ta, tr, tp = dict(np.load(A / "tab.npz")), np.load(R / "tab.npz"), np.load(P / "tab.npz")
    n0 = len(tp["names"])
    pr = tr["pool"].astype(str)
    nr = int((pr == "rb0").sum())
    assert (pr[:nr] == "rb0").all(), f"{POOL2}: rb0 is not its first part"
    sa, sr = slice(n0, None), slice(nr, None)
    bad = []
    if set(ta) != set(tr.files):
        bad.append(f"tab keys {sorted(set(ta) ^ set(tr.files))}")
    bad += [f"tab.{k} dtype {ta[k].dtype}/{tr[k].dtype}" for k in tr.files if k in ta and ta[k].dtype.kind != tr[k].dtype.kind]
    want_nm = np.array([f"on:{PRE}:{t[3:]}" for t in tp["names"].astype(str).tolist()])
    if not np.array_equal(ta["names"].astype(str)[:n0], want_nm):
        bad.append("d3s names")
    bad += [f"tab.{k} suffix" for k in tr.files if k in ta and not same(ta[k][sa], tr[k][sr])]
    bad += [f"tab.{k} d3s" for k in tp.files if k in ta and k != "names" and not same(ta[k][:n0], tp[k])]
    for k in sorted(set(ta) - set(tp.files) - {"names", "pool", "mode"}):         # keys d3s lacks (other parts' string tags, src): filled as rb0's
        if set(map(str, ta[k][:n0].tolist())) != set(map(str, tr[k][:nr].tolist())):
            bad.append(f"tab.{k} d3s fill")
    bad += [] if (ta["pool"].astype(str)[:n0] == PRE).all() and (ta["mode"].astype(str)[:n0] == "self").all() else ["d3s pool / mode"]
    for f in ("agent.npz", "road.npz", "drivable.npz", "route.npz"):
        za, zr, zp = np.load(A / f), np.load(R / f), np.load(P / f)
        for k in zr.files:
            if k == "tokens" or not zr[k].ndim or len(zr[k]) != len(tr["names"]):
                continue
            if k not in za.files:
                bad.append(f"{f}.{k} missing")
                continue
            if not same(za[k][sa], zr[k][sr]):
                bad.append(f"{f}.{k} suffix")
            if k in zp.files and not same(za[k][:n0], zp[k]):
                bad.append(f"{f}.{k} d3s")
    fa, fr, fp = np.load(FEAT / f"{POOLA}.npz"), np.load(FEAT / f"{POOL2}.npz"), np.load(FEAT / f"on1_{N}.npz")
    bad += [f"feat.{k} suffix" for k in ("f", "k", "ok") if not same(fa[k][sa], fr[k][sr])]
    bad += [f"feat.{k} d3s" for k in ("f", "k", "ok") if not same(fa[k][:n0], fp[k].astype(fa[k].dtype) if k == "f" else fp[k])]
    xa, xr, xp = np.load(CR / f"{POOLA}@warp/teacher.npz"), np.load(CR / f"{POOL2}@warp/teacher.npz"), np.load(CR / f"on1_{N}@warp/teacher.npz")
    for k in xr.files:
        row = xr[k].ndim and len(xr[k]) == len(tr["names"])
        if not same(xa[k][sa] if row else xa[k], xr[k][sr] if row else xr[k]):
            bad.append(f"teacher.{k} suffix")
        if row and not same(xa[k][:n0], xp[k]):
            bad.append(f"teacher.{k} d3s")
    Fa, Fr, Fp = open_front(CR / f"{POOLA}@warp/front.npy"), open_front(CR / f"{POOL2}@warp/front.npy"), open_front(CR / f"on1_{N}@warp/front.npy")
    rg = np.random.default_rng(0)
    i = np.sort(rg.choice(len(Fr) - nr, 4000, replace=False))
    j = np.sort(rg.choice(n0, 2000, replace=False))
    if len(Fa) != len(ta["names"]) or not same(Fa[n0 + i], Fr[nr + i]):
        bad.append("front suffix sample")
    if not same(Fa[j], Fp[j]):
        bad.append("front d3s sample")
    agr = np.load(CR / f"on1_{NREF}/agent.npz")["ok"]
    rec = dict(pool=POOLA, rows=len(ta["names"]), parts={p: int((ta["pool"].astype(str) == p).sum()) for p in np.unique(ta["pool"].astype(str))},
               ref=POOL2, ref_rb0_rows=nr, d3s_rows=n0, rb0_agent_ok_equals_part=bool(np.array_equal(np.load(R / "agent.npz")["ok"][:nr], agr)),
               d3s_agent_ok=float(np.load(A / "agent.npz")["ok"][:n0].mean()), mismatch=bad)
    rec["pass_"] = bool(not bad and rec["rb0_agent_ok_equals_part"])
    out_json("afix", rec)
    sys.exit(0 if rec["pass_"] else 2)


def cmd_zfix(a):
    DG = _dag()
    orig = DG.same
    DG.same = lambda x, y: G.same_u(np.asarray(x), np.asarray(y), orig)
    DG.cmd_mfix(argparse.Namespace(root=str(O), pool=POOLZ, ref=POOL2))      # exits 2 on a mismatch
    rec = json.loads((O / f"checks/mfix_{POOLZ}.json").read_text())
    p2 = np.load(CR / POOL2 / "tab.npz")["pool"].astype(str)
    want = {k: int(v) for k, v in zip(*np.unique(p2, return_counts=True))} | {PRE: len(np.load(CR / f"on1_{N}" / "tab.npz")["names"])}
    rec |= dict(parts_want=want, parts_ok=rec["parts"] == want, tail_mode_ok=rec["new_rows_mode"] == ["self"])
    rec["pass_"] = bool(rec["parts_ok"] and rec["tail_mode_ok"] and not rec["mismatch"] and rec["agent_ok_fixed"] == 0)
    out_json("zfix", rec)
    sys.exit(0 if rec["pass_"] else 2)


# ---------------------------------------------------------------- lateral raster
def cmd_latchk(a):
    b, mo, mn = ld(LATA / "build.json"), np.load(LAT_OLD / "meta.npz"), np.load(LATA / "meta.npz")
    ta = np.load(CR / POOLA / "tab.npz")
    pa = ta["pool"].astype(str)
    n0, n1 = int((pa == PRE).sum()), int((pa == "rtb1").sum())
    po = np.load(CR / POOL2 / "tab.npz")["pool"].astype(str)
    nr = int((po == "rb0").sum())
    assert n1 == int((po == "rtb1").sum())
    nmn = mn["names"].astype(str)
    ok_names = bool(len(nmn) == n0 + n1 and np.array_equal(nmn, ta["names"].astype(str)[:n0 + n1])
                    and np.array_equal(nmn[n0:], mo["names"].astype(str)[nr:nr + n1]))
    seg = {k: bool(np.array_equal(mn[k][n0:n0 + n1], mo[k][nr:nr + n1])) for k in ("ok", "own_found", "junction_share")}
    for k in ("onc", "soft"):
        x, y = np.load(LATA / f"{k}.npy", mmap_mode="r"), np.load(LAT_OLD / f"{k}.npy", mmap_mode="r")
        seg[k] = bool(x.shape[1:] == y.shape[1:] and all(np.array_equal(x[n0 + i:n0 + min(i + 2048, n1)], y[nr + i:nr + min(i + 2048, n1)]) for i in range(0, n1, 2048)))
    st = lambda m, s: dict(rows=int(s.stop - s.start), ok=float(m["ok"][s].mean()), own_found=float(m["own_found"][s][m["ok"][s]].mean()),  # noqa: E731
                           junction=float(m["junction_share"][s][m["ok"][s]].mean()))
    rec = dict(build=b, names_ok=ok_names, rtb1_segment_bitwise=seg, meta_same=dict(points=str(mn["points"]) == str(mo["points"]),
               dilate=float(mn["dilate"]) == float(mo["dilate"])), d3s=st(mn, slice(0, n0)), rb0=st(mo, slice(0, nr)))
    rec["pass_"] = bool(b and b["errors"] <= 0.01 * b["rows"] and ok_names and all(seg.values()) and all(rec["meta_same"].values()))
    out_json("latchk", rec)
    sys.exit(0 if rec["pass_"] else 2)


def cmd_latgate(a):
    sys.path[:0] = [str(REPO / "experiments/defect_fix/scripts")]
    import df_lat as LT
    LT.RES = O / "lat"                                           # this lane's dir (the module default is defect_fix/results)
    LT.cmd_gate(argparse.Namespace(pool=POOLA, rows=f"{N}:{PRE},L18RTB1:rtb1", lat=str(LATA), v2="", out="lat_gate_snu",
                                   env_out=str(O / "lat/lat_pick_snu.env"), alone=True, prefix=True))
    new, old = json.loads((LATA / "lat_gate_snu.json").read_text()), json.loads((LAT_OLD / "lat_gate_v2.json").read_text())

    def key(r):
        t = next(q for q in r["hard_tried"] if q["onc_depth"] == 0.0 and q["buf"] == 0.4)
        return dict(hard_union=t["hard_union"], onc=t["onc"], side=t["side"], soft=r["plans"]["human"]["soft_d0.0"]["all"], rows_near=r["rows_near"])
    kn, ko = key(new), key(old)
    served = {k: dict(new=new["plans"]["served"][k]["all"], old=old["plans"]["served"][k]["all"]) for k in new["plans"]["served"] if not k.startswith("_")}
    rec = dict(new=kn, old=ko, served=served, verdict_new=new["verdict"], verdict_old=old["verdict"],
               pass_=bool(kn["hard_union"] <= 1.5 * ko["hard_union"] and kn["soft"] <= 1.5 * ko["soft"]))
    out_json("latgate", rec)
    sys.exit(0 if rec["pass_"] else 2)


# ---------------------------------------------------------------- draw files
def s_rule(tb, part):
    """JT00S's S rule (jt.py weights): self rows of `part` whose log is not held out (crc32(log) % 100 < 15 over the self rows' logs)."""
    logs, mode, pool = tb["log"].astype(str), tb["mode"].astype(str), tb["pool"].astype(str)
    S = mode == "self"
    hl = {g for g in set(logs[S].tolist()) if zlib.crc32(g.encode()) % 100 < HOLD}
    held = S & np.isin(logs, sorted(hl))
    return S & ~held & (pool == part), S & held & (pool == part)


def cmd_weights(a):
    Wd = np.load(DD / W2)
    tj, ta, tz = np.load(CR / POOL2 / "tab.npz"), np.load(CR / POOLA / "tab.npz"), np.load(CR / POOLZ / "tab.npz")
    nj, na, nz = (t["names"].astype(str) for t in (tj, ta, tz))
    w2, g2 = Wd["w"], Wd["grp"].astype(str)
    assert np.array_equal(Wd["names"].astype(str), nj)
    m_rb0, h_rb0 = s_rule(tj, "rb0")                             # the rule reproduces NTV1D2's S rows
    nzS = (w2 > 0) & (tj["pool"].astype(str) == "rb0")
    rule = dict(rows=int(m_rb0.sum()), ntv1d2_rows=int(nzS.sum()), same_rows=bool(np.array_equal(m_rb0, nzS)),
                rel_dev=float(np.abs(w2[nzS] / (S_SHARE / nzS.sum()) - 1).max()), held=int(h_rb0.sum()))
    assert rule["same_rows"] and rule["rel_dev"] < 1e-12, rule
    pa = ta["pool"].astype(str)
    n0 = int((pa == PRE).sum())
    nr = int((tj["pool"].astype(str) == "rb0").sum())
    assert np.array_equal(na[n0:], nj[nr:]), "arm pool suffix != on1_JTDC2 rows"
    m, h = s_rule(ta, PRE)
    w = np.zeros(len(na), w2.dtype)
    w[m] = S_SHARE / m.sum()
    w[n0:] = np.where(np.isin(pa[n0:], ["pd", "pd2"]), w2[nr:], 0.0)
    grp = np.concatenate([np.where(m[:n0], "S", np.where(h[:n0], "S_held", "")), g2[nr:]]).astype("U12")
    assert np.array_equal(nz[:len(nj)], nj) and len(nz) == len(nj) + n0, "identity pool != on1_JTDC2 + d3s"
    mz, hz = s_rule(tz, PRE)
    wz = np.concatenate([w2, np.zeros(n0, w2.dtype)])
    gz = np.concatenate([g2, np.where(mz[len(nj):], "S", np.where(hz[len(nj):], "S_held", ""))]).astype(grp.dtype)
    (O / "w").mkdir(parents=True, exist_ok=True)
    for nm, names, ww, gg, pool in (("NSNU", na, w, grp, POOLA), ("NSNUZ", nz, wz, gz, POOLZ)):
        np.savez(O / "w" / f"{nm}.npz", names=names, w=ww, grp=gg)
        out_json(f"w_{nm}", dict(pool=pool, rows=len(names), s_rows=int(((ww > 0) & (gg == "S")).sum()), s_held_rows=int((gg == "S_held").sum()),
                                 total=float(ww.sum()), group_share=D.gshare(ww, gg), rule_on_rb0=rule, out=f"{WREL}/{nm}.npz"))


def cmd_wcheck(a):
    Wd = np.load(DD / W2)
    tj, ta, tz = np.load(CR / POOL2 / "tab.npz"), np.load(CR / POOLA / "tab.npz"), np.load(CR / POOLZ / "tab.npz")
    A, Z = np.load(O / "w/NSNU.npz"), np.load(O / "w/NSNUZ.npz")
    w2, w, wz = Wd["w"], A["w"], Z["w"]
    pa, ma = ta["pool"].astype(str), ta["mode"].astype(str)
    n0, nj = int((pa == PRE).sum()), len(tj["names"])
    nr = int((tj["pool"].astype(str) == "rb0").sum())
    m, h = s_rule(ta, PRE)
    dag = np.isin(pa, ["pd", "pd2"])
    sh = lambda ww, q: float(ww[q].sum() / ww.sum())  # noqa: E731
    fut = ta["fut"][w > 0]
    ck = dict(names=bool(np.array_equal(A["names"].astype(str), ta["names"].astype(str)) and np.array_equal(Z["names"].astype(str), tz["names"].astype(str))),
              s_rows=int((w[:n0] > 0).sum()), s_rows_rule=int(m.sum()), s_ids_rule=bool(np.array_equal((w > 0) & ~dag, m)),
              s_parts=sorted(set(pa[(w > 0) & ~dag])), s_modes=sorted(set(ma[(w > 0) & ~dag])), s_uniform=bool(np.unique(w[m]).size == 1),
              s_held_zero=bool(not w[h].any()), share_s=sh(w, m), dag_bitwise=bool(np.array_equal(w[n0:][dag[n0:]], w2[nr:][dag[n0:]])),
              dag_rows_equal_ntv1d2=bool(np.array_equal(np.flatnonzero((w[n0:] > 0) & dag[n0:]), np.flatnonzero((w2[nr:] > 0) & dag[n0:]))),
              ntv1d2_nonzero_outside_rb0_and_dag=int(((w2[nr:] > 0) & ~dag[n0:]).sum()), other_zero=bool(not w[~m & ~dag].any()),
              share_pd=sh(w, pa == "pd"), share_pd2=sh(w, pa == "pd2"), share_pd_ntv1d2=sh(w2, tj["pool"].astype(str) == "pd"),
              share_pd2_ntv1d2=sh(w2, tj["pool"].astype(str) == "pd2"), total=float(w.sum()), fut_finite=bool(np.isfinite(fut).all()),
              z_prefix_bitwise=bool(wz.dtype == w2.dtype and np.array_equal(wz[:nj], w2)), z_tail_zero=bool(not wz[nj:].any()),
              z_grp_prefix=bool(np.array_equal(Z["grp"].astype(str)[:nj], Wd["grp"].astype(str))), z_tail_part=sorted(set(tz["pool"].astype(str)[nj:])))
    ok = (ck["names"] and ck["s_ids_rule"] and ck["s_parts"] == [PRE] and ck["s_modes"] == ["self"] and ck["s_uniform"] and ck["s_held_zero"]
          and abs(ck["share_s"] - S_SHARE) < 1e-9 and ck["dag_bitwise"] and ck["dag_rows_equal_ntv1d2"] and ck["ntv1d2_nonzero_outside_rb0_and_dag"] == 0
          and ck["other_zero"] and abs(ck["share_pd"] - ck["share_pd_ntv1d2"]) < 1e-9 and abs(ck["share_pd2"] - ck["share_pd2_ntv1d2"]) < 1e-9
          and abs(ck["total"] - 1) < 1e-9 and ck["fut_finite"] and ck["z_prefix_bitwise"] and ck["z_tail_zero"] and ck["z_grp_prefix"] and ck["z_tail_part"] == [PRE])
    rec = dict(files=[f"{WREL}/NSNU.npz", f"{WREL}/NSNUZ.npz"], ref=W2, checks=ck, pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


# ---------------------------------------------------------------- run comparisons without the held-out pool diagnostics
def fcompare(d, r, upto, allow_new):
    x, y = C.scalars(d, upto), C.scalars(r, upto)
    ig = lambda k: k[0].startswith(IGN)  # noqa: E731
    xi, yi = {k: v for k, v in x.items() if ig(k)}, {k: v for k, v in y.items() if ig(k)}
    keep = lambda s: {k: v for k, v in s.items() if not ig(k)}  # noqa: E731
    sx = C.scalars
    C.scalars = lambda dd, uu: keep(sx(dd, uu))
    try:
        rec = C.COMPARE(d, r, upto, allow_new)
    finally:
        C.scalars = sx
    dif = sorted(f"{k[0]}@{k[1]}" for k in set(xi) & set(yi) if xi[k] != yi[k])
    rec["excluded"] = dict(prefixes=list(IGN), tags=sorted({k[0] for k in set(xi) | set(yi)}), n_new=len(xi), n_ref=len(yi), differ=len(dif), differ_sample=dif[:12],
                           only_new=len(set(xi) - set(yi)), only_ref=len(set(yi) - set(xi)))
    return rec


C.COMPARE = C.compare


def cmd_ident(a):
    C.compare = fcompare
    C.cmd_ident(a)


def cmd_pair(a):
    da, db = C.run_dirs(a.a)[-1], C.run_dirs(a.b)[-1]
    rec = fcompare(db, da, a.upto, False) | dict(a=a.a, b=a.b, run_a=str(da), run_b=str(db))
    rec["pool_rows_b"] = C.pool_rows(db)
    rec["pool_drawn"] = bool(any(v > 0 for v in rec["pool_rows_b"].values()))
    rec["pass_"] = bool(rec["bit_identical"] and rec["pool_drawn"])
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["pass_"] else 2)


# ---------------------------------------------------------------- by-products
def roll_scores(tag):
    out = {}
    for c in range(6):
        f = W / f"roll/{tag}/c{c}/run/aggregate/results-summary.json"
        for r in json.loads(f.read_text())["rollouts"] if f.exists() else []:
            out[r["clipgt_id"]] = (r.get("score"), str(r.get("failure_reason")))
    return out


def cmd_rollstats(a):
    from jevdrive import stats
    X, Y = roll_scores(STU), roll_scores(RB0T)
    cm = sorted(k for k in set(X) & set(Y) if X[k][0] is not None and Y[k][0] is not None)
    x, y = np.array([X[k][0] for k in cm]), np.array([Y[k][0] for k in cm])
    rec = dict(scenes_new=len(X), scenes_rb0=len(Y), common=len(cm), mean_new=float(x.mean()), mean_rb0=float(y.mean()), paired=stats.paired(x, y),
               zeros_new=int((x == 0).sum()), zeros_rb0=int((y == 0).sum()), full_new=int((x >= 0.98).sum()), full_rb0=int((y >= 0.98).sum()),
               fail_new=dict(Counter(X[k][1] for k in cm)), fail_rb0=dict(Counter(Y[k][1] for k in cm)))
    out_json("rollstats", rec)


def cmd_rowstats(a):
    rec = {}
    for nm, rows, pool, part in ((N, N, POOLA, PRE), (NREF, NREF, POOL2, "rb0")):
        M, Al = np.load(ROWS / rows / "meta.npz", allow_pickle=True), np.load(ROWS / rows / "all.npz", allow_pickle=True)
        tb = np.load(CR / pool / "tab.npz") if (CR / pool / "tab.npz").exists() else None
        v, e0 = M["speed"].astype(float), np.abs(M["e0"].astype(float))
        r = dict(rows=len(M["names"]), scenes=len(set(M["scene"].tolist())), logs=len(set(M["log"].tolist())),
                 speed_q=[float(q) for q in np.nanquantile(v, [0.1, 0.5, 0.9])], static_lt05=float((v < 0.5).mean()), fast_gt13=float((v > 13).mean()),
                 e0_gt1=float((e0 > 1).mean()), e0_med=float(np.nanmedian(e0)), decisions=len(Al["k"]), excluded=dict(Counter(Al["excl"].astype(str).tolist())))
        if tb is not None:
            m, h = s_rule(tb, part)
            r |= dict(drawable=int(m.sum()), held=int(h.sum()))
        rec[nm] = r
    out_json("rowstats", rec)


# ---------------------------------------------------------------- verdicts
def cost(o=O):
    from jevdrive.cl import pool as P
    J = P.snapshot()[0]["jobs"]
    h_, t0s, t1s = Counter(), [], []
    for f, grp in ((o / "jobs.txt", None), (o / "v160/jobs.txt", "val160")):
        for ln in (f.read_text().splitlines() if f.exists() else []):
            if len(ln.split()) < 2:
                continue
            i, nm = ln.split()[:2]
            j = J.get(i)
            if not j or not j.get("t0") or j["spec"]["vram_gb"] <= 1:
                continue
            t1 = j.get("t1") or time.time()
            k = grp or ("roll" if nm.startswith("snu-roll") else "read281" if nm.startswith("pai-") else "train" if "-train-" in nm
                        else "build" if nm == "snu-tok" else "gates")
            h_[k] += (t1 - j["t0"]) / 3600
            t0s.append(j["t0"]), t1s.append(t1)
    return {k: float(v) for k, v in h_.items()} | dict(gpu_job_h=float(sum(h_.values())), wall_h=(max(t1s) - min(t0s)) / 3600 if t0s else 0.0)


def cmd_verdict(a):
    from jevdrive import stats
    labs = [ARM, STU, STU1, D1, D1S1, P1L, NT, R8, IP]
    OLD = {t: T_.facts(f"{t}-281ag") for t in labs}
    NEW = T_.ref_new()
    NEW[D1] = T_.new_series(O3 / "sv2/new/rollouts.csv", f"{D1}-ag")
    NEW[D1S1] = T_.new_series(O3 / "sv2c/new/rollouts.csv", f"{D1S1}-ag")
    NEW[STU] = T_.new_series(O4 / "sv2/new/rollouts.csv", f"{STU}-ag")
    NEW[STU1] = T_.new_series(O4 / "sv2s1/new/rollouts.csv", f"{STU1}-ag")
    NEW[ARM] = T_.new_series(O / "sv2/new/rollouts.csv", f"{ARM}-ag")
    nm = {ARM: "S-NU", STU: "D3-s0", STU1: "D3-s1", D1: "D1-s0", D1S1: "D1-s1", P1L: "P1L", NT: "NTV0-s0", R8: "R8K-s2", IP: "INTP75-NR (ref.)"}
    res = T_.score_block(labs, OLD, NEW, [(ARM, y) for y in (STU, STU1, D1, R8, P1L, NT, IP)], nm)
    S = res["_S"]
    m2 = {sc: (S[sc][STU] + S[sc][STU1]) / 2 for sc in ("new", "old")}
    res["d3_2seed_mean"] = {sc: float(v.mean()) for sc, v in m2.items()}
    res["arm_vs_d3_mean"] = {sc: stats.paired(S[sc][ARM], m2[sc]) for sc in m2}
    lines = {}
    for sc in ("new", "old"):
        x = res["pairs"][f"{ARM}:{STU}"][sc]
        lines[sc] = dict(arm_d3=x, line="fresh own S helps" if x["mean"] >= LINE else "fresh own S hurts" if x["mean"] <= -LINE else "neutral", registered=sc == "new")
    res["lines"] = lines
    res["highway"] = T_.highway([(ARM, O / "bs", f"{ARM}-281ag"), (STU, O4 / "bs", f"{STU}-281ag")]
                                + [(t, DD / "runs/pai_av/behavior_stats/out", f"{t}-281ag") for t in (NT, R8)], NEW)
    res["gates"] = T_.gates(O)
    r = ld(RES / "ratio_arm.json")
    if r:
        res["ratio"] = dict(r500=r["ratio"], pass_=r["pass_"], after_T=C.grad_ratio(C.run_dirs(ARM)[-1], 10000, 19500)["ratio"],
                            d3_after_T=C.grad_ratio(C.run_dirs(STU)[-1], 10000, 19500)["ratio"])
    res["ade"] = {t: dict(at_10000=C.dev_ade(t, 10000), end=C.dev_ade(t)) for t in (ARM, STU)}
    res["cost"] = cost()
    L = ["# NTV1 S-NU verdict: D3's recipe with the S part from D3's own nuPlan rollouts (PAI 281, ACC + GAP, Harmonizer OFF; D3's serve)", ""]
    L += T_.md_block(res, nm, "Reads")
    dm = res["arm_vs_d3_mean"]
    L += ["", f"D3 2-seed mean (s0 + s1) / 2 on these scenes: new {res['d3_2seed_mean']['new']:.3f}, old {res['d3_2seed_mean']['old']:.3f}; "
          f"S-NU - D3 2-seed mean: new {C.f3(dm['new'])}, old {C.f3(dm['old'])}.", "", "## Registered line (new scorer registered, old reported)", ""]
    for sc, q in lines.items():
        L.append(f"- {sc} scorer: S-NU - D3-s0 {C.f3(q['arm_d3'])} (helps >= +{LINE}, hurts <= -{LINE}) -> **{q['line']}**" + ("" if q["registered"] else " (reported)"))
    L += T_.hw_md(res["highway"], nm)
    L += ["", "## Gates", ""] + [f"- {k}: {v}" for k, v in res["gates"].items()]
    if "ratio" in res:
        q = res["ratio"]
        L.append(f"- S / imit gradient ratio: 500 steps after T {q['r500']:.3f} (band {C.RATIO_BAND[0]:.2f}-{C.RATIO_BAND[1]:.2f}); whole post-T {q['after_T']:.3f} (D3 {q['d3_after_T']:.3f})")
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
    src = {ARM: O / "v160/sv2/new/rollouts.csv", STU: O4 / "v160/sv2/new/rollouts.csv", STU1: O4 / "v160/sv2/new/rollouts.csv",
           R8: OV / "sv2/new/rollouts.csv", D1: OV / "sv2/new/rollouts.csv", P1L: OV / "sv2/new/rollouts.csv"}
    labs = list(src)
    OLD = {t: T_.facts(f"{t}-v160ag") for t in labs}
    NEW = {t: T_.new_series(f, f"{t}-v160") for t, f in src.items()}
    nm = {ARM: "S-NU", STU: "D3-s0", STU1: "D3-s1", R8: "R8K-s2", D1: "D1-s0", P1L: "P1L"}
    res = T_.score_block(labs, OLD, NEW, [(ARM, y) for y in (STU, STU1, R8, D1, P1L)], nm)
    S = res["_S"]
    m2 = {sc: (S[sc][STU] + S[sc][STU1]) / 2 for sc in ("new", "old")}
    res["d3_2seed_mean"] = {sc: float(v.mean()) for sc, v in m2.items()}
    res["arm_vs_d3_mean"] = {sc: stats.paired(S[sc][ARM], m2[sc]) for sc in m2}
    L = ["# NTV1 S-NU on PAI val160 (ACC + GAP, Harmonizer off, v160.sh's chunks and serve) vs the stored D3 / R8K-s2 / D1 / P1L reads", ""]
    L += T_.md_block(res, nm, "val160")
    dm = res["arm_vs_d3_mean"]
    L += ["", f"D3 2-seed mean: new {res['d3_2seed_mean']['new']:.3f}, old {res['d3_2seed_mean']['old']:.3f}; S-NU - D3 2-seed mean: new {C.f3(dm['new'])}, old {C.f3(dm['old'])}."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", T_.strip(res))
    print("\n".join(L))


def du(p):
    return D.du(p)


def cmd_build(a):
    g = {f.stem: (lambda j: j.get("pass_", j.get("bit_identical")) if isinstance(j, dict) else None)(ld(f)) for f in sorted(RES.glob("*.json"))}
    disk = {k: du(v) for k, v in dict(roll=W / f"roll/{STU}", rows=ROWS / N, rows_long=W / f"rows/{N}", pool=CR / f"on1_{N}", pool_warp=CR / f"on1_{N}@warp",
                                         feat=FEAT / f"on1_{N}.npz", arm_pool=CR / POOLA, arm_feat=FEAT / f"{POOLA}.npz", id_pool=CR / POOLZ,
                                         id_feat=FEAT / f"{POOLZ}.npz", lat=LATA, w=O / "w", reads=O / "pai281", v160=O / "v160").items()}
    disk["total"] = sum(disk.values())
    rec = dict(student=STU, arm=ARM, roll=ld(RES / "rollstats.json"), rows=ld(RES / "rowstats.json"), w=ld(RES / "w_NSNU.json"), gates=g, disk_gib=disk,
               cost=cost(), t=time.strftime("%F %T"))
    out_json("build", rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("argv")
    p.add_argument("arm", choices=("D3", "S", "Z")), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at"), p.add_argument("--seed", type=int)
    p.add_argument("--mix", action="store_true")
    for k in ("diff", "c0chk", "repro", "pgate", "afix", "zfix", "latchk", "latgate", "weights", "rollstats", "rowstats", "build"):
        s.add_parser(k)
    s.add_parser("wcheck").add_argument("--out", required=True)
    p = s.add_parser("ident")
    p.add_argument("--tag", required=True), p.add_argument("--ref", required=True), p.add_argument("--upto", type=int, required=True)
    p.add_argument("--stop", type=int, required=True), p.add_argument("--allow-new", action="store_true"), p.add_argument("--out", required=True)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p = s.add_parser("pair")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--upto", type=int, required=True), p.add_argument("--out", required=True)
    for k in ("verdict", "vverdict"):
        s.add_parser(k).add_argument("--out", required=True)
    a = ap.parse_args()
    if a.sub == "ident":
        a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        assert a.cmd, "ident needs the trainer command after --"
    dict(argv=cmd_argv, diff=cmd_diff, c0chk=cmd_c0chk, repro=cmd_repro, pgate=cmd_pgate, afix=cmd_afix, zfix=cmd_zfix, latchk=cmd_latchk,
         latgate=cmd_latgate, weights=cmd_weights, wcheck=cmd_wcheck, ident=cmd_ident, pair=cmd_pair, rollstats=cmd_rollstats, rowstats=cmd_rowstats,
         verdict=cmd_verdict, vverdict=cmd_vverdict, build=cmd_build)[a.sub](a)


if __name__ == "__main__":
    main()
