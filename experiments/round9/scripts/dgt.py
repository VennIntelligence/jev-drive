#!/usr/bin/env python3
"""ROUND9 NTV1 step 3 (plans/2026-10-24-ntv1-step3.md = the registration; lane scripts/dgt.sh): P1L + P1L's own DAgger rows. Box, repo root, repo venv.
Reuses pcl.py (P1L argv, ident, ratio, post, compare, cost), dgc.py (step 2a: variant b's pool flags, pair, smoke), p1l.py (lrcheck).
  argv ARM TAG [--steps N] [--on-start T] [--save-at S|none] [--seed K] [--mix]   trainer argv, one token a line
        D1 = P1L + --on on1_JTDC --on-wfile NTV1D --on-k 40 (dgc.py variant b)      D2 = D1 + --ho-distill 0
  diff ARM               token diff against PCL1L-F-s0's stored argv; exit 1 on any diff but the registered ones
  wcheck --out J         NTV1D: S part = JT00S-rb0's rows (ids, x 0.8, uniform), DAgger part = on1_PAIDC's drawable rows (uniform), shares 0.8 / 0.2
  rowcheck --out J       D1's DAgger rows vs R6DAG-L18-s0's (argv per-row flags; pool mode / fut / agent ok / road ok / lateral coverage); differences
                         recorded, exit 2 only if our DAgger rows are not `both` or a drawable row has a non-finite label
  dsw --a TAG --b TAG --on-start T --out J   D2's switch on the two smokes: scalars <= T equal (reported), some post-T loss scalar differs (gate)
  bs --out DIR --tags T,..                   behavior_stats (the box copy) on these reads
  verdict --dir O --out P                    lines, paired CIs, classes, flips, highway, gates, cost -> P.{md,json}; the candidate -> O/CAND
  cverdict --dir O --out P                   candidate tail: seeds 0 / 1, Harmonizer OFF / ON -> P.{md,json}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dgc as G  # noqa: E402
import pcl as C  # noqa: E402

DD, REPO = C.DD, C.REPO
O_ = DD / "runs/round9/ntv1_step3"
CR = DD / "runs/op_parity/cache"
F_ = DD / "runs/pai_fix/facts"
TAGS = dict(D1="NT3D1-F-s0", D2="NT3D2-F-s0")
P1L, NT, R8, IP = "PCL1L-F-s0", "NTV0-L18-s0", "R8K-L18-s2", "INTP75-NR"
REFS = [P1L, NT, R8, IP]
NTV1D = "runs/round9/ntv1_step2/w/NTV1D.npz"
LINE_HELP, LINE_HURT, LINE_D2 = 0.02, -0.01, -0.01             # registered (plan section 4)
CAND = 0.86                                                    # candidate line (plan section 5)
OK = 0.98
LAT = ("offroad", "zero-other", "side")
ROW_FLAGS = ["--on-mode", "--on-lon-all", "--on-normal", "--ho-w", "--agent-lam", "--agent-on-only", "--shape", "--lat-onc", "--lat-soft", "--lat-side",
             "--lat-file", "--lat-depth", "--lat-soft-depth", "--lat-buf", "--lat-host", "--on-arc", "--on-arc-rows", "--on-hold-pct", "--ho-distill",
             "--label-bank"]


def build(arm, tag, steps=None, on_start=None, save_at=None, seed=None, mix=False):
    a = G.build_argv("b", tag, steps, on_start, save_at, mix)
    if seed is not None:
        C.put(a, "--seed", str(seed))
    if arm == "D2":
        assert "--ho-distill" not in a
        a += ["--ho-distill", "0"]
    return a


def cmd_argv(a):
    print("\n".join(build(a.arm, a.tag, a.steps, a.on_start, a.save_at, a.seed, a.mix)))


def cmd_diff(a):
    b, x = C.stored_argv(P1L), build(a.arm, TAGS[a.arm])
    print(" ".join(x))
    extra = ["--ho-distill", "0"] if a.arm == "D2" else []
    d = [(b[i - 1], b[i], x[i]) for i in range(len(b)) if b[i] != x[i]]
    for q in d:
        print(f"  {q[0]}: {q[1]} -> {q[2]}")
    print(f"  appended: {x[len(b):]}")
    want = {"--tag": TAGS[a.arm], "--on": "on1_JTDC", "--on-k": "40", "--on-wfile": NTV1D}
    ok = x[len(b):] == extra and len(x) == len(b) + len(extra) and {q[0]: q[2] for q in d} == want and len(d) == len(want)
    print("TOKEN DIFF AS REGISTERED" if ok else f"UNEXPECTED DIFF (want {want} + {extra})")
    sys.exit(0 if ok else 1)


def flagmap(argv):
    m = {}
    for i, t in enumerate(argv):
        if t.startswith("--"):
            m[t] = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
    return m


def cmd_wcheck(a):
    W, B = np.load(DD / NTV1D), np.load(DD / C.WFILE)
    tb, X = np.load(CR / "on1_JTDC/tab.npz"), np.load(CR / "on1_PAIDC/dagger.npz")
    names, n0 = tb["names"].astype(str), len(B["names"])
    part, mode = tb["pool"].astype(str), tb["mode"].astype(str)
    w, bw = W["w"].astype(np.float64), B["w"].astype(np.float64)
    s_ids, b_ids = np.flatnonzero(w[:n0] > 0), np.flatnonzero(bw > 0)
    d_ids, x_ids = np.flatnonzero(w[n0:] > 0), np.flatnonzero(X["draw"])
    ws = bw / bw.sum() * 0.8
    fut = tb["fut"][n0 + d_ids]
    xn = X["names"].astype(str)
    ck = dict(names_equal=bool(np.array_equal(W["names"].astype(str), names)), prefix_equal=bool(np.array_equal(names[:n0], B["names"].astype(str))),
              tail_is_paidc=bool(len(names) == n0 + len(xn) and all(p.endswith(q[2:]) for p, q in zip(names[n0:], xn))),
              s_rows=int(len(s_ids)), s_ids_equal=bool(np.array_equal(s_ids, b_ids)),
              s_parts=sorted(set(part[s_ids])), s_modes=sorted(set(mode[s_ids])),
              s_rel_err=float(np.abs(w[b_ids] / ws[b_ids] - 1).max()), s_uniform=bool(np.unique(w[s_ids]).size == 1),
              d_rows=int(len(d_ids)), d_ids_equal_drawable=bool(np.array_equal(d_ids, x_ids)), d_parts=sorted(set(part[n0 + d_ids])),
              d_modes=sorted(set(mode[n0 + d_ids])), d_uniform=bool(np.unique(w[n0 + d_ids]).size == 1), d_fut_finite=bool(np.isfinite(fut).all()),
              share_s=float(w[:n0].sum() / w.sum()), share_pd=float(w[n0:].sum() / w.sum()),
              grp_prefix_equal=bool(np.array_equal(W["grp"].astype(str)[:n0], B["grp"].astype(str))),
              grp_tail=sorted(set(W["grp"].astype(str)[n0:])))
    ok = (ck["names_equal"] and ck["prefix_equal"] and ck["tail_is_paidc"] and ck["s_rows"] == 13793 and ck["s_ids_equal"] and ck["s_parts"] == ["rb0"]
          and ck["s_modes"] == ["self"] and ck["s_rel_err"] <= 1e-12 and ck["s_uniform"] and ck["d_ids_equal_drawable"] and ck["d_parts"] == ["pd"]
          and ck["d_modes"] == ["both"] and ck["d_uniform"] and ck["d_fut_finite"] and abs(ck["share_s"] - 0.8) < 1e-9 and abs(ck["share_pd"] - 0.2) < 1e-9
          and ck["grp_prefix_equal"] and ck["grp_tail"] == ["pd"])
    rec = dict(file=NTV1D, base=C.WFILE, checks=ck, pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def pool_part(pool, wfile, part="pd"):
    tb = np.load(CR / pool / "tab.npz")
    p, mode = tb["pool"].astype(str), tb["mode"].astype(str)
    i = np.flatnonzero(p == part)
    w = np.load(wfile)["w"].astype(np.float64)
    dr = i[w[i] > 0]
    ag, rd = np.load(CR / pool / "agent.npz")["ok"].astype(bool), np.load(CR / pool / "road.npz")["ok"].astype(bool)
    lat = len(np.load(DD / "runs/defect_fix/lat/L18NR1v2/meta.npz")["names"])
    fut = tb["fut"][dr]
    return dict(pool=pool, rows=int(len(i)), first_row=int(i.min()), modes={k: int(v) for k, v in zip(*np.unique(mode[i], return_counts=True))},
                drawable=int(len(dr)), draw_share=float(w[i].sum() / w.sum()), fut_finite_drawable=float(np.isfinite(fut.reshape(len(fut), -1)).all(1).mean()),
                agent_ok_all=float(ag[i].mean()), agent_ok_drawable=float(ag[dr].mean()), road_ok_all=float(rd[i].mean()), road_ok_drawable=float(rd[dr].mean()),
                lat_raster_rows=lat, lateral_rows=int((i < lat).sum()))


def cmd_rowcheck(a):
    r6, d1 = C.stored_argv("R6DAG-L18-s0"), build("D1", TAGS["D1"])
    fr, fd = flagmap(r6), flagmap(d1)
    per_row = {k: dict(R6DAG=fr.get(k), D1=fd.get(k)) for k in ROW_FLAGS}
    row_diff = {k: v for k, v in per_row.items() if v["R6DAG"] != v["D1"]}
    other = {k: dict(R6DAG=fr.get(k), D1=fd.get(k)) for k in sorted(set(fr) | set(fd)) if k not in ROW_FLAGS and k != "--tag" and fr.get(k) != fd.get(k)}
    pr = pool_part("on1_JTPHD", DD / fr["--on-wfile"])          # absolute in R6DAG's argv: DD / abs = abs
    po = pool_part("on1_JTDC", DD / NTV1D)
    dags = dict(R6DAG=pr, D1=po)
    hard = po["modes"] == {"both": po["rows"]} and po["fut_finite_drawable"] == 1.0
    rec = dict(r6dag_argv=" ".join(r6), d1_argv=" ".join(d1), per_row_flags=per_row, per_row_differences=row_diff, other_argv_differences=other,
               dagger_rows=dags, note=("DAgger rows are mixed-mode `both` rows: imitation to the b3 label + distill (non-plan heads) + the pool's agent / road "
                                      "hinge where agent ok / road ok, 25 % route-memory drop (--on-normal); the nr1_lat raster covers the pool's first "
                                      "lat_raster_rows rows only; --ho-distill acts on `self` rows' distill_ho only; --ego-scale-* acts on non-pool rows only"),
               pass_=bool(hard))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1, default=float))
    sys.exit(0 if hard else 2)


def cmd_dsw(a):
    da, db = C.run_dirs(a.a)[-1], C.run_dirs(a.b)[-1]
    pre = C.compare(db, da, a.on_start, False)
    x, y = C.scalars(da), C.scalars(db)
    post = sorted(f"{k[0]}@{k[1]}" for k in set(x) & set(y) if k[1] > a.on_start and k[0].startswith("loss/") and x[k] != y[k])
    dh = sorted(f"{k[0]}@{k[1]}" for k in y if k[0] == "loss/distill_ho" and k[1] > a.on_start)
    ok = bool(post) and bool(dh)
    rec = dict(a=a.a, b=a.b, pre_T=dict(bit_identical=pre["bit_identical"], compared=pre["scalars_compared"], identical=pre["identical"],
                                         note="reported, not a gate"),
               post_T_loss_scalars_differing=len(post), first_differing=post[:10], distill_ho_logged_after_T=dh[:5], pass_=ok)
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def cmd_bs(a):
    import importlib.util
    sp = importlib.util.spec_from_file_location("behavior_stats", DD / "runs/pai_av/behavior_stats/code/behavior_stats.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules["behavior_stats"] = m
    sp.loader.exec_module(m)
    tags = a.tags.split(",")
    for t in tags:
        m.READS[t] = (m.TAX / f"{t}-281ag.pkl", m.FACTS / f"{t}-281ag.csv", f"{t}-ag")
    m.cmd_run(argparse.Namespace(out=a.out, workers=a.workers, limit=0, reads=",".join(tags)))


# ---------------------------------------------------------------- verdict
def new_series(path, read):
    import pandas as pd
    if not Path(path).exists():
        return None
    x = pd.read_csv(path)
    x = x[x.read == read]
    return x.drop_duplicates("scene", keep="last").set_index("scene").score if len(x) else None


def ref_new(h=""):
    import pandas as pd
    PS = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    out = {NT: PS[f"{NT}-ag{h}"], R8: PS[f"{R8}-ag{h}"]}
    if h:
        s = new_series(DD / "runs/round9/diag_reads/sv2/new/rollouts.csv", f"{P1L}-agH")
        if s is not None:
            out[P1L] = s
    else:
        out[P1L] = new_series(DD / "runs/round9/p1l/sv2/new/rollouts.csv", f"{P1L}-ag")
        out[IP] = new_series(DD / "runs/round9/interp/sv2/new/rollouts.csv", f"{IP}-ag")
    return out


def score_block(labs, OLD, NEW, pairs, nm):
    """common set, per-read metrics, paired CIs with class differences, flips on NTV0's full-score scenes."""
    from jevdrive import stats
    sys.path.insert(0, str(REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    rpc = {t: int(F.rpc.fillna(False).astype(bool).sum()) for t, F in OLD.items()}
    bad = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    common = sorted(set.intersection(*[set(OLD[t].index[OLD[t].score.notna()]) for t in labs])
                    & set.intersection(*[set(NEW[t].dropna().index) for t in labs]) - bad)
    n = len(common)
    S = dict(new={t: NEW[t].reindex(common).to_numpy(float) for t in labs}, old={t: OLD[t].score.reindex(common).to_numpy(float) for t in labs})
    fo = {t: OLD[t].reindex(common) for t in labs}
    res = dict(scenes=n, rpc=rpc, reads={}, pairs={}, flips={})
    for t in labs:
        m = PF.metrics(fo[t])
        res["reads"][t] = dict(old=m, new_mean=float(S["new"][t].mean()), new_zeros=int((S["new"][t] == 0).sum()),
                               lateral=int(sum(m[f"z_{c}"] for c in LAT)), route_turn=int(m["z_route-turn"]), af_front=int(m["z_AF-front"]),
                               dev_ade=C.dev_ade(t) if (C.RUNS / t / "route_mem.json").exists() else None)
    cls = PF.ZCLS + PF.PCLS + ["zeros", "lateral"]
    for x_, y in pairs:
        p = {sc: stats.paired(S[sc][x_], S[sc][y]) for sc in ("new", "old")}
        cl = {}
        for c in cls:
            def ind(q):
                F = fo[q]
                if c == "zeros":
                    return S["old"][q] == 0
                if c == "lateral":
                    return ((F.score == 0) & F.fclass.isin(LAT)).to_numpy()
                return (F.fclass == c).to_numpy()
            u, v = ind(x_).astype(float), ind(y).astype(float)
            q_ = stats.paired(u, v)
            cl[c] = dict(a=int(u.sum()), b=int(v.sum()), diff=n * q_["mean"], lo=n * q_["lo"], hi=n * q_["hi"])
        res["pairs"][f"{x_}:{y}"] = p | dict(classes=cl)
    if NT in labs:
        fa = S["new"][NT] >= OK                                       # count A (INTERP): NTV0 new >= 0.98, read new < 0.98
        fb = S["old"][NT] >= OK                                       # count B (step 1): NTV0 old >= 0.98, read new == 0
        r8f = (~fa) & (S["new"].get(R8, np.zeros(n)) >= OK) if R8 in labs else None
        for t in labs:
            if t == NT:
                continue
            s = S["new"][t]
            neg = fa & (s < OK)
            res["flips"][t] = dict(full_a=int(fa.sum()), neg_a=int(neg.sum()), neg_a_zero=int((fa & (s == 0)).sum()),
                                   neg_a_pts=float((S["new"][NT] - s)[neg].sum()), full_b=int(fb.sum()), neg_b=int((fb & (s == 0)).sum()),
                                   fixed=int(((~fa) & (s >= OK)).sum()),
                                   keeps_r8k_fixes=int((r8f & (s >= OK)).sum()) if r8f is not None else None,
                                   r8k_fixes=int(r8f.sum()) if r8f is not None else None,
                                   neg_scenes=[c for c, q in zip(common, neg) if q])
    res["_S"], res["_common"] = S, common
    return res


def md_block(res, nm, title):
    sys.path.insert(0, str(REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    f3 = C.f3
    L = [f"## {title}", "", f"{res['scenes']} scenes scored by every read below (both scorers) with no driver RPC error in any; RPC scenes per read {res['rpc']}.",
         "", "| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | "
         "launched | dev ADE |", "|:--|" + "--:|" * 14]
    for t, r in res["reads"].items():
        m = r["old"]
        L.append(f"| {nm.get(t, t)} | {r['new_mean']:.3f} | {m['mean']:.3f} | {r['new_zeros']} / {m['zeros']} | {r['lateral']} | {m['z_offroad']} | "
                 f"{m['z_zero-other']} | {m['z_side']} | {m['z_route-turn']} | {m['z_overshoot']} | {m['z_AF-front']} | {m['p_rear-hit']} | {m['p_slow']} | "
                 f"{m['launch']} | " + (f"{r['dev_ade']:.4f} |" if r["dev_ade"] else "- |"))
    L += ["", "Paired differences (95 % bootstrap CI, scenes as units; class counts from old-scorer facts):", "",
          "| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn | AF-front |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for k, p in res["pairs"].items():
        c = p["classes"]
        x_, y = k.split(":")
        L.append(f"| {nm.get(x_, x_)} - {nm.get(y, y)} | {f3(p['new'])} | {f3(p['old'])} | " + " | ".join(
            f"{c[q]['diff']:+.0f} [{c[q]['lo']:+.0f}, {c[q]['hi']:+.0f}]" for q in ("zeros", "lateral", "route-turn", "AF-front")) + " |")
    cz = PF.ZCLS + PF.PCLS
    L += ["", "All zero / partial classes (count difference, 95 % CI):", "", "| pair | " + " | ".join(cz) + " |", "|:--|" + "--:|" * len(cz)]
    for k, p in res["pairs"].items():
        x_, y = k.split(":")
        L.append(f"| {nm.get(x_, x_)} - {nm.get(y, y)} | " + " | ".join(
            f"{p['classes'][q]['diff']:+.0f} [{p['classes'][q]['lo']:+.0f}, {p['classes'][q]['hi']:+.0f}]" for q in cz) + " |")
    if res["flips"]:
        f0 = next(iter(res["flips"].values()))
        L += ["", f"Negative flips on NTV0-s0's full-score scenes. Count A (INTERP's): NTV0 new >= 0.98 ({f0['full_a']} scenes), read new < 0.98. "
              f"Count B (step 1's): NTV0 old >= 0.98 ({f0['full_b']} scenes), read new = 0.", "",
              "| read | A: negative flips | of them to 0 | points lost | B: new zeros | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2's fixes kept |", "|:--|--:|--:|--:|--:|--:|--:|"]
        for t, f in res["flips"].items():
            L.append(f"| {nm.get(t, t)} | {f['neg_a']} | {f['neg_a_zero']} | {f['neg_a_pts']:.2f} | {f['neg_b']} | {f['fixed']} | "
                     + (f"{f['keeps_r8k_fixes']} / {f['r8k_fixes']} |" if f["r8k_fixes"] is not None else "- |"))
    return L


def facts(name):
    import pandas as pd
    return pd.read_csv(F_ / f"{name}.csv").set_index("scene")


def highway(rows_src, new):
    """pcl.py highway()'s rules (behavior_stats Q1) over (label, steps / scene dir, facts name) triples."""
    import pandas as pd
    BS = DD / "runs/pai_av/behavior_stats/out"
    MP = pd.read_csv(BS / "scenes_map.csv").set_index("scene")
    hw_sc = MP.index[MP.scene_cls == "highway"]
    out = []
    for lab, src, fn in rows_src:
        f = Path(src) / f"steps_{lab}.parquet"
        if not f.exists():
            continue
        T = pd.read_parquet(f)
        T = T[(T.free_e) & (T.free_h == True) & (T.vh >= 3) & (T.t >= 2.0)]  # noqa: E712
        x = T[T.cls == "highway"]
        q = x.v / x.vh
        FA = facts(fn)
        sc = pd.read_csv(Path(src) / f"scene_{lab}.csv").set_index("scene")
        z = FA[["fclass", "score"]].join(sc[["cls_end"]], how="left")
        nw = new.get(lab)
        z["new"] = (nw.reindex(z.index) if nw is not None else np.nan)
        z["new"] = z.new.fillna(z.score)
        pts = float((1 - z.new)[z.fclass.isin(["slow", "rear-hit"]) & (z.cls_end == "highway")].sum())
        r = dict(read=lab, steps=len(x), median=float(q.median()), p10=float(q.quantile(0.1)), lt85=float((q < 0.85).mean()), gt110=float((q > 1.1).mean()),
                 dv=float((x.v - x.vh).mean()), slow_rear_pts=pts)
        r["slow"] = bool(r["median"] < C.HW_SLOW["median"] or r["lt85"] >= C.HW_SLOW["lt85"] or pts >= C.HW_SLOW["pts"])
        ps = x.assign(q=q).groupby("scene").agg(n=("q", "size"), med=("q", "median")).reindex(hw_sc)
        r["scenes_below_090"] = int((ps.med < 0.90).sum())
        r["scenes_with_steps"] = int(ps.n.notna().sum())
        out.append(r)
    return out


def hw_md(hw, nm):
    L = ["", "## Highway (behavior_stats Q1: free-road v_ego / v_human on highway stations; slow + rear-hit points by end class, new scorer)", "",
         "| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | scenes < 0.90 | reading |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in hw:
        L.append(f"| {nm.get(r['read'], r['read'])} | {r['steps']} | {r['median']:.2f} | {r['p10']:.2f} | {r['lt85']:.0%} | {r['gt110']:.0%} | {r['dv']:+.2f} | "
                 f"{r['slow_rear_pts']:.2f} | {r['scenes_below_090']} / {r['scenes_with_steps']} | {'slow on highway' if r['slow'] else 'not slow'} |")
    return L


def gates(o):
    g = {}
    for f in sorted((o / "results").glob("*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(j, dict):
            g[f.stem] = j.get("pass_", j.get("passed", j.get("bit_identical")))
    return g


def strip(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def cmd_verdict(a):
    o = Path(a.dir)
    arms = list(TAGS.values())
    labs = arms + REFS
    OLD = {t: facts(f"{t}-281ag") for t in labs}
    NEW = ref_new()
    for t in arms:
        NEW[t] = new_series(o / "sv2/new/rollouts.csv", f"{t}-ag")
    d1, d2 = TAGS["D1"], TAGS["D2"]
    pairs = [(t, y) for t in arms for y in REFS] + [(d2, d1)]
    nm = {d1: "D1", d2: "D2", P1L: "P1L", NT: "NTV0-s0", R8: "R8K-s2", IP: "INTP75-NR (ref.)"}
    res = score_block(labs, OLD, NEW, pairs, nm)
    lines = {}
    for sc in ("new", "old"):
        x = res["pairs"][f"{d1}:{P1L}"][sc]
        y = res["pairs"][f"{d2}:{d1}"][sc]
        lines[sc] = dict(d1_p1l=x, d1=("own-state DAgger helps" if x["mean"] >= LINE_HELP else "own-state DAgger hurts" if x["mean"] <= LINE_HURT else "neutral"),
                         d2_d1=y, d2=("distill_ho can go" if y["mean"] >= LINE_D2 else "distill_ho still needed in the DAgger recipe"), registered=sc == "new")
    res["lines"] = lines
    cand = {arm: res["reads"][t]["new_mean"] for arm, t in TAGS.items() if res["reads"][t]["new_mean"] >= CAND}
    best = max(cand, key=cand.get) if cand else None
    res["candidate"] = dict(line=CAND, passing=cand, chosen=best, tag=TAGS[best] if best else None)
    (o / "CAND").write_text((best or "none") + "\n")
    bsd = o / "bs"
    hw = highway([(t, bsd, f"{t}-281ag") for t in arms] + [(P1L, DD / "runs/round9/p1l/bs", f"{P1L}-281ag")]
                 + [(t, DD / "runs/pai_av/behavior_stats/out", f"{t}-281ag") for t in (NT, R8)], NEW)
    res["highway"] = hw
    res["gates"] = gates(o)
    res["ratio"] = {}
    for arm, t in TAGS.items():
        f = o / f"results/ratio_{arm}.json"
        if f.exists():
            r = json.loads(f.read_text())
            res["ratio"][arm] = dict(r500=r["ratio"], pass_=r["pass_"], after_T=C.grad_ratio(C.run_dirs(t)[-1], 10000, 19500)["ratio"])
    res["ade"] = {t: dict(at_10000=C.dev_ade(t, 10000), end=C.dev_ade(t)) for t in arms + [P1L]}
    res["cost"] = C.cost(o)
    L = ["# NTV1 step 3 verdict (PAI 281, ACC + GAP, Harmonizer OFF; P1L's serve)", ""]
    L += md_block(res, nm, "Reads")
    L += ["", "## Registered lines (new scorer registered, old reported)", ""]
    for sc, q in lines.items():
        L.append(f"- {sc} scorer: D1 - P1L {C.f3(q['d1_p1l'])} (help >= +{LINE_HELP}, hurt <= {LINE_HURT}) -> **{q['d1']}**; "
                 f"D2 - D1 {C.f3(q['d2_d1'])} (line >= {LINE_D2}) -> **{q['d2']}**" + ("" if q["registered"] else " (reported)"))
    c = res["candidate"]
    L += ["", f"Candidate line (new mean >= {CAND}, Harmonizer off): " + (f"**{c['chosen']} ({c['tag']}) {c['passing'][c['chosen']]:.3f}** -> seed 1 + "
          "Harmonizer-ON reads submitted" if c["chosen"] else "no arm passes -> no candidate tail")]
    L += hw_md(hw, nm)
    L += ["", "## Gates", ""] + [f"- {k}: {v}" for k, v in res["gates"].items()]
    for arm, r in res["ratio"].items():
        L.append(f"- S / imit gradient ratio {arm}: 500 steps after T {r['r500']:.3f} (band {C.RATIO_BAND[0]:.2f}-{C.RATIO_BAND[1]:.2f}, "
                 f"{'pass' if r['pass_'] else 'FAIL'}); whole post-T segment {r['after_T']:.3f}")
    L += ["", "| navtrain dev ADE | step 10 000 | end |", "|:--|--:|--:|"] + [
        f"| {nm.get(t, t)} | {v['at_10000'] or float('nan'):.4f} | {v['end'] or float('nan'):.4f} |" for t, v in res["ade"].items()]
    cs = res["cost"]
    L += ["", "## Cost", "", f"GPU jobs {cs['gpu_job_h']:.1f} job-hours (training {cs['train_h']:.2f}, reads {cs['read_h']:.2f}, gates / smokes {cs['gate_h']:.2f}); "
          f"wall {cs['wall_h']:.1f} h from the first start to the last end."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", strip(res))
    print("\n".join(L))


def cmd_cverdict(a):
    from jevdrive import stats
    o = Path(a.dir)
    arm = (o / "CAND").read_text().strip()
    s0, s1 = TAGS[arm], TAGS[arm].replace("-s0", "-s1")
    L = [f"# NTV1 step 3 candidate tail: {arm} seeds 0 / 1, Harmonizer OFF / ON (PAI 281 ACC + GAP)", ""]
    out = {}
    for h, hn in (("", "OFF"), ("H", "ON")):
        NEW = ref_new(h)
        for t in (s0, s1):
            src = o / "sv2/new/rollouts.csv" if (t == s0 and not h) else o / "sv2c/new/rollouts.csv"
            NEW[t] = new_series(src, f"{t}-ag{h}")
        refs = [y for y in ((P1L, NT, R8, IP) if not h else (P1L, NT, R8)) if NEW.get(y) is not None and (F_ / f"{y}-281ag{h}.csv").exists()]
        labs = [s0, s1] + refs
        OLD = {t: facts(f"{t}-281ag{h}") for t in labs}
        nm = {s0: f"{arm}-s0 {hn}", s1: f"{arm}-s1 {hn}", P1L: f"P1L {hn}", NT: f"NTV0-s0 {hn}", R8: f"R8K-s2 {hn}", IP: "INTP75-NR (ref.)"}
        pairs = [(t, y) for t in (s0, s1) for y in refs] + [(s1, s0)]
        res = score_block(labs, OLD, NEW, pairs, nm)
        S = res["_S"]
        res["seed_mean"] = dict(new=float((S["new"][s0] + S["new"][s1]).mean() / 2), old=float((S["old"][s0] + S["old"][s1]).mean() / 2))
        res["seed_mean_vs"] = {y: {sc: stats.paired((S[sc][s0] + S[sc][s1]) / 2, S[sc][y]) for sc in ("new", "old")} for y in refs}
        out[hn] = strip(res)
        L += md_block(res, nm, f"Harmonizer {hn}")
        L += ["", f"Seed mean ({arm}-s0 + s1) / 2: new {res['seed_mean']['new']:.3f}, old {res['seed_mean']['old']:.3f}; vs " + "; ".join(
            f"{nm[y]} new {C.f3(v['new'])} / old {C.f3(v['old'])}" for y, v in res["seed_mean_vs"].items()), ""]
    out["ratio_s1"] = json.loads((o / "results/ratio_cand_s1.json").read_text()) if (o / "results/ratio_cand_s1.json").exists() else None
    out["ade_s1"] = C.dev_ade(s1)
    out["cost"] = C.cost(o)
    L += ["## Seed 1 gates", "", f"- S / imit ratio 500 steps after T: {out['ratio_s1'] and out['ratio_s1']['ratio']}; dev ADE end {out['ade_s1']}",
          "", f"Lane cost so far: {out['cost']['gpu_job_h']:.1f} GPU job-hours (training {out['cost']['train_h']:.2f}, reads {out['cost']['read_h']:.2f}); "
          f"wall {out['cost']['wall_h']:.1f} h."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", out)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("argv")
    p.add_argument("arm", choices=list(TAGS)), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at"), p.add_argument("--seed", type=int)
    p.add_argument("--mix", action="store_true")
    p = s.add_parser("diff")
    p.add_argument("arm", choices=list(TAGS))
    for k in ("wcheck", "rowcheck"):
        s.add_parser(k).add_argument("--out", required=True)
    p = s.add_parser("dsw")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--on-start", type=int, required=True)
    p.add_argument("--out", required=True)
    p = s.add_parser("bs")
    p.add_argument("--out", required=True), p.add_argument("--tags", required=True), p.add_argument("--workers", type=int, default=16)
    for k in ("verdict", "cverdict"):
        p = s.add_parser(k)
        p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(argv=cmd_argv, diff=cmd_diff, wcheck=cmd_wcheck, rowcheck=cmd_rowcheck, dsw=cmd_dsw, bs=cmd_bs, verdict=cmd_verdict, cverdict=cmd_cverdict)[a.sub](a)


if __name__ == "__main__":
    main()
