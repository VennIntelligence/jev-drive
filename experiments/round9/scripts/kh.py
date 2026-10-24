#!/usr/bin/env python3
"""ROUND9 NTV1 K / H (plans/2026-10-24-ntv1-kh.md = the registration; lane scripts/kh.sh): D1 (NT3D1-F-s0) + R8K's two differences, one at a time.
  K  the round-7 pacing floor: P1L's DAgger states (on1_PAIDC) relabelled with teacher b5 (joint.py, verbatim) + F2 on the b5 labels -> pool on1_JTDCK,
     draw NTV1K; arm NTK1-F-s0 = D1's argv with only --on / --on-wfile changed
  H  the Harmonizer twin token: not run (cost estimate in the plan); twincov measures PAIHTW's overlap with D1's pool
Box, repo root. Reuses joint.py (b5 teacher, G1 label gates, pool linking: its path constants re-pointed here), dgt.py (D1's argv, score blocks),
dgc.py (pair, smoke), pcl.py (ident, ratio, post, cost).
  argv ARM TAG [--steps N] [--on-start T] [--save-at S] [--seed K] [--mix] [--pool P] [--wfile W]   ARM D1 | K; --pool / --wfile override (T3 pairs)
  diff                   K's token diff against NT3D1-F-s0's stored argv (exit 1 on anything but --tag / --on / --on-wfile)
  teach --workers N      (AlpaSim venv, DAG_TV=b3) b5 candidates for every on1_PAIDC decision -> $O/teach/<scene>.npz
  label                  joint.cmd_label on on1_JTDC / on1_PAIDC: G1 gates, $O/k/pd.npz, pools on1_JTDCK (pd fut := b5) / on1_JTDCI (identity)
  weights                NTV1K = NTV1D with the F2 pd rows at 0, pd renormalised to 0.2 -> $O/w/NTV1K.npz, results/w_NTV1K.json
  wcheck                 G-W: draw file and pools against NTV1D / on1_JTDC (exit 2 on a failure)
  twincov                PAIHTW twin rows matching D1's pool rows by name (R8K's swap rule)
  ksw --a TAG --b TAG --on-start T --out J   K's switch: smoke scalars <= T equal (reported), some post-T loss scalar differs (gate)
  verdict --out P        OFF and ON blocks, lines, ON - OFF gap vs D1's, labels, gates, cost -> P.{md,json}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dgt as T  # noqa: E402
import pcl as C  # noqa: E402

DD, REPO = C.DD, C.REPO
O = DD / "runs/round9/ntv1_kh"
S2, S3 = DD / "runs/round9/ntv1_step2", DD / "runs/round9/ntv1_step3"
CR, F_ = DD / "runs/op_parity/cache", DD / "runs/pai_fix/facts"
RESD = O / "results"
D1, K = "NT3D1-F-s0", "NTK1-F-s0"
TAGS = dict(D1=D1, K=K)
P1L, NT, R8 = T.P1L, T.NT, T.R8
BASE, SRC, POOL_K, POOL_I = "on1_JTDC", "on1_PAIDC", "on1_JTDCK", "on1_JTDCI"
NTV1D, NTV1D0, NTV1K = T.NTV1D, "runs/round9/ntv1_step2/w/NTV1D0.npz", "runs/round9/ntv1_kh/w/NTV1K.npz"
LINE = 0.015                                                   # registered (plan section 5)
GAP_D1 = -0.016                                                # D1-s0 ON - OFF (decision 471), for the text


def joint():
    sys.path.insert(0, str(REPO / "experiments/round7/scripts"))
    import joint as J
    J.O, J.O1, J.RES, J.BASE, J.POOL_K, J.POOL_I, J.SRC_D = O, S2, RESD, BASE, POOL_K, POOL_I, SRC
    return J


def build(arm, tag, steps=None, on_start=None, save_at=None, seed=None, mix=False, pool=None, wfile=None):
    a = T.build("D1", tag, steps, on_start, save_at, seed, mix)
    if arm == "K":
        C.put(a, "--on", POOL_K), C.put(a, "--on-wfile", NTV1K)
    if pool:
        C.put(a, "--on", pool)
    if wfile:
        C.put(a, "--on-wfile", wfile)
    return a


def cmd_argv(a):
    print("\n".join(build(a.arm, a.tag, a.steps, a.on_start, a.save_at, a.seed, a.mix, a.pool, a.wfile)))


def cmd_diff(a):
    b, x = C.stored_argv(D1), build("K", K)
    print(" ".join(x))
    d = {b[i - 1]: (b[i], x[i]) for i in range(len(b)) if b[i] != x[i]}
    for k, v in d.items():
        print(f"  {k}: {v[0]} -> {v[1]}")
    want = {"--tag": K, "--on": POOL_K, "--on-wfile": NTV1K}
    ok = len(x) == len(b) and {k: v[1] for k, v in d.items()} == want
    print("TOKEN DIFF AS REGISTERED" if ok else f"UNEXPECTED DIFF (want {want})")
    sys.exit(0 if ok else 1)


def cmd_teach(a):
    J = joint()
    import dag as D
    from jevdrive import par
    assert D.TEACHER_V == "b3", D.TEACHER_V
    out = O / "teach"
    out.mkdir(parents=True, exist_ok=True)
    cfg = D.scene_cfgs(str(S2 / "roll/runs/*/sim/generated-user-config-0.yaml"))
    files = sorted((S2 / "rows/PAIDC").glob("clipgt-*.npz"))
    jobs = [(f.stem, cfg[f.stem], str(f), str(out)) for f in files if f.stem in cfg]
    res = par.pmap(J._teach_one, jobs, workers=a.workers or None, desc="teach b5", skip=lambda j: (out / f"{j[0]}.npz").exists())
    res.raise_if_failed()
    v = [x for x in res.values if x]
    rec = dict(scenes=len(v), rows=sum(x["n"] for x in v), ok=sum(x["ok"] for x in v), cond=sum(x["cond"] for x in v),
               no_cfg=[f.stem for f in files if f.stem not in cfg], row_files=len(files))
    C.dump(RESD / "teach.json", rec)
    print(json.dumps(rec))
    sys.exit(0 if not rec["no_cfg"] and len(v) == len(files) else 2)


def cmd_label(a):
    joint().cmd_label(a)


def group_share(grp, w):
    from collections import Counter
    gs = Counter()
    for g_, x in zip(grp, w / w.sum()):
        gs[str(g_)] += x
    return {k: float(v) for k, v in sorted(gs.items()) if v > 0}


def cmd_weights(a):
    W = np.load(DD / NTV1D)
    names, w0, grp = W["names"].astype(str), W["w"].astype(np.float64), W["grp"].astype(str)
    tb = np.load(CR / POOL_K / "tab.npz")
    assert np.array_equal(tb["names"].astype(str), names), f"{POOL_K} rows != NTV1D's"
    part = tb["pool"].astype(str)
    j0 = np.flatnonzero(part == "pd")
    Kz = np.load(O / "k/pd.npz")
    assert np.array_equal(Kz["names"].astype(str), np.array(["on:" + x.split(":", 2)[2] for x in names[j0]])), "pd.npz rows"
    f2 = np.zeros(len(w0), bool)
    f2[j0] = Kz["f2"]
    w = w0.copy()
    m = part == "pd"
    x = w0[m] * ~f2[m]
    w[m] = x * (w0[m].sum() / x.sum())
    assert abs(w.sum() - 1) < 1e-9 and np.array_equal(w[~m], w0[~m])
    dr0, dr1 = int((w0[m] > 0).sum()), int((w[m] > 0).sum())
    fl = Kz["floored"].astype(bool)
    rec = dict(out=NTV1K, base=NTV1D, pool=POOL_K, pd_drawable_before=dr0, pd_drawable_after=dr1, f2_removed=int((f2 & (w0 > 0)).sum()),
               floored_drawable=int(fl.sum()), floored_still_drawn=int((fl & (w[j0] > 0)).sum()),
               share_s=float(w[~m].sum()), share_pd=float(w[m].sum()), group_share=group_share(grp, w))
    (O / "w").mkdir(parents=True, exist_ok=True)
    np.savez(DD / NTV1K, names=W["names"], w=w, grp=W["grp"])
    C.dump(RESD / "w_NTV1K.json", rec)
    C.dump(RESD / "w_NTV1D.json", dict(base=NTV1D, group_share=group_share(grp, w0)))   # the D1 smoke's --want
    print(json.dumps(rec, indent=1))


def cmd_wcheck(a):
    W, Wd = np.load(DD / NTV1K), np.load(DD / NTV1D)
    tb, tk, ti = (np.load(CR / p / "tab.npz") for p in (BASE, POOL_K, POOL_I))
    X = np.load(CR / SRC / "dagger.npz")
    Kz = np.load(O / "k/pd.npz")
    names, part, mode = tb["names"].astype(str), tb["pool"].astype(str), tb["mode"].astype(str)
    w, wd = W["w"].astype(np.float64), Wd["w"].astype(np.float64)
    m = part == "pd"
    j0 = np.flatnonzero(m)
    nz = np.flatnonzero(w[m] > 0)
    want = np.flatnonzero(X["draw"].astype(bool) & ~Kz["f2"].astype(bool))
    fut = tk["fut"][j0[nz]]
    fa, fb = tk["fut"][j0], tb["fut"][j0]
    dk = np.flatnonzero((~((fa == fb) | (np.isnan(fa) & np.isnan(fb)))).reshape(len(j0), -1).any(1))
    fl = np.flatnonzero(Kz["floored"])
    same = lambda x, y: bool(x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))  # noqa: E731
    ck = dict(names_equal=bool(np.array_equal(W["names"].astype(str), names) and np.array_equal(Wd["names"].astype(str), names)),
              s_weights_bit_equal=bool(np.array_equal(w[~m], wd[~m])), grp_equal=bool(np.array_equal(W["grp"], Wd["grp"])),
              pd_nonzero=int(len(nz)), pd_nonzero_is_drawable_minus_f2=bool(np.array_equal(nz, want)), pd_uniform=bool(np.unique(w[m][nz]).size == 1),
              pd_modes=sorted(set(mode[j0[nz]])), pd_fut_finite=bool(np.isfinite(fut).all()),
              share_s=float(w[~m].sum() / w.sum()), share_pd=float(w[m].sum() / w.sum()),
              k_tab_other_keys_equal=sorted(k for k in tb.files if k != "fut" and not same(tk[k], tb[k])) == [],
              k_fut_prefix_equal=bool(np.array_equal(tk["fut"][~m], tb["fut"][~m], equal_nan=True)),
              k_fut_changed_rows=int(len(dk)), k_fut_changed_is_floored=bool(np.array_equal(dk, fl)),
              i_tab_equal=all(same(ti[k], tb[k]) for k in tb.files) and set(ti.files) == set(tb.files))
    ok = (ck["names_equal"] and ck["s_weights_bit_equal"] and ck["grp_equal"] and ck["pd_nonzero_is_drawable_minus_f2"] and ck["pd_uniform"]
          and ck["pd_modes"] == ["both"] and ck["pd_fut_finite"] and abs(ck["share_s"] - 0.8) < 1e-9 and abs(ck["share_pd"] - 0.2) < 1e-9
          and ck["k_tab_other_keys_equal"] and ck["k_fut_prefix_equal"] and ck["k_fut_changed_is_floored"] and ck["i_tab_equal"])
    rec = dict(file=NTV1K, checks=ck, pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def cmd_twincov(a):
    sys.path.insert(0, str(REPO / "lib"))
    import twin_tokens as TT
    tw = set(TT.strip(np.load(CR / "on1_PAIHTW/tab.npz")["names"].astype(str)))
    tb = np.load(CR / BASE / "tab.npz")
    nm = TT.strip(tb["names"].astype(str), tb["pool"].astype(str))
    part = tb["pool"].astype(str)
    hit = np.array([x in tw for x in nm])
    W = np.load(DD / NTV1D)["w"]
    rec = dict(store="on1_PAIHTW", store_rows=len(tw), pool=BASE, pool_rows=len(nm), matched=int(hit.sum()),
               matched_by_part={p: int((hit & (part == p)).sum()) for p in sorted(set(part))}, matched_draw_mass=float(W[hit].sum()),
               store_prefixes=sorted({x.split(":")[0] for x in tw}), pool_prefixes=sorted({x.split(":")[0] for x in nm}),
               note="R8K's --twin swaps only drawn pool rows whose name has a twin; D1's pool has S rows (rb0 ...) and P1L's DAgger rows (paidc), "
                    "PAIHTW holds PAI human rows (paih / paih05 / paih15)")
    C.dump(RESD / "twin_cover.json", rec)
    print(json.dumps(rec, indent=1))


def cmd_ksw(a):
    da, db = C.run_dirs(a.a)[-1], C.run_dirs(a.b)[-1]
    pre = C.compare(db, da, a.on_start, False)
    x, y = C.scalars(da), C.scalars(db)
    post = sorted(f"{k[0]}@{k[1]}" for k in set(x) & set(y) if k[1] > a.on_start and k[0].startswith("loss/") and x[k] != y[k])
    ok = bool(post)
    rec = dict(a=a.a, b=a.b, pre_T=dict(bit_identical=pre["bit_identical"], compared=pre["scalars_compared"], identical=pre["identical"],
                                         note="reported, not a gate"),
               post_T_loss_scalars_differing=len(post), first_differing=post[:10], pass_=ok)
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


# ---------------------------------------------------------------- verdict
def line(x):
    return "K helps" if x >= LINE else "K hurts" if x <= -LINE else "neutral"


def cmd_verdict(a):
    from jevdrive import stats
    blocks, L = {}, ["# NTV1 K / H verdict (PAI 281, ACC + GAP; D1's serve)", ""]
    nm = {K: "D1-K", D1: "D1-s0", P1L: "P1L", NT: "NTV0-s0", R8: "R8K-s2"}
    for h, hn in (("", "OFF"), ("H", "ON")):
        NEW = T.ref_new(h)
        NEW.pop(T.IP, None)
        NEW[K] = T.new_series(O / "sv2/new/rollouts.csv", f"{K}-ag{h}")
        NEW[D1] = T.new_series(S3 / ("sv2c" if h else "sv2") / "new/rollouts.csv", f"{D1}-ag{h}")
        labs = [t for t in (K, D1, P1L, NT, R8) if NEW.get(t) is not None and (F_ / f"{t}-281ag{h}.csv").exists()]
        OLD = {t: T.facts(f"{t}-281ag{h}") for t in labs}
        pairs = [(K, y) for y in labs if y != K]
        res = T.score_block(labs, OLD, NEW, pairs, nm)
        blocks[hn] = res
        L += T.md_block(res, {t: f"{v} {hn}" for t, v in nm.items()}, f"Harmonizer {hn}")
        L.append("")
    lines = {}
    for hn, res in blocks.items():
        for sc in ("new", "old"):
            p = res["pairs"][f"{K}:{D1}"][sc]
            lines[f"Harmonizer {hn}, {sc} scorer"] = dict(diff=p, verdict=line(p["mean"]), registered=(hn == "OFF" and sc == "new"))
    # ON - OFF gap: per scene (K on - off) - (D1 on - off) on both blocks' common scenes
    ca, cb = blocks["OFF"]["_common"], blocks["ON"]["_common"]
    cc = sorted(set(ca) & set(cb))
    ia, ib = [ca.index(s) for s in cc], [cb.index(s) for s in cc]
    gap = {}
    for sc in ("new", "old"):
        so, sn = blocks["OFF"]["_S"][sc], blocks["ON"]["_S"][sc]
        gk, gd = sn[K][ib] - so[K][ia], sn[D1][ib] - so[D1][ia]
        gr = {t: float((sn[t][ib] - so[t][ia]).mean()) for t in sn if t in so}
        gap[sc] = dict(scenes=len(cc), k=float(gk.mean()), d1=float(gd.mean()), per_read=gr, k_minus_d1=stats.paired(gk, gd))
    lab = json.loads((RESD / "rows.json").read_text())
    wk = json.loads((RESD / "w_NTV1K.json").read_text())
    tc = json.loads((RESD / "twin_cover.json").read_text())
    rt = json.loads((RESD / "ratio_K.json").read_text()) if (RESD / "ratio_K.json").exists() else None
    out = dict(blocks={k: T.strip(v) for k, v in blocks.items()}, lines=lines, gap=gap, labels=lab, weights=wk, twin_cover=tc, ratio=rt,
               ade={t: dict(at_10000=C.dev_ade(t, 10000), end=C.dev_ade(t)) for t in (K, D1)}, gates=T.gates(O), cost=C.cost(O))
    L += ["## Registered lines (D1-K - D1-s0; new scorer OFF registered, the rest reported)", ""]
    for k, q in lines.items():
        L.append(f"- {k}: {C.f3(q['diff'])} -> **{q['verdict']}**"
                 + (" (registered)" if q["registered"] else ""))
    L += ["", f"## ON - OFF gap ({gap['new']['scenes']} scenes common to both blocks)", "",
          "| scorer | D1-K | D1-s0 | D1-K gap - D1-s0 gap | other reads |", "|:--|--:|--:|--:|:--|"]
    for sc, g in gap.items():
        L.append(f"| {sc} | {g['k']:+.3f} | {g['d1']:+.3f} | {C.f3(g['k_minus_d1'])} | "
                 + ", ".join(f"{nm.get(t, t)} {v:+.3f}" for t, v in g["per_read"].items() if t not in (K, D1)) + " |")
    L += [f"(D1-s0's gap on its own common set, decision 471: {GAP_D1:+.3f}.)", "", "## What K changed in the labels", "",
          f"- Drawable DAgger rows {lab['reasons']['drawable']}; floored (b5 faster than b3) {lab['reasons']['floored']} ({lab['share_floored']:.1%}); "
          f"4 s arc increase p10 / p50 / p90 {lab['arc4_increase_q']} m; straight rows floored {lab['straight']['floored']} / {lab['straight']['drawable']} "
          f"(increase {lab['straight']['inc_q']} m).",
          f"- Not floored because: no human window {lab['reasons']['no_window']}, human slows {lab['reasons']['human_slows']}, clock lag "
          f"{lab['reasons']['lag']}, student not slower {lab['reasons']['not_slower']}, lead {lab['reasons']['lead']}, T4 revert {lab['reasons']['t4_revert']}.",
          f"- By class (floored / drawable) {json.dumps(lab['by_class'])}; by speed {json.dumps(lab['by_speed'])}.",
          f"- F2 on the b5 labels: {wk['f2_removed']} drawable rows to weight 0 (on b3 labels the rule would take {lab['f2_pd_b3']}); drawable "
          f"{wk['pd_drawable_before']} -> {wk['pd_drawable_after']}; floored rows still drawn {wk['floored_still_drawn']}.",
          f"- G1: {json.dumps(lab['gates'])}; G1a max |diff| {lab['G1a']['max_abs_m']:.2e} m over {lab['G1a']['coverage']:.1%}; G1b {json.dumps(lab['G1b'])}; "
          f"G1c median |lag| {lab['G1c']['median_abs_lag_s']:.3f} s; T4 p99 {lab['T4']['p99_b5']:.2f} (b3 {lab['T4']['p99_b3']:.2f}).", "",
          "## H (not run)", "", f"PAIHTW rows matching D1's pool by name: {tc['matched']} of {tc['pool_rows']} (by part {json.dumps(tc['matched_by_part'])}); "
          f"store prefixes {tc['store_prefixes']}, pool prefixes {tc['pool_prefixes']}. Cost estimate: see the plan, section 2.", "", "## Gates", ""]
    L += [f"- {k}: {v}" for k, v in out["gates"].items()]
    if rt:
        L.append(f"- S / imit gradient ratio, 500 steps after T: {rt['ratio']:.3f} (band {C.RATIO_BAND[0]:.2f}-{C.RATIO_BAND[1]:.2f}); whole post-T segment "
                 f"{C.grad_ratio(C.run_dirs(K)[-1], 10000, 19500)['ratio']:.3f}")
    L += ["", "| navtrain dev ADE | step 10 000 | end |", "|:--|--:|--:|"] + [
        f"| {nm[t]} | {v['at_10000'] or float('nan'):.4f} | {v['end'] or float('nan'):.4f} |" for t, v in out["ade"].items()]
    cs = out["cost"]
    L += ["", "## Cost", "", f"GPU jobs {cs['gpu_job_h']:.1f} job-hours (training {cs['train_h']:.2f}, reads {cs['read_h']:.2f}, gates / smokes {cs['gate_h']:.2f}); "
          f"wall {cs['wall_h']:.1f} h."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", out)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("argv")
    p.add_argument("arm", choices=list(TAGS)), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at"), p.add_argument("--seed", type=int)
    p.add_argument("--mix", action="store_true"), p.add_argument("--pool"), p.add_argument("--wfile")
    s.add_parser("diff")
    s.add_parser("teach").add_argument("--workers", type=int, default=0)
    for k in ("label", "weights", "twincov"):
        s.add_parser(k)
    s.add_parser("wcheck").add_argument("--out", required=True)
    p = s.add_parser("ksw")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--on-start", type=int, required=True)
    p.add_argument("--out", required=True)
    s.add_parser("verdict").add_argument("--out", required=True)
    a = ap.parse_args()
    dict(argv=cmd_argv, diff=cmd_diff, teach=cmd_teach, label=cmd_label, weights=cmd_weights, wcheck=cmd_wcheck, twincov=cmd_twincov, ksw=cmd_ksw,
         verdict=cmd_verdict)[a.sub](a)


if __name__ == "__main__":
    main()
