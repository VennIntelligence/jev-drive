#!/usr/bin/env python3
"""ROUND2 (plans/2026-10-20-round2-prereg.md): the merged pool's fix-up and checks, the draw files, the step rule, the smoke check and the
summary. Box, repo root, op-train env (pool jobs of scripts/r2.sh).

  r2.py fix --pool on1_JTPH --ref on1_JT --parts ph,ph05,ph15     after rf_pait merge: the PAI human rows become pure imitation rows (agent.npz
        ok False on the listed parts; road ok is already False, the lateral raster covers on1_JT's prefix only); the merged pool's first rows
        must equal --ref (names, ego, fut, cam, mode, route features, teacher, a front sample) -> $O/fix_<pool>.json; exit 2 on a mismatch
  r2.py weights --pool on1_JTPH --base JT00S --out NAME [--half SPLIT] [--parts ph:0.25,ph05:0.125,ph15:0.125] [--s-share 0.5]
        per-row draw weights: the --base rows keep their weights (scaled to --s-share of the draw), each PAI part uniform over its drawable rows
        (scene in physical_ai_av/pai-rows@v1, not held: crc32(log) % 100 < 15, in --half if given) at its share; every other row 0
        -> $DATA_DIR/runs/defect_fix/w/NAME.npz (names, w, grp) + results/w_NAME.json
  r2.py steps --full R2JF --half R2JA --k 32                     the registered step rule (prereg "步数"): N = R / k rounded to 100 for J (all
        drawable PAI human rows) and J-half (half A) -> $O/steps.json + results/steps.{json,md} (the passes-per-row table)
  r2.py smoke --tag T --wfile NAME --k-on K --nn N               the 60-step smoke's registered checks -> results/smoke.json; exit 2 on a failure
  r2.py costs --tags A B ..                                      train wall time / steps per arm -> results/costs.md
  r2.py r281 --read K=FACTS.csv .. --pairs A:B .. --out F.md     stage 2: full PAI 281 reads (pai_fix.py facts rows) on the scenes every read
        scored without a driver RPC error: mean, zeros by class, partials, launch counts per read; per pair A - B the paired mean difference
        and the paired difference of every class count (jevdrive.stats.paired on the per-scene indicator, x n), new / fixed zeros
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO)]
from jevdrive.tokens import open_front  # noqa: E402
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, FEAT, W = DATA / "runs/op_parity/cache", DATA / "runs/route_mem/feat", DATA / "runs/defect_fix/w"
O, RES = DATA / "runs/round2", REPO / "experiments/round2/results"
HELD_PCT = 15                                                   # rm_train --on-hold-pct 15 (crc32(log) % 100 < P), applied here to the PAI rows


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def cmd_fix(a):
    P, R = CR / a.pool, CR / a.ref
    tp, tr = np.load(P / "tab.npz"), np.load(R / "tab.npz")
    parts = a.parts.split(",")
    part = tp["pool"].astype(str)
    n = int(np.argmax(np.isin(part, parts)))                    # the leading non-PAI rows: all of --ref (or its prefix with --prefix-only)
    assert n > 0 and np.isin(part[n:], parts).all(), "the PAI parts must follow every other part"
    assert a.prefix_only or n == len(tr["names"]), f"{n} leading rows != {a.ref}'s {len(tr['names'])}"
    tr = {k: tr[k][:n] for k in tr.files}
    bad = [k for k in ("names", "ego", "fut", "cam", "lht", "log", "mode", "speed") if k in tr and not np.array_equal(tp[k][:n], tr[k])]
    fp, fr = np.load(FEAT / f"{a.pool}.npz"), np.load(FEAT / f"{a.ref}.npz")
    bad += [f"feat.{k}" for k in ("names", "f", "k", "ok") if not np.array_equal(fp[k][:n], fr[k][:n])]
    zp, zr = np.load(CR / f"{a.pool}@warp/teacher.npz"), np.load(CR / f"{a.ref}@warp/teacher.npz")
    bad += [f"teacher.{k}" for k in ("out", "plan") if not np.array_equal(zp[k][:n], zr[k][:n])]
    i = np.sort(np.random.default_rng(0).choice(n, min(2000, n), replace=False))
    Fp, Fr = open_front(CR / f"{a.pool}@warp/front.npy"), open_front(CR / f"{a.ref}@warp/front.npy")
    if not np.array_equal(np.asarray(Fp[i]), np.asarray(Fr[i])):
        bad.append("front sample")
    ag = dict(np.load(P / "agent.npz"))
    assert np.array_equal(ag["tokens"], tp["names"]), "agent.npz rows"
    m = np.isin(part, parts)
    rec = dict(pool=a.pool, ref=a.ref, ref_rows=n, rows=len(part), parts={p: int((part == p).sum()) for p in np.unique(part)},
               agent_ok_before={p: int(ag["ok"][part == p].sum()) for p in parts}, mismatch=bad)
    ag["ok"] = np.where(m, False, ag["ok"])
    np.savez(P / "agent.npz", **ag)
    rd = np.load(P / "road.npz")
    rec |= dict(agent_ok_after={p: int(ag["ok"][part == p].sum()) for p in parts}, road_ok_pai=int(rd["ok"][m].sum()),
                mode_pai=sorted(set(tp["mode"][m].astype(str).tolist())), prefix_identical=not bad)
    dump(O / f"fix_{a.pool}.json", rec)
    print(json.dumps(rec))
    if bad or rec["road_ok_pai"]:
        raise SystemExit(2)


def cmd_weights(a):
    from jevdrive.data import splits
    tab = np.load(CR / a.pool / "tab.npz")
    names, part, log = tab["names"].astype(str), tab["pool"].astype(str), tab["log"].astype(str)
    B = np.load(W / f"{a.base}.npz")
    pai = [x.split(":")[0] for x in a.parts.split(",")]
    nb = int(np.argmax(np.isin(part, pai)))                     # the leading non-PAI rows: --base's rows (all of them, or a prefix for the smoke pool)
    assert nb > 0 and np.isin(part[nb:], pai).all(), "the PAI parts must follow every other part"
    assert np.array_equal(B["names"][:nb].astype(str), names[:nb]), f"{a.base}: rows != the pool's first rows"
    assert a.prefix_only or nb == len(B["names"]), f"{nb} leading rows != {a.base}'s {len(B['names'])}"
    B = {k: B[k][:nb] for k in ("w", "grp")}
    w, grp = np.zeros(len(names)), np.full(len(names), "other", dtype="U12")
    w[:nb] = B["w"] / B["w"].sum() * a.s_share
    grp[:nb] = B["grp"].astype(str)
    allow = splits.load("physical_ai_av/pai-rows@v1")
    half = splits.load(a.half) if a.half else None
    held = np.array([zlib.crc32(s.encode()) % 100 < HELD_PCT for s in log])
    tsv = {l.split("\t")[0]: l.split("\t")[4] == "True" for l in (REPO / "experiments/pai_rows/results/scenes.tsv").read_text().splitlines()[1:]}
    rec = dict(pool=a.pool, base=a.base, s_share=a.s_share, half=half.id if half else None, allow=allow.id, held_pct=HELD_PCT, parts={})
    tot = a.s_share
    for x in a.parts.split(","):
        p, sh = x.split(":")
        sh = float(sh)
        m = part == p
        assert m.any(), f"part {p}: no rows"
        sc = log[m]
        assert allow.mask(sc).all(), f"part {p}: scenes outside pai-rows@v1"
        assert all(tsv[s] == h for s, h in zip(sc, held[m])), f"part {p}: crc32 held != scenes.tsv held"
        ok = m & ~held & (half.mask(log) if half else True)
        w[ok] = sh / ok.sum()
        grp[m] = p
        grp[m & held] = f"{p}_held"
        if half:
            grp[m & ~held & ~half.mask(log)] = f"{p}_halfB"
        rec["parts"][p] = dict(share=sh, rows=int(m.sum()), scenes=int(len(set(sc))), held_rows=int((m & held).sum()), drawable_rows=int(ok.sum()),
                               drawable_scenes=int(len(set(log[ok]))))
        tot += sh
    assert abs(tot - 1) < 1e-9 and abs(w.sum() - 1) < 1e-9, (tot, w.sum())
    rec |= dict(k_on=a.k_on, expected_per_batch={g: float(w[grp == g].sum() * a.k_on) for g in sorted(set(grp)) if w[grp == g].sum() > 0},
                pai_drawable_rows=int(sum(v["drawable_rows"] for v in rec["parts"].values())))
    np.savez(W / f"{a.out}.npz", names=names, w=w, grp=grp)
    dump(RES / f"w_{a.out}.json", rec)
    print(json.dumps(rec))


def cmd_steps(a):
    F, H = (json.loads((RES / f"w_{x}.json").read_text()) for x in (a.full, a.half))
    r100 = lambda x: int(round(x / 100.0)) * 100  # noqa: E731
    out = dict(rule="N = round(R / k, nearest 100); R = drawable PAI human rows (plain + both stretches) of the arm, k = PAI human rows per batch "
               "(expected); CTRL takes J's N", k=a.k)
    for arm, w in (("J", F), ("JH", H)):
        R = w["pai_drawable_rows"]
        N = r100(R / a.k)
        out[arm] = dict(steps=N, R=R, passes_all=N * a.k / R,
                        parts={p: dict(R=v["drawable_rows"], k=v["share"] * w["k_on"], passes=N * v["share"] * w["k_on"] / v["drawable_rows"])
                               for p, v in w["parts"].items()})
    out["CTRL"] = dict(steps=out["J"]["steps"])
    dump(O / "steps.json", out)
    dump(RES / "steps.json", out)
    L = ["# ROUND2 steps and passes per row (registered rule, prereg 步数)", "", f"Rule: {out['rule']}.", "",
         "| arm | steps | PAI rows R (drawable) | passes (all PAI rows) | plain R / k / passes | f0.5 R / k / passes | f1.5 R / k / passes |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for arm in ("J", "JH"):
        x = out[arm]
        L.append(f"| {arm} | {x['steps']} | {x['R']} | {x['passes_all']:.2f} | " + " | ".join(
            f"{v['R']} / {v['k']:.0f} / {v['passes']:.2f}" for v in x["parts"].values()) + " |")
    L.append(f"| CTRL | {out['CTRL']['steps']} | 0 | | | | |")
    (RES / "steps.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def run_dir(tag):
    d = sorted(glob.glob(str(DATA / f"runs/route_mem/train-{tag}/*")))
    assert d, f"no run dir for {tag}"
    return Path(d[-1])


def cmd_smoke(a):
    d = run_dir(a.tag)
    S = {}
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            S.setdefault(e["tag"], []).append(e["value"])
    done = json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else {}
    fl = done.get("flat_brk", {})
    share = fl.get("train_rows", 0) / max(fl.get("train_rows_total", 1), 1)
    exp_flat = a.flat_p * a.nn * 0.75 * share                  # non-anchor normal rows (pp_train d_frac 0.25) x listed share x P
    wj = json.loads((RES / f"w_{a.wfile}.json").read_text())
    exp_imit = sum(v for g, v in wj["expected_per_batch"].items() if g in wj["parts"])
    mean = lambda k: float(np.mean(S[k])) if k in S else float("nan")  # noqa: E731
    ck = {
        "finite": all(np.isfinite(v).all() for v in S.values()),
        "flat_rows": (mean("loss/flat/rows"), exp_flat, 0.6 * exp_flat <= mean("loss/flat/rows") <= 1.4 * exp_flat),
        "pool_imit_rows": (mean("loss/on/n_imit"), exp_imit, 0.75 * exp_imit <= mean("loss/on/n_imit") <= 1.25 * exp_imit),
        "pool_imit_rows_without_hinge": (max(S.get("loss/on/n_hinge", [np.nan])), 0, max(S.get("loss/on/n_hinge", [1])) == 0),
        "agent_ho_zero": (max(S.get("loss/on/agent_ho", [0])), 0, max(S.get("loss/on/agent_ho", [0])) == 0),
        "flat_p_recorded": (fl.get("p"), a.flat_p, fl.get("p") == a.flat_p),
        "group_share": (done.get("draw_shares", {}).get("group_share"), None, True),
        "dev_ade": (S.get("dev/ade", [None])[-1], None, True),
    }
    ok = bool(ck["finite"] and all(v[2] for k, v in ck.items() if k != "finite"))
    res = dict(tag=a.tag, run=str(d), pass_=ok, checks=ck, steps=len(S.get("loss/total", [])))
    dump(RES / "smoke.json", res)
    print(json.dumps(res, indent=1, default=float))
    if not ok:
        raise SystemExit(2)


def cmd_costs(a):
    L = ["# ROUND2 training cost", "", "| tag | steps | wall (min) | steps / s | host |", "|:--|--:|--:|--:|:--|"]
    out = {}
    for t in a.tags:
        d = run_dir(t)
        dn, m = json.loads((d / "DONE").read_text()), json.loads((d / "meta.json").read_text())
        st = m["config"]["steps"]
        out[t] = dict(steps=st, wall_s=dn["wall_s"], run=str(d), card_h=dn["wall_s"] / 3600)
        L.append(f"| {t} | {st} | {dn['wall_s'] / 60:.1f} | {st / dn['wall_s']:.2f} | {m.get('host', '')} |")
    dump(RES / "costs.json", out)
    (RES / "costs.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_r281(a):
    import pandas as pd
    sys.path.insert(0, str(REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    from jevdrive import stats
    R = {}
    for x in a.read:
        k, f = x.split("=", 1)
        R[k] = pd.read_csv(f).set_index("scene")
    common = sorted(set.intersection(*[set(F.index[F.score.notna() & ~F.rpc.fillna(False).astype(bool)]) for F in R.values()]))
    R = {k: F.reindex(common) for k, F in R.items()}
    cols = ["n", "mean", "zeros", *[f"z_{k}" for k in PF.ZCLS], *[f"p_{k}" for k in PF.PCLS], "launch"]
    L = [f"# {a.title}", "", f"Scenes every read scored without a driver RPC error: {len(common)} (of {max(len(pd.read_csv(x.split('=', 1)[1])) for x in a.read)} per read). "
         "Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow "
         "(creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).", "",
         "| read | " + " | ".join(cols) + " |", "|---|" + "--:|" * len(cols)]
    out = dict(scenes=len(common), reads={}, pairs={})
    for k, F in R.items():
        m = PF.metrics(F)
        out["reads"][k] = m
        L.append(f"| {k} | " + " | ".join(f"{m[c]:.3f}" if c == "mean" else str(m[c]) for c in cols) + " |")
    L.append("")
    n = len(common)
    for pr in a.pairs:
        A, B = pr.split(":")
        u, v = R[A].score.to_numpy(float), R[B].score.to_numpy(float)
        d = dict(mean=stats.paired(u, v))
        L += [f"## {A} - {B}", "", f"- mean {stats.fmt(d['mean'], '+.3f')}; zeros {int((u == 0).sum()) - int((v == 0).sum()):+d} "
              f"(new {int(((u == 0) & (v > 0)).sum())}, fixed {int(((u > 0) & (v == 0)).sum())})", "",
              "| class | A | B | A - B (count, 95 % CI) | new | fixed |", "|:--|--:|--:|--:|--:|--:|"]
        cl = [("zeros", R[A].score == 0, R[B].score == 0)] + [(c, R[A].fclass == c, R[B].fclass == c) for c in PF.ZCLS + PF.PCLS]
        la, lb = PF.launch_mask(R[A]), PF.launch_mask(R[B])
        cl += [("launched (of launch scenes)", la & PF.launched(R[A]), lb & PF.launched(R[B]))]
        for c, x, y in cl:
            x, y = x.to_numpy(float), y.to_numpy(float)
            q = stats.paired(x, y)
            d[c] = dict(a=int(x.sum()), b=int(y.sum()), diff=n * q["mean"], lo=n * q["lo"], hi=n * q["hi"], new=int(((x == 1) & (y == 0)).sum()),
                        fixed=int(((x == 0) & (y == 1)).sum()))
            L.append(f"| {c} | {d[c]['a']} | {d[c]['b']} | {d[c]['diff']:+.0f} [{d[c]['lo']:+.0f}, {d[c]['hi']:+.0f}] | {d[c]['new']} | {d[c]['fixed']} |")
        L.append("")
        out["pairs"][pr] = d
    L.append("Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).")
    Path(a.out).write_text("\n".join(L) + "\n")
    dump(Path(a.out).with_suffix(".json"), out)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("fix")
    q.add_argument("--pool", required=True), q.add_argument("--ref", default="on1_JT"), q.add_argument("--parts", default="ph,ph05,ph15")
    q.add_argument("--prefix-only", action="store_true", help="the smoke pool: the non-PAI rows are a prefix of --ref")
    q = sp.add_parser("weights")
    q.add_argument("--pool", required=True), q.add_argument("--base", default="JT00S"), q.add_argument("--out", required=True)
    q.add_argument("--half", default=""), q.add_argument("--parts", default="ph:0.25,ph05:0.125,ph15:0.125"), q.add_argument("--s-share", type=float, default=0.5)
    q.add_argument("--k-on", type=int, default=64), q.add_argument("--prefix-only", action="store_true")
    q = sp.add_parser("steps")
    q.add_argument("--full", required=True), q.add_argument("--half", required=True), q.add_argument("--k", type=int, default=32)
    q = sp.add_parser("smoke")
    q.add_argument("--tag", required=True), q.add_argument("--wfile", required=True), q.add_argument("--nn", type=int, default=51)
    q.add_argument("--flat-p", type=float, default=0.5)
    q = sp.add_parser("r281")
    q.add_argument("--read", nargs="+", required=True), q.add_argument("--pairs", nargs="+", required=True), q.add_argument("--out", required=True)
    q.add_argument("--title", default="PAI 281 paired read")
    q = sp.add_parser("costs")
    q.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    dict(fix=cmd_fix, weights=cmd_weights, steps=cmd_steps, smoke=cmd_smoke, costs=cmd_costs, r281=cmd_r281)[a.cmd](a)
