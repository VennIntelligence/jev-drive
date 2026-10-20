#!/usr/bin/env python3
"""ROUND5 v1 recipe (plans/2026-10-21-r5-prereg.md): R5-R = R3MV-L18-s0's command + the PAI road hinge (--on-road-pai, rule d, W 1); R5-A = R5-R
+ S3 outcome rows as extra rows a batch. Smoke checks, the R5-A draw file and the registered gate; round 3's gate code is reused (r3.py).
Box, repo root (op-train env for the pool / run files, .venv for the read stores; both have numpy / pandas).

  r5.py expect --pool on1_JTPH --wfile W --index F    expected share of the drawn PAI human rows that carry the road hinge (index ok, draw-weighted)
  r5.py smoke --tag T --arm R|A [--wfile W --index F --k-s3 9]   the 60-step smoke's checks: finite scalars; the PAI human rows (+ S3 rows for A)
        0.75-1.25 x their expected count; road hinge rows / imitation rows within 0.05 of the expected share; the hinge fires (road_pos > 0);
        the trainer's road_pai_rows = the index's ok rows of the pool, road_pai_w 1; no flattening; the drawn part shares -> results/smoke_<arm>.json
  r5.py weights --base R3MV --pool on1_JTPHO --s3 on1_PAIO --k-on 64 --k-s3 9 --out R5A   R5-A's draw file: R3MV's weights x 64 / 73 on the
        merged pool's on1_JTPH prefix (bit-equal names), the S3 rows 9 / 73 split by outcome type (partial-fixed 1/2, zero-rescued 1/4, A-wins
        1/4; uniform inside a type) -> $DATA_DIR/runs/round5/r5/w/<out>.npz + results/w_<out>.json
  r5.py gate --arms TAG.. --ctrl R2C-L18-s0 --stem gate_r5r   round 3's three lines (r3.py gate: G-LAUNCH >= CTRL - 0.05 binding, T281 route-turn
        zeros <= CTRL - 5, TARGET >= CTRL - 0.02) + the class guard (no zero class over TARGET + G-LAUNCH + G-LEAD more than CTRL + 3)
        -> results/<stem>.{json,md}; prints the tags that pass all four
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments/round3/scripts")]
import r3  # noqa: E402  (round 3's gate, run dirs, shares)

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, W3, W5 = DATA / "runs/op_parity/cache", DATA / "runs/defect_fix/w", DATA / "runs/round5/r5/w"
RP = DATA / "runs/round5/road_pai"
RES = REPO / "experiments/round5/results"
PAI = ("ph", "ph05", "ph15")
S3_SHARE = {"partial-fixed": 0.5, "zero-rescued": 0.25, "A-wins": 0.25}   # the framework's v1 recipe
GUARD_SETS = ["STOP", "STOP-L", "TURN", "G-LAUNCH", "G-LEAD"]             # TARGET + guards (T281 has its own line)
GUARD_TOL = 3


def wpath(name):
    p = W5 / f"{name}.npz"
    return p if p.exists() else W3 / f"{name}.npz"


def expected_share(pool, wfile, index):
    """Draw-weighted share of the PAI human rows (grp ph / ph05 / ph15) whose road raster is ok in the index (rule d shipped)."""
    T = np.load(CR / pool / "tab.npz")
    names = T["names"].astype(str)
    Wz = np.load(wpath(wfile))
    assert np.array_equal(Wz["names"].astype(str), names), f"{wfile} rows != {pool}"
    I = np.load(index if str(index).startswith("/") else RP / index)
    ix = {n: i for i, n in enumerate(names)}
    pos = np.array([ix[t] for t in I["tokens"].astype(str)])
    ok = np.zeros(len(names), bool)
    ok[pos] = I["ok"]
    w, g = Wz["w"].astype(np.float64), Wz["grp"].astype(str)
    pai = np.isin(g, PAI)
    return dict(pool=pool, wfile=wfile, index=str(index), share=float((w[pai] * ok[pai]).sum() / w[pai].sum()),
                pai_mass=float(w[pai].sum() / w.sum()), index_ok_rows=int(I["ok"].sum()),
                part={p: float((w[g == p] * ok[g == p]).sum() / max(w[g == p].sum(), 1e-12)) for p in PAI})


def cmd_expect(a):
    print(json.dumps(expected_share(a.pool, a.wfile, a.index), indent=1))


def cmd_smoke(a):
    d = r3.run_dir(a.tag)
    S = defaultdict(list)
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            S[e["tag"]].append(e["value"])
    done = json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else {}
    ds = done.get("draw_shares") or {}
    m = re.search(r'"road_pai_rows": (\d+), "road_pai_w": ([0-9.]+)', (d / "log.txt").read_text())
    mean = lambda k: float(np.mean(S[k])) if k in S else float("nan")  # noqa: E731
    ck = dict(done=(bool(done), True, bool(done)), finite=(None, None, all(np.isfinite(v).all() for v in S.values())),
              no_flat=(mean("loss/flat/rows"), 0, np.nan_to_num(mean("loss/flat/rows")) == 0 and not done.get("flat_brk")),
              wfile=(ds.get("on_wfile"), a.wfile, f"/{a.wfile}.npz" in str(ds.get("on_wfile"))))
    if a.arm == "R":                                               # R5-R: the PAI road hinge
        ex = expected_share("on1_JTPH", a.wfile, a.index)
        k_both = round(r3.K_ON * ex["pai_mass"])
        sh = mean("loss/on/n_hinge") / max(mean("loss/on/n_imit"), 1e-9)
        ck |= dict(road_hinge_share=(sh, ex["share"], abs(sh - ex["share"]) <= 0.05),
                   road_fires=(mean("loss/on/road_pos"), "> 0", mean("loss/on/road_pos") > 0 and mean("loss/on/road_ho") > 0),
                   road_pai_rows=(int(m.group(1)) if m else None, ex["index_ok_rows"], bool(m) and int(m.group(1)) == ex["index_ok_rows"]),
                   road_pai_w=(float(m.group(2)) if m else None, 1.0, bool(m) and float(m.group(2)) == 1.0))
    else:                                                          # R5-A (addendum 1): R3MV's argv + S3 rows, no road hinge
        ex = json.loads((RES / f"w_{a.wfile}.json").read_text())
        k_both = 32 + a.k_s3                                       # the 32 PAI human rows (64 x 0.5) + the S3 rows, all imitation rows (mode both)
        ck |= dict(no_road_pai=(m is not None, False, m is None),
                   k_on=(ex.get("k_all"), r3.K_ON + a.k_s3, ex.get("k_all") == r3.K_ON + a.k_s3))
        b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", (d / "log.txt").read_text())
        ck["batch_rows"] = (b.groups() if b else None, ("51", "13", str(r3.K_ON + a.k_s3)), bool(b) and b.groups() == ("51", "13", str(r3.K_ON + a.k_s3)))
    ck["pool_imit_rows"] = (mean("loss/on/n_imit"), k_both, 0.75 * k_both <= mean("loss/on/n_imit") <= 1.25 * k_both)
    gs = ds.get("group_share", {})
    want = r3.SHARES["R3MV"] if a.arm == "R" else json.loads((RES / f"w_{a.wfile}.json").read_text())["group_share"]
    ck |= {f"share_{p}": (gs.get(p), v, abs(gs.get(p, -1) - v) < 1e-6) for p, v in want.items()}
    ok = bool(all(v[2] for v in ck.values()))
    res = dict(tag=a.tag, arm=a.arm, run=str(d), pass_=ok, checks=ck, expected=ex, steps=len(S.get("loss/total", [])),
               road=dict(road_ho=mean("loss/on/road_ho"), road_pos=mean("loss/on/road_pos"), road_posmean=mean("loss/on/road_posmean"),
                         n_hinge=mean("loss/on/n_hinge"), n_imit=mean("loss/on/n_imit")), draw_shares=ds)
    r3.dump(RES / f"smoke_{a.arm if a.arm == 'R' else a.wfile}{a.suffix}.json", res)
    print(json.dumps(res, indent=1, default=float))
    if not ok:
        raise SystemExit(2)


def cmd_weights(a):
    B = np.load(W3 / f"{a.base}.npz")
    T = np.load(CR / a.pool / "tab.npz")
    P = np.load(CR / a.s3 / "tab.npz")
    names, n0 = T["names"].astype(str), len(B["names"])
    assert np.array_equal(names[:n0], B["names"].astype(str)), f"{a.pool}: prefix != {a.base}'s rows"
    s3n = P["names"].astype(str)
    assert len(names) == n0 + len(s3n) and all(x.endswith(y[2:]) for x, y in zip(names[n0:], s3n)), f"{a.pool}: tail != {a.s3}'s rows"   # merge prefixes on:<p>:
    ot = P["otype"].astype(str)
    has = np.isfinite(P["fut"].reshape(len(s3n), -1)).all(1) if "fut" in P.files else np.ones(len(s3n), bool)
    k_all = a.k_on + a.k_s3
    w = np.zeros(len(names))
    w[:n0] = B["w"].astype(np.float64) * a.k_on / k_all
    present = {t: s for t, s in S3_SHARE.items() if ((ot == t) & has).any()}
    tot = sum(present.values())
    grp = np.concatenate([B["grp"].astype(str), np.array([f"s3_{t}" for t in ot])]).astype(B["grp"].dtype if B["grp"].dtype.kind == "U" else str)
    rec = dict(base=a.base, pool=a.pool, s3=a.s3, k_on=a.k_on, k_s3=a.k_s3, k_all=k_all, s3_share_of_on=a.k_s3 / k_all,
               shares_wanted=S3_SHARE, shares_used={t: s / tot for t, s in present.items()}, rows={}, out=a.out)
    for t, s in present.items():
        mm = np.flatnonzero((ot == t) & has)
        w[n0 + mm] = a.k_s3 / k_all * (s / tot) / len(mm)
        rec["rows"][t] = dict(rows=int((ot == t).sum()), drawable=int(len(mm)), per_batch=a.k_s3 * s / tot,
                              by_cls={c: int((P["ocls"].astype(str)[mm] == c).sum()) for c in np.unique(P["ocls"].astype(str))} if "ocls" in P.files else {})
    assert abs(w.sum() - 1) < 1e-9, w.sum()
    gs = defaultdict(float)
    for g_, x in zip(grp, w):
        gs[str(g_)] += x
    rec["group_share"] = {k: float(v) for k, v in sorted(gs.items())}
    W5.mkdir(parents=True, exist_ok=True)
    np.savez(W5 / f"{a.out}.npz", names=names, w=w, grp=grp)
    r3.dump(RES / f"w_{a.out}.json", rec)
    print(json.dumps(rec, indent=1))


def cmd_gate(a):
    import pandas as pd
    r3.RES = RES                                                   # round 3's gate, written into round 5's results
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):                   # r3's passing list is not the answer here
        r3.cmd_gate(argparse.Namespace(arms=a.arms, ctrl=a.ctrl, stem=a.stem))
    out = json.loads((RES / f"{a.stem}.json").read_text())
    sys.path[:0] = [str(REPO / "experiments/pai_fix/scripts")]
    import pai_fix as PF
    E = pd.read_csv(REPO / "experiments/pai_fix/scenes/exam.csv").set_index("scene")
    rd = lambda t: pd.read_csv(DATA / f"runs/pai_fix/facts/{t}-vl.csv").set_index("scene").reindex(E.index)  # noqa: E731
    C = rd(a.ctrl)
    md = (RES / f"{a.stem}.md").read_text().splitlines()
    md += ["", f"Class guard: zeros per class over {' + '.join(GUARD_SETS)} (scenes both reads scored without RPC), arm <= CTRL + {GUARD_TOL}; "
           "G-LAUNCH rear-hit partials.", "", "| arm | " + " | ".join(PF.ZCLS) + " | G-LAUNCH rear-hit arm / CTRL | guard | all four |",
           "|:--|" + "--:|" * len(PF.ZCLS) + ":--|:--|:--|"]
    passing = []
    for t in a.arms:
        F = rd(t)
        m = E.subset.isin(GUARD_SETS) & F.score.notna() & C.score.notna() & ~F.rpc.fillna(False).astype(bool) & ~C.rpc.fillna(False).astype(bool)
        z = {c: (int((F.fclass[m] == c).sum()), int((C.fclass[m] == c).sum())) for c in PF.ZCLS}
        g_ok = all(u <= v + GUARD_TOL for u, v in z.values())
        gl = E.subset == "G-LAUNCH"
        rh = (int((F.fclass[gl] == "rear-hit").sum()), int((C.fclass[gl] == "rear-hit").sum()))
        r = out["arms"][t]
        r["class_guard"] = dict(n=int(m.sum()), zeros=z, ok=g_ok)
        r["glaunch_rear_hit"] = rh
        r["ok"]["CLASS"] = g_ok
        r["pass_"] = all(r["ok"].values())
        if r["pass_"]:
            passing.append(t)
        md.append(f"| {t} | " + " | ".join(f"{u} / {v}{'' if u <= v + GUARD_TOL else ' FAIL'}" for u, v in z.values())
                  + f" | {rh[0]} / {rh[1]} | {'ok' if g_ok else 'FAIL'} | {'PASS' if r['pass_'] else 'fail'} |")
    out["passing"] = passing
    out["lines"]["CLASS"] = f"every zero class over {'+'.join(GUARD_SETS)} <= CTRL + {GUARD_TOL}"
    r3.dump(RES / f"{a.stem}.json", out)
    (RES / f"{a.stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-len(a.arms) - 4:]), file=sys.stderr)
    print(" ".join(passing))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("expect")
    p.add_argument("--pool", default="on1_JTPH"), p.add_argument("--wfile", default="R3MV"), p.add_argument("--index", default="road_on1_JTPH.npz")
    p = sub.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--arm", choices=("R", "A"), required=True)
    p.add_argument("--wfile", default="R3MV"), p.add_argument("--index", default="road_on1_JTPH.npz"), p.add_argument("--pool", default="on1_JTPHO")
    p.add_argument("--k-s3", type=int, default=9), p.add_argument("--s3-hinged", type=float, default=0.0, help="road-hinged share of the S3 rows")
    p.add_argument("--suffix", default="")
    p = sub.add_parser("weights")
    p.add_argument("--base", default="R3MV"), p.add_argument("--pool", default="on1_JTPHO"), p.add_argument("--s3", default="on1_PAIO")
    p.add_argument("--k-on", type=int, default=64), p.add_argument("--k-s3", type=int, default=9), p.add_argument("--out", default="R5A")
    p = sub.add_parser("gate")
    p.add_argument("--arms", nargs="+", required=True), p.add_argument("--ctrl", default="R2C-L18-s0"), p.add_argument("--stem", default="gate_r5r")
    a = ap.parse_args()
    dict(expect=cmd_expect, smoke=cmd_smoke, weights=cmd_weights, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    main()
