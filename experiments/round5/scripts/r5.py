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
Stage N (prereg addendum 3): R5-N = R3MV with every PAI human row whose state has a lead vehicle at weight 0 (LEAD rule below).
  r5.py leadrows                (box, CPU) the LEAD rule on every PAI human row (plain rows: their own t0 boxes and 4 s label; stretched rows
        inherit their base plain row's flag) -> $DATA_DIR/runs/round5/r5/N/lead_rows.npz; the R5N draw file (R3MV, lead rows -> 0, each part
        renormalised to its R3MV share) and R5NI (the same code with no row dropped; asserted bit-equal to R3MV) -> w/<out>.npz +
        results/w_<out>.json; rows removed by label class, PAI rows a batch -> results/lead_rows.{md,json}
  r5.py lead281                 (Mac, tmp/round5/{tax,facts}) the LEAD rule on the human's state of the stored PAI 281 scenes: launches
        (NTV0's 37, at the human's launch t_L) and overshoot zeros (NTV0 / R2C / R2J, at the human's rest and 4 s before)
        -> results/lead_281.{csv,md}
  r5.py split281 --read K=FACTS.csv .. --out F.md   launches of the 37 and overshoot zeros per read, split lead / no-lead by lead_281.csv
Stage G (prereg addendum 4): R5-G = R3MV's argv with draw R5G ("stand, then go promptly").
  r5.py gonow                   (box, CPU) GO-NOW rows (fed < 0.5 m/s and the 4 s label's path reaches 0.5 m by its 1.0 s pose: the human
        starts within 1 s; launch_rows.md's "leaves 0.5 m" rule on the first 1 s) in the three PAI parts and navtrain; R5G = R2JF with every
        PAI human row fed < 2 m/s that is not GO-NOW at weight 0, each part renormalised to its share (MV's rule with the GO-NOW rows kept;
        the same code with no GO-NOW row is asserted bit-equal to R3MV) + R3MVL's navtrain weights (launch rows x 3, bit-equal arrays)
        -> w/R5G.npz + results/w_R5G.json; rows a batch by source, 4 s label distance, LEAD share -> results/gonow_rows.{md,json}
  r5.py gonow --go-batch 4 --go-drop ph05 --out R5G2   (addendum 4b) R5G with the GO-NOW rows of the listed parts at 0 and the others
        scaled by one common factor to 4 rows a batch, the rest of each part renormalised so every part keeps its share -> w/R5G2.npz + w_R5G2.json
  r5.py l37 --read K=FACTS.csv .. --pair A:B .. --target A:B ..   LAUNCH-37 (lead_281.csv's 37 launch scenes): launched, rear-hit, mean, zeros,
        lead / no-lead split per read; paired A - B over the 37 (mean, launched) and over PAI-FIX TARGET (vl facts) -> results/launch37.{md,json}
        (--stem); --sep PAIR=TARGET: further registered comparisons (default: the first --pair with the first --target)
Stage L (addendum 5): R5-LC (MV-LC) = R3MV's argv with draw R5LC (every PAI human row whose 4 s label is a lane change off the route lane -> 0).
  r5.py lcrows                  (box, CPU) D1's rule (LC_*: v0 > 5, 4 s point |y| > 2.5 m, |yaw| < 12 deg; route polyline not within 1.2 m of it)
        on the three PAI parts' own rows -> R5LC (each part renormalised to its R3MV share) and R5LCI (no row dropped, asserted bit-equal to
        R3MV) -> w/<out>.npz + results/w_<out>.json; rows removed by part / class, lane-change share before / after vs navtrain -> results/lc_rows.{md,json}
  r5.py lcverdict --json J --md M --read K=F ..   the registered adopt-candidate rule on r2.py r281's json + the secondary column (new zeros /
        partials on the scenes NTV0 scores 1.0), appended to M
  r5.py lcpcs --pcs Q --json J --md M   the local PAI PCS ruler's per-read PCS and paired difference vs NTV0 + the secondary rule, appended to M
  r5.py newbreaks --json J --md M --read K=F ..   lcverdict's new zeros / partials column alone, for any read set (r5_281.sh TGT=R5A)
  r5.py pcsrow --pcs Q --json J --md M --subject S ..   the ruler's rows for the named subjects only, appended to M (r5_281.sh TGT=R5A)
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
# LEAD rule (addendum 3), on one state = the rig frame at t0 (rear axle, x forward): the path = the human's own 4 s future (the row's 8 label
# poses; origin first) extended LEAD_EXT m along the last pose's heading; an object leads when its box centre lies within HW + W / 2 + LEAD_LAT
# of the path, its heading is within LEAD_PSI of the path's there, its length >= LEAD_MINL (vehicles and cyclists, not pedestrians; the pool
# boxes carry no class) and its gap (path station of the centre - L / 2 - FRONT) is in [-1, max(LEAD_G, LEAD_T x v0)] m. As rf_speedgeo.lead
# (40 m, 60 deg, hw + W / 2 + 0.3) plus the 4 s headway reach; the PAI ego box is the rollouts' (tax dims front 4.071 m, half width 1.0785 m).
FRONT, HW = 4.071, 1.0785
LEAD_G, LEAD_T, LEAD_PSI, LEAD_LAT, LEAD_MINL, LEAD_EXT = 40.0, 4.0, np.radians(60.0), 0.3, 1.5, 200.0
GO_T, GO_L = 1.0, 0.5                                                     # GO-NOW (addendum 4): fed < r3.V_LROW, label path >= GO_L by GO_T s
LC_V, LC_Y, LC_YAW, LC_ROUTE = 5.0, 2.5, 12.0, 1.2                         # addendum 5: D1's lane-change shape and route-lane rule (d1.py labels)


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
    elif a.arm in ("N", "G", "L"):                                 # R5-N / R5-G / R5-LC (addenda 3-5): R3MV's argv, draw R5N / R5G / R5LC
        ex = json.loads((RES / f"w_{a.wfile}.json").read_text())
        k_both = 32
        b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", (d / "log.txt").read_text())
        ck |= dict(no_road_pai=(m is not None, False, m is None),
                   batch_rows=(b.groups() if b else None, ("51", "13", str(r3.K_ON)), bool(b) and b.groups() == ("51", "13", str(r3.K_ON))),
                   pool_imit_rows_without_hinge=(max(S.get("loss/on/n_hinge", [np.nan])), 0, max(S.get("loss/on/n_hinge", [1])) == 0),
                   agent_ho_zero=(max(S.get("loss/on/agent_ho", [0])), 0, max(S.get("loss/on/agent_ho", [0])) == 0))
        if a.arm == "G":                                           # R3MVL's navtrain weights ride along (launch rows x 3)
            ck |= dict(nav_listed=(ds.get("nav_listed"), ds.get("nav_rows"), ds.get("nav_listed", 0) >= 0.99 * ds.get("nav_rows", 1)),
                       nav_up_w=(ds.get("nav_up_w"), [r3.NAV_UP], ds.get("nav_up_w") == [r3.NAV_UP]),
                       nav_up_share=(ds.get("nav_up_share"), ex["nav_launch_share_after"],
                                     abs(ds.get("nav_up_share", -1) - ex["nav_launch_share_after"]) < 0.1 * ex["nav_launch_share_after"]))
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


def lead_gap(fut, box, val, v0):
    """fut (n, 8, 3) the human's future poses, box (n, K, 5) objects (x, y, yaw, L, W) and val (n, K) at t0, v0 (n,) the state's speed, all in
    the rig frame at t0 -> gap (n,) of the nearest leading object (inf: none; nan: no finite path) and its slot (n,) (-1: none)."""
    fut = np.asarray(fut, np.float64)
    n = len(fut)
    fin = np.isfinite(fut.reshape(n, -1)).all(1)
    fut = np.where(fin[:, None, None], fut, 0.0)
    yl = fut[:, -1, 2]
    P = np.concatenate([np.zeros((n, 1, 2)), fut[..., :2], fut[:, -1:, :2] + LEAD_EXT * np.stack([np.cos(yl), np.sin(yl)], -1)[:, None]], 1)
    hd = np.concatenate([fut[..., 2], yl[:, None]], 1)                     # segment j ends at label pose j (the extension: the last heading)
    A0, A1 = P[:, :-1], P[:, 1:]
    sl = np.linalg.norm(A1 - A0, axis=-1)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(sl, 1)[:, :-1]], 1)
    reach = np.maximum(LEAD_G, LEAD_T * np.asarray(v0, np.float64))
    gap, slot, r = np.full(n, np.inf), np.full(n, -1), np.arange(n)
    for k in range(box.shape[1]):
        b = np.asarray(box[:, k], np.float64)
        q = b[:, None, :2]
        u = np.clip(((q - A0) * (A1 - A0)).sum(-1) / np.maximum(sl ** 2, 1e-9), 0, 1)
        d = np.linalg.norm(A0 + u[..., None] * (A1 - A0) - q, axis=-1)
        j = d.argmin(1)
        g = cs[r, j] + u[r, j] * sl[r, j] - b[:, 3] / 2 - FRONT
        dpsi = np.abs((b[:, 2] - hd[r, j] + np.pi) % (2 * np.pi) - np.pi)
        ok = (np.asarray(val[:, k], bool) & (b[:, 3] >= LEAD_MINL) & (d[r, j] <= HW + b[:, 4] / 2 + LEAD_LAT) & (dpsi < LEAD_PSI)
              & (g >= -1.0) & (g <= reach) & (g < gap))
        gap, slot = np.where(ok, g, gap), np.where(ok, k, slot)
    return np.where(fin, gap, np.nan), slot


def cmd_leadrows(a):
    """The LEAD rule on the PAI human rows; R5N / R5NI draw files; the offline tables (prereg addendum 3)."""
    from collections import Counter
    O = DATA / "runs/round5/r5/N"
    O.mkdir(parents=True, exist_ok=True)
    T = {p: r3.load_part(p) for p in r3.PARTS}
    A = np.load(CR / "on1_PAIH/agent.npz")
    P = T["ph"]
    pn = P["names"].astype(str)
    assert np.array_equal(A["tokens"].astype(str), pn), "agent.npz rows != on1_PAIH"
    box0, val0 = A["box"][:, 0], A["valid"][:, 0] & A["ok"][:, None]
    gap_p, slot_p = [], []
    for i in range(0, len(pn), 20000):
        g, k = lead_gap(P["fut"][i:i + 20000], box0[i:i + 20000], val0[i:i + 20000], P["speed"][i:i + 20000])
        gap_p.append(g), slot_p.append(k)
    gap_p, slot_p = np.concatenate(gap_p), np.concatenate(slot_p)
    row = {x: i for i, x in enumerate(pn)}
    B = np.load(W3 / "R3MV.npz")
    names, w0, grp = B["names"].astype(str), B["w"].astype(np.float64), B["grp"]
    idx = {x: i for i, x in enumerate(names)}
    M = np.load(CR / "on1_JTPH/tab.npz")
    assert np.array_equal(M["names"].astype(str), names), "R3MV rows != on1_JTPH"
    rec = dict(rule=dict(text=("lead = an object (box length >= 1.5 m) whose centre lies within half the ego width + half its width + 0.3 m of the human's "
                               "4 s future path (extended along the last heading), heading within 60 deg of the path, gap (path station - L/2 - "
                               "front bumper) in [-1, max(40 m, 4 s x v0)]; plain rows: their own t0 boxes, label and fed speed; stretched rows: "
                               "their base plain row's flag"),
                         FRONT=FRONT, HW=HW, LEAD_G=LEAD_G, LEAD_T=LEAD_T, LEAD_PSI_deg=float(np.degrees(LEAD_PSI)), LEAD_LAT=LEAD_LAT,
                         LEAD_MINL=LEAD_MINL), parts={})
    out = {}
    keep_all = {}
    for p, t in T.items():
        nm = t["names"].astype(str)
        if p == "ph":
            b = np.arange(len(nm))
        else:
            b = np.array([row.get(x, -1) for x in t["base"].astype(str)])
            assert (b >= 0).all(), f"{p}: base rows missing"
        gap = gap_p[b]
        lead = np.isfinite(gap)
        ix = np.array([idx[f"on:{p}:{x[3:]}"] for x in nm])
        assert np.array_equal(M["speed"][ix], t["speed"]) and np.array_equal(M["fut"][ix], t["fut"]), f"{p}: merged rows != source"
        cls, _ = r3.label_class(t["speed"].astype(np.float64), t["fut"].astype(np.float64))
        dr = w0[ix] > 0
        drop = lead & dr
        keep_all[p] = (ix, drop, dr)
        out[p] = dict(names=np.array([f"on:{p}:{x[3:]}" for x in nm]), gap=gap, lead=lead, base_gap_nan=np.isnan(gap))
        wsum = w0[ix][dr].sum()
        rec["parts"][p] = dict(rows=int(len(nm)), drawable=int(dr.sum()), lead_drawable=int(drop.sum()), lead_share=float(drop.sum() / max(dr.sum(), 1)),
                               lead_share_drawweighted=float(w0[ix][drop].sum() / max(wsum, 1e-12)), no_path=int((np.isnan(gap) & dr).sum()),
                               lead_all_rows=int(lead.sum()), lead_share_all_rows=float(lead.mean()),
                               by_class={c: dict(drawable=int((dr & (cls == c)).sum()), removed=int((drop & (cls == c)).sum()),
                                                 kept=int((dr & ~drop & (cls == c)).sum())) for c in r3.CLASSES},
                               gap_q=np.nanpercentile(gap[drop], [10, 25, 50, 75, 90]).tolist() if drop.any() else None,
                               v0_q_removed=np.percentile(t["speed"][drop], [10, 50, 90]).tolist() if drop.any() else None,
                               v0_q_kept=np.percentile(t["speed"][dr & ~drop], [10, 50, 90]).tolist() if (dr & ~drop).any() else None,
                               gap_le15=int((drop & (gap <= 15)).sum()))
        rec["parts"][p]["scenes_with_kept"] = int(len(set(t["log"].astype(str)[dr & ~drop])))
        rec["parts"][p]["scenes_drawable"] = int(len(set(t["log"].astype(str)[dr])))
    np.savez(O / "lead_rows.npz", **{f"{p}_{k}": v for p, d in out.items() for k, v in d.items()})

    def draw(nodrop):
        w = w0.copy()
        for p, (ix, drop, dr) in keep_all.items():
            if nodrop or not drop.any():
                continue
            sh = w[ix].sum()
            keep = dr & ~drop
            w[ix[drop]] = 0.0
            w[ix[keep]] *= sh / w[ix[keep]].sum()
        assert abs(w.sum() - 1) < 1e-9
        nonpai = np.ones(len(w), bool)
        nonpai[np.concatenate([v[0] for v in keep_all.values()])] = False
        assert np.array_equal(w[nonpai], w0[nonpai]), "non-PAI weights changed"
        return w

    W5.mkdir(parents=True, exist_ok=True)
    for out_, nodrop in (("R5N", False), ("R5NI", True)):
        w = draw(nodrop)
        if nodrop:
            assert np.array_equal(w, w0), "R5NI != R3MV"
        gs = Counter()
        for g_, x in zip(grp.astype(str), w):
            gs[g_] += x
        np.savez(W5 / f"{out_}.npz", names=B["names"], w=w, grp=grp)
        r = dict(base="R3MV", out=out_, rule="R3MV, every PAI human row with a lead (LEAD rule) -> 0, each part renormalised to its R3MV share"
                 if not nodrop else "identity: R5N's code with no row dropped (bit-equal to R3MV)",
                 group_share={k: float(v) for k, v in sorted(gs.items()) if v > 0}, pai_drawable=int(sum((w[v[0]] > 0).sum() for v in keep_all.values())))
        r3.dump(RES / f"w_{out_}.json", r)
        rec[out_] = r
    # rows a batch: 64 pool rows (32 S + 32 PAI); PAI rows by class before / after
    wn = np.load(W5 / "R5N.npz")["w"]
    cl_all = np.full(len(names), "", dtype="U13")
    for p, t in T.items():
        cl_all[keep_all[p][0]] = r3.label_class(t["speed"].astype(np.float64), t["fut"].astype(np.float64))[0]
    pm = cl_all != ""
    rec["per_batch"] = {k: dict(pai=float(r3.K_ON * ww[pm].sum()), by_class={c: float(r3.K_ON * ww[pm & (cl_all == c)].sum()) for c in r3.CLASSES},
                                unique_pai_rows=int((ww[pm] > 0).sum()))
                        for k, ww in (("R3MV", w0), ("R5N", wn))}
    r3.dump(RES / "lead_rows.json", rec)
    L = ["# R5-N: the LEAD rule on the PAI human rows (offline, before training)", "", rec["rule"]["text"] + ".", "",
         "R3MV draws only PAI rows fed >= 2 m/s; R5-N sets the lead rows among them to 0 and renormalises each part to its R3MV share "
         "(ph 0.25, ph05 0.125, ph15 0.125 of the 64 pool rows).", "",
         "| part | R3MV drawable | lead (removed) | share | draw-weighted | gap <= 15 m | gap median (m) | v0 median removed / kept (m/s) | scenes kept / drawable |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for p, r in rec["parts"].items():
        L.append(f"| {p} | {r['drawable']} | {r['lead_drawable']} | {r['lead_share']:.1%} | {r['lead_share_drawweighted']:.1%} | {r['gap_le15']} | "
                 f"{r['gap_q'][2]:.1f} | {r['v0_q_removed'][1]:.1f} / {r['v0_q_kept'][1]:.1f} | {r['scenes_with_kept']} / {r['scenes_drawable']} |")
    L += ["", "Removed / drawable by label class (r3.label_class; launch and stand are 0 under MV's >= 2 m/s cut):", "",
          "| part | " + " | ".join(r3.CLASSES) + " |", "|:--|" + "--:|" * len(r3.CLASSES)]
    for p, r in rec["parts"].items():
        L.append(f"| {p} | " + " | ".join(f"{r['by_class'][c]['removed']} / {r['by_class'][c]['drawable']}" for c in r3.CLASSES) + " |")
    L += ["", "PAI human rows a batch (64 pool rows x draw mass) by class:", "", "| draw | PAI rows | " + " | ".join(r3.CLASSES) + " | unique PAI rows |",
          "|:--|--:|" + "--:|" * len(r3.CLASSES) + "--:|"]
    for k, r in rec["per_batch"].items():
        L.append(f"| {k} | {r['pai']:.2f} | " + " | ".join(f"{r['by_class'][c]:.2f}" for c in r3.CLASSES) + f" | {r['unique_pai_rows']} |")
    L += ["", f"Lead share over all plain rows (any speed): {rec['parts']['ph']['lead_share_all_rows']:.1%}. Files: w_R5N.json, w_R5NI.json (R5NI = R3MV bit for bit)."]
    (RES / "lead_rows.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _rig(a, X):
    """poses X (n, 3) in the rollout frame -> pose a's rig frame."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = X[:, :2] - a[:2]
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], (X[:, 2] - a[2] + np.pi) % (2 * np.pi) - np.pi]


def human_lead(o, t):
    """The LEAD rule on the human's logged state at time t (us) of one stored rollout (tax record): log pose, its 4 s log future, its speed,
    the logged traffic at the nearest 2 Hz snapshot (within 0.3 s) -> (gap, v0); gap inf = no lead, nan = outside the snapshots."""
    Lg = o["log"]
    T, P = np.asarray(Lg["T"], float), np.asarray(Lg["P"], float)
    Y = np.unwrap(np.asarray(Lg["Y"], float))
    if t < T[0] or t > T[-1]:
        return np.nan, np.nan
    pose = lambda q: np.c_[np.interp(q, T, P[:, 0]), np.interp(q, T, P[:, 1]), np.interp(q, T, Y)]  # noqa: E731
    a = pose(np.array([t]))[0]
    tq = t + np.arange(1, 9) * 0.5e6
    fut = _rig(a, pose(np.minimum(tq, T[-1])))
    v0 = float(np.hypot(*(pose(np.array([t + 0.25e6]))[0, :2] - pose(np.array([t - 0.25e6]))[0, :2])) / 0.5)
    tg = np.asarray(o["tg"], float)
    j = int(np.argmin(np.abs(tg - t)))
    if abs(tg[j] - t) > 0.3e6:
        return np.nan, v0
    tr = o["traffic"][j]
    if not len(tr["id"]):
        return np.inf, v0
    X = _rig(a, np.c_[np.asarray(tr["x"], float), np.asarray(tr["y"], float), np.asarray(tr["yaw"], float)])
    box = np.c_[X, np.asarray(tr["L"], float), np.asarray(tr["W"], float)][None]
    g, _ = lead_gap(fut[None], box, np.ones((1, box.shape[1]), bool), np.array([v0]))
    return float(g[0]), v0


def cmd_lead281(a):
    import pickle
    import pandas as pd
    sys.path[:0] = [str(REPO / "experiments/pai_fix/scripts"), str(HERE)]
    import pai_fix as PF
    import d1_mac as D1
    TM = REPO / "tmp/round5"
    A = {o["scene"]: o for o in pickle.load(open(TM / "tax/NTV0-L18-s0-281ag.pkl", "rb"))["rolls"] if "ctrl" in o}
    F = {k: pd.read_csv(TM / f"facts/{k}-L18-s0-281ag.csv").set_index("scene") for k in ("NTV0", "R2C", "R2J")}
    F = {k: f[~f.rpc] for k, f in F.items()}
    dl = pd.read_csv(RES / "d1_launch.csv")
    dl = dl[(dl.read == "NTV0") & (dl.set == "281")].set_index("scene")
    rows = []
    for s in F["NTV0"].index[PF.launch_mask(F["NTV0"])]:
        o = A[s]
        th = float(o["ctrl"]["t"][0]) + D1.HAND
        tl = th + float(dl.t_L[s]) * 1e6 if s in dl.index else np.nan
        g, v = human_lead(o, tl) if np.isfinite(tl) else (np.nan, np.nan)
        gh, _ = human_lead(o, th)
        rows.append(dict(scene=s, kind="launch", t_rel=float(dl.t_L.get(s, np.nan)), gap=g, v0=v, lead=bool(np.isfinite(g)), gap_hand=gh,
                         lead_hand=bool(np.isfinite(gh)), won_mode_ntv0=dl.won_mode.get(s, "")))
    ov = sorted(set().union(*[set(f.index[f.fclass == "overshoot"]) for f in F.values()]))
    for s in ov:
        o = A[s]
        Lg = o["log"]
        Pl, Tl = np.asarray(Lg["P"], float), np.asarray(Lg["T"], float)
        Sl = D1.arclen(Pl)
        vl = np.r_[0, np.diff(Sl) / np.maximum(np.diff(Tl) * 1e-6, 1e-3)]
        vl = np.convolve(np.r_[[vl[0]] * 5, vl, [vl[-1]] * 5], np.ones(11) / 11, "valid")
        i_rest = next((i for i in range(len(vl)) if vl[i] < 0.3 and (vl[i:] < 0.5).all()), len(vl) - 1)   # d1_mac.overshoot_one's rest
        tr_ = float(Tl[i_rest])
        g0, v0 = human_lead(o, tr_)
        g4, v4 = human_lead(o, tr_ - 4e6)
        rows.append(dict(scene=s, kind="overshoot", t_rel=(tr_ - float(o["ctrl"]["t"][0])) * 1e-6, gap=g0, v0=v0, lead=bool(np.isfinite(g0)),
                         gap_m4=g4, v0_m4=v4, lead_m4=bool(np.isfinite(g4)), nan_m4=bool(np.isnan(g4)),
                         **{f"ov_{k}": bool(s in f.index and f.fclass[s] == "overshoot") for k, f in F.items()}))
    D = pd.DataFrame(rows)
    D["lead_any"] = D.lead | D.get("lead_m4", False).fillna(False).astype(bool)
    D.to_csv(RES / "lead_281.csv", index=False, float_format="%.3f")
    la = D[D.kind == "launch"].set_index("scene")
    L = ["# LEAD rule on the stored PAI 281 human states (Mac, NTV0's tax record; the logged traffic is the same in every read)", "",
         "Rule: r5.py LEAD (addendum 3). Launches: NTV0's 37 launch scenes (handoff < 1 m/s, log > 5 m), the human's state at its launch t_L "
         "(d1_launch.csv; log 0.5 m past the handoff pose). Overshoot zeros: the human's state at its rest (d1 overshoot rule) and 4 s before.", "",
         f"Launch scenes: {len(la)}; lead at t_L {int(la.lead.sum())}, at the handoff {int(la.lead_hand.sum())}; NTV0's serve chose the LEAD mode "
         f"{int((la.won_mode_ntv0 == 'lead').sum())} (main session's 21); both {int((la.lead & (la.won_mode_ntv0 == 'lead')).sum())}.", "",
         "| read | launched / lead scenes | launched / no-lead scenes |", "|:--|--:|--:|"]
    for k, f in F.items():
        x = f.reindex(la.index)
        lz = PF.launched(x).fillna(False).astype(bool)
        L.append(f"| {k} 281 ACC + GAP | {int((lz & la.lead).sum())} / {int(la.lead.sum())} | {int((lz & ~la.lead).sum())} / {int((~la.lead).sum())} |")
    ovd = D[D.kind == "overshoot"]
    L += ["", "| read | overshoot zeros | lead at rest | lead 4 s before rest | lead at either | no lead |", "|:--|--:|--:|--:|--:|--:|"]
    for k in F:
        x = ovd[ovd[f"ov_{k}"]]
        L.append(f"| {k} | {len(x)} | {int(x.lead.sum())} | {int(x.lead_m4.sum())} | {int(x.lead_any.sum())} | {int((~x.lead_any).sum())} |")
    L += ["", f"4 s before rest outside the rollout's traffic snapshots: {int(ovd.nan_m4.sum())} of {len(ovd)} scenes (counted no lead)."]
    fx = TM / "tax/NTV0-L18-s0-vl.pkl"
    if fx.exists():                                                # PAI-FIX G-LAUNCH (6 scenes), the same rule on NTV0's native read
        AV = {o["scene"]: o for o in pickle.load(open(fx, "rb"))["rolls"] if "ctrl" in o}
        df = pd.read_csv(RES / "d1_launch.csv")
        df = df[(df.read == "NTV0") & (df.set == "fix") & df.g_launch].set_index("scene")
        g = []
        for s, r in df.iterrows():
            o = AV[s]
            gg, v = human_lead(o, float(o["ctrl"]["t"][0]) + D1.HAND + float(r.t_L) * 1e6)
            g.append(f"{s[7:15]} {'lead ' + format(gg, '.1f') + ' m' if np.isfinite(gg) else 'no lead'}")
        L += ["", f"PAI-FIX G-LAUNCH ({len(df)} scenes, NTV0 native read, the human's state at t_L): " + "; ".join(g) + "."]
    (RES / "lead_281.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_split281(a):
    import pandas as pd
    sys.path[:0] = [str(REPO / "experiments/pai_fix/scripts")]
    import pai_fix as PF
    D = pd.read_csv(RES / "lead_281.csv")
    la = D[D.kind == "launch"].set_index("scene")
    lead = la.lead.astype(bool)
    ov = D[D.kind == "overshoot"].set_index("scene").lead_any.astype(bool)
    L = [f"# PAI 281 launches and overshoots split by the LEAD rule (lead_281.csv: {len(la)} launch scenes, {int(lead.sum())} lead)", "",
         "| read | launched lead | launched no-lead | rear-hit lead / no-lead | overshoot zeros | of which lead (rest or 4 s before) | no lead | overshoot zeros outside lead_281 |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for r in a.read:
        k, f = r.split("=", 1)
        F = pd.read_csv(f).set_index("scene")
        F = F[~F.rpc]
        x = F.reindex(la.index)
        lz = PF.launched(x).fillna(False).astype(bool)
        rh = x.fclass == "rear-hit"
        oz = F.index[F.fclass == "overshoot"]
        known = [s for s in oz if s in ov.index]
        L.append(f"| {k} | {int((lz & lead).sum())} / {int(lead.sum())} | {int((lz & ~lead).sum())} / {int((~lead).sum())} | "
                 f"{int((rh & lead).sum())} / {int((rh & ~lead).sum())} | {len(oz)} | {int(ov[known].sum())} | {int((~ov[known]).sum())} | {len(oz) - len(known)} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def go_now(v0, fut):
    """(n,) fed speed, (n, 8, 3) label poses at 0.5 .. 4 s -> GO-NOW mask (fed < 0.5 m/s, label path >= GO_L by GO_T), path by GO_T, 4 s path."""
    fut = np.asarray(fut, np.float64)
    P = np.concatenate([np.zeros((len(fut), 1, 2)), fut[:, :, :2]], 1)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    L1 = seg[:, :int(round(GO_T / 0.5))].sum(1)
    return (np.asarray(v0, np.float64) < r3.V_LROW) & (L1 >= GO_L), L1, seg.sum(1)


def cmd_gonow(a):
    """R5G draw file and the offline GO-NOW tables (addendum 4)."""
    S = r3.pai_states()
    B = np.load(W3 / "R2JF.npz")
    names, w0, grp = B["names"].astype(str), B["w"].astype(np.float64), B["grp"]
    idx = {x: i for i, x in enumerate(names)}
    MV, MVL = np.load(W3 / "R3MV.npz"), np.load(W3 / "R3MVL.npz")
    assert np.array_equal(MV["names"], B["names"]) and np.array_equal(MVL["w"], MV["w"]), "R3MV / R3MVL rows"
    LR = np.load(DATA / "runs/round5/r5/N/lead_rows.npz")
    G, X = {}, {}
    for p, d in S.items():
        g, L1, L4 = go_now(d["speed"], d["fut"])
        assert np.array_equal(LR[f"{p}_names"].astype(str), d["names"]), f"{p}: lead_rows order"
        G[p], X[p] = g, dict(ix=np.array([idx[x] for x in d["names"]]), L1=L1, L4=L4, v0=d["speed"].astype(np.float64),
                             lead=LR[f"{p}_lead"].astype(bool), gap=LR[f"{p}_gap"])

    def draw(keep_go):                                             # r3.cmd_weights' MV arithmetic, GO-NOW rows exempt from the < 2 m/s cut
        w = w0.copy()
        for p, d in S.items():
            ix = X[p]["ix"]
            sh = w[ix].sum()
            drop = (d["speed"].astype(np.float64) < r3.V_MV) & ~(G[p] if keep_go else np.zeros(len(ix), bool)) & (w[ix] > 0)
            keep = (w[ix] > 0) & ~drop
            w[ix[drop]] = 0.0
            if keep.any():
                w[ix[keep]] *= sh / w[ix[keep]].sum()
        return w

    assert np.array_equal(draw(False), MV["w"]), "R5G's code without GO-NOW rows != R3MV"
    w = draw(True)
    pai = np.zeros(len(w), bool)
    pai[np.concatenate([x["ix"] for x in X.values()])] = True
    assert abs(w.sum() - 1) < 1e-9 and np.array_equal(w[~pai], MV["w"][~pai]), "non-PAI weights changed"
    q = lambda x, p_: float(np.percentile(x, p_)) if len(x) else float("nan")  # noqa: E731
    rec = dict(rule=(f"GO-NOW row: fed < {r3.V_LROW} m/s and the row's own label path reaches {GO_L} m by its {GO_T} s pose (the human starts "
                     "within 1 s; launch_rows.md's 'leaves 0.5 m' on the first second). R5G = R2JF with every PAI human row fed < "
                     f"{r3.V_MV} m/s that is not GO-NOW at weight 0, each part renormalised to its share (MV's rule, GO-NOW rows kept) + "
                     "R3MVL's navtrain weights (launch rows x 3)"), GO_T=GO_T, GO_L=GO_L, parts={})
    for p, x in X.items():
        ix, g, v0, L4 = x["ix"], G[p], x["v0"], x["L4"]
        dr0 = w0[ix] > 0
        gd = g & dr0
        low = dr0 & (v0 < r3.V_MV) & ~g
        r = dict(drawable_r2jf=int(dr0.sum()), drawable_mv=int((MV["w"][ix] > 0).sum()), drawable_g=int((w[ix] > 0).sum()), gonow=int(gd.sum()),
                 gonow_share_of_fed_lt05=float(gd.sum() / max((dr0 & (v0 < r3.V_LROW)).sum(), 1)),
                 zeroed=dict(stand=int((low & (v0 < r3.V_LROW) & (L4 < r3.L_STAND)).sum()), go_later=int((low & (v0 < r3.V_LROW) & (L4 >= r3.L_STAND)).sum()),
                             slow_05_2=int((low & (v0 >= r3.V_LROW)).sum())),
                 per_batch=float(r3.K_ON * w[ix][g].sum()), per_batch_lead=float(r3.K_ON * w[ix][g & x["lead"]].sum()),
                 L4=[q(L4[gd], 25), q(L4[gd], 50), q(L4[gd], 75)], L1_med=q(x["L1"][gd], 50), v0_med=q(v0[gd], 50),
                 lead=int((gd & x["lead"]).sum()), lead_share=float((gd & x["lead"]).sum() / max(gd.sum(), 1)), lead_gap_med=q(x["gap"][gd & x["lead"]], 50),
                 L4_lead_med=q(L4[gd & x["lead"]], 50), L4_nolead_med=q(L4[gd & ~x["lead"]], 50), scenes=int(len(set(S[p]["log"][gd]))))
        rec["parts"][p] = r
    nwm = dict(zip(MVL["nav_names"].astype(str).tolist(), MVL["nav_w"].astype(np.float64).tolist()))
    nn = dict(rows=0, launch=0, gonow=0, wsum=0.0, wl=0.0, wg=0.0, L4g=[], L4l=[], L1g=[])
    for i in range(12):
        t = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        ok = ~np.isnan(t["fut"]).any((1, 2))
        g, L1, L4 = go_now(t["speed"][ok], t["fut"][ok])
        lr = (t["speed"][ok].astype(np.float64) < r3.V_LROW) & (L4 >= r3.L_STAND)
        nw = np.array([nwm.get(x, np.nan) for x in t["names"][ok].astype(str)])
        assert np.isfinite(nw).all(), "navtrain rows missing from R3MVL nav_w"
        nn["rows"] += int(ok.sum()); nn["launch"] += int(lr.sum()); nn["gonow"] += int(g.sum())
        nn["wsum"] += nw.sum(); nn["wl"] += nw[lr].sum(); nn["wg"] += nw[g].sum()
        nn["L4g"].append(L4[g]), nn["L4l"].append(L4[lr]), nn["L1g"].append(L1[g])
    L4g, L4l, L1g = (np.concatenate(nn.pop(k)) for k in ("L4g", "L4l", "L1g"))
    rec["navtrain"] = dict(rows=nn["rows"], launch=nn["launch"], gonow=nn["gonow"], gonow_share_of_launch=nn["gonow"] / max(nn["launch"], 1),
                           per_batch_launch=51 * nn["wl"] / nn["wsum"], per_batch_gonow=51 * nn["wg"] / nn["wsum"],
                           per_batch_launch_r3mv=51 * nn["launch"] / nn["rows"], L4_gonow=[q(L4g, 25), q(L4g, 50), q(L4g, 75)],
                           L4_launch=[q(L4l, 25), q(L4l, 50), q(L4l, 75)], L1_gonow_med=q(L1g, 50))
    up = MVL["nav_w"] > 1
    rec |= dict(nav_rows=int(len(MVL["nav_w"])), nav_launch_rows=int(up.sum()), nav_up_w=r3.NAV_UP,
                nav_launch_share_after=json.loads((REPO / "experiments/round3/results/w_R3MVL.json").read_text())["nav_launch_share_after"])
    gs = defaultdict(float)
    for g_, x in zip(grp.astype(str), w):
        gs[g_] += x
    rec["group_share"] = {k: float(v) for k, v in sorted(gs.items()) if v > 0}
    rec["pai_drawable"] = int(sum(r["drawable_g"] for r in rec["parts"].values()))
    rec["pai_gonow_per_batch"] = float(sum(r["per_batch"] for r in rec["parts"].values()))
    W5.mkdir(parents=True, exist_ok=True)
    if a.go_batch > 0:                                             # R5G2 (addendum 4b): GO-NOW rows up to go_batch rows a batch
        gm = {p: G[p] & (w[X[p]["ix"]] > 0) & (p not in a.go_drop) for p in X}
        k = a.go_batch / r3.K_ON / sum(w[X[p]["ix"]][gm[p]].sum() for p in X)
        for p, x in X.items():
            ix = x["ix"]
            sh = w[ix].sum()
            w[ix[G[p] & (w[ix] > 0) & ~gm[p]]] = 0.0
            w[ix[gm[p]]] *= k
            rest = (w[ix] > 0) & ~gm[p]
            w[ix[rest]] *= (sh - w[ix[gm[p]]].sum()) / w[ix[rest]].sum()
            rec["parts"][p] |= dict(gonow_kept=int(gm[p].sum()), per_batch_up=float(r3.K_ON * w[ix][gm[p]].sum()),
                                    passes_9500=float(9500 * r3.K_ON * w[ix][gm[p]].sum() / max(gm[p].sum(), 1)), share_after=float(w[ix].sum()))
        assert abs(w.sum() - 1) < 1e-9 and np.array_equal(w[~pai], MV["w"][~pai]), "R5G2: weights"
        gs = defaultdict(float)
        for g_, x in zip(grp.astype(str), w):
            gs[g_] += x
        rec |= dict(group_share={k_: float(v) for k_, v in sorted(gs.items()) if v > 0}, go_batch=a.go_batch, go_drop=a.go_drop, go_factor=float(k),
                    pai_gonow_per_batch=float(sum(r3.K_ON * w[X[p]["ix"]][gm[p]].sum() for p in X)),
                    pai_drawable=int(sum((w[x["ix"]] > 0).sum() for x in X.values())))
        np.savez(W5 / f"{a.out}.npz", names=B["names"], w=w, grp=grp, nav_names=MVL["nav_names"], nav_w=MVL["nav_w"])
        r3.dump(RES / f"w_{a.out}.json", rec | dict(base="R5G", out=a.out))
        print(json.dumps({p: {q: r[q] for q in ("gonow", "gonow_kept", "per_batch_up", "passes_9500", "share_after")} for p, r in rec["parts"].items()}
                         | dict(factor=k, total=rec["pai_gonow_per_batch"], group_share=rec["group_share"]), indent=1))
        return
    np.savez(W5 / "R5G.npz", names=B["names"], w=w, grp=grp, nav_names=MVL["nav_names"], nav_w=MVL["nav_w"])
    r3.dump(RES / "w_R5G.json", rec | dict(base="R2JF", out="R5G"))
    r3.dump(RES / "gonow_rows.json", rec)
    f = lambda v: f"{v[1]:.2f} [{v[0]:.2f}, {v[2]:.2f}]"  # noqa: E731
    nv = rec["navtrain"]
    L = ["# R5-G: GO-NOW rows (offline, before training)", "", rec["rule"] + ".", "",
         "PAI rows: drawable = R2JF weight > 0 (outside held and the exam links); stretch rows on their own label and fed speed, LEAD flag of their "
         "base plain row (lead_rows.npz, addendum 3). `per batch` = expected rows a batch under R5G (64 pool rows; navtrain 51 rows under R3MVL's "
         "nav_w). 4 s dist = the label's 4 s path (median [p25, p75]).", "",
         "| source | GO-NOW rows | of fed < 0.5 | per batch | 4 s dist (m) | 1 s path med (m) | lead | lead share | per batch behind a lead | 4 s lead / no-lead (m) | scenes |",
         "|:--|--:|--:|--:|:--|--:|--:|--:|--:|:--|--:|"]
    for p, r in rec["parts"].items():
        L.append(f"| PAI {p} | {r['gonow']} | {r['gonow_share_of_fed_lt05']:.1%} | {r['per_batch']:.2f} | {f(r['L4'])} | {r['L1_med']:.2f} | {r['lead']} | "
                 f"{r['lead_share']:.0%} | {r['per_batch_lead']:.2f} | {r['L4_lead_med']:.2f} / {r['L4_nolead_med']:.2f} | {r['scenes']} |")
    L += [f"| navtrain GO-NOW | {nv['gonow']} | {nv['gonow_share_of_launch']:.1%} of launch rows | {nv['per_batch_gonow']:.2f} | {f(nv['L4_gonow'])} | "
          f"{nv['L1_gonow_med']:.2f} | n/a | n/a | n/a | | |",
          f"| navtrain launch rows (x 3) | {nv['launch']} | | {nv['per_batch_launch']:.2f} (R3MV {nv['per_batch_launch_r3mv']:.2f}) | {f(nv['L4_launch'])} | | n/a | | | | |",
          "", "navtrain: the cache holds no agent boxes, so no LEAD flag. PAI GO-NOW rows a batch in total: "
          f"{rec['pai_gonow_per_batch']:.2f} (R3MV / R3MVL 0).", "",
          "PAI rows fed < 2 m/s still at weight 0 (as MV): stand (fed < 0.5, 4 s path < 0.5 m) / go later (fed < 0.5, leaves 0.5 m after 1 s) / fed 0.5-2 m/s:", "",
          "| part | R2JF drawable | MV drawable | R5G drawable | stand | go later | 0.5-2 m/s |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for p, r in rec["parts"].items():
        z = r["zeroed"]
        L.append(f"| {p} | {r['drawable_r2jf']} | {r['drawable_mv']} | {r['drawable_g']} | {z['stand']} | {z['go_later']} | {z['slow_05_2']} |")
    L += ["", "Identity: the same code with no GO-NOW row reproduces R3MV.npz bit for bit; non-PAI weights unchanged; nav_names / nav_w = R3MVL's arrays."]
    (RES / "gonow_rows.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def route_lat(poly, x):
    """D1's lat_at: the route polyline's lateral offset at longitudinal x (finite points sorted by x, np.interp: flat past the ends; nan < 2 points)."""
    P = poly[np.isfinite(poly[:, 0])]
    if len(P) < 2:
        return np.nan
    i = np.argsort(P[:, 0])
    return np.interp(x, P[i, 0], P[i, 1])


def lane_change_off_route(v0, fut, route):
    """D1's rule (d1.py labels) on one pool: lane-change shape = fed v0 > LC_V, 4 s label point |y| > LC_Y, |yaw| < LC_YAW deg; it leaves the
    route lane unless the route polyline (route.npz, row frame) passes within LC_ROUTE m of the 4 s point (route not ok: undecided, kept;
    route lateral nan: leaves, as D1 counted). -> (lc, off, undecided) masks."""
    y4, x4, h4 = fut[:, 7, 1].astype(np.float64), fut[:, 7, 0].astype(np.float64), np.degrees(fut[:, 7, 2].astype(np.float64))
    lc = (np.abs(y4) > LC_Y) & (np.abs(h4) < LC_YAW) & (np.asarray(v0, np.float64) > LC_V)
    ok, poly = route["ok"].astype(bool), route["poly"]
    off = np.zeros(len(lc), bool)
    for i in np.flatnonzero(lc & ok):
        off[i] = not abs(route_lat(poly[i], x4[i]) - y4[i]) < LC_ROUTE
    return lc, off, lc & ~ok


def part_drop(w0, parts, nodrop=False):
    """w0 the base draw, parts {p: (ix, drop)} -> w: drop rows at 0, each part's remaining drawable rows rescaled to the part's base mass;
    non-part weights asserted unchanged."""
    w = w0.copy()
    for ix, drop in parts.values():
        if nodrop or not drop.any():
            continue
        sh, keep = w[ix].sum(), (w[ix] > 0) & ~drop
        w[ix[drop]] = 0.0
        w[ix[keep]] *= sh / w[ix[keep]].sum()
    other = np.ones(len(w), bool)
    other[np.concatenate([v[0] for v in parts.values()])] = False
    assert abs(w.sum() - 1) < 1e-9 and np.array_equal(w[other], w0[other]), "draw mass / non-PAI weights changed"
    return w


def cmd_lcrows(a):
    """Stage L (addendum 5): D1's lane-change-off-route rule on the three PAI human parts; R5LC / R5LCI draw files; the offline tables."""
    from collections import Counter
    O = DATA / "runs/round5/r5/L"
    O.mkdir(parents=True, exist_ok=True)
    B = np.load(W3 / "R3MV.npz")
    names, w0, grp = B["names"].astype(str), B["w"].astype(np.float64), B["grp"]
    idx = {x: i for i, x in enumerate(names)}
    M = np.load(CR / "on1_JTPH/tab.npz")
    assert np.array_equal(M["names"].astype(str), names), "R3MV rows != on1_JTPH"
    parts, rec, cls_all, lc_all, vv = {}, dict(parts={}), np.full(len(names), "", dtype="U13"), np.zeros(len(names), bool), np.zeros(len(names))
    rec["rule"] = dict(text=(f"lane-change-shaped label: fed v0 > {LC_V} m/s, the 4 s label point |y| > {LC_Y} m and |yaw| < {LC_YAW} deg (row "
                             f"frame); off the route lane: the row's route.npz polyline (lateral interpolated at the 4 s point's x, flat past its "
                             f"ends) does not pass within {LC_ROUTE} m of the 4 s point (d1.py labels' rule; route not ok -> kept; route lateral "
                             "nan -> off, as D1 counted). Every part on its own rows: own label, fed speed and route.npz (stretched rows on "
                             "their own clock)"), LC_V=LC_V, LC_Y=LC_Y, LC_YAW=LC_YAW, LC_ROUTE=LC_ROUTE)
    for p in r3.PARTS:
        t = r3.load_part(p)
        nm = t["names"].astype(str)
        rt = dict(np.load(CR / r3.PARTS[p] / "route.npz"))      # in memory: NpzFile re-reads an array on every access
        assert np.array_equal(rt["tokens"].astype(str), nm), f"{p}: route.npz rows != tab rows"
        ix = np.array([idx[f"on:{p}:{x[3:]}"] for x in nm])
        v0, fut = t["speed"].astype(np.float64), t["fut"].astype(np.float64)
        assert np.array_equal(M["speed"][ix], t["speed"]) and np.array_equal(M["fut"][ix], t["fut"]), f"{p}: merged rows != source"
        lc, off, und = lane_change_off_route(v0, fut, rt)
        cls, _ = r3.label_class(v0, fut)
        dr = w0[ix] > 0
        drop = off & dr
        parts[p] = (ix, drop)
        cls_all[ix], lc_all[ix], vv[ix] = cls, lc, v0
        mv = dr & (v0 > LC_V)
        nanr = np.array([np.isnan(route_lat(rt["poly"][i], fut[i, 7, 0])) for i in np.flatnonzero(drop)], bool)
        rec["parts"][p] = dict(rows=int(len(nm)), drawable=int(dr.sum()), moving_drawable=int(mv.sum()), lc_drawable=int((lc & dr).sum()),
                               lc_route_undecided=int((und & dr).sum()), removed=int(drop.sum()), removed_route_nan=int(nanr.sum()),
                               lc_share_moving_all_rows=float(lc[v0 > LC_V].mean()), off_share_of_lc=float(drop.sum() / max((lc & dr).sum(), 1)),
                               all_rows_lc=int(lc.sum()), all_rows_lc_route_ok=int((lc & ~und).sum()), all_rows_off=int(off.sum()),
                               removed_drawweighted=float(w0[ix][drop].sum() / w0[ix][dr].sum()),
                               by_class={c: dict(drawable=int((dr & (cls == c)).sum()), removed=int((drop & (cls == c)).sum())) for c in r3.CLASSES},
                               v0_q_removed=np.percentile(v0[drop], [10, 50, 90]).tolist() if drop.any() else None,
                               y4_abs_q_removed=np.percentile(np.abs(fut[drop, 7, 1]), [10, 50, 90]).tolist() if drop.any() else None,
                               scenes_removed=int(len(set(t["log"].astype(str)[drop]))), scenes_drawable=int(len(set(t["log"].astype(str)[dr]))))
    np.savez(O / "lc_rows.npz", **{f"{p}_{k}": v for p, (ix, drop) in parts.items() for k, v in (("ix", ix), ("drop", drop))})
    W5.mkdir(parents=True, exist_ok=True)
    for out_, nodrop in (("R5LC", False), ("R5LCI", True)):
        w = part_drop(w0, parts, nodrop)
        if nodrop:
            assert np.array_equal(w, w0), "R5LCI != R3MV"
        gs = Counter()
        for g_, x in zip(grp.astype(str), w):
            gs[g_] += x
        np.savez(W5 / f"{out_}.npz", names=B["names"], w=w, grp=grp)
        r = dict(base="R3MV", out=out_, rule=("R3MV, every PAI human row whose 4 s label is a lane change off the route lane (D1 rule) -> 0, each part "
                                              "renormalised to its R3MV share") if not nodrop else "identity: R5LC's code with no row dropped (bit-equal to R3MV)",
                 group_share={k: float(v) for k, v in sorted(gs.items()) if v > 0}, pai_drawable=int(sum((w[v[0]] > 0).sum() for v in parts.values())))
        r3.dump(RES / f"w_{out_}.json", r)
        rec[out_] = r
    wl = np.load(W5 / "R5LC.npz")["w"]
    pm = cls_all != ""
    mov = pm & (vv > LC_V)
    rec["per_batch"] = {k: dict(pai=float(r3.K_ON * ww[pm].sum()), by_class={c: float(r3.K_ON * ww[pm & (cls_all == c)].sum()) for c in r3.CLASSES},
                                lc=float(r3.K_ON * ww[pm & lc_all].sum()), lc_share_moving=float(ww[mov & lc_all].sum() / ww[mov].sum()),
                                lc_share_moving_by_part={p: float(ww[ix][lc_all[ix] & (vv[ix] > LC_V)].sum() / ww[ix][vv[ix] > LC_V].sum())
                                                         for p, (ix, _) in parts.items()},
                                unique_pai_rows=int((ww[pm] > 0).sum())) for k, ww in (("R3MV", w0), ("R5LC", wl))}
    nl = []                                                       # navtrain (uniform under both draws): D1's shape share of moving rows
    for f in sorted(CR.glob("navtrain_full.s*of12/tab.npz")):
        z = np.load(f)
        o = np.isfinite(z["fut"].reshape(len(z["fut"]), -1)).all(1)
        y4, h4, v = z["fut"][o, 7, 1], np.degrees(z["fut"][o, 7, 2]), z["speed"][o]
        nl.append(((v > LC_V).sum(), ((np.abs(y4) > LC_Y) & (np.abs(h4) < LC_YAW) & (v > LC_V)).sum()))
    nm_, nlc = map(int, np.sum(nl, 0))
    rec["navtrain"] = dict(moving=nm_, lc=nlc, lc_share_moving=nlc / nm_)
    r3.dump(RES / "lc_rows.json", rec)
    pb = rec["per_batch"]
    L = ["# R5-LC (MV-LC): lane-change-off-route PAI human rows (offline, before training)", "", rec["rule"]["text"] + ".", "",
         "R3MV draws only PAI rows fed >= 2 m/s; MV-LC sets the off-route lane-change rows among them to 0 and renormalises each part to its R3MV "
         "share (ph 0.25, ph05 0.125, ph15 0.125 of the 64 pool rows: still 32 PAI human rows a batch).", "",
         "| part | R3MV drawable | moving (> 5 m/s) | lane-change shaped | off route (removed) | of shaped | draw-weighted of part | route nan | route not ok (kept) "
         "| v0 median removed (m/s) | |y4| median removed (m) | scenes removed / drawable |", "|:--|" + "--:|" * 11]
    for p, r in rec["parts"].items():
        L.append(f"| {p} | {r['drawable']} | {r['moving_drawable']} | {r['lc_drawable']} | {r['removed']} | {r['off_share_of_lc']:.1%} | "
                 f"{r['removed_drawweighted']:.2%} | {r['removed_route_nan']} | {r['lc_route_undecided']} | {r['v0_q_removed'][1]:.1f} | "
                 f"{r['y4_abs_q_removed'][1]:.2f} | {r['scenes_removed']} / {r['scenes_drawable']} |")
    L += ["", "Removed / drawable by label class (r3.label_class; launch and stand are 0 under MV's >= 2 m/s cut):", "",
          "| part | " + " | ".join(r3.CLASSES) + " |", "|:--|" + "--:|" * len(r3.CLASSES)]
    for p, r in rec["parts"].items():
        L.append(f"| {p} | " + " | ".join(f"{r['by_class'][c]['removed']} / {r['by_class'][c]['drawable']}" for c in r3.CLASSES) + " |")
    r = rec["parts"]["ph"]
    L += ["", f"Check against D1 (all plain rows, any draw): lane-change shaped {r['all_rows_lc']} ({r['lc_share_moving_all_rows']:.1%} of moving), "
          f"with a route {r['all_rows_lc_route_ok']}, off the route lane {r['all_rows_off']} ({r['all_rows_off'] / max(r['all_rows_lc_route_ok'], 1):.0%}); "
          "D1: 10 710 (11.9 %), 10 703, 44 %."]
    L += ["", "PAI human rows a batch (64 pool rows x draw mass) by class, and the lane-change-shaped share of the drawn moving (> 5 m/s) rows "
          "(draw-weighted):", "", "| draw | PAI rows | " + " | ".join(r3.CLASSES) + " | lane-change shaped | share of moving | ph / ph05 / ph15 | unique PAI rows |",
          "|:--|--:|" + "--:|" * len(r3.CLASSES) + "--:|--:|--:|--:|"]
    for k, r in pb.items():
        L.append(f"| {k} | {r['pai']:.2f} | " + " | ".join(f"{r['by_class'][c]:.2f}" for c in r3.CLASSES) + f" | {r['lc']:.2f} | {r['lc_share_moving']:.1%} | "
                 + " / ".join(f"{r['lc_share_moving_by_part'][p]:.1%}" for p in r3.PARTS) + f" | {r['unique_pai_rows']} |")
    n = rec["navtrain"]
    L += ["", f"navtrain (navtrain_full, 12 shards, uniform in both draws): lane-change shaped {n['lc']} of {n['moving']} moving rows "
          f"({n['lc_share_moving']:.1%}). Files: lc_rows.json, w_R5LC.json, w_R5LCI.json (R5LCI = R3MV bit for bit)."]
    (RES / "lc_rows.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_lcverdict(a):
    """Stage L: the registered adopt-candidate rule on r2.py r281's json + the secondary column (d429: new zeros / partials on the scenes NTV0
    scores 1.0, over r281's common scenes) -> appended to its md."""
    import pandas as pd
    J = json.loads(Path(a.json).read_text())
    R, P = J["reads"], J["pairs"]
    lat = lambda k: R[k]["z_offroad"] + R[k]["z_zero-other"]  # noqa: E731
    cap = lat("NTV0-ag") + 3
    rows, ok = [], True
    for k in ("R5LC-s0-ag", "R5LC-s1-ag"):
        d = P[f"{k}:NTV0-ag"]["mean"]
        b, l_ = d["mean"] > 0, lat(k) <= cap
        ok &= b and l_
        rows.append(f"| {k} | {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {'yes' if b else 'no'} | {lat(k)} (offroad {R[k]['z_offroad']}, "
                    f"zero-other {R[k]['z_zero-other']}) | {'yes' if l_ else 'no'} |")
    s = P["R3MV-s1-ag:R3MV-s0-ag"]["mean"]
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in a.read)}
    com = sorted(set.intersection(*[set(X.index[X.score.notna() & ~X.rpc.fillna(False).astype(bool)]) for X in F.values()]))
    assert len(com) == J["scenes"], (len(com), J["scenes"])
    solved = (F["NTV0-ag"].score.reindex(com) >= 1 - 1e-4).to_numpy()
    sol = {}
    for k, X in F.items():
        y, c = X.score.reindex(com)[solved], X.fclass.reindex(com)[solved]
        sol[k] = dict(zeros=int((y <= 1e-4).sum()), partials=int(((y > 1e-4) & (y < 1 - 1e-4)).sum()),
                      by_class={str(q): int(n) for q, n in c[y < 1 - 1e-4].value_counts().items()})
    L = ["", "## Registered rule (prereg addendum 5)", "",
         f"adopt-candidate = both MV-LC seeds beat NTV0 on the mean (paired point estimate > 0) and each seed's offroad + zero-other zeros <= NTV0's + 3 = {cap}.", "",
         "| arm | - NTV0 mean [95 % CI] | beats NTV0 | offroad + zero-other | <= cap |", "|:--|--:|:-:|--:|:-:|", *rows, "",
         f"**MV-LC: {'adopt-candidate' if ok else 'not an adopt-candidate'}** (primary rule). Seed noise (R3MV s1 - s0, same argv): "
         f"{s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}].", "",
         f"### Secondary column (addendum 5, d429): new zeros / partials on the {int(solved.sum())} of {len(com)} common scenes NTV0 scores 1.0", "",
         "| read | new zeros | new partials | by class (zeros + partials) |", "|:--|--:|--:|:--|"]
    L += [f"| {k} | {v['zeros']} | {v['partials']} | " + ", ".join(f"{q} {n}" for q, n in sorted(v["by_class"].items(), key=lambda t: -t[1])) + " |"
          for k, v in sol.items() if k != "NTV0-ag"]
    L += ["", "The local PAI PCS ruler (the second secondary column) is appended by `r5.py lcpcs` after its CPU jobs."]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J["verdict"] = dict(adopt_candidate=bool(ok), cap=cap, ntv0_solved=dict(scenes=int(solved.sum()), reads=sol))
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def cmd_lcpcs(a):
    """Stage L: the local PAI PCS ruler (pai_pcs_local.py r5 json) per read, paired vs NTV0; the registered secondary rule -> appended to the md."""
    Q = json.loads(Path(a.pcs).read_text())
    J = json.loads(Path(a.json).read_text())
    L = ["", "### Secondary column (addendum 5, d429): local PAI PCS ruler (d392 pool / anchors, paired vs NTV0-s0+AG on the same bootstrap draws)", "",
         f"NTV0-s0+AG {Q['NTV0-s0+AG']['pcs']:.0f} [{Q['NTV0-s0+AG']['ci'][0]:.0f}, {Q['NTV0-s0+AG']['ci'][1]:.0f}].", "",
         "| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |", "|:--|--:|--:|--:|"]
    for k in ("R2C-s0+AG", "R3MV-s0+AG", "R3MV-s1+AG", "R5LC-s0+AG", "R5LC-s1+AG"):
        q = Q[k]
        L.append(f"| {k} | {q['pcs']:.0f} [{q['ci'][0]:.0f}, {q['ci'][1]:.0f}] | {q['diff_median']:+.0f} ({q['diff_point']:+.0f}) "
                 f"[{q['diff_ci'][0]:+.0f}, {q['diff_ci'][1]:+.0f}] | {q['p_pos']:.2f} |")
    sp = {k: Q[k]["diff_ci"][1] >= 0 for k in ("R5LC-s0+AG", "R5LC-s1+AG")}
    L += ["", "Registered secondary rule: each MV-LC seed's ruler difference to NTV0 has a 95 % CI upper bound >= 0 (not credibly below NTV0; "
          "R2C passed at -29 [-95, +19], R3MV failed at -276 [-375, -217], d429): " + ", ".join(f"{k} {'pass' if v else 'fail'}" for k, v in sp.items())
          + f" -> **{'both pass' if all(sp.values()) else 'not both'}**. Full ruler report: pcs_r5l.md."]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J.setdefault("verdict", {})["pcs"] = dict({k: Q[k] for k in ("NTV0-s0+AG", "R2C-s0+AG", "R3MV-s0+AG", "R3MV-s1+AG", "R5LC-s0+AG", "R5LC-s1+AG")},
                                              secondary_pass=sp)
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def cmd_newbreaks(a):
    """Any read set: the secondary column of lcverdict alone (new zeros / partials on the r281 common scenes NTV0-ag scores 1.0) -> appended to the md
    and to the json under "newbreaks" (no adopt rule)."""
    import pandas as pd
    J = json.loads(Path(a.json).read_text())
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in a.read)}
    com = sorted(set.intersection(*[set(X.index[X.score.notna() & ~X.rpc.fillna(False).astype(bool)]) for X in F.values()]))
    assert len(com) == J["scenes"], (len(com), J["scenes"])
    solved = (F["NTV0-ag"].score.reindex(com) >= 1 - 1e-4).to_numpy()
    sol = {}
    for k, X in F.items():
        y, c = X.score.reindex(com)[solved], X.fclass.reindex(com)[solved]
        sol[k] = dict(zeros=int((y <= 1e-4).sum()), partials=int(((y > 1e-4) & (y < 1 - 1e-4)).sum()),
                      by_class={str(q): int(n) for q, n in c[y < 1 - 1e-4].value_counts().items()})
    L = ["", f"## New zeros / partials on the {int(solved.sum())} of {len(com)} common scenes NTV0 scores 1.0", "",
         "| read | new zeros | new partials | by class (zeros + partials) |", "|:--|--:|--:|:--|"]
    L += [f"| {k} | {v['zeros']} | {v['partials']} | " + ", ".join(f"{q} {n}" for q, n in sorted(v["by_class"].items(), key=lambda t: -t[1])) + " |"
          for k, v in sol.items() if k != "NTV0-ag"]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J["newbreaks"] = dict(scenes=int(solved.sum()), reads=sol)
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def cmd_pcsrow(a):
    """Any read set: rows of the local PAI PCS ruler (pai_pcs_local.py r5 json) for the named subjects, paired vs NTV0-s0+AG -> appended to the md / json."""
    Q = json.loads(Path(a.pcs).read_text())
    J = json.loads(Path(a.json).read_text())
    L = ["", "## Local PAI PCS ruler (d392 pool / anchors, out-of-pool read, paired vs NTV0-s0+AG on the same bootstrap draws)", "",
         f"NTV0-s0+AG {Q['NTV0-s0+AG']['pcs']:.0f} [{Q['NTV0-s0+AG']['ci'][0]:.0f}, {Q['NTV0-s0+AG']['ci'][1]:.0f}].", "",
         "| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |", "|:--|--:|--:|--:|"]
    for k in a.subject:
        q = Q[k]
        L.append(f"| {k} | {q['pcs']:.0f} [{q['ci'][0]:.0f}, {q['ci'][1]:.0f}] | {q['diff_median']:+.0f} ({q['diff_point']:+.0f}) "
                 f"[{q['diff_ci'][0]:+.0f}, {q['diff_ci'][1]:+.0f}] | {q['p_pos']:.2f} |")
    L += ["", f"Reference points (d429): R2C-s0+AG -29 [-95, +19], R3MV-s0+AG -276 [-375, -217]. Full ruler report: {a.report}."]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J.setdefault("pcs", {}).update({k: Q[k] for k in ["NTV0-s0+AG", *a.subject]})
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def cmd_l37(a):
    """LAUNCH-37 read-out (addendum 4). The registered comparison is the first --pair (LAUNCH-37) and the first --target (TARGET)."""
    import pandas as pd
    sys.path[:0] = [str(REPO / "experiments/pai_fix/scripts")]
    import pai_fix as PF
    from jevdrive import stats
    D = pd.read_csv(RES / "lead_281.csv")
    la = D[D.kind == "launch"].set_index("scene")
    sc, lead = la.index, la.lead.astype(bool).to_numpy()
    R, rec = {}, dict(scenes=len(sc), lead=int(lead.sum()), reads={}, pairs={}, target={})
    L = [f"# LAUNCH-37: the {len(sc)} PAI 281 launch scenes, ACC + GAP (g10, FAST=1, CONC 4)", "",
         f"Scenes: lead_281.csv launch rows (NTV0's launch scenes; LEAD rule at the human's t_L: {int(lead.sum())} lead, {int((~lead).sum())} no lead). "
         "Stored NTV0 / R2C / R2J = their PAI 281 ACC + GAP reads restricted to these scenes (not re-read); the others read LAUNCH-37 alone "
         "(the same 4 chunk lists). launched = pai_fix.py launched (dist >= 2 m and >= 0.3 x the log's); scenes with an RPC error are left out.", "",
         "| read | n | launched | rear-hit | mean | zeros | launched lead / no-lead | rear-hit lead / no-lead | mean lead / no-lead |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in a.read:
        k, f = r.split("=", 1)
        x = pd.read_csv(f).set_index("scene").reindex(sc)
        ok = (x.score.notna() & ~x.rpc.fillna(False).astype(bool)).to_numpy()
        lz = PF.launched(x).fillna(False).astype(bool).to_numpy() & ok
        rh = (x.fclass == "rear-hit").to_numpy() & ok
        s = np.where(ok, x.score.to_numpy(float), np.nan)
        R[k] = dict(s=s, lz=lz, ok=ok)
        m = dict(n=int(ok.sum()), launched=int(lz.sum()), rear_hit=int(rh.sum()), mean=float(np.nanmean(s)), zeros=int((s == 0).sum()),
                 launched_lead=int((lz & lead).sum()), launched_nolead=int((lz & ~lead).sum()), rear_lead=int((rh & lead).sum()),
                 rear_nolead=int((rh & ~lead).sum()), mean_lead=float(np.nanmean(s[lead])), mean_nolead=float(np.nanmean(s[~lead])))
        rec["reads"][k] = m
        L.append(f"| {k} | {m['n']} | {m['launched']} | {m['rear_hit']} | {m['mean']:.3f} | {m['zeros']} | {m['launched_lead']} / {m['launched_nolead']} | "
                 f"{m['rear_lead']} / {m['rear_nolead']} | {m['mean_lead']:.3f} / {m['mean_nolead']:.3f} |")
    L += ["", "## Paired over LAUNCH-37 (A - B; scenes both reads scored)", "",
          "| pair | n | mean (95 % CI) | launched (count, 95 % CI) | launched A only / B only | mean lead / no-lead |", "|:--|--:|:--|:--|--:|:--|"]
    for pr in a.pair:
        A, B = pr.split(":")
        m = R[A]["ok"] & R[B]["ok"]
        u, v = R[A]["s"][m], R[B]["s"][m]
        dm, n = stats.paired(u, v), int(m.sum())
        dl = stats.paired(R[A]["lz"][m].astype(float), R[B]["lz"][m].astype(float))
        ld, nl = lead[m], ~lead[m]
        rec["pairs"][pr] = dict(n=n, mean=dm, launched=dict(diff=n * dl["mean"], lo=n * dl["lo"], hi=n * dl["hi"]),
                                a_only=int((R[A]["lz"][m] & ~R[B]["lz"][m]).sum()), b_only=int((~R[A]["lz"][m] & R[B]["lz"][m]).sum()),
                                mean_lead=float((u - v)[ld].mean()), mean_nolead=float((u - v)[nl].mean()))
        p = rec["pairs"][pr]
        L.append(f"| {A} - {B} | {n} | {stats.fmt(dm, '+.3f')} | {p['launched']['diff']:+.0f} [{p['launched']['lo']:+.0f}, {p['launched']['hi']:+.0f}] | "
                 f"{p['a_only']} / {p['b_only']} | {p['mean_lead']:+.3f} / {p['mean_nolead']:+.3f} |")
    if a.target:
        E = pd.read_csv(REPO / "experiments/pai_fix/scenes/exam.csv").set_index("scene")
        FA = DATA / "runs/pai_fix/facts"
        L += ["", "## Paired over PAI-FIX TARGET (val160 STOP + STOP-L + TURN, native VC + LEAD reads; scenes both scored without RPC)", "",
              "| pair | n | A / B | mean (95 % CI) | zeros A / B (new / fixed) |", "|:--|--:|:--|:--|:--|"]
        for pr in a.target:
            A, B = pr.split(":")
            F, C = (pd.read_csv(FA / f"{t}-vl.csv").set_index("scene").reindex(E.index) for t in (A, B))
            m = (E.subset.isin(["STOP", "STOP-L", "TURN"]) & F.score.notna() & C.score.notna() & ~F.rpc.fillna(False).astype(bool)
                 & ~C.rpc.fillna(False).astype(bool))
            u, v = F.score[m].to_numpy(float), C.score[m].to_numpy(float)
            d = stats.paired(u, v)
            rec["target"][pr] = dict(n=int(m.sum()), a=float(u.mean()), b=float(v.mean()), mean=d, zeros=(int((u == 0).sum()), int((v == 0).sum())),
                                     new=int(((u == 0) & (v > 0)).sum()), fixed=int(((u > 0) & (v == 0)).sum()))
            t = rec["target"][pr]
            L.append(f"| {A} - {B} | {t['n']} | {t['a']:.3f} / {t['b']:.3f} | {stats.fmt(d, '+.3f')} | {t['zeros'][0]} / {t['zeros'][1]} ({t['new']} / {t['fixed']}) |")
    excl = lambda d: bool(d["lo"] > 0 or d["hi"] < 0)  # noqa: E731
    rec["separated"] = {}
    for e in a.sep or ([f"{a.pair[0]}={a.target[0]}"] if a.pair and a.target else []):
        pr, tg = e.split("=")
        p0, t0 = rec["pairs"][pr]["mean"], rec["target"][tg]["mean"]
        s = rec["separated"][pr] = dict(target=tg, launch37=excl(p0), target_=excl(t0), any=excl(p0) or excl(t0))
        L += ["", f"**Registered comparison {pr.replace(':', ' vs ')} (addendum 4 / 4b): separated = {'yes' if s['any'] else 'no'}** (paired CI "
              f"excludes 0 on LAUNCH-37: {'yes' if s['launch37'] else 'no'}, on TARGET: {'yes' if s['target_'] else 'no'})."]
    r3.dump(RES / f"{a.stem}.json", rec)
    (RES / f"{a.stem}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_gate(a):
    import pandas as pd
    r3.RES = RES                                                  # round 3's gate, written into round 5's results
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
    p.add_argument("--tag", required=True), p.add_argument("--arm", choices=("R", "A", "N", "G", "L"), required=True)
    p.add_argument("--wfile", default="R3MV"), p.add_argument("--index", default="road_on1_JTPH.npz"), p.add_argument("--pool", default="on1_JTPHO")
    p.add_argument("--k-s3", type=int, default=9), p.add_argument("--s3-hinged", type=float, default=0.0, help="road-hinged share of the S3 rows")
    p.add_argument("--suffix", default="")
    p = sub.add_parser("weights")
    p.add_argument("--base", default="R3MV"), p.add_argument("--pool", default="on1_JTPHO"), p.add_argument("--s3", default="on1_PAIO")
    p.add_argument("--k-on", type=int, default=64), p.add_argument("--k-s3", type=int, default=9), p.add_argument("--out", default="R5A")
    p = sub.add_parser("gate")
    p.add_argument("--arms", nargs="+", required=True), p.add_argument("--ctrl", default="R2C-L18-s0"), p.add_argument("--stem", default="gate_r5r")
    sub.add_parser("leadrows"), sub.add_parser("lead281")
    p = sub.add_parser("gonow")
    p.add_argument("--go-batch", type=float, default=0.0), p.add_argument("--go-drop", nargs="*", default=[]), p.add_argument("--out", default="R5G")
    p = sub.add_parser("split281")
    p.add_argument("--read", nargs="+", required=True), p.add_argument("--out", required=True)
    p = sub.add_parser("l37")
    p.add_argument("--read", nargs="+", required=True), p.add_argument("--pair", nargs="*", default=[]), p.add_argument("--target", nargs="*", default=[])
    p.add_argument("--sep", nargs="*", default=[]), p.add_argument("--stem", default="launch37")
    sub.add_parser("lcrows")
    p = sub.add_parser("lcverdict")
    p.add_argument("--json", required=True), p.add_argument("--md", required=True), p.add_argument("--read", nargs="+", required=True, help="K=FACTS.csv, r281's reads")
    p = sub.add_parser("lcpcs")
    p.add_argument("--pcs", required=True), p.add_argument("--json", required=True), p.add_argument("--md", required=True)
    p = sub.add_parser("newbreaks")
    p.add_argument("--json", required=True), p.add_argument("--md", required=True), p.add_argument("--read", nargs="+", required=True, help="K=FACTS.csv, r281's reads")
    p = sub.add_parser("pcsrow")
    p.add_argument("--pcs", required=True), p.add_argument("--json", required=True), p.add_argument("--md", required=True)
    p.add_argument("--subject", nargs="+", required=True), p.add_argument("--report", default="pcs_report.md")
    a = ap.parse_args()
    dict(expect=cmd_expect, smoke=cmd_smoke, weights=cmd_weights, gate=cmd_gate, leadrows=cmd_leadrows, lead281=cmd_lead281,
         split281=cmd_split281, gonow=cmd_gonow, l37=cmd_l37, lcrows=cmd_lcrows, lcverdict=cmd_lcverdict, lcpcs=cmd_lcpcs,
         newbreaks=cmd_newbreaks, pcsrow=cmd_pcsrow)[a.cmd](a)


if __name__ == "__main__":
    main()
