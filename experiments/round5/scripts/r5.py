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
    elif a.arm == "N":                                             # R5-N (addendum 3): R3MV's argv, draw R5N (lead rows -> 0)
        ex = json.loads((RES / f"w_{a.wfile}.json").read_text())
        k_both = 32
        b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", (d / "log.txt").read_text())
        ck |= dict(no_road_pai=(m is not None, False, m is None),
                   batch_rows=(b.groups() if b else None, ("51", "13", str(r3.K_ON)), bool(b) and b.groups() == ("51", "13", str(r3.K_ON))),
                   pool_imit_rows_without_hinge=(max(S.get("loss/on/n_hinge", [np.nan])), 0, max(S.get("loss/on/n_hinge", [1])) == 0),
                   agent_ho_zero=(max(S.get("loss/on/agent_ho", [0])), 0, max(S.get("loss/on/agent_ho", [0])) == 0))
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
    p.add_argument("--tag", required=True), p.add_argument("--arm", choices=("R", "A", "N"), required=True)
    p.add_argument("--wfile", default="R3MV"), p.add_argument("--index", default="road_on1_JTPH.npz"), p.add_argument("--pool", default="on1_JTPHO")
    p.add_argument("--k-s3", type=int, default=9), p.add_argument("--s3-hinged", type=float, default=0.0, help="road-hinged share of the S3 rows")
    p.add_argument("--suffix", default="")
    p = sub.add_parser("weights")
    p.add_argument("--base", default="R3MV"), p.add_argument("--pool", default="on1_JTPHO"), p.add_argument("--s3", default="on1_PAIO")
    p.add_argument("--k-on", type=int, default=64), p.add_argument("--k-s3", type=int, default=9), p.add_argument("--out", default="R5A")
    p = sub.add_parser("gate")
    p.add_argument("--arms", nargs="+", required=True), p.add_argument("--ctrl", default="R2C-L18-s0"), p.add_argument("--stem", default="gate_r5r")
    sub.add_parser("leadrows"), sub.add_parser("lead281")
    p = sub.add_parser("split281")
    p.add_argument("--read", nargs="+", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(expect=cmd_expect, smoke=cmd_smoke, weights=cmd_weights, gate=cmd_gate, leadrows=cmd_leadrows, lead281=cmd_lead281,
         split281=cmd_split281)[a.cmd](a)


if __name__ == "__main__":
    main()
