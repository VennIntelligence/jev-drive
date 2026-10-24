#!/usr/bin/env python3
"""ROUND9 CB2 (clean B2; plans/2026-10-25-cb2.md = the registration; lane scripts/cb2.sh). Box, repo root, repo venv unless noted.
  creep --pools N.. --teach-dir D --out P      step 0 (offline, before registration): b3 DAgger candidates (k >= 2, fed >= 2 m/s, teacher label)
                                               on pools whose rows exist; how many are human-standing / human-stopping, how far the b3 label's
                                               longitudinal (= the student's served arc) goes past the STAND cap, the rule candidates R1 / R2
                                               and the offline gate G0 -> P.{json,md}
  c0chk --new D --ref D --out J                 a roll's c0 configs vs the reference roll's c0 (= snu.py c0chk's rule), every listed scene scored
  pgate --name N --ref N0 --out J              poolgate.py + gates.json R1 / R2 of an onpol-train pool (= snu.py pgate)
  wfile                                        CB2S (rb0 live + pr live moving S rows, uniform) and CB2S0 (= JT00S-rb0 + pr at 0) on on1_JTPR
  argv ARM TAG [--steps N] [--on-start T] [--save-at S|none] [--seed K]   P1LZ (P1L on on1_JTPR / CB2S0) | P1N | P1NZ (P1N on on1_JTPRD / CB2D0) | D1N
  diff                                         argv token diffs (P1N vs PCL1L-F-s0, D1N vs P1N); exit 2 beyond the registered words
  dlabel --name N --teach F --res R            R3 on the pool's drawable DAgger rows (dagger.npz from nd_teach label), tab.fut written (nd_teach --write
                                               semantics), LABELLED; G0 re-checked on these rows -> R/dlabel_<N>.json (exit 2 = G0 fails)
  dweights --pools P..                         CB2D (CB2S x 32 / 40 + drawable DAgger rows 8 / 40) and CB2D0 on on1_JTPRD
  smoke --tag T --wfile W --k K --out J        60-step smoke: DONE, finite, batch rows, draw shares = the file's, lon_stand / lon_free present
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/round9/scripts")]
from jevdrive.common import data_dir  # noqa: E402

DD = data_dir()
O = DD / "runs/round9/cb2"
WD, WREL = O / "w", "runs/round9/cb2/w"
P1L, B2 = "PCL1L-F-s0", "STANDFMXH-L18-s1"
STAND = ["--on-stand", "0.0068", "--on-stand-margin", "1.0", "--on-free", "0.001", "--on-free-mode", "move", "--on-free-frac", "0.95"]
import os  # noqa: E402
VAR = os.environ.get("CB2V", "")                 # plan amendment 2: "" = P1N (static practice rows dropped), "B" = P1N' (B2's S-row rule)
POOL1, POOL2 = "on1_JTPR", f"on1_JTPRD{VAR}"
SW, DW, DW0 = f"CB2S{VAR}", f"CB2D{VAR}", f"CB2D0{VAR}"
TP1, TD1, TD11 = f"CB2P1N{VAR}-F-s0", f"CB2D1N{VAR}-F-s0", f"CB2D1N{VAR}-F-s1"
K_S, K_D = 32, 8
CR, ROWS = DD / "runs/op_parity/cache", DD / "runs/onpol_nuplan/rows"
V_ROW, K_MIN = 2.0, 2                      # nd_teach.py's drawable rule
H_STAND, V_REST, MARGIN = 0.5, 0.5, 1.0    # nr1_free.py: human 4 s arc < 0.5 m = standing; end speed < 0.5 m/s = at rest by 4 s; STAND margin
LEAD_Y, LEAD_X = 2.0, 15.0                 # a logged vehicle at t0 in the corridor |y| < 2 m, 0 < x < stop arc + 15 m = a lead
HOLD_PCT = 15
# G0 (registered with the rule, plan section 2): a rule passes when on every proxy pool it keeps >= 80 % of the candidates, >= 80 % of the
# lateral correction mass (sum |label - served| lateral at 4 s) and >= 80 % of the rows with a lateral correction > 0.5 m, and leaves <= 10 %
# of the conflict excess (sum relu(label 4 s arc - STAND cap) on human-standing / -stopping rows) in the kept rows
G0_KEEP, G0_LAT, G0_CONF = 0.80, 0.80, 0.10


def cum_arc(P):
    P = np.concatenate([np.zeros((len(P), 1, 2)), np.asarray(P, np.float64)[..., :2]], 1)
    return np.cumsum(np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)), 1)


def replace_at(lab, arcs):
    """R3: a label polyline (n, 8, 3; row frame, from the origin) re-sampled at new arc lengths (n, 8) along its own chord length; past
    its last pose straight along the last pose's heading; yaw interpolated (unwrapped). The path (lateral shape) is unchanged."""
    lab = np.asarray(lab, np.float64)
    P = np.concatenate([np.zeros((len(lab), 1, 3)), lab], 1)
    P[:, :, 2] = np.unwrap(P[:, :, 2], axis=1)
    s = np.concatenate([np.zeros((len(lab), 1)), np.cumsum(np.hypot(*np.diff(P[:, :, :2], axis=1).transpose(2, 0, 1)), 1)], 1)
    out = np.empty_like(lab)
    for i in range(len(lab)):
        t, si = np.asarray(arcs[i], np.float64), s[i]
        for d in range(3):
            out[i, :, d] = np.interp(t, si, P[i, :, d])
        ex = t > si[-1]
        if ex.any():
            h = P[i, -1, 2]
            out[i, ex, 0] = P[i, -1, 0] + (t[ex] - si[-1]) * np.cos(h)
            out[i, ex, 1] = P[i, -1, 1] + (t[ex] - si[-1]) * np.sin(h)
            out[i, ex, 2] = h
    return out


def creep_pool(name, teach):
    tb, z = np.load(CR / f"on1_{name}/tab.npz"), np.load(ROWS / name / "meta.npz")
    names = tb["names"].astype(str)
    assert np.array_equal(z["names"].astype(str), names), f"{name}: rows meta != tab"
    T = np.load(teach)
    assert np.array_equal(T["names"].astype(str), names), f"{name}: teacher rows != tab"
    v0, k, sv = tb["speed"].astype(np.float64), z["k"], tb["served"].astype(np.float64)
    cand = T["ok"] & (v0 >= V_ROW) & (k >= K_MIN)
    a_lab = cum_arc(sv)[:, 7]                                                   # b3: the label sits at the served plan's arc lengths
    fr = tb["fut_reg"].astype(np.float64)
    a_h, a_f = cum_arc(fr)[:, 7], cum_arc(tb["fut"])[:, 7]
    v_end = np.hypot(*(fr[:, 7, :2] - fr[:, 6, :2]).T) / 0.5
    hs = a_h < H_STAND
    hp = ~hs & (v_end < V_REST)
    exc = np.maximum(a_lab - (np.maximum(a_f, a_h) + MARGIN), 0)                # beyond the STAND cap max(his arcs) + 1 m at 4 s
    A = np.load(CR / f"on1_{name}/agent.npz", mmap_mode="r")
    assert np.array_equal(A["tokens"].astype(str), names)
    bx, ok_a = np.asarray(A["box"][:, 0, :, :2]), np.asarray(A["valid"][:, 0]) & (np.asarray(A["cls"]) == 0)
    reach = np.maximum(a_lab, a_h)[:, None] + LEAD_X
    lead = (ok_a & (bx[..., 0] > 0) & (bx[..., 0] < reach) & (np.abs(bx[..., 1]) < LEAD_Y)).any(1)
    lat = np.abs(np.nan_to_num(T["fut"][:, 7, 1]) - sv[:, 7, 1])
    logs = tb["log"].astype(str)
    held = np.array([zlib.crc32(g.encode()) % 100 < HOLD_PCT for g in logs])
    c = cand & ~held                                                            # the rows a draw file could use
    hsp = hs | hp
    conf = c & hsp & (exc > 0)
    rules = dict(R1=hs, R2=hsp, R3=None)
    fut_t = np.nan_to_num(T["fut"]).astype(np.float64)
    j = np.flatnonzero(c & hsp)
    lab3 = fut_t.copy()
    lab3[j] = replace_at(fut_t[j], cum_arc(tb["fut"][j]))                       # R3: HS / HP rows re-placed at the reachable human profile
    lat3 = np.abs(lab3[:, 7, 1] - sv[:, 7, 1])
    exc3 = np.maximum(cum_arc(lab3)[:, 7] - (np.maximum(a_f, a_h) + MARGIN), 0)
    out = dict(pool=name, rows=len(names), candidates=int(c.sum()), human_standing=int((c & hs).sum()), human_stopping=int((c & hp).sum()),
               standing_no_lead=int((c & hs & ~lead).sum()), stopping_no_lead=int((c & hp & ~lead).sum()),
               standing_share=float((c & hs).sum() / max(c.sum(), 1)), stopping_share=float((c & hp).sum() / max(c.sum(), 1)),
               conflict_rows=int(conf.sum()), conflict_rows_no_lead=int((conf & ~lead).sum()), conflict_excess_m=float(exc[conf].sum()),
               conflict_excess_p50_p90=[float(np.percentile(exc[conf], q)) for q in (50, 90)] if conf.any() else None,
               served_vs_human_4s_on_hsp_p50_p90=[float(np.percentile((a_lab - a_h)[c & hsp], q)) for q in (50, 90)] if (c & hsp).any() else None,
               ego_speed_on_hsp_p50=float(np.median(v0[c & hsp])) if (c & hsp).any() else None,
               ego_minus_human_station_on_hsp_p10_p50_p90=[float(np.nanpercentile(z["lead"][c & hsp], q)) for q in (10, 50, 90)] if (c & hsp).any() else None,
               moving_rows_faster_than_cap=int((c & ~hsp & (exc > 0)).sum()),
               lat_mass=float(lat[c].sum()), lat_rows_05=int((c & (lat > 0.5)).sum()), rules={})
    for r, m in rules.items():
        kp, lt, ex = (c & ~m, lat, exc) if m is not None else (c, lat3, exc3)
        g = dict(kept=int(kp.sum()), keep_share=float(kp.sum() / max(c.sum(), 1)), lat_mass_kept=float(lt[kp].sum() / max(lat[c].sum(), 1e-9)),
                 lat_rows_05_kept=float((kp & (lt > 0.5)).sum() / max((c & (lat > 0.5)).sum(), 1)),
                 conflict_excess_left=float(ex[kp & conf].sum() / max(exc[conf].sum(), 1e-9)) if conf.any() else 0.0)
        if m is None:
            g["arc_drop_on_hsp_p50_p90"] = [float(np.percentile((a_lab - cum_arc(lab3)[:, 7])[j], q)) for q in (50, 90)] if len(j) else None
        g["pass_"] = bool(g["keep_share"] >= G0_KEEP and g["lat_mass_kept"] >= G0_LAT and g["lat_rows_05_kept"] >= G0_LAT and g["conflict_excess_left"] <= G0_CONF)
        out["rules"][r] = g
    return out


def cmd_creep(a):
    res = [creep_pool(p, Path(a.teach_dir) / f"{p}.npz") for p in a.pools]
    pick = next((r for r in ("R1", "R2", "R3") if all(x["rules"][r]["pass_"] for x in res)), None)
    rec = dict(pools=res, gate=dict(keep=G0_KEEP, lat=G0_LAT, conflict_left=G0_CONF), pick=pick)
    Path(a.out + ".json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    L = ["# CB2 step 0: b3 DAgger candidates on human-standing / -stopping states (cb2.py creep)", "",
         "Candidates = k >= 2, fed >= 2 m/s, teacher label, not in the 15 % held-out logs. HS = the human's logged 4 s arc < 0.5 m; HP = not HS and "
         "his end speed < 0.5 m/s (stops within 4 s). Conflict = HS / HP rows whose b3 label (longitudinal = the served arc) ends beyond the STAND cap "
         "max(his arcs) + 1 m. R1 drops HS rows, R2 drops HS + HP rows, R3 keeps every row and re-places the label of HS + HP rows at the arc "
         "lengths of tab `fut` (the human's speed profile made reachable from the visited state, nr1_lon), path unchanged. Lead = a logged vehicle at t0 within |y| < 2 m ahead (no lead = line stop: red light / stop sign proxy).", "",
         "| pool | cand | HS | HP | HS no lead | HP no lead | conflict rows (no lead) | excess m p50 / p90 | served - human 4 s p50 / p90 on HS+HP | "
         "R1 keep / lat / lat>0.5 / conflict left | R2 (same) | R3 (same) |", "|:--|" + "--:|" * 11]
    f = lambda g: f"{g['keep_share']:.3f} / {g['lat_mass_kept']:.3f} / {g['lat_rows_05_kept']:.3f} / {g['conflict_excess_left']:.3f} {'PASS' if g['pass_'] else 'FAIL'}"  # noqa: E731
    for x in res:
        q = lambda v: "-" if v is None else " / ".join(f"{t:.2f}" for t in v)  # noqa: E731
        L.append(f"| {x['pool']} | {x['candidates']} | {x['human_standing']} | {x['human_stopping']} | {x['standing_no_lead']} | {x['stopping_no_lead']} | "
                 f"{x['conflict_rows']} ({x['conflict_rows_no_lead']}) | {q(x['conflict_excess_p50_p90'])} | {q(x['served_vs_human_4s_on_hsp_p50_p90'])} | "
                 f"{f(x['rules']['R1'])} | {f(x['rules']['R2'])} | {f(x['rules']['R3'])} |")
    L += ["", f"G0 lines: keep >= {G0_KEEP}, lateral mass and rows > 0.5 m kept >= {G0_LAT}, conflict excess left <= {G0_CONF}. "
          f"Simplest rule passing on every pool: **{pick or 'none'}**."]
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- roll / pool checks
def cmd_c0chk(a):
    import snu as S
    new, ref = Path(a.new), Path(a.ref)
    rec = dict(new=str(new), ref=str(ref), same={}, loose={})
    for f in S.CFG_SAME:
        rec["same"][f] = (new / f).read_bytes() == (ref / f).read_bytes()
    for f in S.CFG_LOOSE:
        x, y = ((d / f).read_text().replace(str(d), "<RUN>").splitlines() for d in (ref, new))
        ch = [ln for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes() if op != "equal" for ln in x[i1:i2] + y[j1:j2]]
        bad = [ln for ln in ch if not S.CFG_OK.match(ln)]
        rec["loose"][f] = dict(changed_lines=len(ch), bad=bad[:20])
    lst = Path(a.list).read_text().split()
    R = json.loads((new / "aggregate/results-summary.json").read_text())["rollouts"]
    sc = {r["clipgt_id"]: r.get("score") for r in R}
    rec |= dict(listed=len(lst), rollouts=len(R), listed_scored=sum(sc.get(x) is not None for x in lst), sim_pkl=(new / "onpol_sim.pkl").exists(),
                mean=float(np.mean([v for v in sc.values() if v is not None])), fail=dict(Counter(str(r.get("failure_reason")) for r in R)))
    ok = all(rec["same"].values()) and not any(v["bad"] for v in rec["loose"].values()) and rec["listed_scored"] == len(lst) == len(R) and rec["sim_pkl"]
    rec["pass_"] = bool(ok)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))
    sys.exit(0 if ok else 2)


def cmd_pgate(a):
    W = DD / "runs/onpol_long"
    AP = DD / "third_party/alpasim/.venv/bin/python"
    p = subprocess.run([str(AP), str(DD / "runs/route_follow/poolgate.py"), a.name], capture_output=True, text=True)
    pg = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else None
    g = json.loads((W / f"rows/{a.name}/gates.json").read_text()) if (W / f"rows/{a.name}/gates.json").exists() else {}
    f = g.get("fail", ["no gates.json"])
    r12 = [x for x in f if x.startswith(("R1", "R2"))] + (["no gates.json"] if not g else [])
    rec = dict(poolgate=pg, rc=p.returncode, R1=g.get("R1"), R2=g.get("R2"), fail=f, r1_r2_fail=r12, pass_=bool(p.returncode == 0 and pg and pg.get("pass") and not r12))
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))
    sys.exit(0 if rec["pass_"] else 2)


# ---------------------------------------------------------------- draw files
def held_logs(logs):
    return {g for g in set(logs) if zlib.crc32(g.encode()) % 100 < HOLD_PCT}


def cmd_wfile(a):
    tb = np.load(CR / POOL1 / "tab.npz")
    names, part, mode, logs = tb["names"].astype(str), tb["pool"].astype(str), tb["mode"].astype(str), tb["log"].astype(str)
    ref = np.load(DD / "runs/round9/pool_pilot/w/JT00S-rb0.npz")
    n0 = len(ref["names"])
    assert np.array_equal(names[:n0], ref["names"].astype(str)), "on1_JTPR prefix != on1_JT"
    assert (part[n0:] != "").all() and all(p.startswith("pr") for p in set(part[n0:])), sorted(set(part[n0:]))
    w0 = np.zeros(len(names))
    w0[:n0] = ref["w"]
    g0 = np.concatenate([ref["grp"].astype("U12"), np.full(len(names) - n0, "pr0w", "U12")])
    hl = held_logs(logs[mode == "self"])
    held = np.isin(logs, sorted(hl))
    rb0 = w0 > 0
    assert (part[rb0] == "rb0").all() and rb0.sum() == 13793 and not held[rb0].any()
    pr = (np.arange(len(names)) >= n0) & (mode == "self") & ~held
    pr_mov, pr_st = pr & (tb["speed"] >= 0.5), pr & (tb["speed"] < 0.5)
    s = rb0 | (pr if VAR == "B" else pr_mov)                                    # B: JTX00's S rule, every live self row equal
    fr = tb["fut_reg"].astype(np.float64)
    stand = (mode == "self") & (tb["speed"] < 0.5) & (cum_arc(np.nan_to_num(fr))[:, 7] < H_STAND)
    w = np.where(s, 1.0 / s.sum(), 0.0)
    if VAR == "B":
        grp = np.where(rb0, "S", np.where(pr, "S_pr", np.where(held & (mode == "self"), "S_held", ""))).astype("U12")
    else:
        grp = np.where(rb0, "S", np.where(pr_mov, "S_pr", np.where(pr_st, "S_pr_static0", np.where(held & (mode == "self"), "S_held", "")))).astype("U12")
    WD.mkdir(parents=True, exist_ok=True)
    if not VAR:
        np.savez(WD / "CB2S0.npz", names=tb["names"], w=w0, grp=g0)
    assert not (WD / f"{SW}.npz").exists(), f"{SW} exists"
    np.savez(WD / f"{SW}.npz", names=tb["names"], w=w, grp=grp)
    # STAND-firing rows (visited < 0.5 m/s, human 4 s arc < 0.5 m) in this draw vs B2's JTX00 (plan amendment 2: report)
    tx, wx = np.load(CR / "on1_JTX/tab.npz"), np.load(DD / "runs/defect_fix/w/JTX00.npz")
    sx = (tx["mode"].astype(str) == "self") & (tx["speed"] < 0.5) & (cum_arc(np.nan_to_num(tx["fut_reg"].astype(np.float64)))[:, 7] < H_STAND)
    mx = (tx["mode"].astype(str) == "self")
    stand_rec = dict(this_share_of_S=float(w[stand].sum() / w[s].sum()), this_per_batch=float(K_S * w[stand].sum() / w[s].sum()),
                     b2_share_of_S=float(wx["w"][sx].sum() / wx["w"][mx].sum()), b2_per_batch=float(32 * wx["w"][sx].sum() / wx["w"][mx].sum()),
                     this_by_part={p: float(w[stand & (part == p)].sum() / w[s].sum()) for p in sorted(set(part[stand])) if w[stand & (part == p)].sum() > 0})
    rec = dict(pool=POOL1, rows=len(names), rb0=int(rb0.sum()), pr_live=int(pr.sum()), pr_moving=int(pr_mov.sum()), pr_static_dropped=int(pr_st.sum()),
               pr_held=int(((np.arange(len(names)) >= n0) & held).sum()), pr_share=float(w[pr_mov].sum()),
               pr_parts={p: int((pr_mov & (part == p)).sum()) for p in sorted(set(part[n0:]))},
               draws_per_row_9500=float(K_S * 9500 / s.sum()), cb2s0_equals_jt00s_rb0=bool(np.array_equal(w0[:n0], ref["w"]) and not w0[n0:].any()),
               pr_static_kept=int((pr_st & s).sum()), stand=stand_rec, variant=VAR or "nostatic")
    (WD / f"{SW}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- argv
def stored(tag):
    import pcl as C
    return C.stored_argv(tag)


def build(arm, tag, steps=None, on_start=None, save_at=None, seed=None):
    import pcl as C
    a = list(stored(P1L))
    C.put(a, "--tag", tag)
    if arm in ("P1LZ", "P1N", "P1NZ", "D1N"):
        C.put(a, "--on", POOL1)
        C.put(a, "--on-wfile", f"{WREL}/{'CB2S0' if arm == 'P1LZ' else SW}.npz")
    if arm in ("P1N", "P1NZ", "D1N"):
        a += STAND
    if arm in ("P1NZ", "D1N"):
        C.put(a, "--on", POOL2)
        C.put(a, "--on-wfile", f"{WREL}/{DW0 if arm == 'P1NZ' else DW}.npz")
    if arm == "D1N":
        C.put(a, "--on-k", str(K_S + K_D))
    for k, v in (("--steps", steps), ("--on-start", on_start), ("--lr2-at", on_start), ("--seed", seed)):
        if v is not None:
            C.put(a, k, str(v))
    if save_at is not None:
        if save_at == "none":
            i = a.index("--save-at")
            del a[i:i + 2]
        else:
            C.put(a, "--save-at", str(save_at))
    return a


def cmd_argv(a):
    print("\n".join(build(a.arm, a.tag, a.steps, a.on_start, a.save_at, a.seed)))


def tokdiff(x, y):
    return [(op, x[i1:i2], y[j1:j2]) for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes() if op != "equal"]


def _p1n0():
    import pcl as C
    a = build("P1N", "CB2P1N-F-s0")
    C.put(a, "--on-wfile", f"{WREL}/CB2S.npz")
    return a


def cmd_diff(a):
    b = stored(P1L)
    p1n, d1n = build("P1N", TP1), build("D1N", TD1)
    d1, d2 = tokdiff(b, p1n), tokdiff(p1n, d1n)
    want1 = [("replace", ["PCL1L-F-s0"], [TP1]), ("replace", ["on1_JT"], [POOL1]),
             ("replace", ["runs/round9/pool_pilot/w/JT00S-rb0.npz"], [f"{WREL}/{SW}.npz"]), ("insert", [], STAND)]
    want2 = [("replace", ["CB2P1N"], ["CB2D1N"]), ("replace", [POOL1], [POOL2]), ("replace", ["32"], ["40"]), ("replace", [f"{WREL}/CB2S.npz"], [f"{WREL}/CB2D.npz"])]
    k1 = sorted(map(str, d1)) == sorted(map(str, want1))
    tok2 = Counter(t for x in d2 for t in x[1] + x[2])
    k2 = all(t in (TP1, TD1, POOL1, POOL2, "32", "40", f"{WREL}/{SW}.npz", f"{WREL}/{DW}.npz") for t in tok2)
    if VAR:                                                                    # amendment 2: P1N' differs from P1N only in tag / wfile
        d0 = tokdiff(_p1n0(), p1n)
        k1 = k1 and sorted(t for x in d0 for t in x[1] + x[2]) == sorted(["CB2P1N-F-s0", TP1, f"{WREL}/CB2S.npz", f"{WREL}/{SW}.npz"])
    rec = dict(p1n_vs_p1l=[list(map(str, x)) for x in d1], d1n_vs_p1n=[list(map(str, x)) for x in d2], p1n=" ".join(p1n), d1n=" ".join(d1n),
               p1n_ok=bool(k1), d1n_ok=bool(k2 and len(d2) <= 4), pass_=bool(k1 and k2 and len(d2) <= 4))
    (O / "results").mkdir(parents=True, exist_ok=True)
    (O / f"results/diff{VAR}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    _ = want2
    sys.exit(0 if rec["pass_"] else 2)


# ---------------------------------------------------------------- DAgger labels (R3) and draw
def cmd_dlabel(a):
    name = a.name
    tbp = CR / f"on1_{name}/tab.npz"
    assert not (CR / f"on1_{name}/LABELLED").exists(), f"{name}: already labelled"
    tb = dict(np.load(tbp))
    T, X = np.load(a.teach), np.load(CR / f"on1_{name}/dagger.npz")
    names = tb["names"].astype(str)
    assert np.array_equal(T["names"].astype(str), names) and np.array_equal(X["names"].astype(str), names)
    rep = creep_pool(name, a.teach)                                             # G0 on these rows (R3) before anything is written
    draw = X["draw"]
    fr = tb["fut_reg"].astype(np.float64)
    hs = cum_arc(fr)[:, 7] < H_STAND
    hsp = hs | (~hs & (np.hypot(*(fr[:, 7, :2] - fr[:, 6, :2]).T) / 0.5 < V_REST))
    lab = np.nan_to_num(T["fut"]).astype(np.float64)
    j = np.flatnonzero(draw & hsp)
    lab[j] = replace_at(lab[j], cum_arc(tb["fut"][j]))
    fut = np.where(draw[:, None, None], lab, tb["served"]).astype(np.float32)
    rec = dict(pool=f"on1_{name}", rows=len(names), drawable=int(draw.sum()), r3_rows=int(len(j)), r3_share=float(len(j) / max(draw.sum(), 1)),
               g0=rep["rules"]["R3"], creep=rep, pass_=bool(rep["rules"]["R3"]["pass_"]))
    Path(a.res).mkdir(parents=True, exist_ok=True)
    (Path(a.res) / f"dlabel_{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != "creep"}, indent=1, default=float))
    if not rec["pass_"]:
        sys.exit(2)
    np.savez(tbp, **(tb | dict(fut=fut, src=np.full(len(names), "dagger").astype("U16"))))
    (CR / f"on1_{name}/LABELLED").write_text("nd-b3\n")


def cmd_dweights(a):
    tb = np.load(CR / POOL2 / "tab.npz")
    names, logs = tb["names"].astype(str), tb["log"].astype(str)
    B = np.load(WD / f"{SW}.npz")
    n0 = len(B["names"])
    assert np.array_equal(names[:n0], B["names"].astype(str)), "on1_JTPRD prefix != on1_JTPR"
    Xs = [np.load(CR / f"on1_{p}/dagger.npz") for p in a.pools]
    names_add = np.concatenate([x["names"].astype(str) for x in Xs])
    draw = np.concatenate([x["draw"] for x in Xs])
    assert len(names) == n0 + len(names_add) and all(x.split(":", 2)[2] == y.split(":", 1)[1] for x, y in zip(names[n0:], names_add))
    hl = held_logs(logs[tb["mode"].astype(str) == "self"])
    dr = np.flatnonzero(draw & ~np.isin(logs[n0:], sorted(hl)))
    if len(dr) < 9500:
        sys.exit(f"{len(dr)} drawable DAgger rows < 9 500 (registered stop)")
    w = np.zeros(len(names))
    w[:n0] = B["w"].astype(np.float64) * K_S / (K_S + K_D)
    w[n0 + dr] = K_D / (K_S + K_D) / len(dr)
    grp = np.concatenate([B["grp"].astype("U12"), np.full(len(names) - n0, "", "U12")])
    grp[n0 + dr] = "dag"
    w0 = np.zeros(len(names))
    w0[:n0] = B["w"]
    assert abs(w.sum() - 1) < 1e-9 and abs(w0.sum() - 1) < 1e-9
    np.savez(WD / f"{DW}.npz", names=tb["names"], w=w, grp=grp)
    np.savez(WD / f"{DW0}.npz", names=tb["names"], w=w0, grp=np.concatenate([B["grp"].astype("U12"), np.full(len(names) - n0, "dag0w", "U12")]))
    gs = Counter()
    for g_, x in zip(grp, w):
        gs[str(g_)] += x
    rec = dict(pool=POOL2, rows=len(names), dag_rows=int(len(dr)), dag_held=int((draw & np.isin(logs[n0:], sorted(hl))).sum()),
               group_share={k: float(v) for k, v in gs.items() if v > 0}, dag_draws_per_row=float(K_D * 9500 / len(dr)),
               dag_by_pool={p: int((np.concatenate([np.full(len(x["names"]), p) for p, x in zip(a.pools, Xs)])[dr] == p).sum()) for p in a.pools})
    (WD / f"{DW}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_smoke(a):
    import pcl as C
    d = C.run_dirs(a.tag)[-1]
    log = (d / "log.txt").read_text()
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", log)
    ev = C.events(d)
    sc = [e for e in ev if e.get("kind") == "scalar"]
    vals = np.array([e["value"] for e in sc], float)
    done = json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else {}
    gs = (done.get("draw_shares") or {}).get("group_share", {})
    W = np.load(DD / a.wfile)
    want = Counter()
    for g_, x in zip(W["grp"].astype(str), W["w"]):
        want[g_] += float(x)
    lon = {k: [e["value"] for e in sc if k in e["tag"]] for k in ("lon_stand", "lon_free")}
    ck = dict(done=bool(done), finite=bool(len(vals) and np.isfinite(vals).all()), batch=list(b.groups()) if b else None,
              batch_want=[str(128 - 13 - a.k), "13", str(a.k)], shares={g_: (gs.get(g_), v) for g_, v in want.items() if v > 0},
              lon_terms={k: dict(n=len(v), finite=bool(v and np.isfinite(v).all())) for k, v in lon.items()})
    ok = ck["done"] and ck["finite"] and ck["batch"] == ck["batch_want"] and all(x is not None and abs(x - y) < 1e-6 for x, y in ck["shares"].values()) \
        and all(v["n"] and v["finite"] for v in ck["lon_terms"].values())
    rec = dict(tag=a.tag, run=str(d), checks=ck, pass_=bool(ok))
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))
    sys.exit(0 if ok else 2)


# ---------------------------------------------------------------- summaries and registered lines
REF_OLD = dict(arena=["NTV0-L18-s0", "NT4D3-F-s0"], held460=["NTV0-L18-s0", "NT4D3-F-s0"])


def k_(bench, tag, cb2=True):
    return (f"{tag}@cb2" if bench == "arena" else f"{tag}-cb2") if cb2 else tag


def run(cmd, log):
    print(" ".join(map(str, cmd)), flush=True)
    with open(log, "a") as f:
        r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"{cmd[:4]} rc {r.returncode} (log {log})")


def cmd_sum(a):
    import shutil
    AP, V = DD / "third_party/alpasim/.venv/bin/python", REPO / ".venv/bin/python"
    st = a.stage
    arms = {1: [TP1], 2: [TD1, TP1], 3: [TD11, TD1, TP1]}[st] + (["CB2P1N-F-s0"] if VAR else [])   # amendment 2: P1N kept as the no-static reference
    sx = f"{st}{VAR}"
    b2 = [B2, "STANDFMXH-L18-s0"]
    R = O / "results"
    log = R / f"sum{sx}.log"
    rec = dict(stage=st)
    for b in ("arena", "held460"):
        new = [k_(b, t) for t in arms + b2]
        old = [k_(b, t, False) for t in b2 + REF_OLD[b]]
        run([AP, "-m", "jevdrive.facts", "fill", "--bench", b, "--keys", *new, *old], log)
        mean = [f"{k_(b, TD1)}+{k_(b, TD11)}"] if st == 3 else []
        # old scorer: arms vs the B2 re-reads and the stored references; the B2 re-reads vs the stored B2 reads (reproduction)
        run([V, "-m", "jevdrive.facts", "table", "--bench", b, "--tags", *[k_(b, t) for t in arms], "--refs", *[k_(b, t) for t in b2], *old,
             "--out", R / f"{b}_old_s{sx}.md", "--title", f"CB2 stage {st}, {b}, old scorer"], log)
        run([V, "-m", "jevdrive.facts", "table", "--bench", b, "--tags", *[k_(b, t) for t in b2], "--refs", *[k_(b, t, False) for t in b2],
             "--out", R / f"{b}_repro_b2.md", "--title", f"B2 native re-reads vs the stored reads, {b}, old scorer"], log)
        v2 = [f"{k}={O}/v2/{b}/{k}/new/rollouts.csv" for k in new]
        run([AP, REPO / "experiments/round9/scripts/cb2v2.py", "table", "--bench", b, "--v2", *v2, "--tags", *[k_(b, t) for t in arms], *mean,
             "--refs", *[k_(b, t) for t in b2], "--out", R / f"{b}_v2_s{sx}.md", "--title", f"CB2 stage {st}, {b}, fixed scorer (1b024f0)"], log)
        rec[b] = dict(old=json.loads((R / f"{b}_old_s{sx}.json").read_text()), v2=json.loads((R / f"{b}_v2_s{sx}.json").read_text()),
                      repro=json.loads((R / f"{b}_repro_b2.json").read_text()))
        rec[b]["v2_summaries"] = {k: json.loads((O / f"v2/{b}/{k}/{v}/summary.json").read_text()) for k in new for v in ("old", "new")
                                  if (O / f"v2/{b}/{k}/{v}/summary.json").exists()}
    # exact per-window reproduction of the stored B2 reads by the re-reads (old scorer)
    import pandas as pd
    sys.path.insert(0, str(REPO))
    import jevdrive.facts as F
    rec["repro_exact"] = {}
    for b in ("arena", "held460"):
        for t in b2:
            x, y = F.scores(b, F.akey(k_(b, t)) if b == "arena" else k_(b, t)), F.scores(b, F.akey(t) if b == "arena" else t)
            ids = x.index.intersection(y.index)
            d = (x.score.loc[ids] - y.score.loc[ids]).abs()
            rec["repro_exact"][f"{b}:{t}"] = dict(n=len(ids), equal=int((d < 1e-9).sum()), max_abs=float(d.max()), mean_diff=float((x.score.loc[ids] - y.score.loc[ids]).mean()))
    A, H = rec["arena"]["v2"], rec["held460"]["v2"]
    if st == 1:
        p = A[f"{TP1}@cb2|{B2}@cb2"]
        z = H[f"{TP1}-cb2|{B2}-cb2"]["zeros_a"]
        ok = p["a"] >= p["b"] - 0.060 and z <= 60
        rec["E1"] = dict(arena_p1n=p["a"], arena_b2s1=p["b"], line=p["b"] - 0.060, held_zeros_p1n=z, held_line=60, pass_=bool(ok))
        (R / f"E1_PASS{VAR}").write_text("pass\n") if ok else None
    if st in (2, 3):
        tag = TD1 if st == 2 else f"{TD1}+{TD11}"
        ka = f"{tag}@cb2" if st == 2 else f"{TD1}@cb2+{TD11}@cb2"
        kh = f"{tag}-cb2" if st == 2 else f"{TD1}-cb2+{TD11}-cb2"
        p, q = A[f"{ka}|{B2}@cb2"], H[f"{kh}|{B2}-cb2"]
        ok = p["hi"] >= 0 and q["zeros_a"] <= q["zeros_b"] + 4
        rec["ADOPT"] = dict(arm=tag, arena=p, held=q, held_zero_line=q["zeros_b"] + 4, pass_=bool(ok))
        if st == 2 and ok:
            (R / f"ADOPT_PASS{VAR}").write_text("pass\n")
    if st == 2:                                                             # PAI 281 side read (new scorer from the scorer_v2 mirror, old from the facts)
        F_ = DD / "runs/pai_fix/facts"
        run([DD / "envs/op-train/bin/python", REPO / "experiments/round2/scripts/r2.py", "r281", "--read", f"D1N={F_}/{TD1}-281ag.csv",
             f"D3s0={F_}/NT4D3-F-s0-281ag.csv", f"D1s0={F_}/NT3D1-F-s0-281ag.csv", f"R8Ks2={F_}/R8K-L18-s2-281ag.csv", "--pairs", "D1N:D3s0", "D1N:D1s0",
             "D1N:R8Ks2", "--out", R / f"pai281_old{VAR}.md", "--title", "PAI 281 ACC + GAP, Harmonizer OFF (old scorer): D1N side read"], log)
        sv = pd.read_csv(O / f"sv2{VAR}/new/rollouts.csv")
        rec["pai281_new"] = dict(mean=float(sv.drop_duplicates("scene").score.mean()), zeros=int((sv.drop_duplicates("scene").score == 0).sum()),
                                 scenes=int(sv.scene.nunique()))
    # extracts for the Mac (frozen PCS, easy items): native_gap-format ext of every cb2 arena read, windows tables old / v2
    ext = [f"{t}={k_('arena', t)}" for t in arms + b2]
    run([AP, "-m", "jevdrive.facts", "ext", "--read", *ext, "--out", O / f"ext_s{sx}"], log)
    w = pd.read_parquet(O / f"ext_s{sx}/win.parquet")
    w[["tag", "id", "score", "type"]].to_csv(R / f"arena_windows_old_s{sx}.csv.gz", index=False)
    run([AP, REPO / "experiments/round9/scripts/cb2v2.py", "rows", "--v2", *[f"{t}={O}/v2/arena/{k_('arena', t)}/new/rollouts.csv" for t in arms + b2],
         "--rows", R / f"arena_windows_old_s{sx}.csv.gz", "--out", R / f"arena_windows_v2_s{sx}.csv.gz"], log)
    (R / f"sum{sx}.json").write_text(json.dumps(rec, indent=1, default=float))
    RES = REPO / "experiments/round9/results/cb2"
    RES.mkdir(parents=True, exist_ok=True)
    for f in R.glob("*"):
        if f.suffix in (".md", ".json", ".gz") and not f.name.startswith("sum") or f.name == f"sum{sx}.json":
            shutil.copy2(f, RES / f.name)
    print(json.dumps({k: rec.get(k) for k in ("E1", "ADOPT", "repro_exact", "pai281_new")}, indent=1, default=float))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("creep"); q.add_argument("--pools", nargs="+", required=True); q.add_argument("--teach-dir", required=True); q.add_argument("--out", required=True)
    q = sp.add_parser("c0chk"); q.add_argument("--new", required=True); q.add_argument("--ref", required=True); q.add_argument("--list", required=True)
    q.add_argument("--out", required=True)
    q = sp.add_parser("pgate"); q.add_argument("--name", required=True); q.add_argument("--out", required=True)
    sp.add_parser("wfile")
    q = sp.add_parser("argv"); q.add_argument("arm"); q.add_argument("tag"); q.add_argument("--steps", type=int); q.add_argument("--on-start", type=int)
    q.add_argument("--save-at"); q.add_argument("--seed", type=int)
    sp.add_parser("diff")
    q = sp.add_parser("dlabel"); q.add_argument("--name", required=True); q.add_argument("--teach", required=True); q.add_argument("--res", required=True)
    q = sp.add_parser("dweights"); q.add_argument("--pools", nargs="+", required=True)
    q = sp.add_parser("smoke"); q.add_argument("--tag", required=True); q.add_argument("--wfile", required=True); q.add_argument("--k", type=int, required=True)
    q.add_argument("--out", required=True)
    q = sp.add_parser("sum"); q.add_argument("--stage", type=int, required=True)
    a = p.parse_args()
    dict(sum=cmd_sum, creep=cmd_creep, c0chk=cmd_c0chk, pgate=cmd_pgate, wfile=cmd_wfile, argv=cmd_argv, diff=cmd_diff, dlabel=cmd_dlabel, dweights=cmd_dweights,
         smoke=cmd_smoke)[a.cmd](a)


if __name__ == "__main__":
    main()
