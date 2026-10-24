#!/usr/bin/env python3
"""ROUND9 CB2 (clean B2; plans/2026-10-25-cb2.md = the registration; lane scripts/cb2.sh). Box, repo root, repo venv unless noted.
  creep --pools N.. --teach-dir D --out P      step 0 (offline, before registration): b3 DAgger candidates (k >= 2, fed >= 2 m/s, teacher label)
                                               on pools whose rows exist; how many are human-standing / human-stopping, how far the b3 label's
                                               longitudinal (= the student's served arc) goes past the STAND cap, the rule candidates R1 / R2
                                               and the offline gate G0 -> P.{json,md}
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/round9/scripts")]
from jevdrive.common import data_dir  # noqa: E402

DD = data_dir()
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
    rules = dict(R1=hs, R2=hsp)
    out = dict(pool=name, rows=len(names), candidates=int(c.sum()), human_standing=int((c & hs).sum()), human_stopping=int((c & hp).sum()),
               standing_no_lead=int((c & hs & ~lead).sum()), stopping_no_lead=int((c & hp & ~lead).sum()),
               standing_share=float((c & hs).sum() / max(c.sum(), 1)), stopping_share=float((c & hp).sum() / max(c.sum(), 1)),
               conflict_rows=int(conf.sum()), conflict_rows_no_lead=int((conf & ~lead).sum()), conflict_excess_m=float(exc[conf].sum()),
               conflict_excess_p50_p90=[float(np.percentile(exc[conf], q)) for q in (50, 90)] if conf.any() else None,
               served_vs_human_4s_on_hsp_p50_p90=[float(np.percentile((a_lab - a_h)[c & hsp], q)) for q in (50, 90)] if (c & hsp).any() else None,
               ego_speed_on_hsp_p50=float(np.median(v0[c & hsp])) if (c & hsp).any() else None,
               moving_rows_faster_than_cap=int((c & ~hsp & (exc > 0)).sum()),
               lat_mass=float(lat[c].sum()), lat_rows_05=int((c & (lat > 0.5)).sum()), rules={})
    for r, m in rules.items():
        kp = c & ~m
        g = dict(kept=int(kp.sum()), keep_share=float(kp.sum() / max(c.sum(), 1)), lat_mass_kept=float(lat[kp].sum() / max(lat[c].sum(), 1e-9)),
                 lat_rows_05_kept=float((kp & (lat > 0.5)).sum() / max((c & (lat > 0.5)).sum(), 1)),
                 conflict_excess_left=float(exc[kp & conf].sum() / max(exc[conf].sum(), 1e-9)) if conf.any() else 0.0)
        g["pass_"] = bool(g["keep_share"] >= G0_KEEP and g["lat_mass_kept"] >= G0_LAT and g["lat_rows_05_kept"] >= G0_LAT and g["conflict_excess_left"] <= G0_CONF)
        out["rules"][r] = g
    return out


def cmd_creep(a):
    res = [creep_pool(p, Path(a.teach_dir) / f"{p}.npz") for p in a.pools]
    pick = next((r for r in ("R1", "R2") if all(x["rules"][r]["pass_"] for x in res)), None)
    rec = dict(pools=res, gate=dict(keep=G0_KEEP, lat=G0_LAT, conflict_left=G0_CONF), pick=pick)
    Path(a.out + ".json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    L = ["# CB2 step 0: b3 DAgger candidates on human-standing / -stopping states (cb2.py creep)", "",
         "Candidates = k >= 2, fed >= 2 m/s, teacher label, not in the 15 % held-out logs. HS = the human's logged 4 s arc < 0.5 m; HP = not HS and "
         "his end speed < 0.5 m/s (stops within 4 s). Conflict = HS / HP rows whose b3 label (longitudinal = the served arc) ends beyond the STAND cap "
         "max(his arcs) + 1 m. Lead = a logged vehicle at t0 within |y| < 2 m ahead (no lead = line stop: red light / stop sign proxy).", "",
         "| pool | cand | HS | HP | HS no lead | HP no lead | conflict rows (no lead) | excess m p50 / p90 | served - human 4 s p50 / p90 on HS+HP | "
         "R1 keep / lat / lat>0.5 / conflict left | R2 keep / lat / lat>0.5 / conflict left |", "|:--|" + "--:|" * 10]
    f = lambda g: f"{g['keep_share']:.3f} / {g['lat_mass_kept']:.3f} / {g['lat_rows_05_kept']:.3f} / {g['conflict_excess_left']:.3f} {'PASS' if g['pass_'] else 'FAIL'}"  # noqa: E731
    for x in res:
        q = lambda v: "-" if v is None else " / ".join(f"{t:.2f}" for t in v)  # noqa: E731
        L.append(f"| {x['pool']} | {x['candidates']} | {x['human_standing']} | {x['human_stopping']} | {x['standing_no_lead']} | {x['stopping_no_lead']} | "
                 f"{x['conflict_rows']} ({x['conflict_rows_no_lead']}) | {q(x['conflict_excess_p50_p90'])} | {q(x['served_vs_human_4s_on_hsp_p50_p90'])} | "
                 f"{f(x['rules']['R1'])} | {f(x['rules']['R2'])} |")
    L += ["", f"G0 lines: keep >= {G0_KEEP}, lateral mass and rows > 0.5 m kept >= {G0_LAT}, conflict excess left <= {G0_CONF}. "
          f"Simplest rule passing on every pool: **{pick or 'none'}**."]
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("creep"); q.add_argument("--pools", nargs="+", required=True); q.add_argument("--teach-dir", required=True); q.add_argument("--out", required=True)
    a = p.parse_args()
    dict(creep=cmd_creep)[a.cmd](a)


if __name__ == "__main__":
    main()
