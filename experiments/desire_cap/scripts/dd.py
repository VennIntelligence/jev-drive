#!/usr/bin/env python3
"""DESIRE-D0 (plans/2026-10-17-desire-d0-prereg.md): can Cinque's native laneChange pathway, triggered by the route (lib/route_desire.py), carry
lane changes in closed loop, and at which speeds does it exist at all? No training.

  scenes    (box, .venv, CPU)       the registered D0 scene lists from stored logs and map sites only (no D0 output):
                                    PAI 281 = map lane-change sites (route_follow results/laneoff/lc_map.csv) U scenes where RouteDesire(v_min 3)
                                    starts in the stored ACC reference's served memory; PAI train160 = scenes whose on1_PAIonp rows fire
                                    route_desire.intent; held = map sites of the stored held RSG-L18-s0 read U RouteDesire starts in its logs
                                    -> runs/desire_cap/d0/lists/{pai281,pai160,held}.txt (+ .tsv with the reason)
  lowrows   (box, op-train, CPU)    PAI on-policy rows (on1_PAIonp) where route_desire.intent fires -> runs/desire_cap/rows_paiint.npz
  lowplans  (box, op-train, GPU)    dc.py's plans protocol on those rows (P0, RSG-L18-s0 / s1) -> runs/desire_cap/plans/paiint/<tag>.npz
  lowreport (Mac or box, CPU)       speed-binned lane-change response (NAVSIM capledger_test rows of dc.py + the PAI rule rows), the
                                    registered low-speed line and the speed gate -> results/d0/lowspeed.md, lowspeed.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(HERE)]
import dc  # noqa: E402
import route_desire as RD  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = DATA / "runs/desire_cap"
D0 = OUT / "d0"
RES = REPO / "experiments/desire_cap/results/d0"
BINS = ((0.0, 3.0), (3.0, 6.0), (6.0, 8.9), (8.9, 99.0))
LOW_BINS = BINS[1:3]                                     # the registered low-speed bins
V_SCENE = 3.0                                            # the scene lists' v_min (the lowest gate the read can pick)
LINE_M, LINE_DIR = 0.5, 0.95
STAGES = ("P0", "RSG-L18")
REF281 = DATA / "runs/route_follow/pai_g10/acc"
HELD_REF = DATA / "runs/route_mem/read/RSG-L18-s0"
LC_MAP = REPO / "experiments/route_follow/results/laneoff/lc_map.csv"


# ---------------------------------------------------------------- scene lists (CPU, stored logs only)
def starts_in_log(path, v_min=V_SCENE):
    """drive.jsonl -> {scene: [(t0 s, side, speed)]} of RouteDesire starts on the served memory (one machine per session)."""
    ses, out = {}, {}
    for line in open(path):
        if '"rmem"' not in line or '"infer": false' in line:              # PAI logs mark inferences; the nuPlan driver infers on every call
            continue
        r = json.loads(line)
        if r.get("kind") != "drive":
            continue
        rd = ses.setdefault(r["session"], RD.RouteDesire(v_min=v_min))
        t, v = r["t0"] * 1e-6, 10.0 * float(r["ego"][4])
        was = rd.state
        rd.step(t, v, np.asarray(r["rmem"], np.float32))
        if was == "off" and rd.state == "starting":
            out.setdefault(r["scene"], []).append((round(t, 2), rd.side, round(v, 2)))
    return out


def cmd_scenes(a):
    import pandas as pd
    D0.mkdir(parents=True, exist_ok=True)
    (D0 / "lists").mkdir(exist_ok=True)
    M = pd.read_csv(LC_MAP)
    M = M[M.site.astype(str) == "True"]
    rows = {}

    def add(lst, scene, why):
        rows.setdefault(lst, {}).setdefault(scene, []).append(why)
    for s, g in M[M.set == "pai"].groupby("scene"):
        add("pai281", s, f"map:{g.side.iloc[0]}:gap{float(g.gap.iloc[0]):.1f}")
    for s, g in M[M.set == "held_s0"].groupby("scene"):
        add("held", s, f"map:{g.side.iloc[0]}:gap{float(g.gap.iloc[0]):.1f}")
    for f in sorted(glob.glob(str(REF281 / "runs/*/driver/drive.jsonl"))):
        for s, st in starts_in_log(f).items():
            add("pai281", s, "rule:" + ",".join(f"{'L' if d > 0 else 'R'}@{v:.1f}" for _, d, v in st))
    for f in sorted(glob.glob(str(HELD_REF / "c*/run/driver-logs/drive.jsonl"))):
        for s, st in starts_in_log(f).items():
            add("held", s, "rule:" + ",".join(f"{'L' if d > 0 else 'R'}@{v:.1f}" for _, d, v in st))
    t = np.load(DATA / "runs/op_parity/cache/on1_PAIonp/tab.npz", allow_pickle=True)
    z = np.load(DATA / "runs/route_mem/feat/on1_PAIonp.npz", allow_pickle=True)
    F = z["f"].astype(np.float32)
    F[~z["ok"]] = 0
    r = RD.intent(F)
    for s in sorted(set(t["log"][r["side"] != 0].tolist())):
        m = (t["log"] == s) & (r["side"] != 0)
        add("pai160", s, f"intent:{int(m.sum())}rows:v{np.median(t['speed'][m]):.1f}")
    for lst, d in rows.items():
        ids = sorted(d)
        (D0 / "lists" / f"{lst}.txt").write_text("\n".join(ids) + "\n")
        (D0 / "lists" / f"{lst}.tsv").write_text("scene\twhy\n" + "".join(f"{s}\t{';'.join(d[s])}\n" for s in ids))
        print(lst, len(ids), "scenes;", sum(any(w.startswith("map") for w in d[s]) for s in ids), "map,",
              sum(any(w.startswith("rule") for w in d[s]) for s in ids), "rule")


# ---------------------------------------------------------------- low-speed rows / plans
def cmd_lowrows(a):
    t = np.load(DATA / "runs/op_parity/cache/on1_PAIonp/tab.npz", allow_pickle=True)
    z = np.load(DATA / "runs/route_mem/feat/on1_PAIonp.npz", allow_pickle=True)
    assert np.array_equal(z["names"], t["names"]), "route features rows != cache rows"
    F = z["f"].astype(np.float32)
    F[~z["ok"]] = 0
    r = RD.intent(F)
    sel = np.flatnonzero(r["side"] != 0)
    np.savez(OUT / "rows_paiint.npz", idx=sel, names=t["names"][sel], log=t["log"][sel], v0=t["speed"][sel], side=r["side"][sel],
             y_t=r["y_t"][sel], d_turn=r["d_turn"][sel], in_P_int=np.ones(len(sel), bool))
    print("rows", len(sel), "scenes", len(set(t["log"][sel].tolist())),
          {f"{lo:g}-{hi:g}": int(((t['speed'][sel] >= lo) & (t['speed'][sel] < hi)).sum()) for lo, hi in BINS})


def cmd_lowplans(a):
    dc.SRC["paiint"] = "on1_PAIonp"
    dc.cmd_plans(argparse.Namespace(tags=a.tags, srcs=["paiint"], force=a.force))


# ---------------------------------------------------------------- low-speed report
def stat(v, g):
    v = np.asarray(v, float)
    if len(v) < 2 or len(set(np.asarray(g).tolist())) < 2:
        return dict(n=int(len(v)), groups=int(len(set(np.asarray(g).tolist()))), mean=float(np.mean(v)) if len(v) else float("nan"),
                    lo=float("nan"), hi=float("nan"), dir=float((v > 0.1).mean()) if len(v) else float("nan"))
    m, lo, hi = dc.cboot(v, g)
    return dict(n=int(len(v)), groups=int(len(set(np.asarray(g).tolist()))), mean=m, lo=lo, hi=hi, dir=float((v > 0.1).mean()))


def exists(s):
    return bool(s["n"] >= 10 and s["mean"] >= LINE_M and s["dir"] >= LINE_DIR)


def cmd_lowreport(a):
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Rn = np.load(root / "rows_nav.npz")
    Rp = np.load(root / "rows_paiint.npz") if (root / "rows_paiint.npz").exists() else None
    L, res = ["# DESIRE-D0 low-speed read (open loop, dc.py protocol)", "",
              "Response R_y(4 s) = the plan's lateral at 4 s with the lane-change desire (pulse age 0) minus without, toward the desire side; mean "
              "[log / scene cluster bootstrap 95 % CI], dir = share > 0.1 m. Stage = row mean of its seeds (RSG-L18: s0, s1). "
              f"Line: mean >= {LINE_M} m and dir >= {LINE_DIR} (n >= 10 rows) = the pathway exists in that bin and side.", ""], {}
    nav = Rn["in_C"] if "in_C" in Rn.files else (Rn["in_C_slow"] | Rn["in_C_fast"])
    fam = {"NAV (C + LC rows)": (nav | Rn["in_LC_L"] | Rn["in_LC_R"], Rn["v0"], Rn["log"], None, "nav"),
           "NAV C (cruise, human keeps lane)": (nav, Rn["v0"], Rn["log"], None, "nav"),
           "NAV LC (human changes lane)": (Rn["in_LC_L"] | Rn["in_LC_R"], Rn["v0"], Rn["log"], None, "nav")}
    if Rp is not None:
        fam["PAI rule rows (route_desire fires; desire on the route side)"] = (np.ones(len(Rp["v0"]), bool), Rp["v0"], Rp["log"], Rp["side"], "paiint")
    for name, (base, v0, g, rside, src) in fam.items():
        L += [f"## {name}", "", "| stage | bin (m/s) | rows / groups | left: mean [CI] | dir | right: mean [CI] | dir | left exists | right exists | "
              "ages 0.4 / 1 / 2 s (pooled) |", "|:--|:--|--:|:--|--:|:--|--:|:--|:--|:--|"]
        for st in STAGES:
            Q = dc.load_stage(root, src, st)
            if Q is None:
                continue
            for lo, hi in BINS:
                m = base & (v0 >= lo) & (v0 < hi)
                if rside is not None:
                    sl, sr = m & (rside > 0), m & (rside < 0)
                else:
                    sl = sr = m
                rl = stat(dc.resp(Q, "LCL@0")[sl, dc.I4], g[sl])
                rr = stat(dc.resp(Q, "LCR@0")[sr, dc.I4], g[sr])
                ages = " / ".join(f"{np.mean(np.r_[dc.resp(Q, f'LCL@{x:g}')[sl, dc.I4], dc.resp(Q, f'LCR@{x:g}')[sr, dc.I4]]):.2f}"
                                  if (sl.any() or sr.any()) else "-" for x in dc.AGES[1:])
                res.setdefault(name, {}).setdefault(st, {})[f"{lo:g}-{hi:g}"] = dict(left=rl, right=rr, left_exists=exists(rl), right_exists=exists(rr))
                L.append(f"| {st} | {lo:g}-{hi:g} | {int(m.sum())} / {len(set(g[m].tolist()))} | {rl['mean']:.2f} [{rl['lo']:.2f}, {rl['hi']:.2f}] | {rl['dir']:.2f} | "
                         f"{rr['mean']:.2f} [{rr['lo']:.2f}, {rr['hi']:.2f}] | {rr['dir']:.2f} | {exists(rl)} | {exists(rr)} | {ages} |")
        L.append("")
    # the registered gate: RSG-L18, right side, NAV (C + LC rows) and the PAI rule rows; the lowest bin from which every higher low bin holds
    prim = "NAV (C + LC rows)"
    pai = next((k for k in res if k.startswith("PAI")), None)

    def ok(st, b, side):
        x = res[prim][st][b][f"{side}_exists"]
        return x and (pai is None or res[pai][st][b][f"{side}_exists"])
    keys = [f"{lo:g}-{hi:g}" for lo, hi in LOW_BINS]
    gate = 8.9
    for i, (lo, _) in enumerate(LOW_BINS):
        if all(ok("RSG-L18", k, "right") for k in keys[i:]):
            gate = lo
            break
    res["gate"] = dict(v_min=gate, rule="RSG-L18 right side, NAV (C + LC rows) and PAI rule rows both exist from this bin up; else 8.9")
    res["p0_low"] = {k: dict(left=ok("P0", k, "left"), right=ok("P0", k, "right")) for k in keys}
    res["rsg_low"] = {k: dict(left=ok("RSG-L18", k, "left"), right=ok("RSG-L18", k, "right")) for k in keys}
    L += ["## Registered lines", "",
          f"- P0 low-speed pathway (both families): " + ", ".join(f"{k}: left {v['left']}, right {v['right']}" for k, v in res["p0_low"].items()),
          f"- RSG-L18 low-speed pathway (both families): " + ", ".join(f"{k}: left {v['left']}, right {v['right']}" for k, v in res["rsg_low"].items()),
          f"- **Speed gate for D0: v_min = {gate:g} m/s** ({res['gate']['rule']})"]
    (out / "lowspeed.md").write_text("\n".join(L) + "\n")
    (out / "lowspeed.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("scenes")
    sp.add_parser("lowrows")
    p = sp.add_parser("lowplans")
    p.add_argument("--tags", nargs="+", default=["P0", "RSG-L18-s0", "RSG-L18-s1"])
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("lowreport")
    p.add_argument("--root", default=str(OUT))
    p.add_argument("--out", default=str(RES))
    a = ap.parse_args()
    {"scenes": cmd_scenes, "lowrows": cmd_lowrows, "lowplans": cmd_lowplans, "lowreport": cmd_lowreport}[a.cmd](a)


if __name__ == "__main__":
    main()
