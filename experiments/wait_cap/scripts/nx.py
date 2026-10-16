#!/usr/bin/env python3
"""Decision 335's two follow-up pilots (prereg draft plans/2026-10-17-retain-anticopy-draft.md); box, repo root; chain in nx.sh.
Outputs: box runs/wait_cap/next/, repo results/next/.

  (iii) RETAIN  the base stage (RB0-F recipe, 10 000 steps from the shipped weights) + P0 plan distillation on the retention rows (on1_WAIT)
                + D2 desire-delta distillation, both at lambda in {1, 3} (rm_train --kd-lam / --d2-lam); control = RB0-F-s0 (bit-identical recipe
                with the flags off, proven by `bitcmp`).
  (iv) ANTI-COPY  the RSG-L18 recipe (1 000 steps, WAIT pilot's shape) + history-speed noise (rm_train --ego-scale-speed 0.5, --ego-scale-max in
                {0.2, 0.4}); control = WTCP-L18-s0.

  bitcmp --a A --b B --out NAME [--hash H] [--note ..]   two runs: every logged scalar, every weight, dev ADE, on_hash_60 (and = H if given)
                -> results/next/NAME.json; exit 2 unless identical
  hist --tags T..   (GPU)  the gate-4 dev rows (rm_train dev_eval's 1 789 rows), the speed history rescaled by s in {0.8, 1.0, 1.2} (rm_train
                --ego-scale-speed's columns and powers) -> next/hist/<tag>.npz; s = 1 must reproduce the run's DONE dev_ade
  rtgate --j T1 T3 --c C --lam-of 1=T1,3=T3   (CPU)  the (iii) lines -> results/next/rt_pilot_gate.{json,md}
  acgate --j T2 T4 --c C --dose-of 0.2=T2,0.4=T4 [--submit]   (CPU)  the (iv) lines -> results/next/ac_pilot_gate.{json,md}; --submit: held
                460 @18 for the passing arms (nx.sh ac-held), then `acheld`
  acheld --j T.. --c C   (CPU)  held 460 @18 per arm vs C (one seed; wt.xpair) + rf_follow turns -> results/next/ac_held.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
import wt  # noqa: E402

DATA = wt.DATA
NX = DATA / "runs/wait_cap/next"
RES = REPO / "experiments/wait_cap/results/next"
SCALES = (0.8, 1.0, 1.2)
V_LO, V_HI = 0.5, 8.0                     # the low-v0 band of 补记 D (where J's 4 s shortfall sat)


def write(name, res, md=None, dry=False):
    if dry:                                                            # a check of the read code on existing tags: print only
        print("\n".join(md) if md else json.dumps(res, indent=1, default=float))
        return
    RES.mkdir(parents=True, exist_ok=True)
    NX.mkdir(parents=True, exist_ok=True)
    for p in (NX / f"{name}.json", RES / f"{name}.json"):
        p.write_text(json.dumps(res, indent=1, default=float))
    if md:
        for p in (NX / f"{name}.md", RES / f"{name}.md"):
            p.write_text("\n".join(md) + "\n")
        print("\n".join(md))
    else:
        print(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------- identity (CPU)
def cmd_bitcmp(a):
    (sa, da, ma), (sb, db, mb) = wt.run_of(a.a), wt.run_of(a.b)
    keys = [k for k in sorted(set(sa) & set(sb), key=str) if not k[0].startswith("throughput/")]
    dl = max(abs(sa[k] - sb[k]) for k in keys)
    same = all(set(ma[g] or {}) == set(mb[g] or {}) for g in ("net", "parity"))
    dw = max(float((ma[g][k].float() - mb[g][k].float()).abs().max()) for g in ("net", "parity") for k in ma[g]) if same else float("inf")
    r = dict(a=a.a, b=a.b, note=a.note, scalars=len(keys), scalar_keys_equal=set(k[0] for k in sa) == set(k[0] for k in sb),
             scalar_max_abs_diff=dl, weight_max_abs_diff=dw, weight_keys_equal=same, dev_ade=[da.get("dev_ade"), db.get("dev_ade")],
             on_hash_60=[da.get("on_hash_60"), db.get("on_hash_60")], want_hash=a.hash or None)
    r["identical"] = bool(dl == 0 and dw == 0 and same and r["scalar_keys_equal"] and r["dev_ade"][0] == r["dev_ade"][1]
                          and r["on_hash_60"][0] == r["on_hash_60"][1] and (not a.hash or r["on_hash_60"][0] == a.hash))
    write(a.out, r)
    raise SystemExit(0 if r["identical"] else 2)


# ---------------------------------------------------------------- history-scale read (GPU)
def cmd_hist(a):
    """Plans of the gate-4 dev rows (wt.cmd_devall's build and call) with the fed speed history x s, per tag -> NX/hist/<tag>.npz."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    from types import SimpleNamespace
    sys.path[:0] = [str(REPO / "experiments/route_mem/scripts"), str(REPO / "experiments/route_follow/scripts")]
    import rm_train as RT
    import pp_train as T
    from rf_lonread import REC
    out = NX / "hist"
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in a.tags if not (out / f"{t}.npz").exists()]
    if not todo:
        return
    dev = torch.device("cuda")
    ns, _, _, _, _ = RT.build(SimpleNamespace(**(REC | dict(data=RT.OT.JT.FULL, yr_mass=0.1)), road_labels=RT.OT.JT.ROAD, on=wt.BASE, init="",
                                              rm_from=todo[0], tag=todo[0], exp="wait_cap", lr_scale=1.0, prefetch=1, dev_card_gb=4.0), dev)
    S, model, W = ns.S, ns.model, ns.losses(ns.mode).W
    Sd = T.dev_store(S, ns.dv_rows, 4.0)
    pi = torch.as_tensor(S.pi, device=dev)
    esp = torch.as_tensor(RT.EGO_SPEED, device=dev)
    pw = torch.as_tensor(RT.EGO_SCALE_POW, device=dev, dtype=torch.float32)
    rows = ns.dv_o[S.has_fut[torch.as_tensor(ns.dv_o, device=dev)].cpu().numpy().astype(bool)]
    fut = S.fut[torch.as_tensor(rows, device=dev)].cpu().numpy().astype(np.float32)
    for j, tag in enumerate(todo):
        if j:
            model.load_state(torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"])
        model.eval()
        P = {}
        for s in SCALES:
            res = []
            with torch.no_grad():
                for i in range(0, len(rows), 128):
                    r = torch.as_tensor(rows[i:i + 128], device=dev)
                    ego = Sd.ego[r].clone()
                    ego[:, esp] = ego[:, esp] * (s ** pw)[None]
                    p = model(Sd.front[r], ego, Sd.tc[r], None, None, inputs_on=True, nv=None if Sd.nv is None else Sd.nv[r]).float()[:, pi].view(-1, 33, 15)
                    res.append(torch.stack(T.rear(p, Sd.cam_x[r], W), -1).cpu())
            P[s] = torch.cat(res).numpy().astype(np.float32)
        ade = float(np.hypot(*(P[1.0][..., :2] - fut[..., :2]).transpose(2, 0, 1)).mean())
        ref = wt.dev_ade(tag)
        print(tag, len(rows), "ADE", ade, "DONE", ref, flush=True)
        assert abs(ade - ref) < 1e-3, f"{tag}: dev ADE {ade} does not reproduce DONE {ref}"
        np.savez(out / f"{tag}.npz", rows=rows, names=S.tab["names"][rows], log=S.tab["log"][rows], speed=S.tab["speed"][rows], fut=fut,
                 scales=np.array(SCALES), **{f"plan_{s}": P[s] for s in SCALES})


def arc4(P):
    """Rear-axle poses (n, 8, >=2) -> arc length to the 4 s pose (n,)."""
    return np.hypot(P[:, 0, 0], P[:, 0, 1]) + np.hypot(*np.diff(P[..., :2], axis=1).transpose(2, 0, 1)).sum(1)


def hist_read(tag, ctag):
    """4 s arc bias (plan - human) on the low-v0 dev rows at each history scale, arm vs control, paired (log-clustered); the history gain
    g = d arc4 / d s (central difference, m per unit s) on the same rows."""
    from jevdrive import stats
    a, c = np.load(NX / "hist" / f"{tag}.npz"), np.load(NX / "hist" / f"{ctag}.npz")
    assert np.array_equal(a["rows"], c["rows"])
    m = (a["speed"] >= V_LO) & (a["speed"] < V_HI)
    h = arc4(a["fut"])
    g = a["log"][m]
    out = dict(rows=int(m.sum()), logs=int(len(np.unique(g))))
    for s in SCALES:
        ba, bc = arc4(a[f"plan_{s}"]) - h, arc4(c[f"plan_{s}"]) - h
        p = stats.paired(np.abs(ba[m]), np.abs(bc[m]), groups=g)
        q = stats.paired(ba[m], bc[m], groups=g)
        out[f"s{s}"] = dict(bias=float(ba[m].mean()), bias_c=float(bc[m].mean()), absbias=float(np.abs(ba[m]).mean()), absbias_c=float(np.abs(bc[m]).mean()),
                            d_abs=stats.fmt(p), d_abs_hi=float(p["hi"]), d_abs_lo=float(p["lo"]), d_bias=stats.fmt(q), d_bias_lo=float(q["lo"]),
                            d_bias_hi=float(q["hi"]))
    for k, z in (("gain", a), ("gain_c", c)):
        out[k] = float(((arc4(z["plan_1.2"]) - arc4(z["plan_0.8"]))[m] / 0.4).mean())
    out["gain_rel"] = out["gain"] / max(out["gain_c"], 1e-6)
    return out


# ---------------------------------------------------------------- (iii) RETAIN lines (CPU)
def ledger_P0():
    """Ledger F cells of shipped Cinque (the P0 plans of led.py) -> {cell: success share}."""
    return {k: float(v[0].mean()) for k, v in wt.ledger_read("P0").items()}


def rely_of(tag):
    return json.loads((DATA / "runs/route_mem/rely" / f"{tag}.json").read_text())


def cmd_rtgate(a):
    sys.path.insert(0, str(REPO / "experiments/desire_cap/scripts"))
    import dt
    from jevdrive import stats
    Cc, cx = wt.cap_read(a.c)
    P0c, _ = wt.cap_read("P0")
    LP0, LC = ledger_P0(), {k: v for k, v in wt.ledger_read(a.c).items()}
    dC = dt.desire_read(a.c)
    c_ade = wt.dev_ade(a.c)
    res = dict(control=a.c, C=Cc | dict(dev_ade=c_ade, ledger_F={k: float(v[0].mean()) for k, v in LC.items()}, desire4=dC["arm"]["mean4"],
                                        rely=rely_of(a.c)), P0=dict(wait_W=P0c["wait_W"], D=P0c["D"], ledger_F=LP0, desire4=dC["P0"]["mean4"]), arms={})
    lam = {v: k for k, v in a.lam_of.items()}
    for t in a.j:
        J, jx = wt.cap_read(t)
        LJ = wt.ledger_read(t)
        dJ = dt.desire_read(t)
        rl = rely_of(t)
        ade = wt.dev_ade(t)
        dv = wt.dev_read(t, a.c)
        half = {k: LC[k][0].mean() + 0.5 * (LP0[k] - LC[k][0].mean()) for k in LC}
        g = dict(wait_W=J["wait_W"] >= 0.60, **{f"half_{k}": float(LJ[k][0].mean()) >= half[k] for k in ("S_L", "Y_P", "Y_V")},
                 desire4=dJ["arm"]["mean4"] >= 0.8 * dJ["P0"]["mean4"], dev_ade=ade <= c_ade + 0.01, rely=bool(rl["read"]))
        dWm, dD = J["wait_W"] - Cc["wait_W"], J["D"] - Cc["D"]
        res["arms"][t] = dict(lam=lam[t], cap=J, dev_ade=ade, dev=dv, rely=rl, desire=dJ["arm"], half_line=half, gates=g, passed=bool(all(g.values())),
                              ledger_F={k: dict(J=float(LJ[k][0].mean()), C=float(LC[k][0].mean()), P0=LP0[k], n=int(len(LJ[k][0])),
                                                diff=stats.fmt(stats.paired(LJ[k][0], LC[k][0], groups=LJ[k][1]))) for k in LJ},
                              d_wait_W=stats.fmt(stats.paired(jx["wait"][jx["W"]], cx["wait"][cx["W"]], groups=jx["g"][jx["W"]])),
                              launch_lock_shape=bool(dWm >= 0.20 and dD < 0.5 * dWm))
    ok = [t for t in a.j if res["arms"][t]["passed"]]
    pick = min(ok, key=lambda t: float(lam[t])) if ok else None
    res |= dict(passed=ok, pick=pick, pick_lam=lam.get(pick), verdict="PASS" if pick else "STOP: no arm meets every line")
    L = ["# (iii) RETAIN pilot lines (RB0-F recipe, 10 000 steps, seed 0; open loop)", "",
         "| arm | wait W (>= 0.60) | go G | D | S_L F | Y_P F | Y_V F | desire 4 s (>= 0.8 x P0) | dev ADE (<= C + 0.01) | rely (read) | lat 2 s vs C | pass |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|--:|:-:|",
         f"| P0 | {P0c['wait_W']:.3f} | {P0c['go_G']:.3f} | {P0c['D']:.3f} | {LP0['S_L']:.3f} | {LP0['Y_P']:.3f} | {LP0['Y_V']:.3f} | {dC['P0']['mean4']:.2f} | - | - | - | - |",
         f"| {a.c} (C) | {Cc['wait_W']:.3f} | {Cc['go_G']:.3f} | {Cc['D']:.3f} | " + " | ".join(f"{res['C']['ledger_F'][k]:.3f}" for k in ("S_L", "Y_P", "Y_V"))
         + f" | {dC['arm']['mean4']:.2f} | {c_ade:.4f} | {res['C']['rely']['read']} | - | - |"]
    for t in a.j:
        r = res["arms"][t]
        L.append(f"| {t} (lambda {r['lam']}) | {r['cap']['wait_W']:.3f} | {r['cap']['go_G']:.3f} | {r['cap']['D']:.3f} | "
                 + " | ".join(f"{r['ledger_F'][k]['J']:.3f} (>= {r['half_line'][k]:.3f})" for k in ("S_L", "Y_P", "Y_V"))
                 + f" | {r['desire']['mean4']:.2f} | {r['dev_ade']:.4f} | {r['rely']['read']} | {r['dev']['lat2s_med_absdiff']:.3f} | {'PASS' if r['passed'] else 'fail'} |")
    L += ["", "report only: launch-lock shape (wait W up >= 0.20 with D up < half of it): "
          + "; ".join(f"{t} {res['arms'][t]['launch_lock_shape']}" for t in a.j),
          "paired vs C (log-clustered): " + "; ".join(f"{t}: wait W {res['arms'][t]['d_wait_W']}; " + ", ".join(f"{k} {v['diff']}" for k, v in res['arms'][t]['ledger_F'].items()) for t in a.j),
          "", f"verdict: {res['verdict']}" + (f"; lambda {res['pick_lam']} ({pick})" if pick else "")]
    write("rt_pilot_gate", res, L, a.dry)


# ---------------------------------------------------------------- (iv) ANTI-COPY lines (CPU)
def cmd_acgate(a):
    dose = {v: k for k, v in a.dose_of.items()}
    c_ade = wt.dev_ade(a.c)
    Cc, _ = wt.cap_read(a.c)
    res = dict(control=a.c, C=dict(dev_ade=c_ade, wait_W=Cc["wait_W"]), arms={})
    for t in a.j:
        h = hist_read(t, a.c)
        dv = wt.dev_read(t, a.c)
        ade = wt.dev_ade(t)
        J, _ = wt.cap_read(t)
        s8, s1 = h["s0.8"], h["s1.0"]
        g = dict(dev_ade=ade <= c_ade + 0.01, slow_hist_bias=s8["d_abs_hi"] < 0, own_hist_bias=abs(s1["bias"] - s1["bias_c"]) <= 0.05,
                 lat2s=dv["lat2s_med_absdiff"] <= 0.05)
        res["arms"][t] = dict(dose=dose[t], dev_ade=ade, hist=h, dev=dv, wait_W=J["wait_W"], gates=g, passed=bool(all(g.values())))
    ok = [t for t in a.j if res["arms"][t]["passed"]]
    pick = min(ok, key=lambda t: float(dose[t])) if ok else None
    res |= dict(passed=ok, pick=pick, verdict="PASS" if ok else "STOP: no arm meets every line")
    L = ["# (iv) ANTI-COPY pilot lines (RSG-L18 recipe + history-speed noise, 1 000 steps, seed 0; open loop)", "",
         f"low-v0 dev rows: {V_LO} <= v0 < {V_HI} m/s, n {res['arms'][a.j[0]]['hist']['rows']} ({res['arms'][a.j[0]]['hist']['logs']} logs)", "",
         "| arm | dev ADE (<= C + 0.01) | 4 s arc bias, history x 0.8 (arm / C) | abs bias diff [CI] (hi < 0) | 4 s bias, own history (arm / C; within 0.05) | "
         "history gain d arc4 / d s (arm / C) | lat 2 s vs C (<= 0.05 m) | wait W | pass |", "|---|--:|--:|--:|--:|--:|--:|--:|:-:|",
         f"| {a.c} (C) | {c_ade:.4f} | - | - | - | - | - | {Cc['wait_W']:.3f} | - |"]
    for t in a.j:
        r = res["arms"][t]
        h = r["hist"]
        L.append(f"| {t} (M {r['dose']}) | {r['dev_ade']:.4f} | {h['s0.8']['bias']:+.2f} / {h['s0.8']['bias_c']:+.2f} | {h['s0.8']['d_abs']} | "
                 f"{h['s1.0']['bias']:+.2f} / {h['s1.0']['bias_c']:+.2f} | {h['gain']:.2f} / {h['gain_c']:.2f} | {r['dev']['lat2s_med_absdiff']:.3f} | "
                 f"{r['wait_W']:.3f} | {'PASS' if r['passed'] else 'fail'} |")
    L += ["", f"verdict: {res['verdict']}" + (f"; passing {', '.join(ok)}" if ok else "")]
    write("ac_pilot_gate", res, L, a.dry)
    if a.submit and ok and not a.dry:
        subprocess.run(["bash", str(HERE / "nx.sh"), "ac-held", ",".join(ok)], check=True)


def cmd_acheld(a):
    """Held 460 @18 per arm vs C (one seed): zeros, mean, the session-clustered paired CI, distance, offroad (wt.xpair); turns (rf_follow)."""
    sys.path.insert(0, str(REPO / "experiments/route_follow/scripts"))
    import rf_recov_post as RP
    J, C = a.j.split(","), a.c
    MC = wt.metrics(glob.glob(wt.xdirs("held", C)))
    res = dict(C=C, arms={})
    for t in J:
        r = wt.xpair(wt.metrics(glob.glob(wt.xdirs("held", t))), MC)
        h = RP.h2_held(NX / "acfollow", [t], [C])
        r["turns"] = None if h is None else dict(completed={k: v["completed"] for k, v in h["counts"].items()}, paired=h["paired"])
        r["line_zeros"] = bool(r["J"]["zeros"] <= r["C"]["zeros"] + 3)
        r["line_turns"] = bool(h is not None and h["counts"][t]["completed"] >= h["counts"][C]["completed"] - 2)
        res["arms"][t] = r
    L = ["# (iv) ANTI-COPY held 460 @18 (one seed) vs " + C, "", "| arm | zeros (<= C + 3) | mean | paired [session CI] | progress_rel | offroad | at-fault | turns completed (>= C - 2) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in res["arms"].items():
        tc = r["turns"]["completed"] if r["turns"] else {}
        L.append(f"| {t} | {r['J']['zeros']} (C {r['C']['zeros']}) | {r['J']['mean']:.3f} (C {r['C']['mean']:.3f}) | {r['paired']} | "
                 f"{r['J']['progress_rel']:.3f} (C {r['C']['progress_rel']:.3f}) | {r['J']['offroad']} (C {r['C']['offroad']}) | {r['J']['at_fault']} (C {r['C']['at_fault']}) | "
                 f"{tc.get(t)} (C {tc.get(C)}) |")
    write("ac_held", res, L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bitcmp")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--out", required=True)
    p.add_argument("--hash", default=""), p.add_argument("--note", default="")
    sp.add_parser("hist").add_argument("--tags", nargs="+", required=True)
    kv = lambda s: dict(x.split("=", 1) for x in s.split(","))  # noqa: E731
    p = sp.add_parser("rtgate")
    p.add_argument("--j", nargs="+", required=True), p.add_argument("--c", required=True), p.add_argument("--lam-of", type=kv, required=True)
    p.add_argument("--dry", action="store_true", help="print only (a check of the read code on existing tags)")
    p = sp.add_parser("acgate")
    p.add_argument("--j", nargs="+", required=True), p.add_argument("--c", required=True), p.add_argument("--dose-of", type=kv, required=True)
    p.add_argument("--submit", action="store_true"), p.add_argument("--dry", action="store_true", help="print only")
    p = sp.add_parser("acheld")
    p.add_argument("--j", required=True), p.add_argument("--c", required=True)
    a = ap.parse_args()
    dict(bitcmp=cmd_bitcmp, hist=cmd_hist, rtgate=cmd_rtgate, acgate=cmd_acgate, acheld=cmd_acheld)[a.cmd](a)
