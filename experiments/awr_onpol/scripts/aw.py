#!/usr/bin/env python3
"""AWR-ONPOL gates (plans/2026-10-17-awr-onpol-prereg.md, section 4). Box, envs/op-train, CPU. Chain: aw.sh.

  bitcmp --a TAG --b TAG [--hash H] --out NAME   two logged trainings identical: every scalar, every trained tensor, dev ADE, on_hash_60 (= H)
  pref --tags T,..                               the outcome proxy PREF on the dev logs (never drawn): per near-identical dev pair (same scene and
        time, other policy, rear axle <= 0.5 m, heading <= 3 deg, speed <= 0.5 m/s; score gap >= 0.1), margin = ADE(plan, loser's plan) -
        ADE(plan, winner's plan), the plan being the tag's plan on each of the two rows' inputs (mean of both) -> runs/awr_onpol/pref/<tag>.npz
  gate --stage pilot --j T,T --c C               G1 dev ADE <= C + 0.01; G2 lateral: rely read and equal-arc turn on held01 vs C >= -0.3 deg; G3 PREF
        margin J - C, log-clustered CI lo > 0; G4 contact on the new nuPlan dev rows J - C (rf_rsgacc.contact) not higher (CI lo <= 0)
        -> results/pilot.{md,json}, the passing tags -> $O/pass.txt; exit 0 = some arm passes
  gate --stage held --j T --c RSG-L18-s0 --dir DIR   held 460 @18 (one read) vs the stored C: zeros J <= C + 3, turns completed J >= C - 2
        (wt.xpair, rf_recov_post.h2_held on DIR/follow) -> results/held_<T>.{md,json}; exit 0 = both lines hold
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_), str(R_ / "experiments/route_follow/scripts"), str(R_ / "experiments/wait_cap/scripts"), str(R_ / "experiments/op_parity/scripts")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = D / "runs/awr_onpol"
RES = R_ / "experiments/awr_onpol/results"
PACK = O / "pack/on1_RSGACC.npz"
FWD = D / "runs/route_follow/rsgacc/fwd/RSGACC-dev-rec"
GAP, TOL_XY, TOL_YAW, TOL_V = 0.1, 0.5, np.radians(3.0), 0.5
ADE_TOL, TURN = 0.01, -0.3


def out(name, md, js):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.md").write_text("\n".join(md) + "\n")
    (RES / f"{name}.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md), flush=True)


def cmd_bitcmp(a):
    import wt
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
    out(a.out, [f"# {a.out}: {a.a} vs {a.b}", "", a.note, "", "```", json.dumps(r, indent=1, default=float), "```"], r)
    raise SystemExit(0 if r["identical"] else 2)


def dev_pairs(z):
    """-> (winner rows, loser rows, logs) of the near-identical dev pairs with score gap >= GAP (pool row indices)."""
    s = z["s"].astype(float)
    dv = (z["dev"].astype(str) == "1") & np.isfinite(s) & np.isfinite(z["target"]).all((1, 2))
    r = np.flatnonzero(dv)
    key = np.char.add(np.char.add(z["track"][r].astype(str), "|"), np.char.add(z["scene"][r].astype(str), np.char.add("|", z["t"][r].astype(str))))
    o = np.argsort(key, kind="stable")
    ks, ro = key[o], r[o]
    cut = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    W, L = [], []
    for a_, b_ in zip(cut[:-1], cut[1:]):
        g = ro[a_:b_]
        for i in range(len(g)):
            for j in range(len(g)):
                p, q = g[i], g[j]
                if z["pol"][p] == z["pol"][q] or s[p] - s[q] < GAP:
                    continue
                dyw = abs((z["yaw"][p] - z["yaw"][q] + np.pi) % (2 * np.pi) - np.pi)
                if np.hypot(*(z["xy"][p] - z["xy"][q])) <= TOL_XY and dyw <= TOL_YAW and abs(z["v"][p] - z["v"][q]) <= TOL_V:
                    W.append(p)
                    L.append(q)
    W, L = np.array(W, int), np.array(L, int)
    return W, L, z["log"][W].astype(str)


def ade(P, T):
    return np.hypot(P[..., 0] - T[..., 0], P[..., 1] - T[..., 1]).mean(-1)


def pref_of(tag, z, W, L):
    f = np.load(FWD / f"plans_{tag}.npz")
    pos = np.full(len(z["names"]), -1)
    pos[f["rows"]] = np.arange(len(f["rows"]))
    assert (pos[W] >= 0).all() and (pos[L] >= 0).all(), f"{tag}: dev plans miss pair rows"
    tw, tl = z["target"][W], z["target"][L]
    m = [ade(f["poses"][pos[X]], tl) - ade(f["poses"][pos[X]], tw) for X in (W, L)]
    return 0.5 * (m[0] + m[1])


def cmd_pref(a):
    z = np.load(PACK)
    W, L, lg = dev_pairs(z)
    (O / "pref").mkdir(parents=True, exist_ok=True)
    for t in a.tags.split(","):
        np.savez(O / "pref" / f"{t}.npz", w=W, l=L, log=lg, margin=pref_of(t, z, W, L).astype(np.float32),
                 track=z["track"][W].astype(str), fail_l=z["fail"][L], gap=(z["s"][W] - z["s"][L]).astype(np.float32))
        print(t, len(W), "pairs", flush=True)


def gate_pilot(J, C):
    import rf_recov as RC
    import rf_rsgacc as RA
    from jevdrive import stats
    cg, _ = RC.trainer_gates(C)
    z = np.load(PACK)
    W, L, lg = dev_pairs(z)
    mc = pref_of(C, z, W, L)
    rows, md = {}, ["# AWR-ONPOL pilot offline gates (1 000 steps, seed 0), each arm vs the same-command control " + C, "",
                    f"Lines: G1 dev ADE <= C + {ADE_TOL}; G2 lateral: rely read and equal-arc turn (held01) vs C >= {TURN} deg; G3 PREF margin J - C, "
                    "log-clustered CI lo > 0; G4 contact on the new nuPlan dev rows J - C, CI lo <= 0 (not higher).", "",
                    f"dev pairs (score gap >= {GAP}, near-identical): {len(W)} ({int((z['track'][W] == 'nu').sum())} nuPlan, "
                    f"{int((z['track'][W] == 'pai').sum())} PAI), {len(set(lg))} logs; loser failed in {int(z['fail'][L].sum())}", "",
                    "| arm | dev ADE (C) | rely | turn equal-arc vs C (deg) | PREF margin J - C (m) [CI] | PREF fail pairs | contact nuPlan dev J / C, d [CI] | contact PAI dev | lines |",
                    "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    passing = []
    for t in J:
        tg, _ = RC.trainer_gates(t)
        rl = tg["rely"] or {}
        turn, _ = RC.gate_turn(t, C)
        mj = pref_of(t, z, W, L)
        p = stats.paired(mj, mc, groups=lg)
        ff = z["fail"][L]
        pf = stats.paired(mj[ff], mc[ff], groups=lg[ff]) if ff.sum() >= 10 else None
        ct = RA.contact(t, C)
        cn, cp = ct["nuplan"], ct["pai"]
        Lz = dict(G1=tg["dev_ade"] is not None and tg["dev_ade"] <= cg["dev_ade"] + ADE_TOL,
                  G2=bool(rl.get("read")) and turn["equal_arc"]["dh_mean"] >= TURN, G3=p["lo"] > 0, G4=cn is not None and cn["diff"][1] <= 0)
        ok = all(Lz.values())
        passing += [t] if ok else []
        rows[t] = dict(dev_ade=tg["dev_ade"], dev_ade_c=cg["dev_ade"], rely=rl.get("read"), turn=turn["equal_arc"], pref=p, pref_fail=pf,
                       pref_arm=float(mj.mean()), pref_c=float(mc.mean()), contact=ct, lines=Lz, pass_=ok)
        f3 = lambda v, q=4: f"{v[0]:+.{q}f} [{v[1]:+.{q}f}, {v[2]:+.{q}f}]"  # noqa: E731
        md.append(f"| {t} | {tg['dev_ade']:.4f} ({cg['dev_ade']:.4f}) | {rl.get('read')} | {turn['equal_arc']['dh_mean']:+.2f} "
                  f"[{turn['equal_arc']['ci'][0]:+.2f}, {turn['equal_arc']['ci'][1]:+.2f}] | {f3((p['mean'], p['lo'], p['hi']))} | "
                  f"{f3((pf['mean'], pf['lo'], pf['hi'])) if pf else '-'} | "
                  + (f"{cn['rate']:.4f} / {cn['rate_ctl']:.4f}, {f3(cn['diff'])} | " if cn else "- | ")
                  + (f"{cp['rate']:.4f} / {cp['rate_ctl']:.4f}, {f3(cp['diff'])} | " if cp else "- | ")
                  + " ".join(f"{k} {'ok' if v else 'NO'}" for k, v in Lz.items()) + " |")
    md += ["", f"control PREF margin {mc.mean():+.4f} m; passing arms (go to held 460 @18): {passing or 'none: stop, no closed loop'}"]
    (O / "pass.txt").write_text("".join(f"{t}\n" for t in passing))
    out("pilot", md, dict(control=C, pairs=int(len(W)), rows=rows, passing=passing))
    return 0 if passing else 3


def gate_held(J, C, d):
    import rf_recov_post as RP
    import wt
    MC = wt.metrics(glob.glob(wt.xdirs("held", C)))
    r = wt.xpair(wt.metrics(glob.glob(wt.xdirs("held", J))), MC)
    h = RP.h2_held(Path(d), [J], [C])
    r["turns"] = None if h is None else dict(completed={k: v["completed"] for k, v in h["counts"].items()}, paired=h["paired"], common=h["common_turns"])
    tc = (r["turns"] or {}).get("completed", {})
    Lz = {"zeros <= C + 3": r["J"]["zeros"] <= r["C"]["zeros"] + 3, "turns >= C - 2": h is not None and tc[J] >= tc[C] - 2}
    md = [f"# AWR-ONPOL held 460 @18 (one read, {r['scenes']} paired scenes): {J} vs stored {C}", "",
          "| vs | zeros (<= C + 3) | mean | paired [session CI] | progress_rel | offroad | at-fault | turns completed (>= C - 2) |", "|---|--:|--:|--:|--:|--:|--:|--:|",
          f"| {C} | {r['J']['zeros']} (C {r['C']['zeros']}) | {r['J']['mean']:.3f} (C {r['C']['mean']:.3f}) | {r['paired']} | {r['J']['progress_rel']:.3f} "
          f"(C {r['C']['progress_rel']:.3f}) | {r['J']['offroad']} (C {r['C']['offroad']}) | {r['J']['at_fault']} (C {r['C']['at_fault']}) | {tc.get(J)} (C {tc.get(C)}) |",
          "", " ; ".join(f"{k}: {'holds' if v else 'FAILS'}" for k, v in Lz.items()),
          f"new zeros {r['zero_flips']['J_only']}; fixed {r['zero_flips']['C_only']}"]
    out(f"held_{J}", md, dict(J=J, C=C, res=r, lines=Lz))
    return 0 if all(Lz.values()) else 3


def cmd_gate(a):
    J = [t for t in a.j.split(",") if t]
    rc = gate_pilot(J, a.c) if a.stage == "pilot" else gate_held(J[0], a.c, a.dir)
    raise SystemExit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bitcmp")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--hash", default="")
    p.add_argument("--out", required=True), p.add_argument("--note", default="")
    p = sub.add_parser("pref")
    p.add_argument("--tags", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--stage", required=True, choices=("pilot", "held"))
    p.add_argument("--j", required=True), p.add_argument("--c", required=True), p.add_argument("--dir", default="")
    a = ap.parse_args()
    dict(bitcmp=cmd_bitcmp, pref=cmd_pref, gate=cmd_gate)[a.cmd](a)
