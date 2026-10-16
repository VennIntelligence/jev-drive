#!/usr/bin/env python3
"""STACK (experiments/route_follow/plans/2026-10-16-stack-prereg.md): flags, identity gates and pilot gates of the joint arm. Box; envs/op-train.

  rf_stack.py flags [--rates FILE]     the arm's trainer flags per dose level (A: tau = 1.0 x reference, pref 1; B: tau = 0.5 x, pref 3) from the
        reference violation rates (C2's rates JSON: human log-future rates of agent / road / cons) -> $SK/flags_{A,B}.txt; cons only when C2
        passed (results/stack/c2.json pass) and the head exists
  rf_stack.py bitcmp                   the 60-step identity run STKG-off (new code, every STACK switch off, RSG-L18 recipe) against JDGG-off (the
        same recipe on the code before STACK): scalars, final weights, on_hash_60 = RSG-L18-s0's -> $SK/gates.json (exit 2 unless identical)
  rf_stack.py rx0                      GPU: RB0-F-s0 with the reflex module attached at its init vs without, the same pool rows with valid
        self-states: outputs bit-identical (the module adds exact zeros) -> $SK/rx0.json
  rf_stack.py gate --tag T [--base RSG-L18-s0] [--submit-held]    the registered pilot gates: dev ADE <= 0.545, G-rely, equal-arc turning
        >= -0.3 deg (rf_recov.gate_turn on the held 460 decision 0 / 1 rows), served-plan contact rate on the dev own-state rows of
        on1_L18RSG down >= 30 % vs the base (rf_dagger.contacts geometry, any kind, 0.1 - 4 s; rows with a t0 contact out) -> $SK/gate/<T>.json;
        --submit-held: on PASS the held 460 @18 read + report (rf_recov_post.sh submit-held, KIND S, PRIO 9)
  rf_stack.py table                    every pilot gate (+ held lines when read) -> results/stack/pilot.md
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

R_ = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R_), str(HERE), str(R_ / "lib"), str(R_ / "experiments/onpol_nuplan/lib"), str(R_ / "experiments/onpol_nuplan/scripts"),
                str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/body1/lib"), str(R_ / "experiments/route_mem/lib"),
                str(R_ / "experiments/op_parity/scripts")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = D / "runs/op_parity/cache"
SK = D / "runs/route_follow/stack"
RES = R_ / "experiments/route_follow/results/stack"
LINES = dict(dev_ade=0.545, turn=-0.3, contact_drop=0.30)
LEVELS = dict(A=dict(level=1.0, pref=1.0), B=dict(level=0.5, pref=3.0))
RCVL = "--on on1_RCVL --on-k 20 --on-mode self --on-weight env=stop:0.99754,env=turn:1.34879,env=lead:0.56581,env=free:0.60637"
HEAD = SK / "cons/head.pt"


def cmd_flags(a):
    c2 = json.loads((RES / "c2.json").read_text()) if (RES / "c2.json").exists() else {}
    rates = json.loads(Path(a.rates).read_text()) if a.rates else c2.get("rates", {})
    hum = rates.get("human", rates)
    cons = bool(c2.get("pass")) and HEAD.exists()
    tau = {k: hum[k] for k in ("agent", "road") if k in hum}
    assert set(tau) == {"agent", "road"}, f"reference rates missing: {hum}"
    if cons:
        tau["cons"] = hum["cons"]
    rec = dict(tau_ref=tau, cons=cons, levels=LEVELS, flags={})
    for lv, d in LEVELS.items():
        cs = "agent,road,envU" + (",cons" if cons else "")
        ts = ",".join(f"{k}={v:.6f}" for k, v in tau.items()) + ",envU=auto"
        f = (f"{RCVL} --env-lam 1 --env-lo-scale 0 --rx --dual {cs} --dual-tau {ts} --dual-level {d['level']:g} --imit-drop --pref {d['pref']:g}"
             + (f" --cons {HEAD}" if cons else ""))
        (SK / f"flags_{lv}.txt").write_text(f + "\n")
        rec["flags"][lv] = f
    (SK / "flags.json").write_text(json.dumps(rec, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "flags.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def _run(tag):
    import torch
    d = sorted(glob.glob(f"{D}/runs/route_mem/train-{tag}/*/"))[-1]
    sc = {}
    for line in open(d + "events.jsonl"):
        e = json.loads(line)
        if e["kind"] == "scalar":
            sc[(e["tag"], e["step"])] = e["value"]
    return sc, json.load(open(d + "DONE")), torch.load(f"{D}/runs/op_parity/runs/{tag}/ckpt-final.pt", map_location="cpu", weights_only=False)["model"]


def cmd_bitcmp(a):
    RSG_HASH60 = "ea8cacc3e451c577"
    (sa, da, ma), (sb, db, mb) = _run("JDGG-off"), _run("STKG-off")
    keys = [k for k in sorted(set(sa) & set(sb), key=str) if not k[0].startswith("throughput/")]
    dl = max(abs(sa[k] - sb[k]) for k in keys)
    dw = max(float((ma[g][k].float() - mb[g][k].float()).abs().max()) for g in ("net", "parity") for k in ma[g])
    same = all(set(ma[g]) == set(mb[g]) for g in ("net", "parity"))
    r = dict(a="JDGG-off (code before STACK)", b="STKG-off (STACK code, every switch off)", scalars=len(keys), scalar_keys_equal=set(sa) == set(sb),
             scalar_max_abs_diff=dl, weight_max_abs_diff=dw, weight_keys_equal=same, on_hash_60=[da.get("on_hash_60"), db.get("on_hash_60")],
             rsg_l18_s0_on_hash_60=RSG_HASH60)
    r["identical"] = bool(dl == 0 and dw == 0 and same and r["on_hash_60"][0] == r["on_hash_60"][1] == RSG_HASH60)
    SK.mkdir(parents=True, exist_ok=True)
    (SK / "gates.json").write_text(json.dumps(r, indent=1, default=float))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gates.json").write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps(r, indent=1, default=float))
    raise SystemExit(0 if r["identical"] else 2)


def cmd_rx0(a):
    import torch
    from types import SimpleNamespace
    sys.path[:0] = [str(R_ / "experiments/route_mem/scripts")]
    import rm_train as RT
    from rf_lonread import REC
    dev = torch.device("cuda")
    res = {}
    outs = []
    for rx in (False, True):
        ns_a = SimpleNamespace(**REC, road_labels=RT.OT.JT.ROAD, on="on1_L18RSG", init="RB0-F-s0", rm_from="RB0-F-s0", tag="rx0", exp="route_follow",
                               lr_scale=1.0, prefetch=1, dev_card_gb=4.0, rx=rx)
        ns, *_ = RT.build(ns_a, dev)
        S, m = ns.S, ns.model.eval()
        n = S.n - ns.on0
        if rx:
            v = S.ego[ns.on0:, -14].cpu().numpy() > 0.5
            rr = ns.on0 + np.flatnonzero(v)[:512]
            res["valid_rows"] = int(v.sum())
        else:
            rr = None
        outs.append((m, S, ns))
    (m0, S0, n0), (m1, S1, n1) = outs
    with torch.no_grad():
        rt = torch.as_tensor(rr, device=dev)
        o0 = m0(S0.front[rr], S0.ego[rt], S0.tc[rt], None, None).float()
        o1 = m1(S1.front[rr], S1.ego[rt], S1.tc[rt], None, None).float()
        res |= dict(rows=int(len(rr)), max_abs_diff=float((o0 - o1).abs().max()), fed_cols=[int(S0.ego.shape[1]), int(S1.ego.shape[1])],
                    res_abs=float(m1.adapter.last_res.abs().max()))
    res["identical"] = res["max_abs_diff"] == 0.0
    SK.mkdir(parents=True, exist_ok=True)
    (SK / "rx0.json").write_text(json.dumps(res, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "rx0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["identical"] else 2)


def contact_read(tag, base):
    """Served-plan contact (agent or road, steps 1..40) on the dev own-state rows of on1_L18RSG: pilot vs base, paired, log-clustered."""
    import rf_recov as RC
    import rf_dagger as DG
    from jevdrive import stats
    rr, Pt = RC.plans("L18RSG", "dev-rec", tag)
    rb, Pb = RC.plans("L18RSG", "dev-rec", base)
    assert np.array_equal(rr, rb)
    n = len(np.load(CR / "on1_L18RSG" / "tab.npz")["names"])
    full = lambda P: (lambda X: (X.__setitem__(rr, P), X)[1])(np.full((n, 8, 3), np.nan, np.float32))  # noqa: E731
    out, okA, okR = DG.contacts("on1_L18RSG", {"t": full(Pt), "b": full(Pb)}, rr)
    lg = np.load(CR / "on1_L18RSG" / "tab.npz")["log"][rr].astype(str)
    keep = ~(out["t"]["t0"] | out["b"]["t0"])
    ct, cb = (out["t"]["agent"] | out["t"]["road"])[keep], (out["b"]["agent"] | out["b"]["road"])[keep]
    r = stats.paired(ct.astype(float), cb.astype(float), groups=lg[keep])
    res = dict(rows=int(keep.sum()), logs=len(set(lg[keep])), rate_pilot=float(ct.mean()), rate_base=float(cb.mean()),
               drop=float(1 - ct.mean() / max(cb.mean(), 1e-9)), diff=[r["mean"], r["lo"], r["hi"]])
    for k in ("agent", "road"):
        a_, b_ = out["t"][k][keep], out["b"][k][keep]
        res[k] = dict(pilot=float(a_.mean()), base=float(b_.mean()), drop=float(1 - a_.mean() / max(b_.mean(), 1e-9)))
    return res


def cmd_gate(a):
    import rf_recov as RC
    res, L = dict(tag=a.tag, base=a.base, lines=LINES), {}
    tg, _ = RC.trainer_gates(a.tag)
    res["trainer"] = tg
    L["dev_ade"] = tg["dev_ade"] is not None and tg["dev_ade"] <= LINES["dev_ade"]
    rl = tg["rely"] or {}
    L["rely"] = bool(rl.get("read")) if "read" in rl else None
    res["turn"], _ = RC.gate_turn(a.tag, a.base)
    L["turn"] = res["turn"]["equal_arc"]["dh_mean"] >= LINES["turn"]
    res["contact"] = contact_read(a.tag, a.base)
    L["contact"] = res["contact"]["drop"] >= LINES["contact_drop"]
    res["stack"] = _summary(a.tag)
    res["lines_ok"] = L
    res["pass"] = all(v is True for v in L.values())
    (SK / "gate").mkdir(parents=True, exist_ok=True)
    (SK / "gate" / f"{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"gate_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print("PASS" if res["pass"] else "FAIL", flush=True)
    if res["pass"] and a.submit_held:
        env = dict(os.environ, ARMROOT=str(SK), KIND="S", RESSUB="stack", PRIO=os.environ.get("PRIO", "9"))
        p = subprocess.run(["bash", str(HERE / "rf_recov_post.sh"), "submit-held", a.tag, a.tag, "-", "-", "-"], env=env, capture_output=True, text=True)
        print(p.stdout[-2000:], p.stderr[-2000:], flush=True)
        if p.returncode:
            raise SystemExit(f"held submit failed rc {p.returncode}")


def _summary(tag):
    f = sorted(glob.glob(f"{D}/runs/route_mem/train-{tag}/*/summary.json"))
    if not f:
        return None
    s = json.loads(open(f[-1]).read())
    return dict(stack=s.get("stack"), rx_feats=s.get("rx_feats"), train_s=s.get("train_s"), dev_ade=s.get("dev_ade"))


def cmd_table(a):
    rows = []
    for f in sorted((SK / "gate").glob("*.json")):
        g = json.loads(f.read_text())
        t = g["tag"]
        ln = SK / t / "lines.json"
        h = json.loads(ln.read_text()) if ln.exists() else None
        H = (h or {}).get("lines", {})
        fmt = lambda k, p=1: (lambda v: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]")(H[k]["v"]) if k in H else "-"  # noqa: E731
        tu, c = g["turn"]["equal_arc"], g["contact"]
        st = ((g.get("stack") or {}).get("stack") or {})
        lam = ", ".join(f"{k} {v:.3g}" for k, v in (st.get("lam") or {}).items() if k in (st.get("dual") or []))
        rows.append(f"| {t} | {g['trainer']['dev_ade']:.4f} | {g['lines_ok']['rely']} | {tu['dh_mean']:+.2f} [{tu['ci'][0]:+.2f}, {tu['ci'][1]:+.2f}] | "
                    f"{c['rate_base']:.3f} -> {c['rate_pilot']:.3f} ({-100 * c['drop']:+.1f} %) | {lam} | {'PASS' if g['pass'] else 'FAIL'} | "
                    f"{fmt('H1_zeros')} | {fmt('H1_at_fault')} | {fmt('H1_gt', 3)} | " + ({None: "-", True: "PASS", False: "FAIL"}[h.get('H1')] if h else "-") + " |")
    md = ["# STACK pilot gates", "", "Lines: dev ADE <= 0.545; G-rely true; equal-arc turning (pilot - RSG-L18-s0) >= -0.3 deg; served-plan contact rate on the "
          "dev own-state rows of on1_L18RSG down >= 30 %; then held 460 @18 H1 vs the RSG-L18 2-seed mean (zeros <= +3, at-fault <= +3, GT <= +0.03 m).", "",
          "| pilot | dev ADE | rely | equal-arc turn (deg) | contact rate dev own rows | final multipliers | offline | held zeros d | at-fault d | GT d (m) | H1 |",
          "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|"] + rows
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "pilot_gates.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("flags")
    p.add_argument("--rates", default="")
    sp.add_parser("bitcmp")
    sp.add_parser("rx0")
    p = sp.add_parser("gate")
    p.add_argument("--tag", required=True), p.add_argument("--base", default="RSG-L18-s0"), p.add_argument("--submit-held", action="store_true")
    sp.add_parser("table")
    a = ap.parse_args()
    SK.mkdir(parents=True, exist_ok=True)
    dict(flags=cmd_flags, bitcmp=cmd_bitcmp, rx0=cmd_rx0, gate=cmd_gate, table=cmd_table)[a.cmd](a)
