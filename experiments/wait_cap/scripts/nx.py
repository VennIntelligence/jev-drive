#!/usr/bin/env python3
"""Decision 335's two follow-up pilots (prereg draft plans/2026-10-17-retain-anticopy-draft.md); box, repo root; chain in nx.sh.
Outputs: box runs/wait_cap/next/, repo results/next/.

  (iii) RETAIN  P0's speed profile (arc length only, lateral untouched) distilled on rolling-arrival / standing-wait rows of on1_WAIT and on
                navtrain approach rows fed the constant-velocity history, + D2 desire-delta distillation, lambda in {1, 3} (rm_train --kd-lam,
                --kd-nav, --d2-lam; decision 336's aim). Variant a: a 1 000-step patch on the RSG-L18 recipe (C = WTCP-L18-s0); variant b: the
                RB0-F base stage rebuilt, 10 000 steps from the shipped weights (C = RB0-F-s0). Both controls are the flags-off command (`bitcmp`).
  (iv) ANTI-COPY  the RSG-L18 recipe (1 000 steps, WAIT pilot's shape) + history-speed noise (rm_train --ego-scale-speed 0.5, --ego-scale-max in
                {0.2, 0.4}); control = WTCP-L18-s0.

  bitcmp --a A --b B --out NAME [--hash H] [--note ..]   two runs: every logged scalar, every weight, dev ADE, on_hash_60 (and = H if given)
                -> results/next/NAME.json; exit 2 unless identical
  hist --tags T..   (GPU)  the gate-4 dev rows (rm_train dev_eval's 1 789 rows), the speed history rescaled by s in {0.8, 1.0, 1.2} (rm_train
                --ego-scale-speed's columns and powers) -> next/hist/<tag>.npz; s = 1 must reproduce the run's DONE dev_ade
  approach  (CPU)  navtrain rolling-approach rows (0.2-6 m/s, stopped lead <= 15 m or TTC < 4 s; wt.lead_ctx) -> next/approach_nav.npz
  arrival --tags T.. [--rmem-from RSG-L18-s0]   (GPU)  decision 336's replay on P0's closed-loop static-scene inputs (wt_xe xreplay's call)
                -> next/arrival/<tag>.json; line: 4 s arc at the first decision after the handoff (k = 1) <= 1.0 m
  rtgate --variant a|b --j T1 T3 --c C --lam-of 1=T1,3=T3 [--dry]   (CPU)  the (iii) lines -> results/next/rt<variant>_pilot_gate.{json,md}
  acgate --j T2 T4 --c C --dose-of 0.2=T2,0.4=T4 [--submit]   (CPU)  the (iv) lines -> results/next/ac_pilot_gate.{json,md}; --submit: held
                460 @18 for the passing arms (nx.sh ac-held), then `acheld`
  acheld --j T.. --c C   (CPU)  held 460 @18 per arm vs C (one seed; wt.xpair) + rf_follow turns -> results/next/ac_held.{json,md}
Prereg 补记 1 (the user's exploratory continuation of RTB1-F-s0 after its pilot STOP; no registered verdict file is touched):
  devdecomp --j T.. --c C --out NAME   (CPU)  wt_devdecomp on the gate-4 dev rows: plans = next/hist s = 1 (each reproduced its DONE dev ADE),
                P0 = 补记 D's devall (same rows) -> next/devdecomp/devall.npz -> results/next/NAME.{json,md}
  cont --j T --dir D --out NAME   (CPU)  one continuation arm vs RSG-L18-s0 and WTC-L18-s0: held 460 @18 (wt.xpair), 24 held turns (rf_follow in
                D/follow), the static public scene @10 (distance, contact), PAI stop17 zeros, arrival replay, dev ADE; references P0 / RSG-L18-s0 /
                WTCP-L18-s0 from 补记 D / E -> results/next/NAME.{json,md}, D/FINISHED
  wide --j T --dir D --out NAME [--no-pai]   (CPU)  public 1491 @10 (+ PAI 281 unless --no-pai) vs the stored RSG-L18-s0 reads
Prereg 补记 2 (two targeted variants of the retain base, exploratory):
  vgate --j BASE [--max-arc 2.0 --max-dev RTB1-F-s0's]   (CPU)  the stop rule: STOP (exit 3) if the arrival replay's 4 s arc at k = 1 > max-arc or
                the DONE dev ADE > max-dev (RTB1-F-s0's); the continuation and its reads hang on it with --after -> results/next/rtb_v_gate_BASE
  vsum --arms V=BASE:CONT,.. --out NAME   (CPU)  every arm (and a = RTB1O-L18-s0) vs a and RSG-L18-s0: held 460 @18, the 7 shared new zeros, static,
                stop17, arrival, public 1491 @10 with front / lateral / rear collisions, the bases' open-loop reads -> results/next/NAME.{json,md}
Prereg 补记 3 (PAI 281 of the V2 recipe; the second base seed of nx.sh rtb-v2s1 was cancelled by the user):
  v2pai --j TAG.. [--ctx TAG..] --out NAME [--dry]   (CPU)  PAI 281 in the submission config of the arms (rtb_wide_<TAG>.json of NX, nx.py wide) against the stored
                RSG-L18-s0 read, the decision 339 condition R2 on the --j arms, the cl-compile lane's PAI reads of RSG-L18-s0 / s1 as references -> results/next/NAME.{json,md}
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


# ---------------------------------------------------------------- (iii) RETAIN rows and reads
def cmd_approach(a):
    """(CPU) navtrain rows of a rolling approach: 0.2 <= v0 <= 6 m/s and, from the t0 agent labels (wt.lead_ctx), a stopped lead (< 0.5 m/s)
    <= 15 m ahead in the lane corridor, or a lead closing with time-to-contact < 4 s (wt_devdecomp's low-TTC rule) -> NX/approach_nav.npz."""
    names, v0 = [], []
    for d in sorted(glob.glob(str(wt.CR / "navtrain_full.s*of12" / "tab.npz")), key=lambda f: int(f.split(".s")[-1].split("of")[0])):
        z = np.load(d)
        names.append(z["names"].astype(str))
        v0.append(z["speed"].astype(float))
    names, v0 = np.concatenate(names), np.concatenate(v0)
    roll = (v0 >= 0.2) & (v0 <= 6.0)
    c = wt.lead_ctx(names[roll])
    vl = -c["lead_close"]                                   # lead x(t0) - x(0.5 s) in the t0 frame / 0.5 = -(lead speed)
    closing = v0[roll] - vl
    stopped = c["ctx_ok"] & (c["lead_gap"] <= 15.0) & (vl < 0.5)
    ttc = c["ctx_ok"] & (c["lead_gap"] < 20.0) & (closing > 0.1) & (c["lead_gap"] / np.maximum(closing, 1e-3) < 4.0)
    m = stopped | ttc
    NX.mkdir(parents=True, exist_ok=True)
    np.savez(NX / "approach_nav.npz", names=names[roll][m], v0=v0[roll][m], lead_gap=c["lead_gap"][m], lead_v=vl[m], stopped=stopped[m], ttc=ttc[m])
    r = dict(rows=int(len(names)), rolling=int(roll.sum()), labelled=int(c["ctx_ok"].sum()), approach=int(m.sum()), stopped_lead=int(stopped.sum()),
             closing_ttc4=int(ttc.sum()), v0_median=float(np.median(v0[roll][m])) if m.any() else None)
    write("approach_nav", r)


def cmd_arrival(a):
    """(GPU) decision 336's arrival replay: each tag's plan on the exact inputs P0 saw in its own closed-loop static-scene run (wt_xe.cmd_xreplay's
    call, measured ego row), the route memory logged by --rmem-from's static run at the same decision (the same memory for every tag)
    -> NX/arrival/<tag>.json (4 s arc per decision; k = 1 is the first decision after the handoff, 0.44 m/s, lead stopped 1.6 m ahead)."""
    import torch
    sys.path[:0] = [str(REPO / "experiments/alpasim/lib")]
    import sh30_core as SC
    import wt_xe as XE
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    out = NX / "arrival"
    out.mkdir(parents=True, exist_ok=True)
    dumps = sorted(glob.glob(str(XE.XE / "static/P0/run/driver-logs/dump/s00_k*.npz")), key=lambda f: int(f.split("_k")[-1][:-4]))
    Dz = [dict(np.load(f)) for f in dumps]
    rm = {}
    for l in open(XE.XE / f"static/{a.rmem_from}/run/driver-logs/drive.jsonl"):
        r = json.loads(l)
        if r["kind"] == "drive" and "rmem" in r:
            rm[int(r["k"])] = np.asarray(r["rmem"], np.float32)
    for tag in a.tags:
        if (out / f"{tag}.json").exists():
            continue
        core = SC.Core(tag, "cuda", "backwarp")
        arcs = []
        for k, d in enumerate(Dz):
            ego = d["ego"].astype(np.float32)
            row = ego if not core.rm else np.r_[ego, rm.get(k, np.zeros(core.rm_dim, np.float32))]
            cur, valid = d["cur"], d["valid"]
            prev = np.concatenate([np.zeros((1,) + cur.shape[1:], np.uint8), cur[:-1]])
            with torch.no_grad():
                pp, cc = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
                H = core.model.net.run_batched(core.A.vision_feeds(pp, cc), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *core.A.H_SHAPE)
                o = core.model(H, torch.from_numpy(row[None]).to(core.dev), torch.tensor([[1.0, 0.0]], device=core.dev))
                o = (o[0] if isinstance(o, tuple) else o).float()
            mu = o[0, core.pi].reshape(33, 15).cpu().numpy()
            P = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, d["cam_t"][:2].astype(np.float64), Z.T_OUT, "lever")
            arcs.append(dict(k=k, v=10 * float(d["ego"][4]), arc4=XE.arc(P), dpose_vs_logged=float(np.abs(P - d["poses"]).max()) if tag == "P0" else None))
        (out / f"{tag}.json").write_text(json.dumps(dict(tag=tag, rmem_from=a.rmem_from, arcs=arcs), indent=1))
        print(tag, [round(x["arc4"], 2) for x in arcs], flush=True)
        del core
        torch.cuda.empty_cache()


def arrival_of(tag):
    return json.loads((NX / "arrival" / f"{tag}.json").read_text())["arcs"]


def free_bias(tag, ctag):
    """4 s arc bias (plan - human, own history, NX/hist s = 1) on the free-road dev rows (wt_devdecomp's context rule on wt.lead_ctx), arm and C."""
    a, c = np.load(NX / "hist" / f"{tag}.npz"), np.load(NX / "hist" / f"{ctag}.npz")
    assert np.array_equal(a["rows"], c["rows"])
    x = wt.lead_ctx(a["names"])
    near = x["lead_gap"] < 20                               # wt_devdecomp: lead near (low TTC is inside it), pedestrian, else free road
    free = x["ctx_ok"] & ~near & ~x["ped"].astype(bool)
    h = arc4(a["fut"])
    ba, bc = arc4(a["plan_1.0"]) - h, arc4(c["plan_1.0"]) - h
    return dict(rows=int(free.sum()), bias=float(ba[free].mean()), bias_c=float(bc[free].mean()), diff=float(ba[free].mean() - bc[free].mean()))


def ledger_P0():
    """Ledger F cells of shipped Cinque (the P0 plans of led.py) -> {cell: success share}."""
    return {k: float(v[0].mean()) for k, v in wt.ledger_read("P0").items()}


def rely_of(tag):
    return json.loads((DATA / "runs/route_mem/rely" / f"{tag}.json").read_text())


def cmd_rtgate(a):
    """The (iii) lines of one variant (a: 1 000-step patch on the RSG-L18 recipe, C = WTCP-L18-s0; b: the RB0-F base rebuilt, C = RB0-F-s0)."""
    sys.path.insert(0, str(REPO / "experiments/desire_cap/scripts"))
    import dt
    from jevdrive import stats
    Cc, cx = wt.cap_read(a.c)
    P0c, _ = wt.cap_read("P0")
    LP0, LC = ledger_P0(), wt.ledger_read(a.c)
    dC = dt.desire_read(a.c)
    c_ade = wt.dev_ade(a.c)
    arrC, arrP0 = arrival_of(a.c)[1]["arc4"], arrival_of("P0")[1]["arc4"]
    res = dict(variant=a.variant, control=a.c, C=Cc | dict(dev_ade=c_ade, ledger_F={k: float(v[0].mean()) for k, v in LC.items()}, desire4=dC["arm"]["mean4"],
                                                         arrival_k1=arrC),
               P0=dict(wait_W=P0c["wait_W"], go_G=P0c["go_G"], D=P0c["D"], ledger_F=LP0, desire4=dC["P0"]["mean4"], arrival_k1=arrP0), arms={})
    if a.variant == "b":
        res["C"]["rely"] = rely_of(a.c)
    lam = {v: k for k, v in a.lam_of.items()}
    for t in a.j:
        J, jx = wt.cap_read(t)
        LJ = wt.ledger_read(t)
        dJ = dt.desire_read(t)
        ade = wt.dev_ade(t)
        dv = wt.dev_read(t, a.c)
        fb = free_bias(t, a.c)
        arr = arrival_of(t)
        half = {k: LC[k][0].mean() + 0.5 * (LP0[k] - LC[k][0].mean()) for k in LC}
        g = dict(arrival=arr[1]["arc4"] <= 1.0, **{f"half_{k}": float(LJ[k][0].mean()) >= half[k] for k in ("S_L", "Y_P", "Y_V")},
                 wait_W=J["wait_W"] >= 0.60, go_G=J["go_G"] >= 0.85, desire4=dJ["arm"]["mean4"] >= 0.8 * dJ["P0"]["mean4"],
                 dev_ade=ade <= c_ade + 0.01, free_bias=abs(fb["diff"]) <= 0.10, lat2s=dv["lat2s_med_absdiff"] <= 0.05)
        r = dict(lam=lam[t], cap=J, dev_ade=ade, dev=dv, free=fb, arrival=[x["arc4"] for x in arr], desire=dJ["arm"], half_line=half,
                 ledger_F={k: dict(J=float(LJ[k][0].mean()), C=float(LC[k][0].mean()), P0=LP0[k], n=int(len(LJ[k][0])),
                                   diff=stats.fmt(stats.paired(LJ[k][0], LC[k][0], groups=LJ[k][1]))) for k in LJ},
                 d_wait_W=stats.fmt(stats.paired(jx["wait"][jx["W"]], cx["wait"][cx["W"]], groups=jx["g"][jx["W"]])))
        if a.variant == "b":
            r["rely"] = rely_of(t)
            g["rely"] = bool(r["rely"]["read"])
        r |= dict(gates=g, passed=bool(all(g.values())))
        res["arms"][t] = r
    ok = [t for t in a.j if res["arms"][t]["passed"]]
    pick = min(ok, key=lambda t: float(lam[t])) if ok else None
    res |= dict(passed=ok, pick=pick, pick_lam=lam.get(pick), verdict="PASS" if pick else "STOP: no arm meets every line")
    name = {"a": "1 000-step patch on the RSG-L18 recipe", "b": "the RB0-F base stage rebuilt, 10 000 steps"}[a.variant]
    L = [f"# (iii-{a.variant}) RETAIN pilot lines ({name}; seed 0; open loop + the arrival replay)", "",
         "| arm | arrival 4 s arc @0.5 s (<= 1.0 m) | S_L F | Y_P F | Y_V F | wait W (>= 0.60) | go G (>= 0.85) | D | desire 4 s (>= 0.8 x P0) | "
         "dev ADE (<= C + 0.01) | free-road 4 s bias (C +- 0.10) | lat 2 s vs C (<= 0.05) |" + (" rely |" if a.variant == "b" else "") + " pass |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|" + ("--|" if a.variant == "b" else "") + ":-:|",
         f"| P0 | {arrP0:.2f} | {LP0['S_L']:.3f} | {LP0['Y_P']:.3f} | {LP0['Y_V']:.3f} | {P0c['wait_W']:.3f} | {P0c['go_G']:.3f} | {P0c['D']:.3f} | "
         f"{dC['P0']['mean4']:.2f} | - | - | - |" + (" - |" if a.variant == "b" else "") + " - |",
         f"| {a.c} (C) | {arrC:.2f} | " + " | ".join(f"{res['C']['ledger_F'][k]:.3f}" for k in ("S_L", "Y_P", "Y_V"))
         + f" | {Cc['wait_W']:.3f} | {Cc['go_G']:.3f} | {Cc['D']:.3f} | {dC['arm']['mean4']:.2f} | {c_ade:.4f} | - | - |"
         + (f" {res['C']['rely']['read']} |" if a.variant == "b" else "") + " - |"]
    for t in a.j:
        r = res["arms"][t]
        L.append(f"| {t} (lambda {r['lam']}) | {r['arrival'][1]:.2f} | "
                 + " | ".join(f"{r['ledger_F'][k]['J']:.3f} (>= {r['half_line'][k]:.3f})" for k in ("S_L", "Y_P", "Y_V"))
                 + f" | {r['cap']['wait_W']:.3f} | {r['cap']['go_G']:.3f} | {r['cap']['D']:.3f} | {r['desire']['mean4']:.2f} | {r['dev_ade']:.4f} | "
                 f"{r['free']['bias']:+.2f} (C {r['free']['bias_c']:+.2f}) | {r['dev']['lat2s_med_absdiff']:.3f} |"
                 + (f" {r['rely']['read']} |" if a.variant == "b" else "") + f" {'PASS' if r['passed'] else 'fail'} |")
    L += ["", "arrival replay, 4 s arc per decision (k = 0 .. 9): " + "; ".join(f"{t} " + " ".join(f"{x:.2f}" for x in res["arms"][t]["arrival"]) for t in a.j),
          "paired vs C (log-clustered): " + "; ".join(f"{t}: wait W {res['arms'][t]['d_wait_W']}; " + ", ".join(f"{k} {v['diff']}" for k, v in res['arms'][t]['ledger_F'].items()) for t in a.j),
          "", f"verdict: {res['verdict']}" + (f"; lambda {res['pick_lam']} ({pick})" if pick else "")]
    write(f"rt{a.variant}_pilot_gate", res, L, a.dry)


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


# ---------------------------------------------------------------- prereg 补记 1: RTB1 continuation (exploratory)
XD, XE = wt.OUT / "explore", wt.OUT / "explore_e"
REFS = ("RSG-L18-s0", "WTC-L18-s0", "RSG-L18-s1", "WTC-L18-s1")         # s1: the seed-matched references of the 补记 1 addition
CL_REF = {"P0": (XE / "static/P0/run", XE / "paisub/P0"), "RSG-L18-s0": (XE / "static/RSG-L18-s0/run", XE / "paisub/RSG-L18-s0"),
          "WTCP-L18-s0": (XD / "sub/WTCP-L18-s0/run", XD / "paisub/WTCP-L18-s0")}


def cmd_devdecomp(a):
    import shutil
    import wt_devdecomp as DD
    d = np.load(XD / "devall.npz")
    H = {t: np.load(NX / "hist" / f"{t}.npz") for t in a.j + [a.c]}
    for t, h in H.items():
        assert np.array_equal(h["rows"], d["rows"]) and np.allclose(h["fut"], d["fut"], atol=1e-5), t
    w = np.load(NX / "hist" / "WTCP-L18-s0.npz")["plan_1.0"]                 # the hist read (no side input) against devall's (side input) plan
    print("WTCP-L18-s0 hist s = 1 vs devall plan: max |diff|", float(np.abs(w - d["plan_WTCP-L18-s0"]).max()), "m")
    o = NX / "devdecomp"
    o.mkdir(parents=True, exist_ok=True)
    np.savez(o / "devall.npz", **{k: d[k] for k in ("rows", "names", "log", "fut", "speed", "plan_P0", "lead_gap", "lead_close", "ped", "ctx_ok")},
             tags=np.array(["P0"] + a.j + [a.c]), **{f"plan_{t}": H[t]["plan_1.0"] for t in H})
    DD.main(str(o / "devall.npz"), str(o / a.out), ctl=a.c)
    RES.mkdir(parents=True, exist_ok=True)
    for x in ("json", "md"):
        shutil.copy(o / f"{a.out}.{x}", RES / f"{a.out}.{x}")


def static_of(run):
    f = Path(run) / "aggregate/results-summary.json"
    if not f.exists():
        return None
    r = next((r for r in json.loads(f.read_text())["rollouts"] if r["clipgt_id"] == wt.STATIC_PUB), None)
    m = (r or {}).get("metrics") or {}
    return None if r is None else dict(score=float(r.get("score") or 0.0), dist=float(m.get("dist_traveled_m", np.nan)),
                                       collision=bool(m.get("collision_any", 0) > 0), min_gap=float(m.get("min_distance_to_obstacle_m", np.nan)))


def stop17_of(d):
    m = wt.metrics(glob.glob(f"{d}/runs/*/sim"))
    full = [wt.pai_full(p) for p in wt.STOP17]
    return dict(zeros=sum(m[s][0] == 0 for s in full if s in m), n=sum(s in m for s in full))


def cmd_cont(a):
    sys.path.insert(0, str(REPO / "experiments/route_follow/scripts"))
    import rf_recov_post as RP
    t, D = a.j, Path(a.dir)
    MJ = wt.metrics(glob.glob(wt.xdirs("held", t)))
    res = dict(arm=t, held={}, cl={}, arrival={}, dev_ade={})
    for c in REFS:
        r = wt.xpair(MJ, wt.metrics(glob.glob(wt.xdirs("held", c))))
        h = RP.h2_held(D, [t], [c])
        r["turns"] = None if h is None else dict(completed={k: v["completed"] for k, v in h["counts"].items()}, turns={k: v["turns"] for k, v in h["counts"].items()},
                                                 paired=h["paired"], common=h["common_turns"])
        r["line_zeros"] = bool(r["J"]["zeros"] <= r["C"]["zeros"] + 3)
        r["line_turns"] = bool(h is not None and h["counts"][t]["completed"] >= h["counts"][c]["completed"] - 2)
        res["held"][c] = r
    for k, (st, pd) in {t: (D / f"static/{t}/run", D / f"paisub/{t}"), **CL_REF}.items():
        res["cl"][k] = dict(static=static_of(st), stop17=stop17_of(pd))
    for k in (t, "RTB1-F-s0", "RB0-F-s0", "RSG-L18-s0", "WTCP-L18-s0", "P0"):
        f = NX / "arrival" / f"{k}.json"
        res["arrival"][k] = arrival_of(k)[1]["arc4"] if f.exists() else None
        if k != "P0":
            res["dev_ade"][k] = wt.dev_ade(k)
    s = res["cl"][t]["static"]
    res["stop_seen"] = bool((s is not None and not s["collision"] and s["dist"] <= 1.0) or res["cl"][t]["stop17"]["zeros"] <= 15)
    L = [f"# 补记 1 (exploratory): {t} closed loop, one seed", "", "| vs | zeros (<= C + 3) | mean | paired [session CI] | progress_rel | offroad | at-fault | "
         "turns /24 (>= C - 2) |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for c, r in res["held"].items():
        tc = (r["turns"] or {}).get("completed", {})
        L.append(f"| {c} | {r['J']['zeros']} (C {r['C']['zeros']}) | {r['J']['mean']:.3f} (C {r['C']['mean']:.3f}) | {r['paired']} | {r['J']['progress_rel']:.3f} "
                 f"(C {r['C']['progress_rel']:.3f}) | {r['J']['offroad']} (C {r['C']['offroad']}) | {r['J']['at_fault']} (C {r['C']['at_fault']}) | {tc.get(t)} (C {tc.get(c)}) |")
    L += ["", "| tag | static: distance (m) | static: contact | static: min gap (m) | stop17 zeros | arrival 4 s arc @k=1 (m) | dev ADE |", "|---|--:|:-:|--:|--:|--:|--:|"]
    for k in [t] + list(CL_REF):
        s, p = res["cl"][k]["static"], res["cl"][k]["stop17"]
        ar, de = res["arrival"].get(k), res["dev_ade"].get(k)
        f2 = lambda x, q=".2f": "-" if x is None else format(x, q)  # noqa: E731
        L.append(f"| {k} | {f2(s and s['dist'])} | {'-' if s is None else ('yes' if s['collision'] else 'no')} | {f2(s and s['min_gap'])} | "
                 f"{p['zeros']} / {p['n']} | {f2(ar)} | {f2(de, '.4f')} |")
    L += ["", f"arrival-stop seen in closed loop (static no contact and <= 1.0 m, or stop17 zeros <= 15): {res['stop_seen']}",
          "dev ADE: " + ", ".join(f"{k} {v:.4f}" for k, v in res["dev_ade"].items())]
    write(a.out, res, L)
    (D / "FINISHED").write_text("ok\n")


def coll_counts(dirs):
    """Front / lateral / rear collision counts over the run dirs (metrics collision_<side> > 0, every rollout)."""
    n = dict(front=0, lateral=0, rear=0)
    for d in dirs:
        f = Path(d) / "aggregate/results-summary.json"
        if f.exists():
            for x in json.loads(f.read_text())["rollouts"]:
                m = x.get("metrics") or {}
                for q in n:
                    n[q] += int((m.get(f"collision_{q}") or 0) > 0)
    return n


def cmd_wide(a):
    """补记 1 addition 2: public 1491 @10 and PAI 281 (submission config) of one arm against the stored RSG-L18-s0 reads (--no-pai: public only)."""
    t, D = a.j, Path(a.dir)
    res = dict(arm=t)
    reads = (("pub", f"{D}/pub/{t}/b*", f"{DATA}/runs/route_follow/guard/read10/RSG-L18-s0/b*"),
             ("pai", f"{D}/pai/{t}/runs/*/sim", f"{wt.PAI_REF}/runs/*/sim"))[:1 if a.no_pai else 2]
    for k, mine, ref in reads:
        MJ, MC = wt.metrics(glob.glob(mine)), wt.metrics(glob.glob(ref))
        r = wt.xpair(MJ, MC)
        for side, dirs in (("J", glob.glob(mine)), ("C", glob.glob(ref))):
            r[side]["collisions"] = coll_counts(dirs)
        r.pop("zero_flips", None)
        res[k] = r
    if not a.no_pai:
        full = [wt.pai_full(p) for p in wt.STOP17]
        for side, g in (("J", f"{D}/pai/{t}/runs/*/sim"), ("C", f"{wt.PAI_REF}/runs/*/sim")):
            m = wt.metrics(glob.glob(g))
            res.setdefault("stop17", {})[side] = dict(zeros=sum(m[s][0] == 0 for s in full if s in m), n=sum(s in m for s in full))
    L = [f"# 补记 1 addition 2 (exploratory): {t} wide reads vs RSG-L18-s0 (stored)", "",
         "| read | scenes | zeros (C) | mean (C) | paired [session CI] | progress_rel (C) | at-fault (C) | offroad (C) | front / lateral / rear collisions (C) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in (x[0] for x in reads):
        r = res[k]
        cj, cc = r["J"]["collisions"], r["C"]["collisions"]
        L.append(f"| {k} | {r['scenes']} | {r['J']['zeros']} ({r['C']['zeros']}) | {r['J']['mean']:.4f} ({r['C']['mean']:.4f}) | {r['paired']} | "
                 f"{r['J']['progress_rel']:.3f} ({r['C']['progress_rel']:.3f}) | {r['J']['at_fault']} ({r['C']['at_fault']}) | {r['J']['offroad']} ({r['C']['offroad']}) | "
                 f"{cj['front']} / {cj['lateral']} / {cj['rear']} ({cc['front']} / {cc['lateral']} / {cc['rear']}) |")
    if not a.no_pai:
        L.append(f"\nPAI stop17 zeros: {t} {res['stop17']['J']['zeros']} / {res['stop17']['J']['n']}, RSG-L18-s0 {res['stop17']['C']['zeros']} / {res['stop17']['C']['n']}")
    write(a.out, res, L)


# ---------------------------------------------------------------- 补记 2 (CPU)
SHARED7 = ("53d3", "de4a", "6d2b", "9d1d", "9a67", "ddc1", "1f54")      # the new held zeros every RTB1 arm shares (decision 337)
A_ARM = ("a", "RTB1-F-s0", "RTB1O-L18-s0")


def cmd_vgate(a):
    """补记 2 stop rule for one base, after its training and arrival replay: GO iff the 4 s arc at k = 1 <= --max-arc and DONE dev ADE <= --max-dev."""
    arr, ade = arrival_of(a.j)[1]["arc4"], wt.dev_ade(a.j)
    a.max_dev = wt.dev_ade("RTB1-F-s0") if a.max_dev is None else a.max_dev           # "worse than RTB1-F-s0" = its exact DONE value (0.5556)
    go = bool(arr <= a.max_arc and ade <= a.max_dev)
    res = dict(base=a.j, arrival_k1=arr, dev_ade=ade, max_arc=a.max_arc, max_dev=a.max_dev, verdict="GO" if go else "STOP")
    write(f"rtb_v_gate_{a.j}", res, [f"# 补记 2 stop rule: {a.j}", "", f"arrival 4 s arc @k=1 {arr:.2f} m (STOP if > {a.max_arc}); "
                                      f"dev ADE {ade:.4f} (STOP if > {a.max_dev:.4f}): **{res['verdict']}**"])
    sys.exit(0 if go else 3)


def cmd_vsum(a):
    """补记 2: each variant's continuation (and a) vs a = RTB1O-L18-s0 and RSG-L18-s0, closed loop + the bases' open-loop reads."""
    arms = [A_ARM] + [(v, *bc.split(":")) for v, bc in (x.split("=") for x in a.arms.split(","))]
    held = {t: wt.metrics(glob.glob(wt.xdirs("held", t))) for t in [x[2] for x in arms] + ["RSG-L18-s0"]}
    pubd = lambda t: (glob.glob(f"{DATA}/runs/route_follow/guard/read10/{t}/b*") if t == "RSG-L18-s0"  # noqa: E731
                      else glob.glob(str(NX / f"cont/wide/pub/{t}/b*")))
    pub = {t: wt.metrics(pubd(t)) for t in held}
    shared = [s for s in held["RSG-L18-s0"] if s.rsplit("-", 1)[-1][:4] in SHARED7]
    assert len(shared) == 7, shared
    res = dict(arms={})
    for v, b, t in arms:
        D = NX / "cont" / v
        r = dict(base=b, cont=t, held={c: wt.xpair(held[t], held[c]) for c in ("RTB1O-L18-s0", "RSG-L18-s0") if c != t},
                 shared7_zero=[s for s in shared if held[t].get(s, (1,))[0] == 0], static=static_of(D / f"static/{t}/run"), stop17=stop17_of(D / f"paisub/{t}"),
                 arrival_cont=arrival_of(t)[1]["arc4"], dev_ade_cont=wt.dev_ade(t),
                 pub={c: wt.xpair(pub[t], pub[c]) for c in ("RTB1O-L18-s0", "RSG-L18-s0") if c != t and pub[t]},
                 pub_coll=coll_counts(pubd(t)) if pub[t] else None, base_reads=dict(arrival=arrival_of(b)[1]["arc4"], dev_ade=wt.dev_ade(b)))
        try:
            cap, led = wt.cap_read(b)[0], wt.ledger_read(b)
            r["base_reads"] |= dict(wait_W=cap["wait_W"], go_G=cap["go_G"], D=cap["D"], **{k: float(led[k][0].mean()) for k in ("S_L", "Y_P", "Y_V")})
        except FileNotFoundError as e:
            r["base_reads"]["missing"] = str(e)
        for k in ("held", "pub"):
            for x in r[k].values():
                x.pop("zero_flips", None)
        res["arms"][v] = r
    res["pub_coll_RSG"] = coll_counts(pubd("RSG-L18-s0"))
    f2 = lambda x, q=".2f": "-" if x is None else format(x, q)  # noqa: E731
    L = ["# 补记 2 (exploratory): RTB1 variants vs a = RTB1O-L18-s0 and RSG-L18-s0, one seed", "",
         "| arm | base / cont | held zeros (a / RSG) | at-fault (a / RSG) | offroad | mean vs a [CI] | mean vs RSG [CI] | shared-7 still zero | static dist / contact | "
         "stop17 zeros | arrival @k=1 cont (base) | dev ADE base / cont |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for v, r in res["arms"].items():
        hr, ha = r["held"]["RSG-L18-s0"], r["held"].get("RTB1O-L18-s0")
        s = r["static"]
        L.append(f"| {v} | {r['base']} / {r['cont']} | {hr['J']['zeros']} ({ha['C']['zeros'] if ha else '-'} / {hr['C']['zeros']}) | "
                 f"{hr['J']['at_fault']} ({ha['C']['at_fault'] if ha else '-'} / {hr['C']['at_fault']}) | {hr['J']['offroad']} ({hr['C']['offroad']}) | "
                 f"{ha['paired'] if ha else '-'} | {hr['paired']} | {len(r['shared7_zero'])} / 7 | "
                 f"{'-' if s is None else f2(s['dist']) + (' yes' if s['collision'] else ' no')} | {r['stop17']['zeros']} / {r['stop17']['n']} | "
                 f"{f2(r['arrival_cont'])} ({f2(r['base_reads']['arrival'])}) | {r['base_reads']['dev_ade']:.4f} / {r['dev_ade_cont']:.4f} |")
    L += ["", "| arm | public scenes | zeros (a / RSG) | mean | vs a [CI] | vs RSG [CI] | at-fault (RSG) | offroad (RSG) | front / lateral / rear |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for v, r in res["arms"].items():
        pr, pa = r["pub"].get("RSG-L18-s0"), r["pub"].get("RTB1O-L18-s0")
        if pr is None:
            L.append(f"| {v} | - | | | | | | | |")
            continue
        c = r["pub_coll"]
        L.append(f"| {v} | {pr['scenes']} | {pr['J']['zeros']} ({pa['C']['zeros'] if pa else '-'} / {pr['C']['zeros']}) | {pr['J']['mean']:.4f} | "
                 f"{pa['paired'] if pa else '-'} | {pr['paired']} | {pr['J']['at_fault']} ({pr['C']['at_fault']}) | {pr['J']['offroad']} ({pr['C']['offroad']}) | "
                 f"{c['front']} / {c['lateral']} / {c['rear']} |")
    c = res["pub_coll_RSG"]
    L += [f"| RSG-L18-s0 | | | | | | | | {c['front']} / {c['lateral']} / {c['rear']} |", "",
          "| base | arrival @k=1 | dev ADE | wait W | go G | D | S_L F | Y_P F | Y_V F |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for v, r in res["arms"].items():
        q = r["base_reads"]
        L.append(f"| {r['base']} | {q['arrival']:.2f} | {q['dev_ade']:.4f} | " + " | ".join(f2(q.get(k), ".3f") for k in ("wait_W", "go_G", "D", "S_L", "Y_P", "Y_V")) + " |")
    L += ["", "shared-7 still zero: " + "; ".join(f"{v} " + " ".join(s.rsplit('-', 1)[-1][:4] for s in r["shared7_zero"]) for v, r in res["arms"].items())]
    write(a.out, res, L)


# ---------------------------------------------------------------- 补记 3 (CPU)
C0, C1 = "RSG-L18-s0", "RSG-L18-s1"
CMP_PAI = DATA / "runs/cl_infra/compile/pai"           # the cl-compile lane's PAI 281 reads: same chunk lists and submission config, today's code
R2_ZEROS, R2_ATFAULT = 5, 4                            # prereg 补记 3, fixed before the read: R2 = zeros >= C + 5, or at-fault >= C + 4, or paired CI upper bound < 0


def pair_ci(MJ, MC):
    """wt.xpair's pairing (common scenes, session clusters) of the mean score as numbers: n, mean, lo, hi."""
    from jevdrive import stats
    s = sorted(set(MJ) & set(MC))
    p = stats.paired([MJ[x][0] for x in s], [MC[x][0] for x in s], groups=np.array(["_".join(x.split("_")[:2]) for x in s]))
    return dict(n=len(s), mean=float(p["mean"]), lo=float(p["lo"]), hi=float(p["hi"]))


def read_vs(dirs, ref):
    """One closed-loop read against a reference read (lists of run dirs): wt.xpair + front / lateral / rear collisions of both."""
    r = wt.xpair(wt.metrics(dirs), wt.metrics(ref))
    r.pop("zero_flips")
    r["J"]["collisions"], r["C"]["collisions"] = coll_counts(dirs), coll_counts(ref)
    return r


def pai_complete(root):
    """True once every chunk of a pai_box.sh read root has its DONE marker."""
    root = Path(root)
    n = len(list((root / "chunks").glob("*.txt")))
    return n > 0 and sum((d / "DONE").exists() for d in (root / "runs").glob("*_c*")) == n


def cmd_v2pai(a):
    """Prereg 补记 3 (the user cancelled the second base seed; this read is what is left): PAI 281 in the submission config of the --j arms (the decision 339 condition R2
    is read on them) and the --ctx arms (context), against the stored RSG-L18-s0 read, from NX/rtb_wide_<tag>.json (nx.py wide); the cl-compile lane's complete PAI reads
    of RSG-L18-s0 (the same checkpoint read again) and RSG-L18-s1 are the references -> results/next/<out>.{json,md}"""
    pai0 = glob.glob(f"{wt.PAI_REF}/runs/*/sim")
    res = dict(arms={}, refs={}, notes=[])
    for t in a.j + a.ctx:
        w = json.loads((NX / f"rtb_wide_{t}.json").read_text())
        r = res["arms"][t] = dict(w["pai"], ci=pair_ci(wt.metrics(glob.glob(f"{NX}/cont/wide/pai/{t}/runs/*/sim")), wt.metrics(pai0)), stop17=w["stop17"]["J"], r2=t in a.j)
        res["refs"].setdefault(f"{C0} (stored)", dict(J=w["pai"]["C"], scenes=w["pai"]["scenes"], stop17=w["stop17"]["C"]))
        if r["r2"]:
            d = dict(d_zeros=r["J"]["zeros"] - r["C"]["zeros"], d_at_fault=r["J"]["at_fault"] - r["C"]["at_fault"], ci_hi=r["ci"]["hi"])
            r["R2"] = d | dict(flag_zeros=bool(d["d_zeros"] >= R2_ZEROS), flag_at_fault=bool(d["d_at_fault"] >= R2_ATFAULT), flag_ci=bool(d["ci_hi"] < 0))
            r["R2"]["met"] = bool(r["R2"]["flag_zeros"] or r["R2"]["flag_at_fault"] or r["R2"]["flag_ci"])
    for lab, root in ((f"{C0} re-read", CMP_PAI / C0), (C1, CMP_PAI / C1)):
        if pai_complete(root):
            q = read_vs(glob.glob(f"{root}/runs/*/sim"), pai0)
            res["refs"][lab] = dict(J=q["J"], scenes=q["scenes"], paired=q["paired"], stop17=stop17_of(root))
        else:
            res["notes"].append(f"the cl-compile lane's PAI 281 read of {lab} ({root}) was not complete at write time and is left out")
    col = lambda c: f"{c['front']} / {c['lateral']} / {c['rear']}"  # noqa: E731
    s17 = lambda p: f"{p['zeros']} / {p['n']}"  # noqa: E731
    L = ["# V2 (RTB1AR) PAI 281 in the submission config (prereg 2026-10-17 amendment 3, exploratory, one read)", "",
         "Decision 339 left V2's PAI 281 unread. Amendment 3 asked for it, and for a second base seed, which the user cancelled before any read (seeds behave alike); "
         "this file holds the one read that was kept: RTB1ARO-L18-s0 (base RTB1AR-F-s0, RSG-L18 continuation, 2 500 steps, seed 0) on the 281 PAI scenes in the submission config "
         "(`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`, `controller.gains.rel_front_steering_angle_weight=10`, ALPASIM_FAST=1), paired against the stored RSG-L18-s0 read "
         "`route_follow/pai_g10/acc` (session-clustered CI). The public and held reads of the same arm are in [rtb_v.md](rtb_v.md). "
         "Prereg: [plans/2026-10-17-retain-anticopy-draft.md](../../plans/2026-10-17-retain-anticopy-draft.md) amendment 3. Code: `scripts/nx.sh rtb-wide`, `scripts/nx.py v2pai`.", "",
         "| arm | scenes | zeros | mean | paired vs RSG-L18-s0 [CI] | at-fault | offroad | front / side / rear collisions | stop17 zeros |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in res["arms"].items():
        L.append(f"| {t} | {r['scenes']} | {r['J']['zeros']} | {r['J']['mean']:.4f} | {r['paired']} | {r['J']['at_fault']} | {r['J']['offroad']} | {col(r['J']['collisions'])} | {s17(r['stop17'])} |")
    for lab, q in res["refs"].items():
        L.append(f"| {lab}{'' if lab.endswith('(stored)') else ' (cl-compile lane)'} | {q['scenes']} | {q['J']['zeros']} | {q['J']['mean']:.4f} | {q.get('paired', '-')} | {q['J']['at_fault']} | "
                 f"{q['J']['offroad']} | {col(q['J']['collisions'])} | {s17(q['stop17'])} |")
    L += [""] + [f"Note: {n}." for n in res["notes"]]
    L += ["", "## Check against the decision 339 refutation conditions", ""]
    t = ["R1 (the second base seed has public rear-ended >= 22) cannot be read: the second base seed was cancelled by the user before any read."]
    for k, r in res["arms"].items():
        if r["r2"]:
            f, J, C = r["R2"], r["J"], r["C"]
            hit = [n for n, on in (("zeros", f["flag_zeros"]), ("at-fault", f["flag_at_fault"]), ("the paired CI upper bound below 0", f["flag_ci"])) if on]
            t.append(f"R2 (PAI 281 at-fault / zeros clearly above RSG-L18-s0; flags fixed in amendment 3 before the read: zeros >= C + {R2_ZEROS}, at-fault >= C + {R2_ATFAULT}, "
                     f"paired CI upper bound < 0): {k} has {J['zeros']} zeros ({f['d_zeros']:+d} against RSG-L18-s0's {C['zeros']}), {J['at_fault']} at-fault ({f['d_at_fault']:+d} against "
                     f"{C['at_fault']}) and a paired mean of {r['paired']}, so " + (f"it crosses the flag on {', '.join(hit)} and R2 is MET" if hit else f"none of the three flags is crossed (zeros {f['d_zeros']:+d} against the +{R2_ZEROS} that flags, "
                                                                                        f"at-fault {f['d_at_fault']:+d} against +{R2_ATFAULT}) and R2 is not met")
                     + f"; stop17 is {s17(r['stop17'])} against {s17(res['refs'][f'{C0} (stored)']['stop17'])}.")
    rr, r1 = res["refs"].get(f"{C0} re-read"), res["refs"].get(C1)
    if rr and r1:
        same = all(rr["J"][x] == res["refs"][f"{C0} (stored)"]["J"][x] for x in ("zeros", "at_fault", "offroad"))
        t.append(f"For scale: the cl-compile lane's read of RSG-L18-s0 in the same config {'reproduces' if same else 'differs from'} the stored read ({rr['J']['zeros']} zeros, {rr['J']['at_fault']} at-fault, "
                 f"paired {rr['paired']}), so a PAI 281 read of one checkpoint repeated exactly in this comparison; RSG-L18-s1 reads {r1['J']['zeros']} zeros, {r1['J']['at_fault']} at-fault, {r1['paired']}, which is the spread between the two RSG seeds.")
    L += [" ".join(t)]
    write(a.out, res, L, a.dry)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bitcmp")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--out", required=True)
    p.add_argument("--hash", default=""), p.add_argument("--note", default="")
    sp.add_parser("hist").add_argument("--tags", nargs="+", required=True)
    kv = lambda s: dict(x.split("=", 1) for x in s.split(","))  # noqa: E731
    sp.add_parser("approach")
    p = sp.add_parser("arrival")
    p.add_argument("--tags", nargs="+", required=True), p.add_argument("--rmem-from", default="RSG-L18-s0")
    p = sp.add_parser("rtgate")
    p.add_argument("--variant", choices=("a", "b"), required=True)
    p.add_argument("--j", nargs="+", required=True), p.add_argument("--c", required=True), p.add_argument("--lam-of", type=kv, required=True)
    p.add_argument("--dry", action="store_true", help="print only (a check of the read code on existing tags)")
    p = sp.add_parser("acgate")
    p.add_argument("--j", nargs="+", required=True), p.add_argument("--c", required=True), p.add_argument("--dose-of", type=kv, required=True)
    p.add_argument("--submit", action="store_true"), p.add_argument("--dry", action="store_true", help="print only")
    p = sp.add_parser("acheld")
    p.add_argument("--j", required=True), p.add_argument("--c", required=True)
    p = sp.add_parser("devdecomp")
    p.add_argument("--j", nargs="+", required=True), p.add_argument("--c", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("wide")
    p.add_argument("--j", required=True), p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    p.add_argument("--no-pai", action="store_true", help="public 1491 only")
    p = sp.add_parser("vgate")
    p.add_argument("--j", required=True), p.add_argument("--max-arc", type=float, default=2.0), p.add_argument("--max-dev", type=float, default=None)
    p = sp.add_parser("vsum")
    p.add_argument("--arms", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("cont")
    p.add_argument("--j", required=True), p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("v2pai")
    p.add_argument("--j", nargs="+", required=True, help="arms the condition R2 is read on"), p.add_argument("--ctx", nargs="*", default=[], help="context arms")
    p.add_argument("--out", required=True), p.add_argument("--dry", action="store_true", help="print only")
    a = ap.parse_args()
    dict(vgate=cmd_vgate, vsum=cmd_vsum, devdecomp=cmd_devdecomp, cont=cmd_cont, wide=cmd_wide, bitcmp=cmd_bitcmp, hist=cmd_hist, approach=cmd_approach, arrival=cmd_arrival, rtgate=cmd_rtgate, acgate=cmd_acgate, acheld=cmd_acheld, v2pai=cmd_v2pai)[a.cmd](a)
