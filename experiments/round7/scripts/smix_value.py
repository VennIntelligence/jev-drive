#!/usr/bin/env python3
"""SMIX-VALUE (results/smix_value/README.md): what is S-MIX worth, CPU only, before any GPU is spent on it. No training, no closed loop.

  fwd --ckpt TAG [--cmd heading] [--frac F]   (box, CPU, repo root, op-train venv) the trainee view of experiments/practice_nuplan/scripts/
        lat_diag.py `trainee` (same argv = NTVX's command, same term weights ho_w 3 x lam 10 for every checkpoint, eval-mode plans, the pool
        terms exactly as the loss sees them), on CPU, on the S rows of the R6DAG / R7J pool parts (rb0 / rtb1 / y / ys) only, optionally on a
        seeded row sample (--frac) and with the round7 command rule applied to the fed rows (--cmd heading, what R7J trained and serves with)
        -> $W/fwd_<TAG>[_cmdh].parquet
  batch    (box, CPU) Q1: per-batch arithmetic of R6DAG.npz vs R7J.npz on the audit definitions (research/s-rows/audit.py)  -> $W/batch.json
  learn    (box, CPU) Q2: activation / hinge mass of each forward on the learnable-active target rows of S-MIX, by term, with row-bootstrap CIs
        -> $W/learn.json
  cover    (box, CPU) Q3: speed and geometry of the PAI 281 loss classes (R6DAG s0 / s1 reads: ego speed of the last inference before the
        event, from the driver records) against the S pool's draw-weighted speed / situation distribution -> $W/cover.json
  pai      (box, CPU) Q4: per-class paired differences R7J vs R6DAG (same seed) on the loss-map classes, scene bootstrap -> $W/pai.json
Outputs are small JSON; the Mac copies them to experiments/round7/results/smix_value/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.common import n_cpus  # noqa: E402

NC = n_cpus()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(NC))
import numpy as np  # noqa: E402

sys.path[:0] = [str(REPO / "experiments/practice_nuplan/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/round7/smix_value"
LD = DATA / "runs/practice_nuplan/lat_diag"
S_PARTS = ("rb0", "rtb1", "y", "ys")
TERMS = ["agent", "road", "lat_side", "lat_onc", "lat_soft"]
K_PRE = {"R6DAG": 72, "R7J": 98}


def _boot(f, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([f(rng.integers(0, n, n)) for _ in range(B)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ---------------------------------------------------------------- forward (lat_diag.py trainee on CPU)
def cmd_fwd(a):
    import pandas as pd
    import torch
    import lat_diag as LDm
    torch.set_num_threads(a.threads or NC)
    line = next(x for x in (DATA / "runs/practice_nuplan/ntvx/stages.txt").read_text().splitlines() if " train NTVX-L18-s0: " in x)
    argv = line.split("rm_train.py train ", 1)[1].split()
    argv[argv.index("--tag") + 1] = a.ckpt
    if a.cmd:
        argv += ["--cmd-rule", a.cmd]
    rt = REPO / "experiments/route_mem/scripts/rm_train.py"
    src = rt.read_text()
    tail = "dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)"
    assert src.count(tail) == 1
    g = {"__name__": "__main__", "__file__": str(rt)}
    sys.argv = [str(rt), "rely", *argv]
    exec(compile(src.replace(tail, "_LATDIAG_ARGS = a"), str(rt), "exec"), g)
    ra = g["_LATDIAG_ARGS"]
    ra.rm_from = a.ckpt
    dev = torch.device("cpu")
    ns, *_ = g["build"](ra, dev)
    LS = ns.losses(ns.mode)
    import pp_train as T
    from loss43 import shape_only
    S, model = ns.S, ns.model.eval()
    pt = np.load(LDm.POOL / "tab.npz")
    assert len(pt["names"]) == ns.n_on and np.array_equal(S.tab["names"][ns.on0:], pt["names"]), "pool rows are not on1_JTX's"
    part = pt["pool"].astype(str)
    sel = np.flatnonzero((pt["mode"].astype(str) == "self") & np.isin(part, S_PARTS))
    rand = np.ones(len(sel), bool)
    if a.frac < 1:                         # stratified: every S-MIX target row (x4) + a seeded uniform frac of all rows (flag `rand`)
        rng = np.random.default_rng(a.seed)
        u = rng.random(len(sel)) < a.frac
        tgt = np.isin(pt["names"][sel].astype(str), rows_table().query("al").name.to_numpy()) if a.target_all else np.zeros(len(sel), bool)
        keep = u | tgt
        sel, rand = sel[keep], u[keep]
    pi_ = torch.as_tensor(S.pi, device=dev)
    wA, wR, wL = ra.ho_w * ra.agent_lam, ra.ho_w * ra.road_lam, ra.ho_w * ra.on_lon_lam
    cols = {}
    import time
    t0 = time.time()
    for j in range(0, len(sel), a.bs):
        r = torch.as_tensor(ns.on0 + sel[j:j + a.bs], device=dev)
        with torch.no_grad():
            o = model(S.front[r.cpu().numpy()], S.ego[r], S.tc[r], None, None, nv=None if ns.nv is None else ns.nv[r]).float()
            x0, y0, p0 = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[r], LS.W)
        x, y, psi = x0.detach(), y0.detach(), p0.detach()
        xs, ys = shape_only(x, y, psi) if LS.shape else (x, y)
        terms = {}
        with torch.no_grad():
            terms["agent"] = wA * LS.agent2.per_step(xs, ys, psi, r).mean(1) * LS.on_aok[r]
            terms["road"] = wR * torch.relu(LS.road.margin - LS.road.margins(xs, ys, psi, r)).mean(1) * LS.road.ok[r]
            if LS.lon is not None:
                for q, v in LS.lon.loss(torch.stack([x, y, psi], -1), r - LS.on0, LS.pack).items():
                    terms[q] = wL * v
        rec = dict(arc4=torch.hypot(x0[:, -1], y0[:, -1]), y4=y0[:, -1], y2=y0[:, 3]) | terms
        for q, v in rec.items():
            cols.setdefault(q, []).append(v.detach().cpu().numpy())
        if j == 0:
            print(f"first batch {time.time() - t0:.1f} s, rows {len(sel)}", flush=True)
    df = pd.DataFrame({q: np.concatenate(v) for q, v in cols.items()})
    df.insert(0, "name", pt["names"][sel]), df.insert(1, "pool", part[sel]), df.insert(2, "speed", pt["speed"][sel])
    df["rand"] = rand
    W.mkdir(parents=True, exist_ok=True)
    out = W / f"fwd_{a.ckpt}{'_cmdh' if a.cmd else ''}.parquet"
    df.to_parquet(out)
    print(json.dumps(dict(ckpt=a.ckpt, cmd=a.cmd, rows=len(df), sec=round(time.time() - t0, 1), weights=dict(agent=wA, road=wR, lat=wL),
                          cmd_rec=getattr(ns, "cmd_rec", None) and {k: v for k, v in ns.cmd_rec.items() if k != "parts"}, out=str(out))))


# ---------------------------------------------------------------- shared row table
def rows_table():
    """S rows of the merged pool with audit columns, R6DAG-s1's own terms (E0), the S-MIX multipliers and both draw files' weights."""
    import pandas as pd
    st = pd.read_parquet(DATA / "runs/s_rows_audit/rows_audit.parquet")[["name", "pool", "off_t0", "fut_agent", "fut_road", "fut_ok", "outcome",
                                                                          "fut_dpsi4", "obj_d", "obj_corridor", "e0", "k"]]
    t = pd.read_parquet(LD / "trainee_R6DAG-L18-s1.parquet")
    t = t[t.pool.isin(S_PARTS)][["name", "speed", *TERMS]].merge(st.drop(columns="pool"), on="name", how="left", validate="1:1")
    act = (t[TERMS] > 0).any(axis=1)
    ag = (t.agent > 0) | (t.road > 0)
    u = (ag & ((t.fut_agent > 0) | (t.fut_road > 0)) & t.fut_ok.astype(bool)) | t.off_t0.astype(bool)
    t["unlearn"], t["act"], t["al"] = u.to_numpy(), act.to_numpy(), (act & ~u).to_numpy()
    t["mass"] = t[TERMS].sum(axis=1)
    for k, f in (("R6DAG", DATA / "runs/round6/dag/w/R6DAG.npz"), ("R7J", DATA / "runs/round7/joint/w/R7J.npz")):
        z = np.load(f)
        ix = {x: i for i, x in enumerate(z["names"].astype(str))}
        t[f"w_{k}"] = z["w"][np.array([ix[x] for x in t.name])]
    t["sit"] = np.select([t.speed < 0.5, (t.obj_d < 20) & (t.obj_corridor > 0), t.fut_dpsi4.abs() > np.deg2rad(30)],
                         ["standing", "lead<20m", "turn>30deg"], "free/other")
    return t


# ---------------------------------------------------------------- Q1
def cmd_batch(a):
    t = rows_table()
    nav = dict(R6DAG=51, R7J=25)
    how = dict(R6DAG=3.0, R7J=0.86)
    imit = {k: 0.75 * nav[k] + 13 + 32 + 8 for k in nav}                   # audit / joint-plan definition: navtrain anchor rows 25 % not imitating
    out = {}
    for k in ("R6DAG", "R7J"):
        w = t[f"w_{k}"].to_numpy()
        Sb = K_PRE[k] * w.sum()                                              # S rows per batch
        p = w / w.sum()
        al, m = t.al.to_numpy(), t.mass.to_numpy()
        mass_d = (p * m).sum()
        o = dict(S_per_batch=Sb, ho_w=how[k], imit_rows=imit[k], live_rows=int((w > 0).sum()),
                 active_share=float(p[t.act].sum()), active_per_batch=float(Sb * p[t.act].sum()),
                 al_share=float(p[al].sum()), al_per_batch=float(Sb * p[al].sum()),
                 unlearn_share=float(p[t.unlearn].sum()),
                 mass_per_draw=float(mass_d), mass_share_al=float((p * m * al).sum() / mass_d),
                 mass_share_unlearn=float((p * m * t.unlearn).sum() / mass_d),
                 row_pressure=Sb * how[k] / imit[k],
                 # effective pressure: expected S hinge (R6DAG-s1 terms, trainer weights ho_w 3 x lam baked in -> rescale to the arm's ho_w) per imitation row
                 hinge_per_imit=float(Sb * mass_d * how[k] / 3.0 / imit[k]),
                 hinge_al_per_imit=float(Sb * (p * m * al).sum() * how[k] / 3.0 / imit[k]),
                 speed_gt13_share=float(p[t.speed > 13].sum()), speed_gt8_share=float(p[t.speed > 8].sum()))
        for c in TERMS:
            o[f"mass_share_{c}"] = float((p * t[c].to_numpy()).sum() / mass_d)
            o[f"al_mass_{c}_per_imit"] = float(Sb * (p * t[c].to_numpy() * al).sum() * how[k] / 3.0 / imit[k])
        out[k] = o
    out["ratio_R7J_over_R6DAG"] = {q: out["R7J"][q] / out["R6DAG"][q] for q in ("al_per_batch", "active_per_batch", "hinge_per_imit", "hinge_al_per_imit",
                                                                             "row_pressure", "S_per_batch")}
    out["RSG_pressure_unit"] = "row_pressure / (13 x 3 / (0.75 x 102 + 13)) = row_pressure / 0.436 -> RSG multiples"
    for k in ("R6DAG", "R7J"):
        out[k]["row_pressure_rsg_x"] = out[k]["row_pressure"] / (13 * 3 / (0.75 * 102 + 13))
    W.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(W / "batch.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- Q2
def cmd_learn(a):
    import pandas as pd
    t = rows_table().set_index("name")
    fw = {}
    for f in sorted(glob.glob(str(W / "fwd_*.parquet"))):
        fw[Path(f).stem[4:]] = pd.read_parquet(f).set_index("name")
    for tag, f in (("R6DAG-L18-s1_gpu", LD / "trainee_R6DAG-L18-s1.parquet"), ("NTV0-L18-s0_gpu", LD / "trainee_NTV0-L18-s0.parquet")):
        fw[tag] = pd.read_parquet(f).set_index("name")
    common = sorted(set.intersection(*[set(v.index) for v in fw.values()]) & set(t.index))
    t = t.loc[common]
    rnd = pd.concat([v.loc[common, "rand"] for k, v in fw.items() if "rand" in v], axis=1).all(axis=1) if any("rand" in v for v in fw.values()) \
        else pd.Series(True, index=common)
    # strata: the S-MIX target = learnable-active under R6DAG-s1 (multiplier 4); x2 = learnable, inactive, zero / partial source; x1 rest; x0 unlearnable
    strata = dict(target_x4=t.al.to_numpy(), x2=(~t.unlearn & ~t.act & t.outcome.isin(["zero", "partial"])).to_numpy(),
                  x1=(~t.unlearn & ~t.act & ~t.outcome.isin(["zero", "partial"])).to_numpy(), x0=t.unlearn.to_numpy(),
                  all_live=((t.w_R6DAG > 0) & rnd).to_numpy())
    for k in ("x2", "x1", "x0"):                                             # non-target strata: the uniform sample only (targets are all in)
        strata[k] = strata[k] & rnd.to_numpy()
    # R7J's own draw weights (renormalised) for a draw-weighted view of the S slice
    pr = (t.w_R7J * rnd).to_numpy() / (t.w_R7J * rnd).sum()
    out = dict(rows=len(common), strata={k: int(v.sum()) for k, v in strata.items()}, fwd={})
    for tag, f in fw.items():
        f = f.loc[common]
        rec = {}
        for sn, m in strata.items():
            r = {}
            for c in TERMS + ["any"]:
                v = (f[TERMS].sum(axis=1) if c == "any" else f[c]).to_numpy()[m]
                r[c] = dict(act=float((v > 0).mean()), mass=float(v.mean()))
            rec[sn] = r
        mm = f[TERMS].sum(axis=1).to_numpy()
        rec["r7j_draw_weighted_mass"] = float((pr * mm).sum())
        rec["r7j_draw_weighted_act"] = float((pr * (mm > 0)).sum())
        out["fwd"][tag] = rec
    # paired contrasts on the target rows: R7J vs R6DAG same seed (fwd on CPU, cmd as each trained), with row bootstrap CIs
    pairs = [(f"R7J-L18-s{s}_cmdh", f"R6DAG-L18-s{s}") for s in (0, 1)] + [(f"R7J-L18-s{s}", f"R6DAG-L18-s{s}") for s in (0, 1)] + \
            [("R6DAG-L18-s1", "NTV0-L18-s0"), ("R6DAG-L18-s0", "NTV0-L18-s0"), ("R6DAG-L18-s1", "R6DAG-L18-s0"), ("R7J-L18-s1_cmdh", "R7J-L18-s0_cmdh"),
             ("R6DAG-L18-s1", "R6DAG-L18-s1_gpu")]
    out["pairs"] = {}
    for A, B in pairs:
        if A not in fw or B not in fw:
            continue
        fa, fb = fw[A].loc[common], fw[B].loc[common]
        res = {}
        for sn in ("target_x4", "x2", "x1", "x0", "all_live"):
            m = strata[sn]
            for c in TERMS + ["any"]:
                va = (fa[TERMS].sum(axis=1) if c == "any" else fa[c]).to_numpy()[m]
                vb = (fb[TERMS].sum(axis=1) if c == "any" else fb[c]).to_numpy()[m]
                if c == "any" or sn == "target_x4":
                    da, db = (va > 0).astype(float), (vb > 0).astype(float)
                    n = len(va)
                    B_ = 400 if n > 20000 else 1000
                    res[f"{sn}:{c}:act"] = dict(a=float(da.mean()), b=float(db.mean()), d=float(da.mean() - db.mean()),
                                               ci=_boot(lambda i: (da[i] - db[i]).mean(), n, B_))
                    res[f"{sn}:{c}:mass_rel"] = dict(a=float(va.mean()), b=float(vb.mean()), rel=float(va.mean() / max(vb.mean(), 1e-12) - 1),
                                                    ci=_boot(lambda i: va[i].mean() / max(vb[i].mean(), 1e-12) - 1, n, B_))
        out["pairs"][f"{A} - {B}"] = res
    json.dump(out, open(W / "learn.json", "w"), indent=1)
    print(json.dumps(out["strata"]))
    for k, v in out["pairs"].items():
        print("\n##", k)
        for q, r in v.items():
            if q.startswith(("target_x4", "all_live:any", "x2:any", "x1:any", "x0:any")):
                if "act" in q:
                    print(f"{q:28s} {r['a']:.4f} {r['b']:.4f} d {r['d']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
                else:
                    print(f"{q:28s} {r['a']:.4f} {r['b']:.4f} rel {r['rel']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")


# ---------------------------------------------------------------- Q3 / Q4: PAI loss classes
READS = {"R6DAG-s0": DATA / "runs/round6/dag/pai281/R6DAG-L18-s0-ag", "R6DAG-s1": DATA / "runs/round6/dag/pai281/R6DAG-L18-s1-ag",
         "R7J-s0": DATA / "runs/round7/joint/pai281/R7J-L18-s0-ag", "R7J-s1": DATA / "runs/round7/joint/pai281/R7J-L18-s1-ag"}
FACTS = {"R6DAG-s0": "R6DAG-L18-s0-281ag", "R6DAG-s1": "R6DAG-L18-s1-281ag", "R7J-s0": "R7J-L18-s0-281ag", "R7J-s1": "R7J-L18-s1-281ag",
         "NTV0": "NTV0-L18-s0-281ag", "R6DAG-s2": "R6DAG-L18-s2-281ag"}
OK = 0.98


def lclass(r):
    """loss_map.py's classes (refined pai_fix fclass), coarse names."""
    if r.score >= OK:
        return "ok"
    fc, pr, stra = str(r.fclass), str(r.primary), str(r.stratum)
    if fc == "offroad":
        if pr in ("off_junc_inside", "off_curve_inside"):
            return "off_turn_inside"
        return "off_straight_junc_outside"
    if fc in ("drift", "lanechange") or "drift" in fc or "lane" in fc:
        return "drift_lc"
    if fc == "overshoot":
        return "overshoot"
    if fc in ("AF-front", "side") or "front" in fc or "side" in fc or "collision" in fc:
        return "collision"
    if fc == "route-turn":
        return "route_turn"
    if fc == "rear-hit" or "rear" in fc:
        return "rear_hit"
    if fc == "slow" or "slow" in fc or "launch" in fc or "creep" in fc:
        return "slow"
    return "other:" + fc


def facts(tag):
    import pandas as pd
    f = pd.read_csv(DATA / f"runs/pai_fix/facts/{FACTS[tag]}.csv")
    f["cls"] = f.apply(lclass, axis=1)
    return f.set_index("scene")


def ev_speed(run_dir):
    """scene -> list of (t_rel_s, v0) of the served inferences."""
    from collections import defaultdict
    d = defaultdict(list)
    for f in glob.glob(f"{run_dir}/runs/*/sim/driver-logs/drive*.jsonl"):
        for line in open(f):
            if '"infer": true' not in line:
                continue
            r = json.loads(line)
            v = (r.get("fix") or {}).get("v0")
            if v is None:
                v = float(np.hypot(*r["ego"][4:6])) if "ego" in r else np.nan
            d[r["scene"]].append((r["now"], v))
    out = {}
    for s, x in d.items():
        x = np.array(sorted(x))
        out[s] = np.c_[(x[:, 0] - x[0, 0]) / 1e6, x[:, 1]]
    return out


def cmd_cover(a):
    import pandas as pd
    t = rows_table()
    p6 = t.w_R6DAG.to_numpy() / t.w_R6DAG.sum()
    p7 = t.w_R7J.to_numpy() / t.w_R7J.sum()
    ev = []
    for tag in ("R6DAG-s0", "R6DAG-s1"):
        f = facts(tag)
        sp = ev_speed(READS[tag])
        for s, r in f[f.cls != "ok"].iterrows():
            x = sp.get(s)
            te = r.t_ev_s if pd.notna(r.t_ev_s) else np.nan
            v_ev = v_pre = np.nan
            if x is not None and len(x):
                if np.isfinite(te):
                    pre = x[(x[:, 0] <= te) & (x[:, 0] >= te - 2.5)]
                    v_ev = float(x[x[:, 0] <= te][-1, 1]) if (x[:, 0] <= te).any() else float(x[0, 1])
                    v_pre = float(pre[:, 1].max()) if len(pre) else v_ev
                else:
                    v_ev = v_pre = float(np.nanmax(x[:, 1]))
            ev.append(dict(read=tag, scene=s, cls=r.cls, loss=1 - r.score, v_ev=v_ev, v_pre_max=v_pre, v_hand=r.v_hand, turn_deg=r.turn_deg,
                           primary=r.primary, t_ev=te))
    E = pd.DataFrame(ev)
    E.to_csv(W / "cover_events.csv", index=False)
    spd = t.speed.to_numpy()
    turn = (t.fut_dpsi4.abs() > np.deg2rad(30)).to_numpy()
    straight = (t.fut_dpsi4.abs() < np.deg2rad(5)).to_numpy()
    al = t.al.to_numpy()
    out = {}
    for c, g in E.groupby("cls"):
        v = g.v_pre_max.dropna().to_numpy()
        if not len(v):
            continue
        lo, hi = max(np.percentile(v, 10) - 2, 0), np.percentile(v, 90) + 2
        band = (spd >= lo) & (spd <= hi)
        geo = turn if c == "off_turn_inside" else straight if c in ("off_straight_junc_outside", "drift_lc") else np.ones_like(band)
        # per event: share of S draws within +-2 m/s of the event speed (and matching geometry)
        per = [(p6[(np.abs(spd - x) <= 2)].sum(), p6[(np.abs(spd - x) <= 2) & geo].sum(), p6[(np.abs(spd - x) <= 2) & geo & al].sum(),
                p7[(np.abs(spd - x) <= 2) & geo & al].sum()) for x in v]
        per = np.array(per)
        out[c] = dict(events=int(len(g)), pts=float(g.loss.sum() / 2), v_ev_p10_50_90=[float(q) for q in np.nanpercentile(g.v_ev, [10, 50, 90])],
                      v_pre_max_p10_50_90=[float(q) for q in np.percentile(v, [10, 50, 90])], band=[float(lo), float(hi)],
                      S_draw_share_band_R6DAG=float(p6[band].sum()), S_draw_share_band_geo_R6DAG=float(p6[band & geo].sum()),
                      S_draw_share_band_geo_al_R6DAG=float(p6[band & geo & al].sum()), S_draw_share_band_geo_al_R7J=float(p7[band & geo & al].sum()),
                      per_event_pm2_R6DAG_median=float(np.median(per[:, 0])), per_event_pm2_geo_R6DAG_median=float(np.median(per[:, 1])),
                      per_event_pm2_geo_al_R6DAG_median=float(np.median(per[:, 2])), per_event_pm2_geo_al_R7J_median=float(np.median(per[:, 3])),
                      events_lt_0p5pct_geo_al_R7J=float((per[:, 3] < 0.005).mean()), events_over_13ms=float((v > 13).mean()),
                      events_over_20ms=float((v > 20).mean()))
    sdist = dict(R6DAG={b: float(p6[(spd >= lo_) & (spd < hi_)].sum()) for b, (lo_, hi_) in
                        {"<3": (0, 3), "3-8": (3, 8), "8-13": (8, 13), "13-20": (13, 20), "20-28": (20, 28), ">=28": (28, 99)}.items()},
                 R7J={b: float(p7[(spd >= lo_) & (spd < hi_)].sum()) for b, (lo_, hi_) in
                      {"<3": (0, 3), "3-8": (3, 8), "8-13": (8, 13), "13-20": (13, 20), "20-28": (20, 28), ">=28": (28, 99)}.items()},
                 max_speed=float(spd.max()), p99=float(np.percentile(spd, 99)),
                 sit_R6DAG=t.assign(p=p6).groupby("sit").p.sum().to_dict(), sit_R7J=t.assign(p=p7).groupby("sit").p.sum().to_dict(),
                 al_by_sit_R7J=t.assign(p=p7 * al).groupby("sit").p.sum().to_dict())
    out = dict(classes=out, S_speed=sdist)
    json.dump(out, open(W / "cover.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_pai(a):
    import pandas as pd
    F = {k: facts(k) for k in FACTS}
    common = sorted(set.intersection(*[set(f.index[f.rpc == False]) for f in F.values()]))  # noqa: E712
    cls = sorted({c for f in F.values() for c in f.cls.unique()} - {"ok"})
    out = dict(scenes=len(common), counts={}, pairs={})
    for k, f in F.items():
        f = f.loc[common]
        out["counts"][k] = dict(mean=float(f.score.mean()), **{c: [int((f.cls == c).sum()), float(((f.cls == c) * (1 - f.score)).sum())] for c in cls})
    rng = np.random.default_rng(0)
    n = len(common)
    idx = rng.integers(0, n, (4000, n))
    for A, B in (("R7J-s0", "R6DAG-s0"), ("R7J-s1", "R6DAG-s1"), ("R6DAG-s1", "R6DAG-s0"), ("R6DAG-s2", "R6DAG-s1"), ("R6DAG-s2", "R6DAG-s0")):
        fa, fb = F[A].loc[common], F[B].loc[common]
        res = {}
        la, lb = 1 - fa.score.to_numpy(), 1 - fb.score.to_numpy()
        for c in cls + ["lateral", "all"]:
            sel = ["off_turn_inside", "off_straight_junc_outside", "drift_lc"] if c == "lateral" else None
            ma = (fa.cls.isin(sel) if sel else (fa.cls != "ok") if c == "all" else fa.cls == c).to_numpy()
            mb = (fb.cls.isin(sel) if sel else (fb.cls != "ok") if c == "all" else fb.cls == c).to_numpy()
            d = la * ma - lb * mb                                             # points lost in the class (A - B), per scene
            bs = d[idx].sum(1)
            res[c] = dict(A_pts=float((la * ma).sum()), B_pts=float((lb * mb).sum()), d_pts=float(d.sum()),
                          ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], new=int((ma & ~mb).sum()), fixed=int((mb & ~ma).sum()))
        out["pairs"][f"{A} - {B}"] = res
    json.dump(out, open(W / "pai.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd_", required=True)
    p = sp.add_parser("fwd")
    p.add_argument("--ckpt", required=True), p.add_argument("--cmd", default=""), p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--target-all", type=int, default=1)
    p.add_argument("--seed", type=int, default=0), p.add_argument("--bs", type=int, default=512), p.add_argument("--threads", type=int, default=0)
    for c in ("batch", "learn", "cover", "pai"):
        sp.add_parser(c)
    a = ap.parse_args()
    W.mkdir(parents=True, exist_ok=True)
    dict(fwd=cmd_fwd, batch=cmd_batch, learn=cmd_learn, cover=cmd_cover, pai=cmd_pai)[a.cmd_](a)


if __name__ == "__main__":
    main()
