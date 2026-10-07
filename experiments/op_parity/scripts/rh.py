"""op_parity replay hinge (plans/2026-10-07-replay-hinge-prereg.md): drivable hinge on the devkit tracker's replayed footprint (R) and the same
plus a 0.5 m front-corner margin on turning tokens (RM), against the current 8-raw-pose hinge (control).

  proxy  (envs/navsim2, CPU)  lib/lqr_proxy.py vs the devkit PDMSimulator on P2H10-F-s0's stored navtest plans: footprint corner error of the
                              replay, DAC agreement (proxy footprint in the scorer's drivable polygons vs the devkit's own DAC sub-score); proxy
                              inputs v0 / a0 from the op_parity cache tab (what training sees), plus a run on the metric-cache ego state
                              -> $OUT/proxy.parquet, results/replay_hinge/proxy.json
  train  (op-train, CPU)      offline gate decoders (op_probe decode path, decision 147 addendum 1): MLP [P2-F-s0 plan-head hidden H, ego E] ->
                              8 poses on navtrain s2-s4 minus dev logs, imitation + lambda 10 hinge, three arms on the same rows / seed / batches
                              -> $OUT/decoder_poses.npz (all navtest), $OUT/fits.csv; scored by experiments/op_probe/scripts/opb_score.py
  report (.venv)              gate tables (log-cluster paired bootstrap) -> results/replay_hinge/gate*.csv|md, gate.json

  $DATA_DIR/envs/navsim2/bin/python experiments/op_parity/scripts/rh.py proxy --procs 64
  $DATA_DIR/envs/op-train/bin/python experiments/op_parity/scripts/rh.py train
  $DATA_DIR/envs/navsim2/bin/python experiments/op_probe/scripts/opb_score.py --poses $OUT/decoder_poses.npz --out $OUT/score.csv
  .venv/bin/python experiments/op_parity/scripts/rh.py report
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "proxy":                                   # devkit workers: one BLAS thread each
    for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_k, "1")
    os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

import numpy as np  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/op_probe/scripts")]
OUT = D / "runs/op_parity/replay_hinge"
RES = REPO / "experiments/op_parity/results/replay_hinge"
MC = D / "runs/navsim/metric_cache/v2_navtest"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
P2H = D / "runs/op_lb/lb_navtest/preds/warp-cinque_PPP2H10-F-s0__base.npz"
P2H_CSV = D / "runs/navsim/eval/v2_navtest_opi_lb_navtest_warp-cinque_PPP2H10-F-s0__base"
TURN_DEG, M_BASE, M_FRONT, LAM = 20.0, 0.3, 0.5, 10.0
ARMS = ("ctrl", "R", "RM")
_W = {}


def turn_mask(fut):
    """|logged heading change at 4 s| > 20 deg."""
    return np.abs(np.degrees(np.arctan2(np.sin(fut[:, 7, 2]), np.cos(fut[:, 7, 2])))) > TURN_DEG


# ---------------------------------------------------------------- proxy validation
def _init_proxy():
    for k, v in dict(NUPLAN_MAP_VERSION="nuplan-maps-v1.0", NUPLAN_MAPS_ROOT=str(D / "datasets/navsim/maps"),
                     OPENSCENE_DATA_ROOT=str(D / "datasets/navsim"), NAVSIM_EXP_ROOT=str(D / "runs/navsim/eval"),
                     NAVSIM_DEVKIT_ROOT=str(D / "third_party/navsim")).items():
        os.environ.setdefault(k, v)
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    samp = TrajectorySampling(num_poses=40, interval_length=0.1)
    _W.update(sim=PDMSimulator(samp), samp=samp)


def _to_global(o, S):
    c, s = np.cos(o[2]), np.sin(o[2])
    g = np.zeros((len(S), 11))
    g[:, 0], g[:, 1], g[:, 2] = o[0] + c * S[:, 0] - s * S[:, 1], o[1] + s * S[:, 0] + c * S[:, 1], o[2] + S[:, 2]
    return g


def _work_proxy(token):
    import lzma
    import pickle
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    W = _W
    with lzma.open(MC / W["path"][token], "rb") as f:
        mc = pickle.load(f)
    p8 = W["plans"][token]
    es = mc.ego_state
    st = get_trajectory_as_array(transform_trajectory(Trajectory(p8), es), W["samp"], es.time_point)
    dev = W["sim"].simulate_proposals(st[None], es)[0]                                       # (41, 11) global
    o = np.array(es.rear_axle.serialize())
    vp = es.car_footprint.vehicle_parameters
    am = mc.drivable_area_map
    idc = am.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
    out = dict(token=token, v0_mc=float(es.dynamic_car_state.rear_axle_velocity_2d.x), a0_mc=float(es.dynamic_car_state.rear_axle_acceleration_2d.x),
               steer0_mc=float(es.tire_steering_angle))
    cd = state_array_to_coords_array(dev[None], vp)[0][:, :4]                                  # (41, 4, 2)
    out["dev_out"] = not am.points_in_polygons(cd[None])[idc].any(0)[0].all()
    for k in ("tab", "mcin"):
        g = _to_global(o, W[k][token])
        cp = state_array_to_coords_array(g[None], vp)[0][:, :4]
        err = np.linalg.norm(cp - cd, axis=-1)
        out[f"{k}_corner_max"], out[f"{k}_pos_max"] = float(err.max()), float(np.linalg.norm(g[:, :2] - dev[:, :2], axis=-1).max())
        out[f"{k}_out"] = not am.points_in_polygons(cp[None])[idc].any(0)[0].all()
    return out


def cmd_proxy(a):
    import glob
    import multiprocessing as mp
    import pandas as pd
    import torch
    import lqr_proxy as LP
    from jevdrive.run import Run
    with Run("op_parity", "replay_hinge-proxy", seed=0, config=vars(a)) as run:
        z = np.load(P2H)
        cp = {Path(p).parent.name: str(Path(p).relative_to(MC)) for p in glob.glob(str(MC / "*/*/*/metric_cache.pkl"))}
        tab = np.load(TAB)
        tp = {t: i for i, t in enumerate(tab["names"].tolist())}
        toks = [t for t in z["tokens"].tolist() if t in cp and t in tp]
        if a.limit:
            toks = toks[:a.limit]
        P = torch.as_tensor(z["poses"], dtype=torch.float64)
        zi = {t: i for i, t in enumerate(z["tokens"].tolist())}
        rows = np.array([tp[t] for t in toks])
        v0 = torch.as_tensor(tab["vel"][rows, -1, 0], dtype=torch.float64)
        a0 = torch.as_tensor(tab["acc"][rows, -1, 0], dtype=torch.float64)
        Pt = P[[zi[t] for t in toks]]
        t0 = time.time()
        with torch.no_grad():
            Rt = torch.cat([LP.replay(Pt[i:i + 4096], v0[i:i + 4096], a0[i:i + 4096]) for i in range(0, len(Pt), 4096)]).numpy()
        run.info(f"proxy replay of {len(toks)} plans in {time.time() - t0:.1f} s")
        _W.update(path=cp, plans={t: Pt[i].numpy() for i, t in enumerate(toks)}, tab={t: Rt[i] for i, t in enumerate(toks)})
        # metric-cache inputs: v0 / a0 read per token in a first pass (cheap) so the proxy on the devkit's exact initial state is also measured
        _init_proxy()
        with mp.get_context("fork").Pool(a.procs) as pool:
            ego = pd.DataFrame(pool.map(_ego_mc, toks, chunksize=16)).set_index("token").loc[toks]
        run.info(f"metric-cache ego states read, {time.time() - t0:.0f} s")
        with torch.no_grad():
            Rm = LP.replay(Pt, torch.as_tensor(ego.v0.to_numpy()), torch.as_tensor(ego.a0.to_numpy()),
                           torch.as_tensor(ego.steer0.to_numpy())).numpy()
        _W["mcin"] = {t: Rm[i] for i, t in enumerate(toks)}
        res = []
        with mp.get_context("fork").Pool(a.procs) as pool:
            for k, r in enumerate(pool.imap_unordered(_work_proxy, toks, chunksize=8)):
                res.append(r)
                if (k + 1) % 2000 == 0:
                    run.status(f"{k + 1}/{len(toks)}, {time.time() - t0:.0f} s")
        df = pd.DataFrame(res).set_index("token").loc[toks].reset_index()
        df["v0_tab"], df["a0_tab"] = v0.numpy(), a0.numpy()
        off = pd.read_csv(sorted(P2H_CSV.glob("*/*.csv"))[-1])
        off = off[~off.token.astype(str).str.startswith(("average", "extended"))].set_index("token")
        df["dac_official"] = off.loc[df.token, "drivable_area_compliance"].to_numpy()
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT / "proxy.parquet")
        rng = np.random.default_rng(0)
        gate = np.zeros(len(df), bool)
        gate[rng.choice(len(df), min(300, len(df)), replace=False)] = True
        summ = {}
        for name, m in (("gate300", gate), ("all", np.ones(len(df), bool)), ("dac_fail", df.dac_official.to_numpy() < 1)):
            d = df[m]
            fail = d.dac_official.to_numpy() < 1
            s = dict(n=int(m.sum()), n_dac_fail=int(fail.sum()))
            for k in ("tab", "mcin"):
                s[k] = dict(corner_p95=float(np.quantile(d[f"{k}_corner_max"], 0.95)), corner_p99=float(np.quantile(d[f"{k}_corner_max"], 0.99)),
                            corner_max=float(d[f"{k}_corner_max"].max()), corner_median=float(d[f"{k}_corner_max"].median()),
                            dac_agree=float((d[f"{k}_out"].to_numpy() == fail).mean()))
            s["devkit_polygon_vs_official_agree"] = float((d.dev_out.to_numpy() == fail).mean())
            s["v0_abs_diff_max"] = float((d.v0_tab - d.v0_mc).abs().max())
            s["a0_abs_diff_max"] = float((d.a0_tab - d.a0_mc).abs().max())
            s["steer0_abs_max"] = float(d.steer0_mc.abs().max())
            summ[name] = s
        g = summ["gate300"]["tab"]
        summ["verdict"] = dict(rule="gate300, tab inputs: corner p95 <= 0.1 m and DAC agreement >= 99%",
                               pass_=bool(g["corner_p95"] <= 0.1 and g["dac_agree"] >= 0.99))
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "proxy.json").write_text(json.dumps(summ, indent=1))
        run.summary.update(summ)
        run.info(json.dumps(summ, indent=1))


def _ego_mc(token):
    import lzma
    import pickle
    with lzma.open(MC / _W["path"][token], "rb") as f:
        es = pickle.load(f).ego_state
    return dict(token=token, v0=float(es.dynamic_car_state.rear_axle_velocity_2d.x), a0=float(es.dynamic_car_state.rear_axle_acceleration_2d.x),
                steer0=float(es.tire_steering_angle))


# ---------------------------------------------------------------- offline gate decoders
def cmd_train(a):
    import torch
    import torch.nn as nn
    import opb_probe as O
    import lqr_proxy as LP
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    torch.set_num_threads(n_cpus())
    dev = torch.device(a.device)
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "replay_hinge-train", seed=0, config=vars(a)) as run:
        toks, datas, is_dev, dvs = O.train_tokens(False)
        run.use_split(dvs), run.use_split(splits.load("navsim/navtrain")), run.use_split(splits.load("navsim/navtest"))
        lab_pos, LZ = O.labels("navtrain_s23456")
        li = np.array([lab_pos[t] for t in toks])
        tabs = {d: np.load(O.CACHE / d / "tab.npz") for d in dict.fromkeys(datas)}
        cat = lambda k: np.concatenate([tabs[d][k] for d in dict.fromkeys(datas)])  # noqa: E731
        fut, ego, vel, acc = cat("fut"), cat("ego").astype(np.float32), cat("vel"), cat("acc")
        use = LZ["ok"][li] & ~np.isnan(fut[:, 0, 0]) & ~is_dev
        sdf = torch.as_tensor(LZ["sdf"][li[use]], device=dev)[:, None]
        Y = torch.as_tensor(fut[use], device=dev).float()
        V0 = torch.as_tensor(vel[use, -1, 0], device=dev, dtype=torch.float64)
        A0 = torch.as_tensor(acc[use, -1, 0], device=dev, dtype=torch.float64)
        turn = torch.as_tensor(turn_mask(fut[use]), device=dev)
        ttab = np.load(O.CACHE / "lb_navtest" / "tab.npz")
        tt = ttab["names"]
        e_ego = ttab["ego"].astype(np.float32)
        assert O.has_feats("P2-F-s0", "lb_navtest", tt).all()
        M = torch.as_tensor(O._interp_matrix(), device=dev, dtype=torch.float32)
        C = torch.as_tensor(O.CORNERS, device=dev, dtype=torch.float32)
        Xa = torch.as_tensor(O.stage_matrix("P2-H", "P2-F-s0", "H", toks[use], datas[use], ego[use]), device=dev)
        mu, sd = Xa.mean(0), Xa.std(0).clamp_min(1e-6)
        Xa = (Xa - mu) / sd
        Xe = (torch.as_tensor(O.stage_matrix("P2-H", "P2-F-s0", "H", tt, np.array(["lb_navtest"] * len(tt)), e_ego), device=dev) - mu) / sd
        run.info(f"train rows {len(Xa)} (turning {float(turn.float().mean()):.3f}), navtest {len(Xe)}, features {Xa.shape[1]}")
        # per-corner margins for RM: corners ordered FL, FR, RL, RR (lib/drivable_hinge.CORNERS); front corners 0.5 m on turning rows
        mfront = torch.tensor([1.0, 1.0, 0.0, 0.0], device=dev)
        res, rows = {"tokens": tt}, []
        for arm in a.arms:
            torch.manual_seed(0)
            net = nn.Sequential(nn.Dropout(0.1), nn.Linear(Xa.shape[1], 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 24)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            wu = max(1, a.steps // 20)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / wu) * 0.5 * (1 + np.cos(np.pi * min(k, a.steps) / a.steps)))
            g = torch.Generator(device=dev).manual_seed(0)
            t1 = time.time()
            for step in range(a.steps):
                b = torch.randint(0, len(Xa), (a.batch,), device=dev, generator=g)
                P = net(Xa[b]).view(-1, 8, 3)
                li_ = nn.functional.huber_loss(P[..., :2], Y[b, :, :2], delta=1.0) + 3.0 * nn.functional.huber_loss(P[..., 2], Y[b, :, 2], delta=0.1)
                if arm == "ctrl":
                    cor = O.corners_torch(P, M, C)
                    marg = M_BASE
                else:
                    S = LP.replay(P.double(), V0[b], A0[b]).float()                              # (B, 41, 3)
                    c, s = torch.cos(S[..., 2:3]), torch.sin(S[..., 2:3])
                    cx = S[..., :1] + c * C[None, None, :, 0] - s * C[None, None, :, 1]
                    cy = S[..., 1:2] + s * C[None, None, :, 0] + c * C[None, None, :, 1]
                    cor = torch.stack([cx, cy], -1).reshape(len(P), -1, 2)
                    marg = M_BASE
                    if arm == "RM":
                        marg = (M_BASE + (M_FRONT - M_BASE) * turn[b].float()[:, None, None] * mfront[None, None]).expand(-1, 41, -1).reshape(len(P), -1)
                lh = torch.relu(marg - O.sdf_at(sdf[b].float(), cor)).mean()
                loss = li_ + LAM * lh
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                if step % 500 == 0:
                    run.status(f"{arm} step {step} loss {float(loss):.4f} imit {float(li_):.4f} hinge {float(lh):.4f} {time.time() - t1:.0f} s")
            net.eval()
            with torch.no_grad():
                Pe = torch.cat([net(Xe[i:i + 4096]).view(-1, 8, 3) for i in range(0, len(Xe), 4096)]).cpu().numpy()
            res[arm] = Pe.astype(np.float32)
            rows.append(dict(arm=arm, final_loss=float(loss), imit=float(li_), hinge=float(lh), train_s=time.time() - t1))
            run.info(json.dumps(rows[-1]))
        np.savez(OUT / a.out, **res)
        import pandas as pd
        pd.DataFrame(rows).to_csv(OUT / "fits.csv", index=False)
        run.summary["out"] = str(OUT / a.out)


# ---------------------------------------------------------------- gate report
def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    df = pd.read_csv(OUT / "score.csv")
    tab = np.load(TAB)
    tp = {t: i for i, t in enumerate(tab["names"].tolist())}
    dpsi = np.degrees(np.arctan2(np.sin(tab["fut"][:, 7, 2]), np.cos(tab["fut"][:, 7, 2])))
    piv = {k: g.set_index("token") for k, g in df.groupby("key")}
    toks = sorted(set.intersection(*[set(g.index) for g in piv.values()]))
    rows = np.array([tp[t] for t in toks])
    logs, d = tab["log"][rows], np.abs(dpsi[rows])
    turn = d > TURN_DEG
    fail = {k: (piv[k].loc[toks, "drivable_area_compliance"].to_numpy() < 1).astype(float) for k in piv}
    score = {k: piv[k].loc[toks, "score"].to_numpy() * 100 for k in piv}
    lqr_only = {k: ((piv[k].loc[toks, "lqr_out"].to_numpy().astype(bool)) & ~(piv[k].loc[toks, "raw_out"].to_numpy().astype(bool))).astype(float)
                for k in piv}
    ep = {k: piv[k].loc[toks, "ego_progress"].to_numpy() * 100 for k in piv}
    out, gate = [], {}
    strata = [("all", np.ones(len(toks), bool)), ("turn > 20 deg", turn), ("< 5 deg", d < 5), ("5-20 deg", (d >= 5) & (d <= 20)),
              ("20-45 deg", (d > 20) & (d <= 45)), ("> 45 deg", d > 45)]
    for arm in [k for k in ARMS if k in piv and k != "ctrl"]:
        for sname, m in strata:
            for met, X, sc in (("DAC fail pp", fail, 100), ("score x100", score, 1), ("LQR-only DAC fail pp", lqr_only, 100), ("EP x100", ep, 1)):
                r = stats.paired(X[arm][m] * sc, X["ctrl"][m] * sc, groups=logs[m])
                out.append(dict(contrast=f"{arm} - ctrl", stratum=sname, metric=met, n=int(m.sum()), arm_mean=r["mean_a"], ctrl_mean=r["mean_b"],
                                diff=r["mean"], lo=r["lo"], hi=r["hi"], logs=r["units"]))
        ft = [o for o in out if o["contrast"] == f"{arm} - ctrl" and o["stratum"] == "turn > 20 deg" and o["metric"] == "DAC fail pp"][0]
        sa = [o for o in out if o["contrast"] == f"{arm} - ctrl" and o["stratum"] == "all" and o["metric"] == "score x100"][0]
        gate[arm] = dict(turn_dac_fail_drop_pp=-ft["diff"], ci_pp=[-ft["hi"], -ft["lo"]], score_diff=sa["diff"], score_ci=[sa["lo"], sa["hi"]],
                         dac_pass=bool(-ft["diff"] >= 0.4), score_pass=bool(sa["hi"] >= 0.0))
        gate[arm]["pass"] = gate[arm]["dac_pass"] and gate[arm]["score_pass"]
    passing = [k for k in gate if gate[k]["pass"]]
    gate["winner"] = max(passing, key=lambda k: gate[k]["turn_dac_fail_drop_pp"]) if passing else None
    gate["verdict"] = "pilot" if passing else "stop"
    gate["n_tokens"], gate["n_turn"] = len(toks), int(turn.sum())
    base = []
    for k in piv:
        for sname, m in strata:
            base.append(dict(arm=k, stratum=sname, n=int(m.sum()), **{f"dac_fail_pct": float(fail[k][m].mean() * 100),
                                                                        "score_x100": float(score[k][m].mean()),
                                                                        "lqr_only_pct": float(lqr_only[k][m].mean() * 100),
                                                                        "raw_out_pct": float(piv[k].loc[np.array(toks)[m], "raw_out"].mean() * 100)}))
    RES.mkdir(parents=True, exist_ok=True)
    stats.write_table(out, RES / "gate_paired", floatfmt=".3f", note="paired arm - ctrl, cluster bootstrap over navtest logs, B 10 000")
    stats.write_table(base, RES / "gate_arms", floatfmt=".2f")
    (RES / "gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("proxy")
    p.add_argument("--procs", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("train")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--out", default="decoder_poses.npz")
    p = sp.add_parser("report")
    a = ap.parse_args()
    {"proxy": cmd_proxy, "train": cmd_train, "report": cmd_report}[a.cmd](a)
