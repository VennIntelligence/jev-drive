"""op_parity turn selector in HUGSIM closed loop (plans/2026-10-08-turn-selector-hugsim-prereg.md): the decision-191 selector (N7, F19 x pc) with gate B/A/0
of decision 193, applied per simulator step to SH30's own plan.

  Selector (op-train, used by pp_hugsim.py serve --select GATE)   one step: the policy server's plan (camera frame), road edges and the tapped tokens
        (view_39 of the current frame, select_4, mean) + the ego features -> 8 rear-axle poses at 0.5 .. 4 s (as the navsim export) -> 19 candidate
        own-edge margins -> N7 gains -> pick; gate B passes the step only when the plan's heading at 4 s is >= 20 deg. A picked candidate is applied
        to the dense 33-point plan as the difference (candidate - identity) of the 8 poses, interpolated in time (held beyond 4 s), and the lateral
        curvature of the preset (mean over 0.5-1.5 s of the plan) is recomputed from the new plan.
  check   (op-train, GPU)   G-eqv: the per-step Selector reproduces the batch `select` stage (picks and candidate poses) on navtest rows
  trt     (openpilot env, GPU)   build the TensorRT engines of the tapped SH30 ONNX files once (before the workers start)
  idcheck (jev env)  G-id: ts0 on the 11 pilot scenarios vs the archived SH30 run (prereg criteria i-iv)
  pilot   (jev env)  G-pilot: the checklist of the prereg on the 11-scenario tsB run
  report  (jev env)  closed-loop readout: paired HD vs SH30, failure counts, gate / pick statistics, steering / lateral-jerk -> results/ and figs/
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, json, re  # noqa: E401,E402

import numpy as np  # noqa: E402

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
T_OUT = np.arange(1, 9) * 0.5
SMOOTH_WINDOW = (0.5, 1.5)                                # jevdrive.openpilot.model.SMOOTH_WINDOW (model seconds)
MIN_SPEED = 1.0


def to_rear(pos, yaw, d, t_out):
    """jevdrive.op_interp.to_rear, mode lever, linear: camera-frame plan (x fwd, y right, yaw clockwise) -> rear-axle (len(t_out), 3) x, y (left), yaw (left)."""
    p = np.stack([pos[:, 0], -pos[:, 1]], -1).astype(np.float64)
    psi = -np.asarray(yaw, np.float64)
    d = np.asarray(d, np.float64)
    Rd = np.stack([np.cos(psi) * d[0] - np.sin(psi) * d[1], np.sin(psi) * d[0] + np.cos(psi) * d[1]], -1)
    rear = np.concatenate([d + p - Rd, psi[:, None]], 1)
    return np.stack([np.interp(t_out, T_IDXS, rear[:, k]) for k in range(3)], -1)


def curvature_window(xy, yaws, t0=SMOOTH_WINDOW[0], t1=SMOOTH_WINDOW[1]):
    """jevdrive.openpilot.model.curvature_window."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    ds = np.interp(t1, T_IDXS, s) - np.interp(t0, T_IDXS, s)
    return (np.interp(t1, T_IDXS, yaws) - np.interp(t0, T_IDXS, yaws)) / max(ds, MIN_SPEED * (t1 - t0))


def apply_delta(pos, yaw, d, P, Q):
    """Dense camera-frame plan (pos (33, 3), yaw (33)) moved so that its 8 rear-axle poses P become Q: the difference Q - P at 0.5 .. 4 s is
    interpolated to the 33 plan times (zero at t = 0, held beyond 4 s) and added to the dense rear-axle plan, which is mapped back to the camera frame."""
    dd = np.asarray(Q, np.float64) - np.asarray(P, np.float64)
    dd[:, 2] = np.angle(np.exp(1j * dd[:, 2]))
    t = np.r_[0.0, T_OUT]
    dd = np.concatenate([np.zeros((1, 3)), dd])
    rear = to_rear(pos, yaw, d, T_IDXS)
    rear = rear + np.stack([np.interp(T_IDXS, t, dd[:, k]) for k in range(3)], -1)
    psi = rear[:, 2]
    d = np.asarray(d, np.float64)
    Rd = np.stack([np.cos(psi) * d[0] - np.sin(psi) * d[1], np.sin(psi) * d[0] + np.cos(psi) * d[1]], -1)
    p = rear[:, :2] - d + Rd
    new = np.array(pos, np.float64)
    new[:, 0], new[:, 1] = p[:, 0], -p[:, 1]
    return new, -psi


class Selector:
    """The N7 selector as a per-step function. Loads the bundle of turn_selbench (refit; decision 191 protocol) and the per-seed edge calibration."""

    def __init__(self, seed: int, gate: str):
        import turn_ceiling as TC
        import turn_selbench as TB
        import turn_selinput as TS
        assert gate in ("A", "B", "0"), gate
        self.TC, self.TB, self.TS, self.gate = TC, TB, TS, gate
        self.bundle = TB.load_bundle()
        c = json.loads(TS.CAL.read_text())[f"SH30-F-s{seed}"]
        self.cal = (c["s"], c["b"])
        self.cands = TC.candidates()

    def run(self, ego, pos, yaw, re, v3, h4, hm, cam_xy):
        """ego (20,), pos (33, 3) / yaw (33) camera-frame plan, re (2, 33, 2) road-edge mu, v3 (32, 512), h4 / hm (512,), cam_xy: camera (x, y) on the vehicle
        relative to the rear axle (m). -> dict(pick, allowed, dyaw, gain, applied, pos, yaw, dk) where pos / yaw are the plan to use and dk the change of the smooth curvature."""
        import sc_analyze as SC
        import turn_dewater as TD
        TB = self.TB
        cam = np.asarray(cam_xy, np.float64)[None, :2]
        P = to_rear(pos, yaw, cam[0], T_OUT).astype(np.float32)[None]
        ex, ey = SC.edges_ego(np.asarray(re, np.float32)[None], cam)
        C = TB.curb_margins(P, ex, ey, self.cal)
        e = np.concatenate([np.asarray(ego, np.float64)[None], TD.plan_desc(P.astype(np.float64))], -1)
        h = np.concatenate([np.asarray(h4), np.asarray(hm)])[None].astype(np.float32)
        pred, _ = TB.infer(self.bundle, e, h, C.reshape(1, -1), np.asarray(v3, np.float16).reshape(1, 32, 512))
        dyaw = float(TB.model_yaw_deg(P)[0])
        allowed = {"A": True, "B": dyaw >= TB.TURN_DEG, "0": False}[self.gate]
        pk = int(TD.picks_of(pred)[0])
        pick = pk if allowed else 0
        out = dict(pick=pick, pick_free=pk, allowed=bool(allowed), dyaw=dyaw, gain=float(pred[0, pk]), applied=pick != 0, pos=pos, yaw=yaw, dk=0.0,
                   margin_id=float(C[0, 0].min()))
        if pick:
            Q = self.TC.transform(P.astype(np.float64), *self.cands[pick][1:])[0]
            new_pos, new_yaw = apply_delta(pos, yaw, cam[0], P[0], Q)
            out.update(pos=new_pos.astype(np.float32), yaw=new_yaw.astype(np.float32), Q=Q,
                       dk=curvature_window(new_pos[:, :2], new_yaw) - curvature_window(np.asarray(pos, np.float64)[:, :2], np.asarray(yaw, np.float64)))
        return out


# ---------------------------------------------------------------- G-eqv
def cmd_check(a):
    """Per-step Selector vs the batch select stage on navtest rows of one SH30 seed: pick agreement and the candidate-pose error of the dense-plan round trip."""
    from dataclasses import replace
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import resolve
    import turn_selbench as TB
    m = resolve(f"SH30-F-s{a.seed}@warp:tsB", check=True)
    base = replace(m, opt="")
    sel = np.load(TB.select_file(m, "navtest"))
    z = np.load(N.plan_file(base, "navtest"))
    f = TB.features(base, "navtest")
    assert f["names"].tolist() == sel["tokens"].tolist()
    row = {t: i for i, t in enumerate(z["names"].tolist())}
    rng = np.random.default_rng(0)
    turn = np.flatnonzero(sel["gate"])
    idx = np.r_[rng.choice(turn, a.n, replace=False), rng.choice(np.flatnonzero(~sel["gate"]), a.n // 3, replace=False)]
    S = Selector(a.seed, "B")
    res = []
    for i in idx:
        j = row[f["names"][i]]
        cam = f["cam"][i][:2]
        o = S.run(f["ego"][i], z["plan_pos"][j], z["plan_yaw"][j], f["re"][i], f["V3"][i], f["h4"][i], f["hm"][i], cam)
        res.append((o["allowed"] == bool(sel["gate"][i]), o["pick"] == int(sel["picks"][i]), o["pick_free"]))
        if o["applied"]:
            back = to_rear(o["pos"], o["yaw"], cam, T_OUT)
            res[-1] += (float(np.abs(back[:, :2] - o["Q"][:, :2]).max()), float(np.abs(np.angle(np.exp(1j * (back[:, 2] - o["Q"][:, 2])))).max()))
    n_app = sum(len(r) > 3 for r in res)
    err = np.array([r[3:] for r in res if len(r) > 3])
    out = dict(n=len(res), gate_agree=float(np.mean([r[0] for r in res])), pick_agree=float(np.mean([r[1] for r in res])), n_applied=n_app,
               xy_err_max_m=float(err[:, 0].max()) if n_app else None, xy_err_p50_m=float(np.median(err[:, 0])) if n_app else None,
               yaw_err_max_rad=float(err[:, 1].max()) if n_app else None)
    out["ok"] = bool(out["gate_agree"] == 1.0 and out["pick_agree"] >= 0.98)
    print(json.dumps(out, indent=1))
    (TB.OUTB / f"gate_eqv_s{a.seed}.json").write_text(json.dumps(out, indent=1))
    assert out["ok"], out


# ---------------------------------------------------------------- runs on disk
PRESET = "spec_plan_smooth"
RES = _R / "experiments/op_parity/results/turn_selector_hugsim"
FIG = _R / "experiments/op_parity/figs/turn_selector_hugsim"
S11 = _R / "experiments/op_parity/scripts/turn_selhug_s11.txt"


def cmd_trt(a):
    """Engine build for the tapped ONNX (the workers would otherwise all build the same engine into one cache directory at once)."""
    import time
    from jevdrive.bench.runner import bench_root
    from jevdrive.openpilot.model import OPModel
    for s in (0, 1):
        t = time.time()
        m = OPModel(str(bench_root("hugsim", "_onnx", f"pp-SH30-F-s{s}.onnx")), "trt", taps=["view_39", "select_4", "mean"])
        m.extra = {"intent_bias": np.zeros((1, 32, 512), np.float16)}
        for _ in range(4):
            out = m.step(np.zeros((2, 6, 128, 256), np.uint8))
        print(f"s{s}: engine built and run in {time.time() - t:.0f} s", {k: (v.shape, float(np.abs(v).max())) for k, v in m.tap_values.items()}, out.shape, flush=True)
        del m


def run_path(spec: str):
    from jevdrive.bench import run_dir
    return run_dir(spec, "hugsim", PRESET)


def load_units(spec: str):
    import pandas as pd
    d = run_path(spec)
    u = pd.read_csv(d / "units.csv").set_index("scenario")
    return u


def steps_of(run_dir_col: str):
    """Step records of one scenario's zs_steps.jsonl."""
    out = []
    for line in open(_pl.Path(run_dir_col) / "zs_steps.jsonl"):
        r = json.loads(line)
        if "step" in r:
            out.append(r)
    return out


def scen_stats(units):
    """Per scenario: n steps, selector gate / pick statistics (tsB / ts0 runs), steering and lateral-jerk statistics (all runs)."""
    rows = {}
    for sc, u in units.iterrows():
        R = steps_of(u.run_dir)
        t = np.array([r["t"] for r in R], float)
        v = np.array([r["v"] for r in R], float)
        th = np.array([r["theta"] for r in R], float)
        st = np.array([r["steer"] for r in R], float)
        o = dict(n=len(R))
        sel = [r.get("sel") for r in R]
        if any(x is not None for x in sel):
            allow = np.array([bool(x and x["allowed"]) for x in sel])
            pick = np.array([int(x["pick"]) if x else 0 for x in sel])
            free = np.array([int(x["pick_free"]) if x else 0 for x in sel])
            pair = allow[1:] & allow[:-1]
            o.update(n_sel=int(sum(x is not None for x in sel)), n_allowed=int(allow.sum()), n_applied=int((pick != 0).sum()), n_free_moved=int((free != 0).sum()),
                     n_pairs_allowed=int(pair.sum()), n_switch_allowed=int((pick[1:] != pick[:-1])[pair].sum()), n_switch_all=int((pick[1:] != pick[:-1]).sum()),
                     n_pairs=len(R) - 1, rtt_ms=[float(x["rtt_ms"]) for x in sel if x], absdk=[abs(float(x["dk"])) for x in sel if x and x["applied"]])
        if len(R) > 3:
            dt = np.maximum(np.diff(t), 1e-3)
            om = np.diff(th) / dt
            al = v[1:] * om
            jk = np.diff(al) / dt[1:]
            m = v[2:] > 1.0
            o["jerk_rms"] = float(np.sqrt(np.mean(jk[m] ** 2))) if m.sum() >= 3 else np.nan
            ds = np.abs(np.diff(st))[v[1:] > 1.0]
            o["dsteer_mean"] = float(ds.mean()) if len(ds) else np.nan
            o["dsteer_p95"] = float(np.percentile(ds, 95)) if len(ds) else np.nan
        rows[sc] = o
    return rows


def cmd_idcheck(a):
    """G-id (prereg, criterion iv amended): ts0 vs archived SH30 on the 11 scenarios. model_pos holds plan points 4, 8, 12, 16, 20, 24, 32; only the first five
    (<= 3.9 model seconds, the span the selector and the HUGSIM plan use) are compared."""
    names = [_pl.Path(x).stem for x in S11.read_text().split()]
    new, old = load_units(f"SH30-F-s{a.seed}:ts0").loc[names], load_units(f"SH30-F-s{a.seed}").loc[names]
    d = (new.hdscore - old.hdscore).abs()
    same_end = (new.end == old.end)
    dpos = []
    for sc in names:
        if same_end[sc]:
            A, B = steps_of(new.run_dir[sc]), steps_of(old.run_dir[sc])
            n = min(5, len(A), len(B))
            dpos.append(max(np.abs(np.array(A[i]["model_pos"])[:5] - np.array(B[i]["model_pos"])[:5]).max() for i in range(n)))
    out = dict(n=len(names), same_end=int(same_end.sum()), hd_within_0p05=int((d < 0.05).sum()), mean_abs_dhd=float(d.mean()), max_abs_dhd=float(d.max()),
               first5_plan_maxdiff_m=float(max(dpos)) if dpos else None, per_scenario={sc: dict(new=float(new.hdscore[sc]), old=float(old.hdscore[sc]), end_new=new.end[sc],
                                                                                                  end_old=old.end[sc]) for sc in names})
    out["ok"] = bool(out["same_end"] >= 9 and out["hd_within_0p05"] >= 9 and out["mean_abs_dhd"] <= 0.02 and out["first5_plan_maxdiff_m"] <= 0.05)
    f = TB_OUTB() / f"gate_id_s{a.seed}.json"
    f.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_scenario"}, indent=1))


def TB_OUTB():
    from jevdrive.bench.models import data_dir
    d = data_dir() / "runs/op_parity/turn_selhug"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_pilot(a):
    """G-pilot (prereg checklist) on the tsB seed-0 run over the 11 scenarios, against ts0."""
    names = [_pl.Path(x).stem for x in S11.read_text().split()]
    turn = {_pl.Path(x).stem for x in (_R / "jevdrive/bench/lists/hugsim_turn23.txt").read_text().split()}
    ub, u0 = load_units(f"SH30-F-s{a.seed}:tsB").loc[names], load_units(f"SH30-F-s{a.seed}:ts0").loc[names]
    sb = scen_stats(ub)
    chk = {}
    chk["all_completed"] = bool(len(ub) == len(names) and ub.end.notna().all())
    gt = sum(sb[s]["n_allowed"] for s in names if s in turn)
    nt = sum(sb[s]["n"] for s in names if s in turn)
    gs = sum(sb[s]["n_allowed"] for s in names if s not in turn)
    ns = sum(sb[s]["n"] for s in names if s not in turn)
    chk["gate_fire_turn"], chk["gate_fire_straight"] = gt / max(nt, 1), gs / max(ns, 1)
    chk["gate_ok"] = bool(gt > 0 and chk["gate_fire_straight"] < 0.10)
    rtt = np.concatenate([sb[s]["rtt_ms"] for s in names])
    chk["rtt_p95_ms"] = float(np.percentile(rtt, 95))
    ad = np.concatenate([sb[s]["absdk"] for s in names] + [np.zeros(1)])
    chk["max_abs_dk"] = float(ad.max())
    chk["every_step_logged"] = bool(all(sb[s]["n_sel"] == sb[s]["n"] for s in names))

    def bad(u):
        return int(((u.cls == "spin") | (u.end == "max_steps") | u.launch_stall.astype(bool)).sum())
    chk["unstable_tsB"], chk["unstable_ts0"] = bad(ub), bad(u0)
    chk["ok"] = bool(chk["all_completed"] and chk["gate_ok"] and chk["rtt_p95_ms"] < 100 and chk["max_abs_dk"] < 0.1 and chk["every_step_logged"]
                     and chk["unstable_tsB"] <= chk["unstable_ts0"])
    chk["hd"] = {sc: (float(ub.hdscore[sc]), float(u0.hdscore[sc])) for sc in names}
    f = TB_OUTB() / f"gate_pilot_s{a.seed}.json"
    f.write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=240)
    sp.add_parser("trt")
    for n in ("idcheck", "pilot"):
        p = sp.add_parser(n)
        p.add_argument("--seed", type=int, default=0)
    sp.add_parser("report")
    a = ap.parse_args()
    {"check": cmd_check, "trt": cmd_trt, "idcheck": cmd_idcheck, "pilot": cmd_pilot, "report": lambda a: __import__("turn_selhug_report").report(a)}[a.cmd](a)


if __name__ == "__main__":
    main()
