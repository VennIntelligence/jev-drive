"""op_parity turn selector on WOD-E2E val (plans/2026-10-08-turn-selwod-prereg.md): the decision-193 selector (N7, F19 x pc, gate B), applied unchanged to the
plan of an adapted openpilot on the 479 WOD val rater frames, valued by RFS.

  extract  (openpilot env, GPU, pool)  the serving ONNX of an arm with taps view_39 / select_4 / mean, run through the decision-155 WOD harness (real frames,
                                       20 Hz, 9 policy slots, the arm's intent bias); last-step raw plan, road edges, taps -> runs/op_parity/turn_selwod/feat/<tag>.npz
  select   (op-train env, GPU, pool)   features -> 19 candidate margins -> N7 gains -> picks; F19 candidates on the dense plan -> 20 WOD waypoints
                                       -> runs/op_parity/turn_selwod/sel/<tag>.npz     (also the plan-level identity gates, no RFS)
  report   (repo venv, CPU)            RFS of the base, the selector (gate B / A / 0) and every candidate; paired CIs, strata, context readings
                                       -> results/turn_selector_wod/, figs/turn_selector_wod/

Candidates: the F19 members of turn_ceiling.candidates() (lateral offset, curvature gain, speed scale) are applied to rear-axle poses at 0.5 .. 5 s (10 poses, the
selector's own 8 poses at 0.5 .. 4 s are its input, unchanged). The change of a candidate (Q - P at 0.5 .. 5 s, zero at 0, linear in time) is added to the plan's
20 WOD waypoints (0.25 .. 5 s), so an unpicked frame is the plan itself. Variant H4: the candidate from the 8 poses of 0.5 .. 4 s, its change held after 4 s
(the closed-loop apply_delta of turn_selhug.py).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, re, time  # noqa: E401,E402

import numpy as np  # noqa: E402

D = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/op_parity/turn_selwod"
ONNX = D / "runs/op_parity/hugsim/onnx"
BIAS = D / "runs/op_parity/wod"
TAPS = ["view_39", "select_4", "mean"]
ACTION_T = (0.275, 0.525)
T33 = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
T_FUT = 0.25 * np.arange(1, 21)
T8 = 0.5 * np.arange(1, 9)
T10 = 0.5 * np.arange(1, 11)
NPT = 23                                                  # plan points with t <= 5 s (the RFS horizon): T33[22] = 4.73 s
TOL_M = 0.03                                              # floor of the plan-point tolerance (two fp16 ulps at 32 m); grows with the coordinate (see tol())
N_RATER = 479


def feat_file(tag, tag_out=""):
    return OUT / "feat" / f"{tag}{tag_out}.npz"


def sel_file(tag, tag_out=""):
    return OUT / "sel" / f"{tag}{tag_out}.npz"


def rater_names():
    from jevdrive import wod_zeroshot as Z
    S = Z.load_sets()
    return sorted(str(n) for n in S["rater"]["name"])


# ---------------------------------------------------------------- extract
def cmd_extract(a):
    from concurrent.futures import ProcessPoolExecutor
    import wod_zeroshot_openpilot as H
    from jevdrive import wod_zeroshot as Z
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    from jevdrive.run import Run
    with Run("op_parity", f"turn_selwod/extract-{a.tag}{a.suffix}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        out = feat_file(a.tag, a.suffix)
        names = rater_names()[: a.limit or None]
        sets = Z.load_sets()
        spans, _ = Z.load_spans()
        op_calib = json.loads((Z.root() / "op_calib.json").read_text())
        z = np.load(BIAS / f"bias-{a.tag}.npz")
        bias = dict(zip(z["names"].astype(str).tolist(), z["bias"]))
        ego = dict(zip(z["names"].astype(str).tolist(), z["ego"]))
        assert all(n in bias for n in names)
        m = OPModel(str(ONNX / f"pp-{a.tag}.onnx"), "trt", cache=OUT / "trt" / a.tag, taps=TAPS)
        n = len(names)
        R = dict(names=np.array(names), wod=np.zeros((n, 20, 2), np.float32), plan_pos=np.zeros((n, 33, 3), np.float32), plan_yaw=np.zeros((n, 33), np.float32),
                 re=np.zeros((n, 2, 33, 2), np.float32), v3=np.zeros((n, 32, 512), np.float16), h4=np.zeros((n, 512), np.float16), hm=np.zeros((n, 512), np.float16),
                 dev_xy=np.zeros((n, 2), np.float64), ego=np.stack([ego[k] for k in names]).astype(np.float32))
        rs = m.slices["road_edges"]
        half = (rs.stop - rs.start) // 2
        assert half == 132, (rs, half)
        shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
        t0 = time.time()
        with ProcessPoolExecutor(a.workers, initializer=H._init, initargs=(spans, op_calib, str(shard_dir), None)) as ex:
            for i, (name, hist, frames) in enumerate(ex.map(H.model_frames, names, chunksize=1)):
                seq = name.rsplit("-", 1)[0]
                dev = np.array(op_calib[seq]["1"]["extrinsic"]).reshape(4, 4)[:2, 3]
                m.reset()                                              # run_one of the harness, with the raw last output kept
                m.extra["intent_bias"] = bias[name][None].astype(np.float16)
                steps = [s for s in frames for _ in range(2)]          # 20 Hz: every WOD frame twice (OPModel(context_rate=False))
                for s in steps[:-1]:
                    m.step(s, action_t=ACTION_T)
                raw = m.step(steps[-1], action_t=ACTION_T)
                d = decode(raw, m.slices, 10.0, ACTION_T)
                R["wod"][i] = Z.openpilot_to_wod(d["plan_pos"], d["plan_yaw"], T_IDXS, dev)
                R["plan_pos"][i], R["plan_yaw"][i] = d["plan_pos"], d["plan_yaw"]
                R["re"][i] = raw[rs][:half].reshape(2, 33, 2)
                R["v3"][i] = m.tap_values["view_39"].reshape(32, 512)
                R["h4"][i], R["hm"][i] = m.tap_values["select_4"], m.tap_values["mean"]
                R["dev_xy"][i] = dev
                if (i + 1) % 50 == 0 or i + 1 == n:
                    el = time.time() - t0
                    run.info("[%d/%d] %.2f s/target, ETA %.0f min", i + 1, n, el / (i + 1), (n - i - 1) * el / (i + 1) / 60)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".tmp.npz"), **R)
        out.with_suffix(".tmp.npz").rename(out)
        run.summary.update(n=n, wall_s=time.time() - t0)


# ---------------------------------------------------------------- geometry
def to_rear(pos, yaw, d, t_out):
    """jevdrive.op_interp.to_rear (lever, linear): camera-frame plan (33, 3) / (33,) -> rear-axle (len(t_out), 3), +y left."""
    p = np.stack([pos[:, 0], -pos[:, 1]], -1).astype(np.float64)
    psi = -np.asarray(yaw, np.float64)
    d = np.asarray(d, np.float64)
    Rd = np.stack([np.cos(psi) * d[0] - np.sin(psi) * d[1], np.sin(psi) * d[0] + np.cos(psi) * d[1]], -1)
    rear = np.concatenate([d + p - Rd, psi[:, None]], 1)
    return np.stack([np.interp(t_out, T33, rear[:, k]) for k in range(3)], -1)


def speed_n(Q, v, kappa_max):
    """turn_ceiling.speed for any number of poses (the original hardcodes 8 + origin)."""
    out = Q.copy()
    T = Q.shape[1]
    for n, q in enumerate(Q):
        seg = np.hypot(*np.diff(q[:, :2], axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        if s[-1] < 1e-3:
            continue
        tgt, sm = v * s[1:], s + 1e-9 * np.arange(T)
        for j in range(3):
            out[n, 1:, j] = np.interp(tgt, sm, q[:, j])
        over = tgt > s[-1]
        if over.any():
            u, th0 = tgt[over] - s[-1], q[-1, 2]
            kap = float(np.clip((q[-1, 2] - q[-2, 2]) / max(seg[-1], 0.5), -kappa_max, kappa_max))
            th = th0 + kap * u
            if abs(kap) < 1e-6:
                x, y = q[-1, 0] + u * np.cos(th0), q[-1, 1] + u * np.sin(th0)
            else:
                x, y = q[-1, 0] + (np.sin(th) - np.sin(th0)) / kap, q[-1, 1] - (np.cos(th) - np.cos(th0)) / kap
            out[n, 1:][over] = np.stack([x, y, th], -1)
    return out


def transform_n(P, o=0.0, k=1.0, v=1.0):
    """turn_ceiling.transform for (N, T, 3) poses at 0.5 s spacing, any T (equal to the original for T = 8: G-gen)."""
    import turn_ceiling as TC
    if (o, k, v) == TC.ID:
        return P
    Q = np.concatenate([np.zeros((len(P), 1, 3)), np.asarray(P, np.float64)], 1)
    Q[..., 2] = np.unwrap(Q[..., 2], axis=1)
    if k != 1.0:
        Q = TC.curv(Q, k)
    if o != 0.0:
        Q = TC.offset(Q, o)
    if v != 1.0:
        Q = speed_n(Q, v, TC.KAPPA_MAX)
    Q[..., 2] = np.angle(np.exp(1j * Q[..., 2]))
    return Q[:, 1:].astype(P.dtype)


def apply_candidates(wod, P, t_p, cands):
    """wod (n, 20, 2) plan waypoints, P (n, T, 3) rear-axle poses at t_p (0.5 s spacing) -> (n, K, 20, 2): plan + interpolated change of each candidate,
    zero at t = 0 and held after the last pose time (np.interp)."""
    out = np.zeros((len(P), len(cands), 20, 2))
    t = np.r_[0.0, t_p]
    for c, (_, o, k, v) in enumerate(cands):
        Q = transform_n(P, o, k, v)
        dd = np.concatenate([np.zeros((len(P), 1, 2)), Q[..., :2] - P[..., :2]], 1)
        for i in range(len(P)):
            out[i, c] = wod[i] + np.stack([np.interp(T_FUT, t, dd[i, :, j]) for j in range(2)], -1)
    return out


def tol(ref, new):
    """Per-coordinate tolerance of a plan-point comparison: two fp16 ulps of the larger coordinate magnitude, at least TOL_M."""
    m = np.maximum(np.abs(ref), np.abs(new))
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(m, 1e-6))) - 10)
    return np.maximum(TOL_M, 2 * ulp)


def plan_gate(ref_pos, pos):
    """Share of rows with any of the first NPT plan points (x, y) outside tol(): (n,) bool over tolerance, (n,) max |d|."""
    r, p = np.asarray(ref_pos, np.float32)[:, :NPT, :2], np.asarray(pos, np.float32)[:, :NPT, :2]
    d = np.abs(r - p)
    return (d > tol(r, p)).any((1, 2)), d.max((1, 2))


# ---------------------------------------------------------------- select
def cmd_select(a):
    import turn_ceiling as TC
    import turn_dewater as TD
    import turn_selbench as TB
    import turn_selinput as TS
    import turn_selnt as TN
    import sc_analyze as SC
    from jevdrive import wod_zeroshot as Z
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", f"turn_selwod/select-{a.tag}{a.suffix}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        f = np.load(feat_file(a.tag, a.suffix))
        names = f["names"].astype(str)
        n = len(names)
        cal_seed = int(re.fullmatch(r".*-s(\d+)", a.tag).group(1))
        cal = json.loads(TS.CAL.read_text())[f"SH30-F-s{cal_seed}"]                # the edge-head calibration of the SH30 seed (WLG: same seed, stated)
        cands = TC.candidates()[:TN.NCAND]
        P8 = np.stack([to_rear(f["plan_pos"][i], f["plan_yaw"][i], f["dev_xy"][i], T8) for i in range(n)])
        P10 = np.stack([to_rear(f["plan_pos"][i], f["plan_yaw"][i], f["dev_xy"][i], T10) for i in range(n)])
        P8f = P8.astype(np.float32)
        ex, ey = SC.edges_ego(f["re"].astype(np.float32), f["dev_xy"])
        C = TB.curb_margins(P8f, ex, ey, (cal["s"], cal["b"]))
        e = np.concatenate([f["ego"].astype(np.float64), TD.plan_desc(P8f.astype(np.float64))], -1)
        h = np.concatenate([f["h4"], f["hm"]], 1).astype(np.float32)
        pred, each = TB.infer(TB.load_bundle(), e, h, C.reshape(n, -1), f["v3"])
        dyaw = TB.model_yaw_deg(P8f)
        pick_free = TD.picks_of(pred)
        # G-gen: transform_n equals the original on the selector's 8 poses
        gen = max(float(np.abs(transform_n(P8, *c[1:]) - TC.transform(P8, *c[1:])).max()) for c in cands)
        W10 = apply_candidates(f["wod"].astype(np.float64), P10, T10, cands)        # primary: candidates over 0.5 .. 5 s
        W4 = apply_candidates(f["wod"].astype(np.float64), P8, T8, cands)           # H4: 0.5 .. 4 s, change held after 4 s
        assert np.array_equal(W10[:, 0], f["wod"].astype(np.float64)) and np.array_equal(W4[:, 0], f["wod"].astype(np.float64)), "identity candidate must be the plan"
        # the WOD waypoints of the plan are the lever-arm map of the plan at T_FUT; the dense plan at 0.5 s spacing must agree with them (map consistency)
        chk = np.abs(np.stack([to_rear(f["plan_pos"][i], f["plan_yaw"][i], f["dev_xy"][i], T_FUT)[:, :2] for i in range(n)]) - f["wod"]).max()
        info = dict(n=n, g_gen_max_abs=gen, g_map_max_abs_m=float(chk), gate_B_share=float((dyaw >= TB.TURN_DEG).mean()), moved_free_share=float((pick_free != 0).mean()))
        # plan-level identity gate against the archived run of the same weights (+ wrong-seed control)
        arch = np.stack([np.load(Z.root("preds", f"op_cinque_{a.tag}") / f"{k}.npz")["plan_pos"] for k in names])
        over, mx = plan_gate(arch, f["plan_pos"])
        info.update(g_feat_rows_over=float(over.mean()), g_feat_max_m=float(mx.max()), g_feat_median_m=float(np.median(mx)))
        m_ = re.fullmatch(r"(.*-s)(\d+)", a.tag)
        wrong = f"{m_.group(1)}{1 - int(m_.group(2))}"
        if (Z.root("preds", f"op_cinque_{wrong}")).exists():
            aw = np.stack([np.load(Z.root("preds", f"op_cinque_{wrong}") / f"{k}.npz")["plan_pos"] for k in names])
            ow, mw = plan_gate(aw, f["plan_pos"])
            info.update(g_feat_control_rows_over=float(ow.mean()), g_feat_control_median_m=float(np.median(mw)))
        # archived waypoints vs the tapped run's (what the RFS identity gate compares)
        arch_w = np.stack([np.load(Z.root("preds", f"op_cinque_{a.tag}") / f"{k}.npz")["wod"] for k in names])
        info.update(g_wod_max_m=float(np.abs(arch_w[..., :2] - f["wod"]).max()), g_wod_median_m=float(np.median(np.abs(arch_w[..., :2] - f["wod"]).max((1, 2)))))
        sel_file(a.tag, a.suffix).parent.mkdir(parents=True, exist_ok=True)
        np.savez(sel_file(a.tag, a.suffix), names=names, pred=pred.astype(np.float32), pred_each=each.astype(np.float32), pick_free=pick_free, dyaw=dyaw, C=C.astype(np.float32),
                 W10=W10.astype(np.float32), W4=W4.astype(np.float32), wod=f["wod"], arch_wod=arch_w[..., :2].astype(np.float32), info=np.array(json.dumps(info)))
        run.summary.update(info)
        run.info("select %s", info)
        assert gen == 0.0, f"G-gen: transform_n differs from turn_ceiling.transform by {gen}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("extract", "select"):
        p = sp.add_parser(nm)
        p.add_argument("--tag", required=True, help="SH30-F-s0 | SH30-F-s1 | WLG-full-s0 | ...")
        p.add_argument("--suffix", default="", help="output suffix (smoke)")
        if nm == "extract":
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--workers", type=int, default=12)
    p = sp.add_parser("report")
    p.add_argument("--tags", nargs="+", default=["SH30-F-s0", "SH30-F-s1"])
    p.add_argument("--name", default="turn_selector_wod")
    a = ap.parse_args()
    if a.cmd == "report":
        return __import__("turn_selwod_report").report(a)
    {"extract": cmd_extract, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
