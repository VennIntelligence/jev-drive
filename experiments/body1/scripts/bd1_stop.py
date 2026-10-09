#!/usr/bin/env python3
"""BODY1 arm 4.1, everything that is fixed before a closed-loop score is read (plans/2026-10-10-body1-prereg.md, Amendment 2).

  hold     (CPU) the stop-point rule on body1-hold-logs: own-plan decisions flagged at the 2 % threshold, candidate constructions of the
           contact arc length x m in {1, 2} m x deceleration cap -> $DATA_DIR/runs/body1/stop/hold/{rule.csv, bias.csv, speed.csv}.
           Open loop, one decision at a time: "avoided" = the distance the ego needs to stand still under the rule's deceleration is not
           beyond the true first-contact arc length. Reads s0/report/own.parquet and s0/pred/step-s{0,1}.npz (hold logs only).
  replay   (GPU) plumbing on COL1's serialized driver messages (the G2 replay harness; navtest logs: nothing is chosen here).
           JEV_STOP unset: the driver's served plans against SWV1's stored replay, bit for bit (the switch-off identity gate).
           JEV_STOP=<m>: the hook's logits against the gate's stored predictions of the same decisions (input standard, slot masks), the
           plan before the stop against the stored replay (bit for bit), the stop trajectories (on the path, never ahead of the plan, never
           past D) and the hook's latency. -> $DATA_DIR/runs/body1/stop/replay/<off|on>.json
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

GROUPS = ("P2H10-F-s0", "P2H10-F-s1", "ctrl")
STEP_C, MS, CAPS = (0.0, 0.25, 0.5), (1.0, 2.0), (4.0, 6.0)


def out_dir(name):
    d = B.root() / "stop" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- hold logs: how the stop point is formed
def cmd_hold(a):
    import pandas as pd
    import bd1_gate as G
    import contact_head as C
    import serve_body as SB
    from jevdrive.run import Run
    with Run("body1", "stop/hold", config=vars(a)) as run:
        R, sp = G.hold_rows()
        for s in sp:
            run.use_split(s)
        D = pd.read_parquet(C.sroot() / "report" / "own.parquet")
        P = [np.load(C.sroot() / "pred" / f"step-s{i}.npz") for i in (0, 1)]
        for z in P:
            assert (z["gi"] == R["gi"]).all()
        out = np.mean([z["hold_out"][:, :2] for z in P], 0).reshape(-1, len(C.OUT))
        step = np.mean([z["hold_step"][:, :2].astype(np.float32) for z in P], 0).reshape(-1, C.NT, 2)[..., 0]
        s = out[:, 0]
        assert np.abs(s - D.s_agent.to_numpy()).max() < 1e-4
        q = R["q"][:, :2].reshape(-1, 8, 3)
        xy = np.concatenate([np.zeros((len(q), 1, 2)), q[..., :2]], 1)
        node = np.concatenate([np.zeros((len(q), 1)), np.cumsum(np.hypot(np.diff(xy[..., 0], axis=1), np.diff(xy[..., 1], axis=1)), 1)], 1)
        ok, y, v, st, cls = D.a_ok.to_numpy(), D.a.to_numpy(), D.speed.to_numpy(), D.a_s.to_numpy(), D.cls.to_numpy()
        clean = ok & ~y & ~D.rear_only.to_numpy()
        fl = s >= SB.THR
        tp, fp = ok & y & fl, clean & fl
        sreg = np.maximum(out[:, 2] * C.S_SCALE, 0.0)

        def sstep(c):
            low = step < c
            k = np.where(low.any(1), low.argmax(1), -1)
            return np.where(k < 0, np.inf, node[np.arange(len(node)), np.clip(k - 1, 0, 8)])
        cands = {"regression": sreg} | {f"min(regression, first interval with clearance < {c:g} m)": np.minimum(sreg, sstep(c)) for c in STEP_C}
        rows = []
        for cap in CAPS:
            for nm, cs in cands.items():
                for m in MS:
                    d = np.maximum(cs - m, 0.0)
                    Dd = np.maximum(d, v * v / (2 * cap))                                   # where the ego stands still (serve_body.retime)
                    acc = np.clip(v * v / (2 * np.maximum(Dd, 1e-6)), SB.A_MIN, cap)
                    brakes_now = v * v / (2 * np.maximum(Dd, 1e-6)) >= SB.A_MIN             # else the plan is followed until the envelope
                    avoid = Dd <= st
                    r = dict(cap=cap, contact_arc=nm, m=m, n_tp=int(tp.sum()), tp_stop_point_before_contact=float((d <= st)[tp].mean()), tp_avoided=float(avoid[tp].mean()),
                             tp_avoided_n=int(avoid[tp].sum()), tp_at_cap=float((acc >= cap - 1e-9)[tp].mean()), tp_slack_median_m=float(np.median((st - d)[tp])),
                             n_fp=int(fp.sum()), fp_stop_point_median_m=float(np.median(d[fp])), fp_stop_point_under_half_m=float((d[fp] <= 0.5).mean()),
                             fp_brakes_now=float(brakes_now[fp].mean()), fp_speed_removed_per_decision_mean=float((np.where(brakes_now, acc, 0.0) * 0.5)[fp].mean()),
                             fp_at_cap=float((acc >= cap - 1e-9)[fp].mean()))
                    r |= {f"tp_avoided_class{c}": f"{int(avoid[tp & (cls == c)].sum())}/{int((tp & (cls == c)).sum())}" for c in (1, 2, 3, 0)}
                    rows.append(r)
        rule = pd.DataFrame(rows)
        o = out_dir("hold")
        rule.to_csv(o / "rule.csv", index=False)
        e = sreg - st
        bias = pd.DataFrame([dict(true_arc=f"{lo}-{hi} m", n=int(m.sum()), signed_error_median=float(np.median(e[m])), q25=float(np.quantile(e[m], 0.25)),
                                  q75=float(np.quantile(e[m], 0.75)), q90=float(np.quantile(e[m], 0.9)), abs_error_median=float(np.median(np.abs(e[m]))))
                             for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 20), (20, 99), (0, 99)) for m in [tp & (st >= lo) & (st < hi)]])
        bias.to_csv(o / "bias.csv", index=False)
        speed = pd.DataFrame([dict(speed=f"{lo}-{hi} m/s", clean=int((clean & m).sum()), flag_rate_clean=float(fl[clean & m].mean()), positives=int((ok & y & m).sum()),
                                   recall=float(fl[ok & y & m].mean())) for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 10), (10, 99)) for m in [(v >= lo) & (v < hi)]])
        speed.to_csv(o / "speed.csv", index=False)
        run.info("flags: %d of %d clean (%.4f), %d of %d positives\n%s\n%s\n%s", fp.sum(), clean.sum(), fp.sum() / clean.sum(), tp.sum(), (ok & y).sum(),
                 rule.drop(columns=[c for c in rule if c.startswith("tp_avoided_class")]).round(3).to_string(index=False), bias.round(2).to_string(index=False), speed.round(4).to_string(index=False))
        run.summary.update(n_tp=int(tp.sum()), n_fp=int(fp.sum()), out=str(o))


# ---------------------------------------------------------------- replay: identity with the switch off, plumbing with it on
class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def cmd_replay(a):
    alp = B.REPO / "experiments" / "alpasim"
    _sys.path[:0] = [str(alp / "scripts"), str(alp / "lib")]
    import torch
    import sh30_driver as D
    from jevdrive.run import Run
    LD, stored_dir = B.root().parent / "alpasim/col1/lead", B.root().parent / "alpasim/swv1/replay"
    on = D.BD is not None
    with Run("body1", "stop/replay-" + ("on" if on else "off"), config=vars(a) | dict(jev_stop=os.environ.get("JEV_STOP", ""))) as run:
        spec = json.loads((LD / "spec.json").read_text())
        pb, ctx, res = D.egodriver_pb2, Ctx(), {}
        caught = []
        if on:
            D.BD.load("cuda")
            apply0 = D.BD.apply

            def apply(lock, o, v0):
                info = apply0(lock, o, v0)
                caught.append((info, np.asarray(o.get("poses_plan", o["poses"])).copy(), np.asarray(o["poses"]).copy(), float(v0)))
                return info
            D.BD.apply = apply
            pred = [np.load(B.root() / "s0" / "pred" / f"step-s{i}.npz") for i in (0, 1)]
        for g in a.groups:
            core = D.C.Core(spec[g]["tag"], "cuda", "backwarp")
            drv = D.Driver(core, _pl.Path(tempfile.mkdtemp()), 0, False)
            last = {}
            orig = core.plan

            def plan(*x, _o=orig, **k):
                last["o"] = _o(*x, **k)
                return last["o"]
            core.plan = plan
            stored = pickle.load(open(stored_dir / f"{g}.pkl", "rb"))
            scenes = spec[g]["scenes"][:a.limit] if a.limit else spec[g]["scenes"]
            d_pose, recs = [], []
            for scene in run.tqdm(scenes, desc=g):
                uuid, k = None, 0
                for kind, raw in pickle.load(open(LD / "msgs" / g / f"{scene}.pkl", "rb")):
                    if kind == "driver_session_request":
                        req = pb.DriveSessionRequest.FromString(raw)
                        uuid = req.session_uuid
                        drv.start_session(req, ctx)
                    elif kind == "driver_camera_image":
                        drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
                    elif kind == "driver_ego_trajectory":
                        drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
                    elif kind == "route_request":
                        drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
                    elif kind == "driver_request":
                        n0 = len(caught)
                        drv.drive(pb.DriveRequest.FromString(raw), ctx)
                        o, s = last["o"], stored[scene][k]
                        plan_ = caught[-1][1] if on else o["poses"]
                        assert not on or len(caught) == n0 + 1
                        d_pose.append(float(np.abs(plan_ - s["poses"]).max()))
                        recs.append((scene, k))
                        k += 1
                drv.sessions.pop(uuid, None)
            d_pose = np.array(d_pose)
            r = dict(n=len(d_pose), max_abs_plan_diff_m=float(d_pose.max()), n_nonzero=int((d_pose > 0).sum()))
            if on:
                C_, z = caught[-len(recs):], np.load(B.root() / "g2" / "dump" / f"{g}.npz")
                key = {(str(sc), int(kk)): i for i, (sc, kk) in enumerate(zip(z["scene"], z["k"]))}
                idx = np.array([key[x] for x in recs])
                ref = np.stack([p[f"g2_{g}"][idx, 0] for p in pred], 1)                      # (n, 2) stored per-seed agent logits
                got = np.array([c[0]["z"] for c in C_])
                flag, ref_flag = np.array([c[0]["flag"] for c in C_]), ref.mean(1) >= D.BD.THR
                ms = np.array([c[0]["ms"] for c in C_])
                bad, cuts, acc, why = 0, [], [], np.zeros(4, int)                               # why: off the path, ahead of the plan, past D, backwards
                for info, pl, sv, v0 in C_:
                    if not info["flag"]:
                        bad += int(not np.array_equal(pl, sv))
                        continue
                    arc = lambda p: np.cumsum(np.hypot(*np.diff(np.concatenate([np.zeros((1, 2)), p[:, :2]]), axis=0).T))  # noqa: E731
                    sa, pa = arc(sv), arc(pl)
                    P = np.concatenate([np.zeros((1, 2)), pl[:, :2]])
                    seg = P[1:] - P[:-1]
                    t = np.clip(np.einsum("nsc,sc->ns", sv[:, None, :2] - P[None, :-1], seg) / np.maximum((seg ** 2).sum(1), 1e-12), 0, 1)
                    off = np.linalg.norm(sv[:, None, :2] - (P[None, :-1] + t[..., None] * seg[None]), axis=-1).min(1).max()
                    fails = [off > 1e-6, bool((sa > pa + 1e-6).any()), sa[-1] > info["D"] + 1e-3, bool((np.diff(sa) < -1e-9).any())]
                    why += np.array(fails, int)
                    bad += int(any(fails))
                    cuts.append(info["cut"]), acc.append(info["a"])
                r.update(logit_max_abs_diff=float(np.abs(got - ref).max()), logit_mean_abs_diff=float(np.abs(got - ref).mean()), flags=int(flag.sum()),
                         flags_stored=int(ref_flag.sum()), flag_disagree=int((flag != ref_flag).sum()), flag_rate=float(flag.mean()), trajectory_checks_failed=bad, trajectory_checks_why=why.tolist(),
                         hook_ms_p50=float(np.median(ms)), hook_ms_p90=float(np.quantile(ms, 0.9)), hook_ms_max=float(ms.max()),
                         flagged_cut_median_m=float(np.median(cuts)) if cuts else None, flagged_a_median=float(np.median(acc)) if acc else None,
                         slots_min=int(min(int(np.asarray(z["valid"][i]).sum()) for i in idx[:50])))
            run.info("%s: %s", g, r)
            res[g] = r
            del core, drv
            torch.cuda.empty_cache()
        f = out_dir("replay") / ("on.json" if on else "off.json")
        f.write_text(json.dumps(res, indent=1))
        run.summary.update(res)
        if any(v["max_abs_plan_diff_m"] > 0 or v.get("trajectory_checks_failed", 0) for v in res.values()):
            raise SystemExit(f"replay gate FAILED: {res}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hold")
    p = sub.add_parser("replay")
    p.add_argument("--groups", nargs="*", default=list(GROUPS)), p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"hold": cmd_hold, "replay": cmd_replay}[a.cmd](a)


if __name__ == "__main__":
    main()
