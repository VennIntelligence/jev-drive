#!/usr/bin/env python
"""BODY1 stage 2, P0-E: correctness of the two parity input-port fixes of lib/op_arb_agent.py on recorded values, before any CARLA job
(experiments/body1/plans/2026-10-10-stage2-prereg.md section 4, items i and ii). CPU; envs/simlingo (carla.Map for the junction ids).

  $DATA_DIR/envs/simlingo/bin/python experiments/body1/scripts/s2p0_port_check.py [--clips 24] [--base <git sha of the agent before the switches>]

The recorded ego stream of N b2dc clips (ego.npz, route.npz of the cache's clip list; chosen by sha256(route): 2 per town, then junction-turn
clips up to --turns) is replayed tick by tick through the real `OpArbAgent.parity_ego` (an instance without setup; only the two simulator
readers are replaced by the recorded values: `_hero_state` -> ego.npz vel / acc / yaw, `_junction_ids` -> b2dc_labels.junction_ids on the
offline map; the pose buffer holds the recorded rear-axle poses, the desire is the shipped rule on the recorded RoadOptions).

  0  bit identity: the agent file at --base against the working file with every switch off, all ticks of all clips, exact equality
  i  both switches on, at the cache's rows of these clips: the 17 non-command dims within 1e-3 of b2d_v2 tab.ego; the command dims equal to
     b2d_v2L20 tab.ego on >= 0.99 of the rows, every disagreeing row listed with its route progress and distance to the turn
  ii convention against navtrain (navtrain_full.s0of12 tab) on moving rows (speed > 3 m/s): OLS slope of ay on vx x yaw rate (yaw rate = the
     heading change over the last 0.5 s of the fed pose history) in [0.8, 1.2] on both; ax quantiles; vy against the yaw rate reported
Also reported (no line): how far the shipped ports were from the labels on the same rows (dims of the as-shipped vector against the cache).
Output: experiments/body1/results/s2p0/port_check.{json,csv}.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
REPO = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "scripts"), str(REPO / "experiments/b2d_collect/lib")]
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402
from collections import deque  # noqa: E402
from types import SimpleNamespace as NS  # noqa: E402

import numpy as np  # noqa: E402

OUT = REPO / "experiments/body1/results/s2p0"
DIMS = ["present", "cmd_L", "cmd_S", "cmd_R", "vx/10", "vy/10", "ax/3", "ay/3"] + [f"{k}{i}" for i in range(4) for k in ("x/10 t", "y/10 t", "yaw t")]
NONCMD = [0] + list(range(4, 20))


def stub_harness():
    """leaderboard / srunner are only in the route process env: the agent module needs their names to import, nothing of them is called."""
    for name, attrs in (("leaderboard", {}), ("leaderboard.autoagents", {}), ("srunner", {}), ("srunner.scenariomanager", {}),
                        ("leaderboard.autoagents.autonomous_agent", {"AutonomousAgent": type("AutonomousAgent", (), {}), "Track": NS(SENSORS="SENSORS")}),
                        ("srunner.scenariomanager.timer", {"GameTime": NS()})):
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            m.__dict__.update(attrs)
            _sys.modules[name] = m


def load_agent(src, name):
    m = types.ModuleType(name)
    m.__file__ = str(REPO / "lib/op_arb_agent.py")
    exec(compile(src, name, "exec"), m.__dict__)
    return m


def pick(clips, n, turns):
    h = sorted((c for c in clips if c["n_rows"] > 0), key=lambda c: hashlib.sha256(c["route"].encode()).hexdigest())
    out, per = [], {}
    for c in h:
        if per.get(c["town"], 0) < 2 and len(out) < n:
            per[c["town"]] = per.get(c["town"], 0) + 1
            out.append(c)
    for c in h:
        if sum(bool(x["turn"]) for x in out) >= turns:
            break
        if c["turn"] and c not in out:
            out.append(c)
    return out


def replay(M, clip, port, xodr, L, Z):
    """Every tick of one clip through M.OpArbAgent.parity_ego -> (n, 20) fed vectors, the label route progress per tick (port cmd only)."""
    ego, meta = dict(np.load(clip / "ego.npz")), json.loads((clip / "meta.json").read_text())
    rz = np.load(clip / "route.npz")
    plan = [(NS(location=NS(x=float(p[0]), y=float(p[1]), z=float(p[2]))), int(o)) for p, o in zip(rz["xyzyaw"], rz["option"])]
    yaw = np.radians(ego["rot"][:, 1].astype(float))
    xy = ego["loc"][:, :2] + L.REAR_AXLE_X * np.stack([np.cos(yaw), np.sin(yaw)], -1)
    ag = object.__new__(M.OpArbAgent)
    ag.poses, ag.port, ag.port_route, ag.port_s, ag._dense_plan = deque(maxlen=64), dict(port), None, None, plan
    state = {}
    ag._hero_state = lambda: (ego["vel"][state["i"]].tolist(), ego["acc"][state["i"]].tolist(), float(ego["rot"][state["i"], 1]))
    ag._junction_ids = lambda xyz: L.junction_ids(meta["town"], str(xodr(meta["town"])), xyz)
    route = Z.Route(plan)                                       # the shipped desire rule on the recorded RoadOptions
    n = len(ego["t"])
    out, prog, des = np.zeros((n, 20), np.float32), np.full(n, np.nan), np.zeros(n, int)
    for i in range(n):
        state["i"] = i
        ag.poses.append((float(ego["t"][i]), xy[i], float(yaw[i])))
        route.progress(xy[i])
        des[i] = route.desire()
        out[i] = ag.parity_ego(float(ego["speed"][i]), des[i])
        if ag.port_s is not None:
            prog[i] = ag.port_s
    return out, prog, des, ag.port_route


def slope(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(k), float(b), float(np.corrcoef(x, y)[0, 1])


def convention(ego, speed):
    """Moving rows of an (N, 20) fed-vector table -> slope of ay on vx x yaw rate, vy on yaw rate, ax / vy quantiles."""
    m = speed > 3.0
    e = ego[m].astype(float)
    vx, vy, ax, ay = e[:, 4] * 10, e[:, 5] * 10, e[:, 6] * 3, e[:, 7] * 3
    w = (e[:, 19] - e[:, 16]) / 0.5
    k, b, r = slope(vx * w, ay)
    kv, bv, rv = slope(w, vy)
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 50, 95])]  # noqa: E731
    return dict(rows=int(m.sum()), ay_slope=round(k, 3), ay_intercept=round(b, 3), ay_r=round(r, 3), vy_on_yawrate_slope_m=round(kv, 3),
                vy_r=round(rv, 3), ax_p5_p50_p95=q(ax), ay_p5_p50_p95=q(ay), vy_p5_p50_p95=q(vy), vx_p5_p50_p95=q(vx),
                yawrate_abs_p95=round(float(np.percentile(np.abs(w), 95)), 3))


def main(a):
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("body1", "s2p0-port-check", config=vars(a)) as run:
        stub_harness()
        import b2d_zeroshot_agent as Z
        import b2dc_labels as L
        run.use_split(splits.load("b2d/b2dc-train@v2"))
        D = data_dir()
        carla_root = D / "third_party/carla/CARLA_0.9.15"

        def xodr(town):
            p = carla_root / f"CarlaUE4/Content/Carla/Maps/{town}/OpenDrive/{town}.xodr"
            return p if p.exists() else carla_root / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
        new = load_agent((REPO / "lib/op_arb_agent.py").read_text(), "op_arb_agent_new")
        old = load_agent(subprocess.check_output(["git", "-C", str(REPO), "show", f"{a.base}:lib/op_arb_agent.py"]).decode(), "op_arb_agent_old")
        C = D / "runs/op_parity/cache"
        clips = pick(json.loads((C / "b2d_v2/plan.json").read_text())["clips"], a.clips, a.turns)
        tab, tab20 = np.load(C / "b2d_v2/tab.npz"), np.load(C / "b2d_v2L20/tab.npz")
        names = tab["names"]
        assert np.array_equal(names, tab20["names"])
        E30, E20, log = tab["ego"], tab20["ego"], tab["log"]
        rows, bad_cmd, fed_all, ship_all, lab_all, spd_all = [], [], [], [], [], []
        ident_ticks, ident_ok = 0, True
        for c in run.tqdm(clips, desc="clips"):
            clip = _pl.Path(c["clip"])
            e_old = replay(old, clip, {}, xodr, L, Z)[0]
            e_off, _, des, _ = replay(new, clip, {}, xodr, L, Z)
            e_fix, prog, _, route = replay(new, clip, {"ego": True, "cmd": True}, xodr, L, Z)
            same = bool(np.array_equal(e_old, e_off))
            ident_ok &= same
            ident_ticks += len(e_old)
            idx = np.flatnonzero(log == c["route"])
            t0 = np.array([int(s.split("_")[1]) for s in names[idx]])
            lab, lab20, fed, ship = E30[idx], E20[idx], e_fix[t0], e_off[t0]
            d = np.abs(fed - lab)
            agree = (fed[:, 1:4] == lab20[:, 1:4]).all(1)
            for j in np.flatnonzero(~agree):
                o, dist = route.next_turn(prog[t0[j]])
                bad_cmd.append(dict(route=c["route"], tick=int(t0[j]), fed=fed[j, 1:4].tolist(), cache=lab20[j, 1:4].tolist(), progress=round(float(prog[t0[j]]), 2),
                                    turn_next=int(o), turn_dist=round(float(dist), 2)))
            sd = np.abs(ship - lab)
            rows.append(dict(route=c["route"], town=c["town"], type=c["type"], turn=c["turn"], ticks=len(e_old), rows=len(idx), identical_off=same,
                             noncmd_max=float(d[:, NONCMD].max()), vel_max=float(d[:, 4:6].max()), acc_max=float(d[:, 6:8].max()), pose_max=float(d[:, 8:].max()),
                             cmd_agree_L20=float(agree.mean()), cmd_agree_L30=float((fed[:, 1:4] == lab[:, 1:4]).all(1).mean()),
                             turn_rows_L20=float((lab20[:, 2] == 0).mean()),
                             shipped_cmd_agree_L20=float((ship[:, 1:4] == lab20[:, 1:4]).all(1).mean()),
                             shipped_vx_absmax=float(sd[:, 4].max() * 10), shipped_vy_absp95=float(np.percentile(sd[:, 5], 95) * 10),
                             shipped_ax_absp95=float(np.percentile(sd[:, 6], 95) * 3), shipped_ay_absp95=float(np.percentile(sd[:, 7], 95) * 3)))
            fed_all.append(fed), ship_all.append(ship), lab_all.append(lab20), spd_all.append(tab["speed"][idx])
            run.info("%s %s %-28s rows %5d  identical(off) %s  non-cmd max %.2e  cmd agree L20 %.4f  (shipped cmd %.4f)", c["route"], c["town"], c["type"][:28],
                     len(idx), same, rows[-1]["noncmd_max"], rows[-1]["cmd_agree_L20"], rows[-1]["shipped_cmd_agree_L20"])
        fed, ship, lab, spd = (np.concatenate(x) for x in (fed_all, ship_all, lab_all, spd_all))
        d = np.abs(fed - lab)
        per_dim = {DIMS[k]: float(d[:, k].max()) for k in range(20)}
        agree = float((fed[:, 1:4] == lab[:, 1:4]).all(1).mean())
        nav = np.load(C / "navtrain_full.s0of12/tab.npz")
        conv = dict(navtrain=convention(nav["ego"], nav["speed"]), replay_fixed=convention(fed, spd), replay_shipped=convention(ship, spd),
                    b2d_cache_same_rows=convention(lab, spd))
        towns = sorted({r["town"] for r in rows})
        n_turn = sum(bool(r["turn"]) for r in rows)
        ks = [conv[k]["ay_slope"] for k in ("navtrain", "replay_fixed")]
        ship_d = np.abs(ship - lab)
        res = dict(base=a.base, clips=len(rows), towns=towns, turn_clips=n_turn, ticks_replayed=ident_ticks, rows_compared=int(len(fed)),
                   bit_identity=dict(passed=ident_ok, ticks=ident_ticks, what="parity_ego of the agent file at --base vs the working file with port = {}, np.array_equal over all ticks of all clips"),
                   i_noncmd=dict(max_abs=float(d[:, NONCMD].max()), line=1e-3, passed=bool(d[:, NONCMD].max() <= 1e-3), per_dim_max=per_dim),
                   i_cmd=dict(agree_L20=agree, line=0.99, passed=bool(agree >= 0.99), disagreements=len(bad_cmd), rows=bad_cmd[:200]),
                   i_material=dict(passed=bool(len(rows) >= 20 and len(towns) >= 3 and n_turn >= 5)),
                   ii_convention=dict(conv, line=[0.8, 1.2], passed=bool(all(0.8 <= k <= 1.2 for k in ks))),
                   shipped_vs_labels=dict(cmd_agree_L20=float((ship[:, 1:4] == lab[:, 1:4]).all(1).mean()),
                                          cmd_agree_L20_on_turn_rows=float((ship[:, 1:4] == lab[:, 1:4]).all(1)[lab[:, 2] == 0].mean()),
                                          abs_err_p50_p95_max={n_: [round(float(x) * s_, 4) for x in (*np.percentile(ship_d[:, k], [50, 95]), ship_d[:, k].max())]
                                                               for n_, k, s_ in (("vx m/s", 4, 10), ("vy m/s", 5, 10), ("ax m/s2", 6, 3), ("ay m/s2", 7, 3))}))
        res["passed"] = bool(res["bit_identity"]["passed"] and res["i_noncmd"]["passed"] and res["i_cmd"]["passed"] and res["i_material"]["passed"]
                             and res["ii_convention"]["passed"])
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "port_check.json").write_text(json.dumps(res, indent=1) + "\n")
        with open(OUT / "port_check.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
        run.info(json.dumps({k: v for k, v in res.items() if k != "i_cmd"}, indent=1))
        run.info("command: agree %.5f, %d disagreeing rows; first: %s", agree, len(bad_cmd), json.dumps(bad_cmd[:12]))
        run.summary.update(passed=res["passed"], bit_identity=ident_ok, noncmd_max=res["i_noncmd"]["max_abs"], cmd_agree=agree, ay_slope=ks)
        assert not math.isnan(agree)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--turns", type=int, default=10, help="at least this many clips with a junction turn")
    ap.add_argument("--base", default="2b8a2b55", help="git revision of lib/op_arb_agent.py before the port switches")
    main(ap.parse_args())
