"""AUD1: synthetic-input audit of the served AlpaSim nuPlan driver (results/aud1_driver_assumptions.md).

The submission image serves lib/ap2_driver.py + lib/ap2_core.py with APY10m10-AB-s0, interpreted model passes (SH30_COMPILE=0), slot warps on
the card (SH30_SYNTH=gpu), the cold rule stored in the checkpoint (backwarp) and every serving switch off. This script builds exactly that
Driver, replays logged driver-side messages (c1_extract.py msgs: the requests AlpaSim sent in a finished rollout) through it with ONE input
changed at a time, and reports the plan's first 4 s against the unchanged replay. Open loop: the logged state at every decision is kept.

  aud1_driver_tests.py tests --msgs DIR [DIR ...] --out OUT.json [--tag APY10m10-AB-s0]   (box, envs/op-train, one GPU: a pool job)
  aud1_driver_tests.py stats --msgs DIR [DIR ...] --logs GLOB --out OUT.json              (box, envs/op-train, CPU: message / log statistics)

Plan numbers: rear-axle (x, y, yaw) of the plan at 1, 2, 3, 4 s in the t0 rig frame; d* = test minus nominal at the same decision.
"""
from __future__ import annotations

import argparse
import copy
import glob
import io
import json
import os
import pickle
import sys
import threading
import time
import uuid as U
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO / "experiments/alpasim/lib"))
import sh30_driver as D  # noqa: E402
import ap2_inputs as AI  # noqa: E402

pb = D.egodriver_pb2
CLS = {"driver_session_request": pb.DriveSessionRequest, "driver_camera_image": pb.RolloutCameraImage,
       "driver_ego_trajectory": pb.RolloutEgoTrajectory, "route_request": pb.RouteRequest, "driver_request": pb.DriveRequest}
T4 = [1, 3, 5, 7]                                   # plan rows at 1, 2, 3, 4 s


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def load(f) -> list:
    return [(k, raw) for k, raw in pickle.load(open(f, "rb")) if k in CLS]


def parse(raw_msgs) -> list:
    return [(k, CLS[k].FromString(raw)) for k, raw in raw_msgs]


def rot(a):
    return D.rot(a)


def yaw_q(q) -> float:
    return D.yaw_from_quat(q)


def set_yaw(q, yaw):
    q.w, q.x, q.y, q.z = float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))


# ---------------------------------------------------------------- replay through the real driver
TL = threading.local()


def build(tag: str):
    import ap2_driver as AD
    core = AD.AC.Core(tag, "cuda", "", 1.0, "gpu")          # = ap2_driver.main with the image's env (SH30_COMPILE=0: no tune / compile)
    assert not D.FX.ON, "serving switches must be off (the image's default)"
    orig = core.plan

    def plan(*a, **kw):
        o = orig(*a, **kw)
        TL.o, TL.a = o, dict(n_keys=len(a[0]), vel=np.asarray(a[2]).copy(), acc=np.asarray(a[3]).copy(), cmd=np.asarray(a[4]).copy(),
                              hist=np.asarray(a[1]).copy())
        return o
    core.plan = plan
    import tempfile
    return core, AD.Driver(core, Path(tempfile.mkdtemp()), 0, False, "")


def replay(drv, raw_msgs, mod=None, extra=None, keep_frames=False, sid=None) -> list:
    """Feed one scene's messages (mod(kind, msg, k) -> msg | None | list of msgs) and call drive at every decision -> one record per decision."""
    sid, ctx, k, out = sid or U.uuid4().hex, Ctx(), 0, []
    msgs = parse(raw_msgs) + (extra or [])
    for kind, m in msgs:
        m.session_uuid = sid
        r = mod(kind, m, k) if mod else m
        for m in ([] if r is None else r if isinstance(r, list) else [r]):
            m.session_uuid = sid
            sub = {"driver_session_request": drv.start_session, "driver_camera_image": drv.submit_image_observation,
                   "driver_ego_trajectory": drv.submit_egomotion_observation, "route_request": drv.submit_route}.get(kind)
            if sub is not None:
                try:
                    sub(m, ctx)
                except Exception as e:                               # a rejected observation: the evaluator's call fails
                    out.append(dict(k=k, now=-1, err=f"{kind}: {str(e)[:300]}"))
            elif kind == "driver_request":
                t = time.perf_counter()
                try:
                    drv.drive(m, ctx)
                    o, a = TL.o, TL.a
                    rec = dict(k=k, now=int(m.time_now_us), poses=o["poses"].astype(float).copy(), n_keys=a["n_keys"], cmd=a["cmd"].tolist(),
                               v=a["vel"][-1].tolist(), a=a["acc"].tolist(), ego=o["ego"][:20].astype(float).tolist(), n_slots=int(o["valid"].sum()),
                               ms=1e3 * (time.perf_counter() - t))
                    if keep_frames:
                        rec["cur"] = o["cur"].copy()
                except Exception as e:                               # an abort = a failed rollout in the evaluator
                    rec = dict(k=k, now=int(m.time_now_us), err=str(e)[:300])
                out.append(rec)
        if kind == "driver_request":
            k += 1
    drv.close_session(pb.DriveSessionCloseRequest(session_uuid=sid), ctx)
    return out


def summ(r) -> dict:
    if "err" in r:
        return {"k": r["k"], "err": r["err"]}
    p = r["poses"]
    return {"k": r["k"], "n_keys": r["n_keys"], "cmd": int(np.argmax(r["cmd"])) if np.any(r["cmd"]) else -1, "cmd_vec": r["cmd"],
            "v": np.round(r["v"], 3).tolist(), "a": np.round(r["a"], 3).tolist(),
            "x": np.round(p[T4, 0], 2).tolist(), "y": np.round(p[T4, 1], 3).tolist(), "yaw_deg": np.round(np.degrees(p[T4, 2]), 2).tolist()}


def diff(test, nom) -> dict:
    """Per decision: test minus nominal at 4 s and the largest |dy| over 0.5 .. 4 s."""
    nm = {r["k"]: r for r in nom if "err" not in r}
    rows = []
    for r in test:
        n = nm.get(r["k"])
        if "err" in r or n is None:
            rows.append({"k": r["k"], "err": r.get("err", "nominal missing")})
            continue
        d = r["poses"] - n["poses"]
        rows.append({"k": r["k"], "dx4": round(float(d[7, 0]), 3), "dy4": round(float(d[7, 1]), 3), "dyaw4_deg": round(float(np.degrees(d[7, 2])), 2),
                     "max_dy": round(float(np.abs(d[:, 1]).max()), 3), "n_keys": r["n_keys"], "cmd": int(np.argmax(r["cmd"])) if np.any(r["cmd"]) else -1,
                     "v": np.round(r["v"], 2).tolist(), "a": np.round(r["a"], 2).tolist()})
    ok = [x for x in rows if "err" not in x]
    agg = {"n": len(rows), "errors": len(rows) - len(ok)}
    if ok:
        agg.update(max_abs_dy4=max(abs(x["dy4"]) for x in ok), max_abs_dyaw4_deg=max(abs(x["dyaw4_deg"]) for x in ok),
                   max_abs_dx4=max(abs(x["dx4"]) for x in ok), max_dy_any=max(x["max_dy"] for x in ok))
    return {"agg": agg, "rows": rows}


# ---------------------------------------------------------------- message modifications
def route_mod(fn, ks=None):
    def mod(kind, m, k):
        if kind != "route_request" or (ks is not None and k not in ks):
            return m
        w = np.array([[q.x, q.y, q.z] for q in m.route.waypoints], np.float64).reshape(-1, 3)
        w2 = fn(w)
        if w2 is None:
            return None
        del m.route.waypoints[:]
        for p in w2:
            m.route.waypoints.add(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        return m
    return mod


def valid(w):
    return w[~np.isnan(w[:, 0])]


def shift_xy(dx=0.0, dy=0.0):
    def f(w):
        w = w.copy()
        w[:, 0] += dx
        w[:, 1] += dy
        return w
    return f


def first_nan(w):
    w = w.copy()
    i = np.flatnonzero(~np.isnan(w[:, 0]))
    if len(i):
        w[i[0]] = np.nan
    return w


def ego_mod(fn, ks=None):
    """fn(pose, dyn, k, i) mutates one (PoseAtTime, DynamicState) pair of an egomotion message."""
    def mod(kind, m, k):
        if kind != "driver_ego_trajectory" or (ks is not None and k not in ks):
            return m
        for i, (p, d) in enumerate(zip(m.trajectory.poses, m.dynamic_states)):
            fn(p, d, k, i)
        return m
    return mod


def to_rig_frame_start(t_first):
    """The start states as a fixed evaluator would deliver them: rotate the delivered (-yaw) vectors back into the rig frame."""
    def fn(p, d, k, i):
        if p.timestamp_us <= t_first:
            R = rot(yaw_q(p.pose.quat))
            for vec in (d.linear_velocity, d.linear_acceleration):
                x, y = R @ np.array([vec.x, vec.y])
                vec.x, vec.y = float(x), float(y)
    return fn


def acc_zero(p, d, k, i):
    d.linear_acceleration.x = d.linear_acceleration.y = 0.0


def pos_noise(sigma, seed=0):
    rng = np.random.default_rng(seed)

    def fn(p, d, k, i):
        p.pose.vec.x += float(rng.normal(0, sigma))
        p.pose.vec.y += float(rng.normal(0, sigma))
    return fn


def yaw_noise(sigma_deg, seed=0):
    rng = np.random.default_rng(seed)

    def fn(p, d, k, i):
        set_yaw(p.pose.quat, yaw_q(p.pose.quat) + np.radians(rng.normal(0, sigma_deg)))
    return fn


def drop_first_pose():
    """The first egomotion message carries one pose only (no 17 ms companion pose)."""
    def mod(kind, m, k):
        if kind == "driver_ego_trajectory" and k == 0 and len(m.trajectory.poses) > 1:
            P, Dn = list(m.trajectory.poses), list(m.dynamic_states)
            del m.trajectory.poses[:]
            del m.dynamic_states[:]
            m.trajectory.poses.add().CopyFrom(P[-1])
            m.dynamic_states.add().CopyFrom(Dn[-1])
        return m
    return mod


def speed_k0(target):
    """k = 0 only: the start states' velocity scaled to `target` m/s, the 17 ms pose difference scaled with it."""
    def mod(kind, m, k):
        if kind == "driver_ego_trajectory" and k == 0:
            P = m.trajectory.poses
            v0 = max(np.hypot(m.dynamic_states[-1].linear_velocity.x, m.dynamic_states[-1].linear_velocity.y), 1e-3)
            s = target / v0
            for d in m.dynamic_states:
                d.linear_velocity.x *= s
                d.linear_velocity.y *= s
            if len(P) > 1:
                x0, y0 = P[0].pose.vec.x, P[0].pose.vec.y
                for p in P[1:]:
                    p.pose.vec.x, p.pose.vec.y = x0 + s * (p.pose.vec.x - x0), y0 + s * (p.pose.vec.y - y0)
        return m
    return mod


def retime(scale=1.0, t_ref=17000, jitter_frames=0, first_at_zero=False):
    """Timestamps t >= t_ref -> t_ref + scale (t - t_ref); jitter_frames us added to CAM_F0 frame times from decision 1 on; first_at_zero: the
    first decision at t = 0 (its frame, its drive) and the 17 ms pose removed."""
    f = lambda t: int(t_ref + scale * (t - t_ref)) if t >= t_ref else int(t)  # noqa: E731

    def mod(kind, m, k):
        if kind == "driver_camera_image":
            ci = m.camera_image
            ci.frame_start_us, ci.frame_end_us = f(ci.frame_start_us), f(ci.frame_end_us)
            if first_at_zero and k == 0:
                ci.frame_end_us = 0
            if jitter_frames and k >= 1:
                ci.frame_end_us += jitter_frames
        elif kind == "driver_ego_trajectory":
            P, Dn = list(m.trajectory.poses), list(m.dynamic_states)
            keep = [(p, d) for p, d in zip(P, Dn) if not (first_at_zero and k == 0 and p.timestamp_us == t_ref)]
            del m.trajectory.poses[:]
            del m.dynamic_states[:]
            for p, d in keep:
                q = m.trajectory.poses.add()
                q.CopyFrom(p)
                q.timestamp_us = f(p.timestamp_us)
                m.dynamic_states.add().CopyFrom(d)
        elif kind == "route_request":
            m.route.timestamp_us = f(m.route.timestamp_us)
        elif kind == "driver_request":
            m.time_now_us, m.time_query_us = f(m.time_now_us), f(m.time_query_us)
            if first_at_zero and k == 0:
                m.time_now_us, m.time_query_us = 0, 500000
        return m
    return mod


def session_mod(fn):
    def mod(kind, m, k):
        if kind == "driver_session_request":
            for c in m.rollout_spec.vehicle.available_cameras:
                if c.logical_id == D.CAM:
                    fn(c)
        return m
    return mod


def cam_rot(axis: int, deg: float):
    from scipy.spatial.transform import Rotation
    def fn(c):
        q = c.rig_to_camera.quat
        R = D.quat_R(q) @ Rotation.from_rotvec(np.radians(deg) * np.eye(3)[axis]).as_matrix()
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)
    return fn


def cam_t(dx=0.0, dz=0.0):
    def fn(c):
        c.rig_to_camera.vec.x += dx
        c.rig_to_camera.vec.z += dz
    return fn


def cam_f(scale):
    def fn(c):
        p = c.intrinsics.opencv_pinhole_param
        p.focal_length_x *= scale
        p.focal_length_y *= scale
    return fn


def half_res():
    """CAM_F0 delivered at 960 x 540 with the intrinsics scaled to match."""
    from PIL import Image

    def mod(kind, m, k):
        if kind == "driver_session_request":
            for c in m.rollout_spec.vehicle.available_cameras:
                if c.logical_id == D.CAM:
                    p = c.intrinsics.opencv_pinhole_param
                    p.focal_length_x, p.focal_length_y = p.focal_length_x / 2, p.focal_length_y / 2
                    p.principal_point_x, p.principal_point_y = p.principal_point_x / 2, p.principal_point_y / 2
        elif kind == "driver_camera_image" and m.camera_image.logical_id == D.CAM:
            im = Image.open(io.BytesIO(m.camera_image.image_bytes)).convert("RGB").resize((960, 540), Image.BILINEAR)
            b = io.BytesIO()
            im.save(b, "JPEG", quality=95)
            m.camera_image.image_bytes = b.getvalue()
        return m
    return mod


def extension(raw_msgs, n: int) -> list:
    """n more decisions after the logged ones: the last CAM_F0 image again, the ego moving on at its last velocity and heading, the last route."""
    msgs = parse(raw_msgs)
    img = [m for k, m in msgs if k == "driver_camera_image" and m.camera_image.logical_id == D.CAM][-1]
    ego = [m for k, m in msgs if k == "driver_ego_trajectory"][-1]
    route = [m for k, m in msgs if k == "route_request"][-1]
    p0, d0 = ego.trajectory.poses[-1], ego.dynamic_states[-1]
    yaw = yaw_q(p0.pose.quat)
    vw = rot(yaw) @ np.array([d0.linear_velocity.x, d0.linear_velocity.y])
    t_last, out = int(p0.timestamp_us), []
    for j in range(1, n + 1):
        t = t_last + j * 500000
        im = copy.deepcopy(img)
        im.camera_image.frame_start_us, im.camera_image.frame_end_us = t - 17000, t
        e = pb.RolloutEgoTrajectory()
        p = e.trajectory.poses.add()
        p.CopyFrom(p0)
        p.timestamp_us = t
        p.pose.vec.x, p.pose.vec.y = p0.pose.vec.x + vw[0] * 0.5 * j, p0.pose.vec.y + vw[1] * 0.5 * j
        e.dynamic_states.add().CopyFrom(d0)
        r = copy.deepcopy(route)
        r.route.timestamp_us = t
        out += [("driver_camera_image", im), ("driver_ego_trajectory", e), ("route_request", r),
                ("driver_request", pb.DriveRequest(time_now_us=t, time_query_us=t + 500000))]
    return out


# ---------------------------------------------------------------- scene facts and selection
def facts(raw_msgs) -> dict:
    msgs = parse(raw_msgs)
    ego0 = next(m for k, m in msgs if k == "driver_ego_trajectory")
    drives = [m for k, m in msgs if k == "driver_request"]
    routes = [np.array([[q.x, q.y] for q in m.route.waypoints], np.float64).reshape(-1, 2) for k, m in msgs if k == "route_request"]
    st = [m.dynamic_states[-1] for k, m in msgs if k == "driver_ego_trajectory"]
    sp = [float(np.hypot(d.linear_velocity.x, d.linear_velocity.y)) for d in st]
    sess = next(m for k, m in msgs if k == "driver_session_request")
    cam = next(c for c in sess.rollout_spec.vehicle.available_cameras if c.logical_id == D.CAM)
    imgs = [m.camera_image for k, m in msgs if k == "driver_camera_image" and m.camera_image.logical_id == D.CAM]
    from PIL import Image
    wh = Image.open(io.BytesIO(imgs[0].image_bytes)).size
    return {"n_pose0": len(ego0.trajectory.poses), "pose0_t": [int(p.timestamp_us) for p in ego0.trajectory.poses],
            "yaw0": yaw_q(ego0.trajectory.poses[-1].pose.quat), "speed": sp, "now": [int(d.time_now_us) for d in drives],
            "frame_end": [int(c.frame_end_us) for c in imgs], "img_wh": wh,
            "cmd": [int(np.argmax(D.command_from_route(m.route))) for k, m in msgs if k == "route_request"],
            "cmd_ai": [int(np.argmax(AI.route_cmd(w))) for w in routes],
            "n_valid": [int((~np.isnan(w[:, 0])).sum()) for w in routes], "n_wp": [len(w) for w in routes],
            "first_x": [float(valid(w)[0, 0]) if len(valid(w)) else None for w in routes],
            "first_y": [float(valid(w)[0, 1]) if len(valid(w)) else None for w in routes],
            "x_range": [[float(np.nanmin(w[:, 0])), float(np.nanmax(w[:, 0]))] if (~np.isnan(w[:, 0])).any() else None for w in routes],
            "cam_t": [cam.rig_to_camera.vec.x, cam.rig_to_camera.vec.y, cam.rig_to_camera.vec.z],
            "cam_f": [cam.intrinsics.opencv_pinhole_param.focal_length_x, cam.intrinsics.opencv_pinhole_param.focal_length_y]}


def pick(F: dict) -> dict:
    """Scene roles by their k = 0 facts."""
    S = {}
    by = sorted(F)

    def first(*preds):
        for p in preds:
            c = [s for s in by if p(F[s])]
            if c:
                return c[0]
        raise ValueError("no scene for a role")
    yawed = lambda f: abs(np.sin(f["yaw0"])) > 0.7  # noqa: E731  (heading far from the local x axis: the -yaw rotation matters)
    S["cruise"] = first(lambda f: 7 < f["speed"][0] < 14 and yawed(f) and set(f["cmd"]) == {1}, lambda f: 7 < f["speed"][0] < 14 and yawed(f),
                        lambda f: 5 < f["speed"][0])
    S["stop"] = first(lambda f: f["speed"][0] < 0.2 and max(f["speed"]) > 1.0, lambda f: f["speed"][0] < 0.2)   # starts at rest and launches
    S["slow"] = first(lambda f: 1.2 < f["speed"][0] < 3.0 and yawed(f), lambda f: 1.0 < f["speed"][0] < 4.0)
    S["turn"] = max(by, key=lambda s: (sum(c != 1 for c in F[s]["cmd"]), F[s]["speed"][0] > 3))
    S["fast"] = max(by, key=lambda s: F[s]["speed"][0])
    sg = [s for s in by if "veh-52" in s and F[s]["speed"][0] > 3]
    if sg:
        S["sg"] = sg[0]
    return S


# ---------------------------------------------------------------- the tests
def run_tests(a) -> dict:
    files = sorted(f for d in a.msgs for f in glob.glob(f"{d}/*.pkl"))
    raw = {Path(f).stem: f for f in files}
    F = {s: facts(load(f)) for s, f in raw.items()}
    S = pick(F)
    print("roles", json.dumps(S), flush=True)
    core, drv = build(a.tag)
    M = {s: load(raw[s]) for s in set(S.values())}
    res = {"tag": a.tag, "roles": S, "facts": {r: {k: v for k, v in F[s].items()} for r, s in S.items()}, "tests": {}}
    nom = {r: replay(drv, M[s]) for r, s in S.items()}
    rep = {r: replay(drv, M[s]) for r, s in S.items()}                  # determinism of the harness itself
    res["nominal"] = {r: [summ(x) for x in v] for r, v in nom.items()}
    res["determinism"] = {r: diff(rep[r], nom[r])["agg"] for r in S}

    def T(name, role, note, **kw):
        out = replay(drv, M[S[role]], **kw)
        res["tests"][name] = {"role": role, "note": note, **diff(out, nom[role]), "test": [summ(x) for x in out]}
        g = res["tests"][name]["agg"]
        print(f"{name:28s} {role:7s} {json.dumps(g)}", flush=True)
        return out

    # 1. route
    for role in ("cruise", "turn"):
        T(f"route_all_nan@{role}", role, "20 NaN waypoints every decision", mod=route_mod(lambda w: np.full_like(w, np.nan)))
        T(f"route_empty@{role}", role, "0 waypoints every decision (UNKNOWN command)", mod=route_mod(lambda w: w[:0]))
        T(f"route_none@{role}", role, "no route message at all (session default command)", mod=route_mod(lambda w: None))
        T(f"route_short5@{role}", role, "first 5 valid waypoints, no padding", mod=route_mod(lambda w: valid(w)[:5]))
        T(f"route_first_nan@{role}", role, "first valid waypoint NaN", mod=route_mod(first_nan))
        T(f"route_far@{role}", role, "all waypoints 40 m further (>= 80 m)", mod=route_mod(shift_xy(dx=40.0)))
        T(f"route_beside+3@{role}", role, "route shifted 3 m left", mod=route_mod(shift_xy(dy=3.0)))
        T(f"route_beside-3@{role}", role, "route shifted 3 m right", mod=route_mod(shift_xy(dy=-3.0)))
        T(f"route_behind@{role}", role, "route shifted 50 m back (starts ~8 m behind the ego)", mod=route_mod(shift_xy(dx=-50.0)))
        T(f"route_left5@{role}", role, "route shifted 5 m left (forces LEFT)", mod=route_mod(shift_xy(dy=5.0)))
        T(f"route_right5@{role}", role, "route shifted 5 m right (forces RIGHT)", mod=route_mod(shift_xy(dy=-5.0)))
    T("route_all_nan@stop", "stop", "20 NaN waypoints at a standing start", mod=route_mod(lambda w: np.full_like(w, np.nan)))

    # 2. egomotion
    for role in ("cruise", "slow", "stop"):
        t1 = F[S[role]]["now"][0]
        T(f"ego_start_rigframe@{role}", role, "start states delivered in the rig frame (evaluator without the -yaw rotation)",
          mod=ego_mod(to_rig_frame_start(t1), ks=[0]))
        T(f"ego_one_pose0@{role}", role, "first egomotion message has one pose (no 17 ms companion)", mod=drop_first_pose())
        T(f"ego_acc0@{role}", role, "acceleration missing (0) in every state", mod=ego_mod(acc_zero))
        T(f"ego_posnoise3cm@{role}", role, "N(0, 3 cm) on every pose x, y", mod=ego_mod(pos_noise(0.03)))
        T(f"ego_posnoise10cm@{role}", role, "N(0, 10 cm) on every pose x, y", mod=ego_mod(pos_noise(0.10)))
        T(f"ego_yawnoise1deg@{role}", role, "N(0, 1 deg) on every pose yaw", mod=ego_mod(yaw_noise(1.0)))

    # 3. cold start
    T("cold_k0_25mps@cruise", "cruise", "k = 0 speed scaled to 25 m/s (start states and 17 ms pose step)", mod=speed_k0(25.0))
    T("cold_k0_0mps@cruise", "cruise", "k = 0 speed scaled to 0.05 m/s", mod=speed_k0(0.05))
    for role in ("cruise", "stop"):
        T(f"cold_spacing0.4@{role}", role, "decision / frame / pose times on a 0.4 s grid", mod=retime(scale=0.8))
        T(f"cold_spacing0.6@{role}", role, "decision / frame / pose times on a 0.6 s grid", mod=retime(scale=1.2))
        T(f"cold_jitter3ms@{role}", role, "CAM_F0 frame_end 3 ms late from decision 1 on", mod=retime(jitter_frames=3000))
        T(f"cold_jitter1ms@{role}", role, "CAM_F0 frame_end 1 ms late from decision 1 on", mod=retime(jitter_frames=1000))
        T(f"cold_first_at_0@{role}", role, "first decision at t = 0 (no 17 ms offset, one start pose)", mod=retime(first_at_zero=True))

    # 4. camera calibration
    for role in ("cruise", "turn"):
        T(f"cal_pitch+1@{role}", role, "rig_to_camera rotated 1 deg about the camera x axis", mod=session_mod(cam_rot(0, 1.0)))
        T(f"cal_roll+1@{role}", role, "rig_to_camera rotated 1 deg about the camera z axis", mod=session_mod(cam_rot(2, 1.0)))
        T(f"cal_yaw+1@{role}", role, "rig_to_camera rotated 1 deg about the camera y axis", mod=session_mod(cam_rot(1, 1.0)))
        T(f"cal_height+0.3@{role}", role, "camera 0.3 m higher (t.z)", mod=session_mod(cam_t(dz=0.3)))
        T(f"cal_x+0.5@{role}", role, "camera 0.5 m further forward (t.x)", mod=session_mod(cam_t(dx=0.5)))
        T(f"cal_f+2pct@{role}", role, "focal lengths x 1.02", mod=session_mod(cam_f(1.02)))
        T(f"cal_halfres@{role}", role, "CAM_F0 at 960 x 540 with K scaled", mod=half_res())
    if "sg" in S:
        drv.lht = True
        T("lht_on@sg", "sg", "Singapore scene served with the left-hand-traffic flag (SH30_LHT=1)")
        drv.lht = False
        drv.lht = True
        T("lht_on@cruise", "cruise", "US scene served with the left-hand-traffic flag")
        drv.lht = False

    # 5. state: long scenes and concurrent sessions
    ext = extension(M[S["cruise"]], 24)
    out = replay(drv, M[S["cruise"]], extra=ext)
    ex = [r for r in out if r["k"] >= 10 and "err" not in r]
    steady = [r for r in ex if r["k"] >= 14]
    res["tests"]["long_scene@cruise"] = {
        "role": "cruise", "note": "24 decisions after the logged 10: same image, constant velocity and heading, same route",
        "agg": {"n": len(ex), "errors": sum("err" in r for r in out),
                "steady_max_abs_dpose": float(max(np.abs(r["poses"] - steady[0]["poses"]).max() for r in steady)),
                "ms_first10": float(np.mean([r["ms"] for r in out if r["k"] < 10 and "err" not in r])), "ms_last10": float(np.mean([r["ms"] for r in out[-10:] if "err" not in r]))},
        "test": [summ(x) for x in out]}
    print("long_scene", json.dumps(res["tests"]["long_scene@cruise"]["agg"]), flush=True)

    roles = [r for r in S if r != "fast"][:4]
    seq = {r: nom[r] for r in roles}
    for trial in range(2):
        got, th = {}, []
        for r in roles + roles[:1]:                                   # one scene twice at the same time
            key = r if r not in got else r + "_dup"
            got[key] = None

            def go(r=r, key=key):
                got[key] = replay(drv, M[S[r]])
            th.append(threading.Thread(target=go))
        [t.start() for t in th]
        [t.join() for t in th]
        res["tests"][f"concurrent_{trial}"] = {"note": f"{len(th)} sessions replayed at once in threads through one Driver (2 of them the same scene)",
                                               "agg": {k: diff(v, seq[k.replace('_dup', '')])["agg"] for k, v in got.items()}}
        print("concurrent", json.dumps(res["tests"][f"concurrent_{trial}"]["agg"]), flush=True)

    # 4b. which calibration components reach the model frames (pixels of the newest slot and the oldest synthesised slot)
    base = replay(drv, M[S["cruise"]], keep_frames=True)
    pix = {}
    for name, mod in [("pitch+1", session_mod(cam_rot(0, 1.0))), ("roll+1", session_mod(cam_rot(2, 1.0))), ("height+0.3", session_mod(cam_t(dz=0.3))),
                      ("x+0.5", session_mod(cam_t(dx=0.5))), ("f+2pct", session_mod(cam_f(1.02)))]:
        o = replay(drv, M[S["cruise"]], mod=mod, keep_frames=True)
        pix[name] = {f"k{k}": {"newest_slot_px_changed": float((o[k]["cur"][-1] != base[k]["cur"][-1]).mean()),
                               "oldest_slot_px_changed": float((o[k]["cur"][0] != base[k]["cur"][0]).mean())} for k in (0, 5)}
    res["frames_vs_calibration"] = pix
    print("frames", json.dumps(pix), flush=True)
    return res


# ---------------------------------------------------------------- statistics over many scenes and the served runs' logs
def run_stats(a) -> dict:
    files = sorted(f for d in a.msgs for f in glob.glob(f"{d}/*.pkl"))
    F = {}
    for f in files:
        try:
            F[Path(f).stem] = facts(load(f))
        except Exception as e:
            print("skip", f, e)
    cmd_names = ["L", "S", "R", "U"]
    flips = sum(F[s]["cmd"][0] != F[s]["cmd"][1] for s in F if len(F[s]["cmd"]) > 1)
    agree = sum(c == d for s in F for c, d in zip(F[s]["cmd"], F[s]["cmd_ai"]))
    tot = sum(len(F[s]["cmd"]) for s in F)
    still = [(c, sp) for s in F for c, sp in zip(F[s]["cmd"], F[s]["speed"]) if sp < 0.5]
    nv = np.array([n for s in F for n in F[s]["n_valid"]])
    fx = np.array([x for s in F for x in F[s]["first_x"] if x is not None])
    xr = np.array([r for s in F for r in F[s]["x_range"] if r is not None])
    out = {"n_scenes": len(F), "decisions": tot,
           "n_pose_first_msg": dict(zip(*map(lambda x: x.tolist(), np.unique([F[s]["n_pose0"] for s in F], return_counts=True)))),
           "first_now_us": dict(zip(*map(lambda x: x.tolist(), np.unique([F[s]["now"][0] for s in F], return_counts=True)))),
           "frame_end_minus_now_max_us": max(abs(fe - n) for s in F for fe, n in zip(F[s]["frame_end"], F[s]["now"])),
           "img_wh": sorted({tuple(F[s]["img_wh"]) for s in F}),
           "cam_t_range": np.round([np.min([F[s]["cam_t"] for s in F], 0), np.max([F[s]["cam_t"] for s in F], 0)], 3).tolist(),
           "cam_f_range": np.round([np.min([F[s]["cam_f"] for s in F], 0), np.max([F[s]["cam_f"] for s in F], 0)], 1).tolist(),
           "n_wp_per_message": sorted({n for s in F for n in F[s]["n_wp"]}),
           "n_valid_waypoints": {"min": int(nv.min()), "p50": float(np.median(nv)), "max": int(nv.max()), "zero": int((nv == 0).sum()),
                                 "lt5": int((nv < 5).sum())},
           "first_valid_x": {"min": float(fx.min()), "p50": float(np.median(fx)), "max": float(fx.max())},
           "valid_x_range": {"min": float(xr[:, 0].min()), "max": float(xr[:, 1].max())},
           "cmd_counts": {cmd_names[c]: int(n) for c, n in zip(*np.unique([c for s in F for c in F[s]["cmd"]], return_counts=True))},
           "cmd_k0_vs_k1_flips": int(flips), "cmd_shipped_vs_ap2_inputs_agree": f"{agree}/{tot}",
           "cmd_at_standstill": {cmd_names[c]: int(n) for c, n in zip(*np.unique([c for c, _ in still], return_counts=True))} if still else {},
           "speed_k0": {"lt0.2": sum(F[s]["speed"][0] < 0.2 for s in F), "0.2-1": sum(0.2 <= F[s]["speed"][0] < 1 for s in F),
                        "ge1": sum(F[s]["speed"][0] >= 1 for s in F), "max": max(F[s]["speed"][0] for s in F)}}
    # the served runs' own decision logs: was the k = 0 state rotated, and how often is the fed lateral velocity implausible
    rows = []
    for f in sorted(glob.glob(a.logs, recursive=True)):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "drive":
                rows.append((r["k"], r["rotated"][-1], 10 * r["ego"][4], 10 * r["ego"][5], r["n_keys"], r["cmd"], r.get("n_wp")))
    if rows:
        R = np.array([(k, rot_, vx, vy, nk, c) for k, rot_, vx, vy, nk, c, _ in rows], float)
        k0 = R[R[:, 0] == 0]
        out["served_logs"] = {"files": len(glob.glob(a.logs, recursive=True)), "decisions": len(R), "k0": len(k0),
                              "k0_rotated": int(k0[:, 1].sum()), "k0_fed_abs_vy_gt1": int((np.abs(k0[:, 3]) > 1).sum()),
                              "k0_fed_vx_lt0": int((k0[:, 2] < -0.3).sum()),
                              "kge1_fed_abs_vy_gt0.5": int((np.abs(R[R[:, 0] >= 1, 3]) > 0.5).sum()),
                              "n_keys_by_k": {int(k): np.unique(R[R[:, 0] == k, 4], return_counts=True)[0].astype(int).tolist() for k in range(12) if (R[:, 0] == k).any()},
                              "max_k": int(R[:, 0].max()),
                              "cmd_counts": {cmd_names[int(c)]: int(n) for c, n in zip(*np.unique(R[:, 5], return_counts=True))}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tests", "stats"])
    ap.add_argument("--msgs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="APY10m10-AB-s0")
    ap.add_argument("--logs", default="")
    a = ap.parse_args()
    res = run_tests(a) if a.mode == "tests" else run_stats(a)
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print("->", a.out, flush=True)


if __name__ == "__main__":
    main()
