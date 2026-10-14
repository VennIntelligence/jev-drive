"""Log-replay variant of experiments/alpasim/lib/pai_driver.py (lane ONPOL-PAI, pass 2): the simulated ego is steered along the scene's
LOGGED path while the model runs, is fed and is recorded exactly as in the normal driver. Imitation rows at on-log states with the PAI
serving input (f-theta camera through pai_core, rendered 10 Hz frames in every slot).

  logged path   `submit_recording_ground_truth`, which the runtime sends before every `drive` once the wizard override
                `runtime.simulation_config.send_recording_ground_truth=true` is set (off in the dev preset): the recording's whole ego
                trajectory (rig poses, absolute timestamps on the rollout clock), expressed in the TRUE rig frame of the step start
                (AlpaSim 0bb4c4b, runtime/events/policy.py). Composed here with the egomotion pose of `time_now` (full 3-D) it is the
                log in the frame the driver answers in, also when an egomotion error model is on
  model         untouched: pai_driver.Driver.drive runs first on every call and writes its usual record (`poses`, `poses_model`, dumps)
  returned      the log from `time_now` on: ONP_HORIZON s (default 4.0, the model plan's horizon) of 10 Hz poses, x / y / yaw in the local
                frame, z of the current pose and a yaw-only quaternion, as navsim_transfuser_challenge.trajectory.build_trajectory_from_plan
                writes them. The first pose is the log at `time_now`, not the ego's pose: an ego that is off the log gets a reference
                that pulls it back (no blending). Past the end of the log (the rollout ends where the log ends, so the last
                ONP_HORIZON s of every scene) the reference continues on the arc of the log's last 0.5 s (constant speed and yaw
                rate): the MPC clamps a short reference at its last pose, which would brake the ego
  hand-over     the runtime itself keeps the ego on the log for the first force_gt_duration_us (1.7 s in the dev preset) and seeds the
                controller's dynamic state from the log at the first closed-loop step; from there the MPC tracks this reference
  drive.jsonl   every drive record gains `ret` ("log"), `ret_n` (poses returned), `ret_ext` (how many of them are extrapolated),
                `log_now` (the log's pose at time_now in the rig frame of time_now, i.e. where the log is as seen from the ego:
                x > 0 = the ego is behind the log, y > 0 = the ego is right of it) and `log_poses` ((8, 3): the log at time_now + 0.5 .. 4.0 s in that frame, the imitation label in the layout of
                `poses`; `log_valid` of them lie inside the log). The close record gains `log_error`

  switch        ONP_SWITCH (unset = never): the decision index k (the session's `drive` call count, 10 Hz from the first call, as the
                record's `k`) from which the driver returns the MODEL's plan instead of the log, i.e. the policy drives from k on (turning
                school P2, route C of research/turning-school-checks.md). Either one integer for every scene or a file of `<scene id> <k>`
                lines (a scene not in the file is never switched; mount it into the driver container through DRV_ENV `-v`). Before k
                everything is as above; from k on the response is exactly pai_driver's (the model, its serving changes, its record), and the
                log fields are still written when the ground truth is there (`log_now` = where the log is, seen from the ego). A k inside
                the runtime's force-GT window (< 17 in the dev preset) hands over at the end of it; ONP_SWITCH=0 is pai_driver plus the
                log fields and needs no ground truth. Records gain `switch` (k, null = never) and `ret` ("log" | "model"); every record
                with ground truth gains `log_t` (s from the first pose of the recording to time_now: decision k <-> log time)

  after         ONP_AFTER (comma list, unset = none): input controls applied from the switch decision k on, never before (turning
                school P2 control arms). `rmem0`: the route-memory features fed to the model are the empty memory (what SH30_RMEM=0
                feeds for the whole run; the tracker keeps taking the route messages). `vhzero`: the ego's speed history is zeroed in the
                fed row (vhist.py JEV_VHIST=zero's entries; that switch alone zeroes it from decision 0, so it must stay unset here), logged
                as `vhist`. Records gain `after` (the controls active at this call)

  branch        ONP_BRANCH (unset = off, bit-identical to the driver without it): `g<gain>s<scale>[t<seconds>]`, e.g. g1.5s0.8 (t default
                3.0). For the decisions k_eff .. k_eff + 10 t - 1 (k_eff = max(switch k, ONP_FGT_K = 17), 10 Hz) the served plan's 8 poses
                are transformed before the MPC sees them (branch_poses: yaw and segment course about the ego multiplied by g at the same
                arc length, then the path re-sampled at s times its arc length), the raw policy plan before and after (turning school P3,
                experiments/route_follow/plans/2026-10-14-p3-branch-prereg.md). g1s1 returns the plan untouched. Records gain `branch`
                ({g, s, t, on}) and, while on, `poses_branch` (the transformed poses; `poses` stays the raw served plan)

  cells         ONP_CELLS (a JSON file; default <ALPASIM_DRIVER_LOG_DIR>/p4cells.json when it exists; unset and absent = off): turning
                school P4, the self-branch search (experiments/route_follow/plans/2026-10-14-p4-self-branch-prereg.md). The file has
                `scenes` {scene: onp_p4.Route fields: route grid u / x / y / kappa in the local frame, s_entry, s_exit, s_release, v_e} and
                `cells` {scene: [cell, ...]}; the n-th session of a scene (start order) takes cell n, so one simulator launch with
                n_rollouts = len(cells) runs every cell once (a session beyond the list is refused). A cell is {tag, D, src, g, lon,
                switch}: src "self" = the policy drives from decision 0 (ONP_SWITCH 0) and the handover h is the first decision k >= ONP_FGT_K
                at which the ego's route station (onp_p4.Route.station of the plan anchor) is within D m of the entry and the ego within
                onp_p4.FAR m of the route; src "log" = the log up to decision `switch`, h = max(switch, ONP_FGT_K) (P2 / P3's handover);
                D null = never (the identity cell). From h: lateral gain g for onp_p4.LAT_N decisions (P3's curvature push), and the
                longitudinal profile lon (raw | ve | ve08: onp_p4.profile to v_e or 0.8 v_e at the entry, floor 0.8 v_e before it; per
                decision arc scale s_k = onp_p4.scale) until the station reaches s_release, then the raw plan. Every record gains `p4`
                {tag, D, src, h, sigma, d_route, v_meas, lat, lon, released[, arc_prof, arc_plan, s_k, v_ent_prof]} and, when lat or lon
                is on, `poses_branch` (branch_poses(poses, g if lat else 1, s_k if lon else 1)). Exclusive with ONP_SWITCH / ONP_BRANCH.
                Renderer race: the NRE renderer serves garbage frames (a half-loaded scene) to a second session of a scene that starts while
                the scene is still loading for the first (P4 smoke try 1: 0.3 s of smeared frames, plans differ from k 0). So session n >= 1 of a
                scene waits in start_session (before any of its renders) until session 0 of that scene has received ONP_WARM (10) frames of
                its camera (at most 600 s); the wait is logged as p4.wait_s.

A `drive` before the switch without a ground-truth message of its own timestamp is aborted with a gRPC error and counted. Run through
experiments/onpol_pai/scripts/onp_pass2.sh (DRV_PY of pai_run.sh); environment as pai_driver.py plus ONP_HORIZON, ONP_SWITCH, ONP_AFTER,
ONP_BRANCH, ONP_FGT_K.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.normpath(os.path.join(_HERE, "..", "..", "alpasim", "lib"))):   # run from onpol_pai/lib (GPU box): pai_driver's dir
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.append(_d)
import pai_driver as P  # noqa: E402
from pai_driver import common_pb2, egodriver_pb2, grpc, S

HORIZON_S, TAIL_US = float(os.environ.get("ONP_HORIZON", "4.0")), 500_000


def switch_table(v: str) -> tuple:
    """ONP_SWITCH -> (k for every scene or None, {scene: k})."""
    if not v:
        return None, {}
    if v.strip().lstrip("-").isdigit():
        return int(v), {}
    tab = {}
    for line in open(v):
        f = line.split("#")[0].split()
        if len(f) >= 2:
            tab[f[0]] = int(f[1])
    return None, tab


SWITCH_ALL, SWITCH = switch_table(os.environ.get("ONP_SWITCH", ""))
AFTER = {x for x in os.environ.get("ONP_AFTER", "").replace(" ", "").split(",") if x}
assert AFTER <= {"rmem0", "vhzero"}, f"ONP_AFTER={AFTER}: rmem0, vhzero"
assert not ("vhzero" in AFTER and P.VH.ON), "ONP_AFTER=vhzero needs JEV_VHIST unset (it would act before the switch)"
_m = re.fullmatch(r"g([\d.]+)s([\d.]+)(?:t([\d.]+))?", os.environ.get("ONP_BRANCH", "").strip())
assert _m or not os.environ.get("ONP_BRANCH", "").strip(), "ONP_BRANCH=g<gain>s<scale>[t<seconds>], e.g. g1.5s0.8"
BRANCH = dict(g=float(_m[1]), s=float(_m[2]), t=float(_m[3] or 3.0)) if _m else None
FGT_K = int(os.environ.get("ONP_FGT_K", "17"))
assert not (BRANCH and P.TR.ON), "ONP_BRANCH transforms the plan handed to make_cached_plan; JEV_TRACK must be off"
P4_WARM = int(os.environ.get("ONP_WARM", "10"))
_CF = os.environ.get("ONP_CELLS") or os.path.join(os.environ.get("ALPASIM_DRIVER_LOG_DIR", "/nonexistent"), "p4cells.json")
CELLS = json.load(open(_CF)) if os.path.exists(_CF) else None
if CELLS:
    import onp_p4 as Q
    assert not (BRANCH or SWITCH or SWITCH_ALL is not None or AFTER or P.TR.ON), "ONP_CELLS excludes ONP_BRANCH / ONP_SWITCH / ONP_AFTER / JEV_TRACK"
    ROUTES = {sc: Q.Route(d) for sc, d in CELLS["scenes"].items()}
    for _sc, _cl in CELLS["cells"].items():
        assert _sc in ROUTES and all(c["src"] in ("self", "log") and c["lon"] in ("raw", "ve", "ve08") and (c["src"] == "self") == (c["switch"] == 0)
                                     for c in _cl), f"ONP_CELLS: bad cell list of {_sc}"


def branch_poses(poses, g: float, s: float):
    """(8, 3) x, y, yaw in the ego frame (ego at the origin, yaw 0) -> the transformed plan. Curvature gain g: every pose's yaw and every
    segment's course (from the previous pose, the origin first) are multiplied by g, the segment lengths kept, the path re-integrated.
    Speed scale s: the (transformed) polyline origin -> pose 8 re-sampled at s times each pose's arc length (same path, shorter), yaw
    interpolated along the arc. g = s = 1 returns the input untouched."""
    q = np.asarray(poses, np.float64)
    if g == 1.0 and s == 1.0:
        return q
    xy = np.r_[[[0.0, 0.0]], q[:, :2]]
    d = np.diff(xy, axis=0)
    ds = np.hypot(*d.T)
    c = g * np.unwrap(np.r_[0.0, np.arctan2(d[:, 1], d[:, 0])])[1:]
    yaw = g * np.unwrap(np.r_[0.0, q[:, 2]])
    xy = np.r_[[[0.0, 0.0]], np.cumsum(ds[:, None] * np.c_[np.cos(c), np.sin(c)], 0)]
    S = np.r_[0.0, np.cumsum(ds)]
    if s != 1.0:
        u = np.maximum.accumulate(S + 1e-9 * np.arange(len(S)))           # strictly increasing for interp (stopped plans)
        w = s * u[1:]
        return np.c_[np.interp(w, u, xy[:, 0]), np.interp(w, u, xy[:, 1]), np.interp(w, u, yaw)]
    return np.c_[xy[1:], yaw[1:]]


class MaskedMemory:
    """rmem.Tracker that feeds the empty memory (None: pai_core.plan's zeros) once `on` is set; route messages still go in."""

    def __init__(self, rt):
        self.rt, self.on = rt, False

    def add(self, *a, **k):
        return self.rt.add(*a, **k)

    def features(self, *a, **k):
        return None if self.on else self.rt.features(*a, **k)


class VhShim:
    """pai_driver's VH for ONP_AFTER=vhzero: a session's state {"zero": True} (set at the switch) gets vhist.py's `zero` row, every other
    call goes to vhist.py (which is off, so pai_driver never calls it before the switch)."""

    def __init__(self, vh):
        self.vh = vh

    def __getattr__(self, k):
        return getattr(self.vh, k)

    def feed(self, st, ego, cmd):
        if st.get("zero"):
            e = ego.copy()
            e[self.vh.EGO_SPEED] = 0.0
            return e, dict(mode="zero-after", v_meas=round(float(np.hypot(ego[4], ego[5]) * 10.0), 3), v_fed=0.0)
        return self.vh.feed(st, ego, cmd)

    def observe(self, st, o):
        if not st.get("zero"):
            self.vh.observe(st, o)


if "vhzero" in AFTER:
    P.VH = VhShim(P.VH)
TL = threading.local()


def p4_step(st, k, t0, anchor, poses):
    """P4 (ONP_CELLS): one decision of a session's cell -> (served poses, `p4` record fields). st = the session's cell state."""
    c, R, s = st["cell"], ROUTES[st["scene"]], st["session"]
    sig, dist = R.station((anchor.pose.vec.x, anchor.pose.vec.y), st["hint"])
    st["hint"] = sig
    with s.lock:
        v0 = float(np.hypot(*s.velocity(t0)[0]))
    if st["h"] is None and c["D"] is not None and k >= FGT_K:
        if (k >= max(c["switch"], FGT_K)) if c["src"] == "log" else (R.s_entry - sig <= c["D"] and dist <= Q.FAR):
            st["h"] = k
    h = st["h"]
    if h is not None and c["lon"] != "raw" and not st["released"] and sig >= R.s_release:
        st["released"] = True
    lat = h is not None and k < h + Q.LAT_N and c["g"] != 1.0
    lon = h is not None and c["lon"] != "raw" and not st["released"]
    rec = dict(tag=c["tag"], D=c["D"], src=c["src"], wait_s=st["wait_s"], h=h, sigma=round(sig, 4), d_route=round(dist, 4), v_meas=round(v0, 5), lat=lat, lon=lon,
               released=st["released"])
    sk = 1.0
    if lon:
        pr = Q.profile(R, sig, v0, R.v_e * (0.8 if c["lon"] == "ve08" else 1.0), 0.8 * R.v_e)
        ap = Q.plan_arc(poses)
        sk = Q.scale(pr["arc"], ap)
        rec.update(arc_prof=round(pr["arc"], 4), arc_plan=round(ap, 4), s_k=round(sk, 6), v_ent_prof=None if pr["v_ent"] is None else round(pr["v_ent"], 4))
    if lat or lon:
        poses = branch_poses(poses, c["g"] if lat else 1.0, sk)
    return poses, rec


if BRANCH or CELLS:
    _make_plan = P.S.make_cached_plan

    def _branch_plan(t0, anchor, poses):
        """pai_driver's make_cached_plan with the plan transformed inside the window set by drive (TL.branch, P3) or by the session's
        P4 cell (TL.p4); logs poses_branch."""
        st = getattr(TL, "p4", None)
        if st is not None:
            out, rec = p4_step(st, TL.k, t0, anchor, poses)
            TL.extra["p4"] = rec
            if rec["lat"] or rec["lon"]:
                poses = out
                TL.extra["poses_branch"] = np.round(poses, 4).tolist()
        elif getattr(TL, "branch", False):
            poses = branch_poses(poses, BRANCH["g"], BRANCH["s"])
            TL.extra["poses_branch"] = np.round(poses, 4).tolist()
        return _make_plan(t0, anchor, poses)

    P.S.make_cached_plan = _branch_plan


class Out:
    """drive.jsonl writer that merges the calling thread's log-replay fields into the record pai_driver writes."""

    def __init__(self, f):
        self.f = f

    def write(self, line: str):
        x = getattr(TL, "extra", None)
        return self.f.write(json.dumps({**json.loads(line), **x}) + "\n" if x else line)


def log_arrays(traj) -> tuple:
    """A common.Trajectory -> t_us (n,), positions (n, 3), unit x axes (n, 3) of the poses, in the message's frame."""
    a = np.array([[p.timestamp_us, p.pose.vec.x, p.pose.vec.y, p.pose.vec.z, p.pose.quat.w, p.pose.quat.x, p.pose.quat.y, p.pose.quat.z]
                  for p in traj.poses], np.float64).reshape(-1, 8)
    w, x, y, z = a[:, 4:].T
    return a[:, 0], a[:, 1:4], np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], 1)


def sample(t, xy, yaw, ts) -> tuple:
    """The path (t, xy, unwrapped yaw) at times ts -> (len(ts), 3) x, y, yaw and the number of extrapolated rows. Inside the log: linear.
    After its end: the arc of the last TAIL_US (constant speed, course rate = yaw rate)."""
    out = np.stack([np.interp(ts, t, xy[:, 0]), np.interp(ts, t, xy[:, 1]), np.interp(ts, t, yaw)], 1)
    ext = ts > t[-1]
    if ext.any():
        ta = max(t[-1] - TAIL_US, t[0])
        d = max((t[-1] - ta) * 1e-6, 1e-3)
        pa = np.array([np.interp(ta, t, xy[:, 0]), np.interp(ta, t, xy[:, 1]), np.interp(ta, t, yaw)])
        v, w = (xy[-1] - pa[:2]) / d, (yaw[-1] - pa[2]) / d
        speed, course, dt = float(np.hypot(*v)), float(np.arctan2(v[1], v[0])) + w * d / 2, (ts[ext] - t[-1]) * 1e-6
        if abs(w) > 1e-4:
            dx, dy = (np.sin(course + w * dt) - np.sin(course)) / w, (np.cos(course) - np.cos(course + w * dt)) / w
        else:
            dx, dy = np.cos(course) * dt, np.sin(course) * dt
        out[ext] = np.stack([xy[-1, 0] + speed * dx, xy[-1, 1] + speed * dy, yaw[-1] + w * dt], 1)
    return out, int(ext.sum())


class LogDriver(P.Driver):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.out = Out(self.out)
        self.p4n, self.p4ready = {}, {}

    def submit_image_observation(self, req, ctx):
        r = super().submit_image_observation(req, ctx)
        s = self.sessions.get(req.session_uuid)
        p4 = getattr(s, "p4", None) if s is not None else None
        if p4 is not None and p4["idx"] == 0 and s.count["images"] >= P4_WARM:
            self.p4ready[s.scene].set()
        return r

    def start_session(self, req, ctx):
        r = super().start_session(req, ctx)
        s = self.sessions.get(req.session_uuid)
        if CELLS and s is not None:                          # P4: the n-th session of a scene runs its cell n
            with self.lock:
                i = self.p4n.get(s.scene, 0)
                self.p4n[s.scene] = i + 1
            cl = CELLS["cells"].get(s.scene, [])
            if i >= len(cl):
                ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, f"ONP_CELLS: session {i} of {s.scene}, only {len(cl)} cells")
            ev = self.p4ready.setdefault(s.scene, threading.Event())
            t0 = time.time()
            if i > 0:
                ev.wait(600.0)                               # the renderer race: see `cells` in the module docstring
            s.p4 = dict(cell=cl[i], scene=s.scene, session=s, idx=i, h=None, hint=None, released=False, wait_s=round(time.time() - t0, 2))
        if s is not None and "rmem0" in AFTER and s.rt is not None:
            s.rt = MaskedMemory(s.rt)
        return r

    def submit_recording_ground_truth(self, req, ctx):
        s = self._s(req.session_uuid, ctx)
        s.gt = req.ground_truth                              # parsed in drive: one message per step, each the whole clip
        return common_pb2.Empty()

    def log_reference(self, s, now: int):
        """(Trajectory to return, drive.jsonl fields) from the ground-truth message of `now`; a string when it cannot be built."""
        g = getattr(s, "gt", None)
        with s.lock:
            pose = s.poses.get(now)
        if g is None or int(g.timestamp_us) != now:
            return f"no recording ground truth at Drive time {now} (latest {int(g.timestamp_us) if g is not None else None}): " \
                   "runtime.simulation_config.send_recording_ground_truth=true is required"
        if pose is None or len(g.trajectory.poses) < 2:
            return f"no egomotion pose at Drive time {now}" if pose is None else "recording ground truth with fewer than 2 poses"
        t, pos, ax = log_arrays(g.trajectory)
        yaw_r = np.unwrap(np.arctan2(ax[:, 1], ax[:, 0]))
        R, o = S.quat_R(pose.pose.quat), np.array([pose.pose.vec.x, pose.pose.vec.y, pose.pose.vec.z])
        pl, al = pos @ R.T + o, ax @ R.T                     # the log in the local frame the driver answers in
        n = int(round(HORIZON_S * 1e6 / P.STEP_US))
        ref, ext = sample(t, pl[:, :2], np.unwrap(np.arctan2(al[:, 1], al[:, 0])), now + P.STEP_US * np.arange(n + 1.0))
        traj = common_pb2.Trajectory(poses=[common_pb2.PoseAtTime(timestamp_us=now + P.STEP_US * i, pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=float(x), y=float(y), z=pose.pose.vec.z),
            quat=common_pb2.Quat(w=float(np.cos(a / 2)), x=0.0, y=0.0, z=float(np.sin(a / 2))))) for i, (x, y, a) in enumerate(ref)])
        rig, _ = sample(t, pos[:, :2], yaw_r, now + 500_000 * np.arange(9.0))        # label layout of `poses`: rig frame of `now`
        rig[:, 2] -= 2 * np.pi * np.round(rig[0, 2] / (2 * np.pi))
        return traj, {"ret": "log", "ret_n": n + 1, "ret_ext": ext, "log_t": round((now - t[0]) * 1e-6, 3), "log_now": rig[0].round(4).tolist(),
                      "log_poses": rig[1:].round(4).tolist(), "log_valid": int((now + 500_000 * np.arange(1, 9) <= t[-1]).sum())}

    def drive(self, req, ctx):
        s, now = self._s(req.session_uuid, ctx), int(req.time_now_us)
        p4 = getattr(s, "p4", None)
        sw = p4["cell"]["switch"] if p4 else SWITCH.get(s.scene, SWITCH_ALL)
        policy = sw is not None and s.count["drive"] >= sw  # pai_driver's count of this call, before it increments it
        ref = self.log_reference(s, now)
        s.count.setdefault("log_error", 0)
        if isinstance(ref, str):
            s.count["log_error"] += 1
            if not policy:
                ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, ref)
            ref = (None, {})
        x = {k: v for k, v in ref[1].items() if not (policy and k in ("ret_n", "ret_ext"))}
        if policy and AFTER:                                 # the control arms: from the switch on
            if isinstance(s.rt, MaskedMemory):
                s.rt.on = True
            if "vhzero" in AFTER and s.vh is None:
                s.vh = {"zero": True}
            x["after"] = sorted(AFTER)
        TL.extra = {**x, "switch": sw, "ret": "model" if policy else "log"}
        TL.p4, TL.k = p4, s.count["drive"]
        if BRANCH:                                           # P3: the plan transform from k_eff for t s
            k0 = max(sw, FGT_K) if policy else None
            TL.branch = policy and s.count["drive"] < k0 + int(round(10 * BRANCH["t"]))
            TL.extra["branch"] = {**BRANCH, "on": bool(TL.branch)}
        try:
            r = super().drive(req, ctx)                      # the model, its record and its dumps
        finally:
            TL.extra, TL.branch, TL.p4 = None, False, None
        return r if policy else egodriver_pb2.DriveResponse(trajectory=ref[0])

    def get_version(self, req, ctx):
        v = super().get_version(req, ctx)
        v.version_id += "-logreplay" + ("-switch" if SWITCH or SWITCH_ALL is not None else "") + "".join(f"-{a}" for a in sorted(AFTER)) \
            + (f"-branch-g{BRANCH['g']}s{BRANCH['s']}t{BRANCH['t']}" if BRANCH else "") + ("-p4cells" if CELLS else "")
        return v


if __name__ == "__main__":
    P.Driver = LogDriver                                     # pai_driver.main builds whatever its module calls Driver
    P.main()
