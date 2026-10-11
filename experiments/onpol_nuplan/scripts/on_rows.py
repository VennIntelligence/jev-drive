#!/usr/bin/env python3
"""ONPOL-NUPLAN rows (plans/2026-10-11-onpol-prereg.md, Data / Labels): finished on-policy rollouts (on_roll.sh) -> one training row per visited
decision, with the imitation target, the simulator's actor boxes and the road-and-lane raster in the row's own frame, gates R1 / R2, the row
report and a BEV preview. Nothing here changes a model; simulator state, map and logged path are labels only.

  geo   --name N --roll <stage dir> [...]     CPU, envs/navsim2 (nuPlan map API), in tmux:
          scripts/tmux_run.sh onpol-geo-N $DATA_DIR/envs/navsim2/bin/python experiments/onpol_nuplan/scripts/on_rows.py geo --name N --roll $DATA_DIR/runs/onpol_nuplan/roll/smoke10
  tok   --name N                              one GPU, envs/op-train, a pool job:
          .venv/bin/python -m jevdrive.cl submit --owner alpasim-onpol --name onpol-tok --vram 12 --cpu 16 --ram 32 --log-dir $DATA_DIR/runs/onpol_nuplan/rows/N/pool -- \
            $DATA_DIR/envs/op-train/bin/python experiments/onpol_nuplan/scripts/on_rows.py tok --name N
  bev   --name N [--n 16]                     CPU, envs/op-train (matplotlib) -> <rows dir>/bev/*.png (copied to experiments/onpol_nuplan/figs/)

Time and frame (established from data per rollout, proved by gate R2): the rollout of scene `<log>-<token>` starts 1.5 s before the token's
frame; simulation time 0 is found per rollout by matching the rollout's own logged ego track against the navsim log by time (sim_start), not
by frame index. Decisions k = 0..9 at simulation time 0.017 + 0.5 k s. The rollout's local frame is the nuPlan map frame translated to the
logged rear-axle position at simulation time 0 (no rotation). Decisions 0 and 1 are at log-replay states.
A row's own frame = the rear-axle pose the driver anchored its plan to (drive.jsonl `anchor`, the pose observation at t0): x forward, y left.

Output = the pool contract of on_train.py (plans/on_pool_format.md). Row order: (tag, scene, k). Only rows that pass the prereg's exclusions
are in the pool; every decision is listed in all.npz with its reason. P = $DATA_DIR/runs/op_parity/cache/on1_<N>:
  P/tab.npz           names (n,) str "on:<scene>:<seed>:<k>"; log (n,) str log unit (c0b_report.log_of); ego (n, 20) float32 the ego vector fed
                      to the adapter (from the dump, unrounded); fut (n, 8, 3) float32 imitation target x, y, yaw at 0.5 .. 4 s, row frame;
                      cam (n, 3) float32 camera position in the rear-axle frame; lht (n,) bool False; speed (n,) float32 m/s (the fed body
                      velocity); served (n, 8, 3) float32 the served plan; pose (n, 4, 3) history poses fed (drive.jsonl, 1e-4 m); cmd (n, 4, 4)
                      one-hot command of t0 repeated; acc (n, 4, 2) the fed acceleration repeated; vel (n, 4, 2) NaN except the t0 row (the
                      older velocities enter the frame warp only and are not logged). No `off`: every label is in the row frame.
  P@warp/front.npy    (n, 8, 32, 512) float16: Cinque vision tokens of the 8 dumped slot frames (encoder pair of slot i = (slot i - 1 or
                      zeros, slot i)) through pp_prep.encoder, the trainer's eager encoder path. The driver's own tokens (compiled pass) are
                      not in SH30_DUMP; gate R1 bounds the difference at the plan. Every row has 8 valid slots (backwarp cold start).
  P@warp/teacher.npz  out (n, 1086), plan (n, 33, 15), di, pi: shipped Cinque on these tokens (ot_rows.py make_teacher, same call)
  P/agent.npz         lib/agent_hinge.py label layout, K = 32: box (n, 9, 32, 5) float32 x, y, yaw (unwrapped over time), length, width of the
                      simulator's actors at t0 + 0, 0.5 .. 4 s in the row frame; valid (n, 9, 32); cls (n, 32) int8 (0 = vehicle, -1 = empty);
                      ok; tokens = names; log; classes. Pose: the simulator's own actor pose while the rollout lasts (to simulation time
                      5.017 s), after it the nuPlan log's box of the same track (the traffic replays the log); size: the simulator's box of
                      the actor (constant; the log's box size varies by frame). The 32 actors nearest to the row origin over the 9 times.
  P/road.npz          runs/body1/labels_road layout: sdf (n, 128, 96) float16 signed distance (m, + inside) to ROADBLOCK | ROADBLOCK_CONNECTOR |
                      INTERSECTION | LANE | LANE_CONNECTOR at the row pose (opb_labels.sdf_raster, unedited); tokens, log, pose_global, ok, x0, y0, res, layers
  P/drivable.npz      the same with ROADBLOCK | INTERSECTION | CARPARK_AREA (P2H10's raster; what bd4_train swaps in for rows off the road raster)
R = $DATA_DIR/runs/onpol_nuplan/rows/<N> (aligned with the tab; agent.npz / road.npz / drivable.npz here are links to the pool's):
  R/meta.npz          names, scene, log (nuPlan log file), logu (log unit), tag, seed, k, t_us (simulation time of t0), t_log_us (its log time),
                      pose_map (3) rear axle in the nuPlan map frame, anchor (3) the same in the rollout frame, speed, s0, e0 (m, left +),
                      psi_e (rad, unclipped), lead (m, + ahead of the log's position at the same time), Lr, D (8) log travel, path_ext (target
                      beyond the log's last frame, on the 40 m straight continuation), n_act, d_anchor (m, driver anchor against the
                      simulator's ego), n_keys, poses (8, 3) served plan, mu (33, 15) plan mean, gpath (86, 2) logged path from s0 - 15 m to
                      s0 + 70 m, logfut (9, 3) the log's own poses at t0 + 0 .. 4 s, dump (file), run
  R/all.npz           every decision of every rollout: scene, tag, seed, k, t_us, excl ("" = kept), row (index in the tab or -1), e0, lead
  R/active.npz        per row, the stored own plan: agent / road / drivable hinge value, road_off_t0, imit_ade
  R/r2.json, r2.md, r1.json, r1.md, report.json, report.md, bev/
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/scripts"), str(_R / "experiments/alpasim/lib"), str(_R / "experiments/op_probe/scripts")]
import argparse, json, os, pickle  # noqa: E401,E402

import numpy as np  # noqa: E402

DATA = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
FAM = "on1"
CR = DATA / "runs/op_parity/cache"
ROWS = DATA / "runs/onpol_nuplan/rows"
SPLIT = "alpasim_nuplan/onpol-train"
FRAME_OFF, STEP_US, NT, K = -3, 500_000, 9, 32
CEN, CORRIDOR, EXT = 1.461, 4.0, 40.0                          # rear axle -> box centre (m); the scorer's corridor; straight continuation of the logged path
PSI_MAX, LR_MIN, LR_T = np.radians(30.0), 4.0, 1.0
ROAD_LAYERS = ("ROADBLOCK", "ROADBLOCK_CONNECTOR", "INTERSECTION", "LANE", "LANE_CONNECTOR")
NAV_LAYERS = ("ROADBLOCK", "INTERSECTION", "CARPARK_AREA")
TOL_M, R2_EGO, R2_ACT = 0.03, 0.3, 0.5
REASONS = ("abort", "no_sim", "no_dump", "overlap", "corridor", "log_future", "path_end", "raster")
_C = {}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def into(frame, P):
    """Poses (n, 3) x, y, yaw of some frame -> the frame of the pose `frame` (x, y, yaw) given in it."""
    P = np.atleast_2d(np.asarray(P, np.float64))
    return np.c_[(P[:, :2] - frame[:2]) @ rot(frame[2]), P[:, 2] - frame[2]]


# ---------------------------------------------------------------- per-process readers
def _sim(run):
    """<run dir>/onpol_sim.pkl (c1_extract.one_log per scene; written under numpy 2)."""
    if ("sim", run) not in _C:
        import col1_lib as CL
        f = _pl.Path(run) / "onpol_sim.pkl"
        _C["sim", run] = CL.load(f) if f.exists() else {}
    return _C["sim", run]


def _drv(run):
    """drive.jsonl of a run -> {scene: {session: {"n": driver session index, "rec": {k: drive record}}}}."""
    if ("drv", run) not in _C:
        S, out = {}, {}
        for line in open(_pl.Path(run) / "driver-logs/drive.jsonl"):
            r = json.loads(line)
            if r["kind"] == "start":
                S[r["session"]] = out.setdefault(r["scene"], {}).setdefault(r["session"], dict(n=r["n"], rec={}))
            elif r["kind"] == "drive":
                S[r["session"]]["rec"][r["k"]] = r
        _C["drv", run] = out
    return _C["drv", run]


def _log(log):
    """A navsim log -> dict(tok, T (us), P (n, 3) rear-axle poses in the nuPlan map frame, loc, fr)."""
    if ("log", log) not in _C:
        qy = lambda q: np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))  # noqa: E731
        fr = sorted(pickle.load(open(DATA / "datasets/navsim/navsim_logs/test" / f"{log}.pkl", "rb")), key=lambda f: f["timestamp"])
        P = np.array([[*f["ego2global_translation"][:2], qy(f["ego2global_rotation"])] for f in fr])
        P[:, 2] = np.unwrap(P[:, 2])
        _C["log", log] = dict(tok=[f["token"] for f in fr], T=np.array([f["timestamp"] for f in fr], np.int64), P=P, loc=fr[0]["map_location"], fr=fr, ann={})
    return _C["log", log]


def _ann(L, f):
    """Frame f of a log -> {track token: (x, y, yaw) box centre in the map frame}."""
    if f not in L["ann"]:
        a, p = L["fr"][f]["anns"], L["P"][f]
        b = np.asarray(a["gt_boxes"], np.float64).reshape(-1, 7)
        g = np.c_[p[:2] + b[:, :2] @ rot(p[2]).T, b[:, 6] + p[2]]
        L["ann"][f] = dict(zip(np.asarray(a["track_tokens"]).tolist(), g))
    return L["ann"][f]


def log_box(L, tk, t):
    """The log's box centre of track tk at log time t (us), linear between the two neighbouring 2 Hz frames (both must hold the track) -> (3,) or None."""
    T = L["T"]
    f = int(np.searchsorted(T, t, "right")) - 1
    if f < 0 or f >= len(T):
        return None
    a = _ann(L, f).get(tk)
    if a is None:
        return None
    if t - T[f] <= 2000:
        return a
    if f + 1 >= len(T) or T[f + 1] - T[f] > 600_000 or (b := _ann(L, f + 1).get(tk)) is None:
        return None
    w = (t - T[f]) / (T[f + 1] - T[f])
    return np.r_[a[:2] + w * (b[:2] - a[:2]), a[2] + w * wrap(b[2] - a[2])]


# ---------------------------------------------------------------- the logged path G
class Path:
    """The logged rear-axle path of a log from frame `a` on: cubic Hermite through the 2 Hz poses (tangent = the logged heading; frames that
    moved less than 5 cm are merged), sampled every 0.1 m, continued EXT m straight beyond the last frame. xy (m, 2), s (m,) arc, h (m,)
    heading of the segment that starts at each vertex; SF (frames,) arc of every frame, TF their times; s_end = arc of the last frame."""

    def __init__(self, P, T):
        keep = [0]
        for i in range(1, len(P)):
            if np.hypot(*(P[i, :2] - P[keep[-1], :2])) > 0.05:
                keep.append(i)
        out, at = [P[:1, :2]], [0]
        for i, j in zip(keep[:-1], keep[1:]):
            a, b = P[i], P[j]
            d = b[:2] - a[:2]
            Lc = np.hypot(*d)
            u = np.linspace(0, 1, max(int(np.ceil(Lc / 0.1)), 1) + 1)[1:, None]
            ta, tb = (Lc * np.array([np.cos(q[2]), np.sin(q[2])]) for q in (a, b))
            if ta @ d <= 0 or tb @ d <= 0:                                       # reversing or a heading across the chord: the chord itself
                seg = a[:2] + u * d
            else:
                seg = (2 * u**3 - 3 * u**2 + 1) * a[:2] + (u**3 - 2 * u**2 + u) * ta + (-2 * u**3 + 3 * u**2) * b[:2] + (u**3 - u**2) * tb
            out.append(seg)
            at.append(at[-1] + len(seg))
        xy = np.concatenate(out)
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        self.s_end = float(s[-1])
        yaw = P[keep[-1], 2]
        self.xy = np.r_[xy, xy[-1:] + EXT * np.array([[np.cos(yaw), np.sin(yaw)]])]
        self.s = np.r_[s, s[-1] + EXT]
        d = np.diff(self.xy, axis=0)
        self.h = np.unwrap(np.r_[np.arctan2(d[:, 1], d[:, 0]), yaw])
        self.h[-1] = self.h[-2]
        SF = np.zeros(len(P))
        SF[keep] = s[at]
        for i in range(1, len(P)):                                               # a merged frame takes the arc of the frame it was merged into
            if i not in set(keep):
                SF[i] = SF[i - 1]
        self.SF, self.TF = SF, np.asarray(T, np.float64)

    def s_log(self, t):
        """Arc position of the log at log time t (us)."""
        return np.interp(t, self.TF, self.SF)

    def at(self, s):
        """Points (n, 2) and segment headings (n,) at arcs s (clamped to the path)."""
        s = np.clip(np.atleast_1d(s), 0, self.s[-1])
        i = np.clip(np.searchsorted(self.s, s, "right") - 1, 0, len(self.s) - 2)
        w = ((s - self.s[i]) / np.maximum(self.s[i + 1] - self.s[i], 1e-9))[:, None]
        return self.xy[i] + w * (self.xy[i + 1] - self.xy[i]), self.h[i]

    def project(self, p, lo, hi):
        """Point p (2,) -> (arc of the foot, signed lateral offset, left +, heading of the path at the foot), nearest segment with arc in [lo, hi]."""
        i0, i1 = np.searchsorted(self.s, [lo, hi])
        i0, i1 = max(i0 - 1, 0), min(max(i1, i0 + 1), len(self.s) - 1)
        a, d = self.xy[i0:i1], self.xy[i0 + 1:i1 + 1] - self.xy[i0:i1]
        L2 = np.maximum((d * d).sum(1), 1e-12)
        w = np.clip(((p - a) * d).sum(1) / L2, 0, 1)
        f = a + w[:, None] * d
        j = int(np.hypot(*(p - f).T).argmin())
        h = self.h[i0 + j]
        return float(self.s[i0 + j] + w[j] * np.sqrt(L2[j])), float((p - f[j]) @ np.array([-np.sin(h), np.cos(h)])), float(h)


def imitation(G: Path, s0, e0, psi_e, D):
    """The prereg's imitation target: 8 poses (map frame) = G at s0 + D_k moved by e(D_k) along G's left normal, e(d) = (e0 + (tan(psi_e) +
    e0 / Lr) d) exp(-d / Lr), Lr = max(4 m, 1 s x D_8 / 4 s), psi_e clipped to +-30 deg; heading = the tangent of that curve (central
    difference over 0.1 m). -> (8, 3), Lr."""
    Lr = max(LR_MIN, LR_T * D[-1] / 4.0)
    a = np.tan(np.clip(psi_e, -PSI_MAX, PSI_MAX)) + e0 / Lr

    def c(d):
        xy, h = G.at(s0 + d)
        return xy + ((e0 + a * d) * np.exp(-d / Lr))[:, None] * np.c_[-np.sin(h), np.cos(h)]
    D = np.asarray(D, np.float64)
    t = c(D + 0.05) - c(D - 0.05)
    return np.c_[c(D), np.arctan2(t[:, 1], t[:, 0])], Lr


def box_poly(c, L, W):
    """Centre poses (n, 3), sizes (n,) -> shapely polygons (n,)."""
    import shapely
    c = np.atleast_2d(c)
    hl, hw = np.broadcast_to(np.asarray(L, float) / 2, len(c))[:, None], np.broadcast_to(np.asarray(W, float) / 2, len(c))[:, None]
    lx, ly = np.array([1, 1, -1, -1.0]) * hl, np.array([1, -1, -1, 1.0]) * hw
    cs, sn = np.cos(c[:, 2:3]), np.sin(c[:, 2:3])
    return shapely.polygons(np.stack([c[:, 0:1] + lx * cs - ly * sn, c[:, 1:2] + lx * sn + ly * cs], -1))


# ---------------------------------------------------------------- one rollout -> rows
def sim_start(L, i, rig) -> tuple:
    """Log time (us) of simulation time 0, from data: the candidate (a log frame 6 before .. 2 after the token's, or the token's time - 1.5 s)
    whose log track, translated to its first pose, lies closest to the rollout's own logged ego track `rig` (t_us, x, y, yaw; local, 10 Hz).
    Poses are compared by time, at 0, 0.5 .. 5 s, never by frame index (some logs have a 1 s gap). -> (T0, its largest distance in m)."""
    T, P = L["T"], L["P"]
    tau = np.arange(0, rig[-1, 0] + 1, STEP_US, dtype=np.float64)
    loc = np.c_[np.interp(tau, rig[:, 0], rig[:, 1]), np.interp(tau, rig[:, 0], rig[:, 2])] - rig[0, 1:3]
    best = (None, np.inf)
    for T0 in [int(T[j]) for j in range(max(i - 6, 0), min(i + 3, len(T)))] + [int(T[i]) + FRAME_OFF * STEP_US]:
        if T0 < T[0] or T0 + tau[-1] > T[-1]:
            continue
        g = np.c_[np.interp(T0 + tau, T, P[:, 0]), np.interp(T0 + tau, T, P[:, 1])]
        e = float(np.hypot(*(g - g[0] - loc).T).max())
        if e < best[1] - 1e-9:
            best = (T0, e)
    return best


def geo_one(job):
    """One rollout (tag, run dir, scene) -> dict(rows=[...], r2={...}). Every decision gets a row dict; excluded ones carry `excl`."""
    import col1_lib as CL
    import opb_labels as OL
    import shapely
    tag, run, scene = job
    seed, (log, tok) = int(tag.rsplit("-s", 1)[1]), scene.rsplit("-", 1)
    base = dict(scene=scene, log=log, tag=tag, seed=seed, run=run)
    o, sess = _sim(run).get(scene), _drv(run).get(scene, {})
    if o is None or "EGO" not in o.get("actors", {}):
        recs = max(sess.values(), key=lambda v: v["n"])["rec"] if sess else {}
        return dict(rows=[base | dict(k=k, t_us=r["t0"], excl="no_sim") for k, r in sorted(recs.items())], r2=None)
    drv = sess.get(o.get("session")) or (max(sess.values(), key=lambda v: v["n"]) if sess else dict(n=-1, rec={}))
    L = _log(log)
    i = L["tok"].index(tok)
    ego_c = o["actors"]["EGO"]
    ego = np.c_[ego_c[:, 0], CL.to_rig(ego_c, CEN)]                             # the simulator's rear axle, local frame
    g = o["logged"][0]["traj"]
    rig = np.c_[g[:, 0], CL.to_rig(g, CEN)]                                     # the rollout's logged ego, local frame, 10 Hz
    logp = lambda t: np.array([np.interp(t, L["T"], L["P"][:, q]) for q in range(3)])  # noqa: E731  the log's pose at log time t
    T0, e_best = sim_start(L, i, rig)                                           # log time of simulation time 0
    if T0 is None:
        recs = drv["rec"]
        return dict(rows=[base | dict(k=k, t_us=r["t0"], excl="no_sim") for k, r in sorted(recs.items())], r2=None)
    O = logp(T0)[:2] - rig[0, 1:3]                                              # map position of the local origin
    a = max(int(np.searchsorted(L["T"], T0)) - 10, 0)
    G = Path(L["P"][a:], L["T"][a:])
    t_end, t_sim_end = L["T"][-1], ego[-1, 0]
    to_map = lambda p: np.r_[p[:2] + O, p[2]]                                   # noqa: E731
    ids = [q for q in o["actors"] if q != "EGO"]
    size = np.array([o["size"][q][:2] for q in ids], np.float64).reshape(-1, 2)
    Le, We = o["size"]["EGO"][:2]

    def actor(q, t):
        """Actor q at simulation time t (us), local frame: the simulator's pose while it has one, the log's box after the rollout."""
        tr = o["actors"][q]
        if tr[0, 0] - 1000 <= t <= tr[-1, 0] + 1000:
            n = int(np.abs(tr[:, 0] - t).argmin())
            if abs(tr[n, 0] - t) <= 1000:
                return tr[n, 1:]
            m = int(np.searchsorted(tr[:, 0], t))
            return CL.interp(tr, t)[0] if 0 < m < len(tr) and tr[m, 0] - tr[m - 1, 0] <= 600_000 else None
        if t > t_sim_end and tr[-1, 0] >= t_sim_end - 1000:                      # an actor the rollout still had at its end
            b = log_box(L, q, T0 + t)
            return None if b is None else np.r_[b[:2] - O, b[2]]
        return None

    # ---- gate R2 material: ego at the log-replay decisions, simulator actors against the log's boxes, the dense path against the rollout's 10 Hz logged track
    r2 = dict(scene=scene, tag=tag, t0_rel=round((T0 - int(L["T"][i])) * 1e-6, 2), frame_err=e_best, ego=[], act=[], act_yaw=[], n_act=len(ids))
    for t in o["actor_t"]:
        for q in ids:
            tr = o["actors"][q]
            n = int(np.abs(tr[:, 0] - t).argmin())
            b = log_box(L, q, T0 + t)
            if b is not None and abs(tr[n, 0] - t) <= 1000:
                r2["act"].append(float(np.hypot(*(tr[n, 1:3] + O - b[:2]))))
                r2["act_yaw"].append(float(abs(wrap(tr[n, 3] - b[2]))))
    pe = [abs(G.project(q[1:3] + O, G.s_log(T0 + q[0]) - 30, G.s_log(T0 + q[0]) + 30)[1]) for q in rig]
    r2["path_dense"] = float(max(pe))
    rows = []
    reqs = sorted({int(d["now"]) for d in o["drive"]})
    for k, now in enumerate(reqs):
        r = drv["rec"].get(k)
        row = base | dict(k=k, t_us=now, excl="")
        rows.append(row)
        if r is None or r["now"] != now:
            row["excl"] = "abort"
            continue
        t = int(r["t0"])
        anchor = np.array(r["anchor"], np.float64)
        pm = to_map(anchor)
        se = CL.interp(ego, t)[0]                                               # the simulator's rear axle at t0
        lp = logp(T0 + t)
        if k < 2:
            r2["ego"].append(float(np.hypot(*(se[:2] + O - lp[:2]))))
        sl = float(G.s_log(T0 + t))
        s0, e0, hG = G.project(pm[:2], sl - 60, sl + 80)
        psi = float(wrap(pm[2] - hG))
        dump = _pl.Path(run) / "driver-logs/dump" / f"s{drv['n']:02d}_k{k}.npz"
        row |= dict(t_us=t, t_log_us=int(T0 + t), anchor=anchor, pose_map=pm, s0=s0, e0=e0, psi_e=psi, lead=s0 - sl, d_anchor=float(np.hypot(*(se[:2] - anchor[:2]))),
                    hist=np.array(r["hist"], np.float32), cmd=int(r["cmd"]), dump=str(dump), n_keys=int(r["n_keys"]))
        if not dump.exists():
            row["excl"] = "no_dump"
            continue
        with np.load(dump) as z:
            assert str(z["scene"]) == scene and bool(z["valid"].all()), f"{dump}: scene {z['scene']}, valid {z['valid']}"
            row |= dict(ego=z["ego"].astype(np.float32), poses=z["poses"].astype(np.float64), mu=z["mu"].astype(np.float32), cam=z["cam_t"].astype(np.float32))
        assert np.abs(row["poses"] - np.array(r["poses"])).max() < 1e-3 and np.abs(row["ego"] - np.array(r["ego"])).max() < 1e-4, f"{dump} is not decision {k} of {scene}"
        # actors at t0 + 0, 0.5 .. 4 s -> the row frame; the K nearest over the 9 times
        A = np.full((NT, len(ids), 3), np.nan)
        for j in range(NT):
            for n, q in enumerate(ids):
                b = actor(q, t + j * STEP_US)
                if b is not None:
                    A[j, n] = into(anchor, b)[0]
        val = ~np.isnan(A[..., 0])
        dmin = np.where(val, np.hypot(A[..., 0], A[..., 1]), np.inf).min(0) if len(ids) else np.zeros(0)
        sel = np.argsort(dmin, kind="stable")[:K]
        sel = sel[np.isfinite(dmin[sel])]
        box, bval, cls = np.zeros((NT, K, 5), np.float32), np.zeros((NT, K), bool), np.full(K, -1, np.int8)
        for c, n in enumerate(sel):
            v = np.flatnonzero(val[:, n])
            A[v, n, 2] = np.unwrap(wrap(A[v, n, 2]))
            box[v, c, :3], box[v, c, 3:], bval[v, c], cls[c] = A[v, n], size[n], True, 0
        row |= dict(box=box, bval=bval, cls=cls, n_act=int(np.isfinite(dmin).sum()))
        # exclusion (b): the simulator's ego box against the actor boxes at t0; the rear axle against the logged path
        now_ok = [n for n, q in enumerate(ids) if actor(q, t) is not None]
        if now_ok and shapely.intersects(box_poly(np.r_[se[:2] + CEN * np.array([np.cos(se[2]), np.sin(se[2])]), se[2]], Le, We)[0],
                                         box_poly(np.array([actor(ids[n], t) for n in now_ok]), size[now_ok, 0], size[now_ok, 1])).any():
            row["excl"] = "overlap"
        elif abs(e0) > CORRIDOR:
            row["excl"] = "corridor"
        elif T0 + t + 8 * STEP_US > t_end:
            row["excl"] = "log_future"
        D = G.s_log(T0 + t + STEP_US * np.arange(1, 9)) - sl
        if not row["excl"] and s0 + D[-1] > G.s[-1] - 0.1:
            row["excl"] = "path_end"
        fut, Lr = imitation(G, s0, e0, psi, D)
        gp = G.at(s0 + np.arange(-15.0, 71.0))[0]
        lf = np.array([logp(T0 + t + j * STEP_US) if T0 + t + j * STEP_US <= t_end else np.full(3, np.nan) for j in range(NT)])
        row |= dict(fut=into(pm, fut).astype(np.float32), Lr=Lr, D=D.astype(np.float32), path_ext=bool(s0 + D[-1] > G.s_end),
                    gpath=into(pm, np.c_[gp, np.zeros(len(gp))])[:, :2].astype(np.float32), logfut=into(pm, lf).astype(np.float32))
        row["fut"][:, 2] = wrap(row["fut"][:, 2])
        try:
            m = OL._map(L["loc"])
            for name, layers in (("sdf_road", ROAD_LAYERS), ("sdf_nav", NAV_LAYERS)):
                OL.LAYERS = layers
                row[name] = np.clip(OL.sdf_raster(m, pm), -20, 20).astype(np.float16)
        except Exception as e:  # noqa: BLE001
            row["excl"] = row["excl"] or "raster"
            row["err"] = f"{type(e).__name__}: {e}"
    return dict(rows=rows, r2=r2)


def rollouts(stages) -> list:
    """Stage dirs of on_roll.sh -> [(tag, run dir, scene)]: every scene of every label once, from the first run dir whose summary holds it."""
    out = []
    for st in stages:
        for label, dirs in json.loads((_pl.Path(st) / "manifest.json").read_text()).items():
            seen = set()
            for D in dirs:
                for r in json.loads((_pl.Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]:
                    if r["clipgt_id"] not in seen:
                        seen.add(r["clipgt_id"])
                        out.append((label.split("@")[0], D, r["clipgt_id"]))
    return sorted(out, key=lambda x: (x[0], x[2]))


def md_table(head, rows) -> list:
    return ["| " + " | ".join(head) + " |", "|" + "|".join([":--"] + ["--:"] * (len(head) - 1)) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def cmd_geo(a):
    import c0b_report as CB
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = ROWS / a.name
    with Run("onpol_nuplan", f"rows-geo-{a.name}", config=vars(a)) as run:
        tr = splits.load(SPLIT)
        run.use_split(tr)
        jobs = rollouts(a.roll)
        assert jobs and tr.mask(np.array([j[2] for j in jobs])).all(), "a rollout of a scene outside the train split"
        assert len({(j[0], j[2]) for j in jobs}) == len(jobs)
        run.info(f"{a.name}: {len(jobs)} rollouts of {len({j[2] for j in jobs})} scenes, tags {sorted({j[0] for j in jobs})}")
        res = par.pmap(geo_one, jobs, run=run, workers=a.workers or None, desc="rollouts")
        res.raise_if_failed()
        rows = [r for v in res.values for r in v["rows"]]
        keep = [r for r in rows if not r["excl"]]
        n = len(keep)
        assert n, "no row passed the exclusions"
        for i, r in enumerate(keep):
            r["row"] = i
        st = lambda key, dt=None, src=keep: np.array([r[key] for r in src], dt) if dt else np.stack([r[key] for r in src])  # noqa: E731
        names = np.array([f"on:{r['scene']}:{r['seed']}:{r['k']}" for r in keep])
        assert len(set(names.tolist())) == n
        logs, ego, pool = np.array([CB.log_of(r["scene"]) for r in keep]), st("ego"), CR / f"{FAM}_{a.name}"
        pool.mkdir(parents=True, exist_ok=True), out.mkdir(parents=True, exist_ok=True)
        vel = np.full((n, 4, 2), np.nan, np.float32)
        vel[:, 3] = ego[:, 4:6] * 10.0
        np.savez(pool / "tab.npz", names=names, log=logs, ego=ego, pose=st("hist"), vel=vel, acc=np.repeat(ego[:, None, 6:8] * 3.0, 4, 1),
                 cmd=np.repeat(np.eye(4, dtype=np.float32)[st("cmd", int)][:, None], 4, 1), fut=st("fut"), cam=st("cam"), lht=np.zeros(n, bool),
                 speed=np.hypot(ego[:, 4], ego[:, 5]) * 10.0, served=st("poses").astype(np.float32))
        np.savez(pool / "agent.npz", tokens=names, log=logs, box=st("box"), valid=st("bval"), cls=st("cls"), ok=np.ones(n, bool),
                 classes=np.array(["vehicle", "generic_object", "pedestrian", "bicycle"]))
        for f, key, layers in (("road", "sdf_road", ROAD_LAYERS), ("drivable", "sdf_nav", NAV_LAYERS)):
            np.savez(pool / f"{f}.npz", tokens=names, log=logs, pose_global=st("pose_map"), sdf=st(key), ok=np.ones(n, bool), x0=-8.0, y0=-24.0, res=0.5, layers=np.array(layers))
        for f in ("agent", "road", "drivable"):                                              # the label files live in the pool dir; the rows dir links to them
            (out / f"{f}.npz").unlink(missing_ok=True)
            (out / f"{f}.npz").symlink_to(pool / f"{f}.npz")
        np.savez(out / "meta.npz", names=names, scene=st("scene", str), log=st("log", str), logu=logs, tag=st("tag", str),
                 seed=st("seed", int), k=st("k", int), t_us=st("t_us", np.int64), t_log_us=st("t_log_us", np.int64), pose_map=st("pose_map"), anchor=st("anchor"), speed=np.hypot(ego[:, 4], ego[:, 5]) * 10.0,
                 s0=st("s0", float), e0=st("e0", float), psi_e=st("psi_e", float), lead=st("lead", float), Lr=st("Lr", float), D=st("D"), path_ext=st("path_ext", bool),
                 n_act=st("n_act", int), d_anchor=st("d_anchor", float), n_keys=st("n_keys", int), poses=st("poses"), mu=st("mu"), gpath=st("gpath"), logfut=st("logfut"),
                 dump=st("dump", str), run=st("run", str))
        g = lambda key, r: r.get(key, np.nan)  # noqa: E731
        np.savez(out / "all.npz", scene=st("scene", str, rows), tag=st("tag", str, rows), seed=st("seed", int, rows), k=st("k", int, rows), t_us=st("t_us", np.int64, rows),
                 excl=st("excl", str, rows), row=np.array([r.get("row", -1) for r in rows]), e0=np.array([g("e0", r) for r in rows]), lead=np.array([g("lead", r) for r in rows]))
        # ---- gate R2
        R = [v["r2"] for v in res.values if v["r2"]]
        eg, ac, ay = (np.array([x for r in R for x in r[q]]) for q in ("ego", "act", "act_yaw"))
        r2 = dict(rollouts=len(jobs), with_sim=len(R), t0_rel_s={str(d): int(sum(r["t0_rel"] == d for r in R)) for d in sorted({r["t0_rel"] for r in R})},
                  frame_err_max_m=max(r["frame_err"] for r in R), ego_n=len(eg), ego_max_m=float(eg.max()), ego_share_within=float((eg <= R2_EGO).mean()),
                  act_n=len(ac), act_median_m=float(np.median(ac)), act_p99_m=float(np.percentile(ac, 99)), act_max_m=float(ac.max()),
                  act_share_within=float((ac <= R2_ACT).mean()), act_yaw_p99_deg=float(np.degrees(np.percentile(ay, 99))), act_yaw_max_deg=float(np.degrees(ay.max())),
                  path_dense_max_m=max(r["path_dense"] for r in R), anchor_vs_sim_max_m=float(st("d_anchor", float).max()), tol_ego_m=R2_EGO, tol_act_m=R2_ACT)
        # at the log-replay decisions the imitation target must be the log's own future (label sanity, not part of the registered gate)
        rep = [r for r in keep if r["k"] < 2]
        r2["imit_at_replay_max_m"] = float(max(np.hypot(*(r["fut"][:, :2] - r["logfut"][1:, :2]).T).max() for r in rep)) if rep else None
        r2["pass"] = bool(len(R) and r2["ego_max_m"] <= R2_EGO and r2["act_max_m"] <= R2_ACT)
        (out / "r2.json").write_text(json.dumps(r2, indent=1) + "\n")
        (out / "r2.md").write_text("\n".join([f"# Gate R2 (frames), rows `{a.name}`: {'PASS' if r2['pass'] else 'FAIL'}", ""] + md_table(["quantity", "value", "limit"], [
            ("rollouts with simulator state", f"{len(R)} of {len(jobs)}", ""), ("simulation time 0 against the token's log time (s): rollouts per value", r2["t0_rel_s"], ""),
            ("the rollout's own logged ego track against the navsim log from that time, max over 0 .. 5 s (m)", f"{r2['frame_err_max_m']:.4f}", ""),
            (f"simulator ego against the log's pose at decisions 0 and 1, max of {len(eg)} (m)", f"{r2['ego_max_m']:.4f}", R2_EGO),
            (f"simulator actor centres against the log's boxes of the same tracks, {len(ac)} samples: median / p99 / max (m)",
             f"{r2['act_median_m']:.3f} / {r2['act_p99_m']:.3f} / {r2['act_max_m']:.3f}", R2_ACT),
            ("share of actor samples within the limit", f"{r2['act_share_within']:.4f}", ""),
            ("actor yaw difference, p99 / max (deg)", f"{r2['act_yaw_p99_deg']:.2f} / {r2['act_yaw_max_deg']:.2f}", ""),
            ("dense logged path (Hermite through 2 Hz) against the rollout's 10 Hz logged track, max lateral (m)", f"{r2['path_dense_max_m']:.4f}", ""),
            ("driver anchor against the simulator's rear axle at t0, max (m)", f"{r2['anchor_vs_sim_max_m']:.4f}", ""),
            ("imitation target against the log's own future at decisions 0 and 1, max (m)", f"{r2['imit_at_replay_max_m']:.3f}" if rep else "n/a", "")]) + [""]))
        run.summary.update(rows=n, decisions=len(rows), excluded={q: int(sum(r["excl"] == q for r in rows)) for q in REASONS if any(r["excl"] == q for r in rows)}, r2=r2)
        run.info(f"{n} rows of {len(rows)} decisions -> {pool}, {out}; R2 {'PASS' if r2['pass'] else 'FAIL'}: {json.dumps(r2)}")


# ---------------------------------------------------------------- tokens, teacher, gate R1, report
def cmd_tok(a):
    import torch
    import pp_prep as PP
    import pp_train as T
    from concurrent.futures import ThreadPoolExecutor
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive import cache
    from jevdrive import navsim_zs as Z
    from jevdrive import op_adapt as A
    from jevdrive import op_interp as I
    from jevdrive.data import splits
    from jevdrive.run import Run
    out, root, froot = ROWS / a.name, CR / f"{FAM}_{a.name}", CR / f"{FAM}_{a.name}@warp"
    dev = torch.device("cuda")
    with Run("onpol_nuplan", f"rows-tok-{a.name}", config=vars(a)) as run:
        run.use_split(splits.load(SPLIT))
        tab, M, allr = dict(np.load(root / "tab.npz")), dict(np.load(out / "meta.npz")), dict(np.load(out / "all.npz"))
        n = len(tab["names"])
        assert (M["names"] == tab["names"]).all()
        froot.mkdir(parents=True, exist_ok=True)
        net, enc = PP.encoder(dev)
        kb = dict(names=cache.key(params=dict(n=tab["names"].tolist(), d=M["dump"].tolist())))

        def make_front():
            def load(rows):
                cur = np.stack([np.load(M["dump"][i])["cur"] for i in rows])                        # (b, 8, 2, 6, 128, 256) as fed
                return rows, np.concatenate([np.zeros_like(cur[:, :1]), cur[:, :-1]], 1).reshape(-1, *cur.shape[2:]), cur.reshape(-1, *cur.shape[2:])
            fr = np.zeros((n, 8) + A.H_SHAPE, np.float16)
            with ThreadPoolExecutor(16) as ex:
                for rows, prev, cur in run.tqdm(PP._bounded(ex, load, [np.arange(i, min(i + 32, n)) for i in range(0, n, 32)], 8), total=(n + 31) // 32, desc="front"):
                    fr[rows] = enc(prev, cur).reshape(len(rows), 8, *A.H_SHAPE)
            return fr
        front = cache.cached(froot / "front.npy", cache.key(params=kb, code=[PP.encoder]), make_front, force=a.force)
        tc = torch.tensor([[1.0, 0.0]], device=dev).repeat(n, 1)

        def make_teacher():                                                                     # ot_rows.cmd_prep make_teacher, same call
            di, pi = A.distill_index(net.slices), A.plan_index(net.slices)
            o_, plan = np.zeros((n, len(di)), np.float32), np.zeros((n, 33, 15), np.float32)
            with torch.no_grad():
                for i in range(0, n, 256):
                    r = slice(i, min(i + 256, n))
                    Hh = torch.from_numpy(np.ascontiguousarray(front[r])).to(dev)
                    Hh = torch.cat([torch.zeros_like(Hh[:, :1]), Hh], 1)
                    valid = torch.ones(Hh.shape[:2], dtype=torch.bool, device=dev)
                    valid[:, 0] = False
                    o = A._policy(net, Hh, (0.275, 0.525), tc[r], valid)["outputs"].float()
                    o_[r], plan[r] = o[:, di].cpu().numpy(), o[:, pi].cpu().numpy().reshape(-1, 33, 15)
            return dict(out=o_, plan=plan, di=di, pi=pi)
        tz = cache.cached(froot / "teacher.npz", cache.key(params=kb, inputs=[froot / "front.npy"]), make_teacher, force=a.force)
        # ---- gate R1: the trainer's forward of the serving checkpoint on the stored row against the served plan
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
        pi = torch.as_tensor(tz["pi"], device=dev)
        ego, cam_x = torch.from_numpy(tab["ego"]).to(dev), torch.from_numpy(tab["cam"][:, 0].astype(np.float32)).to(dev)
        tags = sorted(set(M["tag"].tolist()))

        @torch.no_grad()
        def diff(model, rows):
            """Per row, trainer forward against served: max |difference| over (0) the first 15 plan points (x, y); (1) the 8 poses (x, y), the
            trainer's plan through the serving export (op_interp.to_rear, as sh30_core.Core.plan); (2) the same over the poses served within
            32 m (beyond it one fp16 ulp of a plan point is 0.031 m); (3) the 8 poses through the trainer's own formula (pp_train.rear, which
            interpolates before the lever arm and drops the camera's lateral offset); (4) pp_train.rear of the served mu against the served poses."""
            out = [[] for _ in range(5)]
            for i in range(0, len(rows), 128):
                r = rows[i:i + 128]
                rt = torch.as_tensor(r, device=dev)
                p = model(torch.from_numpy(np.ascontiguousarray(front[r])).to(dev), ego[rt], tc[rt]).float()[:, pi].view(-1, 33, 15)
                mu, sp = torch.from_numpy(M["mu"][r]).to(dev), torch.from_numpy(M["poses"][r].astype(np.float32)).to(dev)
                pn = p.cpu().numpy()
                ex = np.stack([I.to_rear(q[:, 0:3], q[:, 11], I.T_IDXS, c[:2], Z.T_OUT, "lever") for q, c in zip(pn, tab["cam"][r].astype(np.float64))])
                d = np.abs(ex[..., :2] - M["poses"][r][..., :2]).max(2)
                out[0].append((p[:, :15, :2] - mu[:, :15, :2]).abs().amax((1, 2)).cpu().numpy())
                out[1].append(d.max(1))
                out[2].append(np.where(np.abs(M["poses"][r][..., 0]) < 32.0, d, 0.0).max(1))
                for j, q in ((3, p), (4, mu)):
                    x, y, _ = T.rear(q, cam_x[rt], W)
                    out[j].append(torch.maximum((x - sp[..., 0]).abs(), (y - sp[..., 1]).abs()).amax(1).cpu().numpy())
            return tuple(np.concatenate(v) for v in out)
        @torch.no_grad()
        def single(model, rows):
            """Rows forwarded one at a time as sh30_core.Core.plan does (interpreted): max |difference| of the whole plan mean from the served one."""
            d = []
            for i in rows:
                cur = torch.from_numpy(np.load(M["dump"][i])["cur"]).to(dev)
                H = model.net.run_batched(A.vision_feeds(torch.cat([torch.zeros_like(cur[:1]), cur[:-1]]), cur), ["view_39"])["view_39"].reshape(1, 8, *A.H_SHAPE)
                p = model(H, ego[i:i + 1], tc[:1]).float()[0, pi].view(33, 15).cpu().numpy()
                d.append(float(np.abs(p - M["mu"][i]).max()))
            return np.array(d)
        models = {t: T.load_pmodel(t, dev) for t in sorted(set(tags) | {t[:-1] + str(1 - int(t[-1])) for t in tags})}
        r1, lines = dict(tol_m=TOL_M, need=0.99, tags={}), []
        for t in tags:
            rows = np.flatnonzero(M["tag"] == t)
            other = t[:-1] + str(1 - int(t[-1]))
            (dm, dp, dn, dr, df), (cm, cp, _, _, _) = diff(models[t], rows), diff(models[other], rows)
            d1 = single(models[t], rows[:: max(len(rows) // a.single, 1)]) if a.single else np.full(1, np.nan)
            r1["tags"][t] = dict(n=len(rows), mu_share=float((dm <= TOL_M).mean()), pose_share=float((dp <= TOL_M).mean()), mu_median=float(np.median(dm)),
                                 mu_max=float(dm.max()), pose_median=float(np.median(dp)), pose_max=float(dp.max()), control=other,
                                 pose_share_within_32m=float((dn <= TOL_M).mean()), pose_share_one_ulp=float((dp <= 0.03126).mean()), pose_share_trainer_formula=float((dr <= TOL_M).mean()),
                                 rear_formula_max=float(df.max()), single_n=int(len(d1)) if a.single else 0, single_identical=float((d1 == 0).mean()), single_max=float(d1.max()),
                                 control_mu_share=float((cm <= TOL_M).mean()), control_pose_share=float((cp <= TOL_M).mean()), control_pose_median=float(np.median(cp)))
            q = r1["tags"][t]
            lines.append((t, len(rows), f"{q['mu_share']:.4f}", f"{q['pose_share']:.4f}", f"{q['mu_median']:.4f} / {q['mu_max']:.4f}", f"{q['pose_median']:.4f} / {q['pose_max']:.4f}",
                          f"{q['pose_share_within_32m']:.4f} / {q['pose_share_one_ulp']:.4f}", f"{q['pose_share_trainer_formula']:.4f} ({q['rear_formula_max']:.4f})",
                          f"{q['single_identical']:.3f} of {q['single_n']} (max {q['single_max']:.4f})" if q["single_n"] else "not run", f"{other}: {q['control_mu_share']:.4f} / {q['control_pose_share']:.4f} (median {q['control_pose_median']:.3f} m)"))
        r1["pass"] = bool(all(q["mu_share"] >= 0.99 and q["pose_share"] >= 0.99 and q["control_pose_share"] < 0.5 for q in r1["tags"].values()))
        (out / "r1.json").write_text(json.dumps(r1, indent=1) + "\n")
        (out / "r1.md").write_text("\n".join([f"# Gate R1 (inputs), rows `{a.name}`: {'PASS' if r1['pass'] else 'FAIL'}", "",
                                              f"Trainer forward (pp_train.PModel, eager, stored tokens + ego vector) against the served plan (compiled passes); share of rows within {TOL_M} m, needs "
                                              ">= 0.99 on both; the 8 poses = the trainer's plan through the serving export (op_interp.to_rear). Control = the other seed's checkpoint on the same "
                                              "rows (must stay below 0.5 on the poses). Diagnostics, not the gate: the pose share counted over the poses served within 32 m only / with a 0.03126 m "
                                              "tolerance (one fp16 ulp of a plan point beyond 32 m); the pose share with the trainer's own pose formula (pp_train.rear) and, in brackets, the "
                                              "largest difference of that formula from the serving export on the served plan itself; --single: rows forwarded one at a time as the driver does (8 image pairs in one "
                                              "encoder call from the dump, policy batch 1): share whose whole plan mean (33 x 15) equals the served one bit for bit.", ""] +
                                             md_table(["serving tag", "rows", "first 15 plan points", "8 poses", "plan points median / max (m)", "poses median / max (m)", "poses within 32 m / one-ulp tolerance", "poses, trainer formula (formula gap, m)", "one row at a time: identical plan mean", "wrong-seed control (points / poses)"], lines) + [""]))
        # ---- own-plan activation of each label (S's hinges as loss43 runs them: agent 0.3 m in the corridor / 0 beside it, road 0.3 m)
        ah = AgentHinge(root / "agent.npz", tab["names"], dev, 0.3, 0.0)
        rh, nh = Hinge(root / "road.npz", tab["names"], dev, 0.3), Hinge(root / "drivable.npz", tab["names"], dev, 0.3)
        P = torch.from_numpy(M["poses"].astype(np.float32)).to(dev)
        rr = torch.arange(n, device=dev)
        with torch.no_grad():
            av = ah.per_step(P[..., 0], P[..., 1], P[..., 2], rr).mean(1).cpu().numpy()
            rv = torch.relu(0.3 - rh.margins(P[..., 0], P[..., 1], P[..., 2], rr)).mean(1).cpu().numpy()
            nv = torch.relu(0.3 - nh.margins(P[..., 0], P[..., 1], P[..., 2], rr)).mean(1).cpu().numpy()
            z = torch.zeros(n, 8, device=dev)
            off0 = (rh.margins(z, z, z, rr).amin(1) < 0).cpu().numpy()                             # footprint off the road raster at t0 (bd4_train note (x))
        ade = np.hypot(*(M["poses"][..., :2] - tab["fut"][..., :2]).transpose(2, 0, 1)).mean(1)
        e0, lead, ex = np.abs(M["e0"]), M["lead"], allr["excl"]
        rep = dict(name=a.name, decisions=len(ex), rows=n, scenes=len(set(M["scene"].tolist())), rollouts=len(set(zip(allr["tag"].tolist(), allr["scene"].tolist()))),
                   logs=len(set(M["logu"].tolist())), excluded={q: int((ex == q).sum()) for q in REASONS}, off_path_05=float((e0 > 0.5).mean()), off_path_10=float((e0 > 1.0).mean()),
                   ahead_5=float((lead > 5).mean()), behind_5=float((lead < -5).mean()), e0_abs_p50_p90_max=[float(np.percentile(e0, q)) for q in (50, 90, 100)],
                   lead_p05_p50_p95=[float(np.percentile(lead, q)) for q in (5, 50, 95)], per_k={int(k): int((M["k"] == k).sum()) for k in range(10)},
                   per_k_decisions={int(k): int((allr["k"] == k).sum()) for k in range(10)}, per_tag={t: int((M["tag"] == t).sum()) for t in tags},
                   agent_active=float((av > 0).mean()), road_active=float((rv > 0).mean()), drivable_active=float((nv > 0).mean()), road_off_at_t0=float(off0.mean()),
                   imit_ade_mean=float(ade.mean()), imit_ade_over_05=float((ade > 0.5).mean()), path_ext=float(M["path_ext"].mean()), actors_mean=float(M["n_act"].mean()),
                   actors_over_k=float((M["n_act"] > K).mean()), cold_rows=int((M["n_keys"] < 4).sum()), front_bytes=int(front.nbytes),
                   active_by_k={int(k): [float((av[M["k"] == k] > 0).mean()), float((rv[M["k"] == k] > 0).mean())] for k in range(10) if (M["k"] == k).any()})
        np.savez(out / "active.npz", names=tab["names"], agent=av, road=rv, drivable=nv, road_off_t0=off0, imit_ade=ade)
        (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
        L = [f"# On-policy rows `{a.name}` (generated by on_rows.py tok)", "", f"{rep['rollouts']} rollouts of {rep['scenes']} scenes ({rep['logs']} log units), tags {', '.join(tags)}: "
             f"{rep['decisions']} decisions, **{n} rows**.", ""]
        L += md_table(["excluded for", "decisions"], [(q, v) for q, v in rep["excluded"].items()]) + [""]
        L += md_table(["quantity (rows)", "value"], [
            ("more than 0.5 m / 1.0 m off the logged path", f"{rep['off_path_05']:.3f} / {rep['off_path_10']:.3f}"), ("|e0| median / p90 / max (m)", " / ".join(f"{v:.2f}" for v in rep["e0_abs_p50_p90_max"])),
            ("more than 5 m ahead of / behind the log's position at the same time", f"{rep['ahead_5']:.3f} / {rep['behind_5']:.3f}"),
            ("lead p05 / median / p95 (m)", " / ".join(f"{v:+.2f}" for v in rep["lead_p05_p50_p95"])),
            ("own plan activates the agent hinge (0.3 m / 0 m)", f"{rep['agent_active']:.3f}"), ("own plan activates the road-and-lane hinge (0.3 m)", f"{rep['road_active']:.3f}"),
            ("own plan activates P2H10's drivable hinge on these rows (0.3 m; not a label of this lane)", f"{rep['drivable_active']:.3f}"),
            ("footprint off the road-and-lane raster at t0", f"{rep['road_off_at_t0']:.3f}"),
            ("own plan against the imitation target: mean ADE (m) / share over 0.5 m", f"{rep['imit_ade_mean']:.3f} / {rep['imit_ade_over_05']:.3f}"),
            ("imitation target on the straight continuation beyond the log", f"{rep['path_ext']:.3f}"), ("actors per row, mean / share of rows with more than 32", f"{rep['actors_mean']:.1f} / {rep['actors_over_k']:.3f}"),
            ("cold-start rows (fewer than 4 keyframes)", rep["cold_rows"])]) + [""]
        L += md_table(["decision k"] + [str(k) for k in range(10)], [("decisions", *[rep["per_k_decisions"][k] for k in range(10)]), ("rows", *[rep["per_k"][k] for k in range(10)]),
                                                                    ("agent hinge active", *[f"{rep['active_by_k'].get(k, [np.nan])[0]:.2f}" for k in range(10)]),
                                                                    ("road hinge active", *[f"{rep['active_by_k'].get(k, [np.nan, np.nan])[1]:.2f}" for k in range(10)])]) + [""]
        (out / "report.md").write_text("\n".join(L))
        run.summary.update(rows=n, r1=r1, report=rep)
        run.info(f"R1 {'PASS' if r1['pass'] else 'FAIL'}: {json.dumps(r1['tags'])}")
        run.info("\n".join(L))


# ---------------------------------------------------------------- BEV preview
def cmd_bev(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = ROWS / a.name
    with Run("onpol_nuplan", f"rows-bev-{a.name}", config=vars(a)) as run:
        run.use_split(splits.load(SPLIT))
        M, tab, ag, rd = (dict(np.load(f)) for f in (out / "meta.npz", CR / f"{FAM}_{a.name}" / "tab.npz", out / "agent.npz", out / "road.npz"))
        act = dict(np.load(out / "active.npz")) if (out / "active.npz").exists() else None
        n, rng = len(M["names"]), np.random.default_rng(0)
        near = np.where(ag["valid"], np.hypot(ag["box"][..., 0], ag["box"][..., 1]), np.inf).min((1, 2))
        turn = np.abs(wrap(tab["fut"][:, -1, 2]))
        groups = [("off the path", np.argsort(-np.abs(M["e0"]))), ("ahead of the log", np.argsort(-M["lead"])), ("behind the log", np.argsort(M["lead"])), ("actor close", np.argsort(near)),
                  ("turn", np.argsort(-turn)), ("decision 0 (cold start)", np.flatnonzero(M["k"] == 0)), ("random", rng.permutation(n))]
        if act is not None:
            groups[5:5] = [("agent hinge active", np.argsort(-act["agent"])), ("road hinge active", np.argsort(-act["road"]))]
        per, pick = max(a.n // len(groups), 1), []
        for name, order in groups:
            seen = {M["scene"][i] for _, i in pick}
            new = [i for i in order if M["scene"][i] not in seen][:per]                      # one row per scene across groups where possible
            pick += [(name, i) for i in (new or list(order[:per]))]
        xs, ys = -8 + 0.5 * (np.arange(128) + .5), -24 + 0.5 * (np.arange(96) + .5)
        HL, HW = 5.176 / 2, 2.297 / 2

        def box_xy(x, y, yaw, hl, hw):
            c, s = np.cos(yaw), np.sin(yaw)
            u = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
            return x + c * u[:, 0] - s * u[:, 1], y + s * u[:, 0] + c * u[:, 1]
        (out / "bev").mkdir(exist_ok=True)
        cols = 4
        nr = (len(pick) + cols - 1) // cols
        fig, axs = plt.subplots(nr, cols, figsize=(5.2 * cols, 6.4 * nr), squeeze=False)
        for A, (gname, i) in zip(axs.ravel(), pick):
            A.contourf(ys, xs, rd["sdf"][i].astype(np.float32), levels=[0, 99], colors=["#e3e3e3"])
            A.contour(ys, xs, rd["sdf"][i].astype(np.float32), levels=[0], colors=["#1f77b4"], linewidths=0.8)
            A.plot(M["gpath"][i][:, 1], M["gpath"][i][:, 0], color="#7f7f7f", lw=1, ls="--")
            for j in range(NT):
                for o in np.flatnonzero(ag["valid"][i, j]):
                    b = ag["box"][i, j, o]
                    px, py = box_xy(b[0], b[1], b[2], b[3] / 2, b[4] / 2)
                    A.plot(py, px, color="#d62728", alpha=1.0 - 0.09 * j, lw=1.3 if j == 0 else 0.6)
            lf = M["logfut"][i]
            A.plot(lf[:, 1], lf[:, 0], "x", color="#7f7f7f", ms=5)
            f, p = np.r_[np.zeros((1, 3)), tab["fut"][i]], np.r_[np.zeros((1, 3)), M["poses"][i]]
            A.plot(f[:, 1], f[:, 0], "o-", color="#ff7f0e", ms=3, lw=1.2)
            A.plot(p[:, 1], p[:, 0], "o-", color="k", ms=3, lw=1.2)
            for q in (0, 4, 8):
                c, s = np.cos(p[q, 2]), np.sin(p[q, 2])
                px, py = box_xy(p[q, 0] + c * CEN, p[q, 1] + s * CEN, p[q, 2], HL, HW)
                A.plot(py, px, color="#2ca02c", lw=1.1)
            A.set_xlim(22, -22), A.set_ylim(-8, 56), A.set_aspect("equal")
            extra = f"; hinge agent {act['agent'][i]:.3f} road {act['road'][i]:.3f}" if act is not None else ""
            A.set_title(f"[{gname}] {M['names'][i]}\nv {M['speed'][i]:.1f} m/s, e0 {M['e0'][i]:+.2f} m, psi_e {np.degrees(M['psi_e'][i]):+.1f} deg, lead {M['lead'][i]:+.1f} m{extra}", fontsize=7.5)
        for A in axs.ravel()[len(pick):]:
            A.axis("off")
        fig.suptitle(f"on-policy rows `{a.name}`, row frame (x forward up, y left): grey = road-and-lane raster (blue edge), red = simulator actor boxes at t0 .. t0 + 4 s (fading), "
                     "dashed = logged path, x = the log's own positions at t0 .. + 4 s,\norange = imitation target, black = the served own plan with the ego box at 0 / 2 / 4 s (green)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fn = out / "bev" / f"on_rows_bev_{a.name}.png"
        fig.savefig(fn, dpi=75)
        run.summary.update(fig=str(fn), rows=[M["names"][i] for _, i in pick])
        run.info(f"{len(pick)} rows -> {fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("geo")
    p.add_argument("--name", required=True), p.add_argument("--roll", nargs="+", required=True), p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("tok")
    p.add_argument("--name", required=True), p.add_argument("--force", action="store_true")
    p.add_argument("--single", type=int, default=0, help="gate R1 extra: this many rows per tag forwarded one at a time (the driver's batch shapes)")
    p = sub.add_parser("bev")
    p.add_argument("--name", required=True), p.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    {"geo": cmd_geo, "tok": cmd_tok, "bev": cmd_bev}[a.cmd](a)
