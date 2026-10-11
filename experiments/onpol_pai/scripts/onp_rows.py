#!/usr/bin/env python3
"""ONPOL-PAI row pipeline: finished closed-loop PAI rollouts (pai_run.sh run dirs on the Tokyo box) -> one npz of training rows per
scene: the model inputs exactly as the driver fed them, and privileged labels at the visited states. The file format is the contract in
experiments/onpol_pai/results/rows_format.md; read it before changing a key. CPU only, two stages in two images:

  bash /data/runs/onpol_pai/dev/scripts/onp_rows.sh pol '/data/runs/onpol_pai/pass1/*/runs/*'      # all of it; skips scenes already built

  rows    (image alpasim-base: AlpaSim's .asl reader, scene loader, polars) every scene of the finished run dirs -> <stage>/<pass>/<scene>.npz,
          uncompressed, complete except that the synthesised (back-warped) slots of the cold-start rows come from this image's OpenCV
  cold    (the driver image the rollouts were served from) re-runs pai_core.slots on the cold-start rows, where the back-warp is the served
          one bit for bit (cv2.remap differs between the two images' OpenCV 5.0 / 4.11 by 1 to 3 grey levels on about 2 % of the bytes),
          and writes the final compressed <out>/<pass>/<scene>.npz; the staged file (host /tmp, not the data disk) is then removed
  report  <out>/rows_<pass>.md        fig  3 decisions per scene as BEV pngs, <out>/fig/ (skips scenes whose png is newer than the npz)
  `rows --redo` rebuilds scenes that already have a file (ROWS_ARGS=--redo with onp_rows.sh).

Inputs = an open-loop replay of the driver-side messages of rollout.asl, in arrival order, through pai_driver.Driver itself (the
approach of experiments/lowboard_diag/scripts/lbd_pai_replay.py) with the model pass replaced by a tap on pai_core.slots / pai_core.plan:
the tap records their arguments, so nothing of the driver's input path is re-implemented. Gate G-in compares them with the PAI_DUMP
files (bit for bit) and with drive.jsonl (to its rounding).
Labels: c1_extract.one_log (simulated actors, metric series), the rollout metadata (logged ego rig path, ego box), the runtime's scene
loader (road edges, wait lines), lc_ctx.turn; geometry in lib/onp_geom.py.

Output <out>/<pass>/: <scene>.npz, failed/<scene>.txt, times.jsonl, STATUS, DONE | ERROR, log.txt.
"""
import argparse
import asyncio
import glob
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
JEV = Path(os.environ.get("JEV_REPO") or (HERE.parents[3] if len(HERE.parents) > 3 and (HERE.parents[3] / "jevdrive").is_dir() else "/jev"))
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(JEV)] + [str(JEV / "experiments" / p) for p in (
    "alpasim/lib", "alpasim/scripts", "lowboard_diag/scripts", "long_contact/scripts", "long_contact/lib", "body1/lib")]
import onp_geom as G  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/data"))
EVENTS = ("collision_at_fault", "offroad", "left_corridor_laterally")
SPEED = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 99))


# ---------------------------------------------------------------- inputs: replay through the driver
class _Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


class _Fed(Exception):
    """Raised in place of the model pass: the decision's inputs are captured, the plan is not needed."""


def replay(asl: str, dump_of, cold: str, lht: bool) -> dict:
    """rollout.asl -> {md: rollout metadata, frames: {frame_end_us: model frame}, rows: {k: captured inputs | {error}}, gate}.
    dump_of(k) -> the PAI_DUMP file of decision k or None; every one found is compared with the replay on the spot, except the frames
    of a cold-start row (gate["cold"]: {k: [file, the other fields equal]}; `finalize` compares those in the driver image)."""
    import c1_extract as X
    import lbd_pai_x as LX
    import pai_driver as PD
    logging.getLogger("pai").disabled = True                                  # the tap's exception is logged by the driver as a failure
    PC, cap = PD.PC, {}
    if not hasattr(PC, "_slots"):
        PC._slots = PC.slots

    def slots(frames, t0, P, V, cam_t, cold_, *a, **k):
        cur, valid, real = PC._slots(frames, t0, P, V, cam_t, cold_, *a, **k)
        ts = np.array(sorted(frames), np.int64)
        near = ts[np.abs(ts[None] - (t0 + PC.T_SLOT_US)[:, None]).argmin(1)]   # pai_core.slots' own choice of frame per slot
        cap.update(t0=int(t0), P=np.array(P), V=np.array(V), cur=cur, valid=valid, real=real, slot_ts=np.where(real, near, -1))
        return cur, valid, real

    def plan(core, cur, valid, P, V, acc, cmd, cam_t, lht_=False):
        cap.update(acc=np.array(acc, np.float64), cmd=np.array(cmd, np.float32), cam_t=np.array(cam_t, np.float64), lht=bool(lht_),
                   ego=PD.C.PA.ego_features(P, V, np.tile(np.asarray(acc, np.float32), (4, 1)), np.asarray(cmd, np.float32)))
        raise _Fed
    PC.slots, PC.plan = slots, plan
    tmp = Path(tempfile.mkdtemp(prefix="onp-"))
    drv, ctx = PD.Driver(None, tmp, cold, 1, 0, lht), _Ctx()
    R = dict(md=None, frames={}, rows={}, gate=dict(n=0, eq=0, diff=[], cold={}))

    async def go():
        s = None
        async for kind, m in X._read(asl, LX.KIND | {"rollout_metadata"}):
            if kind == "rollout_metadata":
                R["md"] = m
            elif kind == "driver_session_request":
                drv.start_session(m, ctx)
                s = drv.sessions[m.session_uuid]
            elif kind == "driver_camera_image":
                drv.submit_image_observation(m, ctx)
                if m.camera_image.logical_id == PD.CAM:
                    R["frames"][int(m.camera_image.frame_end_us)] = s.frames[int(m.camera_image.frame_end_us)]
            elif kind == "driver_ego_trajectory":
                drv.submit_egomotion_observation(m, ctx)
            elif kind == "route_request":
                drv.submit_route(m, ctx)
            elif kind == "driver_request":
                cap.clear()
                k, err = s.count["drive"], ""
                try:
                    drv.drive(m, ctx)
                except RuntimeError as e:
                    err = str(e)
                if "ego" not in cap:
                    R["rows"][k] = dict(error=err)
                    continue
                t0 = cap["t0"]
                r = R["rows"][k] = {x: cap[x] for x in ("t0", "P", "V", "acc", "cmd", "valid", "real", "slot_ts", "ego", "lht", "cam_t")}
                r.update(now=int(m.time_now_us), anchor=s.pose_at(t0), z0=s.poses[min(s.poses, key=lambda t: abs(t - t0))].pose.vec.z)
                if not cap["real"].all():
                    r["cur"] = cap["cur"]
                f = dump_of(k)
                if f is not None:                                             # G-in, dump half
                    z, g, cold = np.load(f), R["gate"], "cur" in r
                    stored = cap["cur"] if cold else np.stack([R["frames"][int(t)] for t in cap["slot_ts"]])      # as the npz will hold it
                    bad = [n for n, ok in (("cur", cold or (np.array_equal(z["cur"], cap["cur"]) and np.array_equal(z["cur"], stored))),
                                           ("valid", np.array_equal(z["valid"], cap["valid"])), ("real", np.array_equal(z["real"], cap["real"])),
                                           ("ego", np.array_equal(z["ego"], cap["ego"])), ("t0", int(z["t0"]) == t0)) if not ok]
                    if cold:
                        g["cold"][k] = [str(f), not bad]
                    else:
                        g["n"] += 1
                        g["eq"] += not bad
                    if bad and len(g["diff"]) < 5:
                        g["diff"].append(dict(k=k, fields=bad, cur_bytes_differ=int((z["cur"] != cap["cur"]).sum()),
                                              cur_slots_differ=np.flatnonzero((z["cur"] != cap["cur"]).reshape(8, -1).any(1)).tolist(),
                                              ego_max=float(np.abs(z["ego"] - cap["ego"]).max())))
    try:
        asyncio.run(go())
    finally:
        drv.out.close(), drv.start.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return R


def driver_logs(run: str, sess: str):
    """The driver's own records of one session: drive.jsonl rows by k, and its start.jsonl line."""
    rec = [r for r in (json.loads(x) for x in open(f"{run}/driver/drive.jsonl") if sess in x) if r.get("kind") == "drive" and r["session"] == sess]
    start = [r for r in (json.loads(x) for x in open(f"{run}/driver/start.jsonl") if sess in x) if r["session"] == sess]
    if not start:
        raise RuntimeError(f"session {sess} not in start.jsonl")
    return sorted(rec, key=lambda r: r["k"]), start[-1]


# ---------------------------------------------------------------- labels
def load_map(sim: str, scene: str, uuid: str, nurec: str):
    """The runtime's own scene loader on this scene's usdz -> (road-edge polylines [(n, 3) x y z], wait lines [(polyline, type, implicit)])
    in the rollout's local frame."""
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_runtime.scene_loader import build_scene_loader
    from omegaconf import OmegaConf
    from trajdata.maps.vec_map_elements import MapElementType as MT
    cfg = OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(f"{sim}/generated-user-config-0.yaml"))
    cfg.scene_provider.usdz.data_dir = f"{nurec}/all-usdzs/{uuid}.usdz"        # the config's own path is the runtime container's mount
    E = build_scene_loader(cfg).get_data_source(scene).map.elements
    xyz = lambda p: np.c_[p[:, :2], p[:, 2] if p.shape[1] > 2 else np.full(len(p), np.nan)].astype(np.float64)  # noqa: E731
    return ([xyz(np.asarray(e.polyline.points)) for e in E.get(MT.ROAD_EDGE, {}).values()],
            [(np.asarray(e.polyline.points, np.float64)[:, :2], e.wait_line_type, e.is_implicit) for e in E.get(MT.WAIT_LINE, {}).values()])


def series(M: dict, name: str):
    """c1_extract.one_log's metric table -> (t_us int64, value float with NaN) of one metric, time order."""
    a = np.array(M.get(name, np.zeros((0, 2))), dtype=object).reshape(-1, 2)
    t = np.array([np.nan if x is None else x for x in a[:, 0]], np.float64)
    v = np.array([np.nan if x is None else x for x in a[:, 1]], np.float64)
    o = np.argsort(t[np.isfinite(t)], kind="stable")
    return np.round(t[np.isfinite(t)][o]).astype(np.int64), v[np.isfinite(t)][o]


def events(M: dict, corridor: float):
    """The scorer's reading of the per-step series (aggregation/processing.py DEFAULT_MODIFIERS, then aggregation/main.py's corridor
    modifier) -> (first time each of EVENTS is flagged (3,), t_rel, t_end). The scorer reads the steps from the first with
    eval_relevant > 0 (t_rel) up to and including the first with offroad, any collision, or dist_to_gt_trajectory >= corridor (t_end;
    NEVER if none): a flag outside that window is not in the score (an at-fault collision after a rear-end one, an offroad after the
    corridor exit). At-fault collision = front | lateral. NEVER = not flagged."""
    first = lambda t, m: int(t[m][0]) if m.any() else G.NEVER                # noqa: E731
    tr, vr = series(M, "eval_relevant")
    t_rel = first(tr, vr > 0) if len(tr) else -1
    S = {n: series(M, n) for n in ("collision_any", "offroad", "collision_front", "collision_lateral", "left_corridor_laterally", "dist_to_gt_trajectory")}
    S["dist_to_gt_trajectory"] = (S["dist_to_gt_trajectory"][0], (S["dist_to_gt_trajectory"][1] >= corridor).astype(np.float64))
    t_end = min(first(t, (v > 0) & (t >= t_rel)) for t, v in (S["collision_any"], S["offroad"], S["dist_to_gt_trajectory"]))
    out = [min(first(t, (v > 0) & (t >= t_rel) & (t <= t_end)) for t, v in (S[n] for n in names))
           for names in (("collision_front", "collision_lateral"), ("offroad",), ("left_corridor_laterally",))]
    return np.array(out, np.int64), t_rel, t_end


def rig_n(p, A):
    """Local poses (N, m, 3) -> the rig frames at A (N, 3)."""
    c, s = np.cos(A[:, 2])[:, None], np.sin(A[:, 2])[:, None]
    dx, dy = p[..., 0] - A[:, None, 0], p[..., 1] - A[:, None, 1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, p[..., 2] - A[:, None, 2]], -1)


def turn_rows(path: G.Path, s_ego) -> np.ndarray:
    """(N, 2) arc from s_ego to the peak of the first >= 45 deg turn ahead within 100 m and the log's speed there (lc_ctx.turn on the path
    heading at 1 m, as LONG-PROBE reads it; NaN if none)."""
    import lc_ctx
    out = np.full((len(s_ego), 2), np.nan, np.float32)
    for n, s in enumerate(s_ego):
        if path.len - s <= 12 or len(path.sv) < 2:
            continue
        g = np.arange(0, min(lc_ctx.LC.NR, int(path.len - s) + 1), dtype=np.float64)
        tn = lc_ctx.turn(path.at(s + g)[:, 2], g, path.s - s, path.v)
        if tn is not None and tn["spk"] <= 100.0:
            out[n] = tn["spk"], tn["vpk"]
    return out


def scene_rows(j: dict) -> dict:
    """One scene -> the arrays of its npz (rows_format.md)."""
    import c1_extract as X
    T, tic = {}, time.time()

    def lap(name):
        nonlocal tic
        T[name], tic = round(time.time() - tic, 2), time.time()
    run, scene, sess = j["run"], j["scene"], j["session"]
    sim = f"{run}/sim"
    asl = f"{sim}/rollouts/{scene}/{sess}/rollout.asl"
    X.asl_of = lambda *_: asl                                                 # one_log on the session the scorer read
    _, o = X.one_log((sim, scene))
    if o["session"] != sess:
        raise RuntimeError(f"asl session {o['session']} != {sess}")
    lap("one_log")
    rec, start = driver_logs(run, sess)
    dump = lambda k: (lambda f: f if f.exists() else None)(Path(run) / "driver/dump" / f"s{start['n']:02d}_k{k:03d}.npz")  # noqa: E731
    R = replay(asl, dump, j["cold"], j["lht"])
    lap("replay")

    # ---- rows = the served decisions, aligned with the replay by the driver's call counter
    rec = [r for r in rec if r.get("infer")]
    miss = [r["k"] for r in rec if "t0" not in R["rows"].get(r["k"], {})]
    if miss or not rec:
        raise RuntimeError(f"{len(miss)} of {len(rec)} served decisions not reproduced by the replay; first k {miss[:5]}: {R['rows'].get(miss[0]) if miss else None}")
    rows, N = [R["rows"][r["k"]] for r in rec], len(rec)
    st = lambda key, dt=np.float64: np.array([x[key] for x in rows], dt)      # noqa: E731
    jl = lambda key, dt=np.float32: np.array([r[key] for r in rec], dt)       # noqa: E731
    t0, P, V, anchor, ego_in, cmd = st("t0", np.int64), st("P"), st("V"), st("anchor"), st("ego", np.float32), st("cmd", np.float32)
    frame_ts = np.array(sorted(R["frames"]), np.int64)
    slot_ts = st("slot_ts", np.int64)
    slot = np.where(slot_ts >= 0, np.searchsorted(frame_ts, slot_ts), -1).astype(np.int32)
    cold = np.flatnonzero((slot < 0).any(1)).astype(np.int32)
    from pai_core import C
    Z = dict(scene=scene, session=sess, tag=j["tag"], run=run, t0_us=t0, k=jl("k", np.int32),
             frames=np.stack([R["frames"][t] for t in frame_ts]), frame_ts=frame_ts, slot=slot, valid=st("valid", bool), real=st("real", bool),
             cold_rows=cold, cold_cur=np.stack([rows[i]["cur"] for i in cold]) if len(cold) else np.zeros((0, 8) + C.FRAME, np.uint8),
             P=P, V=V, acc=st("acc"), cmd=cmd, cam_t=rows[0]["cam_t"], lht=rows[0]["lht"], ego_in=ego_in, ego=jl("ego"), poses=jl("poses"),
             poses_model=np.array([r.get("poses_model", r["poses"]) for r in rec], np.float32), anchor=anchor, z0=st("z0", np.float32),
             v0=np.hypot(V[:, -1, 0], V[:, -1, 1]).astype(np.float32))
    Z["pass"] = j["pass"]
    # ---- G-in, drive.jsonl half: the driver's rounded records
    eq = ((np.abs(P.round(4) - jl("hist", np.float64)).reshape(N, -1).max(1) < 1e-9) & (ego_in.round(5) == Z["ego"]).all(1)
          & (np.abs(anchor.round(4) - jl("anchor", np.float64)).max(1) < 1e-9) & (cmd.argmax(1) == jl("cmd", np.int64)) & (t0 == jl("t0", np.int64)))
    gate = dict(gin_dump_n=R["gate"]["n"], gin_dump_eq=int(R["gate"]["eq"]), gin_dump_diff=R["gate"]["diff"], gin_cold_dumps=R["gate"]["cold"],
                gin_rec_n=N, gin_rec_eq=int(eq.sum()), gin_rec_diff_k=Z["k"][~eq][:5].tolist())
    lap("inputs")

    # ---- ego boxes, logged path
    md = R["md"]
    q, ea = md.transform_ego_coords_rig_to_aabb, next(x.aabb for x in md.actor_definitions.actor_aabb if x.actor_id == "EGO")
    veh_sim = np.array([ea.size_x, ea.size_y, q.vec.x], np.float32)
    vs = start["rollout_spec"].get("vehicle", {})
    bb = vs.get("boundingBox", {})
    veh = np.array([bb.get("sizeX", np.nan), bb.get("sizeY", np.nan), vs.get("rigToBoundingBox", {}).get("vec", {}).get("x", np.nan), np.nan], np.float32)
    off, front = float(veh_sim[2]), float(veh_sim[2] + veh_sim[0] / 2)
    gt = X.traj(md.ego_rig_recorded_ground_truth_trajectory)
    gt = gt[np.r_[True, np.diff(gt[:, 0]) > 0]]
    path = G.Path(gt[:, 0], gt[:, 1:])
    fwd = lambda p: np.c_[p[:, 0] + off * np.cos(p[:, 2]), p[:, 1] + off * np.sin(p[:, 2]), p[:, 2]]  # noqa: E731  rig pose -> box-centre pose
    cxy = fwd(gt[:, 1:])[:, :2]                                               # the scorer's reference: the logged box-centre polyline
    lg, esim = o["logged"][0]["traj"], o["actors"]["EGO"]                      # c1_extract's "logged ego first" and the simulated ego: box centres
    near_t = lambda a, b: (lambda i: (i, np.abs(a[i] - b) < 1000))(np.searchsorted(a, b).clip(0, len(a) - 1))  # noqa: E731
    i, m = near_t(lg[:, 0], gt[:, 0])
    chk_log = float(np.abs(lg[i[m], 1:3] - cxy[m]).max()) if m.any() and o["logged"][0]["id"] == "EGO" else float("nan")
    i, m = near_t(esim[:, 0], t0.astype(np.float64))
    chk_anchor = float(np.hypot(*(esim[i[m], 1:3] - fwd(anchor)[m, :2]).T).max()) if m.any() else float("nan")
    s_ego, e0 = path.track(anchor[:, :2])
    h0 = path.at(s_ego)[:, 2]
    tw = 2 * np.pi * np.round((h0 - anchor[:, 2]) / (2 * np.pi))               # branch of the path's unwrapped yaw against the ego's
    dpsi0 = anchor[:, 2] + tw - h0
    lp, ls, _ = path.at_time(t0)
    fp, _, fok = path.at_time(t0[:, None] + G.T_FUT)
    arcs = s_ego[:, None] + G.PATH_ARC
    pth = rig_n(path.at(arcs), anchor)
    lf, l0 = rig_n(fp, anchor), rig_n(lp[:, None], anchor)[:, 0]
    lf[..., 2], l0[:, 2], pth[..., 2] = lf[..., 2] - tw[:, None], l0[:, 2] - tw, pth[..., 2] - tw[:, None]
    im = [G.imit(path, t0[n], s_ego[n], e0[n], dpsi0[n], Z["v0"][n], anchor[n]) for n in range(N)]
    imit = np.stack([x[0] for x in im])
    imit[..., 2] -= tw[:, None]
    dlat = np.array([G.scorer_lateral(cxy, p[:2]) for p in fwd(anchor)])
    mt, mv = series(o["metrics"], "lateral_dist_to_gt_trajectory")
    ix = np.searchsorted(mt, t0).clip(0, max(len(mt) - 1, 0))
    m = (np.abs(mt[ix] - t0) < 1000) & (ix >= 2) if len(mt) else np.zeros(N, bool)      # the scorer sets its first two steps to 0
    chk_lat = float(np.abs(dlat[m] - mv[ix[m]]).max()) if m.any() else float("nan")
    Z.update(veh=veh, veh_sim=veh_sim, log_xyyaw=np.c_[path.xy, path.yaw], log_t_us=path.t, log_v=path.v.astype(np.float32), log_s=path.s.astype(np.float32),
             path_len=np.float32(path.len), log0=l0, log_fut=lf, log_fut_ok=fok, s_ego=s_ego, e0=e0, dpsi0=dpsi0, ds0=s_ego - ls,
             path=pth[..., :2], path_ok=(arcs >= 0) & (arcs <= path.len), path_v=path.speed(arcs), path_h=pth[..., 2],
             imit=imit, imit_ok=np.stack([x[1] for x in im]), dlat_c=dlat)
    lap("path")

    # ---- simulator actors
    ids, Xt, dims, lab = G.actor_tracks(o["actors"], {k: o["size"].get(k, (0.0, 0.0, "")) for k in o["actors"]}, o["actor_t"])
    box, bok, idx = G.boxes(np.asarray(o["actor_t"], np.int64), Xt, dims, t0, anchor)
    pick = lambda a, fill: np.where(idx >= 0, a[np.maximum(idx, 0)], fill) if len(a) else np.full(idx.shape, fill)  # noqa: E731
    Z.update(box=box, box_ok=bok, box_cls=pick(np.array([G.CLS.get(x, 1) for x in lab], np.int8), -1).astype(np.int8),
             box_label=pick(lab.astype("U24"), "").astype("U24"), box_id=pick(ids.astype("U24"), "").astype("U24"))
    lap("boxes")

    # ---- map: road edges, wait lines (stubbed, with the reason in meta, if the scene loader fails)
    edges, lines, map_err = [], [], ""
    try:
        edges, lines = load_map(sim, scene, md.session_metadata.nre_uuid, j["nurec"])
    except Exception as e:
        map_err = f"{e!r}"[:300] + " | " + " <- ".join(x.strip().split("\n")[0] for x in traceback.format_tb(e.__traceback__)[-4:])
    lap("map")
    near = [e for e in edges if len(e) > 1 and (np.hypot(*(e[:, None, :2] - path.pv[None, ::5]).transpose(2, 0, 1)).min() < 150.0)]
    rs = [G.resample(e) for e in near]
    exyz = np.concatenate(rs) if rs else np.zeros((0, 3))
    eoff = np.r_[0, np.cumsum([len(r) for r in rs])].astype(np.int32)
    seg = G.edge_segments(exyz, eoff) if len(exyz) else np.zeros((0, 5))
    ebox, eok = G.edge_rows(seg, anchor, Z["z0"] + q.vec.z)
    Z.update(edge_box=ebox, edge_ok=eok, edges_xyz=exyz, edges_off=eoff, has_edges=not map_err, has_stop=not map_err)
    Z.update(G.stop_rows(G.crossings(path, lines, front), t0, s_ego, front))
    if map_err:
        Z["stop_s"][:], Z["wl_s"][:] = np.nan, np.nan
    tn = turn_rows(path, s_ego)
    (ev, t_rel, t_end), sm = events(o["metrics"], j["corridor"]), j["summary"]
    zk = np.array([bool((sm.get("score_metrics") or {}).get(n) or 0) for n in EVENTS])
    Z.update(turn_s=tn[:, 0], turn_vpk=tn[:, 1], score=np.float32(np.nan if sm.get("score") is None else sm["score"]), fail_reason=str(sm.get("failure_reason") or ""),
             zero_kind=zk, t_event_us=ev, after_event=t0 >= ev.min(), t_rel_us=np.int64(t_rel), t_end_us=np.int64(t_end), after_end=t0 > t_end,
             corridor_m=np.float32(j["corridor"]))
    lap("labels")
    in40 = np.hypot(*(seg[None, :, :2] - anchor[:, None, :2]).transpose(2, 0, 1)) <= G.EDGE_R if len(seg) else np.zeros((N, 0), bool)
    lvl = ~(np.abs(seg[None, :, 4] - (Z["z0"] + q.vec.z)[:, None]) > G.EDGE_DZ) if len(seg) else in40
    meta = dict(args={k: j[k] for k in ("cold", "lht", "nurec", "pass", "tag")}, n_rows=N, n_frames=len(frame_ts), n_cold=len(cold), n_calls=len(R["rows"]),
                n_actors=len(ids), n_edges=len(near), n_wait_lines=len(lines), map_error=map_err, **gate, chk_logged_centre_m=chk_log, chk_scorer_lateral_m=chk_lat,
                chk_anchor_vs_sim_m=chk_anchor, chk_event_flags=bool((zk == (ev != G.NEVER)).all()), edge_segs_40m=int(in40.sum()),
                edge_segs_40m_same_level=int((in40 & lvl).sum()), edge_rows_full=int(eok.all(1).sum()), secs=T, secs_build=round(sum(T.values()), 2))
    Z["meta"] = json.dumps(meta)
    f32 = ("log0", "log_fut", "s_ego", "e0", "dpsi0", "ds0", "path", "path_v", "path_h", "imit", "dlat_c", "edges_xyz")
    return {k: (np.asarray(v, np.float32) if k in f32 else v) for k, v in Z.items()}


# ---------------------------------------------------------------- driver of the build
def build(j: dict):
    """Stage 1 of one scene -> the staged, uncompressed npz."""
    t = time.time()
    out, stg = Path(j["out"]), Path(j["stage"]) / f"{j['scene']}.npz"
    try:
        Z = scene_rows(j)
        tmp = stg.with_name(f".{stg.stem}.{os.getpid()}.tmp.npz")
        Z["meta"] = json.dumps(dict(json.loads(Z["meta"]), secs_stage1=round(time.time() - t, 2)))
        np.savez(tmp, **Z)
        os.replace(tmp, stg)
        (out / "failed" / f"{j['scene']}.txt").unlink(missing_ok=True)
        return j["scene"], json.loads(Z["meta"]), None
    except Exception:
        (out / "failed").mkdir(exist_ok=True)
        (out / "failed" / f"{j['scene']}.txt").write_text(f"stage rows, {j['run']}\n{traceback.format_exc()}")
        return j["scene"], None, traceback.format_exc()


def finalize(j: dict):
    """Stage 2 of one scene, in the driver image: the cold-start rows' slots through pai_core.slots as served (its OpenCV), G-in on the
    cold PAI_DUMP files, the final compressed npz; the staged file is removed."""
    t = time.time()
    stg, out = Path(j["stg"]), Path(j["out"]) / Path(j["stg"]).name
    try:
        import pai_core as PC
        with np.load(stg) as z:
            Z = {k: z[k] for k in z.files}
        meta = json.loads(str(Z["meta"]))
        cur = np.empty_like(Z["cold_cur"])
        for i, n in enumerate(Z["cold_rows"]):
            t0 = int(Z["t0_us"][n])
            fr = {int(ts): Z["frames"][m] for m, ts in enumerate(Z["frame_ts"]) if ts <= t0 + 30_000}       # what the session held at the call
            cur[i], valid, real = PC.slots(fr, t0, Z["P"][n], Z["V"][n], Z["cam_t"], meta["args"]["cold"])
            if not (np.array_equal(valid, Z["valid"][n]) and np.array_equal(real, Z["real"][n])):
                raise RuntimeError(f"row {n}: slots' valid / real differ between the two stages")
            d = meta["gin_cold_dumps"].get(str(int(Z["k"][n])))
            if d:
                same = bool(d[1]) and np.array_equal(np.load(d[0])["cur"], cur[i])
                meta["gin_dump_n"], meta["gin_dump_eq"] = meta["gin_dump_n"] + 1, meta["gin_dump_eq"] + same
                if not same and len(meta["gin_dump_diff"]) < 5:
                    meta["gin_dump_diff"].append(dict(k=int(Z["k"][n]), fields=["cur (cold)"], cur_bytes_differ=int((np.load(d[0])["cur"] != cur[i]).sum())))
        meta.update(cold_bytes_changed=int((cur != Z["cold_cur"]).sum()), cold_exact=True)
        Z["cold_cur"], Z["meta"] = cur, json.dumps(meta)
        tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **Z)
        os.replace(tmp, out)
        stg.unlink()                                                          # the staged copy lives in host /tmp, not on the data disk
        dt = round(time.time() - t, 2)
        return stg.stem, dict(meta, bytes=out.stat().st_size, secs_stage2=dt, secs_total=round(meta["secs_stage1"] + dt, 2)), None
    except Exception:
        (out.parent / "failed").mkdir(exist_ok=True)
        (out.parent / "failed" / f"{stg.stem}.txt").write_text(f"stage cold, {stg}\n{traceback.format_exc()}")
        return stg.stem, None, traceback.format_exc()


def jobs_of(a, out: Path, say) -> list:
    jobs, seen = [], set()
    for run in sorted({r.rstrip("/") for g in a.runs for r in glob.glob(g) if os.path.isdir(r)}):
        sm = Path(run) / "sim/aggregate/results-summary.json"
        if not (sm.exists() and (Path(run) / "DONE").exists()):
            say(f"skip {run}: no DONE / results-summary.json (pai_run.sh has not finished it)")
            continue
        tag = json.loads((Path(run) / "run.json").read_text()).get("tag", a.tag) if (Path(run) / "run.json").exists() else a.tag
        ev = Path(run) / "sim/eval-config.yaml"
        cor = next((float(x.split(":")[1]) for x in ev.read_text().splitlines() if "max_dist_to_gt_trajectory" in x), np.nan) if ev.exists() else np.nan
        for r in json.loads(sm.read_text())["rollouts"]:
            sc = r["clipgt_id"]
            ds = [p.parent for p in sorted((Path(run) / "sim/rollouts" / sc).glob("*/metrics.parquet")) if (p.parent / "rollout.asl").exists()]
            ds = [d for d in ds if d.name == r.get("rollout_id")] or ds       # a retried scene: the session of the summary row
            if not ds:
                say(f"skip {sc} of {run}: no session with rollout.asl and metrics.parquet")
            elif sc in seen:
                say(f"skip {sc} of {run}: already taken from an earlier run dir")
            elif not (a.scenes and not any(x in sc for x in a.scenes)):
                seen.add(sc)
                jobs.append(dict(run=run, scene=sc, session=ds[0].name, tag=tag, summary=r, corridor=cor, out=str(out), cold=a.cold, lht=a.lht,
                                 nurec=a.nurec, **{"pass": a.pass_}))
    return jobs


def _say(out: Path):
    log = open(out / "log.txt", "a", buffering=1)

    def say(msg, status=False):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True), log.write(line + "\n")
        if status:
            (out / "STATUS").write_text(line + "\n")
    return say


def _pool(fn, todo, n, say, line, what):
    """fn over todo in n spawned workers (polars / torch / OpenCV are loaded in the workers: no fork) -> failed scene ids."""
    t, done, bad = time.time(), 0, []
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
        for fu in as_completed([ex.submit(fn, j) for j in todo]):
            sc, meta, err = fu.result()
            done += 1
            if err:
                bad.append(sc)
                say(f"FAILED {sc}: {err.strip().splitlines()[-1]}")
            else:
                say(f"{sc}: " + line(sc, meta))
            if done % 8 == 0 or done == len(todo):
                say(f"{what}: {done}/{len(todo)} scenes, {len(bad)} failed, {time.time() - t:.0f} s", True)
    return bad, time.time() - t


def cmd_rows(a):
    out, stage = Path(a.out) / a.pass_, Path(a.stage) / a.pass_
    out.mkdir(parents=True, exist_ok=True), stage.mkdir(parents=True, exist_ok=True)
    say = _say(out)
    for f in ("DONE", "ERROR"):
        (out / f).unlink(missing_ok=True)
    jobs = jobs_of(a, out, say)
    for j in jobs:
        j["stage"] = str(stage)
    todo = [j for j in jobs if a.redo or not ((out / f"{j['scene']}.npz").exists() or (stage / f"{j['scene']}.npz").exists())][:a.limit or None]
    n = max(1, min(a.jobs, len(todo)))
    say(f"rows: {len(jobs)} scenes in finished run dirs, {len(jobs) - len(todo)} already built, staged or beyond --limit, building {len(todo)} with {n} workers", True)
    bad, dt = _pool(build, todo, n, say, lambda sc, m: f"{m['n_rows']} rows, {m['n_cold']} cold, G-in dump {m['gin_dump_eq']}/{m['gin_dump_n']} (+{len(m['gin_cold_dumps'])} cold "
                    f"pending) rec {m['gin_rec_eq']}/{m['gin_rec_n']}, {m['secs_stage1']} s" + (f", MAP STUBBED: {m['map_error']}" if m["map_error"] else ""), "rows") if todo else ([], 0.0)
    say(f"rows done: {len(todo) - len(bad)} staged, {len(bad)} failed, {dt:.0f} s wall, {n} workers; next: cold", True)


def cmd_cold(a):
    out, stage = Path(a.out) / a.pass_, Path(a.stage) / a.pass_
    say = _say(out)
    todo = [dict(stg=str(f), out=str(out)) for f in sorted(stage.glob("clipgt-*.npz"))][:a.limit or None]
    n = max(1, min(a.jobs, len(todo)))
    say(f"cold: {len(todo)} staged scenes, {n} workers", True)

    def line(sc, m):
        open(out / "times.jsonl", "a").write(json.dumps(dict(scene=sc, **{k: m[k] for k in ("n_rows", "secs_total", "secs_stage1", "secs_stage2", "secs", "bytes")})) + "\n")
        return (f"{m['n_rows']} rows, G-in dump {m['gin_dump_eq']}/{m['gin_dump_n']} rec {m['gin_rec_eq']}/{m['gin_rec_n']}, {m['cold_bytes_changed']} cold bytes replaced, "
                f"{m['secs_stage2']} s, {m['bytes'] / 2**20:.1f} MB")
    _, dt = _pool(finalize, todo, n, say, line, "cold") if todo else ([], 0.0)
    failed = sorted(p.stem for p in (out / "failed").glob("*.txt")) if (out / "failed").exists() else []
    msg = f"done: {len(list(out.glob('clipgt-*.npz')))} scene files in {out}, {len(failed)} failed, cold stage {dt:.0f} s wall, {n} workers"
    (out / ("ERROR" if failed else "DONE")).write_text(msg + "\n" + "\n".join(failed) + "\n")
    say(msg, True)


# ---------------------------------------------------------------- report
def pct(x, q=(5, 25, 50, 75, 95, 99, 100)):
    x = np.asarray(x, np.float64)
    return " / ".join(f"{v:.2f}" for v in np.percentile(x, q)) if len(x) else "n/a"


def cmd_report(a):
    out = Path(a.out) / a.pass_
    files = sorted(out.glob("clipgt-*.npz"))
    if not files:
        print("no scene files in", out)
        return
    A, per, metas = {}, [], []
    add = lambda k, v: A.setdefault(k, []).append(np.asarray(v))             # noqa: E731
    for f in files:
        z = np.load(f)
        m, vs = json.loads(str(z["meta"])), z["veh_sim"]
        metas.append(m)
        N = len(z["t0_us"])
        pok = z["path_ok"][:, 20:41]
        d = np.linalg.norm(z["box"][:, 0, :, None, :2] - z["path"][:, None, 20:41], axis=-1)        # box centre at t0 against the path 0 .. 20 m ahead
        d[np.broadcast_to(~pok[:, None], d.shape)] = np.inf
        eb = z["edge_box"]
        esd = np.where(z["edge_ok"], G.box_sdf(eb[..., :2].reshape(-1, 2), float(vs[2]), vs[0] / 2, vs[1] / 2).reshape(eb.shape[:2]), np.inf)
        for k in ("e0", "ds0", "v0", "after_event", "after_end", "dlat_c"):
            add(k, z[k])
        add("lead", (z["box_ok"][:, 0] & (d.min(-1) <= 2.0)).any(1)), add("anybox", z["box_ok"][:, 0].any(1)), add("nbox", z["box_ok"][:, 0].sum(1))
        add("edge3", esd.min(1) < 3.0), add("edge_any", z["edge_ok"].any(1)), add("stop60", (z["stop_s"] <= 60).any(1)), add("wl60", (z["wl_s"] <= 60).any(1))
        add("turn60", z["turn_s"] <= 60), add("cold", (z["slot"] < 0).any(1)), add("dlog", np.hypot(z["log0"][:, 0], z["log0"][:, 1]))
        add("imit_ok", z["imit_ok"].all(1)), add("past_end", z["s_ego"] > z["path_len"]), add("box_full", z["box_ok"].all(1)[:, 0] if N else np.zeros(0, bool))
        per.append(dict(scene=str(z["scene"]), n=N, score=float(z["score"]), fail=str(z["fail_reason"]), ev=z["t_event_us"], t0=z["t0_us"], m=m, has_stop=bool(z["has_stop"]),
                        has_edges=bool(z["has_edges"]), veh_eq=bool(np.allclose(z["veh"][:3], vs, atol=1e-3)), bytes=f.stat().st_size))
    A = {k: np.concatenate(v) for k, v in A.items()}
    R = len(A["e0"])
    sh = lambda m: f"{int(np.sum(m))} ({100 * np.mean(m):.1f}%)"             # noqa: E731
    nr = np.array([p["n"] for p in per])
    tm = [json.loads(x) for x in open(out / "times.jsonl")] if (out / "times.jsonl").exists() else []
    tm = list({t["scene"]: t for t in tm if t["scene"] in {p["scene"] for p in per}}.values())
    gd, ge = sum(m["gin_dump_n"] for m in metas), sum(m["gin_dump_eq"] for m in metas)
    rd, re_ = sum(m["gin_rec_n"] for m in metas), sum(m["gin_rec_eq"] for m in metas)
    fin = lambda k: [m[k] for m in metas if np.isfinite(m[k])]               # noqa: E731
    failed = sorted(p.stem for p in (out / "failed").glob("*.txt")) if (out / "failed").exists() else []
    L = [f"# ONPOL-PAI rows, pass `{a.pass_}`", "",
         f"Generated by `experiments/onpol_pai/scripts/onp_rows.py report` on {time.strftime('%F %T')} from `{out}`. Format: [rows_format.md](rows_format.md).", "",
         "## Size", "", "| | |", "|---|---|",
         f"| scenes built | {len(per)} |", f"| scenes failed | {len(failed)}{' (' + ', '.join(x[7:15] for x in failed) + ')' if failed else ''} |",
         f"| rows | {R} |", f"| rows per scene, min / median / max | {nr.min()} / {int(np.median(nr))} / {nr.max()} |",
         f"| rows after a scorer event (`after_event`) | {sh(A['after_event'])} |",
         f"| rows after the scorer stopped reading the rollout (`after_end`: first offroad, any collision or corridor exit) | {sh(A['after_end'])} |",
         f"| scenes with a scorer event (at-fault collision / offroad / left corridor) | {' / '.join(str(sum(p['ev'][i] != G.NEVER for p in per)) for i in range(3))} |",
         f"| scene score, mean | {np.nanmean([p['score'] for p in per]):.4f} |",
         f"| file size per scene, mean / max | {np.mean([p['bytes'] for p in per]) / 2**20:.1f} / {np.max([p['bytes'] for p in per]) / 2**20:.1f} MB |",
         f"| total size | {sum(p['bytes'] for p in per) / 2**30:.2f} GB |"]
    if tm:
        L += [f"| seconds per scene (one worker's wall), mean / max | {np.mean([t['secs_total'] for t in tm]):.1f} / {np.max([t['secs_total'] for t in tm]):.1f} |",
              "| of which, mean: " + ", ".join(sorted({k for t in tm for k in t["secs"]})) + " | " + ", ".join(
                  f"{np.mean([t['secs'].get(k, 0) for t in tm]):.1f}" for k in sorted({k for t in tm for k in t['secs']})) + " |"]
    L += ["", "## Gate G-in", "", "| check | compared | equal |", "|---|--:|--:|",
          f"| PAI_DUMP files: `cur`, `valid`, `real` bit for bit (both the replayed `cur` and `frames[slot]` / `cold_cur` as stored), dumped `ego` vector, t0 | {gd} | {ge} |",
          f"| drive.jsonl decisions: `hist` = P at 4 decimals, `ego` = ego_in at 5 decimals, `anchor` at 4 decimals, command, t0 | {rd} | {re_} |", "",
          f"G-in {'PASSES' if gd == ge and rd == re_ and rd > 0 else 'FAILS'}." + ("" if gd else " No PAI_DUMP file was found: only the drive.jsonl half was checked.")]
    for m, p in zip(metas, per):
        if m["gin_dump_diff"] or m["gin_rec_diff_k"]:
            L.append(f"- {p['scene']}: dump differences {json.dumps(m['gin_dump_diff'])}; drive.jsonl differences at k {m['gin_rec_diff_k']}")
    L += ["", "## Self-checks of the label frames", "", "| check | value |", "|---|---|",
          f"| logged rig path + rig-to-centre offset against the trafficsim's logged EGO track, max over scenes | {max(fin('chk_logged_centre_m'), default=float('nan')):.4f} m ({len(fin('chk_logged_centre_m'))} scenes comparable) |",
          f"| `dlat_c` against the scorer's own `lateral_dist_to_gt_trajectory` series at the same steps, max over scenes | {max(fin('chk_scorer_lateral_m'), default=float('nan')):.4f} m ({len(fin('chk_scorer_lateral_m'))} scenes) |",
          f"| ego box centre from the driver's anchor against the simulator's EGO actor pose at t0, max over scenes | {max(fin('chk_anchor_vs_sim_m'), default=float('nan')):.4f} m |",
          f"| cold-start rows: bytes of the synthesised slots replaced by the driver image's own warp | {sum(m.get('cold_bytes_changed', 0) for m in metas)} |",
          f"| summary zero flags agree with the per-step series (`t_event_us` finite) | {sum(m['chk_event_flags'] for m in metas)} of {len(metas)} scenes |",
          f"| `veh` (start.jsonl) equals `veh_sim` (rollout metadata) | {sum(p['veh_eq'] for p in per)} of {len(per)} scenes |",
          f"| road-edge segments within 40 m, all levels / at the ego's level | {sum(m['edge_segs_40m'] for m in metas)} / {sum(m['edge_segs_40m_same_level'] for m in metas)} |",
          "", "## Label items: real or stubbed", "", "| item | status |", "|---|---|",
          "| inputs, `box`, logged path, `imit`, turns, scorer events | real in every scene built |",
          f"| road edges (`edge_box`, `edges_xyz`) | real in {sum(p['has_edges'] for p in per)} of {len(per)} scenes |",
          f"| wait lines (`stop_*`, `wl_*`) | real in {sum(p['has_stop'] for p in per)} of {len(per)} scenes (static lines; PAI has no light state) |"]
    for m, p in zip(metas, per):
        if m["map_error"]:
            L.append(f"| {p['scene']} map stubbed | {m['map_error']} |")
    L += ["", "## Composition (rows; classes overlap)", "", "| class | rows |", "|---|--:|",
          f"| a simulator box at t0 whose centre is within 2 m of the logged path 0 to 20 m ahead of the rear axle | {sh(A['lead'])} |",
          f"| any simulator box at t0 (mean {A['nbox'].mean():.1f} of 32 slots) | {sh(A['anybox'])} |",
          f"| box horizon complete (nearest box valid at all 9 times) | {sh(A['box_full'])} |",
          f"| a road-edge segment centre within 3 m of the ego box (`veh_sim`) | {sh(A['edge3'])} |",
          f"| any road-edge segment within 40 m | {sh(A['edge_any'])} |",
          f"| an ELIGIBLE stop line ahead within 60 m (`stop_s`) | {sh(A['stop60'])} |",
          f"| any wait line ahead within 60 m (`wl_s`) | {sh(A['wl60'])} |",
          f"| a >= 45 deg turn peak ahead within 60 m | {sh(A['turn60'])} |",
          f"| cold start (a slot without a rendered frame) | {sh(A['cold'])} |",
          f"| `imit` valid at all 8 poses | {sh(A['imit_ok'])} |",
          f"| ego projection past the end of the logged path | {sh(A['past_end'])} |"]
    L += [f"| speed {lo} to {hi} m/s | {sh((A['v0'] >= lo) & (A['v0'] < hi))} |" for lo, hi in SPEED]
    L += ["", "## How far the visited states are from the log", "", "| quantity | value |", "|---|---|",
          f"| abs(e0) percentiles 5 / 25 / 50 / 75 / 95 / 99 / max (m) | {pct(np.abs(A['e0']))} |",
          f"| ds0 percentiles 5 / 25 / 50 / 75 / 95 / 99 / max (m; positive = ahead of the log) | {pct(A['ds0'])} |",
          f"| 2-D distance to the log's same-time pose, same percentiles (m) | {pct(A['dlog'])} |",
          f"| rows with abs(e0) > 0.5 m | {sh(np.abs(A['e0']) > 0.5)} |", f"| rows with abs(e0) > 1.0 m | {sh(np.abs(A['e0']) > 1.0)} |",
          f"| rows with abs(e0) > 2.0 m | {sh(np.abs(A['e0']) > 2.0)} |",
          f"| rows more than 0.5 m from the log's same-time pose (2-D) | {sh(A['dlog'] > 0.5)} |",
          f"| rows more than 2.0 m from the log's same-time pose (2-D) | {sh(A['dlog'] > 2.0)} |",
          f"| rows with the scorer's lateral distance `dlat_c` >= 4.0 m | {sh(A['dlat_c'] >= 4.0)} |", "",
          "## Scenes", "", "| scene | rows | score | failure | rows after event | G-in dump | G-in rec | actors | edges | wait lines |", "|---|--:|--:|---|--:|--:|--:|--:|--:|--:|"]
    L += [f"| {p['scene'][7:15]} | {p['n']} | {p['score']:.3f} | {p['fail']} | {int((p['t0'] >= p['ev'].min()).sum())} | {p['m']['gin_dump_eq']}/{p['m']['gin_dump_n']} | "
          f"{p['m']['gin_rec_eq']}/{p['m']['gin_rec_n']} | {p['m']['n_actors']} | {p['m']['n_edges']} | {p['m']['n_wait_lines']} |" for p in per[:a.max_scenes]]
    if len(per) > a.max_scenes:
        L.append(f"\n({len(per) - a.max_scenes} more scenes not listed.)")
    md = Path(a.md or out.parent / f"rows_{a.pass_}.md")
    md.write_text("\n".join(L) + "\n")
    print("\n".join(L[:400]))
    print("->", md)


# ---------------------------------------------------------------- BEV check figures
def cmd_fig(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    out = Path(a.out) / "fig"
    out.mkdir(parents=True, exist_ok=True)

    def rect(ax, x, y, yaw, ln, w, **kw):
        c, s = np.cos(yaw), np.sin(yaw)
        p = np.array([[ln / 2, w / 2], [ln / 2, -w / 2], [-ln / 2, -w / 2], [-ln / 2, w / 2]]) @ np.array([[c, s], [-s, c]]) + [x, y]
        ax.add_patch(Polygon(np.c_[-p[:, 1], p[:, 0]], closed=True, **kw))                 # plot: forward up, left to the left
    for f in sorted((Path(a.out) / a.pass_).glob("clipgt-*.npz")):
        png = out / f"{a.pass_}_{f.stem[7:15]}.png"
        if (a.scenes and not any(x in f.stem for x in a.scenes)) or (png.exists() and png.stat().st_mtime > f.stat().st_mtime):
            continue
        z = np.load(f)
        N, vs = len(z["t0_us"]), z["veh_sim"]
        fig, axs = plt.subplots(2, 3, figsize=(21, 14), gridspec_kw=dict(height_ratios=[1, 3.2]))
        for c, n in enumerate((N // 5, N // 2, (4 * N) // 5)):
            i = z["slot"][n, -1]
            fr = z["frames"][i] if i >= 0 else z["cold_cur"][list(z["cold_rows"]).index(n), -1]
            im = np.zeros((2, 256, 512), np.uint8)
            im[:, 0::2, 0::2], im[:, 1::2, 0::2], im[:, 0::2, 1::2], im[:, 1::2, 1::2] = fr[:, 0], fr[:, 1], fr[:, 2], fr[:, 3]
            axs[0, c].imshow(np.concatenate([im[0], im[1]], 1), cmap="gray"), axs[0, c].axis("off")
            axs[0, c].set_title(f"k {z['k'][n]}  v0 {z['v0'][n]:.1f} m/s  e0 {z['e0'][n]:+.2f} m  ds0 {z['ds0'][n]:+.1f} m  road | wide (newest slot)")
            ax = axs[1, c]
            for b in z["edge_box"][n][z["edge_ok"][n]]:
                rect(ax, *b, fc="0.4", ec="0.4", lw=1.5)
            ok = z["path_ok"][n]
            ax.plot(-z["path"][n, ok, 1], z["path"][n, ok, 0], "-", c="tab:blue", lw=1, label="logged path")
            for jt, col in ((8, "mistyrose"), (4, "lightsalmon"), (0, "tab:red")):
                for kk in np.flatnonzero(z["box_ok"][n, jt]):
                    rect(ax, *z["box"][n, jt, kk], fc=col if jt == 0 else "none", ec=col, lw=1, alpha=0.8)
                    if jt == 0 and np.hypot(*z["box"][n, 0, kk, :2]) < 60:
                        ax.text(-z["box"][n, 0, kk, 1], z["box"][n, 0, kk, 0], str(z["box_label"][n, kk])[:6], fontsize=6, clip_on=True)
            rect(ax, float(vs[2]), 0, 0, vs[0], vs[1], fc="none", ec="k", lw=1.5)
            ax.plot(-z["poses"][n, :, 1], z["poses"][n, :, 0], "o-", c="tab:green", ms=4, label="served plan")
            ax.plot(-z["poses_model"][n, :, 1], z["poses_model"][n, :, 0], ".:", c="darkgreen", ms=3, label="adapter plan")
            m = z["imit_ok"][n]
            ax.plot(-z["imit"][n, m, 1], z["imit"][n, m, 0], "s-", c="tab:orange", ms=4, mfc="none", label="imit")
            m = z["log_fut_ok"][n]
            ax.plot(-z["log_fut"][n, m, 1], z["log_fut"][n, m, 0], "x", c="tab:blue", ms=5, label="log at t0 + 0.5 k")
            for s_, kd, el in [(s_, kd, False) for s_, kd in zip(z["wl_s"][n], z["wl_kind"][n])] + [(s_, kd, True) for s_, kd in zip(z["stop_s"][n], z["stop_kind"][n])]:
                if np.isfinite(s_) and s_ <= 100:
                    p = z["path"][n, int(np.clip(round(s_) + 20, 0, 120))]
                    ax.plot(-p[1], p[0], "*" if el else "_", c="magenta", ms=16 if el else 20, mew=2)
                    ax.text(-p[1] + 1, p[0], f"{'STOP elig' if el else 'wl'} kind {kd} {s_:.0f} m", fontsize=7, c="magenta", clip_on=True)
            if np.isfinite(z["turn_s"][n]):
                p = z["path"][n, int(np.clip(round(float(z["turn_s"][n])) + 20, 0, 120))]
                ax.plot(-p[1], p[0], "D", c="purple", ms=8, mfc="none"), ax.text(-p[1] + 1, p[0] - 2, f"turn peak, log {z['turn_vpk'][n]:.1f} m/s", fontsize=7, c="purple", clip_on=True)
            ax.set_xlim(-35, 35), ax.set_ylim(-20, 80), ax.set_aspect("equal"), ax.grid(alpha=0.3), ax.legend(loc="lower right", fontsize=7)
            ax.set_xlabel("left <- y (m) -> right"), ax.set_ylabel("x forward (m)")
        fig.suptitle(f"{z['scene']}  pass {z['pass']}  score {float(z['score']):.3f} {z['fail_reason']}   red = simulator boxes at t0 (salmon: +2 s, +4 s), "
                     "grey = road-edge segments, black = ego box")
        fig.tight_layout()
        fig.savefig(png, dpi=70)
        plt.close(fig)
        print(png, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rows", "cold", "report", "fig"):
        p = sp.add_parser(name)
        p.add_argument("--pass", dest="pass_", required=True, help="pass name: pol = the on-policy rollouts")
        p.add_argument("--out", default=os.environ.get("ONP_OUT") or str(D / "runs/onpol_pai/rows"))
        p.add_argument("--stage", default=os.environ.get("ONP_STAGE", "/tmp/onp_stage"), help="staging dir between the two stages (not on the data disk)")
        p.add_argument("--scenes", nargs="*", help="only scenes whose id contains one of these")
        p.add_argument("--md", help="report file (default <out>/rows_<pass>.md)")
        p.add_argument("--max-scenes", type=int, default=200, help="rows of the report's scene table")
        p.add_argument("--jobs", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 8)), help="worker processes (leaves 8 threads for the simulators)")
        p.add_argument("--limit", type=int, default=0)
    p = sp.choices["rows"]
    p.add_argument("--runs", nargs="+", required=True, help="pai_run.sh run dirs (globs)")
    p.add_argument("--tag", default="P2H10S-F-s0", help="checkpoint tag when the run dir has no run.json")
    p.add_argument("--cold", default="backwarp", help="PAI_COLD the rollouts were served with")
    p.add_argument("--lht", action="store_true", help="PAI_LHT=1 was served")
    p.add_argument("--nurec", default=str(D / "datasets/nurec"))
    p.add_argument("--redo", action="store_true", help="rebuild scenes that already have a file (it is replaced atomically)")
    a = ap.parse_args()
    {"rows": cmd_rows, "cold": cmd_cold, "report": cmd_report, "fig": cmd_fig}[a.cmd](a)
