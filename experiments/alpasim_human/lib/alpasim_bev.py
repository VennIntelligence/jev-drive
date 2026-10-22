#!/usr/bin/env python
"""AlpaSim's own top view (BEV) of the live rollout for the human viewer's inset, with the evaluator's offroad and corridor verdicts.

Display only and privileged: it reads the runtime's rollout log (sim/rollouts/<scene>/<session>/rollout.asl, written as the rollout runs)
and the scene map the runtime loads, never anything the driver gets, and nothing it computes goes back to a driver or model. Everything
is AlpaSim's code at the source checkout (third_party/alpasim, src/eval and src/runtime), called as shipped:

  log          eval.accumulator.EvalDataAccumulator fed every LogEntry but the camera images (the eval's asl_loader path), tailed as
               the file grows
  map          alpasim_runtime.scene_loader.build_scene_loader(<sim>/generated-user-config-0.yaml).get_data_source(scene).map: the
               runtime's vector map (unbound_rollout: vector_map=data_source.map), PAI from the scene's USDZ, nuPlan from trajdata
  rule         eval.scorers.offroad.OffRoadScorer and eval.scorers.ground_truth.GroundTruthScorer with <sim>/eval-config.yaml, run
               on a SimulationResult (SimulationResult.from_scenario_input) cut to the last three steps: per step the same lanes /
               road edges / road areas, ego footprint (RAABB with the config's shrink and corner rounding) and corridor
               (aggregation_modifiers.max_dist_to_gt_trajectory, 4 m) as the scored run; GroundTruthScorer's first two steps
               are fixed at 0, so the third is the one read
  picture      eval.video's map panel: ShapelyMap.from_vec_map(...).render (lane centres blue dashed, lane edges blue, road edges
               black, stop lines red dashed), the recorded log path (GT_LINESTRING, green) and the log ego's box
               (EGO_GT_GHOST_POLYGON, lime), ActorPolygons.render_at_time (agents grey, ego lime), the driver's selected plan
               (DriverResponses.render_at_time) and the route (Routes.render_at_time); rotated so the ego faces up and placed by
               ActorPolygons.set_axis_limits_around_agent with the config's map_video (ego_loc BOTTOM_CENTER), radius RADIUS_M
Added here only: the ego box red (fill and outline) while OffRoadScorer says offroad, orange while GroundTruthScorer says
left_corridor_laterally (red wins); the log path and the log ego's box only while the viewer's Shadow is on.

Live mode (started by scripts/human_viewer.py in the AlpaSim runtime image with the runtime's mounts, one container per live dir):
reads <live>/state.json (run, scene, session, sim_t) and <live>/viewer.json (heartbeat, shadow, bev_px), renders the step at the
viewer's sim time (first drive request + sim_t, the driver's own clock) and writes <live>/bev.rgba (bev_px^2 RGBA) + <live>/bev.json
{session, ts, sim_t, px, offroad, corridor, lat_m, scored (after force-GT), ms}. Exits when the viewer's heartbeat is 30 s old.

  python alpasim_bev.py live --dir <live> [--asl <rollout.asl> ...]     # --asl: replays (scripts/replay_asl.py) have no run dir
  python alpasim_bev.py check <rollout.asl> [--png <dir> --at <ts|onset> ...]   # per-step flags vs the full evaluator, frames
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import struct
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from alpasim_grpc.v0.logging_pb2 import LogEntry  # noqa: E402
from alpasim_utils.yaml_utils import typed_parse_config  # noqa: E402
from eval.accumulator import EvalDataAccumulator  # noqa: E402
from eval.data import Routes, SimulationResult  # noqa: E402
from eval.schema import EvalConfig  # noqa: E402
from eval.scorers.ground_truth import GroundTruthScorer  # noqa: E402
from eval.scorers import offroad as OR  # noqa: E402
from eval.scorers.offroad import OffRoadScorer  # noqa: E402
from shapely.plotting import patch_from_polygon  # noqa: E402
from trajdata.maps.vec_map_elements import MapElementType  # noqa: E402
from eval.video import get_ego_transform  # noqa: E402
from eval.video_data import ShapelyMap  # noqa: E402

RADIUS_M = 25.0                       # half the view (eval video: map_radius_m 20); 50 m square, ego 12.5 m above the bottom edge
RED, ORANGE = "#f03c32", "#ff9632"
CAM = LogEntry.DESCRIPTOR.fields_by_name["driver_camera_image"].number
WINDOW = 3                            # steps per verdict (GroundTruthScorer computes from the third step on)


def sim_dir(asl: str) -> Path:
    return Path(asl).resolve().parents[3]


def eval_cfg(sim: Path, radius: float = RADIUS_M) -> EvalConfig:
    cfg = typed_parse_config(sim / "eval-config.yaml", EvalConfig)
    cfg.video.map_video.map_radius_m = radius
    return cfg


_LOADERS: dict = {}
_MAPS: dict = {}


def scene_map(sim: Path, scene: str):
    """The runtime's vector map for `scene` from the run's own user config (cached per config and scene)."""
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_runtime.scene_loader import build_scene_loader
    uc = sim / "generated-user-config-0.yaml"
    if not uc.exists():                                      # an exam stack's sim dir: rollouts here, configs in its resident stack
        uc = Path((sim.parent / "resident").read_text().strip()) / "sim" / uc.name
    key = (str(uc), scene)
    if key not in _MAPS:
        sig = uc.read_text().split("\nscenes:")[0]          # the provider block; the scene list differs per stack
        if sig not in _LOADERS:
            _LOADERS[sig] = build_scene_loader(typed_parse_config(uc, UserSimulatorConfig))
        vm = _LOADERS[sig].get_data_source(scene).map
        _MAPS[key] = (vm, ShapelyMap.from_vec_map(vm))
    return _MAPS[key]


class Tail:
    """An ASL file read as it grows into an EvalDataAccumulator (camera images skipped, as the map panel does not use them)."""

    def __init__(self, path: str, cfg: EvalConfig):
        self.path, self.f, self.acc = path, open(path, "rb"), EvalDataAccumulator(cfg=cfg)
        self.session = self.scene = None
        self.t_first = None                                  # the first drive request: the driver's sim clock origin
        self.ego_ts: list[int] = []

    def read(self) -> int:
        n = 0
        while True:
            pos = self.f.tell()
            h = self.f.read(4)
            if len(h) < 4:
                self.f.seek(pos)
                return n
            size = struct.unpack(">L", h)[0]
            b = self.f.read(size)
            if len(b) < size:                                # a record still being written
                self.f.seek(pos)
                return n
            if b and b[0] >> 3 == CAM:                       # field tag of the oneof: skip without parsing
                continue
            e = LogEntry.FromString(b)
            k = e.WhichOneof("log_entry")
            if k == "driver_session_request":
                self.session = e.driver_session_request.session_uuid
            elif k == "rollout_metadata":
                self.scene = e.rollout_metadata.session_metadata.scene_id
            elif k == "driver_request" and self.t_first is None:
                self.t_first = int(e.driver_request.time_now_us)
            elif k == "actor_poses" and any(p.actor_id == "EGO" for p in e.actor_poses.actor_poses):
                self.ego_ts.append(int(e.actor_poses.timestamp_us))
            self.acc.handle_message(e)
            n += 1

    def result(self, ts: int, vec_map) -> SimulationResult:
        """SimulationResult over the last WINDOW ego steps up to ts (exact steps, as the scorers need)."""
        a = self.acc
        i = int(np.searchsorted(self.ego_ts, ts, side="right"))
        keep = set(self.ego_ts[max(0, i - WINDOW):i])
        lo = min(keep)
        saved = a._actor_trajectory_data, a._driver_responses, a._routes
        try:
            cut = lambda v: v[max(0, bisect_right(v, ts, key=lambda p: p[0]) - 2 * WINDOW):bisect_right(v, ts, key=lambda p: p[0])]  # noqa: E731
            a._actor_trajectory_data = {k: [p for p in cut(v) if p[0] in keep] for k, v in saved[0].items()}
            a._driver_responses = [r for r in cut(saved[1]) if lo <= r[0]]
            r = Routes()
            for t, rr in zip(saved[2].timestamps_us, saved[2].routes_in_rig_frame):
                if lo <= t <= ts:
                    r.timestamps_us, r.routes_in_rig_frame = [t], [rr]
            a._routes = r
            si = a.build_scenario_eval_input(run_uuid="live", run_name="live", vec_map=vec_map)
        finally:
            a._actor_trajectory_data, a._driver_responses, a._routes = saved
        return SimulationResult.from_scenario_input(si, a.cfg)


def verdict(sr: SimulationResult, cfg: EvalConfig) -> dict:
    m = {x.name: x.values[-1] for x in OffRoadScorer(cfg).calculate(sr) + GroundTruthScorer(cfg).calculate(sr)}
    ts = int(sr.timestamps_us[-1])
    fd = sr.first_driven_timestamp_us
    return {"ts": ts, "offroad": bool(m["offroad"]), "corridor": bool(m["left_corridor_laterally"]),
            "lat_m": round(float(m["lateral_dist_to_gt_trajectory"]), 2), "scored": fd is None or ts >= fd}


class Bev:
    """eval.video's map panel on one square Agg figure; artists are kept across frames as create_video_animation's update does."""

    def __init__(self, px: int = 400):
        self.px = 0
        self.size(px)

    def size(self, px: int):
        if px == self.px:
            return
        self.px = px
        self.fig = Figure(figsize=(px / 100, px / 100), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.reset(None)

    def reset(self, smap):
        self.ax.cla()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")
        if smap is not None:                                 # the map's artists went with cla
            for i in smap.currently_rendered_linestring_ids:
                smap.renderable_linestrings[i].artists = None
            smap.currently_rendered_linestring_ids = np.array([], int)
        self.smap, self.keep, self.gt, self.area = smap, {}, None, (None, [])

    def road_area(self, sr: SimulationResult, ts: int, R: float) -> list:
        """Road areas (nuPlan: its offroad test when the map has no road edges) as a light fill under the map lines, from the
        evaluator's own query and union (offroad._road_areas_near_ego, _road_area_union); redrawn only when the set changes."""
        if not OR._has_map_element_kdtree(sr, MapElementType.ROAD_AREA):
            return []
        p = sr.actor_trajectories["EGO"].interpolate_pose(ts)
        areas = OR._road_areas_near_ego(sr, np.array([*p.vec3, p.yaw()]), R + 20) or []
        ids = frozenset(a.id for a in areas)
        if ids != self.area[0]:
            for a in self.area[1]:
                a.remove()
            u = OR._road_area_union(sr, areas) if areas else None
            polys = [] if u is None or u.is_empty else list(getattr(u, "geoms", [u]))
            self.area = (ids, [self.ax.add_patch(patch_from_polygon(q, facecolor="0.88", edgecolor="none", zorder=0))
                               for q in polys if q.geom_type == "Polygon"])
        return self.area[1]

    def draw(self, sr: SimulationResult, cfg: EvalConfig, v: dict, shadow: bool) -> bytes:
        ax, ts, R = self.ax, v["ts"], cfg.video.map_video.map_radius_m + 10
        tf = get_ego_transform(sr, cfg, ts)
        center = sr.actor_polygons.set_axis_limits_around_agent(ax, "EGO", ts, cfg, axis_transform=tf)
        arts = {"map": self.smap.render(ax, cfg, center=center, max_dist=R), "area": {"area": self.road_area(sr, ts, R)}}
        for name, obj, call in (("agents", sr.actor_polygons, lambda o: o.render_at_time(ax, ts, center=center, max_dist=R)),
                                ("plan", sr.driver_responses, lambda o: o.render_at_time(ax, ts, "now")),
                                ("route", sr.routes, lambda o: o.render_at_time(ax, ts))):
            obj.artists = self.keep.get(name, {} if name == "agents" else None)
            arts[name] = call(obj)
            self.keep[name] = obj.artists
        g = sr.ego_recorded_ground_truth_trajectory
        if self.gt is None:                                  # the whole log path, once per session
            self.gt = g.set_linestring_plot_style("gt_linestring", linewidth=1, style="g-", alpha=0.7).render_linestring(ax)
        g.set_polygon_plot_style(fill_color="limegreen")
        g.polygon_artists = self.keep.get("ghost")
        arts["ghost"] = g.render_polygon_at_time(ax, ts)
        self.keep["ghost"] = g.polygon_artists
        arts["gt"] = self.gt
        for a in [x for d in arts.values() for lst in d.values() for x in (lst if isinstance(lst, list) else [lst])]:
            a.set_transform(tf + ax.transData)
        for a in [x for d in (arts["gt"], arts["ghost"]) for lst in d.values() for x in lst]:
            a.set_visible(shadow)
        ego = (arts["agents"] or {}).get("EGO")
        if ego:
            col = RED if v["offroad"] else ORANGE if v["corridor"] else None
            ego[0].set_color(col or "k")
            ego[0].set_linewidth(2.5 if col else 1)
            ego[1].set_color(col or "limegreen")
            ego[1].set_alpha(0.75 if col else 0.3)
            ego[1].set_zorder(5)
            ego[0].set_zorder(6)
        self.canvas.draw()
        return bytes(self.canvas.buffer_rgba())


def put(path: Path, data: bytes):
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    tmp.write_bytes(data)
    os.replace(tmp, path)


def find_asl(st: dict, extra: list[str]) -> str | None:
    sess, scene = st.get("session"), st.get("scene")
    cands = list(extra)
    if st.get("run"):
        for pat in ("sim/rollouts", "stacks/*/sim/rollouts"):
            cands += glob.glob(os.path.join(st["run"], pat, scene or "*", sess or "*", "rollout.asl"))
    if not sess:                                             # a driver without "session" in state.json: the scene's newest rollout
        cands = [c for c in cands if Path(c).parent.parent.name == scene or c in extra]
        return max(cands, key=os.path.getmtime, default=None)
    return next((c for c in cands if Path(c).parent.name == sess), None)


def live(d: Path, extra: list[str]):
    t0, cur, bev, cfg, smap, vm, last = time.time(), None, Bev(), None, None, None, None
    while True:
        try:
            vj = json.loads((d / "viewer.json").read_text())
        except (OSError, ValueError):
            vj = {}
        if time.time() - vj.get("t", 0) > 30 and time.time() - t0 > 30:
            print("viewer gone: exit", flush=True)
            return
        try:
            st = json.loads((d / "state.json").read_text())
        except (OSError, ValueError):
            st = {}
        sess = st.get("session") or st.get("scene")         # older drivers: the scene names the rollout
        if sess and (cur is None or cur.key != sess) and not st.get("closed"):
            asl = find_asl(st, extra)
            if asl and (cur is None or cur.path != asl):
                put(d / "bev.json", json.dumps({"t": time.time(), "session": st.get("session"), "state": "loading map"}).encode())
                try:
                    t1 = time.time()
                    cfg = eval_cfg(sim_dir(asl))
                    cur = Tail(asl, cfg)
                    cur.key = sess
                    vm, smap = scene_map(sim_dir(asl), st["scene"])
                    bev.reset(smap)
                    last = None
                    print("session %s %s: map in %.1f s (%s)" % (sess, st["scene"], time.time() - t1, asl), flush=True)
                except Exception as e:  # noqa: BLE001  (display only: report and retry on the next session)
                    print("load failed: %r" % (e,), flush=True)
                    put(d / "bev.json", json.dumps({"t": time.time(), "session": st.get("session"), "state": "error: %r" % (e,)}).encode())
                    cur = None
                    time.sleep(2)
                    continue
        if cur is None or cur.key != sess:
            time.sleep(0.1)
            continue
        cur.read()
        px = int(vj.get("bev_px") or 400)
        shadow = bool(vj.get("shadow"))
        if cur.t_first is None or len(cur.ego_ts) < WINDOW:
            time.sleep(0.05)
            continue
        target = cur.t_first + int(round(1e6 * (st.get("sim_t") or 0))) + 5000
        i = int(np.searchsorted(cur.ego_ts, target, side="right"))
        if i < WINDOW:
            time.sleep(0.05)
            continue
        ts = cur.ego_ts[i - 1]
        if (ts, px, shadow) == last:
            time.sleep(0.02)
            continue
        try:
            t1 = time.time()
            sr = cur.result(ts, vm)
            v = verdict(sr, cfg)
            t2 = time.time()
            bev.size(px)
            if bev.smap is None:
                bev.reset(smap)
            img = bev.draw(sr, cfg, v, shadow)
            t3 = time.time()
        except Exception as e:  # noqa: BLE001
            print("frame %d failed: %r" % (ts, e), flush=True)
            last = (ts, px, shadow)
            continue
        put(d / "bev.rgba", img)
        put(d / "bev.json", json.dumps({"t": time.time(), "session": st.get("session"), "scene": st.get("scene"), "state": "ok", "px": px,
                                        "sim_t": round((ts - cur.t_first) / 1e6, 2), **v,
                                        "ms_eval": round(1e3 * (t2 - t1), 1), "ms_render": round(1e3 * (t3 - t2), 1)}).encode())
        last = (ts, px, shadow)


def check(asl: str, png: str | None, at: list[str], px: int):
    """The evaluator's own per-step offroad / corridor over the whole rollout (asl_loader + from_scenario_input + the two scorers)
    against this module's three-step verdict at every step; onsets in the scored window; optional frames."""
    from eval.asl_loader import load_scenario_eval_input_from_asl
    sim = sim_dir(asl)
    cfg = eval_cfg(sim)
    t = Tail(asl, cfg)
    t.read()
    vm, smap = scene_map(sim, t.scene)

    class A:                                                 # artifacts[scene_id].map as asl_loader reads it
        map = vm
    si = asyncio.run(load_scenario_eval_input_from_asl(asl, cfg, {t.scene: A}, {"run_uuid": "x", "run_name": "x"}))
    full = SimulationResult.from_scenario_input(si, cfg)
    fm = {x.name: x.values for x in OffRoadScorer(cfg).calculate(full) + GroundTruthScorer(cfg).calculate(full)}
    fts = [int(x) for x in full.timestamps_us]
    fd = full.first_driven_timestamp_us
    rows, ms = [], []
    for k, ts in enumerate(fts):
        if k < WINDOW - 1:
            continue
        t1 = time.time()
        v = verdict(t.result(ts, vm), cfg)
        ms.append(1e3 * (time.time() - t1))
        rows.append((ts, bool(fm["offroad"][k]), v["offroad"], bool(fm["left_corridor_laterally"][k]), v["corridor"],
                     fm["lateral_dist_to_gt_trajectory"][k], v["lat_m"], fd is None or ts >= fd))
    on = lambda c: next((r[0] for r in rows if r[7] and r[c]), None)  # noqa: E731
    mis = [r for r in rows if r[1] != r[2] or r[3] != r[4]]
    out = {"asl": asl, "scene": t.scene, "steps": len(rows), "first_driven": fd, "t_first": t.t_first,
           "offroad_onset_eval": on(1), "offroad_onset_live": on(2), "corridor_onset_eval": on(3), "corridor_onset_live": on(4),
           "mismatched_steps": len(mis), "max_lat_diff_m": round(max(abs(r[5] - r[6]) for r in rows), 3),
           "eval_offroad_any_scored": any(r[1] for r in rows if r[7]), "eval_corridor_any_scored": any(r[3] for r in rows if r[7]),
           "verdict_ms_median": round(float(np.median(ms)), 1)}
    print(json.dumps(out), flush=True)
    if png:
        os.makedirs(png, exist_ok=True)
        bev = Bev(px)
        bev.reset(smap)
        want = []
        for a in at:
            if a == "onset":
                want += [x for x in (out["offroad_onset_eval"], out["corridor_onset_eval"]) if x]
                o = out["offroad_onset_eval"] or out["corridor_onset_eval"]
                if o:                                        # one step before the onset and 2 s before (on the road)
                    j = fts.index(o)
                    want += [fts[j - 1], fts[max(WINDOW - 1, j - 20)]]
            else:
                want.append(int(a))
        from PIL import Image
        rms = []
        for ts in sorted(set(want)):
            sr = t.result(ts, vm)
            v = verdict(sr, cfg)
            t1 = time.time()
            img = bev.draw(sr, cfg, v, True)
            rms.append(1e3 * (time.time() - t1))
            f = os.path.join(png, "%s_%+.1fs_%s.png" % (Path(asl).parent.name[:8], (ts - t.t_first) / 1e6,
                                                         "offroad" if v["offroad"] else "corridor" if v["corridor"] else "onroad"))
            Image.frombuffer("RGBA", (px, px), img).save(f)
            print("frame", f, json.dumps(v), flush=True)
        if rms:
            print("render ms", [round(x, 1) for x in rms], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("live")
    a.add_argument("--dir", required=True)
    a.add_argument("--asl", nargs="*", default=[])
    b = sp.add_parser("check")
    b.add_argument("asl", nargs="+")
    b.add_argument("--png")
    b.add_argument("--at", nargs="*", default=[])
    b.add_argument("--px", type=int, default=400)
    x = ap.parse_args()
    if x.cmd == "live":
        live(Path(x.dir), x.asl)
    else:
        for f in x.asl:
            check(f, x.png, x.at, x.px)


if __name__ == "__main__":
    sys.exit(main())
