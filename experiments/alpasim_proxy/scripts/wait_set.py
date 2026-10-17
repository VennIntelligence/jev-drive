#!/usr/bin/env python3
"""WAIT-SET: a local closed-loop set where the logged human waits at standstill (red light, queue) and the ego must stay put, built
from public AlpaSim nuPlan scenes whose human starts from a standstill, with the scene window moved so that the human is still waiting
in it (experiments/wait_cap/results/next/census_red_rear.md section 4; decision 325). Box, repo root, AlpaSim's venv
(P=$DATA_DIR/third_party/alpasim/.venv/bin/python; the build also needs PYTHONPATH=$DATA_DIR/envs/nuplan_devkit_overlay):

  $P wait_set.py select [--workers N]          candidate windows from the raw nuPlan test logs + reconstruction coverage + the drop rule
                                                 -> $DATA_DIR/runs/alpasim_proxy/wait_set/select/<ts>/{cands.csv, set.json, set.txt, controls.txt}
  $P wait_set.py build --set F --out DIR       trajdata cache of the set's windows and of their source scenes' public windows (controls), as
                                                 experiments/alpasim_long/scripts/build_cache.py builds it (AlpaSim's own config -> params ->
                                                 UnifiedDataset, vehicles only, prebuilt maps symlinked); only central_tokens_config differs
  $P wait_set.py check --set F --cache DIR     the cached ego track of every window against the set (static / onset), -> DIR/check.json
  $P wait_set.py collect --set F --run D [D ...] --label L --out DIR   per-window outcome of one read (score, zero type, roll, launch)
  wait_set.py report --set F --in DIR [DIR ...] --out <md> [--csv <csv>]   (Mac or box) tables of results/wait_set.md
  wait_set.py anchors --csv <report all.csv> --out <md>   public score + variant-a zeros against the private anchors (Kendall tau, fits)

Definitions (fixed before any model read):
  frames      raw lidar_pc rows of a log ordered by timestamp (20 Hz), trajdata's order. A scene <log>-<token> covers frames
              [i - 30, i + 80] around its central token i (5.5 s, 10 decisions); the window id of a rebuilt window is <log>-<token of
              its frame start + 30>, its renderer asset is the source scene's (central_tokens_config asset_folder = the source config's
              road_block_name, as AlpaSim does for every token of a config).
  human       ego_pose of the log (rear axle). Standing start: a public scene whose human is slower than 0.3 m/s at the scene's first
              frame. Rest point = the position there; movement onset m = the first later frame > 0.25 m from it; standstill = the
              frames [b, m) within 0.25 m of the rest point (b found backwards from the scene's first frame).
  variant a   static: the 111-frame window that ends 10 frames (0.5 s) before the onset, start m - 120; requires a standstill >= 6.0 s.
              The human never moves in the window (path < 0.25 m), so the scorer's corridor is the plain distance to the rest point.
  variant b   late launch: the window whose frame 60 (3.0 s) is the onset, start m - 60; requires a standstill >= 3.0 s and the window
              inside the log. Human launch time = first frame faster than 1.0 m/s (census_red_rear's launch rule).
  class       red: the light connector the census found ahead of the source scene (scenes.json.gz tl_cid, front bumper <= 80 m from
              the line) shows red in any of the window's rest frames (0.25 s steps; the label lags the switch); lead: a logged vehicle whose centre is within 1.8 m of the
              ego's heading line and whose rear is ahead of the ego's front bumper (-0.5 m), at most 30 m, at the window start.
              red_first / red_queue / other_lead / other_free.
  coverage    the reconstruction's camera frames (assets/<asset>/video_scene_dict.pkl, frame position = ego2global_translation +
              global2world_translation): ego_cov = max over the window's 20 Hz human poses of the distance to the nearest camera frame.
              Actors: a logged vehicle is drawn when the asset checkpoint has a model for its track token (background ckpt keys
              <type>_<token>); inv20 / inv_path = distinct undrawn vehicles whose centre comes within 20 m of the ego / into the ego's
              forward corridor (0-25 m ahead of the front bumper, |lateral| < 2.5 m) at any 0.25 s step; also computed on the public
              window of the same scene as the reference.
  drop        (1) ego_cov > 1.0 m (the ego outside the reconstructed views); (2) inv_path > 0 or an undrawn lead (a vehicle the ego could
              hit in its lane that the render does not show; amended before any read from "inv_path above the public window's", which
              dropped nothing while 19 windows had an invisible standing lead); (3) window outside the log; (4) a window id that is
              already taken (another public scene or window). One window per source scene and variant.
  outcome     (collect) score, the scorer's zero type (collision at fault / offroad / corridor, in the scorer's precedence), rear
              contacts; from the controller trace (10 Hz, scene frame): roll = max distance from the start position while the human
              waits (a: the whole run; b: up to the onset, 3.0 s); wait success = roll <= 1.0 m and no zero; launch (b) = first time the
              ego is faster than 1.0 m/s; delay = ego launch - human launch (censored at the run's end: "never"); early = the ego passes
              1.0 m/s before the human's onset.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sqlite3
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RAW = DATA / "datasets/nuplan_raw"
SHIPPED = DATA / "datasets/alpasim_nuplan"
CFG, ASSETS = SHIPPED / "navtest/configs", SHIPPED / "navtest/assets"
CENSUS = DATA / "runs/wait_cap/census/scenes/20261017-144434/scenes.json.gz"
TEMPLATE = DATA / "runs/onpol_nuplan/long/smoke/runs/a80"      # build_cache.py's template run (the shipped e2e_challenge_nuplan wiring)

HZ, BEFORE, AFTER = 20, 30, 80
WIN = BEFORE + AFTER + 1
V_STOP, V_GO, REST_R = 0.3, 1.0, 0.25
MARGIN_A, ONSET_B = 10, 60
COV_EGO, R_NEAR, PATH_X, PATH_Y = 1.0, 20.0, 25.0, 2.5
FRONT, REAR, HALF_W = 4.049, 1.127, 2.297 / 2
LEAD_LAT, LEAD_GAP, TL_MAX = 1.8, 30.0, 80.0
WAIT_OK = 1.0


def yaml_cfg(scene):
    import yaml

    class L(yaml.SafeLoader):
        pass
    L.add_multi_constructor("tag:yaml.org,2002:python/object", lambda l, t, n: l.construct_mapping(n, deep=True))
    L.add_multi_constructor("tag:yaml.org,2002:python/tuple", lambda l, t, n: l.construct_sequence(n, deep=True))
    return yaml.load((CFG / f"{scene}.yaml").read_text(), Loader=L)


class Log:
    """One raw nuPlan log: frames in trajdata's order, ego poses, vehicle boxes and light states on demand."""

    def __init__(self, log):
        self.c = c = sqlite3.connect(f"file:{RAW}/test/{log}.db?mode=ro", uri=True)
        rows = c.execute("select l.token, l.timestamp, e.x, e.y, e.qw, e.qx, e.qy, e.qz, e.vx, e.vy from lidar_pc l "
                         "join ego_pose e on e.token = l.ego_pose_token order by l.timestamp").fetchall()
        self.tok = [r[0] for r in rows]
        a = np.array([r[1:] for r in rows], float)
        self.ts = a[:, 0].astype(np.int64)
        self.xy = a[:, 1:3]
        qw, qx, qy, qz = a[:, 3], a[:, 4], a[:, 5], a[:, 6]
        self.yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.v = np.hypot(a[:, 7], a[:, 8])
        self.idx = {k.hex(): i for i, k in enumerate(self.tok)}

    def boxes(self, frames):
        """{frame: [(track hex, x, y, yaw, length, width), ...]} of the vehicles in these frames."""
        q = ("select lb.lidar_pc_token, lb.track_token, lb.x, lb.y, lb.yaw, lb.length, lb.width from lidar_box lb join track t on "
             "t.token = lb.track_token join category c on c.token = t.category_token where c.name = 'vehicle' and lb.lidar_pc_token in (%s)")
        fr = {self.tok[f]: f for f in frames}
        out = {f: [] for f in frames}
        ks = list(fr)
        for j in range(0, len(ks), 500):
            for r in self.c.execute(q % ",".join("?" * len(ks[j:j + 500])), ks[j:j + 500]):
                out[fr[r[0]]].append((r[1].hex(), r[2], r[3], r[4], r[5], r[6]))
        return out

    def red_frac(self, frames, cid):
        if cid is None:
            return None
        ks = [self.tok[f] for f in frames]
        st = [r[0] for r in self.c.execute("select status from traffic_light_status where lane_connector_id = ? and lidar_pc_token in (%s)"
                                           % ",".join("?" * len(ks)), [int(cid), *ks])]
        return float(np.mean([s == "red" for s in st])) if st else None


def to_ego(xy, yaw, p):
    d = np.asarray(p, float) - xy
    c, s = np.cos(yaw), np.sin(yaw)
    return d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c


def asset_info(asset):
    """Camera frame positions of the reconstruction video and the track tokens the checkpoint has a model for."""
    import pickle

    import torch
    v = list(pickle.load(open(ASSETS / asset / "video_scene_dict.pkl", "rb")).values())
    g = np.asarray(v[0]["global2world_translation"], float)[:2]
    pos = np.concatenate([np.array([f["ego2global_translation"][:2] for f in x["frame_infos"]], float) + g for x in v])
    ts = np.concatenate([np.array([f["timestamp"] for f in x["frame_infos"]], np.int64) for x in v])
    ck = torch.load(ASSETS / asset / "background" / f"{asset}.ckpt", map_location="cpu", weights_only=False, mmap=True)
    models = {k.rsplit("_", 1)[-1] for k in ck.keys()} - {"background", "skybox"}
    del ck
    return pos, ts, models


def window_stats(L, w0, onset, models, pos, vts):
    """Coverage and context of the 111-frame window starting at frame w0 (onset: the human's movement onset frame or None)."""
    fr = np.arange(w0, w0 + WIN)
    d = np.sqrt(((L.xy[fr, None, :] - pos[None, :, :]) ** 2).sum(-1)).min(1)
    steps = fr[::5]
    B = L.boxes(steps)
    inv20, invp, ntr = set(), set(), set()
    for f in steps:
        for tk, x, y, _, ln, wd in B[f]:
            ex, ey = to_ego(L.xy[f], L.yaw[f], (x, y))
            dist = np.hypot(ex - (FRONT - REAR) / 2, ey)
            if dist < R_NEAR:
                ntr.add(tk)
                if tk not in models:
                    inv20.add(tk)
                    if 0 < ex - FRONT < PATH_X and abs(ey) < PATH_Y:
                        invp.add(tk)
    lead = None
    for tk, x, y, yw, ln, wd in B[w0]:
        ex, ey = to_ego(L.xy[w0], L.yaw[w0], (x, y))
        if abs(ey) < LEAD_LAT and ex - ln / 2 > FRONT - 0.5 and ex - ln / 2 - FRONT < LEAD_GAP:
            gap = ex - ln / 2 - FRONT
            if lead is None or gap < lead[1]:
                lead = (tk, gap, tk in models)
    return dict(ego_cov=float(d.max()), vid_lead_s=float(vts.min() - L.ts[w0]) / 1e6, vid_tail_s=float(L.ts[w0 + WIN - 1] - vts.max()) / 1e6,
                near=len(ntr), inv20=len(inv20), inv_path=len(invp), lead=lead is not None, lead_gap=None if lead is None else round(float(lead[1]), 2),
                lead_drawn=None if lead is None else lead[2])


def one_log(job):
    """All standing-start public scenes of one log -> candidate rows (variants p = the public window, a, b)."""
    log, scenes, census = job
    L = Log(log)
    n, rows = len(L.tok), []
    for sc in scenes:
        cfg = yaml_cfg(sc)
        tok = sc.rsplit("-", 1)[1]
        i0 = L.idx.get(tok)
        if i0 is None or i0 - BEFORE < 0:
            continue
        ws = i0 - BEFORE
        if L.v[ws] >= V_STOP:
            continue
        out = np.hypot(*(L.xy - L.xy[ws]).T) > REST_R     # frames away from the rest point
        far = np.flatnonzero(out[ws:])
        if not len(far):
            continue
        m = ws + int(far[0])
        nb = np.flatnonzero(out[:ws])
        b = int(nb[-1]) + 1 if len(nb) else 0
        pos, vts, models = asset_info(cfg["road_block_name"])
        cen = census.get(sc, {})
        cid = cen.get("tl_cid") if (cen.get("tl_d_front") is not None and cen.get("tl_d_front") <= TL_MAX) else None
        go = np.flatnonzero(L.v[m:] > V_GO)
        t_go = m + int(go[0]) if len(go) else None
        base = dict(src=sc, log=log, asset=cfg["road_block_name"], stand_s=(m - b) / HZ, onset_pub_s=(m - ws) / HZ,
                    launch_pub_s=None if t_go is None else (t_go - ws) / HZ, census_tl=cid is not None, census_red_stop=cen.get("red_stop"),
                    census_red_start=cen.get("red_start"), census_launch_t=cen.get("h_launch_t"))
        for var, w0 in (("p", ws), ("a", m - MARGIN_A - WIN + 1 if m - b >= MARGIN_A + WIN - 1 else None),
                        ("b", m - ONSET_B if m - b >= ONSET_B else None)):
            r = dict(base, variant=var)
            if w0 is None:
                r.update(drop="standstill too short")
                rows.append(r)
                continue
            if w0 + WIN > n:
                r.update(drop="window outside the log")
                rows.append(r)
                continue
            c = w0 + BEFORE
            st = window_stats(L, w0, m, models, pos, vts)
            restf = np.arange(w0, min(m, w0 + WIN))
            rf = L.red_frac(restf[::5], cid)
            wv = L.v[w0:w0 + WIN]
            path = float(np.hypot(*np.diff(L.xy[w0:w0 + WIN], axis=0).T).sum())
            gw = np.flatnonzero(wv > V_GO)
            r.update(id=f"{log}-{L.tok[c].hex()}", token=L.tok[c].hex(), start_idx=int(w0), shift_s=(w0 - ws) / HZ,
                     onset_s=(m - w0) / HZ if m < w0 + WIN else None, h_launch_s=float(gw[0] / HZ) if len(gw) else None,
                     h_path_m=path, h_disp_m=float(np.hypot(*(L.xy[w0:w0 + WIN] - L.xy[w0]).T).max()), red_frac=rf,
                     red=bool(rf is not None and rf > 0), **st)
            r["cls"] = ("red_" if r["red"] else "other_") + (("queue" if r["red"] else "lead") if r["lead"] else ("first" if r["red"] else "free"))
            rows.append(r)
    return rows


def cmd_select(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("alpasim_proxy", "wait_set/select", config=vars(a)) as run:
        allsp, held = splits.load("alpasim_nuplan/onpol-all"), splits.load("alpasim_nuplan/onpol-held")
        run.use_split(allsp)
        run.use_split(held)
        census = {}
        for x in json.load(gzip.open(CENSUS, "rt")):
            census[x["scene"]] = x["pub"]
        by_log = {}
        for s in sorted(allsp.members):
            by_log.setdefault(s.rsplit("-", 1)[0], []).append(s)
        jobs = [(lg, ss, {s: census.get(s, {}) for s in ss}) for lg, ss in sorted(by_log.items())]
        res = par.pmap(one_log, jobs, run=run, workers=a.workers)
        res.raise_if_failed()
        rows = [r for rr in res.values for r in rr]
        hm = set(held.members)
        for r in rows:
            r["split"] = "held" if r["src"] in hm else "train"
        pub_ids = set(allsp.members)
        seen = set(pub_ids)
        ref = {r["src"]: r for r in rows if r["variant"] == "p"}
        for r in rows:
            if r["variant"] == "p" or r.get("drop"):
                continue
            pr = ref[r["src"]]
            if r["ego_cov"] > COV_EGO:
                r["drop"] = "ego outside the reconstruction's camera frames"
            elif r["inv_path"] > 0 or r["lead_drawn"] is False:
                r["drop"] = "undrawn vehicle in the forward corridor or as the lead"
            elif r["id"] in seen:
                r["drop"] = "window id taken"
            else:
                seen.add(r["id"])
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(run.path("cands.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
        keep = [r for r in rows if r["variant"] in "ab" and not r.get("drop")]
        win = [{k: r[k] for k in ("id", "variant", "src", "split", "log", "token", "asset", "start_idx", "shift_s", "onset_s", "h_launch_s",
                                  "h_path_m", "h_disp_m", "stand_s", "cls", "red", "red_frac", "lead", "lead_gap", "lead_drawn", "ego_cov",
                                  "vid_lead_s", "vid_tail_s", "near", "inv20", "inv_path")} for r in keep]
        srcs = sorted({r["src"] for r in keep})
        ctl = [dict(id=s, log=ref[s]["log"], token=s.rsplit("-", 1)[1], asset=ref[s]["asset"]) for s in srcs]
        params = dict(BEFORE=BEFORE, AFTER=AFTER, V_STOP=V_STOP, V_GO=V_GO, REST_R=REST_R, MARGIN_A=MARGIN_A, ONSET_B=ONSET_B, COV_EGO=COV_EGO,
                      R_NEAR=R_NEAR, PATH_X=PATH_X, PATH_Y=PATH_Y, census=str(CENSUS), splits=[allsp.id, held.id])
        json.dump(dict(params=params, windows=win, controls=ctl), open(run.path("set.json"), "w"), indent=1)
        Path(run.path("set.txt")).write_text("".join(f"{w['id']}\n" for w in win))
        Path(run.path("controls.txt")).write_text("".join(f"{c['id']}\n" for c in ctl))
        from collections import Counter
        cand = [r for r in rows if r["variant"] in "ab"]
        summ = dict(standing_starts=len(ref), candidates={v: sum(r["variant"] == v for r in cand) for v in "ab"},
                    kept={v: sum(w["variant"] == v for w in win) for v in "ab"},
                    drops={v: dict(Counter(r["drop"] for r in cand if r["variant"] == v and r.get("drop"))) for v in "ab"},
                    classes={v: dict(Counter(w["cls"] for w in win if w["variant"] == v)) for v in "ab"},
                    held={v: sum(w["variant"] == v and w["split"] == "held" for w in win) for v in "ab"})
        run.summary.update(summ)
        print(json.dumps(summ, indent=1))


def cmd_build(a):
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_runtime.scene_loader import trajdata_provider_config_to_params
    from omegaconf import OmegaConf
    from trajdata.dataset import UnifiedDataset
    from trajdata.dataset_specific.nuplan import nuplan_utils

    assert nuplan_utils.create_splits_logs().get("test"), "nuPlan log split table missing: set PYTHONPATH to the devkit overlay"
    S = json.load(open(a.set))
    out = Path(a.out).resolve()
    assert out != SHIPPED.resolve() and SHIPPED.resolve() not in out.parents, "never write the shipped cache"
    assert not (out / "nuplan_test/scenes_list.dill").exists(), f"{out} already holds a cache"
    (out / "nuplan_test").mkdir(parents=True, exist_ok=True)
    if not (out / "nuplan_test/maps").exists():
        (out / "nuplan_test/maps").symlink_to(SHIPPED / "nuplan_test/maps")
    user = OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(TEMPLATE / "generated-user-config-0.yaml"))
    td = user.scene_provider.trajdata
    td.cache_location, td.dataset.data_dir, td.num_workers = str(out), str(RAW), a.workers
    td.dataset.extra_params.num_timesteps_before, td.dataset.extra_params.num_timesteps_after = BEFORE, AFTER
    td.rebuild_cache = True
    params = trajdata_provider_config_to_params(OmegaConf.to_object(td))
    kw = params["dataset_kwargs"]["nuplan_test"]
    kw["central_tokens_config"] = [dict(central_token=w["token"], logfile=w["log"], asset_folder=w["asset"]) for w in S["windows"] + S["controls"]]
    assert len({(c["logfile"], c["central_token"]) for c in kw["central_tokens_config"]}) == len(kw["central_tokens_config"]), "duplicate token"
    print(f"central tokens: {len(S['windows'])} windows + {len(S['controls'])} controls", flush=True)
    orig = nuplan_utils.NuPlanObject.get_detected_agents   # vehicles only, as the shipped cache (build_cache.py, gate G0)
    nuplan_utils.NuPlanObject.get_detected_agents = lambda self, t: (lambda d: d[d.category_name == "vehicle"].reset_index(drop=True))(orig(self, t))
    ds = UnifiedDataset(**params, require_map_cache=False, verbose=True)
    print(f"built {ds.num_scenes()} scenes into {out}", flush=True)
    assert ds.num_scenes() == len(kw["central_tokens_config"]), "scene count differs from the token list"


def cmd_check(a):
    """Every window's cached ego track: 56 steps at 0.1 s, static in a (path < REST_R), at rest until the onset in b; the asset folder."""
    import dill
    import pandas as pd
    S = json.load(open(a.set))
    rows = []
    for w in S["windows"]:
        d = Path(a.cache) / "nuplan_test" / w["id"]
        m = dill.load(open(d / "scene_metadata_dt0.10.dill", "rb"))
        ag = pd.read_feather(d / "agent_data_dt0.10.feather")
        e = ag[ag.agent_id == "ego"].sort_values("scene_ts")
        xy = e[["x", "y"]].to_numpy()
        disp = np.hypot(*(xy - xy[0]).T)
        k_on = int(round(w["onset_s"] * 10)) if w["onset_s"] is not None else len(xy)
        rows.append(dict(id=w["id"], variant=w["variant"], steps=int(m.length_timesteps), asset_ok=m.data_access_info.get("asset_folder") == w["asset"],
                         start_ok=m.data_access_info.get("start_idx") == w["start_idx"], disp_max=float(disp.max()),
                         disp_before_onset=float(disp[:k_on].max()), agents=len(m.agents)))
    R = pd.DataFrame(rows)
    ok = (R.steps == 56) & R.asset_ok & R.start_ok & ((R.variant != "a") | (R.disp_max < 2 * REST_R)) & (R.disp_before_onset < 2 * REST_R)
    summ = dict(windows=len(R), ok=int(ok.sum()), bad=R[~ok].id.tolist()[:20], disp_max_a=float(R[R.variant == "a"].disp_max.max()),
                disp_before_onset_b=float(R[R.variant == "b"].disp_before_onset.max()))
    json.dump(dict(summary=summ, rows=rows), open(Path(a.cache) / "check.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    sys.exit(0 if ok.all() else 1)


# ---------------------------------------------------------------- outcomes of one read
def ctrl(path):
    rows = list(csv.reader(open(path)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    return A[:, h["timestamp_us"]] / 1e6, A[:, [h["x"], h["y"]]], np.hypot(A[:, h["vx"]], A[:, h["vy"]])


def cmd_collect(a):
    import polars as pl
    S = json.load(open(a.set))
    W = {w["id"]: w for w in S["windows"]}
    out, seen = [], set()
    for run, r in [(d, r) for d in a.run for r in json.load(open(Path(d) / "aggregate/results-summary.json"))["rollouts"]]:
        sid = r["clipgt_id"]
        if sid not in W or sid in seen:
            continue
        seen.add(sid)
        w, m = W[sid], r.get("metrics") or {}
        sc = float(r.get("score") or 0.0)
        z = None
        if sc == 0:
            z = ("collision" if (m.get("collision_at_fault") or 0) > 0 else "offroad" if (m.get("offroad") or 0) > 0 else
                 "corridor" if (m.get("left_corridor_laterally") or 0) > 0 else "failed" if r.get("failure_reason") else "other")
        row = dict(label=a.label, id=sid, variant=w["variant"], src=w["src"], split=w["split"], cls=w["cls"], score=sc, zero=z,
                   fail=r.get("failure_reason"), rear=(m.get("collision_rear") or 0) > 0, driven_m=m.get("dist_traveled_m"),
                   gt_m=m.get("gt_dist_traveled_m"))
        f = Path(run) / f"controller/alpasim_controller_{r['rollout_id']}.csv"
        if f.exists():
            t, xy, v = ctrl(f)
            disp = np.hypot(*(xy - xy[0]).T)
            t_on = w["onset_s"] if w["variant"] == "b" else None
            wait = t < t_on if t_on is not None else np.ones(len(t), bool)
            row["roll_m"] = float(disp[wait].max()) if wait.any() else 0.0
            row["t_end"] = float(t[-1])
            go = np.flatnonzero(v > V_GO)
            row["launch_s"] = float(t[go[0]]) if len(go) else None
            if w["variant"] == "b":
                row["h_launch_s"] = w["h_launch_s"]
                row["delay_s"] = None if row["launch_s"] is None else row["launch_s"] - w["h_launch_s"]
                row["early"] = row["launch_s"] is not None and row["launch_s"] < t_on
                row["disp_end"] = float(disp[-1])
            row["wait_ok"] = bool(row["roll_m"] <= WAIT_OK and z is None)
        mp = sorted(glob.glob(f"{run}/rollouts/{sid}/*/metrics.parquet"))
        if mp:   # first onset of each event series (scene time, s)
            M = pl.read_parquet(mp[0], columns=["name", "timestamps_us", "values"]).filter(pl.col("values") > 0)
            for n in ("collision_front", "collision_lateral", "collision_rear", "offroad", "left_corridor_laterally"):
                x = M.filter(pl.col("name") == n)
                row[f"t_{n}"] = float(x["timestamps_us"].cast(pl.Float64).min()) / 1e6 if len(x) else None
        out.append(row)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(Path(a.out) / f"{a.label}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(out)
    miss = sorted(set(W) - seen)
    print(json.dumps(dict(label=a.label, rollouts=len(out), missing=len(miss), first_missing=miss[:5],
                          mean_score={v: float(np.mean([r["score"] for r in out if r["variant"] == v] or [np.nan])) for v in "ab"})))


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    R = pd.concat([pd.read_csv(f) for d in a.inp for f in sorted(glob.glob(f"{d}/*.csv"))], ignore_index=True)
    if a.set:   # only the windows of the current set version (tables of an older version may hold dropped windows)
        R = R[R.id.isin({w["id"] for w in json.load(open(a.set))["windows"]})]
    R["log"] = R.src.str.rsplit("-", n=1).str[0]
    order = [x for x in a.order.split(",") if x in set(R.label)] + sorted(set(R.label) - set(a.order.split(",")))
    L = []

    def row(lab, x, v):
        n = len(x)
        z = x.zero.fillna("")
        sc = stats.bootstrap(x.score.to_numpy(float))
        d = dict(label=lab, variant=v, n=n, score=f"{sc['mean']:.3f} [{sc['lo']:.3f}, {sc['hi']:.3f}]", zeros=int((z != "").sum()),
                 collision=int((z == "collision").sum()), offroad=int((z == "offroad").sum()), corridor=int((z == "corridor").sum()),
                 failed=int(z.isin(["failed", "other"]).sum()), rear=int(x.rear.sum()),
                 roll_med=f"{x.roll_m.median():.2f}", roll_p90=f"{x.roll_m.quantile(0.9):.2f}", roll_gt3=int((x.roll_m > 3).sum()),
                 wait_ok=f"{x.wait_ok.mean():.3f}")
        if v == "b":
            dl = x.delay_s.dropna()
            d.update(early=int(x.early.sum()), never=int(x.launch_s.isna().sum()), delay_med=f"{dl.median():.2f}" if len(dl) else "-",
                     delay_p90=f"{dl.quantile(0.9):.2f}" if len(dl) else "-", late_gt1=int((dl > 1).sum()))
        return d
    for v in "ab":
        T = [row(lab, R[(R.label == lab) & (R.variant == v)], v) for lab in order if ((R.label == lab) & (R.variant == v)).any()]
        L.append(f"\n### Variant {v}\n")
        df = pd.DataFrame(T)
        L.append("| " + " | ".join(df.columns) + " |")
        L.append("|" + "|".join("--:" for _ in df.columns) + "|")
        for _, r in df.iterrows():
            L.append("| " + " | ".join(str(x) for x in r.values) + " |")
    for v in "ab":   # by class
        L.append(f"\n### Variant {v} by class (score / zeros / wait success)\n")
        cls = sorted(R[R.variant == v].cls.unique())
        L.append("| label | " + " | ".join(f"{c} (n)" for c in cls) + " |")
        L.append("|:--|" + "|".join("--:" for _ in cls) + "|")
        for lab in order:
            x = R[(R.label == lab) & (R.variant == v)]
            if not len(x):
                continue
            cells = []
            for c in cls:
                y = x[x.cls == c]
                cells.append(f"{y.score.mean():.3f} / {int(y.zero.notna().sum())} / {y.wait_ok.mean():.2f} ({len(y)})" if len(y) else "-")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
    held = R[R.split == "held"]
    if len(held):
        L.append("\n### Held-split windows only (score / zeros / wait success)\n")
        L.append("| label | a | b |")
        L.append("|:--|--:|--:|")
        for lab in order:
            cells = []
            for v in "ab":
                y = held[(held.label == lab) & (held.variant == v)]
                cells.append(f"{y.score.mean():.3f} / {int(y.zero.notna().sum())} / {y.wait_ok.mean():.2f} ({len(y)})" if len(y) else "-")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
    if a.pairs:
        L.append("\n### Paired differences (score, log-clustered bootstrap over source logs)\n")
        L.append("| pair | variant | n | mean [95 % CI] |")
        L.append("|:--|:--|--:|--:|")
        for p in a.pairs:
            x_, y_ = p.split(":")
            for v in "ab":
                A_ = R[(R.label == x_) & (R.variant == v)].set_index("id")
                B_ = R[(R.label == y_) & (R.variant == v)].set_index("id")
                ids = sorted(set(A_.index) & set(B_.index))
                if not ids:
                    continue
                r_ = stats.paired(A_.loc[ids, "score"].to_numpy(float), B_.loc[ids, "score"].to_numpy(float), groups=A_.loc[ids, "log"].to_numpy())
                L.append(f"| {x_} - {y_} | {v} | {len(ids)} | {stats.fmt(r_)} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    if a.csv:
        R.to_csv(a.csv, index=False)
    print("\n".join(L))


# private board anchors (results/leaderboard_nuplan_20261012.json; public = all 1491 scenes, decision 262; go-straight public = the
# 300-scene log-stratified sample of decision 325's N1, 0.522)
ANCHORS = {"APY10m10-AB-s0": (0.935, 0.640), "YR10m10-F-s0": (0.925, 0.800), "LTF-sample": (0.824, 0.738), "DD-sample": (0.797, 0.772),
           "G0014": (0.740, 0.848), "starter": (0.522, 0.799)}


def ktau(x, y):
    """Kendall tau-b."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = d = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            tx += sx == 0
            ty += sy == 0
            c += sx * sy > 0
            d += sx * sy < 0
    n0 = len(x) * (len(x) - 1) / 2
    return float((c - d) / np.sqrt((n0 - tx) * (n0 - ty)))


def cmd_anchors(a):
    """Does public score + the wait set's variant-a zeros order the private anchors? Two one-parameter fits over the anchors read:
    linear  private = public - k * z_a            (z_a = share of variant-a windows scored 0, failures included)
    mixture private = (1 - p) * public + p * s_a  (s_a = variant-a mean score = 1 - z_a; p = the share of waiting windows on private)
    each with its in-sample and leave-one-anchor-out predictions and Kendall tau-b against private."""
    import pandas as pd
    R = pd.read_csv(a.csv)
    rows = []
    for lab, (pub, prv) in ANCHORS.items():
        x = R[(R.label == lab) & (R.variant == "a")]
        y = R[(R.label == lab) & (R.variant == "b")]
        if len(x):
            rows.append(dict(anchor=lab, private=prv, public=pub, n_a=len(x), z_a=float((x.score == 0).mean()), s_a=float(x.score.mean()),
                             s_b=float(y.score.mean()) if len(y) else np.nan, early_b=float(y.early.mean()) if len(y) else np.nan))
    T = pd.DataFrame(rows)
    if len(T) < 3:
        sys.exit(f"only {len(T)} anchors read")
    fk = lambda t: float((t.z_a * (t.public - t.private)).sum() / max((t.z_a ** 2).sum(), 1e-12))      # noqa: E731
    fp = lambda t: float(((t.s_a - t.public) * (t.private - t.public)).sum() / max(((t.s_a - t.public) ** 2).sum(), 1e-12))  # noqa: E731
    k, p = fk(T), fp(T)
    T["pred_lin"] = T.public - k * T.z_a
    T["pred_mix"] = (1 - p) * T.public + p * T.s_a
    T["loo_lin"] = [T.public[i] - fk(T.drop(i)) * T.z_a[i] for i in T.index]
    T["loo_mix"] = [(1 - (q := fp(T.drop(i)))) * T.public[i] + q * T.s_a[i] for i in T.index]
    taus = {c: ktau(T[c], T.private) for c in ("public", "s_a", "pred_lin", "pred_mix", "loo_lin", "loo_mix")}
    L = [f"Anchors read: {len(T)}; fitted k = {k:.3f} (linear), p = {p:.3f} (mixture).\n",
         "| anchor | private | public | n a | zero rate a | score a | score b | early b | pred linear | pred mixture | LOO linear | LOO mixture |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in T.sort_values("private", ascending=False).iterrows():
        L.append(f"| {r.anchor} | {r.private:.3f} | {r.public:.3f} | {r.n_a} | {r.z_a:.3f} | {r.s_a:.3f} | {r.s_b:.3f} | {r.early_b:.3f} | "
                 f"{r.pred_lin:.3f} | {r.pred_mix:.3f} | {r.loo_lin:.3f} | {r.loo_mix:.3f} |")
    L += ["", "Kendall tau-b against private: " + ", ".join(f"{c} {v:+.2f}" for c, v in taus.items()),
          f"Mean abs error: linear {np.abs(T.pred_lin - T.private).mean():.3f}, mixture {np.abs(T.pred_mix - T.private).mean():.3f}, "
          f"LOO linear {np.abs(T.loo_lin - T.private).mean():.3f}, LOO mixture {np.abs(T.loo_mix - T.private).mean():.3f}, "
          f"public alone {np.abs(T.public - T.private).mean():.3f}"]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--workers", type=int, default=24)
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("build")
    p.add_argument("--set", required=True), p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=24)
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("check")
    p.add_argument("--set", required=True), p.add_argument("--cache", required=True)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("collect")
    p.add_argument("--set", required=True), p.add_argument("--run", nargs="+", required=True), p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("report")
    p.add_argument("--set"), p.add_argument("--in", dest="inp", nargs="+", required=True), p.add_argument("--out", required=True)
    p.add_argument("--csv"), p.add_argument("--order", default="starter,replay,P0,RSG-L18-s0,RTB1ARO-L18-s0"), p.add_argument("--pairs", nargs="*")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("anchors")
    p.add_argument("--csv", required=True, help="all.csv of wait_set.sh report"), p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_anchors)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)
