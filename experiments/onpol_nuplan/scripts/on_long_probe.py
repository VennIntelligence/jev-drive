#!/usr/bin/env python3
"""ONPOL-NUPLAN round 2, item A (plans/2026-10-12-onpol2-prereg.md): what bounds the length of a public nuPlan scene. CPU only, reads only
(the `window --cache` probe writes into the NEW cache dir it is given, never into the shipped one). Box, AlpaSim's venv:

  P=$DATA_DIR/third_party/alpasim/.venv/bin/python
  $P on_long_probe.py span --out <csv> [--json <summary.json>]
      every public scene (navtest/configs/*.yaml): frames of the prebuilt trajdata cache (20 Hz and resampled), seconds the MTGS
      reconstruction's own video continues after the scene token (assets/<scene>/video_scene_dict.pkl: the camera frames the Gaussians were
      fitted to), seconds the logged ego stays inside the reconstructed road block (+ reconstruct_buffer), seconds the nuPlan log continues
      (NAVSIM test log pickle, 2 Hz), and the decisions each bound would allow (scene start = token - 1.5 s, first decision at 0.017 s,
      0.5 s per decision: AlpaSim's unbound_rollout.py formula). Split membership from results/split/{train,held}.txt.
  $P on_long_probe.py window --run <finished run dir> --scenes <txt> --after N [--cache <new dir>]
      AlpaSim's own scene loader (alpasim_runtime.scene_loader, unedited) on the run's generated user config with num_timesteps_after = N and,
      with --cache, trajdata's cache_location = <new dir> (data_dir unchanged): prints per scene the rig trajectory's time range and the
      decisions it allows, or the loader's failure. This is the CPU half of the smoke: what the override does without a rollout.
"""
import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SPLIT = REPO / "experiments/onpol_nuplan/results/split"
T_BEFORE, T_FIRST, DT = 1.5, 0.017, 0.5          # scene start before the token, first decision, control step (s)


def n_dec(t_after):
    """Decisions a scene that ends t_after seconds after its token allows (10 for the shipped 4.0 s)."""
    return int(max(0.0, (T_BEFORE + t_after - 1e-6 - T_FIRST) // DT))


def cmd_span(a):
    import dill
    import yaml

    class L(yaml.SafeLoader):
        pass
    L.add_multi_constructor("tag:yaml.org,2002:python/object", lambda l, t, n: l.construct_mapping(n, deep=True))
    L.add_multi_constructor("tag:yaml.org,2002:python/tuple", lambda l, t, n: l.construct_sequence(n, deep=True))
    root, nav = Path(a.root), Path(a.navsim)
    cached = {r.name: r.length for r in dill.load(open(root / "nuplan_test/scenes_list.dill", "rb"))}
    split = {s: n for n in ("train", "held") for s in (SPLIT / f"{n}.txt").read_text().split()}
    logs, rows = {}, []
    for f in sorted((root / "navtest/configs").glob("*.yaml")):
        cfg = yaml.load(f.read_text(), Loader=L)
        sid, log, tok = f.stem, cfg["central_log"], f.stem.rsplit("-", 1)[1]
        if log not in logs:
            fr = pickle.load(open(nav / f"{log}.pkl", "rb"))
            logs[log] = ({x["token"]: i for i, x in enumerate(fr)}, np.array([x["timestamp"] for x in fr], np.int64),
                         np.array([x["ego2global_translation"][:2] for x in fr], np.float64))
        ix, ts, xy = logs[log]
        r = dict(scene=sid, log=log, split=split.get(sid, ""), cache_frames_20hz=cached.get(sid), n_tokens_in_asset=len(cfg["central_tokens"]))
        m = pickle.load(open(root / f"nuplan_test/{sid}/scene_metadata_dt0.10.dill", "rb")) if a.meta else None
        if m is not None:
            r["cache_frames_10hz"] = m.length_timesteps
        if tok in ix:
            i, t0 = ix[tok], ts[ix[tok]]
            r["log_after_s"] = (ts[-1] - t0) / 1e6
            x0, y0, x1, y1 = cfg["road_block"]
            b = cfg.get("reconstruct_buffer", 0)
            out = ~((xy[i:, 0] >= x0 - b) & (xy[i:, 0] <= x1 + b) & (xy[i:, 1] >= y0 - b) & (xy[i:, 1] <= y1 + b))
            r["in_block_at_token"] = bool(not out[0])
            r["block_after_s"] = float((ts[i + int(np.argmax(out))] - t0) / 1e6) if out.any() else float(r["log_after_s"])   # first 2 Hz frame outside
            r["log_m_in_4s"] = float(np.linalg.norm(xy[min(i + 8, len(xy) - 1)] - xy[i]))
            vf = root / f"navtest/assets/{sid}/video_scene_dict.pkl"
            if vf.exists():
                vs = list(pickle.load(open(vf, "rb")).values())
                has = [v for v in vs if tok in {x["token"] for x in v["frame_infos"]}] or vs
                r["n_videos"], r["video_has_token"] = len(vs), tok in {x["token"] for x in has[0]["frame_infos"]}
                r["video_frames"] = len(has[0]["frame_infos"])
                r["video_before_s"], r["video_after_s"] = (t0 - has[0]["start_ts"]) / 1e6, (has[0]["end_ts"] - t0) / 1e6
        rows.append(r)
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    def dist(sel, key):
        v = np.array([r[key] for r in rows if sel(r) and r.get(key) is not None], float)
        d = np.array([n_dec(x) for x in v])
        return dict(n=len(v), s_min=v.min(), s_p10=np.percentile(v, 10), s_median=np.median(v), s_p90=np.percentile(v, 90), s_max=v.max(),
                    decisions_min=int(d.min()), decisions_median=float(np.median(d)),
                    share_ge={str(k): float((d >= k).mean()) for k in (10, 12, 14, 16, 18, 20, 30, 40)})
    S = dict(scenes=len(rows), cache_frames_20hz={str(k): int(n) for k, n in zip(*np.unique([r["cache_frames_20hz"] for r in rows if r["cache_frames_20hz"]], return_counts=True))},
             token_in_navsim_log=sum("log_after_s" in r for r in rows), video_has_token=sum(bool(r.get("video_has_token")) for r in rows),
             not_in_block_at_token=sum(r.get("in_block_at_token") is False for r in rows))
    for name, sel in (("all", lambda r: True), ("held", lambda r: r["split"] == "held"), ("train", lambda r: r["split"] == "train")):
        S[name] = {k: dist(sel, k) for k in ("video_after_s", "block_after_s", "log_after_s")}
    txt = json.dumps(S, indent=1, default=float)
    if a.json:
        Path(a.json).write_text(txt + "\n")
    print(txt)


def cmd_window(a):
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_runtime.scene_loader import build_scene_loader
    from omegaconf import OmegaConf
    user = OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(f"{a.run}/generated-user-config-0.yaml"))
    td = user.scene_provider.trajdata
    td.dataset.extra_params.num_timesteps_after = a.after
    if a.cache:
        Path(a.cache).mkdir(parents=True, exist_ok=True)
        assert Path(a.cache).resolve() != Path(td.dataset.data_dir).resolve(), "the probe never writes into the shipped cache"
        td.cache_location = a.cache
    print(f"num_timesteps_before {td.dataset.extra_params.num_timesteps_before} after {td.dataset.extra_params.num_timesteps_after} "
          f"cache_location {td.cache_location} data_dir {td.dataset.data_dir} rebuild_cache {td.rebuild_cache}", flush=True)
    try:
        loader = build_scene_loader(user)
        print("loader scenes:", len(loader.scene_ids if hasattr(loader, "scene_ids") else loader._provider.scene_ids), flush=True)
    except BaseException as e:
        traceback.print_exc()
        print(f"LOADER FAILED: {type(e).__name__}: {str(e)[:500]}")
        return 3
    for s in Path(a.scenes).read_text().split():
        try:
            ds = loader.get_data_source(s)
            r = ds.rig.trajectory.time_range_us
            print(f"{s} frames {ds.scene.length_timesteps} dt {ds.scene.dt} range_s {(r.stop - r.start) / 1e6:.3f} decisions {int(((r.stop - 1 - r.start) / 1e6 - T_FIRST) // DT)}", flush=True)
        except BaseException as e:
            print(f"{s} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("span")
    p.add_argument("--root", default=os.path.expandvars("$DATA_DIR/datasets/alpasim_nuplan")), p.add_argument("--navsim", default=os.path.expandvars("$DATA_DIR/datasets/navsim/navsim_logs/test"))
    p.add_argument("--out", required=True), p.add_argument("--json"), p.add_argument("--meta", action="store_true", help="also read every scene's 10 Hz metadata (slow)")
    p.set_defaults(fn=cmd_span)
    p = sub.add_parser("window")
    p.add_argument("--run", required=True), p.add_argument("--scenes", required=True), p.add_argument("--after", type=int, required=True), p.add_argument("--cache")
    p.set_defaults(fn=cmd_window)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)
