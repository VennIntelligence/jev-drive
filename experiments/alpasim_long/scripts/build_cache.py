#!/usr/bin/env python3
"""ALPASIM-LONG: rebuild AlpaSim's nuPlan trajdata cache from the raw nuPlan test logs at a chosen window, and compare caches.
Box, AlpaSim's venv with the devkit overlay (only for nuPlan's log -> split table; maps are not rebuilt):

  P=$DATA_DIR/third_party/alpasim/.venv/bin/python; export PYTHONPATH=$DATA_DIR/envs/nuplan_devkit_overlay
  $P build_cache.py build --after N --out $DATA_DIR/datasets/alpasim_nuplan_long/after<N> [--workers 24] [--template <run dir>]
  $P build_cache.py compare --a <cache root> --b <cache root> --out <json> [--scenes <txt>] [--workers 24]

build: takes scene_provider.trajdata of a finished run's generated-user-config-0.yaml (the shipped e2e_challenge_nuplan wiring), changes
only cache_location (-> --out), dataset.data_dir (-> --raw, which holds test/<log>.db) and num_timesteps_after (-> N), and builds the cache
with AlpaSim's own trajdata_provider_config_to_params + trajdata's UnifiedDataset, unedited. One deviation from the runtime's call:
require_map_cache=False, because the four prebuilt map files of the shipped cache are symlinked into --out/nuplan_test/maps instead of being
rebuilt from the raw nuPlan maps (that needs the full devkit map API). The shipped cache is never written (asserted).
compare: per scene of the shipped scene list, scene length, agent table at dt 0.1 (all columns), traffic lights, and the metadata's agent
list; a and b must be identical at the shipped window (gate G0). With different windows it reports the prefix agreement instead.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
SHIPPED = DATA / "datasets/alpasim_nuplan"
TEMPLATE = DATA / "runs/onpol_nuplan/long/smoke/runs/a80"


def build(a):
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_runtime.scene_loader import trajdata_provider_config_to_params
    from omegaconf import OmegaConf
    from trajdata.dataset import UnifiedDataset
    from trajdata.dataset_specific.nuplan import nuplan_utils

    assert nuplan_utils.create_splits_logs().get("test"), "nuPlan log split table missing: set PYTHONPATH to the devkit overlay"
    out = Path(a.out).resolve()
    assert out != SHIPPED.resolve() and SHIPPED.resolve() not in out.parents, "never write the shipped cache"
    (out / "nuplan_test").mkdir(parents=True, exist_ok=True)
    if not (out / "nuplan_test/maps").exists():
        (out / "nuplan_test/maps").symlink_to(SHIPPED / "nuplan_test/maps")
    user = OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(Path(a.template) / "generated-user-config-0.yaml"))
    td = user.scene_provider.trajdata
    td.cache_location, td.dataset.data_dir, td.num_workers = str(out), a.raw, a.workers
    td.dataset.extra_params.num_timesteps_after = a.after
    td.rebuild_cache = True
    params = trajdata_provider_config_to_params(OmegaConf.to_object(td))
    print(json.dumps({k: (v if k != "dataset_kwargs" else {d: {x: (y if x != "central_tokens_config" else f"<{len(y)} tokens>")
                                                                for x, y in kw.items()} for d, kw in v.items()}) for k, v in params.items()},
                     default=str, indent=1), flush=True)
    ds = UnifiedDataset(**params, require_map_cache=False, verbose=True)
    print(f"built {ds.num_scenes()} scenes into {out}", flush=True)


def _cmp_scene(args):
    import dill
    import pandas as pd
    a, b, s = args
    r = {"scene": s}
    try:
        ma, mb = (dill.load(open(Path(x) / "nuplan_test" / s / "scene_metadata_dt0.10.dill", "rb")) for x in (a, b))
        r["len_a"], r["len_b"] = ma.length_timesteps, mb.length_timesteps
        da, db = (pd.read_feather(Path(x) / "nuplan_test" / s / "agent_data_dt0.10.feather") for x in (a, b))
        key = [c for c in ("agent_id", "scene_ts") if c in da.columns]
        da, db = (d.sort_values(key).reset_index(drop=True) if key else d for d in (da, db))
        n = min(ma.length_timesteps, mb.length_timesteps)
        pa, pb = (d[d["scene_ts"] < n].reset_index(drop=True) for d in (da, db))   # the common prefix
        r["rows_a"], r["rows_b"], r["prefix_rows_a"], r["prefix_rows_b"] = len(da), len(db), len(pa), len(pb)
        r["cols_equal"] = list(da.columns) == list(db.columns)
        if r["cols_equal"] and len(pa) == len(pb):
            num = pa.select_dtypes("number").columns
            r["prefix_max_abs"] = float((pa[num] - pb[num]).abs().max().max()) if len(pa) else 0.0
            r["prefix_ids_equal"] = bool((pa[key].values == pb[key].values).all()) if key else None
        r["agents_a"], r["agents_b"] = len(ma.agents), len(mb.agents)
        r["agents_prefix_equal"] = sorted(x.name for x in ma.agents if x.first_timestep < n) == sorted(x.name for x in mb.agents if x.first_timestep < n)
        ta, tb = (Path(x) / "nuplan_test" / s / "tls_data_dt0.05.feather" for x in (a, b))
        r["tls_bytes_equal"] = ta.exists() == tb.exists() and (not ta.exists() or ta.read_bytes() == tb.read_bytes())
    except Exception as e:  # a missing scene is a result
        r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return r


def compare(a):
    import dill
    scenes = Path(a.scenes).read_text().split() if a.scenes else [r.name for r in dill.load(open(Path(a.a) / "nuplan_test/scenes_list.dill", "rb"))]
    la = {r.name: tuple(r[1:4]) for r in dill.load(open(Path(a.a) / "nuplan_test/scenes_list.dill", "rb"))}
    lb = {r.name: tuple(r[1:4]) for r in dill.load(open(Path(a.b) / "nuplan_test/scenes_list.dill", "rb"))}
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_cmp_scene, [(a.a, a.b, s) for s in scenes], chunksize=16))
    ok = [r for r in rows if "error" not in r]
    exact = [r for r in ok if r["len_a"] == r["len_b"] and r["rows_a"] == r["rows_b"] and r.get("prefix_max_abs") == 0.0 and r["agents_prefix_equal"] and r["tls_bytes_equal"]]
    prefix = [r for r in ok if r.get("prefix_max_abs") is not None and r["prefix_max_abs"] <= 1e-6 and r["prefix_ids_equal"] and r["agents_prefix_equal"]]
    from collections import Counter
    summ = dict(a=a.a, b=a.b, scenes=len(scenes), list_a=len(la), list_b=len(lb),
                list_names_equal=set(la) == set(lb), list_location_split_equal=all(la[k][0::2] == lb.get(k, (None,) * 3)[0::2] for k in la),
                errors=len(rows) - len(ok), exact=len(exact), prefix_equal=len(prefix),
                len_pairs=Counter(f"{r['len_a']}->{r['len_b']}" for r in ok).most_common(8),
                worst_prefix=sorted(ok, key=lambda r: -(r.get("prefix_max_abs") or 0))[:5], first_errors=[r for r in rows if "error" in r][:5])
    Path(a.out).write_text(json.dumps(dict(summary=summ, rows=rows), indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--after", type=int, required=True), p.add_argument("--out", required=True)
    p.add_argument("--raw", default=str(DATA / "datasets/nuplan_raw")), p.add_argument("--template", default=str(TEMPLATE))
    p.add_argument("--workers", type=int, default=24)
    p.set_defaults(fn=build)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--out", required=True)
    p.add_argument("--scenes"), p.add_argument("--workers", type=int, default=24)
    p.set_defaults(fn=compare)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)
