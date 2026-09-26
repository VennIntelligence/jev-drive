"""Build counterfactual PufferDrive scene variants for the state-space policy behaviour exam.

Input: the enriched GPUDrive JSONs and manifest.csv written by BehaviorBench's
data_utils/womd/extract_interactive_benchmark.py (one row per (scenario, ego track)).
Output: one split dir per variant under --out, each with map_XXXXXX.bin + manifest.csv,
plus meta.pkl describing every episode (see todos/2026-09-26-state-space-policies.md):

  base       original scene
  nopartner  the other tracks_to_predict vehicle removed           (exam A, x-)
  nullrm     one vehicle >= 30 m from the ego log path removed     (exam A, null)
  obstacle   a static 4.8 x 2.0 m car inserted on the ego log path (exam B, x+)
  shift      ego history (t <= t0) shifted +-1.5 m laterally        (exam C)

Run with the statepol env (needs pufferlib from behavior-bench on sys.path).
"""
import argparse
import copy
import csv
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

T0, T = 10, 91
NULL_DIST, OBS_LEN, OBS_W, SHIFT = 30.0, 4.8, 2.0, 1.5


def arr(obj):
    p = np.array([[q["x"], q["y"]] for q in obj["position"]], np.float64)
    v = np.array([[q["x"], q["y"]] for q in obj["velocity"]], np.float64)
    return p, v, np.asarray(obj["heading"], np.float64), np.asarray(obj["valid"], bool)


def remove(scene, idx):
    s = copy.deepcopy(scene)
    del s["objects"][idx]
    md = s.setdefault("metadata", {})
    fix = lambda i: i - (i > idx)
    if md.get("sdc_track_index", -1) >= 0:
        md["sdc_track_index"] = -1 if md["sdc_track_index"] == idx else fix(md["sdc_track_index"])
    md["tracks_to_predict"] = [dict(t, track_index=fix(t["track_index"]))
                               for t in md.get("tracks_to_predict", []) if t["track_index"] != idx]
    return s


def path_point(xy, s_query):
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    i = int(np.clip(np.searchsorted(s, s_query) - 1, 0, len(seg) - 1))
    a = (s_query - s[i]) / max(seg[i], 1e-6)
    d = xy[i + 1] - xy[i]
    return xy[i] + a * d, float(np.arctan2(d[1], d[0]))


def build(args):
    json_path, ego, rng_seed, out_dir = args
    r = _build(json_path, ego, rng_seed)
    if r is None:
        return None
    from pufferlib.ocean.drive.drive import save_map_binary
    meta, variants = r
    for v, (scene, e) in variants.items():
        d = Path(out_dir) / v
        d.mkdir(parents=True, exist_ok=True)
        save_map_binary(scene, str(d / f"tmp_{rng_seed:06d}.bin"), rng_seed)
    meta["ego_entity"] = {v: e for v, (_, e) in variants.items()}
    meta["job"] = rng_seed
    return meta


def _build(json_path, ego, rng_seed):
    scene = json.load(open(json_path))
    objs = scene["objects"]
    ep, ev, eh, eva = arr(objs[ego])
    if not eva[T0] or eva[T0:].sum() < 60:
        return None
    fut = ep[T0:][eva[T0:]]
    tracks = [t["track_index"] for t in scene["metadata"]["tracks_to_predict"]]
    partner = next((t for t in tracks if t != ego), None)
    if partner is None:
        return None
    rng = np.random.default_rng(rng_seed)
    meta = dict(scenario_id=scene.get("scenario_id", ""), json=str(json_path), ego=ego, partner=partner,
                ego_xy=ep, ego_h=eh, ego_valid=eva, partner_xy=arr(objs[partner])[0], partner_valid=arr(objs[partner])[3])
    out = {"base": (scene, ego)}
    out["nopartner"] = (remove(scene, partner), ego - (ego > partner))
    # null: a vehicle that never comes within NULL_DIST of the ego's logged path
    cands = []
    for j, o in enumerate(objs):
        if j in (ego, partner) or o.get("type") != "vehicle":
            continue
        p, _, _, va = arr(o)
        if not va[T0]:
            continue
        d = np.linalg.norm(p[va][:, None] - fut[None], axis=-1).min()
        if d >= NULL_DIST:
            cands.append(j)
    if cands:
        j = int(rng.choice(cands))
        out["nullrm"] = (remove(scene, j), ego - (ego > j))
        meta["null_removed"] = j
    # straight-road eligibility for exams B and C
    seg = np.linalg.norm(np.diff(fut, axis=0), axis=1)
    length = seg.sum()
    hv = eh[T0:][eva[T0:]]
    dh = abs((hv[-1] - hv[0] + np.pi) % (2 * np.pi) - np.pi)
    v0 = np.linalg.norm(ev[T0])
    meta.update(v0=v0, path_len=length, dheading=dh)
    if v0 >= 5 and length >= 40 and dh < np.deg2rad(20):
        others = [(arr(o)[0], arr(o)[3]) for j, o in enumerate(objs) if j != ego]
        for s_obs in (max(20.0, 2.5 * v0), max(20.0, 2.5 * v0) + 10):
            if s_obs > length - 5:
                break
            c, h = path_point(fut, s_obs)
            if all((np.linalg.norm(p[T0:][va[T0:]] - c, axis=1) >= 5).all() for p, va in others):
                s2 = copy.deepcopy(scene)
                s2["objects"].append(dict(
                    type="vehicle", id=987654, width=OBS_W, length=OBS_LEN, height=1.5, mark_as_expert=1,
                    position=[dict(x=float(c[0]), y=float(c[1]), z=float(objs[ego]["position"][T0].get("z", 0)))] * T,
                    velocity=[dict(x=0.0, y=0.0)] * T, heading=[h] * T, valid=[1] * T,
                    goalPosition=dict(x=float(c[0]), y=float(c[1]), z=0.0)))
                out["obstacle"] = (s2, ego)
                meta.update(obs_xy=c, obs_h=h, obs_s=s_obs)
                break
        sign = 1.0 if rng_seed % 2 == 0 else -1.0
        n = np.array([-np.sin(eh[T0]), np.cos(eh[T0])]) * SHIFT * sign
        s3 = copy.deepcopy(scene)
        for t in range(T0 + 1):
            q = s3["objects"][ego]["position"][t]
            q["x"] += float(n[0]); q["y"] += float(n[1])
        out["shift"] = (s3, ego)
        meta["shift_sign"] = sign
    return meta, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in sorted(rng.permutation(len(rows))[: int(a.max_episodes * 1.3)])]
    out = Path(a.out)
    jobs = [(Path(a.enriched) / r["original_filename"], int(r["ego_agent_idx"]), k, str(out)) for k, r in enumerate(rows)]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap(build, jobs, chunksize=2) if r is not None][: a.max_episodes]
    metas = {v: [] for v in ("base", "nopartner", "nullrm", "obstacle", "shift")}
    for ep_id, meta in enumerate(res):
        meta["episode"] = ep_id
        for v, ego in meta["ego_entity"].items():
            k = len(metas[v])
            (out / v / f"tmp_{meta['job']:06d}.bin").rename(out / v / f"map_{k:06d}.bin")
            metas[v].append(dict(meta, map_id=k, ego_entity=ego))
    for p in out.glob("*/tmp_*.bin"):
        p.unlink()
    for v, ms in metas.items():
        with open(out / v / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["new_filename", "ego_agent_idx", "episode", "scenario_id"])
            w.writeheader()
            for m in ms:
                w.writerow(dict(new_filename=f"map_{m['map_id']:06d}.bin", ego_agent_idx=m["ego_entity"],
                                episode=m["episode"], scenario_id=m["scenario_id"]))
        pickle.dump(ms, open(out / v / "meta.pkl", "wb"))
        print(f"{v}: {len(ms)} episodes")


if __name__ == "__main__":
    main()
