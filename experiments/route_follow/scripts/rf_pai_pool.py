#!/usr/bin/env python3
"""Route-night section 9 step 5: the PAI on-policy rows of one checkpoint (onpol_pai's row pipeline: scripts/onp_rows.sh -> onp_cache.py build)
as an on-policy pool in the contract of experiments/onpol_nuplan/plans/on_pool_format.md, with the ROUTE-MEM inputs and the route hinge labels,
so route_mem/scripts/rm_train.py can train on it (--on on1_<name> --on-mode self --on-lon on_route). envs/op-train, CPU.

  rows      every cached row of the given scenes (--split, first column) that is usable (8 valid slots), before the rollout's first scorer
            event and before the log's end (onp_train's rule); names on:pai:<scene>:<k>, log = the scene id
  tab       ego = the row's ego_in (pai_core.plan's input, unrounded), fut = imit (NaN -> 0: not read in self mode), cam = cam_t, lht,
            speed = v0, served = the served poses
  @warp     front = the cached Cinque tokens of the row's 8 slots (onp_cache tok[tix]); teacher out = onp_cache t_out (shipped Cinque on the
            same tokens, pp_prep's rule); plan = 0 (never read on pool rows), di / pi of the cache
  agent     the row file's K = 32 actor boxes (rig frame at t0, 9 times, the navtrain_all-k32 layout), ok True
  road      ok False everywhere: PAI has no nuPlan raster layers (the road hinge is off on these rows; stated in the run's record)
  route     inputs: the driver's logged route messages (pai_driver drive.jsonl `route`: ts, waypoints, ego pose at ts) of the session, those
            with ts <= t0 carried to the row's anchor by rmem.Tracker -> feat/on1_<name>.npz (names, f, k, ok); gate: equal to the served
            `rmem` of the decision where the driver logged one (max |diff| over rows, 5-decimal rounding of the log)
            labels: the merged polyline of ALL messages of the rollout (rmem.merge, after the fact; no map) -> rm_build.label at the anchor ->
            route.npz (tokens, ok, poly (n, 51, 2), seglen, dz = 1.0, cov20 = the fed memory's +20 m point)

  rf_pai_pool.py --name PAIRB0 --rows /data/runs/route_night/rows/pol --cache /data/runs/route_night/cache/pol \
      --runs '/data/runs/route_night/roll/*/runs/*' --split experiments/onpol_pai/results/split/train160.tsv
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "experiments/route_mem/lib"), str(R / "experiments/route_mem/scripts")]
import rmem as RM  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/data"))
CR, FEAT = D / "runs/op_parity/cache", D / "runs/route_mem/feat"
LAB = -10.0 + 2.0 * np.arange(51)


def routes(runs):
    """drive.jsonl of the run dirs -> {session: [(ts, pose (3,), wp (m, 2))]} sorted by ts, {(session, t0): served rmem}."""
    M, S = {}, {}
    for f in sorted({f for g in runs for f in glob.glob(os.path.join(g, "driver", "drive.jsonl"))}):
        for line in open(f):
            if '"route"' not in line and '"rmem"' not in line:
                continue
            r = json.loads(line)
            if r.get("kind") == "route":
                M.setdefault(r["session"], []).append((int(r["ts"]), np.array(r["pose"], float), np.array(r["wp"], float).reshape(-1, 2)))
            elif r.get("kind") == "drive" and "rmem" in r:
                S[(r["session"], int(r["t0"]))] = np.array(r["rmem"], np.float32)
    return {k: sorted(v, key=lambda x: x[0]) for k, v in M.items()}, S


def main():
    from rm_build import label
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    keep_sc = {l.split("\t")[0] for l in Path(a.split).read_text().splitlines() if l.startswith("clipgt-")}
    T = np.load(Path(a.cache) / "tab.npz")
    scenes = T["scenes"][T["sid"]]
    sel = np.flatnonzero(T["usable"] & ~T["after_event"] & ~T["after_end"] & np.isin(scenes, list(keep_sc)))
    n = len(sel)
    sc, kk, t0 = scenes[sel], T["k"][sel].astype(int), T["t0_us"][sel].astype(np.int64)
    names = np.array([f"on:pai:{s}:{k}" for s, k in zip(sc, kk)])
    assert len(set(names.tolist())) == n, "duplicate row names"
    msgs, served = routes(a.runs)
    anchor, sess = np.zeros((n, 3)), np.empty(n, object)
    for s in np.unique(sc):                                           # anchors and sessions from the row files
        z = np.load(Path(a.rows) / f"{s}.npz")
        pos = {int(t): i for i, t in enumerate(z["t0_us"])}
        for i in np.flatnonzero(sc == s):
            anchor[i], sess[i] = z["anchor"][pos[int(t0[i])]], str(z["session"])
    F, ok, poly = np.zeros((n, RM.DIM), np.float16), np.zeros(n, bool), np.full((n, len(LAB), 2), np.nan, np.float32)
    dmax, ncmp, nomsg = 0.0, 0, 0
    full = {}
    for se in np.unique(sess):
        m = msgs.get(se, [])
        if not m:
            nomsg += int((sess == se).sum())
            continue
        B = [(ts, RM.to_local(p, wp[np.isfinite(wp).all(1)])) for ts, p, wp in m if np.isfinite(wp).all(1).any()]
        full[se] = RM.merge([b for _, b in B])
        tr, pose = RM.Tracker(), {ts: p for ts, p, _ in m}
        for ts, _, wp in m:
            tr.add(ts, wp)
        for i in np.flatnonzero(sess == se):
            f = tr.features(int(t0[i]), pose.get, anchor[i])
            F[i], ok[i] = f, len([1 for ts, _, _ in m if ts <= t0[i]]) > 0
            if (se, int(t0[i])) in served:
                dmax, ncmp = max(dmax, float(np.abs(served[se, int(t0[i])] - f).max())), ncmp + 1
            if full[se] is not None:
                poly[i] = label(full[se], anchor[i])
    seg = np.linalg.norm(np.diff(poly, axis=1), axis=-1).astype(np.float32)
    rok = ok & (np.isfinite(poly).all(-1).sum(1) >= 2)
    sm = RM.summary(F.astype(np.float32))
    pool, warp = CR / f"on1_{a.name}", CR / f"on1_{a.name}@warp"
    pool.mkdir(parents=True, exist_ok=True), warp.mkdir(parents=True, exist_ok=True), FEAT.mkdir(parents=True, exist_ok=True)
    imit = np.nan_to_num(T["imit"][sel]).astype(np.float32)
    np.savez(pool / "tab.npz", names=names, log=sc, ego=T["ego_in"][sel].astype(np.float32), fut=imit, cam=T["cam_t"][sel].astype(np.float32),
             lht=T["lht"][sel].astype(bool), speed=T["v0"][sel].astype(np.float32), served=T["poses"][sel].astype(np.float32))
    tok = np.load(Path(a.cache) / "tok.npy", mmap_mode="r")
    fr = np.lib.format.open_memmap(warp / "front.npy", "w+", np.float16, (n, 8) + tok.shape[1:])
    tix = T["tix"][sel]
    for i in range(0, n, 512):
        fr[i:i + 512] = tok[tix[i:i + 512]]
    fr.flush()
    del fr
    np.savez(warp / "teacher.npz", out=T["t_out"][sel].astype(np.float32), plan=np.zeros((n, 33, 15), np.float32), di=T["di"], pi=T["pi"])
    np.savez(pool / "agent.npz", tokens=names, ok=np.ones(n, bool), box=T["box"][sel].astype(np.float32), valid=T["box_ok"][sel].astype(bool))
    np.savez(pool / "road.npz", tokens=names, ok=np.zeros(n, bool), sdf=np.zeros((n, 128, 96), np.float16))
    np.savez(pool / "route.npz", tokens=names, ok=rok, poly=poly, seglen=seg, dz=np.ones(n, np.float32), cov20=np.asarray(sm["c20"], bool) & ok)
    np.savez(FEAT / f"on1_{a.name}.npz", names=names, f=F, k=kk, ok=ok)
    fin = np.isfinite(poly[rok, :, 0])
    rec = dict(name=a.name, rows=n, scenes=int(len(np.unique(sc))), rows_cached=int(len(scenes)), route_in_ok=float(ok.mean()), rows_without_messages=nomsg,
               served_compared=ncmp, served_max_abs_diff=dmax, c20=float(sm["c20"][ok].mean()) if ok.any() else 0.0, route_ok=float(rok.mean()),
               route_pts_med=float(np.median(fin.sum(1))) if len(fin) else 0.0,
               route_lat0_med=float(np.nanmedian(np.abs(poly[rok, 5, 1]))) if rok.any() else float("nan"), road_rows=0, pool=str(pool))
    (pool / "build.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
