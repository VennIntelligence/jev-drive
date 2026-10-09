#!/usr/bin/env python3
"""BODY1 diagnosis of the baseline zeros (results/zeros_diagnosis.md): data side. Why does the contact head not flag P2H10-F's zero-score
scenes in closed loop, and what would have prevented each zero? Read-only on navtest logs: nothing is fitted or chosen here, simulator
objects and the simulator's map are labels for the diagnosis only and never become training rows.

  sel      every zero of P2H10-F-s0 / -s1 on the 700 scenes (TR1's baseline runs, rollout.asl kept for failed rollouts) and one clean scene
           (score 1 in both seeds) per zero, same log segment when there is one, else same vehicle-day, else same chunk -> zeros.csv,
           clean-s{0,1}.txt                                                                  (.venv python)
  extract  c1_extract.py logs / map / msgs (AlpaSim's venv, unchanged) for the zero rollouts of the baseline runs and for the clean
           reruns (rp_direct.py with RP_KEEP=1) -> x/<set>/{logs.pkl, map.pkl, msgs/}         (.venv python; sets: zero-s0, zero-s1, clean-s0, clean-s1)
  replay   the logged driver-side messages through the real driver class (swv1_replay.py's loop), and the contact head (serve_body.Body, the
           two full-scale seeds, the gate's input standard) on the served plan and on every candidate of `candidates` -> replay/<set>.npz
           (envs/op-train, one GPU)

Candidates per decision (index: name), all built from the served plan's 8 poses: 0 the plan; 1-10 the lateral ramps of serve_body.A_RP
(+-0.3 .. 1.5 m at 4 s); 11-12 the plan's path at 0.75 / 0.5 of its arc length per step (a mild slow-down, the generator's `arc` family);
13-32 each ramp at 0.75 and at 0.5 of the arc length.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent), str(_H.parents[1] / "lib"), str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[2] / "alpasim" / "lib"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
BASE = DATA / "runs/alpasim/tr1/a/runs"
OUT = DATA / "runs/body1/diag"
ALP_PY = DATA / "third_party/alpasim/.venv/bin/python"
SETS = ("zero-s0", "zero-s1", "clean-s0", "clean-s1")
A_LAT = np.array([s_ * a_ for a_ in (0.3, 0.6, 0.9, 1.2, 1.5) for s_ in (1.0, -1.0)])      # serve_body.A_RP
SLOW = (0.75, 0.5)
CAND = ["plan"] + [f"lat{a:+.1f}" for a in A_LAT] + [f"slow{c}" for c in SLOW] + [f"lat{a:+.1f}+slow{c}" for c in SLOW for a in A_LAT]


def base_runs():
    return {(s, c): sorted(glob.glob(str(BASE / f"P2H10-F-s{s}-chunk{c}/*/")))[-1].rstrip("/") for s in (0, 1) for c in (0, 1, 2)}


def scores(run):
    return {r["clipgt_id"]: r for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]}


def reason(r):
    m = r.get("score_metrics") or r["metrics"]
    return next((f for f in ("collision_at_fault", "offroad", "left_corridor_laterally") if m.get(f)), r.get("failure_reason") or "")


def score_of(r):
    for k in ("scene_score", "score"):
        if r.get(k) is not None:
            return float(r[k])
    raise KeyError(f"no score key in {sorted(r)}")


# ---------------------------------------------------------------- sel
def cmd_sel(a, run):
    R = {k: scores(v) for k, v in base_runs().items()}
    sc = {s: {sid: (score_of(r), c, r) for (s_, c), d in R.items() if s_ == s for sid, r in d.items()} for s in (0, 1)}
    run.info(f"scenes per seed {[len(v) for v in sc.values()]}; row keys {sorted(next(iter(sc[0].values()))[2])}")
    zeros = [(s, sid, c, reason(r)) for s in (0, 1) for sid, (v, c, r) in sorted(sc[s].items()) if v == 0]
    clean = sorted(sid for sid in sc[0] if sc[0][sid][0] == 1.0 and sc[1].get(sid, (0,))[0] == 1.0)
    rng, used, rows = np.random.default_rng(0), set(), []
    for s, sid, c, why in zeros:
        seg, day = sid.rsplit("-", 1)[0], "_".join(sid.split("_")[:2])
        for lvl, pool in (("segment", [x for x in clean if x.rsplit("-", 1)[0] == seg]), ("vehicle-day", [x for x in clean if "_".join(x.split("_")[:2]) == day]),
                          ("chunk", [x for x in clean if sc[0][x][1] == c])):
            pool = [x for x in pool if (s, x) not in used]
            if pool:
                m = pool[int(rng.integers(len(pool)))]
                break
        used.add((s, m))
        rows.append(dict(seed=s, scene=sid, chunk=c, reason=why, run=base_runs()[s, c], clean=m, match=lvl))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "zeros.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader(), w.writerows(rows)
    for s in (0, 1):
        (OUT / f"clean-s{s}.txt").write_text("\n".join(r["clean"] for r in rows if r["seed"] == s) + "\n")
    cnt = {s: {w: sum(1 for r in rows if r["seed"] == s and r["reason"] == w) for w in sorted({r["reason"] for r in rows})} for s in (0, 1)}
    run.info(f"zeros {len(rows)}: {cnt}; clean pool {len(clean)}; match levels { {k: sum(r['match'] == k for r in rows) for k in ('segment', 'vehicle-day', 'chunk')} }")
    run.summary.update(zeros=len(rows), by_seed=cnt)


# ---------------------------------------------------------------- extract
def set_runs(st):
    """{run dir: [scenes]} of one set."""
    kind, s = st.split("-s")
    rows = [r for r in csv.DictReader(open(OUT / "zeros.csv")) if r["seed"] == s]
    by = {}
    if kind == "zero":
        for r in rows:
            by.setdefault(r["run"], []).append(r["scene"])
    else:
        man = json.loads((OUT / "cl/clean/manifest.json").read_text())
        for d in man[f"clean-s{s}"]:
            for sid in scores(d):
                if glob.glob(f"{d}/rollouts/{sid}/*/rollout.asl"):
                    by.setdefault(d, []).append(sid)
    return by


def cmd_extract(a, run):
    X = str(_H.parents[2] / "alpasim/scripts/c1_extract.py")
    env = dict(os.environ, PYTHONPATH=str(DATA / "third_party/alpasim/src/utils"))
    for st in a.sets:
        logs, maps = {}, {}
        (OUT / "x" / st / "msgs").mkdir(parents=True, exist_ok=True)
        for d, sc in set_runs(st).items():
            with tempfile.TemporaryDirectory() as t:
                Path(t, "s.txt").write_text("\n".join(sc) + "\n")
                for cmd, out in (("logs", f"{t}/l.pkl"), ("map", f"{t}/m.pkl"), ("msgs", str(OUT / "x" / st / "msgs"))):
                    subprocess.run([str(ALP_PY), X, cmd, "--run", d, "--scenes", f"{t}/s.txt", "--out", out, "--jobs", "16", "--radius", "120"], check=True, env=env,
                                   cwd=DATA / "third_party/alpasim", stdout=subprocess.DEVNULL)
                L = pickle.load(open(f"{t}/l.pkl", "rb"))
                logs |= {k: v for k, v in L.items() if k in sc}
                maps |= pickle.load(open(f"{t}/m.pkl", "rb"))
        pickle.dump(logs, open(OUT / "x" / st / "logs.pkl", "wb"), protocol=4)
        pickle.dump(maps, open(OUT / "x" / st / "map.pkl", "wb"), protocol=4)
        run.info(f"{st}: logs {len(logs)}, maps {len(maps)}, msgs {len(list((OUT / 'x' / st / 'msgs').glob('*.pkl')))}")
        run.summary[st] = len(logs)


# ---------------------------------------------------------------- candidates
def at_arc(P, c):
    """One plan (8, 3) -> the poses at c x its arc length per step, along the polyline origin -> P (the generator's `arc` family, c <= 1)."""
    P0 = np.r_[np.zeros((1, 3)), np.asarray(P, np.float64)]
    P0[:, 2] = np.unwrap(P0[:, 2])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P0[:, :2], axis=0).T))]
    return np.c_[[np.interp(c * s[1:], s + 1e-9 * np.arange(9), P0[:, j]) for j in range(3)]].T


def candidates(P):
    """The served plan (8, 3) -> (33, 8, 3) float64 in the order of CAND."""
    import serve_body as SB
    lat = SB.ramps(P, A_LAT)[0]
    return np.concatenate([np.asarray(P, np.float64)[None], lat, [at_arc(P, c) for c in SLOW], [at_arc(q, c) for c in SLOW for q in lat]])


# ---------------------------------------------------------------- replay (GPU)
class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def cmd_replay(a, run):
    import torch
    import sh30_driver as D
    import serve_body as SB
    assert D.BD is None, "replay runs with the BODY1 switches off"
    pb, ctx = D.egodriver_pb2, Ctx()
    body = SB.Body("cuda")
    (OUT / "replay").mkdir(parents=True, exist_ok=True)
    for st in a.sets:
        tag = f"P2H10-F-s{st[-1]}"
        core = D.C.Core(tag, "cuda", "backwarp")
        drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        last, orig = {}, core.plan

        def plan(*x, _o=orig, **k):
            last["o"] = _o(*x, **k)
            return last["o"]
        core.plan = plan
        R = dict(scene=[], k=[], now=[], poses=[], ego=[], n_slots=[], cmd=[], za=[], zb=[], v0=[], frame=[])
        files = sorted((OUT / "x" / st / "msgs").glob("*.pkl"))
        for i, f in enumerate(files):
            uuid, k = None, 0
            for kind, raw in pickle.load(open(f, "rb")):
                if kind == "driver_session_request":
                    req = pb.DriveSessionRequest.FromString(raw)
                    uuid = req.session_uuid
                    drv.start_session(req, ctx)
                elif kind == "driver_camera_image":
                    drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
                elif kind == "driver_ego_trajectory":
                    drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
                elif kind == "route_request":
                    drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
                elif kind == "driver_request":
                    req = pb.DriveRequest.FromString(raw)
                    drv.drive(req, ctx)
                    o = last["o"]
                    Q = candidates(o["poses"])
                    za, zb = body.score_many(o["tokens"], o["valid"], o["ego"], Q)
                    R["scene"].append(f.stem), R["k"].append(k), R["now"].append(int(req.time_now_us)), R["poses"].append(np.asarray(o["poses"], np.float64))
                    R["ego"].append(np.asarray(o["ego"], np.float32)), R["n_slots"].append(int(np.sum(o["valid"]))), R["cmd"].append(int(np.argmax(drv.sessions[uuid].cmd)))
                    R["za"].append(za), R["zb"].append(zb)
                    R["frame"].append(np.asarray(o["cur"], np.uint8)[-1, 0] if a.frames else np.zeros(0, np.uint8))
                    k += 1
            drv.sessions.pop(uuid, None)
            if i % 10 == 0:
                run.info(f"{st}: {i} / {len(files)} {f.stem}")
        np.savez_compressed(OUT / "replay" / f"{st}.npz", **{k: np.array(v) for k, v in R.items() if k != "v0"}, cand=np.array(CAND))
        run.info(f"{st}: {len(R['k'])} decisions of {len(files)} scenes")
        run.summary[st] = len(R["k"])
        del core, drv
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("sel", "extract", "replay"))
    ap.add_argument("--sets", nargs="*", default=list(SETS)), ap.add_argument("--frames", type=int, default=1)
    a = ap.parse_args()
    from jevdrive.run import Run
    with Run("body1", f"diag-{a.cmd}", config=vars(a)) as run:
        dict(sel=cmd_sel, extract=cmd_extract, replay=cmd_replay)[a.cmd](a, run)


if __name__ == "__main__":
    main()
