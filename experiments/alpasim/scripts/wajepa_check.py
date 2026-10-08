"""Offline checks of the WA-JEPA online core (experiments/alpasim/lib/wajepa_core.py) on navtest tokens, no simulator.

  parity  the core fed a token's real CAM_L0 / F0 / R0 / B0 JPEGs (4 history frames) and index ego states must reproduce the stored plans of the
          run that scored navtest EPDMS 91.71 (runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl: the shipped agent
          through the NAVSIM devkit, fp32, flow seed 1)
  direct  with --direct K: the first K tokens also through NAVSIM's own SceneLoader -> agent.compute_trajectory in the same process: bit-level
          check of the AgentInput assembly (any remaining difference to the stored plans is the run, not the assembly)
  cold    the same tokens with only the m = 1, 2, 3 newest keyframes and states under each cold rule: distance of the 8 poses to the
          full-history plan and to the logged future (AlpaSim scenes start without history: decisions 0-2 of 10 have m = 1 / 2 / 3)

  cd $DATA_DIR/third_party/wajepa && PYTHONPATH=$PWD:$DATA_DIR/third_party/navsim $DATA_DIR/envs/wajepa/bin/python \
      <repo>/experiments/alpasim/scripts/wajepa_check.py --out $DATA_DIR/runs/alpasim/wajepa_check [--n 300] [--scenes FILE]
Tokens: those of --scenes (AlpaSim scene ids `<log>-<token>`) first, then a seed-0 draw from navtest up to --n. Inputs: op_parity's lb_navtest tab.npz
(4 relative poses, body velocities / accelerations, one-hot command, logged future) and the OpenScene logs for the image paths.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import wajepa_core as C  # noqa: E402

CAMN = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")


def main(a):
    import os
    D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
    tab = np.load(D / "runs/op_parity/cache/lb_navtest/tab.npz")
    names = tab["names"].tolist()
    row = {t: k for k, t in enumerate(names)}
    stored = pickle.load(open(D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl", "rb"))["trajectories"]
    first = [s.rsplit("-", 1)[1] for s in Path(a.scenes).read_text().split()] if a.scenes else []
    rest = [names[k] for k in np.random.default_rng(0).permutation(len(names))]
    toks = [t for t in dict.fromkeys([t for t in first if t in row] + rest) if t in stored][: a.n]
    cores = {c: C.Core("cuda", cold=c) for c in C.COLD}
    for c in C.COLD[1:]:
        cores[c].agent = cores[C.COLD[0]].agent                        # one copy of the weights
    root, logs = D / "datasets/navsim/sensor_blobs/test", {}
    ade = lambda x, y: float(np.linalg.norm(x[:, :2] - y[:, :2], axis=1).mean())  # noqa: E731
    rows, ms, dec, plans, frames_of = [], [], [], {}, []
    for n, t in enumerate(toks):
        r = row[t]
        lg = str(tab["log"][r])
        if lg not in logs:
            fr = pickle.load(open(D / "datasets/navsim/navsim_logs/test" / f"{lg}.pkl", "rb"))
            logs = {lg: (fr, {f["token"]: i for i, f in enumerate(fr)})}
        fr, at = logs[lg]
        i, frames = at[t], []
        for j in range(i - 3, i + 1):
            d = {}
            for c in CAMN:
                b = (root / fr[j]["cams"][c]["data_path"]).read_bytes()
                t1 = time.perf_counter()
                d[c] = C.decode(b)
                dec.append(1e3 * (time.perf_counter() - t1))
            frames.append(d)
        pose, vel = tab["pose"][r].astype(np.float64), tab["vel"][r].astype(np.float64)
        acc, cmd = tab["acc"][r][-1], tab["cmd"][r][-1]
        frames_of.append(frames) if n < a.amp else None
        full = cores["repeat"].plan(frames, pose, vel, acc, cmd)
        ms.append(full["ms"])
        plans[t] = full["poses"]
        st, fut = np.asarray(stored[t], np.float64), tab["fut"][r]
        have_fut = not np.isnan(fut).any()
        rec = {"token": t, "v0": float(np.linalg.norm(vel[-1])), "cmd": int(np.argmax(cmd)), "cmd_zero": bool(cmd.sum() == 0),
               "pose_ade": ade(full["poses"], st), "pose_max": float(np.abs(full["poses"] - st).max()),
               "xy_max": float(np.abs(full["poses"][:, :2] - st[:, :2]).max()), "head_max": float(np.abs(full["poses"][:, 2] - st[:, 2]).max()),
               "full_fut": ade(full["poses"], fut) if have_fut else None, "stored_fut": ade(st, fut) if have_fut else None}
        for m in (1, 2, 3):
            w = float(pose[4 - m + 1, 2] - pose[4 - m, 2]) / 0.5 if m > 1 else float(pose[3, 2] - pose[2, 2]) / 0.5
            for c in C.COLD:
                o = cores[c].plan(frames[-m:], pose[-m:], vel[-m:], acc, cmd, yaw_rate=w)
                rec[f"{c}{m}_full"] = ade(o["poses"], full["poses"])
                rec[f"{c}{m}_fut"] = ade(o["poses"], fut) if have_fut else None
                rec[f"{c}{m}_x4"] = float(o["poses"][-1, 0] - full["poses"][-1, 0])
        rows.append(rec)
        if n % 25 == 0:
            print(n, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    amp = []
    if a.amp:                                                           # bf16 autocast vs fp32 on the same inputs
        ca = C.Core("cuda", amp=True)
        ca.agent = cores["repeat"].agent
        for t, f in zip(toks[: a.amp], frames_of):
            r = row[t]
            o = ca.plan(f, tab["pose"][r].astype(np.float64), tab["vel"][r].astype(np.float64), tab["acc"][r][-1], tab["cmd"][r][-1])
            amp.append({"token": t, "ade_vs_fp32": ade(o["poses"], plans[t]), "ms": o["ms"]["infer"]})
    direct = []
    if a.direct:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from navsim.common.dataloader import SceneLoader
        sf = instantiate(OmegaConf.load(D / "third_party/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
        dt = toks[: a.direct]
        sf.tokens, sf.log_names = dt, sorted({str(tab["log"][row[t]]) for t in dt})
        od = D / "datasets/navsim"
        loader = SceneLoader(original_sensor_path=od / "sensor_blobs/test", data_path=od / "navsim_logs/test", scene_filter=sf,
                             sensor_config=cores["repeat"].agent.get_sensor_config())
        for t in dt:
            ref = np.asarray(cores["repeat"].agent.compute_trajectory(loader.get_agent_input_from_token(t)).poses, np.float64)
            direct.append({"token": t, "core_vs_direct_max": float(np.abs(plans[t] - ref).max()), "stored_vs_direct_ade": ade(np.asarray(stored[t], np.float64), ref)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mean = lambda k: float(np.mean([r[k] for r in rows if r[k] is not None]))  # noqa: E731
    q = lambda k, p: float(np.quantile([r[k] for r in rows if r[k] is not None], p))  # noqa: E731
    S = {"n": len(rows), "n_cmd_zero": sum(r["cmd_zero"] for r in rows),
         "parity": {k: {"mean": mean(k), "p99": q(k, 0.99), "max": q(k, 1.0)} for k in ("pose_ade", "pose_max", "xy_max", "head_max")},
         "full_fut_ade": mean("full_fut"), "stored_fut_ade": mean("stored_fut"),
         "cold": {f"{c}{m}": {"ade_vs_full": mean(f"{c}{m}_full"), "p90_vs_full": q(f"{c}{m}_full", 0.9), "ade_vs_log": mean(f"{c}{m}_fut"),
                              "dx4_mean": mean(f"{c}{m}_x4")} for c in C.COLD for m in (1, 2, 3)},
         "ms_warm": {k: float(np.median([x[k] for x in ms[5:]])) for k in ms[0]},
         "direct": {"n": len(direct), "core_vs_direct_max_m": max((d["core_vs_direct_max"] for d in direct), default=None),
                    "stored_vs_direct_ade_mean": float(np.mean([d["stored_vs_direct_ade"] for d in direct])) if direct else None},
         "amp": {"n": len(amp), "ade_vs_fp32": float(np.mean([x["ade_vs_fp32"] for x in amp])) if amp else None,
                "infer_ms_median": float(np.median([x["ms"] for x in amp[3:]])) if len(amp) > 3 else None,
                "fp32_infer_ms_median": float(np.median([x["infer"] for x in ms[3: a.amp]])) if a.amp > 6 else None},
         "decode_1080p_ms": {"median": float(np.median(dec)), "p95": float(np.quantile(dec, 0.95)), "max": float(np.max(dec))}}
    (out / "check.json").write_text(json.dumps({"summary": S, "rows": rows, "direct": direct}, indent=1))
    print(json.dumps(S, indent=1))
    (out / "DONE").write_text(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--direct", type=int, default=0)
    ap.add_argument("--amp", type=int, default=0, help="also run the first N tokens under bf16 autocast")
    ap.add_argument("--scenes", default="")
    main(ap.parse_args())
