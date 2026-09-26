"""Night queue 3, lane D, Q4b: openpilot `temporal` on the P5 v1 BA exam frames under the NAVSIM input protocol
(todos/2026-09-26-night-queue-3.md, Q4b and the [D] Q4b entry). Runs in envs/openpilot.

Per target frame t (every obs-role row of the P5 v1 BA index): the four 2 Hz history slots of NAVSIM (-1.5, -1.0, -0.5,
0 s) take the nearest recorded 5 Hz frame (ties to the later one: -1.4, -1.0, -0.4, 0 s; clamped to the stream start),
rendered by the P5 renderer unchanged (scripts/p5_openpilot.render, the same P4 rig as op_streams_vis), then the NAVSIM
rollout unchanged: zero state, each 2 Hz frame held until the next one on the 20 Hz clock (Cinque 31 steps), Lebowski
its 8 context-rate phases (scripts/navsim_zs_openpilot.schedule / rollout), desire none, right-hand traffic.
Output: <out>/<model>.npz (name, temporal float32, clamped).

  python scripts/nq3_d/q4b_openpilot.py --check 8     # the runner's P5-protocol path reproduces op_streams_vis
  python scripts/nq3_d/q4b_openpilot.py --workers 8   # the batch
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
import navsim_zs_openpilot as NO  # noqa: E402
from drive_backbones_openpilot import bounded_map  # noqa: E402
import p5_openpilot as PO  # noqa: E402
import wod_zeroshot_openpilot as WZ  # noqa: E402
from jevdrive import drive_backbones as D  # noqa: E402
from jevdrive import p5_openpilot as P  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

OFFS = np.array([-30, -20, -10, 0])      # NAVSIM history slots in 20 Hz ticks


def slots(ticks: np.ndarray, j: int) -> tuple[list, bool]:
    """Indices of the recorded frames nearest to t + OFFS (ties to the later frame), clamped to the stream start."""
    out, clamped = [], False
    for o in OFFS:
        want = ticks[j] + o
        if want < ticks[0]:
            out.append(0)
            clamped = True
            continue
        d = np.abs(ticks[: j + 1] - want)
        k = np.flatnonzero(d == d.min())[-1]
        out.append(int(k))
    return out, clamped


def job(item):
    st, need = item
    fr = PO.render([st["files"][k] for k in need])
    return st["key"], dict(zip(need, fr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=list(P.MODELS))
    a = ap.parse_args()
    from jevdrive.openpilot.model import OPModel
    import pandas as pd
    log = RunLog("nq3", "q4b-openpilot" + ("-check" if a.check else ""))
    plan = json.loads((P.root() / "op_plan.json").read_text())
    t = pd.read_parquet(P.root() / "index.parquet")
    want = set(t.frame_name[t.role == "obs"])
    WZ._init({}, {PO.SEQ: plan["calib"]}, ".")
    streams = []
    for st in plan["streams"]:
        ticks = np.array([int(n.rsplit("-", 1)[1]) for n in st["names"]])
        tg = [j for j in st["targets"] if st["names"][j] in want]
        if tg:
            streams.append((st, ticks, tg))
    n_tg = sum(len(x[2]) for x in streams)
    log.info(f"{len(streams)} streams, {n_tg} obs targets (of {len(want)})")
    assert n_tg == len(want)
    ex = None if a.check else ProcessPoolExecutor(a.workers, initializer=WZ._init, initargs=({}, {PO.SEQ: plan["calib"]}, "."))
    if ex is not None:
        list(ex.map(int, range(a.workers)))   # fork the renderers before the TensorRT sessions exist (p5_openpilot)
    models = {k: OPModel(k, WZ.MODELS[k], context_rate=(k == "lebowski"), taps=list(D.OP_TAPS[k].values())) for k in a.models}
    sched = {k: NO.schedule(k == "lebowski") for k in models}
    if a.check:
        # (1) the P5-protocol path (whole stream, 5 Hz, run_stream) reproduces the stored op_streams_vis temporal
        res = []
        for st, ticks, tg in streams[:: max(1, len(streams) // a.check)][: a.check]:
            fr = PO.render(st["files"][: tg[-1] + 1])
            for k, m in models.items():
                rows = PO.run_stream(m, fr, tg[-2:])
                ref = np.load(P.root("op_streams_vis", k) / f"{st['key']}.npz")
                at = {n: i for i, n in enumerate(ref["name"])}
                for j in tg[-2:]:
                    d = float(np.abs(rows[j]["temporal"] - ref["temporal"][at[st["names"][j]]]).max())
                    res.append({"stream": st["key"], "model": k, "target": st["names"][j], "max_abs_diff": d})
        log.info("P5-protocol reproduction: " + json.dumps(res))
        mx = max(r["max_abs_diff"] for r in res)
        log.event("check", rows=res, max_abs_diff=mx)
        assert mx <= 1e-3, f"runner does not reproduce op_streams_vis (max |diff| {mx})"
        return
    out = Path(a.out or P.root("nq3_q4b_navsim_protocol"))
    out.mkdir(parents=True, exist_ok=True)
    res = {k: {"name": [], "temporal": [], "clamped": []} for k in models}
    items, meta = [], {}
    for st, ticks, tg in streams:
        sl = {j: slots(ticks, j) for j in tg}
        need = sorted({k for s, _ in sl.values() for k in s})
        items.append((st, need))
        meta[st["key"]] = (st, sl)
    t0, n, tm = time.time(), 0, {k: 0.0 for k in models}
    with ex:
        for key, frames in bounded_map(ex, job, items, 2 * a.workers):
            st, sl = meta[key]
            for j, (s, cl) in sl.items():
                fr = np.stack([frames[k] for k in s])
                for k, m in models.items():
                    t1 = time.perf_counter()
                    NO.rollout(m, fr, sched[k], (1, 0))
                    tm[k] += time.perf_counter() - t1
                    res[k]["name"].append(st["names"][j])
                    res[k]["temporal"].append(m.tap_values[D.OP_TAPS[k]["temporal"]].astype(np.float32).copy())
                    res[k]["clamped"].append(cl)
                n += 1
            if n % 2000 < len(sl):
                el = time.time() - t0
                log.info(f"{n}/{n_tg} targets, {1e3 * el / n:.1f} ms/target wall, ETA {(n_tg - n) * el / n / 60:.0f} min; model s "
                         + ", ".join(f"{k} {v:.0f}" for k, v in tm.items()))
                log.event("progress", n=n, wall_s=el)
    for k in models:
        np.savez(out / f"{k}.npz", name=np.array(res[k]["name"]), temporal=np.stack(res[k]["temporal"]),
                 clamped=np.array(res[k]["clamped"]))
    log.info(f"done: {n} targets in {time.time() - t0:.0f} s, clamped {np.mean(res[a.models[0]]['clamped']):.4f} -> {out}")
    log.event("end", n=n, wall_s=time.time() - t0)


if __name__ == "__main__":
    main()
