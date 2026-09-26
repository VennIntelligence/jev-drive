#!/usr/bin/env python
"""SCH GPU helper: lane C's Alpamayo 1.5 exam (scripts/nq3_c_alpamayo.py run) on a second card.

Same model load (I.load + the runner's expert_graph config), same Prep / infer / record, batch 1 (the rule-8
bit-identical path). Takes whole priority-0 units from the END of the runner's order (the runner walks from the start),
one O_EXCL claim per unit, and writes one part per unit as nq3_alpamayo_parts/part_9UUUU.npz (UUUU = unit index):
the runner's glob picks them up at its next consolidate, and its own numbering (count of parts at start) never
reaches 90000. It never consolidates while the runner lives, and takes no new unit after --stop-at (08:00). It stops before a unit within --gap frames of the
runner's last written frame, so the two never work on the same unit.
Run in the alpamayo1.5 venv from the repo root.

  verify --unit 0  one whole unit the runner already wrote, recomputed here and compared (xyz bit for bit, text)
  run              the loop above
  consolidate --owner-pid P   the runner's consolidate(), refused while P is alive
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import nq3_c_alpamayo as A  # noqa: E402

CLAIMS = A.P6 / "nq3_alpamayo_sch_claims"


def load():
    model, processor = A.I.load(A.CFG.attn)
    A.I.apply(model, A.I.Config(name="nq3_p6_graph", expert_graph=True))   # the runner's cmd_run config
    return model, processor


def runner_parts():
    return sorted(p for p in A.PARTS.glob("part_?????.npz") if not p.name.startswith("part_9"))


def done_frames(parts) -> set:
    return {str(f) for p in parts for f in np.load(p)["frame_name"]}


def cmd_verify(a, log):
    """Recompute one whole unit the runner already wrote (--unit, default its first) and compare with its parts."""
    xyz_all, meta = A.read_parts()
    at = {f: i for i, f in enumerate(meta.frame_name)}
    t = A.frames()
    units = [g for _, g in t.groupby(["priority", "base_id", "seed"], sort=False)]
    g = units[a.unit]
    assert g.frame_name.isin(at).all(), f"unit {a.unit} is not finished by the runner"
    model, processor = load()
    prep = A.Prep(processor)
    res = []
    for p in A.prefetch(prep, list(g.itertuples()), a.workers, 4 * a.workers):
        i = at[p["row"].frame_name]
        m = meta.iloc[i]
        o = A.infer(model, A.to_cuda(p["inputs"]), p["seed"])
        res.append({"frame": m.frame_name, "seed_equal": p["seed"] == m.seed, "xyz_equal": bool(np.array_equal(o["xyz"], xyz_all[i])),
                    "max_abs": float(np.abs(o["xyz"] - xyz_all[i]).max()), "cot_equal": o.get("cot", "") == m.cot,
                    "wall_s": o["wall"]})
        log.info(str(res[-1]))
    ok = all(r["xyz_equal"] and r["cot_equal"] and r["seed_equal"] for r in res)
    log.info(f"verify unit {a.unit}, {len(res)} frames: {'identical' if ok else 'DIFFERENT'}, "
             f"{np.mean([r['wall_s'] for r in res]):.2f} s/frame GPU")
    sys.exit(0 if ok else 1)


def cmd_run(a, log):
    t = A.frames()
    units = [(k, g) for k, g in t.groupby(["priority", "base_id", "seed"], sort=False)]
    pos = np.cumsum([0] + [len(g) for _, g in units])            # frame offset of each unit in the runner's order
    CLAIMS.mkdir(parents=True, exist_ok=True)
    model, processor, prep = None, None, None
    n = 0
    for u in range(len(units) - 1, -1, -1):
        (pri, base, seed), g = units[u]
        if pri != a.priority:
            continue
        part = A.PARTS / f"part_9{u:04d}.npz"
        if part.exists():
            continue
        if time.strftime("%H:%M") >= a.stop_at and time.strftime("%H:%M") < "12:00":
            log.info(f"stop: {a.stop_at} reached, no new unit")
            break
        rp = runner_parts()
        dn = done_frames(rp)
        last = max((i for i, (_, gg) in enumerate(units) if gg.frame_name.isin(dn).any()), default=-1)
        if pos[u] - pos[last + 1] < a.gap:                       # the runner's in-flight part may already cover it
            log.info(f"stop: unit {u} is within {a.gap} frames of the runner (last written unit {last})")
            break
        if g.frame_name.isin(dn).all():
            continue
        lock = CLAIMS / f"u{u:04d}.lock"
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        try:
            if model is None:
                model, processor = load()
                prep = A.Prep(processor)
            t0, recs = time.time(), []
            for p in A.prefetch(prep, list(g.itertuples()), a.workers, 4 * a.workers):
                o = A.infer(model, A.to_cuda(p["inputs"]), p["seed"])
                recs.append(dict(A.record(p, o), batch=1))
            A.PARTS.mkdir(parents=True, exist_ok=True)
            df_tmp = A.PARTS / f"part_9{u:04d}.tmp.npz"
            import pandas as pd
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "xyz"} for r in recs])
            np.savez(df_tmp, frame_name=df.frame_name.to_numpy().astype(str), xyz=np.stack([r["xyz"] for r in recs]),
                     meta=np.array(df.to_json(orient="records")))
            df_tmp.replace(part)                                 # A.write_part's format under our own name
            n += len(g)
            rate = (time.time() - t0) / len(g)
            log.info(f"unit {u} ({pri}, {base}, {seed}): {len(g)} frames, {rate:.2f} s/frame; {n} frames this run")
            log.event("unit", unit=u, frames=len(g), s_per_frame=rate, total=n)
        finally:
            lock.unlink(missing_ok=True)
    log.info(f"exit: {n} frames")


def cmd_consolidate(a, log):
    try:
        alive = Path(f"/proc/{a.owner_pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        alive = False
    if alive:
        sys.exit(f"runner {a.owner_pid} is alive: it consolidates itself")
    log.info(f"consolidated: {A.consolidate()} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("verify", "run", "consolidate"))
    ap.add_argument("--unit", type=int, default=0)
    ap.add_argument("--priority", type=int, default=0)
    ap.add_argument("--gap", type=int, default=600, help="frames kept free ahead of the runner's last written frame")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stop-at", default="08:00", help="box clock (morning): no new unit from then on")
    ap.add_argument("--owner-pid", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    log = A.RunLog("sch_gpu", "alpamayo", a.cmd)
    log.info(f"args {vars(a)} -> {log.dir}")
    {"verify": cmd_verify, "run": cmd_run, "consolidate": cmd_consolidate}[a.cmd](a, log)
    log.close()
