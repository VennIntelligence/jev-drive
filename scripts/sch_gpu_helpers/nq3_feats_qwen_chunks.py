"""SCH GPU helper: lane C's Qwen feature chunks (jevdrive.nq3_feats qwen) on an idle card.

The per-chunk body is nq3_feats.qwen's, unchanged (same rows, chunking, make_fx, features.extract with
p4_carla.ClipFiles, index.parquet, meta.json last). Chunks go to a staging root first, because lane C's own process
(nq3_feats qwen, which computed its chunk list once at start) would otherwise redo them; `install` moves finished
staged chunks into processed/carla_p6/features only once that process is gone.

  run      --chunks 8,7,6   claim (O_EXCL lock in the staging root, p5_qwen._claim) and extract each chunk not yet
                            done in either root
  verify   --n 16           the first n clips of chunk 0 into a scratch dir, compared bit for bit with the stored c000
  install  --owner-pid P    rename every finished staged chunk into features/ (refuses while P is alive; a partial
                            target without meta.json is replaced)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ["P5_SET"] = "carla_p6"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive import nq3_feats as N, p5_qwen  # noqa: E402

REAL = N.proc("features", "x").parent
STAGE = N.proc("features_sch", "x").parent
TAPS = ("L18_last", "L18_mean", "vis_mean", "vit_mean")


def chunks():
    t = N.rows(1)                                  # nq3_feats.qwen's default max_priority
    return [t.iloc[i:i + N.CHUNK] for i in range(0, len(t), N.CHUNK)]


def extract_chunk(fx, rl, c, dst: Path, tag: str, batch: int, workers: int):
    """nq3_feats.qwen's loop body."""
    from jevdrive import features as F, p4_carla as p4
    dst.mkdir(parents=True, exist_ok=True)
    st = F.extract(fx, c.files.map(list).tolist(), batch, workers, dst, rl, tag, dataset=p4.ClipFiles)
    c[["frame_name"]].to_parquet(dst / "index.parquet", index=False)
    (dst / "meta.json").write_text(pd.Series({"recipe": "P3(d'') qwenvid", "batch_size": batch, **st}).to_json())
    return st


def cmd_run(a):
    from jevdrive import waymo_qwenvid as qv
    from jevdrive.runlog import RunLog
    cs = chunks()
    rl = RunLog("sch_gpu", f"nq3-feats-qwen-{os.getpid()}")
    fx = None
    for i in map(int, a.chunks.split(",")):
        name = f"c{i:03d}"
        if (REAL / name / "meta.json").exists() or (STAGE / name / "meta.json").exists():
            continue
        lock = STAGE / f"{name}.lock"
        if not p5_qwen._claim(lock):
            continue
        try:
            fx = fx or qv.make_fx(compile=False)
            (STAGE / name / "meta.json").unlink(missing_ok=True)
            st = extract_chunk(fx, rl, cs[i], STAGE / name, f"p6/{name}", a.batch, a.workers)
            rl.log.info("chunk %d: %d clips, %.1f ms/frame", i, len(cs[i]), st["ms_per_frame"])
            rl.event("chunk_done", chunk=i, n=len(cs[i]), ms_per_frame=st["ms_per_frame"])
        finally:
            lock.unlink(missing_ok=True)
    rl.close()


def cmd_verify(a):
    from jevdrive import waymo_qwenvid as qv
    from jevdrive.runlog import RunLog
    c = chunks()[0].iloc[:a.n]
    dst = STAGE / f"verify_{os.getpid()}"
    rl = RunLog("sch_gpu", "nq3-feats-qwen-verify")
    st = extract_chunk(qv.make_fx(compile=False), rl, c, dst, "verify", a.batch, a.workers)
    ref_names = pd.read_parquet(REAL / "c000" / "index.parquet").frame_name.iloc[:a.n].tolist()
    assert ref_names == c.frame_name.tolist(), "row order differs from the stored c000"
    res = {k: bool(np.array_equal(np.load(dst / f"{k}.npy"), np.load(REAL / "c000" / f"{k}.npy", mmap_mode="r")[:a.n]))
           for k in TAPS}
    diff = {k: float(np.abs(np.load(dst / f"{k}.npy").astype(np.float32)
                            - np.load(REAL / "c000" / f"{k}.npy", mmap_mode="r")[:a.n].astype(np.float32)).max()) for k in TAPS}
    rl.log.info("verify %d clips: bit-identical %s, max abs diff %s, %.1f ms/frame", a.n, res, diff, st["ms_per_frame"])
    rl.close()
    shutil.rmtree(dst)
    print({"identical": res, "max_abs_diff": diff, "ms_per_frame": st["ms_per_frame"]})
    sys.exit(0 if all(res.values()) else 1)


def cmd_install(a):
    try:
        os.kill(a.owner_pid, 0)
        sys.exit(f"lane C's qwen process {a.owner_pid} is alive: not installing")
    except ProcessLookupError:
        pass
    for d in sorted(STAGE.glob("c[0-9][0-9][0-9]")):
        tgt = REAL / d.name
        if not (d / "meta.json").exists() or (tgt / "meta.json").exists():
            continue
        if tgt.exists():                           # a partial chunk of the stopped process (no meta.json)
            shutil.rmtree(tgt)
        d.rename(tgt)
        print(f"installed {d} -> {tgt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "verify", "install"))
    ap.add_argument("--chunks", default="")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--owner-pid", type=int, default=801105)
    a = ap.parse_args()
    {"run": cmd_run, "verify": cmd_verify, "install": cmd_install}[a.cmd](a)


if __name__ == "__main__":
    main()
