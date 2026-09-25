"""Qwen `L18_*` features for a P5 frame index, sharded across processes and cards (reactivity program, P5 v1:
todos/2026-09-25-reactivity-program.md, extraction-profiling subsection).

The recipe is `p5_pairs.extract` unchanged: P3(d'')'s extractor (`waymo_qwenvid.make_fx`, eager), the same clip
items (`p4_carla.ClipFiles`), the same chunk layout under processed/<P5_SET>/features/c<NNN>/, and the same batch
size (2), so a row is bit-identical to what the one-process path writes (checked by `profile`). What changes is how
it runs and what it does not recompute:

  plan      the chunk list (frames not already extracted), written once to features/plan.parquet. Frames reused:
            P4's 3000 clips (as before, by frame name) and any frame whose 12 JPEGs are the *same files* as a frame
            of an earlier set (`--reuse-from`, default carla_p5): P5 v1's BehaviorAgent worlds on v0 routes are
            symlinks to v0's runs, so their rows are copied from v0's chunks into features/r000/ (float16, exact)
  work      one extraction process: claims chunks with an O_EXCL lock file (a dead owner's lock is taken over),
            writes meta.json last as the done-marker; start several per card and on several cards
  check     every planned chunk done and every index row has a feature row -> exit 0
  profile   throughput of the old path and of candidate configurations on v0 frames, and max |diff| of every
            configuration against v0's stored rows and against the old path re-run on the same card
"""
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import p5_pairs as P
from .common import data_dir, get_logger

log = get_logger(__name__)
ARRAYS = ("L18_last", "L18_mean", "vis_mean", "vit_mean")


def _p4_names() -> set:
    from . import p4_carla as p4
    return set(pd.read_parquet(p4.out_dir("features", p4.FEATURE_SET) / "index.parquet").frame_name)


def _file_keys(files: pd.Series) -> pd.Series:
    """Clip identity: the 12 JPEGs with every directory resolved (symlinked attempts collapse onto their target)."""
    real = {}

    def key(fs):
        out = []
        for f in fs:
            d, b = f.rsplit("/", 1)
            if d not in real:
                real[d] = os.path.realpath(d)
            out.append(real[d] + "/" + b)
        return "|".join(out)
    return files.map(key)


def plan(reuse_from=("carla_p5",), chunk: int = P.CHUNK) -> pd.DataFrame:
    """features/plan.parquet: frame_name, chunk (compute chunks) for every frame that needs a forward pass;
    reused rows go to features/r000 at once."""
    root = P.processed("features")
    t = pd.read_parquet(P.processed() / "index.parquet")
    todo = t[~t.frame_name.isin(_p4_names())].reset_index(drop=True)
    cur = os.environ.get("P5_SET", "carla_p5")
    reuse = pd.DataFrame(columns=["frame_name", "src", "row"])
    srcs = [s for s in reuse_from if s and s != cur]
    if srcs:
        keys = _file_keys(todo.files)
        found = []
        for s in srcs:
            sd = data_dir() / "processed" / s
            si = pd.read_parquet(sd / "index.parquet", columns=["frame_name", "files"])
            skey = dict(zip(si.frame_name, _file_keys(si.files)))
            for c in sorted((sd / "features").glob("c*")):
                if not (c / "meta.json").exists():
                    continue
                names = pd.read_parquet(c / "index.parquet").frame_name
                found.append(pd.DataFrame({"key": names.map(skey), "src": str(c), "row": np.arange(len(names))}))
        if found:
            f = pd.concat(found).dropna().drop_duplicates("key")
            reuse = todo.assign(key=keys)[["frame_name", "key"]].merge(f, on="key")[["frame_name", "src", "row"]]
        _write_reuse(root / "r000", reuse)
    comp = todo[~todo.frame_name.isin(set(reuse.frame_name))].reset_index(drop=True)
    pl = pd.DataFrame({"frame_name": comp.frame_name, "chunk": np.arange(len(comp)) // chunk})
    pl.to_parquet(root / "plan.parquet", index=False)
    log.info("%s: %d frames indexed, %d reused from P4, %d reused from %s, %d to extract in %d chunks of %d", cur,
             len(t), len(t) - len(todo), len(reuse), ",".join(srcs) or "-", len(comp), pl.chunk.nunique(), chunk)
    return pl


def _write_reuse(dst: Path, reuse: pd.DataFrame):
    if not len(reuse):
        return
    reuse = reuse.reset_index(drop=True)
    if (dst / "meta.json").exists() and \
            pd.read_parquet(dst / "index.parquet").frame_name.tolist() == reuse.frame_name.tolist():
        return
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta.json").unlink(missing_ok=True)
    for a in ARRAYS:
        parts = [np.load(Path(s) / f"{a}.npy", mmap_mode="r")[g.row.to_numpy()] for s, g in reuse.groupby("src", sort=False)]
        order = np.concatenate([g.index.to_numpy() for _, g in reuse.groupby("src", sort=False)])
        arr = np.empty((len(reuse), parts[0].shape[1]), np.float16)
        arr[order] = np.concatenate(parts)
        np.save(dst / f"{a}.npy", arr)
    reuse[["frame_name"]].to_parquet(dst / "index.parquet", index=False)
    (dst / "meta.json").write_text(json.dumps({"recipe": "P3(d'') qwenvid", "reused_rows": len(reuse),
                                               "from": sorted(reuse.src.unique().tolist())}))
    log.info("reused %d rows -> %s", len(reuse), dst)


def _alive(pid: int, host: str) -> bool:
    if host != socket.gethostname():
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _claim(lock: Path) -> bool:
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                o = json.loads(lock.read_text())
            except (OSError, ValueError):
                return False              # being written right now
            if _alive(o["pid"], o["host"]):
                return False
            lock.unlink(missing_ok=True)  # dead owner: take it over
    return False


def _done(dst: Path, names) -> bool:
    return (dst / "meta.json").exists() and \
        set(pd.read_parquet(dst / "index.parquet").frame_name) == set(names)


def work(rl, batch: int = 2, workers: int = 4):
    """Claim and extract chunks until none is left; resumable at chunk granularity."""
    from . import features as F, p4_carla as p4, waymo_qwenvid as qv
    root = P.processed("features")
    t = pd.read_parquet(P.processed() / "index.parquet", columns=["frame_name", "files"]).set_index("frame_name")
    pl = pd.read_parquet(root / "plan.parquet")
    fx, n_done = None, 0
    for i, g in pl.groupby("chunk"):
        dst = root / f"c{i:03d}"
        if _done(dst, g.frame_name):
            continue
        lock = root / f"c{i:03d}.lock"
        if not _claim(lock):
            continue
        try:
            if _done(dst, g.frame_name):
                continue
            if fx is None:
                fx = qv.make_fx(compile=False)
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "meta.json").unlink(missing_ok=True)
            st = F.extract(fx, t.files.loc[g.frame_name].map(list).tolist(), batch, workers, dst, rl, f"p5/c{i:03d}",
                           dataset=p4.ClipFiles)
            g[["frame_name"]].to_parquet(dst / "index.parquet", index=False)
            (dst / "meta.json").write_text(pd.Series({"recipe": "P3(d'') qwenvid", "batch_size": batch,
                                                      "host_pid": os.getpid(), **st}).to_json())
            n_done += 1
            rl.log.info("chunk %d: %d clips, %.1f ms/frame", i, len(g), st["ms_per_frame"])
            rl.event("chunk_done", chunk=int(i), n=len(g), ms_per_frame=st["ms_per_frame"])
        finally:
            lock.unlink(missing_ok=True)
    rl.log.info("no chunk left to claim (%d done by this process)", n_done)


def check() -> bool:
    root = P.processed("features")
    pl = pd.read_parquet(root / "plan.parquet")
    left = [i for i, g in pl.groupby("chunk") if not _done(root / f"c{i:03d}", g.frame_name)]
    if left:
        log.info("%d chunks not done: %s", len(left), left[:20])
        return False
    t = pd.read_parquet(P.processed() / "index.parquet")
    P.load_features(t.iloc[::97], ("L18_last",))        # asserts every sampled row has features
    have = set(_p4_names())
    for d in sorted(root.glob("[cr]*")):
        if (d / "index.parquet").exists():
            have |= set(pd.read_parquet(d / "index.parquet").frame_name)
    miss = int((~t.frame_name.isin(have)).sum())
    log.info("%d frames, %d without features", len(t), miss)
    return miss == 0


# ---------------------------------------------------------------- profiling and equivalence

def profile(rl, n: int = 240, configs=("eager-b2-w4", "eager-b2-w4", "eager-b4-w6", "compile-b2-w4",
                                         "compile-b4-w6"), chunk: str = "c000"):
    """ms/frame per configuration on the first n frames of one v0 chunk, max |diff| of every array against the
    stored v0 rows and against the first run of the old path (eager, batch 2, 4 workers) on this card."""
    import torch
    from . import features as F, p4_carla as p4, waymo_qwenvid as qv
    os.environ["P5_SET"] = "carla_p5"
    src = P.processed("features", chunk)
    names = pd.read_parquet(src / "index.parquet").frame_name[:n]
    t = pd.read_parquet(P.processed() / "index.parquet", columns=["frame_name", "files"]).set_index("frame_name")
    items = t.files.loc[names].map(list).tolist()
    ref = {a: np.load(src / f"{a}.npy", mmap_mode="r")[:n].astype(np.float32) for a in ARRAYS}
    scratch = data_dir() / "scratch" / "p5_qwen_profile" / rl.dir.name
    fxs, rows, first = {}, [], None
    ds = p4.ClipFiles(items[:9], None)
    for k, cfg in enumerate(configs):
        mode, b, w = cfg.split("-")
        b, w = int(b[1:]), int(w[1:])
        if mode not in fxs:
            fxs.clear()
            F.free_gpu()
            fxs[mode] = qv.make_fx(compile=mode == "compile")
            if k == 0:                     # CPU cost of one item (read + decode + processor), one core
                ds.transform = fxs[mode].transform
                ds[0]
                t0 = time.perf_counter()
                for i in range(1, 9):
                    ds[i]
                rl.event("cpu_item_ms", ms=1e3 * (time.perf_counter() - t0) / 8)
                rl.log.info("CPU: %.0f ms per clip item (12 JPEGs, decode + processor)", 1e3 * (time.perf_counter() - t0) / 8)
        dst = scratch / f"{k}-{cfg}"
        dst.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        st = F.extract(fxs[mode], items, b, w, dst, rl, f"profile/{k}-{cfg}", dataset=p4.ClipFiles)
        out = {a: np.load(dst / f"{a}.npy").astype(np.float32) for a in ARRAYS}
        if first is None:
            first = out
        r = {"config": cfg, "run": k, "ms_per_frame": st["ms_per_frame"], "wall_s": st["wall_s"],
             "peak_vram_gb": st["peak_vram_reserved_gb"]}
        for a in ARRAYS:
            for tag, y in (("v0", ref[a]), ("old", first[a])):
                d = out[a] - y
                r[f"{a}_maxabs_vs_{tag}"] = float(np.abs(d).max())
                r[f"{a}_rel_vs_{tag}"] = float(np.linalg.norm(d) / np.linalg.norm(y))
                r[f"{a}_identical_vs_{tag}"] = int((d == 0).all(1).sum())
        r["L18_last_scale"] = float(np.abs(ref["L18_last"]).max())
        rows.append(r)
        rl.event("profile", **r)
        rl.log.info("%-16s %.1f ms/frame, peak %.1f GB | L18_last vs v0: max|d| %.3g, identical %d/%d | vs old: %.3g",
                    cfg, r["ms_per_frame"], r["peak_vram_gb"], r["L18_last_maxabs_vs_v0"],
                    r["L18_last_identical_vs_v0"], n, r["L18_last_maxabs_vs_old"])
    tab = pd.DataFrame(rows)
    tab.to_csv(rl.dir / "profile.csv", index=False)
    rl.log.info("profile\n%s", tab.to_markdown(index=False, floatfmt=".4g"))


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "work", "check", "profile"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse-from", default="carla_p5")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--configs", default="eager-b2-w4,eager-b2-w4,eager-b4-w6,compile-b2-w4,compile-b4-w6")
    ap.add_argument("--chunk", default="c000")
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(tuple(a.reuse_from.split(",")))
    elif a.cmd == "check":
        raise SystemExit(0 if check() else 1)
    else:
        rl = RunLog("p5_qwen", a.cmd + "-" + os.environ.get("P5_SET", "carla_p5"))
        if a.cmd == "work":
            work(rl, a.batch, a.workers)
        else:
            profile(rl, a.n, tuple(a.configs.split(",")), a.chunk)
        rl.close()


if __name__ == "__main__":
    main()
