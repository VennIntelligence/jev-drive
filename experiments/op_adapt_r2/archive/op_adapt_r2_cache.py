"""op-adapt round 2, package C GPU passes (op-train venv): trunk caches and the original-Cinque teacher
(experiments/op_adapt_r2/archive/op_adapt_r2_data.py; formats in tmp/2026-09-30-op-adapt-r2-build.md, section C).

  sim     per Cosmos G4 pair: the 4 streams C+ C- K+ K- (E1's input path: 20 Hz clip ::4, cosmos_openpilot.model_frames,
          image pair (slot - 1, slot), zero image before slot 0) -> R2/cache-sim/<pair>.npy fp16 (96, 1024, 8, 16) and the
          teacher of slots 9..23 -> R2/teacher/parts-sim/<pair>.npz (mu, std: (4, 15, 3, 33, 15))
  off     offset-start samples of R2/offset/table.parquet, one shard of 250 per file: homography-warped frames ->
          R2/cache-off/<shard>.npy (+ <shard>.ctx.npy (n, 9)) and the teacher -> R2/teacher/parts-off/<shard>.npz
  teacher the teacher on the cached trunks of an index (nus: keyframes) -> R2/teacher/<domain>.npz
  merge   teacher/simC.npz, simK.npz, off.npz aligned to the index uids; index/off.parquet

  CUDA_VISIBLE_DEVICES=1 taskset -c 24-29 python experiments/op_adapt_r2/archive/op_adapt_r2_cache.py sim --workers 6 --shard 0 --nshard 4
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/cosmos/lib", "scripts",)]
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from jevdrive import op_adapt as A  # noqa: E402
from experiments.op_adapt_r2.archive import op_adapt_r2_data as C  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

H_, W_, T_ = 704, 1280, 93


def read_mp4_every(path: Path, step: int = C.STEP) -> np.ndarray:
    """cosmos_openpilot.read_mp4(path)[::step] without keeping the other frames (same decoder, same rgb24 output)."""
    import glob
    ff = sorted(glob.glob(str(data_dir() / "envs/*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-*")))[0]
    out = subprocess.run([ff, "-loglevel", "error", "-i", str(path), "-vf", f"select=not(mod(n\\,{step}))", "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, H_, W_, 3)


_IDX = None


def _init(idx):
    global _IDX
    _IDX = idx


def sim_frames(pair: str):
    """(4, 24, 2, 6, 128, 256) packed model frames of C+ C- K+ K- (K+ composited on the blend support as load_pair)."""
    import cosmos_openpilot as CO
    from experiments.cosmos.lib.cosmos_full import main_dir
    d = main_dir() / "pairs" / pair
    sup = np.load(d / "gt.npz")["support"]
    n = H_ * W_ // 8
    s = np.stack([np.unpackbits(sup[t * n:(t + 1) * n]).reshape(H_, W_, 1) for t in range(0, T_, C.STEP)]).astype(bool)
    cp, cm, kp, km = (read_mp4_every(d / f"{k}.mp4") for k in ("carla_plus", "carla_minus", "cosmos_plus", "cosmos_minus"))
    kp = np.where(s, kp, km)
    assert all(len(x) == C.NSLOT for x in (cp, cm, kp, km)), pair
    return pair, np.stack([CO.model_frames(x, _IDX) for x in (cp, cm, kp, km)])


@torch.no_grad()
def trunk(net, prev, cur, batch):
    out = []
    for i in range(0, len(cur), batch):
        o = net.run_batched(A.vision_feeds(prev[i:i + batch], cur[i:i + batch]), [A.TRUNK_OUT])[A.TRUNK_OUT]
        out.append(o[:, 0].to(torch.float16))
    return torch.cat(out)


@torch.no_grad()
def hidden(net, T, batch=64):
    """(n, 1024, 8, 16) trunk rows -> (n, 32, 512) stage-4 hidden states (the round-1 stage4_policy path)."""
    return torch.cat([net.run_batched({A.TRUNK_OUT: T[i:i + batch, None].to(net.dtype)}, ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE)
                      for i in range(0, len(T), batch)])


@torch.no_grad()
def teach(net, Hrows, ctx, tc, batch=128):
    """Hrows (m, 32, 512), ctx (n, 9) rows into Hrows (-1 = zero), tc (n, 2) -> mu, std (n, 3, 33, 15) numpy."""
    mu, sd = [], []
    for i in range(0, len(ctx), batch):
        c = torch.as_tensor(ctx[i:i + batch], device=Hrows.device, dtype=torch.long)
        v = c >= 0
        m, s = C.teacher_plans(net, Hrows[c.clamp(min=0)], v, torch.as_tensor(tc[i:i + batch], device=Hrows.device))
        mu.append(m.cpu())
        sd.append(s.cpu())
    return torch.cat(mu).numpy(), torch.cat(sd).numpy()


def bounded(ex, fn, items, depth):
    from drive_backbones_openpilot import bounded_map
    return bounded_map(ex, fn, items, depth)


def run_sim(a, log, net):
    import cosmos_openpilot as CO
    pairs = a.pairs.split(",") if a.pairs else C.sim_pairs().pair.tolist()
    pairs = pairs[a.shard::a.nshard][: a.limit or None]
    out, tout = C.root("cache-sim"), C.root("teacher", "parts-sim")
    todo = [p for p in pairs if not ((out / f"{p}.npy").exists() and (tout / f"{p}.npz").exists())]
    log.info(f"sim shard {a.shard}/{a.nshard}: {len(pairs)} pairs, {len(todo)} to do")
    j = np.arange(C.MIN_SLOT, C.NSLOT)
    tc = np.tile([[1.0, 0.0]], (4 * len(j), 1)).astype(np.float32)
    ctx = np.concatenate([s * C.NSLOT + C.stride_ctx(j, 1) for s in range(4)])
    t0, n = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(CO.maps(),)) as ex:
        list(ex.map(int, range(a.workers)))
        net = net()
        for pair, fr in bounded(ex, sim_frames, todo, 2 * a.workers):
            cur = torch.as_tensor(fr.reshape(-1, 2, 6, 128, 256)).cuda()
            prev = torch.as_tensor(np.concatenate([np.zeros_like(fr[:, :1]), fr[:, :-1]], 1).reshape(-1, 2, 6, 128, 256)).cuda()
            T = trunk(net, prev, cur, a.batch)
            mu, sd = teach(net, hidden(net, T), ctx, tc)
            np.save(out / f"{pair}.tmp.npy", T.cpu().numpy())
            (out / f"{pair}.tmp.npy").replace(out / f"{pair}.npy")
            np.savez(tout / f"{pair}.tmp.npz", mu=mu.reshape(4, len(j), 3, 33, 15), std=sd.reshape(4, len(j), 3, 33, 15))
            (tout / f"{pair}.tmp.npz").replace(tout / f"{pair}.npz")
            n += 1
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                log.info(f"[{n}/{len(todo)}] {el / 60:.1f} min, {el / n:.2f} s/pair, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
                         f"ETA {(len(todo) - n) * el / n / 60:.0f} min")
                log.event("progress", n=n, wall_s=el, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    log.event("end", pairs=n, wall_s=time.time() - t0)


_ENT = {}


def _off_init():
    from jevdrive import navsim_zs as Z
    t = pd.read_parquet(C.root("offset") / "table.parquet")
    need = set(t.token)
    _ENT.update({e["token"]: e for e in Z.load_index("navtrain", slim=True) if e["token"] in need})


def off_job(rows):
    shard, recs = rows
    prev, cur, ctx, cov = [], [], [], []
    base = 0
    for tok, e, psi in recs:
        p, c, x, cv = C.offset_frames(_ENT[tok], e, psi)
        prev.append(p)
        cur.append(c)
        ctx.append(np.where(x >= 0, x + base, -1))
        cov.append(cv)
        base += len(c)
    return shard, np.concatenate(prev), np.concatenate(cur), np.stack(ctx).astype(np.int32), np.array(cov)


def run_off(a, log, net):
    t = pd.read_parquet(C.root("offset") / "table.parquet")
    if a.identity:              # numerical check: e = psi = 0 must reproduce the navtrain cache
        t = t[t.split == "train"].head(a.limit or 40).assign(e=0.0, psi=0.0, shard=-1)
    shards = sorted(t.shard.unique())[a.shard::a.nshard]
    if a.limit and not a.identity:
        shards = shards[: a.limit]
    out, tout = C.root("cache-off" + ("-identity" if a.identity else "")), C.root("teacher", "parts-off" + ("-identity" if a.identity else ""))
    todo = [s for s in shards if not (tout / f"{s}.npz").exists()]
    jobs = [(s, list(zip(g.token, g.e, g.psi))) for s, g in t[t.shard.isin(todo)].groupby("shard")]
    log.info(f"off shard {a.shard}/{a.nshard}: {len(jobs)} shards to do")
    t0, n = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=_off_init) as ex:
        list(ex.map(int, range(a.workers)))
        net = net()
        for s, prev, cur, ctx, cov in bounded(ex, off_job, jobs, 2 * a.workers):
            T = trunk(net, torch.as_tensor(prev).cuda(), torch.as_tensor(cur).cuda(), a.batch)
            g = t[t.shard == s]
            mu, sd = teach(net, hidden(net, T), ctx, g[["tc0", "tc1"]].to_numpy(np.float32))
            np.save(out / f"{s}.tmp.npy", T.cpu().numpy())
            (out / f"{s}.tmp.npy").replace(out / f"{s}.npy")
            np.save(out / f"{s}.ctx.npy", ctx)
            np.savez(tout / f"{s}.tmp.npz", sid=g.sid.to_numpy(), mu=mu, std=sd, coverage=cov)
            (tout / f"{s}.tmp.npz").replace(tout / f"{s}.npz")
            n += 1
            el = time.time() - t0
            log.info(f"[{n}/{len(jobs)}] shard {s}: {len(g)} samples, {len(T)} pairs, {el / 60:.1f} min, "
                     f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, ETA {(len(jobs) - n) * el / n / 60:.0f} min")
            log.event("progress", n=n, wall_s=el)
    log.event("end", shards=n, wall_s=time.time() - t0)


def run_teacher(a, log, net):
    idx = C.load_index(a.domain)
    if a.domain == "nus":
        idx = idx[idx.labeled]
    net = net()
    t0 = time.time()
    uids, mus, sds = [], [], []
    for f, g in idx.groupby("file", sort=False):
        z = np.load(data_dir() / f) if f.endswith(".npz") else None
        T = torch.as_tensor(z["trunk"] if z is not None else np.load(data_dir() / f)).cuda()
        c = np.stack(g.ctx.to_numpy())
        need = np.unique(c[c >= 0])
        pos = np.full(len(T), -1)
        pos[need] = np.arange(len(need))
        mu, sd = teach(net, hidden(net, T[torch.as_tensor(need).cuda()]), np.where(c >= 0, pos[c.clip(0)], -1),
                       g[["tc0", "tc1"]].to_numpy(np.float32))
        uids.append(g.uid.to_numpy())
        mus.append(mu)
        sds.append(sd)
    np.savez(C.root("teacher") / f"{a.domain}.npz", uid=np.concatenate(uids), plan_mu=np.concatenate(mus), plan_std=np.concatenate(sds),
             desires=np.array(["op", "op_L", "op_R"]))
    log.event("end", domain=a.domain, n=int(sum(map(len, uids))), wall_s=time.time() - t0)
    log.info(f"teacher {a.domain}: {sum(map(len, uids))} rows in {time.time() - t0:.0f} s")


def run_merge(a, log):
    for dom, s0 in (("simC", 0), ("simK", 2)):
        idx = C.load_index(dom)
        uid, mu, sd = [], [], []
        for p, g in idx.groupby("key", sort=True):
            z = np.load(C.root("teacher", "parts-sim") / f"{p}.npz")
            st = np.where(g.sign.to_numpy() > 0, s0, s0 + 1)
            k = g.slot.to_numpy() - C.MIN_SLOT
            uid.append(g.uid.to_numpy())
            mu.append(z["mu"][st, k])
            sd.append(z["std"][st, k])
        np.savez(C.root("teacher") / f"{dom}.npz", uid=np.concatenate(uid), plan_mu=np.concatenate(mu), plan_std=np.concatenate(sd),
                 desires=np.array(["op", "op_L", "op_R"]))
        log.info(f"teacher {dom}: {sum(map(len, uid))} rows")
    t = pd.read_parquet(C.root("offset") / "table.parquet")
    parts = sorted(C.root("teacher", "parts-off").glob("*.npz"))
    if parts and len(parts) == t.shard.nunique():
        rows, mu, sd = [], [], []
        for f in parts:
            s = int(f.stem)
            z = np.load(f)
            ctx = np.load(C.root("cache-off") / f"{s}.ctx.npy")
            g = t[t.shard == s]
            assert (z["sid"] == g.sid.to_numpy()).all()
            rows.append(g.assign(file=C.rel(C.root("cache-off") / f"{s}.npy"), ctx=list(ctx), row=ctx[:, -1], slot=np.arange(len(g)),
                                 coverage_road=z["coverage"][:, 0], coverage_wide=z["coverage"][:, 1]))
            mu.append(z["mu"])
            sd.append(z["std"])
        d = pd.concat(rows).sort_values("sid").drop(columns="uid")
        d["key"] = d.sid.astype(str)
        d = C._finish(d, "off")
        assert (d.uid.to_numpy() == C.UID_BASE["off"] + d.sid.to_numpy()).all()
        d.to_parquet(C.root("index") / "off.parquet", index=False)
        o = np.argsort(np.concatenate([np.load(f)["sid"] for f in parts]))
        np.savez(C.root("teacher") / "off.npz", uid=d.uid.to_numpy(), plan_mu=np.concatenate(mu)[o], plan_std=np.concatenate(sd)[o],
                 desires=np.array(["op", "op_L", "op_R"]))
        log.info(f"index/teacher off: {len(d)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("sim", "off", "teacher", "merge"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64, help="image pairs per trunk forward")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domain", default="nus")
    ap.add_argument("--pairs", default="", help="sim: comma-separated pair names (pilot)")
    ap.add_argument("--identity", action="store_true", help="off: e = psi = 0 on the first train samples (equivalence check)")
    a = ap.parse_args()
    log = RunLog("op_adapt_r2", "logs", f"C-{a.what}" + (f"-{a.domain}" if a.what == "teacher" else "") + (f"-s{a.shard}" if a.nshard > 1 else ""))
    log.event("start", args=vars(a))
    net = lambda: A.load("cinque", torch.float16).cuda()  # noqa: E731  (after the workers fork)
    {"sim": run_sim, "off": run_off, "teacher": run_teacher, "merge": lambda a_, l_, n_: run_merge(a_, l_)}[a.what](a, log, net)


if __name__ == "__main__":
    main()
