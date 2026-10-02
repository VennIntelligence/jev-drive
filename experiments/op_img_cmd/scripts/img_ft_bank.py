"""Image-command fine-tune (Q2): stage-3 trunk bank of every overlay variant, train pool and eval pool (op-train venv, one GPU).
Plan: ../plans/2026-10-04-img-cmd-ft-prereg.md.

Stage 1-3 of Cinque is frozen in the fine-tune (op_adapt_l's port, L3's recipe), so every (sample, overlay family, command)
variant has fixed stage-3 outputs: they are computed once here and both the trainer and the eval read them. A sample is the
port's nav context (op_adapt_h's h_prep.NavSrc: 10 packed frames on the 5 Hz lattice from op_lb's 4 keys + 6 GIMM frames,
pairs (k, k+1) = the 9 policy context slots, slot 0 invalid); the overlay is drawn on every valid frame with the ego pose at
that frame's source time (keys: logged poses; GIMM frames: op_interp's EgoTrack), as img_run.render does on the 31-step schedule.

  pool   samples (geom pkl)   op_lb data     variants per sample
  train  geom/ft.pkl          lb_imgtrain    junction: none, TRAIN_FAMS x every exit class; straight: none, band (own lane)
  eval   geom/nav.pkl valid   lb_navtrain    exactly img_run.variants: none, every family x class, band_all; straight: none + 4
Output $DATA_DIR/runs/op_img_cmd/ft/bank/<pool>/{trunk.npy (n, 9, 1024, 8, 16) fp16, var.npz}; resumable per chunk.

  CUDA_VISIBLE_DEVICES=1 taskset -c 100-149 $DATA_DIR/envs/op-train/bin/python experiments/op_img_cmd/scripts/img_ft_bank.py train eval
"""
import json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments" / "op_adapt_h" / "scripts"), str(Path(__file__).resolve().parent)]
import img_overlay as O  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

FT = data_dir() / "runs" / "op_img_cmd" / "ft"
TRAIN_FAMS = ("band", "barrier")
CMDS = ("left", "straight", "right")
POOLS = {"train": ("ft.pkl", "lb_imgtrain"), "eval": ("nav.pkl", "lb_navtrain")}


def classes(s):
    return [c for c in CMDS if any(b["cls"] == c for b in s["branches"])]


def variants(s, pool):
    if pool == "eval":
        import img_run
        return img_run.variants(s)
    if s["kind"] == "straight":
        return [("none", ""), ("band", "straight")]
    return [("none", "")] + [(f, c) for f in TRAIN_FAMS for c in classes(s)]


def samples(pool):
    G = pickle.load(open(data_dir() / "runs" / "op_img_cmd" / "geom" / POOLS[pool][0], "rb"))
    G = [s for s in G if O.valid(s)]
    assert all(s["row"] >= 0 for s in G), "op_lb rows missing (img_ft_data.py prep)"
    return G


_W = {}


def _init(data):
    import h_prep
    from jevdrive import op_interp as I
    _W["src"], _W["I"] = h_prep.NavSrc(data), I


def render(job):
    """(sample, pool) -> [(fam, cmd)], imgs (V, 10, 2, 6, 128, 256) uint8 with the overlay on every valid lattice frame."""
    s, pool = job
    src, I = _W["src"], _W["I"]
    mt, r = src.mt, s["row"]
    base = src(r)
    cam = np.asarray(mt["cam"][r], float)
    tr = I.track_navsim(mt["pose"][r], mt["vel"][r])
    poses = [None if q is None else np.asarray(mt["pose"][r][q[1]], float) if q[0] == "k" else tr(t) for q, t in zip(src.src, src.t)]
    vs = variants(s, pool)
    out = np.empty((len(vs),) + base.shape, np.uint8)
    for v, (fam, c) in enumerate(vs):
        lay = O.primitives(s, fam, c or None)
        out[v] = [base[j] if p is None else O.draw(base[j], lay, p, cam) for j, p in enumerate(poses)]
    return vs, out


def build(pool, net, dev, workers, batch=32):
    import torch
    from drive_backbones_openpilot import bounded_map
    from experiments.op_adapt_h.lib import op_adapt_h as H
    import h_prep
    lim = int(os.environ.get("IMG_FT_LIMIT", 0))                # smoke: the first `lim` samples into bank/<pool>-lim<lim>
    out = FT / "bank" / (pool + (f"-lim{lim}" if lim else ""))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "var.npz").exists():
        print(pool, "bank exists", flush=True)
        return
    G = samples(pool)[: lim or None]
    tab = [(i, f, c) for i, s in enumerate(G) for f, c in variants(s, pool)]
    n = len(tab)
    src = h_prep.NavSrc(POOLS[pool][1])
    sv = np.ones(9, bool)
    sv[0] = False                                                # h_prep's nav convention (slot 0 = a zero frame pair)
    tmp, prog = out / "trunk.tmp.npy", out / "progress.json"
    T = np.lib.format.open_memmap(tmp, "r+" if tmp.exists() else "w+", np.float16, (n, 9, 1024, 8, 16))
    first = np.searchsorted([t[0] for t in tab], np.arange(len(G) + 1))
    s_done = json.loads(prog.read_text())["samples"] if prog.exists() and tmp.exists() else 0
    print(f"{pool}: {len(G)} samples, {n} variants ({n * 9 * 1024 * 128 * 2 / 2 ** 30:.1f} GB), from sample {s_done}", flush=True)
    svt = torch.from_numpy(sv).to(dev)[None, :, None, None, None]
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(POOLS[pool][1],)) as ex:
        buf, at = [], first[s_done]
        def flush():
            nonlocal buf, at
            if not buf:
                return
            imgs = np.concatenate(buf)
            for k in range(0, len(imgs), batch):
                x = torch.from_numpy(imgs[k:k + batch]).to(dev)
                T[at + k:at + k + len(x)] = (H.trunks(net, x, chunk=64) * svt).cpu().numpy()
            at += len(imgs)
            buf = []
        for i, (vs, imgs) in enumerate(bounded_map(ex, render, [(s, pool) for s in G[s_done:]], 2 * workers), start=s_done):
            assert [(f, c) for _, f, c in tab[first[i]:first[i + 1]]] == vs
            buf.append(imgs)
            if sum(len(b) for b in buf) >= 4 * batch:
                flush()
            if (i + 1) % 100 == 0:
                flush()
                T.flush()
                prog.write_text(json.dumps({"samples": i + 1}))
                el = time.time() - t0
                print(f"{pool}: {i + 1}/{len(G)} samples, {(at - first[s_done]) / el:.1f} var/s", flush=True)
        flush()
    T.flush()
    del T
    tmp.replace(out / "trunk.npy")
    np.savez(out / "var.npz", sample=np.array([t[0] for t in tab]), fam=np.array([t[1] for t in tab]), cmd=np.array([t[2] for t in tab]),
             token=np.array([G[t[0]]["token"] for t in tab]), slot_valid=sv, cam=np.array([src.mt["cam"][G[t[0]]["row"]] for t in tab], np.float32),
             tc=np.array([[0.0, 1.0] if src.mt["lht"][G[t[0]]["row"]] else [1.0, 0.0] for t in tab], np.float32))
    prog.unlink(missing_ok=True)
    print(f"{pool}: {n} variants in {time.time() - t0:.0f} s", flush=True)


def main():
    import torch
    from jevdrive.run import Run
    from experiments.op_adapt_l.lib import op_adapt_l as L
    pools = [p for p in sys.argv[1:] if p in POOLS] or list(POOLS)
    workers = int(os.environ.get("IMG_FT_WORKERS", 40))
    with Run("op_img_cmd", "ft-bank", config={"pools": pools, "workers": workers}) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("navsim/navtrain"))
        dev = torch.device("cuda")
        net = L.load_model(None, dev).net
        for p in pools:
            build(p, net, dev, workers)
            run.summary[p] = "done"


if __name__ == "__main__":
    main()
