"""Equality check of the token-cache builders' GPU warp path (ap2_prep.py, pp_prep.py --frames warp): --synth gpu against the CPU
reference --synth cpu on the first N tokens of one data dir, written under a scratch cache root (never the real cache).

  run     --which ap2|pp --synth cpu|gpu --data D --limit N --out DIR [--bs 128] [--workers 3] [--ckpt-s S] [--plain]
          the builder's own main() with cache root DIR/<synth>; every encoder call is recorded (one 64-bit hash per frame of prev and
          cur, in the order fed) -> DIR/<which>-<synth>.frames.npz; --plain: no recording (throughput, VRAM and RSS as the builder
          reports them in meta.json / timing.json). pp: the tab is the first N rows of the real one (the navsim_zs index is gone)
  cmp     --which ap2|pp --data D --limit N --out DIR [--ref]
          cpu against gpu: the encoder calls (count, sizes, frame hashes) and the written arrays; --ref: the gpu arrays against the same
          rows of the real cache (same batches when N is a multiple of 32 and the real cache was built at the same --bs)
  full    --data navtrain_full.s2of12 --n 224 --out DIR
          pp_prep's full-navtrain job, which cannot run end to end without the navtrain index: _full_job (keys + CPU warps) against
          _key_job + warp_keys, bytes of the 4 keys and the 6 lattice frames
  resume  --which ap2|pp --data D --limit N --out DIR [...]
          run gpu with a 2 s checkpoint into DIR/resume, kill it after the first checkpoint, run it again, compare with DIR/gpu

  $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/prep_check.py run --which ap2 --synth gpu --data navtrain_full.s2of12 --limit 256 --out $DATA_DIR/runs/alpasim/lat1/prep_dev/eq
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib"), str(_R / "experiments/alpasim/scripts")]
import argparse, hashlib, json, os, signal, subprocess, time  # noqa: E401,E402

import numpy as np  # noqa: E402

import pp_prep as PP  # noqa: E402  (before sh30_core, which puts the repo's op_parity scripts first on sys.path)
import ap2_prep as AP  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

FILES = {"ap2": ("cold.npy", "bw.npy"), "pp": ("front.npy",)}


def tag(a):
    return a.data + (f"-first{a.limit}" if a.limit else "")


def where(a, synth):
    return a.out / synth / (tag(a) if a.which == "ap2" else f"{tag(a)}@warp")


def hook(rec):
    """Record every encoder call of both paths: per-frame hashes of prev and cur, and the batch size."""
    h = lambda x: np.array([int.from_bytes(hashlib.blake2b(np.ascontiguousarray(f).tobytes(), digest_size=8).digest(), "little") for f in x], np.uint64)  # noqa: E731
    enc_dev0, encoder0 = PP.enc_dev, PP.encoder

    def enc_dev(net, prev, cur, bs=128):
        rec.append((h(prev.cpu().numpy()), h(cur.cpu().numpy()), bs))
        return enc_dev0(net, prev, cur, bs)

    def encoder(dev):
        net, enc0 = encoder0(dev)

        def enc(prev, cur, bs=128):
            rec.append((h(prev), h(cur), bs))
            return enc0(prev, cur, bs)
        return net, enc
    PP.enc_dev, PP.encoder = enc_dev, encoder


def cmd_run(a):
    rec = []
    if not a.plain:
        hook(rec)
    if a.ckpt_s:
        mark0 = PP.Part.mark
        PP.Part.mark = lambda self, rows, every=60.0: mark0(self, rows, a.ckpt_s)
    ns = dict(data=a.data, limit=a.limit, workers=a.workers, synth=a.synth, out=a.out / a.synth, seed=0, resume=None, force=a.force)
    if a.which == "ap2":
        AP.OUT = ns["out"]
        AP.main(argparse.Namespace(bw=True, bs=a.bs, **ns))
    else:
        PP.OUT, cached0 = ns["out"], PP.cache.cached
        tab = np.load(data_dir() / "runs/op_parity/cache" / a.data / "tab.npz")
        PP.cache.cached = lambda path, k, fn, **kw: {q: tab[q][: a.limit] for q in tab.files} if path.name == "tab.npz" else cached0(path, k, fn, **kw)
        PP.main(argparse.Namespace(frames="warp", no_side=True, **ns))
    if a.plain:
        return
    np.savez(a.out / f"{a.which}-{a.synth}.frames.npz", prev=np.concatenate([r[0] for r in rec]), cur=np.concatenate([r[1] for r in rec]),
             n=np.array([len(r[0]) for r in rec]), bs=np.array([r[2] for r in rec]))


def arrays(x, y):
    x, y = np.asarray(x), np.asarray(y)
    d = np.abs(x.astype(np.float32) - y.astype(np.float32))
    return dict(shape=list(x.shape), identical=bool(np.array_equal(x, y)), differing_values=int((x != y).sum()), max_abs=float(d.max()),
                rows_differing=int((x != y).reshape(len(x), -1).any(1).sum()))


def cmd_cmp(a):
    c, g = (np.load(a.out / f"{a.which}-{s}.frames.npz") for s in PP.SYNTH[::-1])
    out = dict(which=a.which, data=a.data, tokens=a.limit, encoder_calls=[len(c["n"]), len(g["n"])],
               same_calls=bool(np.array_equal(c["n"], g["n"]) and np.array_equal(c["bs"], g["bs"])), bs=sorted(set(c["bs"].tolist())),
               batches=int(sum(-(-n // b) for n, b in zip(c["n"], c["bs"]))), pairs=int(c["n"].sum()))
    if out["same_calls"]:
        out |= dict(frames=2 * len(c["cur"]), frames_differing=int((c["prev"] != g["prev"]).sum() + (c["cur"] != g["cur"]).sum()),
                    distinct_cur_frames=len(np.unique(c["cur"])), zero_prev_frames=int((c["prev"] == c["prev"][0]).sum()))
    for f in FILES[a.which]:
        x, y = (np.load(where(a, s) / f, mmap_mode="r") for s in PP.SYNTH[::-1])
        out[f"{f}: cpu vs gpu"] = arrays(x, y)
        if a.ref:
            ref = data_dir() / ("runs/alpasim/ap2/cache" if a.which == "ap2" else "runs/op_parity/cache") / where(a, "x").name.replace(tag(a), a.data) / f
            out[f"{f}: gpu vs real cache rows"] = arrays(np.load(ref, mmap_mode="r")[: len(y)], y)
    (a.out / f"{a.which}-cmp.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def cmd_full(a):
    import torch
    import op_lb as OL
    tab = np.load(data_dir() / "runs/op_parity/cache" / a.data / "tab.npz")
    sel = np.sort(np.random.default_rng(0).choice(len(tab["names"]), a.n, replace=False))
    ents = AP.entries(tab["names"][sel].tolist(), tab["log"][sel].tolist(), None)
    pose, vel, cam, T = tab["pose"][sel].astype(float), tab["vel"][sel].astype(float), tab["cam"][sel].astype(float), np.asarray(OL.SYN_T)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as pool:
        t0 = time.time()
        kf0, sf0 = (np.stack(x) for x in zip(*pool.map(PP._full_job, [(ents[i], pose[i], vel[i], cam[i], T) for i in range(a.n)])))
        t1 = time.time()
        kf1, sk, pp = (np.stack(x) for x in zip(*pool.map(PP._key_job, [(ents[i], pose[i], vel[i], T) for i in range(a.n)])))
        t2 = time.time()
    dev, sf1 = torch.device("cuda"), []
    for i in range(0, a.n, 32):
        K, b = torch.from_numpy(kf1[i:i + 32]).to(dev), len(kf1[i:i + 32])
        sf1.append(PP.warp_keys(K, np.repeat(np.arange(b), sk.shape[1]), sk[i:i + 32].reshape(-1), cam[i:i + 32], pp[i:i + 32].reshape(-1, 2, 3)).view(b, -1, *PP.FRAME).cpu().numpy())
    sf1 = np.concatenate(sf1)
    out = dict(data=a.data, tokens=a.n, logs=len(set(tab["log"][sel].tolist())), camera_positions=len(np.unique(cam, axis=0)),
               key_frames=int(kf0.shape[0] * 4), key_pixels_differing=int((kf0 != kf1).sum()), lattice_frames=int(sf0.shape[0] * sf0.shape[1]),
               lattice_pixels=int(sf0.size), lattice_pixels_differing=int((sf0 != sf1).sum()), workers=a.workers,
               cpu_job_tokens_per_s=a.n / (t1 - t0), key_job_tokens_per_s=a.n / (t2 - t1))
    (a.out / "pp-full.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def cmd_resume(a):
    me = [_sys.executable, __file__, "run", "--which", a.which, "--synth", "gpu", "--data", a.data, "--limit", str(a.limit), "--out", str(a.out / "resume"),
          "--bs", str(a.bs), "--workers", str(a.workers), "--ckpt-s", "2"]
    a.out = a.out / "resume"
    mask = where(a, "gpu") / "done.part.npy"
    p = subprocess.Popen(me + ["--force"], start_new_session=True)
    while p.poll() is None and not (mask.exists() and np.load(mask).sum() >= 64):
        time.sleep(0.2)
    assert p.poll() is None, "the build finished before the kill: raise --limit"
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    done = int(np.load(mask).sum())
    subprocess.run(me, check=True)
    fr = np.load(a.out / f"{a.which}-gpu.frames.npz")
    out = dict(which=a.which, tokens=a.limit, rows_done_at_kill=done, pairs_encoded_after_resume=int(fr["n"].sum()))
    for f in FILES[a.which]:
        out[f"{f}: resumed vs uninterrupted"] = arrays(np.load(a.out.parent / "gpu" / where(a, "gpu").name / f, mmap_mode="r"), np.load(where(a, "gpu") / f, mmap_mode="r"))
    (a.out / f"{a.which}-resume.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "cmp", "full", "resume"])
    ap.add_argument("--which", default="ap2", choices=list(FILES))
    ap.add_argument("--synth", default="gpu", choices=PP.SYNTH)
    ap.add_argument("--data", default="navtrain_full.s2of12")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--out", type=_pl.Path, required=True)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ckpt-s", type=float, default=0.0)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "cmp": cmd_cmp, "full": cmd_full, "resume": cmd_resume}[a.cmd](a)
