"""Batched level (a) sweep for WA-JEPA: T tokens x A arms in one planner forward.

Same arms, same per-call noise as p1_wajepa.py: the model re-seeds its flow generator on every predict_trajectory call and
draws (batch, ...) noise, so in the shipped batch-1 path every token and arm gets the identical noise. Here the noise draw is
wrapped to draw batch 1 and broadcast it to every row (the same values the batch-1 path sees), the encoder runs once on the T
tokens, and the planner runs once on the T x A projected copies. Batched kernels can differ from batch 1 in the last bits; the
`gate` command measures that against the batch-1 path (pre-registered tolerance: max |plan diff| <= 1e-3 m).

  profile --n N      timing: data loading, encoder, planner batch 1 per arm, batched planner per token
  gate --n N         batched vs batch-1 (p1_wajepa sweep) plans on N navtest tokens, every arm -> fastgate.json
  sweep --shard I N  navtest tokens not yet in any sweep chunk (sweep/*/c*.npz), shard I of N -> sweep/fIofN/c*.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_wajepa as W  # noqa: E402
from p1_wajepa import UP  # noqa: E402

KS = [None] + W.KS


class Batched:
    """Hooks for one batched call: encoder on the T unique tokens, arms expanded, noise broadcast from a batch-1 draw."""

    def __init__(self, model, pca, arms):
        self.m, self.arms = model, arms
        self.basis = UP._Basis(pca, "cuda")
        self.enc, self.rnd = model._encode_scene_context, model._randn_for_inference
        self.feat = None

    def __enter__(self):
        A, me = len(self.arms), self

        def enc(clip, masks_x, batch_size):
            V = me.m.num_cameras
            T = batch_size // A
            cam = (torch.arange(T, device=clip.device) * A * V).repeat_interleave(V) + torch.arange(V, device=clip.device).repeat(T)
            x = me.enc(clip.index_select(0, cam), masks_x.index_select(0, torch.arange(T, device=clip.device) * A), T)
            me.feat = x
            return torch.cat([me.basis.project(x, k).unsqueeze(1) for k in me.arms], 1).reshape(T * A, *x.shape[1:])

        def rnd(shape, **kw):
            return me.rnd((1,) + tuple(shape[1:]), **kw).expand(*shape).contiguous()
        self.m._encode_scene_context, self.m._randn_for_inference = enc, rnd
        return self

    def __exit__(self, *e):
        del self.m._encode_scene_context, self.m._randn_for_inference
        return False


def stack(fs, A):
    """Features of T tokens (batch 1 each) -> one batch of T x A rows, token-major."""
    return {k: torch.cat([f[k] for f in fs]).repeat_interleave(A, 0) for k in fs[0]}


AMP = False   # bf16 autocast (the repo's HUGSIM adapter precision); TF32 is a global switch (--tf32)


def run_batched(A_, fs, pca, arms=KS):
    with Batched(A_.model, pca, arms) as b, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
        out = A_.model.predict_trajectory(stack(fs, len(arms))).float().cpu().numpy().astype(np.float32)
    return out.reshape(len(fs), len(arms), *out.shape[1:]).transpose(1, 0, 2, 3), b.feat     # [A, T, 8, 3]


def cmd_profile(a):
    A_ = W.agent()
    L, toks = W.loader("navtest", A_.get_sensor_config())
    toks = toks[:: len(toks) // a.n][: a.n]
    pca = UP.PCA.load(W.OUT / "pca.npz")
    sync = torch.cuda.synchronize
    t0 = time.time()
    fs = [f for _, f in W.batches(A_, L, toks, W.workers())]
    t_load = (time.time() - t0) / len(fs)
    m = A_.model
    with UP.Probe(m, W.SITE) as p:
        p.use_pca(pca, "cuda")
        sync(); t0 = time.time()
        for f in fs:
            W.plan(A_, f)
        sync(); t_plain = (time.time() - t0) / len(fs)
        enc_t = []
        orig = m._encode_scene_context
        for f in fs:
            def timed(*aa, **kw):
                sync(); s = time.time(); r = orig(*aa, **kw); sync(); enc_t.append(time.time() - s); return r
            m._encode_scene_context = timed
            W.plan(A_, f)
            del m._encode_scene_context
        sync(); t0 = time.time()
        for f in fs:
            p.sweep(lambda: W.plan(A_, f), KS)
        sync(); t_sweep = (time.time() - t0) / len(fs)
    res = dict(n=len(fs), load_s_per_token_amortised=t_load, plain_call_s=t_plain, encoder_s=float(np.mean(enc_t)),
               planner_s=t_plain - float(np.mean(enc_t)), sweep_8arms_batch1_s=t_sweep)
    global AMP
    ref = None
    for mode in ("fp32", "tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = mode == "tf32"
        AMP = mode == "bf16"
        for T in a.batch:
            sync(); t0 = time.time()
            out = np.concatenate([run_batched(A_, fs[i:i + T], pca)[0] for i in range(0, len(fs), T)], 1)
            sync()
            res[f"{mode}_T{T}_s_per_token"] = (time.time() - t0) / len(fs)
            res[f"{mode}_T{T}_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if ref is None:
                ref = out
            res[f"{mode}_T{T}_max_abs_m_vs_fp32_T{a.batch[0]}"] = float(np.abs(out - ref).max())
            res[f"{mode}_T{T}_ade_m_vs_fp32_T{a.batch[0]}"] = float(np.linalg.norm(out[..., :2] - ref[..., :2], axis=-1).mean())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    AMP = False
    (W.OUT / "profile.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_gate(a):
    A_ = W.agent()
    L, toks = W.loader("navtest", A_.get_sensor_config())
    toks = toks[:: len(toks) // a.n][: a.n]
    pca = UP.PCA.load(W.OUT / "pca.npz")
    fs = [f for _, f in W.batches(A_, L, toks, W.workers())]
    ref = []
    with UP.Probe(A_.model, W.SITE) as p:
        p.use_pca(pca, "cuda")
        for f in fs:
            r = p.sweep(lambda: W.plan(A_, f), KS)
            ref.append(np.stack([r[k] for k in KS]))
    ref = np.stack(ref, 1)                                         # [A, n, 8, 3]
    got = np.concatenate([run_batched(A_, fs[i:i + a.T], pca)[0] for i in range(0, len(fs), a.T)], 1)
    d = np.abs(got - ref).max(axis=(1, 2, 3))
    res = dict(n=len(fs), T=a.T, max_abs_m_per_arm={W.kname(k): float(x) for k, x in zip(KS, d)}, passed=bool(d.max() <= 1e-3))
    (W.OUT / "fastgate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if not res["passed"]:
        sys.exit("batched path differs from batch 1 by more than 1e-3 m")


def cmd_sweep(a):
    from jevdrive import par
    A_ = W.agent()
    L, toks = W.loader("navtest", A_.get_sensor_config())
    done = set()
    for f in (W.OUT / "sweep").glob("*/c*.npz"):
        if not f.parent.name.startswith("f"):         # chunks of the batch-1 shards
            done |= set(np.load(f)["tokens"].tolist())
    mine = par.shards([t for t in toks if t not in done], a.shard[1], a.shard[0])
    d = W.OUT / "sweep" / f"f{a.shard[0]}of{a.shard[1]}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokens.txt").write_text("\n".join(mine) + "\n")
    pca = UP.PCA.load(W.OUT / "pca.npz")
    basis = UP._Basis(pca, "cuda")
    chunks = [(c0, mine[c0:c0 + a.chunk]) for c0 in range(0, len(mine), a.chunk) if not (d / f"c{c0:06d}.npz").exists()]
    it = W.batches(A_, L, [t for _, c in chunks for t in c], W.workers())
    t0, n = time.time(), 0
    for c0, c in chunks:
        T, P, et, er = [], [], 0.0, np.zeros(len(W.KS))
        for i in range(0, len(c), a.T):
            tf = [next(it) for _ in c[i:i + a.T]]
            out, feat = run_batched(A_, [f for _, f in tf], pca)
            e = basis.energy(feat, W.KS)
            et += e["total"]
            er += np.array([e[k] for k in W.KS])
            T += [t for t, _ in tf]
            P.append(out)
        assert T == list(c), "token order"
        n += len(T)
        np.savez(d / "tmp.npz", tokens=np.array(T), poses=np.concatenate(P, 1), ks=np.array([W.kname(k) for k in KS]),
                 e_total=et, e_res=er)
        os.replace(d / "tmp.npz", d / f"c{c0:06d}.npz")
        print(f"[fast] chunk {c0}: {n}/{sum(len(x) for _, x in chunks)} {(time.time() - t0) / n:.2f} s/token", flush=True)
    (d / "DONE").write_text(json.dumps(dict(tokens=len(mine), wall_s=time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "gate", "sweep"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--T", type=int, default=4, help="tokens per batched call (x 8 arms)")
    ap.add_argument("--batch", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    os.chdir(W.WJ)
    from jevdrive.run import Run
    from jevdrive.data import splits
    with Run("wm_surplus", f"p1fast-{a.cmd}" + (f"-s{a.shard[0]}of{a.shard[1]}" if a.cmd == "sweep" else ""), config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
