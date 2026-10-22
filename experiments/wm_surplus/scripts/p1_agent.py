"""P1 level (a) on third-party NAVSIM agents (P0 subjects): project the feature their planner reads onto its top-k navtrain
principal directions, through the P1 hook (lib/usage_probe.Probe), same protocol as p1_wajepa.py.

The agent is built exactly as the bench runs it (jevdrive/bench/agents/<model>.json + jevdrive/bench/agent_poses.py: the fork's
hydra config with its eval overrides, `compute_trajectory`), so arm `off` is the reproduced model. A site adapter splits
one module call of the shipped model into "producer" and "feature" without touching the model code: it replaces the
instance's `forward` of that module by a wrapper that calls the original and routes the feature tensor(s) through
probed holder methods (channel-last rows, the unit of the PCA). In a sweep the producer runs once per token (Probe cached
mode), the rest of the shipped forward runs once per arm. The RNG is re-seeded per token before every call (seed from
the token), so all arms of a token see the same sampler noise (paired).

Sites
  ReCogDrive-2B-RL  `agent.backbone` (InternVL3-2B chat model) -> `outputs.hidden_states[-1]`: the Qwen2.5 LM's last hidden
                    state over the whole prompt (image + text tokens), the only VLM input of the DiT planner. D = LM width.
  (DiffusionDrive / WoTE dropped by PD2: camera-only subjects only.)

PCA fit (differs from p1_wajepa.py, which fits on navtrain): the navtrain camera frames are gone (PD8), so the PCA is
cross-fitted within navtest: navtest logs in 2 folds (sorted log names, i % 2); the fit of fold f uses tokens of fold f only,
and a token of fold g is swept with the PCA of the other fold (a fit never sees the logs it is evaluated on). The `off`
pass also stores a training-free pooled feature X (mean row + last row per site) and the 20 blind inputs B per token for
scripts/p1_knowdo.py (knowdo_feats.npz).

  gate  --model M [--n 24] [--pca]   identity gate (G1 hook off == no hook, G3 sweep off == no hook, G4 determinism, R full rank)
  fit   --model M --fold F --shard I N   moments on navtest tokens of fold F (--max spread over its logs)
  pca   --model M                        merge moments -> pca_fold<F>_f<site>.npz
  sweep --model M --shard I N            navtest poses per arm, cross-fitted PCA -> sweep/sIofN/g<fold>_c*.npz
  merge --model M                        -> bench pred files PROBE-<tag>-<arm>, ev.json, knowdo_feats.npz

Run with the model's env python ($DATA_DIR/envs/<env>/bin/python), env vars of jevdrive.bench.navsim.agent_env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/wm_surplus/lib"), str(REPO)]
import usage_probe as UP  # noqa: E402
import jevdrive.bench.agent_poses as AP  # noqa: E402  (by package path: jevdrive/bench/navsim.py must not shadow the fork's navsim)
from jevdrive.bench.models import agent_cfg, data_dir  # noqa: E402

NAV = data_dir() / "datasets" / "navsim"
TAGS = {"ReCogDrive-2B-RL": "recog-lmh"}
FOLDS = 2


def chlast(x):
    return x.permute(0, *range(2, x.dim()), 1) if x.dim() > 2 else x


def chfirst(x):
    return x.permute(0, x.dim() - 1, *range(1, x.dim() - 1)) if x.dim() > 2 else x


class Site:
    """Feature tensors of one module call, each behind its own probed holder method `f<i>` (channel-last). The module's
    output is memoised per token (`memo`, cleared by `call` / `sweep`), so cached sweep arms reuse the producer's output."""

    def __init__(self, module, pick, rebuild, n: int):
        self.module, self.orig, self.pick, self.rebuild, self.n = module, module.forward, pick, rebuild, n
        self.memo = None

    def _feat(self, i, a, kw):
        if self.memo is None:
            self.memo = self.orig(*a, **kw)
        return self.pick(self.memo)[i]

    def f0(self, *a, **kw):
        return self._feat(0, a, kw)

    def f1(self, *a, **kw):
        return self._feat(1, a, kw)

    def __enter__(self):
        def fwd(*a, **kw):
            feats = [getattr(self, f"f{i}")(*a, **kw) for i in range(self.n)]
            return self.rebuild(self.memo, feats)
        self.module.forward = fwd
        return self

    def __exit__(self, *exc):
        del self.module.forward          # back to the class method
        self.memo = None
        return False


def site(model: str, agent) -> Site:
    if model == "ReCogDrive-2B-RL":
        def pick(o):
            return [o.hidden_states[-1]]

        def rebuild(o, f):
            return types.SimpleNamespace(**{**{k: getattr(o, k) for k in ("logits",) if hasattr(o, k)}, "hidden_states": [f[0]]})
        return Site(agent.backbone, pick, rebuild, 1)
    raise SystemExit(f"no P1 site for {model}")


def navtest_tokens():
    from jevdrive.bench.sets import navsim_tokens
    t, lg = navsim_tokens("navtest")
    return t.tolist(), lg.tolist()


def fold_of_logs(logs) -> dict:
    """log -> fold: sorted navtest log names, i % FOLDS (whole logs; the PCA used on a token is never fit on its log's fold)."""
    return {lg: i % FOLDS for i, lg in enumerate(sorted(set(logs)))}


def loader(c, cfg, agent, T, logs):
    return AP.scene_loader(cfg, agent, T, sorted(set(logs)), sensors=c.get("sensor_overrides"), loader_kw=c.get("loader_kwargs"))


def ks_for(dims) -> list:
    D = max(dims)
    return sorted({2 ** i for i in range(3, 20) if 2 ** i < D} | {D})


def out_dir(model: str) -> Path:
    d = data_dir() / "runs" / "wm_surplus" / "p1_agents" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_pcas(model, fold: int) -> list:
    return [UP.PCA.load(f) for f in sorted(out_dir(model).glob(f"pca_fold{fold}_f*.npz"))]


def blind(inp) -> np.ndarray:
    """What the planner reads besides images (p1 knows-vs-does prereg): command one-hot (4), ego velocity (2) / acceleration (2)
    at t0, the 4 history poses (12) -> 20 values."""
    e = inp.ego_statuses
    return np.concatenate([np.asarray(e[-1].driving_command, np.float32).ravel()[:4], np.asarray(e[-1].ego_velocity, np.float32)[:2],
                           np.asarray(e[-1].ego_acceleration, np.float32)[:2],
                           np.concatenate([np.asarray(x.ego_pose, np.float32)[:3] for x in e[-4:]])])


def pooled(S) -> np.ndarray:
    """Training-free pooling of the site's feature for the knows-vs-does probe: per site, the mean over rows and the last row
    (ReCogDrive: the LM hidden state of the whole prompt and of its final token)."""
    out = []
    for f in S.pick(S.memo):
        x = f.detach().float().reshape(-1, f.shape[-1])
        out += [x.mean(0).cpu().numpy(), x[-1].cpu().numpy()]
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- commands
def cmd_gate(a):
    c, cfg, agent = build(a.model)
    toks, logs = navtest_tokens()
    sel = list(range(0, len(toks), max(1, len(toks) // a.n)))[: a.n]
    T = [toks[i] for i in sel]
    L = loader(c, cfg, agent, T, [logs[i] for i in sel])
    pcas = load_pcas(a.model, 0) if a.pca else []
    plain, again, hooked, swept, full = [], [], [], [], []
    for t in T:
        plain.append(call(agent, L, t))
        again.append(call(agent, L, t))
    S = site(a.model, agent)
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        try:
            for t in T:
                hooked.append(call(agent, L, t, S))
                if pcas:
                    for p, q in zip(probes, pcas):
                        p.use_pca(q, "cuda")
                    D = max(q.dim for q in pcas)
                    r = sweep(probes, S, lambda: call(agent, L, t), [None, D])
                    swept.append(r[None])
                    full.append(r[D])
        finally:
            for p in probes:
                p.__exit__()
    g = dict(model=a.model, n=len(T), G1_off_vs_nohook=UP.identity_gate(plain, hooked), G4_determinism=UP.identity_gate(plain, again))
    if pcas:
        g["G3_sweep_off_vs_nohook"] = UP.identity_gate(plain, swept)
        g["R_fullrank_vs_plain_max_abs_m"] = max(UP.max_abs_diff(x, y) for x, y in zip(plain, full))
    g["passed"] = all(v["bit_identical"] for k, v in g.items() if k.startswith("G"))
    (out_dir(a.model) / ("gate_pca.json" if pcas else "gate.json")).write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))
    if not g["passed"]:
        sys.exit("identity gate failed")


def sweep(probes, S, fn, ks) -> dict:
    """Probe.sweep over several probes of one site: the producer runs once (k = None first), then every k projects each
    probe's cached feature onto min(k, D_i) directions."""
    for p in probes:
        p.mode, p.cache = "cached", None
    S.memo = None
    out = {}
    try:
        for k in ks:
            for p in probes:
                p.k = None if k is None else min(k, p.basis.U.shape[0])
            out[k] = fn()
            if k is None:
                out["_pooled"] = pooled(S)
                for p in probes:
                    for key, v in p.basis.energy(p.cache, [x for x in ks if x is not None]).items():
                        p.energy[key] = p.energy.get(key, 0.0) + v
    finally:
        for p in probes:
            p.mode, p.cache, p.k = "off", None, None
    return out


def cmd_fit(a):
    """Moments of the feature on navtest tokens of fold --fold (at most --max, spread over its logs); shard I of N."""
    c, cfg, agent = build(a.model)
    toks, logs = navtest_tokens()
    fo = fold_of_logs(logs)
    idx = [i for i, lg in enumerate(logs) if fo[lg] == a.fold]
    if a.max and len(idx) > a.max:
        idx = idx[:: len(idx) // a.max][: a.max]
    mine = idx[a.shard[0]:: a.shard[1]]
    T = [toks[i] for i in mine]
    L = loader(c, cfg, agent, T, [logs[i] for i in mine])
    d = out_dir(a.model) / "fit"
    d.mkdir(exist_ok=True)
    S = site(a.model, agent)
    Ms = [UP.Moments() for _ in range(S.n)]
    t0 = time.time()
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        for p, m in zip(probes, Ms):
            p.capture(m)
        for i, t in enumerate(T, 1):
            call(agent, L, t, S)
            if i % 100 == 0:
                print(f"[fit] {i}/{len(T)} {(time.time() - t0) / i:.2f} s/token", flush=True)
        for p in probes:
            p.__exit__()
    np.savez(d / f"fold{a.fold}_m_s{a.shard[0]}of{a.shard[1]}.npz", tokens=len(T), n_sites=S.n,
             **{f"f{i}_{k}": v for i, m in enumerate(Ms) for k, v in m.state().items()})
    print(f"[fit] fold {a.fold}: {len(T)} navtest tokens in {time.time() - t0:.0f} s", flush=True)


def cmd_pca(a):
    d = out_dir(a.model)
    info = {}
    for f in range(FOLDS):
        fs = sorted((d / "fit").glob(f"fold{f}_m_s*of*.npz"))
        zs = [dict(np.load(x)) for x in fs]
        inf = dict(shards=len(fs), tokens=sum(int(z["tokens"]) for z in zs), sites=[])
        for i in range(int(zs[0]["n_sites"])):
            pca = UP.Moments.fit(UP.Moments.merge({k: z[f"f{i}_{k}"] for k in ("n", "s", "ss")} for z in zs))
            pca.save(d / f"pca_fold{f}_f{i}.npz")
            inf["sites"].append(dict(dim=pca.dim, rows=pca.n, explained_fit={f"k{k}": pca.explained(min(k, pca.dim)) for k in ks_for([pca.dim])}))
        info[f"fold{f}"] = inf
    (d / "pca.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def cmd_sweep(a):
    """navtest tokens of shard I of N; a token of fold g is projected with the PCA fit on the other fold(s) (cross-fitting)."""
    c, cfg, agent = build(a.model)
    toks, logs = navtest_tokens()
    fo = fold_of_logs(logs)
    idx = list(range(a.shard[0], len(toks), a.shard[1]))
    P_by = {g: load_pcas(a.model, (g + 1) % FOLDS) for g in range(FOLDS)}     # FOLDS = 2: the other fold
    ks = [None] + ks_for([q.dim for q in P_by[0]])
    d = out_dir(a.model) / "sweep" / f"s{a.shard[0]}of{a.shard[1]}"
    d.mkdir(parents=True, exist_ok=True)
    S = site(a.model, agent)
    t0, n = time.time(), 0
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        for g in range(FOLDS):
            for c0 in range(0, len(idx), a.chunk):
                part = [i for i in idx[c0:c0 + a.chunk] if fo[logs[i]] == g]
                f_out = d / f"g{g}_c{c0:06d}.npz"
                if not part or f_out.exists():
                    continue
                for p, q in zip(probes, P_by[g]):
                    p.use_pca(q, "cuda")
                    p.energy = {}
                T = [toks[i] for i in part]
                L = loader(c, cfg, agent, T, [logs[i] for i in part])
                P, X, B = [], [], []
                for t in T:
                    r = sweep(probes, S, lambda: call(agent, L, t), ks)
                    P.append(np.stack([r[k] for k in ks]))
                    X.append(r["_pooled"])
                    B.append(blind(L.get_agent_input_from_token(t)))
                n += len(T)
                np.savez(d / "tmp.npz", tokens=np.array(T), fold=g, poses=np.stack(P, 1), X=np.stack(X), B=np.stack(B),
                         ks=np.array(["off" if k is None else f"k{k}" for k in ks]),
                         e_total=np.array([p.energy["total"] for p in probes]),
                         e_res=np.array([[p.energy[k] for k in ks[1:]] for p in probes]))
                os.replace(d / "tmp.npz", f_out)
                print(f"[sweep] fold {g} chunk {c0}: {n}/{len(idx)} {(time.time() - t0) / max(n, 1):.2f} s/token", flush=True)
        for p in probes:
            p.__exit__()
    (d / "DONE").write_text(json.dumps(dict(tokens=len(idx), wall_s=time.time() - t0)))


def cmd_merge(a):
    from jevdrive.bench.models import resolve
    from jevdrive.bench.navsim import pred_file
    d = out_dir(a.model)
    toks, _ = navtest_tokens()
    zs = [np.load(f) for f in sorted((d / "sweep").glob("s*of*/g*_c*.npz"))]
    T = np.concatenate([z["tokens"] for z in zs])
    P = np.concatenate([z["poses"] for z in zs], 1)
    names = [str(x) for x in zs[0]["ks"]]
    assert len(set(T)) == len(T) == len(toks) and set(T) == set(toks), f"{len(T)} plans for {len(toks)} navtest tokens"
    order = np.argsort(T)
    T, P = T[order], P[:, order]
    np.savez(d / "knowdo_feats.npz", tokens=T, X=np.concatenate([z["X"] for z in zs])[order], B=np.concatenate([z["B"] for z in zs])[order])
    off = P[names.index("off")]
    for i, nm in enumerate(names):
        pf = pred_file(resolve(f"PROBE-{TAGS[a.model]}-{nm}"), "navtest")
        pf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pf, tokens=T, poses=P[i])
    tot = sum(z["e_total"] for z in zs)
    res = sum(z["e_res"] for z in zs)
    pc = [load_pcas(a.model, f) for f in range(FOLDS)]
    kk = [int(nm[1:]) for nm in names[1:]]
    ev = dict(model=a.model, tokens=len(T), arms=names, site_dims=[q.dim for q in pc[0]], fit="navtest 2-fold log cross-fit",
              explained_navtrain={nm: float(np.mean([sum(q.var[:min(k, q.dim)].sum() for q in ps) / sum(q.var.sum() for q in ps) for ps in pc]))
                                  for nm, k in zip(names[1:], kk)},
              explained_navtest={nm: 1 - float(res[:, j].sum()) / float(tot.sum()) for j, nm in enumerate(names[1:])},
              plan_shift_vs_off_m={nm: float(np.linalg.norm(P[i][..., :2] - off[..., :2], axis=-1).mean()) for i, nm in enumerate(names)})
    ev["note"] = "explained_navtrain = in-fold (fit) explained variance averaged over the 2 folds (key name kept for p1_readout); explained_navtest = out-of-fold"
    (d / "ev.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "fit", "pca", "sweep", "merge"])
    ap.add_argument("--model", required=True, choices=sorted(TAGS))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pca", action="store_true", help="gate: also G3 / full-rank checks with the fold-0 PCA")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--max", type=int, default=3000, help="fit tokens per fold (spread over the fold's logs)")
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    from jevdrive.run import Run
    tag = f"p1a-{TAGS[a.model]}-{a.cmd}" + (f"-f{a.fold}" if a.cmd == "fit" else "") + (f"-s{a.shard[0]}of{a.shard[1]}" if a.cmd in ("fit", "sweep") else "")
    with Run("wm_surplus", tag, config=vars(a)) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("navsim/navtest"))
        globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
