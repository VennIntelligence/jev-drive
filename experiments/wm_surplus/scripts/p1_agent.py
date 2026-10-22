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
  DiffusionDrive    `agent._transfuser_model._backbone` (TransFuser image + LiDAR fusion) -> (bev_feature_upscale, bev_feature):
                    both BEV maps the decoder and the diffusion head read (the image grid is discarded by the shipped model).
                    One PCA per map over its channels at every BEV cell; arm k keeps min(k, C) directions of each.
  WoTE              `agent._wote_model._backbone` (TransFuser backbone of the BEV world model), same two BEV maps.

  gate  --model M [--n 24] [--pca]   identity gate (G1 hook off == no hook, G3 sweep off == no hook, G4 determinism, R full rank)
  fit   --model M --shard I N        navtrain moments (tokens whose sensors are on disk, --max spread over logs)
  pca   --model M                    merge moments -> pca_<site>.npz
  sweep --model M --shard I N        navtest poses per arm -> sweep/sIofN/c*.npz
  merge --model M                    -> bench pred files PROBE-<tag>-<arm> (score: python -m jevdrive.bench run --model ... --bench navtest)

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
sys.path[:0] = [str(REPO / "experiments/wm_surplus/lib"), str(REPO), str(REPO / "jevdrive/bench")]
import usage_probe as UP  # noqa: E402
import agent_poses as AP  # noqa: E402
from jevdrive.bench.models import agent_cfg, data_dir  # noqa: E402

NAV = data_dir() / "datasets" / "navsim"
TAGS = {"ReCogDrive-2B-RL": "recog-lmh", "DiffusionDrive": "dd-bev", "WoTE": "wote-bev"}


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
    if model in ("DiffusionDrive", "WoTE"):
        owner = agent._transfuser_model if model == "DiffusionDrive" else agent._wote_model
        def pick(o):
            return [chlast(o[0]), chlast(o[1])]

        def rebuild(o, f):
            return (chfirst(f[0]), chfirst(f[1]), *o[2:])
        return Site(owner._backbone, pick, rebuild, 2)
    raise SystemExit(f"no P1 site for {model}")


def seed_of(tok: str) -> int:
    return int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)


def call(agent, L, tok, S=None):
    if S is not None:
        S.memo = None                    # a new token: the producer runs again
    torch.manual_seed(seed_of(tok))
    with torch.no_grad():
        return AP.trajectory(agent, L, tok)


def build(model: str):
    c = agent_cfg(model)
    for k, v in __import__("jevdrive.bench.navsim", fromlist=["agent_env"]).agent_env(c).items():
        os.environ.setdefault(k, v)
    cfg, agent = AP.build(Path(c["repo"]), c.get("config_dir", "navsim/planning/script/config/pdm_scoring"),
                          c.get("config_name", "default_run_pdm_score"), c["overrides"])
    return c, cfg, agent


def navtest_tokens():
    from jevdrive.bench.sets import navsim_tokens
    t, lg = navsim_tokens("navtest")
    return t.tolist(), lg.tolist()


def navtrain_tokens(c, cfg, agent, max_tokens: int):
    """navtrain tokens whose sensors (as the agent's sensor config asks) are on disk; spread over logs, at most max_tokens."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    y = next(Path(c["repo"]).glob("navsim/planning/script/config/**/scene_filter/navtrain.yaml"))
    sf = instantiate(OmegaConf.load(y))
    blobs = NAV / "sensor_blobs" / "trainval"
    have = sorted(p.name for p in blobs.iterdir() if (p / "CAM_F0").is_dir()) if blobs.is_dir() else []
    logs = [lg for lg in sf.log_names if lg in have]
    from navsim.common.dataloader import SceneLoader
    sf.log_names = logs
    L = SceneLoader(sensor_blobs_path=blobs, data_path=NAV / "navsim_logs" / "trainval", scene_filter=sf, sensor_config=agent.get_sensor_config())
    sc = agent.get_sensor_config()
    nh = sf.num_history_frames

    def want(attr, j):
        v = getattr(sc, attr, False)
        return (j in v) if isinstance(v, (list, tuple)) else bool(v)

    def ok(t):
        fr = L.scene_frames_dicts[t][:nh]
        for j, f in enumerate(fr):
            for cam, d in f["cams"].items():
                if want(cam.lower(), j) and not (blobs / d["data_path"]).exists():
                    return False
            if want("lidar_pc", j) and not (blobs / f["lidar_path"]).exists():
                return False
        return True
    toks = [t for t in L.tokens if ok(t)]
    if max_tokens and len(toks) > max_tokens:
        toks = toks[:: len(toks) // max_tokens][:max_tokens]
    return L, toks


def ks_for(dims) -> list:
    D = max(dims)
    return sorted({2 ** i for i in range(3, 20) if 2 ** i < D} | {D})


def out_dir(model: str) -> Path:
    d = data_dir() / "runs" / "wm_surplus" / "p1_agents" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_pcas(model):
    d = out_dir(model)
    return [UP.PCA.load(f) for f in sorted(d.glob("pca_f*.npz"))]


# ---------------------------------------------------------------- commands
def cmd_gate(a):
    c, cfg, agent = build(a.model)
    toks, logs = navtest_tokens()
    sel = list(range(0, len(toks), max(1, len(toks) // a.n)))[: a.n]
    T = [toks[i] for i in sel]
    L = AP.scene_loader(cfg, agent, T, sorted({logs[i] for i in sel}))
    pcas = load_pcas(a.model) if a.pca else []
    plain, again, hooked, swept, full = [], [], [], [], []
    for t in T:
        plain.append(call(agent, L, t))
        again.append(call(agent, L, t))
    S = site(a.model, agent)
    with S:
        probes = [UP.Probe(S, f"f{i}") for i in range(S.n)]
        for p in probes:
            p.__enter__()
        try:
            for t in T:
                hooked.append(call(agent, L, t, S))
                if pcas:
                    for p, q in zip(probes, pcas):
                        p.use_pca(q, "cuda")
                    r = sweep(probes, S, lambda: call(agent, L, t), [None, max(q.dim for q in pcas)])
                    swept.append(r[None])
                    full.append(r[max(q.dim for q in pcas)])
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
                for p in probes:
                    for key, v in p.basis.energy(p.cache, [x for x in ks if x is not None]).items():
                        p.energy[key] = p.energy.get(key, 0.0) + v
    finally:
        for p in probes:
            p.mode, p.cache, p.k = "off", None, None
    return out


def cmd_fit(a):
    c, cfg, agent = build(a.model)
    L, toks = navtrain_tokens(c, cfg, agent, a.max)
    mine = toks[a.shard[0]:: a.shard[1]]
    d = out_dir(a.model) / "fit"
    d.mkdir(exist_ok=True)
    S = site(a.model, agent)
    Ms = [UP.Moments() for _ in range(S.n)]
    t0 = time.time()
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        for p, m in zip(probes, Ms):
            p.capture(m)
        for i, t in enumerate(mine, 1):
            call(agent, L, t, S)
            if i % 100 == 0:
                print(f"[fit] {i}/{len(mine)} {(time.time() - t0) / i:.2f} s/token", flush=True)
        for p in probes:
            p.__exit__()
    np.savez(d / f"m_s{a.shard[0]}of{a.shard[1]}.npz", tokens=len(mine), n_sites=S.n,
             **{f"f{i}_{k}": v for i, m in enumerate(Ms) for k, v in m.state().items()})
    print(f"[fit] {len(mine)} navtrain tokens in {time.time() - t0:.0f} s", flush=True)


def cmd_pca(a):
    d = out_dir(a.model)
    fs = sorted((d / "fit").glob("m_s*of*.npz"))
    zs = [dict(np.load(f)) for f in fs]
    info = dict(shards=len(fs), navtrain_tokens=sum(int(z["tokens"]) for z in zs), sites=[])
    for i in range(int(zs[0]["n_sites"])):
        pca = UP.Moments.fit(UP.Moments.merge({k: z[f"f{i}_{k}"] for k in ("n", "s", "ss")} for z in zs))
        pca.save(d / f"pca_f{i}.npz")
        info["sites"].append(dict(dim=pca.dim, rows=pca.n, explained_navtrain={f"k{k}": pca.explained(min(k, pca.dim)) for k in ks_for([pca.dim])}))
    (d / "pca.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def cmd_sweep(a):
    c, cfg, agent = build(a.model)
    toks, logs = navtest_tokens()
    idx = list(range(a.shard[0], len(toks), a.shard[1]))
    pcas = load_pcas(a.model)
    ks = [None] + ks_for([q.dim for q in pcas])
    d = out_dir(a.model) / "sweep" / f"s{a.shard[0]}of{a.shard[1]}"
    d.mkdir(parents=True, exist_ok=True)
    S = site(a.model, agent)
    t0, n = time.time(), 0
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        for p, q in zip(probes, pcas):
            p.use_pca(q, "cuda")
        for c0 in range(0, len(idx), a.chunk):
            f_out = d / f"c{c0:06d}.npz"
            if f_out.exists():
                continue
            part = idx[c0:c0 + a.chunk]
            T = [toks[i] for i in part]
            L = AP.scene_loader(cfg, agent, T, sorted({logs[i] for i in part}))
            for p in probes:
                p.energy = {}
            P = []
            for t in T:
                r = sweep(probes, S, lambda: call(agent, L, t), ks)
                P.append(np.stack([r[k] for k in ks]))
            n += len(T)
            np.savez(d / "tmp.npz", tokens=np.array(T), poses=np.stack(P, 1), ks=np.array(["off" if k is None else f"k{k}" for k in ks]),
                     e_total=np.array([p.energy["total"] for p in probes]),
                     e_res=np.array([[p.energy[k] for k in ks[1:]] for p in probes]))
            os.replace(d / "tmp.npz", f_out)
            print(f"[sweep] {c0 + len(T)}/{len(idx)} {(time.time() - t0) / max(n, 1):.2f} s/token", flush=True)
        for p in probes:
            p.__exit__()
    (d / "DONE").write_text(json.dumps(dict(tokens=len(idx), wall_s=time.time() - t0)))


def cmd_merge(a):
    from jevdrive.bench.models import resolve
    from jevdrive.bench.navsim import pred_file
    d = out_dir(a.model)
    toks, _ = navtest_tokens()
    zs = [np.load(f) for f in sorted((d / "sweep").glob("s*of*/c*.npz"))]
    T = np.concatenate([z["tokens"] for z in zs])
    P = np.concatenate([z["poses"] for z in zs], 1)
    names = [str(x) for x in zs[0]["ks"]]
    assert len(set(T)) == len(T) == len(toks) and set(T) == set(toks), f"{len(T)} plans for {len(toks)} navtest tokens"
    order = np.argsort(T)
    T, P = T[order], P[:, order]
    off = P[names.index("off")]
    for i, nm in enumerate(names):
        pf = pred_file(resolve(f"PROBE-{TAGS[a.model]}-{nm}"), "navtest")
        pf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pf, tokens=T, poses=P[i])
    tot = sum(z["e_total"] for z in zs)
    res = sum(z["e_res"] for z in zs)
    pcas = load_pcas(a.model)                    # several sites: variance-weighted over sites (each keeps min(k, D_i))
    kk = [int(nm[1:]) for nm in names[1:]]
    ev = dict(model=a.model, tokens=len(T), arms=names, site_dims=[q.dim for q in pcas],
              explained_navtrain={nm: float(sum(q.var[:min(k, q.dim)].sum() for q in pcas) / sum(q.var.sum() for q in pcas))
                                  for nm, k in zip(names[1:], kk)},
              explained_navtest={nm: 1 - float(res[:, j].sum()) / float(tot.sum()) for j, nm in enumerate(names[1:])},
              explained_navtest_per_site=[{nm: 1 - float(r) / float(tt) for nm, r in zip(names[1:], rr)} for tt, rr in zip(tot, res)],
              plan_shift_vs_off_m={nm: float(np.linalg.norm(P[i][..., :2] - off[..., :2], axis=-1).mean()) for i, nm in enumerate(names)})
    (d / "ev.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "fit", "pca", "sweep", "merge"])
    ap.add_argument("--model", required=True, choices=sorted(TAGS))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pca", action="store_true", help="gate: also G3 / full-rank checks with the fitted PCA")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    from jevdrive.run import Run
    with Run("wm_surplus", f"p1a-{TAGS[a.model]}-{a.cmd}" + (f"-s{a.shard[0]}of{a.shard[1]}" if a.cmd in ("fit", "sweep") else ""),
             config=vars(a)) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("navsim/navtrain" if a.cmd in ("fit", "pca") else "navsim/navtest"))
        globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
