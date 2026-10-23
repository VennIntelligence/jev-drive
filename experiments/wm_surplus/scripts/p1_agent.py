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
  DriveVLA-W0       Emu3Pi0 (Pi0-style: the 1024-wide action expert attends, in each of the 32 layers, to that layer's VLM
                    K / V). Sites f0..f31 = the input-normed VLM hidden state of layer l (`input_layernorm` output, [1, P, 4096])
                    at the P positions the expert may attend to (before the second BOA token: both frames' text + VQ image
                    tokens + previous actions); K / V are linear in it, so projecting it restricts what the expert can read
                    from that layer. The VLM is causal and never sees the expert, so this stream is the same for all 10 flow
                    steps. Not a NAVSIM agent: built in-process exactly as their inference script (dataset item -> model
                    .sample_actions -> de-normalised (dx, dy, dyaw) -> poses as drivevla_w0_navtest.py). Probe active: the
                    VLM prefix runs once per token, then their Euler loop runs with every layer's expert attention on the
                    cached (projected) K / V through their own `Emu3Pi0SharedLayer.forward_with_cache` (W0Site); probe off:
                    their `sample_actions` untouched. Flow noise: unseeded as shipped; here torch.manual_seed(seed of the
                    token) before every call, so arms of a token share z (the shipped noise floor is 0.10 m mean |xy|).
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
TAGS = {"ReCogDrive-2B-RL": "recog-lmh", "DriveVLA-W0": "w0-vlmkv"}
G3_TOL_M = 0.01
PCA32 = {"DriveVLA-W0"}          # 32 sites x 4096^2: PCA comps stored float32, fit moments deleted after the PCA
FOLDS = 2


def chlast(x):
    return x.permute(0, *range(2, x.dim()), 1) if x.dim() > 2 else x


def chfirst(x):
    return x.permute(0, x.dim() - 1, *range(1, x.dim() - 1)) if x.dim() > 2 else x


class Site:
    """Feature tensors of one module call, each behind its own probed holder method `f<i>` (channel-last). The module's
    output is memoised per token (`memo`, cleared by `call` / `sweep`), so cached sweep arms reuse the producer's output.
    `exact`: the hooked path is the shipped computation (sweep arm off must be bit-identical to no hook); `active` is set
    by fit / sweep (a site whose hooked path differs from the shipped one runs the shipped one while inactive)."""
    exact = True

    def __init__(self, module, pick, rebuild, n: int, attr: str = "forward"):
        self.module, self.attr, self.orig, self.n = module, attr, getattr(module, attr), n
        if pick is not None:
            self.pick, self.rebuild = pick, rebuild
        self.memo, self.active = None, False

    def _feat(self, i, a, kw):
        if self.memo is None:
            self.memo = self.orig(*a, **kw)
        return self.pick(self.memo)[i]

    def pool(self, memo) -> list:
        """Tensors pooled for the knows-vs-does features (default: every site)."""
        return self.pick(memo)

    def run(self, feats, a, kw):
        return self.rebuild(self.memo, feats)

    def __enter__(self):
        def fwd(*a, **kw):
            feats = [getattr(self, f"f{i}")(*a, **kw) for i in range(self.n)]
            return self.run(feats, a, kw)
        setattr(self.module, self.attr, fwd)
        return self

    def __exit__(self, *exc):
        delattr(self.module, self.attr)  # back to the class method
        self.memo = None
        return False


for _i in range(64):                     # probed holder methods f0 .. f63 (Probe wraps them on the instance)
    setattr(Site, f"f{_i}", lambda self, *a, _i=_i, **kw: self._feat(_i, a, kw))


class W0:
    """DriveVLA-W0 as their inference script builds it (inference/vla/inference_action_navsim_flow_matching_vava.py: config,
    DataArguments, Emu3Tokenizer, Emu3DrivingVAVADataset over the full navtest meta pickle, Emu3Pi0.from_pretrained, bf16,
    sdpa), in-process. Dataset items are indexed by token (pickle order = their navtest.yaml order)."""
    native = True

    def __init__(self):
        sys.path.insert(0, str(REPO / "experiments/wm_surplus/scripts"))
        import yaml
        import drivevla_w0_navtest as WN
        os.environ.update(WN.w0_env(WN.YAML, 0))
        R = WN.ROOT
        sys.path[:0] = [str(R / "inference/vla"), str(R / "inference/navsim/navsim"), str(R), str(R / "utils")]
        assert "datasets" not in sys.modules, "HF datasets imported before their utils/datasets.py"
        from config import get_config, setup_paths_early
        setup_paths_early()
        from emu3.mllm import Emu3Pi0, Emu3Pi0Config, Emu3Tokenizer
        import emu3.mllm.modeling_emu3 as ME
        from utils.train_pi0 import DataArguments
        from datasets import Emu3DrivingVAVADataset
        cfg = get_config(None)
        self.ME, self.rel_to_abs = ME, WN.rel_to_abs
        self.af, self.ad = cfg.get("data.action_frames", 8), cfg.get("data.action_dim", 3)
        da = DataArguments(data_path=str(WN.PKL), actions=True, driving=True, use_previous_actions=True, actions_format="fast",
                           action_tokenizer_path=cfg.get_path("paths.action_tokenizer"), frames=cfg.get("data.frames", 1),
                           action_frames=self.af, action_dim=self.ad, cur_frame_idx=cfg.get("data.cur_frame_idx", 3),
                           pre_action_frames=cfg.get("data.pre_action_frames", 3), video_format=None, use_flip=False)
        vlm = cfg.get_path("paths.vlm_model")
        tok = Emu3Tokenizer.from_pretrained(vlm, model_max_length=cfg.get("model.model_max_length", 1400),
                                            padding_side=cfg.get("model.padding_side", "right"), use_fast=cfg.get("model.use_fast", False))
        self.ds = Emu3DrivingVAVADataset(da, tokenizer=tok)
        dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(cfg.get("model.torch_dtype", "bfloat16"), torch.bfloat16)
        model = Emu3Pi0.from_pretrained(str(WN.HUB), config=Emu3Pi0Config.from_pretrained(str(WN.HUB / "config.json")), pretrain_vlm_path=vlm,
                                        attn_implementation=cfg.get("model.attn_implementation", "sdpa"), torch_dtype=dt, trust_remote_code=True)
        self.model = model.to("cuda").eval()
        ns = json.load(open(cfg.get_path("paths.norm_stats")))["norm_stats"]["libero"]
        self.low, self.high = torch.tensor(ns["q01"], device="cuda"), torch.tensor(ns["q99"], device="cuda")
        self.steps = cfg.get("inference.num_inference_steps", 10)
        allt = yaml.safe_load(WN.YAML.read_text())["tokens"]
        assert len(allt) == len(self.ds), (len(allt), len(self.ds))
        self.index = {t: i for i, t in enumerate(allt)}
        self._b = (None, None)

    def batch(self, tok) -> dict:
        """Collated dataset item of the token on the card (memoised: every sweep arm of a token reuses it)."""
        if self._b[0] != tok:
            item = self.ds[self.index[tok]]
            fn = getattr(self.ds, "collate_fn", None) or torch.utils.data.default_collate
            self._b = (tok, {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in fn([item]).items()})
        return self._b[1]

    def trajectory(self, L, tok) -> np.ndarray:
        """Their per-token step: sample_actions -> de-normalise (their arithmetic) -> (dx, dy, dyaw) chained into poses."""
        b = self.batch(tok)
        pred = self.model.sample_actions(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                                         pre_action=b["pre_action"].to(torch.bfloat16), cmd=b["cmd"].to(torch.bfloat16),
                                         num_inference_steps=self.steps, action_frames=self.af, action_dim=self.ad)
        act = 0.5 * (pred.squeeze(0) + 1) * (self.high - self.low) + self.low
        return self.rel_to_abs(act.cpu().numpy().tolist()).astype(np.float32)

    def blind(self, L, tok) -> np.ndarray:
        """What its action expert reads besides the VLM: command one-hot (4) + the 3 previous normalised actions (9)."""
        b = self.batch(tok)
        return torch.cat([b["cmd"].reshape(-1), b["pre_action"].reshape(-1)]).float().cpu().numpy()


class W0Site(Site):
    """f<l> = input-normed VLM hidden state of layer l at the positions the action expert attends to (see module doc).
    Active: `produce` runs the VLM prefix once per token (their shared-layer arithmetic, VLM rows only, on the visible
    prefix; exact in math since the VLM is causal and blind to the expert), `run` = their Euler loop of sample_actions with
    each layer's expert step = their forward_with_cache on K / V computed from the (projected) f<l>."""
    exact = False

    def __init__(self, w0: W0):
        self.m = w0.model
        super().__init__(self.m, None, None, len(self.m.vlm.model.layers), attr="sample_actions")

    def __enter__(self):
        def fwd(*a, **kw):
            if not self.active:
                return self.orig(*a, **kw)
            feats = [getattr(self, f"f{i}")(*a, **kw) for i in range(self.n)]
            return self.run(feats, a, kw)
        setattr(self.module, self.attr, fwd)
        return self

    def pick(self, memo) -> list:
        return memo["h"]

    def pool(self, memo) -> list:
        return memo["h"][-1:]            # last layer's input (the deepest state the expert reads)

    def _feat(self, i, a, kw):
        if self.memo is None:
            self.memo = self.produce(**kw)
        return self.memo["h"][i]

    @torch.no_grad()
    def produce(self, input_ids, attention_mask, action_frames, **_):
        m, ME = self.m, sys.modules["emu3.mllm.modeling_emu3"]
        B, dev = input_ids.shape[0], input_ids.device
        assert B == 1
        x = m.vlm.model.embed_tokens(input_ids)
        S, A = x.shape[1], action_frames + 1
        mask = m.create_causal_style_attention_mask(S, A, attention_mask, input_ids, B, dev, x.dtype)
        boa = input_ids[0] == m.config.boa_token_id
        P = int(torch.nonzero(boa)[1])                       # second BOA: the expert sees positions < P
        assert bool(attention_mask[0, :P].all()) and bool((mask[0, 0, S:, :P] == 0).all()) and bool((mask[0, 0, S:, P:S] != 0).all())
        pos = torch.arange(S, device=dev, dtype=torch.long).unsqueeze(0).expand(B, -1)
        h, H = x[:, :P], []
        for j, L in enumerate(m.vlm.model.layers):
            n = L.input_layernorm(h)
            H.append(n)
            if j == len(m.vlm.model.layers) - 1:
                break
            at = L.self_attn
            q = at.q_proj(n).view(B, P, at.num_heads, at.head_dim).transpose(1, 2)
            k = at.k_proj(n).view(B, P, at.num_key_value_heads, at.head_dim).transpose(1, 2)
            v = at.v_proj(n).view(B, P, at.num_key_value_heads, at.head_dim).transpose(1, 2)
            cos, sin = at.rotary_emb(v, seq_len=S)
            q, k = ME.apply_rotary_pos_emb(q, k, cos, sin, pos[:, :P], unsqueeze_dim=1)
            o = torch.nn.functional.scaled_dot_product_attention(
                q.contiguous(), ME.repeat_kv(k, at.num_key_value_groups).contiguous(), ME.repeat_kv(v, at.num_key_value_groups).contiguous(),
                attn_mask=mask[:, :, :P, :P], dropout_p=0.0, is_causal=False)
            h = h + L.dropout(at.o_proj(o.transpose(1, 2).reshape(B, P, at.num_heads * at.head_dim)))
            h = h + L.dropout(L.mlp(L.post_attention_layernorm(h)))
        return dict(h=H, P=P, S=S, A=A, pos=pos, amask=torch.cat([mask[:, :, S:, :P], mask[:, :, S:, S:]], -1))

    @torch.no_grad()
    def run(self, feats, a, kw):
        m, ME, M = self.m, sys.modules["emu3.mllm.modeling_emu3"], self.memo
        P, S, A, B = M["P"], M["S"], M["A"], 1
        kv = []
        for n, L in zip(feats, m.vlm.model.layers):
            at = L.self_attn
            k = at.k_proj(n).view(B, P, at.num_key_value_heads, at.head_dim).transpose(1, 2)
            v = at.v_proj(n).view(B, P, at.num_key_value_heads, at.head_dim).transpose(1, 2)
            cos, sin = at.rotary_emb(v, seq_len=S)
            _, k = ME.apply_rotary_pos_emb(k, k, cos, sin, M["pos"][:, :P], unsqueeze_dim=1)
            kv.append((k, v))
        # their sample_actions loop from the noise draw on (same dtypes, bf16 time stepping), expert attention on cached K / V
        AF, AD, dtype = kw["action_frames"], kw["action_dim"], feats[0].dtype
        z = torch.randn(B, AF, AD, device=feats[0].device, dtype=dtype)
        st = m.state_projector(torch.cat([kw["pre_action"].view(B, -1), kw["cmd"]], dim=1)).unsqueeze(1)
        dt = -1.0 / kw["num_inference_steps"]
        dt_tensor = torch.tensor(dt, dtype=dtype, device=z.device)
        t = torch.tensor(1.0, dtype=dtype, device=z.device)
        while t >= -dt / 2:
            tau = m.tau_emb(t.expand(B).to(z.dtype)).to(z.dtype)
            taue = tau.unsqueeze(1).expand(-1, AF, -1)
            h = torch.cat([st, m.action_projector(z, taue)], dim=1)
            for sl, (k, v) in zip(m.shared_layers, kv):
                h = sl.forward_with_cache(h, action_attention_mask=M["amask"], vlm_seq_len=P, action_seq_len=A, batch_size=B,
                                          vlm_k_rope_cached=k, vlm_v_cached=v)
            z = z + dt * m.action_decoder(m.action_expert.norm(h)[:, 1:, :], taue)
            t += dt_tensor
        return z


def site(model: str, agent) -> Site:
    if model == "DriveVLA-W0":
        return W0Site(agent)
    if model == "ReCogDrive-2B-RL":
        def pick(o):
            return [o.hidden_states[-1]]

        def rebuild(o, f):
            return types.SimpleNamespace(**{**{k: getattr(o, k) for k in ("logits",) if hasattr(o, k)}, "hidden_states": [f[0]]})
        return Site(agent.backbone, pick, rebuild, 1)
    raise SystemExit(f"no P1 site for {model}")


def seed_of(tok: str) -> int:
    return int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)


def call(agent, L, tok, S=None):
    if S is not None:
        S.memo = None                    # a new token: the producer runs again
    torch.manual_seed(seed_of(tok))
    with torch.no_grad():
        return agent.trajectory(L, tok) if getattr(agent, "native", False) else AP.trajectory(agent, L, tok)


def build(model: str):
    from jevdrive.bench.navsim import agent_env
    c = agent_cfg(model)
    for k, v in agent_env(c).items():
        os.environ.setdefault(k, v)
    if model == "DriveVLA-W0":
        return c, None, W0()
    cfg, agent = AP.build(Path(c["repo"]), c.get("config_dir", "navsim/planning/script/config/pdm_scoring"),
                          c.get("config_name", "default_run_pdm_score"), c["overrides"])
    return c, cfg, agent


def navtest_tokens():
    from jevdrive.bench.sets import navsim_tokens
    t, lg = navsim_tokens("navtest")
    return t.tolist(), lg.tolist()


def fold_of_logs(logs) -> dict:
    """log -> fold: sorted navtest log names, i % FOLDS (whole logs; the PCA used on a token is never fit on its log's fold)."""
    return {lg: i % FOLDS for i, lg in enumerate(sorted(set(logs)))}


def loader(c, cfg, agent, T, logs):
    if getattr(agent, "native", False):
        return None                      # the subject indexes its own inputs by token
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


def blind_of(agent, L, t) -> np.ndarray:
    return agent.blind(L, t) if getattr(agent, "native", False) else blind(L.get_agent_input_from_token(t))


def pooled(S) -> np.ndarray:
    """Training-free pooling of the site's feature for the knows-vs-does probe: per pooled site, the mean over rows and the
    last row (ReCogDrive: the LM hidden state of the whole prompt and of its final token; DriveVLA-W0: the last layer's
    expert-visible input over the visible prefix and at its last position)."""
    out = []
    for f in S.pool(S.memo):
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
    plain, again, hooked, swept, swept2, full = [], [], [], [], [], []
    t0 = time.time()
    for t in T:
        plain.append(call(agent, L, t))
        again.append(call(agent, L, t))
    t_plain = (time.time() - t0) / (2 * len(T))
    S = site(a.model, agent)
    t0 = time.time()
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        try:
            for t in T:
                hooked.append(call(agent, L, t, S))
                for p, q in zip(probes, pcas):
                    p.use_pca(q, "cuda")
                D = max(q.dim for q in pcas) if pcas else None
                r = sweep(probes, S, lambda: call(agent, L, t), [None] + ([D] if pcas else []))
                swept.append(r[None])
                if pcas:
                    full.append(r[D])
                swept2.append(sweep(probes, S, lambda: call(agent, L, t), [None])[None])
        finally:
            for p in probes:
                p.__exit__()
    xy = lambda A, B: float(np.mean([np.abs(np.asarray(x)[:, :2] - np.asarray(y)[:, :2]).mean() for x, y in zip(A, B)]))  # noqa: E731
    g = dict(model=a.model, n=len(T), exact_site=S.exact, s_per_token_plain=t_plain, s_per_token_hooked_sweeps=(time.time() - t0) / len(T),
             G1_off_vs_nohook=UP.identity_gate(plain, hooked), G4_determinism=UP.identity_gate(plain, again),
             G5_sweep_determinism=UP.identity_gate(swept, swept2), G3_sweep_off_vs_nohook=UP.identity_gate(plain, swept))
    g["G3_sweep_off_vs_nohook"]["mean_abs_xy_m"] = xy(plain, swept)
    if pcas:
        g["R_fullrank_vs_plain_max_abs_m"] = max(UP.max_abs_diff(x, y) for x, y in zip(plain, full))
        g["R_fullrank_vs_plain_mean_abs_xy_m"] = xy(plain, full)
        g["R_fullrank_vs_sweep_off_max_abs_m"] = max(UP.max_abs_diff(x, y) for x, y in zip(swept, full))
        g["R_fullrank_vs_sweep_off_mean_abs_xy_m"] = xy(swept, full)
    # exact sites: every G line bit-identical. Re-implemented hooked paths (W0Site): G1 / G4 / G5 bit-identical and the
    # hooked path's off arm within G3_TOL_M mean |xy| of the shipped one (10 % of DriveVLA-W0's own unseeded run-to-run floor)
    g["G3_tol_mean_abs_xy_m"] = None if S.exact else G3_TOL_M
    g["passed"] = all(v["bit_identical"] for k, v in g.items() if k.startswith("G") and isinstance(v, dict) and (S.exact or k != "G3_sweep_off_vs_nohook")) \
        and (S.exact or g["G3_sweep_off_vs_nohook"]["mean_abs_xy_m"] <= G3_TOL_M)
    (out_dir(a.model) / ("gate_pca.json" if pcas else "gate.json")).write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))
    if not g["passed"]:
        sys.exit("identity gate failed")


def sweep(probes, S, fn, ks) -> dict:
    """Probe.sweep over several probes of one site: the producer runs once (k = None first), then every k projects each
    probe's cached feature onto min(k, D_i) directions."""
    for p in probes:
        p.mode, p.cache = "cached", None
    S.memo, S.active = None, True
    out = {}
    try:
        for k in ks:
            for i, p in enumerate(probes):
                # an int k projects onto the probe's top-k PCA directions; a (name, fns) arm applies fns[i] to probe i
                p.k = k[1][i] if isinstance(k, tuple) else None if k is None else min(k, p.basis.U.shape[0])
            out[k[0] if isinstance(k, tuple) else k] = fn()
            if k is None:
                out["_pooled"] = pooled(S)
                for p in (p for p in probes if p.basis is not None):
                    for key, v in p.basis.energy(p.cache, [x for x in ks if isinstance(x, int)]).items():
                        p.energy[key] = p.energy.get(key, 0.0) + v
    finally:
        for p in probes:
            p.mode, p.cache, p.k = "off", None, None
        S.active = False
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
        S.active = True
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
        zs = [np.load(x) for x in fs]      # lazy: one site at a time
        inf = dict(shards=len(fs), tokens=sum(int(z["tokens"]) for z in zs), sites=[])
        for i in range(int(zs[0]["n_sites"])):
            pca = UP.Moments.fit(UP.Moments.merge({k: z[f"f{i}_{k}"] for k in ("n", "s", "ss")} for z in zs))
            if a.model in PCA32:
                pca.comps = pca.comps.astype(np.float32)
            pca.save(d / f"pca_fold{f}_f{i}.npz")
            inf["sites"].append(dict(dim=pca.dim, rows=pca.n, explained_fit={f"k{k}": pca.explained(min(k, pca.dim)) for k in ks_for([pca.dim])}))
        info[f"fold{f}"] = inf
    (d / "pca.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    if a.model in PCA32:                 # 4.3 GB per moment shard; the PCA (and pca.json) is all later steps read
        for x in (d / "fit").glob("fold*_m_s*of*.npz"):
            x.unlink()


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
                    B.append(blind_of(agent, L, t))
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


# ---------------------------------------------------------------- projection-artifact controls (variants)
# Same tokens, folds, cross-fitted PCAs and k grid as `sweep`; each token's producer (the LM) runs once and every variant x k
# arm reuses the cached feature (UP.Projector). Outputs under <out_dir>/variants/; bench arms PROBE-<tag>-<variant>-k<k>.
VAR_DEFAULT = ("cnorm", "norm", "rand", "std", "wpca")


def read_in(model: str, agent) -> torch.Tensor:
    """The planner's linear read-in of the site (m x D, fp32 on the card): ReCogDrive's DiT sees the LM hidden state only
    through action_head.feature_encoder = nn.Linear(D, 384) (get_action: vl_embeds = feature_encoder(vl_features))."""
    if model == "ReCogDrive-2B-RL":
        return agent.action_head.feature_encoder.weight.detach().float()
    raise SystemExit(f"no linear read-in known for {model}")


def projectors(model, agent, pcas, variants) -> dict:
    W = read_in(model, agent)
    return {v: [UP.Projector(q, v, "cuda", W=W.double().cpu().numpy()) for q in pcas] for v in variants}, W


def variant_arms(prj: dict, ks) -> list:
    """[(name, [fn per probe])] per variant x k; k grid of the original sweep, clamped to the variant's rank (wpca: k < kmax,
    then kmax)."""
    arms = []
    for v, ps in prj.items():
        km = ps[0].kmax
        for k in [k for k in ks if k < km] + ([km] if km <= max(ks) else []):
            arms.append((f"{v}-k{k}", [lambda x, p=p, k=k: p.project(x, k) for p in ps]))
    return arms


class InputMemo:
    """Variant commands only: per-token one-entry memo of the deterministic input path that the cached sweep arms would
    otherwise redo per arm (the loader's AgentInput; ReCogDrive's InternVL `load_image` tiling, no RNG). Model math is
    untouched; the variant gate re-checks off / full-rank arms against the unhooked model."""

    def __init__(self, model: str, L):
        self.L, self.mods = L, []
        if model == "ReCogDrive-2B-RL":
            self.mods = [sys.modules["navsim.agents.recogdrive.recogdrive_agent"]]

    def __enter__(self):
        if self.L is not None:
            get, m = self.L.get_agent_input_from_token, {}
            self.L.get_agent_input_from_token = lambda t: m[t] if t in m else (m.clear(), m.setdefault(t, get(t)))[1]
        for mod in self.mods:
            li, m2 = mod.load_image, {}
            mod._p1_orig_load_image = li
            mod.load_image = lambda p, *a, _li=li, _m=m2, **kw: _m[p] if p in _m else (_m.clear(), _m.setdefault(p, _li(p, *a, **kw)))[1]
        return self

    def __exit__(self, *exc):
        if self.L is not None:
            del self.L.get_agent_input_from_token
        for mod in self.mods:
            mod.load_image = mod._p1_orig_load_image
        return False


def cmd_vgate(a):
    """Identity gate for the variant run: G1 hook off == no hook, G4 determinism, G3 / G5 sweep arm off == no hook and
    repeatable, P0 Projector('pca') == the original projection bit for bit, R full-rank arm of every variant vs plain."""
    c, cfg, agent = build(a.model)
    toks, logs = navtest_tokens()
    sel = list(range(0, len(toks), max(1, len(toks) // a.n)))[: a.n]
    T = [toks[i] for i in sel]
    L = loader(c, cfg, agent, T, [logs[i] for i in sel])
    pcas = load_pcas(a.model, 0)
    prj, W = projectors(a.model, agent, pcas, a.variants)
    full = [(f"{v}-full", [lambda x, p=p: p.project(x, p.kmax) for p in ps]) for v, ps in prj.items()]
    plain, again, hooked, swept, swept2, R = [], [], [], [], [], {n: [] for n, _ in full}
    p0 = []
    t0 = time.time()
    for t in T:
        plain.append(call(agent, L, t))
        again.append(call(agent, L, t))
    t_plain = (time.time() - t0) / (2 * len(T))
    S = site(a.model, agent)
    t_arm = []
    with S, InputMemo(a.model, L):
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        try:
            for t in T:
                hooked.append(call(agent, L, t, S))
                t1 = time.time()
                r = sweep(probes, S, lambda: call(agent, L, t), [None] + full)
                t_arm.append((time.time() - t1 - t_plain) / len(full))
                swept.append(r[None])
                for n, _ in full:
                    R[n].append(r[n])
                swept2.append(sweep(probes, S, lambda: call(agent, L, t), [None])[None])
                x = S.pick(S.memo)[0]
                b, pp = UP._Basis(pcas[0], "cuda"), UP.Projector(pcas[0], "pca", "cuda")
                p0.append(all(torch.equal(b.project(x, k), pp.project(x, k)) for k in (8, 64, 512, 1024)))
        finally:
            for p in probes:
                p.__exit__()
    xy = lambda A, B: float(np.mean([np.abs(np.asarray(x)[:, :2] - np.asarray(y)[:, :2]).mean() for x, y in zip(A, B)]))  # noqa: E731
    g = dict(model=a.model, n=len(T), variants=list(prj), kmax={v: ps[0].kmax for v, ps in prj.items()}, read_in_shape=list(W.shape),
             s_per_token_plain=t_plain, s_per_cached_arm=float(np.mean(t_arm)),
             G1_off_vs_nohook=UP.identity_gate(plain, hooked), G4_determinism=UP.identity_gate(plain, again),
             G5_sweep_determinism=UP.identity_gate(swept, swept2), G3_sweep_off_vs_nohook=UP.identity_gate(plain, swept),
             P0_projector_pca_equals_original=bool(all(p0)),
             R={n: dict(max_abs_m=max(UP.max_abs_diff(x, y) for x, y in zip(plain, v)), mean_abs_xy_m=xy(plain, v)) for n, v in R.items()})
    g["R_tol_mean_abs_xy_m"] = G3_TOL_M
    g["passed"] = all(g[k]["bit_identical"] for k in ("G1_off_vs_nohook", "G4_determinism", "G5_sweep_determinism", "G3_sweep_off_vs_nohook")) \
        and g["P0_projector_pca_equals_original"] and all(v["mean_abs_xy_m"] <= G3_TOL_M for v in g["R"].values())
    d = out_dir(a.model) / "variants"
    d.mkdir(exist_ok=True)
    (d / "gate.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))
    if not g["passed"]:
        sys.exit("variant identity gate failed")


def cmd_vsweep(a):
    """`sweep` with the variant arms: navtest shard I of N, a token of fold g uses the other fold's PCA; also the
    projection diagnostics (residual energy in input and read-in space, norm ratio) of every variant incl. the original pca."""
    c, cfg, agent = build(a.model)
    toks, logs = navtest_tokens()
    fo = fold_of_logs(logs)
    idx = list(range(a.shard[0], len(toks), a.shard[1]))
    ks = ks_for([q.dim for q in load_pcas(a.model, 0)])
    d = out_dir(a.model) / "variants" / "sweep" / f"s{a.shard[0]}of{a.shard[1]}"
    d.mkdir(parents=True, exist_ok=True)
    S = site(a.model, agent)
    t0, n = time.time(), 0
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        for g in range(FOLDS):
            pcas = load_pcas(a.model, (g + 1) % FOLDS)
            prj, W = projectors(a.model, agent, pcas, a.variants)
            diag = {v: (prj[v][0] if v in prj else UP.Projector(pcas[0], v, "cuda", W=W.double().cpu().numpy())) for v in ("pca", *a.variants)}
            arms = variant_arms(prj, ks)
            names = ["off"] + [n_ for n_, _ in arms]
            for c0 in range(0, len(idx), a.chunk):
                part = [i for i in idx[c0:c0 + a.chunk] if fo[logs[i]] == g]
                f_out = d / f"g{g}_c{c0:06d}.npz"
                if not part or f_out.exists() or (c0 // a.chunk) % a.part[1] != a.part[0]:
                    continue
                T = [toks[i] for i in part]
                L = loader(c, cfg, agent, T, [logs[i] for i in part])
                P, st = [], {}
                for t in T:
                    with InputMemo(a.model, L):
                        r = sweep(probes, S, lambda: call(agent, L, t), [None] + arms)
                    P.append(np.stack([r[None]] + [r[n_] for n_ in names[1:]]))
                    x = S.pick(S.memo)[0]
                    for v, pr in diag.items():
                        kv = [k for k in ks if k < pr.kmax] + ([pr.kmax] if pr.kmax <= max(ks) else [])
                        s = pr.stats(x, kv, W)
                        acc = st.setdefault(v, {})
                        for key, val in s.items():
                            if isinstance(val, dict):
                                for kk, vv in val.items():
                                    acc.setdefault(f"k{key}", {}).setdefault(kk, 0.0)
                                    acc[f"k{key}"][kk] += vv
                            else:
                                acc[key] = acc.get(key, 0.0) + val
                n += len(T)
                np.savez(d / "tmp.npz", tokens=np.array(T), fold=g, poses=np.stack(P, 1), arms=np.array(names), stats=json.dumps(st))
                os.replace(d / "tmp.npz", f_out)
                print(f"[vsweep] fold {g} chunk {c0}: {n}/{len(idx)} {(time.time() - t0) / max(n, 1):.2f} s/token", flush=True)
        for p in probes:
            p.__exit__()
    if a.part[1] == 1:
        (d / "DONE").write_text(json.dumps(dict(tokens=len(idx), wall_s=time.time() - t0)))


def cmd_vmerge(a):
    """-> bench pred files PROBE-<tag>-<variant>-k<k>, variants/ev.json (diagnostics per arm; original arms as k<k>), and the
    check that this run's arm off equals the original sweep's off plans bit for bit."""
    from jevdrive.bench.models import resolve
    from jevdrive.bench.navsim import pred_file
    d = out_dir(a.model) / "variants"
    toks, _ = navtest_tokens()
    zs = [np.load(f) for f in sorted((d / "sweep").glob("s*of*/g*_c*.npz"))]
    names = [str(x) for x in zs[0]["arms"]]
    assert all([str(x) for x in z["arms"]] == names for z in zs), "arm lists differ between folds / chunks"
    T = np.concatenate([z["tokens"] for z in zs])
    P = np.concatenate([z["poses"] for z in zs], 1)
    assert len(set(T)) == len(T) == len(toks) and set(T) == set(toks), f"{len(T)} plans for {len(toks)} navtest tokens"
    o = np.argsort(T)
    T, P = T[o], P[:, o]
    off = P[names.index("off")]
    z0 = np.load(pred_file(resolve(f"PROBE-{TAGS[a.model]}-off"), "navtest"))
    o0 = np.argsort(z0["tokens"])
    assert (z0["tokens"][o0] == T).all()
    same = bool(np.array_equal(z0["poses"][o0], off))
    for i, nm in enumerate(names[1:], 1):
        pf = pred_file(resolve(f"PROBE-{TAGS[a.model]}-{nm}"), "navtest")
        pf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pf, tokens=T, poses=P[i])
    st = {}
    for z in zs:
        for v, acc in json.loads(str(z["stats"])).items():
            tgt = st.setdefault(v, {})
            for key, val in acc.items():
                if isinstance(val, dict):
                    for kk, vv in val.items():
                        tgt.setdefault(key, {}).setdefault(kk, 0.0)
                        tgt[key][kk] += vv
                else:
                    tgt[key] = tgt.get(key, 0.0) + val
    diag = {}
    for v, acc in st.items():
        for key, r in acc.items():
            if isinstance(r, dict):
                diag["k" + key[1:] if v == "pca" else f"{v}-{key}"] = dict(ev_input=1 - r["res2"] / acc["c2"], ev_readin=1 - r["wres2"] / acc["wc2"],
                                                                         norm_ratio=r["ratio"] / acc["rows"])
    shift = {nm: float(np.linalg.norm(P[i][..., :2] - off[..., :2], axis=-1).mean()) for i, nm in enumerate(names)}
    ev = dict(model=a.model, tokens=len(T), arms=names, off_equals_original_off=same, fit="navtest 2-fold log cross-fit (same PCAs as sweep)",
              explained_navtest={k: v["ev_input"] for k, v in diag.items()}, explained_readin={k: v["ev_readin"] for k, v in diag.items()},
              norm_ratio={k: v["norm_ratio"] for k, v in diag.items()}, plan_shift_vs_off_m=shift)
    ev["note"] = ("explained_navtest = 1 - sum ||x - x'||^2 / sum ||x - mu||^2 over held-out rows (input space); explained_readin = the same "
                  "through the planner's read-in W (1 - ||W(x - x')||^2 / ||W(x - mu)||^2); norm_ratio = mean ||x'|| / ||x|| per row; "
                  "keys k<k> = the original pca arms (diagnostics only, plans from the original sweep)")
    (d / "ev.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps({k: ev[k] for k in ("tokens", "off_equals_original_off")}, indent=1))
    if not same:
        sys.exit("variant run's off arm differs from the original sweep's off arm")


# ---------------------------------------------------------------- DriveVLA-W0 controls (prereg addendum A)
# On the fixed token subset navsim/p1-w0-ctl: 32-layer projection with the raw norm restored per row (V2 `norm`), and a
# single layer projected (original projection, the other 31 layers untouched). Same cross-fitted PCAs, folds and per-token
# noise seed as `sweep`; outputs under <out_dir>/controls/; bench arms PROBE-w0-vlmkv-<arm> (subset tokens only).
W0_SPLIT = "navsim/p1-w0-ctl"
W0_CTL = (("norm", 64), ("norm", 512), ("norm", 2048), ("L16", 64), ("L16", 512), ("L31", 64), ("L31", 512))


def norm_project(b: "UP._Basis", x: torch.Tensor, k: int) -> torch.Tensor:
    """Projector('norm') on a _Basis: mu + c U_k U_k^T in fp32, rescaled to the row's original ||x||, cast back."""
    xf = x.float()
    U = b.U[:, :k]
    y = ((xf - b.mu) @ U) @ U.T + b.mu
    return (y * (xf.norm(dim=-1, keepdim=True) / y.norm(dim=-1, keepdim=True).clamp_min(1e-12))).to(x.dtype)


def w0_arms(probes, ctl) -> list:
    """[(name, [fn or None per layer probe])]; probes' current bases (the fold's PCA) are read at call time."""
    arms = []
    for v, k in ctl:
        if v == "norm":
            fns = [lambda x, p=p, k=k: norm_project(p.basis, x, min(k, p.basis.U.shape[0])) for p in probes]
        else:
            fns = [None] * len(probes)
            p = probes[int(v[1:])]
            fns[int(v[1:])] = lambda x, p=p, k=k: p.basis.project(x, min(k, p.basis.U.shape[0]))
        arms.append((f"{v}-k{k}", fns))
    return arms


def w0_subset():
    from jevdrive.data import splits
    toks, logs = navtest_tokens()
    keep = set(splits.load(W0_SPLIT))
    sel = [i for i, t in enumerate(toks) if t in keep]
    assert len(sel) == len(keep)
    return toks, logs, sel


def cmd_wcgate(a):
    """Identity gate of the W0 controls on --n subset tokens: G4 plain twice, G5 sweep off twice (bit-identical), G3 sweep
    off vs plain (mean |xy| <= G3_TOL_M), R full-rank control arms vs sweep off (mean |xy| <= G3_TOL_M), O off vs the
    original sweep's off plans (reported)."""
    from jevdrive.bench.models import resolve
    from jevdrive.bench.navsim import pred_file
    c, cfg, agent = build(a.model)
    toks, logs, sel = w0_subset()
    sel = sel[:: max(1, len(sel) // a.n)][: a.n]
    T = [toks[i] for i in sel]
    fo = fold_of_logs(logs)
    plain, again, swept, swept2 = [], [], [], []
    for t in T:
        plain.append(call(agent, L=None, tok=t))
        again.append(call(agent, L=None, tok=t))
    S = site(a.model, agent)
    full = [("norm", 4096), ("L16", 4096), ("L31", 4096)]
    R = {f"{v}-k{k}": [] for v, k in full}
    t0 = time.time()
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        cur = None
        for i, t in zip(sel, T):
            g = fo[logs[i]]
            if g != cur:
                for p, q in zip(probes, load_pcas(a.model, (g + 1) % FOLDS)):
                    p.use_pca(q, "cuda")
                cur = g
            r = sweep(probes, S, lambda: call(agent, None, t), [None] + w0_arms(probes, full))
            swept.append(r[None])
            for n_ in R:
                R[n_].append(r[n_])
            swept2.append(sweep(probes, S, lambda: call(agent, None, t), [None])[None])
        for p in probes:
            p.__exit__()
    xy = lambda A, B: float(np.mean([np.abs(np.asarray(x)[:, :2] - np.asarray(y)[:, :2]).mean() for x, y in zip(A, B)]))  # noqa: E731
    z0 = np.load(pred_file(resolve(f"PROBE-{TAGS[a.model]}-off"), "navtest"))
    pos = {t: j for j, t in enumerate(z0["tokens"].tolist())}
    orig = [z0["poses"][pos[t]] for t in T]
    g = dict(model=a.model, split=W0_SPLIT, n=len(T), s_per_token_sweep=(time.time() - t0) / len(T),
             G4_determinism=UP.identity_gate(plain, again), G5_sweep_determinism=UP.identity_gate(swept, swept2),
             G3_sweep_off_vs_nohook=UP.identity_gate(plain, swept), O_sweep_off_vs_original_off=UP.identity_gate(orig, swept),
             R={n_: dict(max_abs_m=max(UP.max_abs_diff(x, y) for x, y in zip(swept, v)), mean_abs_xy_m=xy(swept, v)) for n_, v in R.items()},
             tol_mean_abs_xy_m=G3_TOL_M)
    g["G3_sweep_off_vs_nohook"]["mean_abs_xy_m"] = xy(plain, swept)
    g["O_sweep_off_vs_original_off"]["mean_abs_xy_m"] = xy(orig, swept)
    g["passed"] = g["G4_determinism"]["bit_identical"] and g["G5_sweep_determinism"]["bit_identical"] \
        and g["G3_sweep_off_vs_nohook"]["mean_abs_xy_m"] <= G3_TOL_M and all(v["mean_abs_xy_m"] <= G3_TOL_M for v in g["R"].values())
    d = out_dir(a.model) / "controls"
    d.mkdir(exist_ok=True)
    (d / "gate.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))
    if not g["passed"]:
        sys.exit("W0 control identity gate failed")


def cmd_wcsweep(a):
    """Subset shard I of N: off + the control arms; a token of fold g uses the other fold's PCA (as `sweep`)."""
    c, cfg, agent = build(a.model)
    toks, logs, sel = w0_subset()
    fo = fold_of_logs(logs)
    idx = sel[a.shard[0]:: a.shard[1]]
    d = out_dir(a.model) / "controls" / "sweep" / f"s{a.shard[0]}of{a.shard[1]}"
    d.mkdir(parents=True, exist_ok=True)
    S = site(a.model, agent)
    t0, n = time.time(), 0
    with S:
        probes = [UP.Probe(S, f"f{i}").__enter__() for i in range(S.n)]
        arms = w0_arms(probes, W0_CTL)
        names = ["ctl-off"] + [n_ for n_, _ in arms]
        for g in range(FOLDS):
            loaded = False
            for c0 in range(0, len(idx), a.chunk):
                part = [i for i in idx[c0:c0 + a.chunk] if fo[logs[i]] == g]
                f_out = d / f"g{g}_c{c0:06d}.npz"
                if not part or f_out.exists():
                    continue
                if not loaded:
                    for p, q in zip(probes, load_pcas(a.model, (g + 1) % FOLDS)):
                        p.use_pca(q, "cuda")
                    loaded = True
                T = [toks[i] for i in part]
                P = []
                for t in T:
                    r = sweep(probes, S, lambda: call(agent, None, t), [None] + arms)
                    P.append(np.stack([r[None]] + [r[n_] for n_ in names[1:]]))
                n += len(T)
                np.savez(d / "tmp.npz", tokens=np.array(T), fold=g, poses=np.stack(P, 1), arms=np.array(names))
                os.replace(d / "tmp.npz", f_out)
                print(f"[wcsweep] fold {g} chunk {c0}: {n}/{len(idx)} {(time.time() - t0) / max(n, 1):.2f} s/token", flush=True)
        for p in probes:
            p.__exit__()
    (d / "DONE").write_text(json.dumps(dict(tokens=len(idx), wall_s=time.time() - t0)))


def cmd_wcmerge(a):
    """-> bench pred files PROBE-w0-vlmkv-<arm> (subset tokens), controls/ev.json (plan shift vs the original off, off identity)."""
    from jevdrive.bench.models import resolve
    from jevdrive.bench.navsim import pred_file
    d = out_dir(a.model) / "controls"
    toks, _, sel = w0_subset()
    zs = [np.load(f) for f in sorted((d / "sweep").glob("s*of*/g*_c*.npz"))]
    names = [str(x) for x in zs[0]["arms"]]
    T = np.concatenate([z["tokens"] for z in zs])
    P = np.concatenate([z["poses"] for z in zs], 1)
    assert len(set(T)) == len(T) == len(sel) and set(T) == {toks[i] for i in sel}, f"{len(T)} plans for {len(sel)} subset tokens"
    o = np.argsort(T)
    T, P = T[o], P[:, o]
    z0 = np.load(pred_file(resolve(f"PROBE-{TAGS[a.model]}-off"), "navtest"))
    pos = {t: j for j, t in enumerate(z0["tokens"].tolist())}
    off0 = z0["poses"][[pos[t] for t in T]]
    for i, nm in enumerate(names):
        pf = pred_file(resolve(f"PROBE-{TAGS[a.model]}-{nm}"), "navtest")
        pf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pf, tokens=T, poses=P[i])
    shift = {nm: float(np.linalg.norm(P[i][..., :2] - off0[..., :2], axis=-1).mean()) for i, nm in enumerate(names)}
    for k in ("k64", "k512", "k2048"):
        zk = np.load(pred_file(resolve(f"PROBE-{TAGS[a.model]}-{k}"), "navtest"))
        pk = {t: j for j, t in enumerate(zk["tokens"].tolist())}
        shift[k] = float(np.linalg.norm(zk["poses"][[pk[t] for t in T]][..., :2] - off0[..., :2], axis=-1).mean())
    ev = dict(model=a.model, split=W0_SPLIT, tokens=len(T), arms=names, ctl_off_equals_original_off=bool(np.array_equal(P[0], off0)),
              ctl_off_max_abs_m=float(np.abs(P[0] - off0).max()), plan_shift_vs_off_m=shift)
    (d / "ev.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "fit", "pca", "sweep", "merge", "vgate", "vsweep", "vmerge", "wcgate", "wcsweep", "wcmerge"])
    ap.add_argument("--variants", nargs="+", default=list(VAR_DEFAULT), choices=list(UP.VARIANTS[1:]))
    ap.add_argument("--model", required=True, choices=sorted(TAGS))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pca", action="store_true", help="gate: also G3 / full-rank checks with the fold-0 PCA")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--max", type=int, default=3000, help="fit tokens per fold (spread over the fold's logs)")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--part", type=int, nargs=2, default=(0, 1), help="vsweep: only chunks j of every m of the shard (same files)")
    a = ap.parse_args()
    from jevdrive.run import Run
    tag = f"p1a-{TAGS[a.model]}-{a.cmd}" + (f"-f{a.fold}" if a.cmd == "fit" else "") + (f"-s{a.shard[0]}of{a.shard[1]}" if a.cmd in ("fit", "sweep", "vsweep", "wcsweep") else "") \
        + (f"-p{a.part[0]}of{a.part[1]}" if a.part[1] > 1 else "")
    with Run("wm_surplus", tag, config=vars(a)) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("navsim/navtest"))
        globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
