"""Timed Alpamayo 1.5 inference with config knobs applied from outside the model code.

Stage timings come from forward hooks with CUDA syncs around four modules:
  vision   vlm.model.visual                  (all image calls)
  prefill  vlm.model.language_model, call 0  (prompt forward inside generate)
  decode   vlm.model.language_model, calls 1+ (one per reasoning token)
  flow     expert                            (one per flow-matching step)
`other` = wall - sum(stages): generate bookkeeping, sampling, trajectory integration.
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

REPO = "nvidia/Alpamayo-1.5-10B"


@dataclass
class Config:
    name: str = "default"
    attn: str = "flash_attention_2"   # VLM attention; the expert always uses sdpa (FA2 unsupported there)
    n_samples: int = 1                # trajectories (= reasoning rollouts) per input
    max_gen: int = 256                # reasoning token cap (the shipped example uses 256)
    flow_steps: int = 10              # Euler steps of the flow-matching expert (model default 10)
    no_reasoning: bool = False        # prefill an empty CoT so the model goes straight to the trajectory
    expert_graph: bool = False        # the repo's own exact-shape CUDA graphs for the expert
    compile: tuple = ()               # submodules to torch.compile: "visual", "expert", "lm"
    static_cache: bool = False        # generation_config.cache_implementation = "static"
    top_p: float = 0.98
    temperature: float = 0.6


def versions() -> dict:
    import subprocess
    import transformers
    try:
        import flash_attn
        fa = flash_attn.__version__
    except ImportError:
        fa = None
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
            "flash_attn": fa, "cudnn": torch.backends.cudnn.version(), "driver_gpu": drv}


def hub_offline(on: bool = True):
    """Flip HF hub + transformers to offline at runtime (both cache the env flag at import). Needed because the
    model builds its tokenizer/config from nvidia/Cosmos-Reason2-8B, whose gate is not accepted for our account:
    online, transformers' probe for the absent processor_config.json gets a 403 and raises. Offline, it reads the
    files we fetched from ModelScope (scripts/alpamayo_fetch.py backbone-config) and the .no_exist markers."""
    import huggingface_hub.constants
    import transformers.utils.hub
    huggingface_hub.constants.HF_HUB_OFFLINE = on
    transformers.utils.hub._is_offline_mode = on


def load(attn: str = "flash_attention_2"):
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    hub_offline(True)
    try:
        model = Alpamayo1_5.from_pretrained(REPO, dtype=torch.bfloat16, attn_implementation=attn).to("cuda").eval()
        return model, helper.get_processor(model.tokenizer)
    finally:
        hub_offline(False)


def build_inputs(data: dict, processor, no_reasoning: bool = False, nav_text: str | None = None) -> dict:
    """Same message/tokenization as the shipped test_inference.py; `no_reasoning` closes the CoT in the prompt."""
    from alpamayo1_5 import helper
    msgs = helper.create_message(frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"],
                                 nav_text=nav_text)
    if no_reasoning:
        msgs[-1]["content"][0]["text"] = "<|cot_start|><|cot_end|>"
    tok = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                        return_dict=True, return_tensors="pt")
    return helper.to_device({"tokenized_data": tok, "ego_history_xyz": data["ego_history_xyz"],
                        "ego_history_rot": data["ego_history_rot"]}, "cuda")


class StageTimer:
    def __init__(self, model):
        self.mods = _mods(model)
        self.handles, self.calls = [], {k: [] for k in self.mods}
        for k, m in self.mods.items():
            self.handles += [m.register_forward_pre_hook(self._pre(k)), m.register_forward_hook(self._post(k))]
        self.on = False

    def _pre(self, k):
        def f(*_):
            if self.on:
                torch.cuda.synchronize()
                self._t0 = time.perf_counter()
        return f

    def _post(self, k):
        def f(*_):
            if self.on:
                torch.cuda.synchronize()
                self.calls[k].append(time.perf_counter() - self._t0)
        return f

    @contextmanager
    def record(self):
        self.calls = {k: [] for k in self.mods}
        self.on = True
        try:
            yield self
        finally:
            self.on = False

    def stages(self) -> dict:
        lm = self.calls["lm"]
        return {"vision": sum(self.calls["visual"]), "prefill": lm[0] if lm else 0.0, "decode": sum(lm[1:]),
                "flow": sum(self.calls["expert"]), "n_decode": max(len(lm) - 1, 0),
                "n_flow_calls": len(self.calls["expert"])}


def apply(model, cfg: Config):
    """Install the knobs that live on the model; reset() undoes them, so rows can share one loaded model."""
    reset(model)
    if cfg.expert_graph:
        model.enable_diffusion_expert_cuda_graph(max_batch_size=max(16, cfg.n_samples), max_graphs=8)
    for name in cfg.compile:
        mod = _mods(model)[name]
        mod._jev_forward = mod.forward
        mod.forward = torch.compile(mod.forward, dynamic=True)
    model.vlm.generation_config.cache_implementation = "static" if cfg.static_cache else None


def reset(model):
    g = model._diffusion_expert_cuda_graph
    if g is not None:
        model.expert.forward, model._diffusion_expert_cuda_graph = g._original_forward, None
    for mod in _mods(model).values():
        if hasattr(mod, "_jev_forward"):
            mod.forward = mod._jev_forward
            del mod._jev_forward
    model.vlm.generation_config.cache_implementation = None
    torch.cuda.empty_cache()


def _mods(model) -> dict:
    return {"visual": model.vlm.model.visual, "expert": model.expert, "lm": model.vlm.model.language_model}


def run(model, inputs: dict, cfg: Config, timer: StageTimer, seed: int = 42) -> dict:
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    with timer.record(), torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        t0 = time.perf_counter()
        xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inputs, top_p=cfg.top_p, temperature=cfg.temperature, num_traj_samples=cfg.n_samples,
            max_generation_length=cfg.max_gen, return_extra=True,
            diffusion_kwargs={"inference_step": cfg.flow_steps})
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    st = timer.stages()
    st["other"] = wall - st["vision"] - st["prefill"] - st["decode"] - st["flow"]
    return {"wall": wall, **st, "xyz": xyz[0, 0].float().cpu().numpy(),  # (n_samples, 64, 3)
            "cot": [str(c) for c in extra["cot"][0, 0]]}


def ade(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """Per-sample ADE over the 64 future waypoints. pred (n, 64, 2), gt (64, 2) -> (n,)."""
    return np.linalg.norm(pred_xy - gt_xy[None], axis=-1).mean(-1)
