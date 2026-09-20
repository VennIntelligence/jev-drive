# Baselines

Read this when you need a competitor's latency on our card, want to re-run one, or wonder why a baseline is missing.

All numbers: RTX PRO 6000 Blackwell (sm_120), batch 1, serial requests, the release's own code, weights,
precision and preprocessing, no tuning. 20 warmup + 200 timed requests, each bracketed by `torch.cuda.synchronize()`
(openjev: one blocking HTTP round trip). Same input everywhere: the WOD-E2E demo scene
`74cf0a3e1a537277b1258a683de98b4f-149` that ships with Qwen-Drive (3 front cameras x 4 frames, JPEG on disk).
CPU threads: `OMP_NUM_THREADS=4` (the box is shared; torch would otherwise size its pool from the host's 208 cores).
Raw outputs: `$DATA_DIR/runs/bench_baselines/<model>/<tag>/<time>/{summary,times_ms}.json`.
Stage breakdowns are tags of the same run (`stage-vision`, `stage-vision-prefill`, `stage-flow-expert`, `preprocess-only`);
`--only stages` re-runs just those.

## Results

| Model / mode | Source (commit) | License, gating | Runs | Venv | Mean / p50 / p95 ms | Peak VRAM | Timed span | Notes |
|---|---|---|---|---|---:|---|---|---|
| Qwen-Drive-1.0-4B, direct planning, planner-sft, 1 sample | [QwenLM/Qwen-Drive-1.0](https://github.com/QwenLM/Qwen-Drive-1.0) `28091c1`; weights `Qwen/Qwen-Drive-1.0-4B` rev `2848408` (ModelScope copy, sha256 = HF) | Apache-2.0, not gated | yes | `envs/qwen-drive` | 702 / 699 / 746 | 12.3 GB alloc, 13.0 GB reserved | `model.run()`: JPEG decode + resize + tokenize on CPU (214 ms), vision encoder on 12 frames (61 ms), LLM prefill of 3383 tokens (120 ms), 10 flow-matching steps (262 ms), 0 generated tokens | bf16, flash_attention_2, torch 2.8.0 cu128 as pinned (already has sm_120) |
| same, 6 samples | same | same | yes | same | 688 / 686 / 705 | same | same, 6 trajectories batched in the expert | cost of extra samples is within noise |
| Qwen-Drive, reasoning planning, planner-rl, 1 sample | same | same | yes | same | 1258 / 1249 / 1329 | 11.2 GB alloc | as direct, plus greedy decode of 18 tokens (think block + 13-token rationale, ~30 ms/token) and the turn-closing forward | the mode the card recommends for planner-rl; rationale: "Turn left at the clear intersection and accelerate to the target speed." |
| same, 6 samples | same | same | yes | same | 1245 / 1243 / 1259 | same | same | |
| Qwen-Drive, reasoning planning, planner-sft, 1 sample | same | same | yes | same | 1327 / 1243 / 1690 | same | same | p95 tail from CPU contention with other jobs on the box |
| openjev (DiffusionGemma-26B-A4B NVFP4), 3 front cameras + 3 driving questions, fresh frames | [razorback16/openjev](https://github.com/razorback16/openjev) `91d5005` + vLLM fork `razorback16/vllm` `9bbf741` (precompiled kernels of `2c88fb1`); weights `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev `ec4ff3d` | Apache-2.0 (code and weights), not gated | yes | `envs/openjev` | 333 / 300 / 482 | vLLM preallocates 0.9 x 96 GB (86.8 GB used); weights ~18 GB | one HTTP round trip to `/v1/systemone` on localhost: base64 decode, Gemma image preprocessing, vision tower, prefill of 1015 input tokens, one read-only denoise step of a 64-token canvas, plus 3 parallel re-reads when any answer's entropy > 0.1 | not a planner: answers 2 choice + 1 yes/no questions, current frames only, no history. Bimodal: 163/200 requests at ~299 ms (one read), the rest ~479 ms (re-reads) |
| openjev, same, identical request repeated | same | same | yes | same | 159 / 158 / 167 | same | same, but the vLLM prefix cache holds the images | not representative of driving (every frame is new) |
| openjev, front camera only, fresh frames | same | same | yes | same | 230 / 181 / 352 | same | same, 501 input tokens | 134/200 at ~177 ms, rest ~338 ms |
| openjev, README text benchmark (3 text questions) | same | same | yes | same | 174 / 208 / 225 | same | same, 171 input tokens, no image | fast mode (57/200, one read) 74 ms matches the README's 94 ms p50; most requests trigger re-reads here |
| AutoVLA (Qwen2.5-VL-3B + action tokens), adaptive-CoT prompt and fast-thinking prompt | [ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA) `ba34eed`; weights `Zewei-Zhou/AutoVLA` `AutoVLA_PDMS_89.ckpt` (NAVSIM, RFT/GRPO), base `Qwen/Qwen2.5-VL-3B-Instruct` | UCLA Academic Software License: academic / non-profit use only, no redistribution of derivatives (the user confirmed our use is academic); weights not gated | venv and entry point ready, waiting for the checkpoint (16 GB, in the box's download queue) | `envs/autovla` | pending | pending | `AutoVLA.predict()`: JPEG decode + resize (qwen_vl_utils, CPU), Qwen2.5-VL vision tower, prefill, decode of the optional chain-of-thought and the action tokens, detokenizing to a trajectory | `scripts/bench_baselines/autovla.py` is a minimal single-sample entry point (the release ships only a NAVSIM agent); it builds the same feature dict their feature builder produces and stubs `models.utils.score`, which drags in navsim + nuplan-devkit and is only used by the GRPO reward |
| Qwen/Qwen3-VL-8B-Instruct (bf16) | HF rev `0c351dd` | Apache-2.0, not gated | yes (load check only) | `envs/jevdrive` (ours) | not benchmarked | 16.7 GB alloc | one 40-token caption of one camera frame | our own feature backbone: `uv run python scripts/bench_baselines/load_backbones.py <repo>` (needs `HF_HUB_OFFLINE=1`) |
| facebook/vjepa2-vitl-fpc64-256, google/siglip2-so400m-patch14-384 | HF | Apache-2.0 / Apache-2.0, not gated | load check pending (queued download) | `envs/jevdrive` (ours) | not benchmarked | | one forward on the demo frames | backbone controls for our own pipeline, not competitors: V-JEPA 2 is video-pretrained (can one frame answer what happens before a maneuver starts?) and SigLIP2 separates language alignment from pure vision. They replace DINOv3, see decision 12 |
| Qwen/Qwen3-VL-32B-Instruct (bf16) | HF rev `0cfaf48` | Apache-2.0, not gated | download still running (~66 GB at ~1.7 MB/s; the box link is shared) | same | not benchmarked | | same | the load check runs by itself when the download ends (tmux `jev:bl-backbone-check`) and writes `$DATA_DIR/runs/bench_baselines/backbones/load_check_32b.txt` |

Sanity check, not latency: the bundled `scripts/demo.py` on the same scene gives VQA text, 6 direct and 6
reasoning trajectories, ADE 0.225 m / FDE 0.992 m (direct) and 0.242 m / 1.072 m (reasoning) against the logged
future. Output in `$DATA_DIR/runs/bench_baselines/qwen-drive-1.0-4b/demo/demo.txt`.

What to read from it: Qwen-Drive's as-released pipeline costs ~0.7 s without reasoning and ~1.25 s with it,
and openjev's single read over three camera images costs ~0.3 s; both are far from a 10 Hz budget as released.

Obvious inefficiencies seen (not fixed; the numbers above are as released; speedups are rough estimates, not measured):
- Qwen-Drive CPU preprocessing is 214 ms of the 700 (PIL decode and two bicubic resizes of 12 frames on CPU).
  GPU decode and resize would take ~20-30 ms: about -180 ms.
- Qwen-Drive's flow expert takes 262 ms for 10 steps of a 32-layer, 1024-wide expert over ~66 query tokens.
  It is eager PyTorch at batch 1 and launch-bound; CUDA graphs or `torch.compile` would likely bring it to 30-50 ms.
- Qwen-Drive's reasoning decode runs at ~30 ms/token through eager HF `generate`. A CUDA-graph decoder (vLLM, SGLang,
  static cache + compile) typically does 5-10 ms/token for a 4B model: about -400 ms in reasoning mode.
  Together: roughly 0.25 s direct and 0.35 s reasoning.
- openjev: latency is mostly the number of reads, which the server's re-read policy picks (`OPENJEV_AUTO_MAX`,
  `OPENJEV_AUTO_THRESHOLD`). vLLM's scheduler and the async API are CPU-heavy; the box's CPU was shared with
  two data jobs during the run, which may inflate the tails.

## Not run

| Item | Status | Why |
|---|---|---|
| RAP (2510.04333) | code public ([vita-epfl/RAP](https://github.com/vita-epfl/RAP) `5fd8630`, Apache-2.0), checkpoints `Lanl11/RAP_ckpts` (Waymo 10.6 GB, not gated) | **Not runnable for us, cited only.** Its encoder initialises from `facebook/dinov3-vith16plus-pretrain-lvd1689m`; our access request was rejected by the authors and HF does not allow resubmitting, and we do not route around a deliberate denial through re-uploads (decision 12 in `research/decisions.md`). We quote its published test scores instead. It also ships no single-sample entry point and pins torch 2.1.0 cu121 / mmcv 2.1.0, which would need a rebuild for sm_120. |
| Poutine (2506.11234) | no code, no weights found | nothing public (arXiv, author pages, GitHub, HF). |
| FROST-Drive (2601.03460) | no code, no weights found | nothing public. |
| MindVLA-U1 (2605.12624) | no code, no weights found | project page mind-omni.github.io has no code or download links. |
| hr98w/jev-visual (`19af545`, MIT) | Apple MLX only | depends on `mlx` / `mlx-vlm` with 4-bit MLX weights; no torch or CUDA path. Would need a port. |

## Still open

- AutoVLA: checkpoint downloading (queued behind NAVSIM on the box's shared 12-18 MB/s link); then
  `$DATA_DIR/envs/autovla/bin/python scripts/bench_baselines/autovla.py` gives both modes.
- Qwen3-VL-32B: same queue; its load check needs ~66 GB of VRAM, so run it when the card is free.
- V-JEPA 2 and SigLIP2: queued behind it, then a load check in our venv (no latency run, they are controls).
- The openjev rows were measured with a placeholder ego state in the prompt ("8.2 m/s, go straight");
  the committed client now uses the demo scene's own state ("0.06 m/s, turn left"), two words apart and the same
  token count to within a few tokens. Re-run when the card is free if you want the numbers to match the script exactly.

## Reproduce

- Code in `$DATA_DIR/third_party/<name>` at the commit above, venv in `$DATA_DIR/envs/<name>`,
  built by `scripts/bench_baselines/setup_<name>.sh`. Weights: `scripts/download_models.sh`.
- Run inside tmux (see [long-runs.md](long-runs.md)):
  - `scripts/tmux_run.sh bl-qd env OMP_NUM_THREADS=4 $DATA_DIR/envs/qwen-drive/bin/python scripts/bench_baselines/qwen-drive.py`
  - `scripts/tmux_run.sh bl-openjev scripts/bench_baselines/openjev.sh`
  - `scripts/tmux_run.sh bl-autovla env OMP_NUM_THREADS=4 $DATA_DIR/envs/autovla/bin/python scripts/bench_baselines/autovla.py`
- `scripts/bench_baselines/_bench.py` is the shared timer (stdlib + torch, so it runs in every venv).

Last verified: 2026-09-20
