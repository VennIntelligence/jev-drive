# Baselines

Read this when you need a competitor's latency on our card, want to re-run one, or wonder why a baseline is missing.

Protocol, shared by every row: one RTX PRO 6000 Blackwell (sm_120), **batch 1**, serial requests, the release's own
code, weights, precision and preprocessing, no tuning. 20 warmup then **200 timed** requests, each bracketed by
`torch.cuda.synchronize()` (openjev: one blocking HTTP round trip); the table reports **mean / p50 / p95** in ms,
peak VRAM of the process, and the timed span in its own column. Same input everywhere: the WOD-E2E demo scene
`74cf0a3e1a537277b1258a683de98b4f-149` that ships with Qwen-Drive (3 front cameras x 4 frames of JPEG on disk,
its logged ego state of 0.06 m/s and its route command, turn left).
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
| AutoVLA (Qwen2.5-VL-3B + action tokens), released adaptive-CoT prompt | [ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA) `ba34eed`; weights `Zewei-Zhou/AutoVLA` `AutoVLA_PDMS_89.ckpt` (NAVSIM, RFT/GRPO), base `Qwen/Qwen2.5-VL-3B-Instruct` | UCLA Academic Software License: academic / non-profit only, no redistribution of derivatives (our use is academic); weights not gated | yes | `envs/autovla` | 1362 / 1355 / 1418 | 8.6 GB alloc | `AutoVLA.predict()`: JPEG decode and resize of 3 x 4 frames (qwen_vl_utils, CPU), Qwen2.5-VL vision tower, prefill of 1281 tokens (792 of them image tokens, 3 clips at grid 2x24x22), decode of 44 tokens, detokenizing the action tokens to a 10-point trajectory | torch 2.8.0+cu128 instead of their 2.4.0 (sm_120), everything else at their pins. On this scene the adaptive model writes "This is a straightforward scenario, and a direct decision can be made" and skips the chain of thought, so this row is its **fast path**; a scene that triggers real reasoning decodes hundreds of tokens instead of 44, at ~30 ms/token |
| AutoVLA, fast-thinking prompt (their no-CoT system prompt) | same | same | yes | same | 1407 / 1368 / 1719 | same | same, prefill of 1013 tokens (shorter system prompt) | the checkpoint is the CoT-trained RFT model, so it still emits the same short think-then-answer block: on this scene the two prompts differ by 268 prefill tokens and nothing else, which is why the latency is the same to within noise |
| openjev (DiffusionGemma-26B-A4B NVFP4), 3 front cameras + 3 driving questions, fresh frames | [razorback16/openjev](https://github.com/razorback16/openjev) `91d5005` + vLLM fork `razorback16/vllm` `9bbf741` (precompiled kernels of `2c88fb1`); weights `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev `ec4ff3d` | Apache-2.0 (code and weights), not gated | yes | `envs/openjev` | 471 / 470 / 502 | vLLM preallocates 0.9 x 96 GB (87.0 GB used); weights ~18 GB | one HTTP round trip to `/v1/systemone` on localhost: base64 decode, Gemma image preprocessing, vision tower, prefill of 1016 input tokens, one read-only denoise step of a 64-token canvas, plus the automatic re-reads when an answer's entropy > 0.1 | not a planner: answers 2 choice + 1 yes/no questions, current frames only, no history. Answers `turn_left` (confidence 0.99, the scene's route command) but is unsure about the longitudinal action (0.10), so every request re-reads: the latency is tight, not bimodal |
| openjev, same, identical request repeated | same | same | yes | same | 347 / 358 / 385 | same | same, but the vLLM prefix cache holds the images | not representative of driving (every frame is new) |
| openjev, front camera only, fresh frames | same | same | yes | same | 345 / 345 / 371 | same | same, 502 input tokens | |
| openjev, README text benchmark (3 text questions) | same | same | yes | same | 147 / 151 / 215 | same | same, 171 input tokens, no image | the 24/200 requests that need one read only take 47 ms; the README quotes 94 ms p50 on the same card |
| Qwen/Qwen3-VL-32B-Instruct (bf16) | HF rev `0cfaf48` | Apache-2.0, not gated | yes (load check only) | `envs/jevdrive` (ours) | not benchmarked | 62.7 GB alloc | load plus one 40-token caption of one frame | fits the 96 GB card at bf16 with ~33 GB to spare; 37 s from cold cache to caption |
| facebook/vjepa2-vitl-fpc64-256 (bf16) | HF rev `b3c1679` | Apache-2.0, not gated | yes (load check only) | same | not benchmarked | 0.7 GB alloc | one forward of a 4-frame clip | video control, see "Backbone controls" below: 48 ms per 4-frame clip (12 ms/frame), 512 tokens x 1024 |
| google/siglip2-so400m-patch14-384 (bf16) | HF rev `e8e4872` | Apache-2.0, not gated | yes (load check only) | same | not benchmarked | 2.2 GB alloc | one forward of one frame | image-text control: 12.4 ms/frame, 729 tokens x 1152 |
| facebook/dinov2-base (bf16) | HF | Apache-2.0, not gated | yes (load check only) | same | not benchmarked | 0.2 GB alloc | one forward of one frame | single-frame self-supervised control: 3.1 ms/frame, 257 tokens x 768 |
| Qwen/Qwen3-VL-8B-Instruct (bf16) | HF rev `0c351dd` | Apache-2.0, not gated | yes (load check only) | `envs/jevdrive` (ours) | not benchmarked | 16.7 GB alloc | one 40-token caption of one camera frame | our own feature backbone: `uv run python scripts/bench_baselines/load_backbones.py <repo>` (needs `HF_HUB_OFFLINE=1`) |

Sanity check, not latency: the bundled `scripts/demo.py` on the same scene gives VQA text, 6 direct and 6
reasoning trajectories, ADE 0.225 m / FDE 0.992 m (direct) and 0.242 m / 1.072 m (reasoning) against the logged
future. Output in `$DATA_DIR/runs/bench_baselines/qwen-drive-1.0-4b/demo/demo.txt`.

What to read from it: nothing released here is close to a 10 Hz budget as it ships. Qwen-Drive costs ~0.70 s without
reasoning and ~1.26 s with it, AutoVLA ~1.36 s even when it decides not to reason, and openjev ~0.47 s for three
categorical decisions that are not a trajectory at all. Two second-order points matter as much as the ranking:
**latency is input-dependent** (openjev goes 333 -> 471 ms when the ego state makes an answer uncertain and the
server re-reads; AutoVLA would go from 44 decoded tokens to hundreds on a scene it decides to think about), and
**the cost is in the generic parts**, not in the planning head: preprocessing, the vision tower, prefill, and
token-by-token decode through eager `generate`.

Obvious inefficiencies seen (not fixed; the numbers above are as released; speedups are rough estimates, not measured):
- Qwen-Drive CPU preprocessing is 214 ms of the 700 (PIL decode and two bicubic resizes of 12 frames on CPU).
  GPU decode and resize would take ~20-30 ms: about -180 ms.
- Qwen-Drive's flow expert takes 262 ms for 10 steps of a 32-layer, 1024-wide expert over ~66 query tokens.
  It is eager PyTorch at batch 1 and launch-bound; CUDA graphs or `torch.compile` would likely bring it to 30-50 ms.
- Qwen-Drive's reasoning decode runs at ~30 ms/token through eager HF `generate`. A CUDA-graph decoder (vLLM, SGLang,
  static cache + compile) typically does 5-10 ms/token for a 4B model: about -400 ms in reasoning mode.
  Together: roughly 0.25 s direct and 0.35 s reasoning.
- openjev: latency is mostly the number of reads, which the server's re-read policy picks (`OPENJEV_AUTO_MAX`,
  `OPENJEV_AUTO_THRESHOLD`); on this scene one answer stays uncertain, so every request pays four reads.
  A caller that accepts a single read would see roughly the 1-read cost (47 ms on the text benchmark).
  vLLM's scheduler and the async API are also CPU-heavy, and the box's CPU is shared with the dataset jobs.

## Backbone controls: how to feed them

These three are not competitors, they are the controls for our own feature pipeline (they replace DINOv3, see
decision 12 in `research/decisions.md`). Measured with `scripts/bench_baselines/load_backbones.py` on the same demo
frames, bf16, batch 1, after 3 warmups.

| Backbone | Input tensor | Preprocessing | Exposed features | Cost |
|---|---|---|---|---|
| `facebook/dinov2-base` | `pixel_values` `(1, 3, 224, 224)` | shortest edge 256 then centre crop 224, ImageNet mean/std | `last_hidden_state (1, 257, 768)` = CLS + 16x16 patches, `pooler_output (1, 768)` | 3.1 ms/frame, 0.2 GB |
| `google/siglip2-so400m-patch14-384` | `pixel_values` `(1, 3, 384, 384)` | fixed 384x384, mean/std 0.5 | vision tower `last_hidden_state (1, 729, 1152)` = 27x27 patches (no CLS), `get_image_features (1, 1152)` for the image-text embedding | 12.4 ms/frame, 2.2 GB |
| `facebook/vjepa2-vitl-fpc64-256` | `pixel_values_videos` `(1, T, 3, 256, 256)` | shortest edge 292 then centre crop 256, ImageNet mean/std | `last_hidden_state (1, T/2 * 256, 1024)`: 512 tokens for T=4 | 48 ms per 4-frame clip (12 ms/frame), 0.7 GB |

What the feature-extraction path has to respect:
- V-JEPA 2 takes a **clip**, `AutoVideoProcessor` with `videos=[[frame, ...]]` (a list of HWC uint8 arrays), and its
  tubelet is 2 frames, so **T must be even**; tokens are `T/2 x 16 x 16`, so memory and cost grow linearly in T.
  The checkpoint was pretrained with 64 frames per clip; it accepts shorter clips, and our 4-frame WOD-E2E history
  is the natural unit. There is no CLS token and no pooled output, so the head pools itself.
- SigLIP2 has two towers. `AutoModel(**inputs)` alone fails ("You have to specify input_ids"); use
  `model.vision_model(...)` for patch tokens or `model.get_image_features(...)` for the aligned embedding.
- All three want their own normalisation (SigLIP2 differs from the other two), so the cached features are per
  backbone and cannot be shared.
- Per frame at our 3 cameras x 4 timestamps: DINOv2 37 ms, SigLIP2 149 ms, V-JEPA 2 3 clips of 4 frames = 144 ms.
  Qwen-Drive's own vision tower, for comparison, does the same 12 frames in 61 ms, as one batch.

## Not run

| Item | Status | Why |
|---|---|---|
| RAP (2510.04333) | code public ([vita-epfl/RAP](https://github.com/vita-epfl/RAP) `5fd8630`, Apache-2.0), checkpoints `Lanl11/RAP_ckpts` (Waymo 10.6 GB, not gated) | **Not runnable for us, cited only.** Its encoder initialises from `facebook/dinov3-vith16plus-pretrain-lvd1689m`; our access request was rejected by the authors and HF does not allow resubmitting, and we do not route around a deliberate denial through re-uploads (decision 12 in `research/decisions.md`). We quote its published test scores instead. It also ships no single-sample entry point and pins torch 2.1.0 cu121 / mmcv 2.1.0, which would need a rebuild for sm_120. |
| Poutine (2506.11234) | no code, no weights found | nothing public (arXiv, author pages, GitHub, HF). |
| FROST-Drive (2601.03460) | no code, no weights found | nothing public. |
| MindVLA-U1 (2605.12624) | no code, no weights found | project page mind-omni.github.io has no code or download links. |
| hr98w/jev-visual (`19af545`, MIT) | Apple MLX only | depends on `mlx` / `mlx-vlm` with 4-bit MLX weights; no torch or CUDA path. Would need a port. |

## Still open

- AutoVLA's reasoning path is unmeasured: the released checkpoint skips the chain of thought on this scene.
  Timing it needs a scene it decides to think about (a busy intersection), which also makes the point about
  input-dependent latency quantitative.
- Nothing else is blocked. RAP stays cited-only (below).

## Reproduce

- Code in `$DATA_DIR/third_party/<name>` at the commit above, venv in `$DATA_DIR/envs/<name>`,
  built by `scripts/bench_baselines/setup_<name>.sh`. Weights: `scripts/download_models.sh`.
- Run inside tmux (see [long-runs.md](long-runs.md)):
  - `scripts/tmux_run.sh bl-qd env OMP_NUM_THREADS=4 $DATA_DIR/envs/qwen-drive/bin/python scripts/bench_baselines/qwen-drive.py`
  - `scripts/tmux_run.sh bl-openjev scripts/bench_baselines/openjev.sh`
  - `scripts/tmux_run.sh bl-autovla env OMP_NUM_THREADS=4 $DATA_DIR/envs/autovla/bin/python scripts/bench_baselines/autovla.py`
- `scripts/bench_baselines/_bench.py` is the shared timer (stdlib + torch, so it runs in every venv).

Last verified: 2026-09-20
