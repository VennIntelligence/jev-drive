# small_vlm: Qwen3.5-2B / 4B and Gemma 4 E2B on the d85 ego-light reading (first look)

Decision 182. Harness: `scripts/small_vlm_eval.py` (one generic HF path for all models; `small_vlm_pdl.py` / `small_vlm_dl.sh` download).
Data: the 233 sweep frames of d85 (85 ego red / 68 ego green / 80 no light, 21 routes, CARLA), the sweep's own `truth` labels, the d85 prompt
(`FOUR_PROMPT`, two cameras), zero-shot, bf16, batch 1. CIs: `jevdrive.stats.bootstrap`, resampling routes. **Small-batch first look**: the red cell
is 85 frames from a handful of routes, so the red-recall CI is about +/-6 pp for the control and +/-10-20 pp for the new models; green is 68 frames,
no-light 80 (CI [0, 0] there only means no false alarm in 80 frames on these routes). Not a model switch; no serving path changed.

Models taken (ModelScope copies, bf16; hf-mirror was ~1 MB/s): `Qwen/Qwen3.5-2B` (4.5 GB), `Qwen/Qwen3.5-4B` (9.2 GB), `google/gemma-4-E2B-it` (9.9 GB, not gated on
ModelScope; 5.1 B params counting per-layer embeddings), control `Qwen/Qwen3-VL-4B-Instruct`. Shared `.venv` transformers 5.17 loads all of them; Qwen3.5
ran with `flash-linear-attention` from a separate `--target` dir (without it the gated-delta-rule fallback made the 2B forward 160 ms instead of 73 ms).
Downloads: 14-15 MB/s with 64 range streams (see docs/storage.md).

## Main read: the d85 serving setting (one forward, option scoring, ~1150 visual tokens, `fwd_r1153`)

| model | ego red: red (recall) | ego red: green | ego green: green | no light: red | red precision | latency p50 / p95 ms (JPEG-decode + preprocess + forward) | forward only ms | peak VRAM GiB |
|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| Qwen3-VL-4B (control) | 90.6% [84.1, 96.9] (77/85) | 4.7% (4) | 85.3% [78.6, 93.3] (58/68) | 0/80 | 90.6% (77/85) | 178 / 186 | 124 | 8.6 |
| Qwen3.5-2B | 11.8% [1.4, 23.9] (10/85) | 3.5% (3) | 75.0% [55.9, 91.4] (51/68) | 0/80 | 100% (10/10) | 144 / 152 | 73 | 4.5 |
| Qwen3.5-4B | 52.9% [32.1, 73.2] (45/85) | 35.3% [15.3, 58.0] (30) | 98.5% [94.8, 100] (67/68) | 0/80 | 100% (45/45) | 185 / 191 | 120 | 8.8 |
| Gemma 4 E2B | 67.1% [48.0, 83.8] (57/85) | 10.6% [3.7, 20.3] (9) | 83.8% [71.7, 94.5] (57/68) | 0/80 | 91.9% (57/62) | 194 / 203 | 118 | 9.5 |

Control check (Qwen3-VL-4B in this harness vs the archived d85 run): `generate` answers identical on 100% of frames (79/85, 4 red-as-green, 66/68, 0/80); the one-forward configs differ on 1-2 frames
of 233 (77 vs 76 red hits at 1153) because this harness preprocesses on CPU (1140 tokens instead of 1153). Latency here includes JPEG decode on the CPU, so it is higher than d85's 125 ms (GPU decode); the
forward-only column (124 ms) matches d85. Latency was measured on 100 frames, batch 1, with other lanes running on the box.

## Extra configs that finished (not the registered read; the generate rows use a 512-token budget because these models reason aloud before the ANSWER line)

### Accuracy (233 sweep frames, 85 ego red / 68 ego green / 80 no light; cell = estimate [95% route-cluster CI])

| model | config | visual tokens | ego red: red | ego red: green | ego green: green | no light: red | red precision | peak VRAM GiB |
|:--|:--|--:|:--|:--|:--|:--|:--|--:|
| q3vl4b | fwd_r4573 | 4560 | 92.9% [87.5, 98.6] (79/85) | 4.7% [0.0, 10.3] (4) | 97.1% [91.5, 100.0] (66/68) | 0.0% [0.0, 0.0] (0/80) | 97.5% (79/81) | 9.3 |
| q3vl4b | fwd_r1153 | 1140 | 90.6% [84.1, 96.9] (77/85) | 4.7% [0.0, 10.3] (4) | 85.3% [78.6, 93.3] (58/68) | 0.0% [0.0, 0.0] (0/80) | 90.6% (77/85) | 8.6 |
| q3vl4b | fwd_r559 | 546 | 91.8% [84.6, 98.6] (78/85) | 4.7% [0.0, 10.3] (4) | 82.4% [64.7, 96.7] (56/68) | 0.0% [0.0, 0.0] (0/80) | 87.6% (78/89) | 8.5 |
| q3vl4b | gen_native |  | 92.9% [87.5, 98.6] (79/85) | 4.7% [0.0, 10.3] (4) | 97.1% [91.5, 100.0] (66/68) | 0.0% [0.0, 0.0] (0/80) | 97.5% (79/81) | 9.6 |
| q35_2b | fwd_r4573 |  | 7.1% [0.0, 15.4] (6/85) | 2.4% [0.0, 7.2] (2) | 80.9% [64.6, 92.5] (55/68) | 0.0% [0.0, 0.0] (0/80) | 100.0% (6/6) | 4.1 |
| q35_2b | fwd_r1153 | 1140 | 11.8% [1.4, 23.9] (10/85) | 3.5% [0.0, 8.8] (3) | 75.0% [55.9, 91.4] (51/68) | 0.0% [0.0, 0.0] (0/80) | 100.0% (10/10) | 4.5 |
| q35_2b | fwd_r559 | 546 | 8.2% [0.0, 19.6] (7/85) | 2.4% [0.0, 7.2] (2) | 61.8% [40.0, 80.3] (42/68) | 0.0% [0.0, 0.0] (0/80) | 100.0% (7/7) | 4.3 |
| q35_2b | gen_native |  | 81.2% [65.6, 94.7] (69/85) | 3.5% [0.0, 8.6] (3) | 91.2% [84.5, 96.7] (62/68) | 1.2% [0.0, 4.3] (1/80) | 94.5% (69/73) | 5.0 |
| q35_4b | fwd_r4573 |  | 63.5% [42.6, 82.1] (54/85) | 27.1% [8.1, 50.0] (23) | 98.5% [94.8, 100.0] (67/68) | 0.0% [0.0, 0.0] (0/80) | 100.0% (54/54) | nan |
| q35_4b | fwd_r1153 | 1140 | 52.9% [32.1, 73.2] (45/85) | 35.3% [15.3, 58.0] (30) | 98.5% [94.8, 100.0] (67/68) | 0.0% [0.0, 0.0] (0/80) | 100.0% (45/45) | 8.8 |
| q35_4b | gen_native |  | 82.4% [62.1, 97.1] (70/85) | 11.8% [0.0, 33.3] (10) | 76.5% [55.2, 92.5] (52/68) | 0.0% [0.0, 0.0] (0/80) | 90.9% (70/77) | nan |
| gemma4_e2b | fwd_r4573 |  | 68.2% [48.8, 84.2] (58/85) | 12.9% [4.5, 24.2] (11) | 89.7% [79.4, 97.4] (61/68) | 0.0% [0.0, 0.0] (0/80) | 96.7% (58/60) | nan |
| gemma4_e2b | fwd_r1153 |  | 67.1% [48.0, 83.8] (57/85) | 10.6% [3.7, 20.3] (9) | 83.8% [71.7, 94.5] (57/68) | 0.0% [0.0, 0.0] (0/80) | 91.9% (57/62) | 9.5 |


Peak VRAM `nan` = not recorded (job cancelled before the cfg ended, answers kept). Gemma 4 E2B `fwd_r1153` used 1044 visual tokens (soft-token budget 560 per image), Gemma `fwd_r4573`
2184. `gen_native` for Qwen3-VL-4B is d85's generate row (reproduced exactly).

## Follow-up: native-style reading (no forced `ANSWER:` prefix, one-word prompt)

Prompt: the two-camera sentence + "Is the traffic light that controls the ego vehicle's lane ahead red or yellow, green, or is there none (no light, or a light for another lane)? Reply with one word: red, green or none."
`short_logit` = one forward, argmax over the first token of `red` / `green` / `none` at the generation position (Qwen3.5 with thinking off); `short_gen` = greedy generate, 8 new tokens.
Same 233 frames, ~1150 visual tokens. Four jobs ran at the same time on the box, so the latency column (forward / generate only, no decode) is rough.

| model | config | ego red: red | ego red: green | ego green: green | no light: red | red precision | unparsed | ms p50 |
|:--|:--|:--|:--|:--|:--|:--|--:|--:|
| Qwen3-VL-4B (control) | short_logit | 75.3% [55.9, 91.6] (64/85) | 19 | 89.7% [83.8, 96.8] (61/68) | 0/80 | 91% | 0 | 151 |
| Qwen3-VL-4B (control) | short_gen | 74.1% [53.7, 91.3] (63/85) | 14 | 89.7% (61/68) | 0/80 | 91% | 6 | 156 |
| Qwen3.5-2B | short_logit | 0.0% (0/85) | 0 | 52.9% [30.0, 76.7] (36/68) | 0/80 | n/a | 0 | 141 |
| Qwen3.5-2B | short_gen | 0.0% (0/85) | 0 | 42.6% [21.3, 66.0] (29/68) | 0/80 | n/a | 10 | 153 |
| Qwen3.5-4B | short_logit | 41.2% [23.3, 56.1] (35/85) | 2 | 54.4% [34.7, 75.3] (37/68) | 0/80 | 100% | 0 | 352 |
| Qwen3.5-4B | short_gen | 41.2% [23.3, 56.1] (35/85) | 2 | 48.5% [26.0, 72.1] (33/68) | 0/80 | 100% | 0 | 465 |
| Gemma 4 E2B | short_logit | 64.7% [44.1, 82.0] (55/85) | 5 | 77.9% [60.0, 91.3] (53/68) | 1/80 | 93% | 0 | 112 |
| Gemma 4 E2B | short_gen | 64.7% [44.1, 82.0] (55/85) | 5 | 77.9% [60.0, 91.3] (53/68) | 1/80 | 93% | 3 | 151 |

Reading: the shorter, native-style prompt does not rescue the new models (Qwen3.5-2B never says red, Qwen3.5-4B 41%, Gemma 65%, all at or below the forced-option read), and it also hurts the
control (91% to 75%, with 14-19 red frames read as green), so the d85 option-scoring prompt is not what holds the new models back. Where the red misses go differs: Qwen3.5 answers "none" for red
lights (red to green only 0-2 frames), Gemma and the control confuse red with green.
