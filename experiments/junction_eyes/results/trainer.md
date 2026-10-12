# Step 1 trainer, export and gates (executor 3)

Plan: [junction-eyes-plan.md](../../../research/next-round/junction-eyes-plan.md) sections 4.1 to 4.4, 5.2 and 5.3. Pre-registration B:
[plans/2026-10-12-prereg-B.md](../plans/2026-10-12-prereg-B.md). Status 2026-10-12 23:00 JST: the engineering is done and every gate passes.
**No arm has been trained or submitted** (coordinator HOLD, 22:20). Every number below is a gate, a smoke run or a throughput measurement;
none of them reads a read set.

## Code

| file | what it does |
|:--|:--|
| `jevdrive/op_torch.py` | LoRA on Conv kernels on top of the existing MatMul / Gemm LoRA. `A @ B` has the weight's 2-D shape: Conv (out, in·k·k), reshaped back to the kernel. Depthwise kernels are trained in full through `trainable` (fp32 masters). `merged()` returns W + A @ B · alpha / r (a zero update keeps W's bytes). `write_merged(src, dst, weights)` writes into a copy and refuses to overwrite its source. Unit test: `tests/test_op_torch_lora.py`. |
| `jevdrive/op_adapt.py` | `vision_lora_targets()` classifies the trunk weights: 72 MLP MatMuls (the ConvNeXt pointwise layers); 5 dense convs (4×4 stem, three 2×2 downsamples, 1×1 head); 36 depthwise 7×7. |
| `lib/je_model.py` | The model. Arm kinds: `lora` (r 16, alpha 16, B = 0; 6.00 M LoRA + 1.77 M depthwise), `full` (349.3 M, input normalisation constants excluded), `none`. Two heads from `head1_train.build()`: R-A nq 15 nc 2, R-B nq 36 nc 192 (6.82 M together). `rb_logits` maps (B, 36, 192) to (B, 48, 48, 3): query 6·bi + bj owns the 8×8 block, cells row-major, x is the row. The heads read the t0 slot (the views sit in the last slots; `keep` = those tokens). `cam_emb` (zero init) is added to each view's tokens when n_view > 1. |
| `lib/je_data.py` | Data access. Pixel stores read with preadv into pinned memory: F0, W90 (F0 road + `px90` wide), 3V (F0 + `px_side` L0 / R0). `KeepStore` follows executor 2's layout. Row sources from executor 1's parquet and labels; ego comes from the token cache `tab.npz`. A prefetcher. `comma_frames` decodes comma1M in the openpilot env, because op-train has no PyAV. |
| `scripts/je_train.py` | `train` (arms L-F0, L-F0-nav, U-F0, L-W90, L-3V); `ident` (identity gate); `prof` (fetch / GPU / loop throughput split). |
| `scripts/je_export.py` | `export` (merged ONNX copy); `gate` (torch vs merged ONNX); `serve0` (G-serve with B = 0). |
| `scripts/je_encode.py` | Re-encode with a checkpoint's trunk, or `--tag factory` for Z'. `--preset f0` covers the 12 navtrain shards, lb_navtest, and the J_on / held-452 / zero-98 slot pairs. `--preset w90` / `3v` cover navtrain + navtest only. Output: `step1/tokens/<tag>/<set>.npy`, (N, 32, 512) fp16, or (N, 3, 32, 512) for 3v, in the pixel file's row order. Only the t0 slot is re-encoded, because the pixel caches hold only the t0 pair. Executor 1's HEAD1 probes on these tokens therefore read the t0 slot only (the "Cinque reads slot 7 only" form of 3.3), and Z' must be read the same way. |
| `scripts/je_keep_mini.py` | A 4-segment keep store, used for the first smokes before executor 2's `keep/` existed. |
| `scripts/je_e5.py`, `scripts/je_wod.py` | E5 and the WOD canary (sub-agent; commit ccc7b348). |

How a training step runs. Batch 64: 24 J_nav + 24 J_on + 16 keep rows; nav-only arms use 48 J_nav + 16 keep.
- Draws: J_nav uniform over its fit rows; J_on weighted by `w_scene`; keep uniform over the non-held keep rows. Each stream has its own rng seeded by the seed alone, so every arm of a seed sees the same rows.
- Forward: one grad pass over the J t0 pairs plus the keep t0 pairs; the 8 keep context pairs go through the current trunk without grad.
- Keep term: the frozen factory policy on the 9 hidden states, distilled against executor 2's 20 Hz Stepper teacher. The weight is lam_d 30 on the mean over the `distill_index` columns of ((s − t) / std)², with the plan columns at the same per-column weight as the others.
- Head losses: Huber δ 1 on R-A, standardised by the mean / std of the fit rows, with missing labels masked; cross-entropy on R-B.
- Optimiser: AdamW; LoRA 3e-4 wd 0; depthwise 1e-5 wd 0; U-F0 vision 1e-5 wd 0; heads 3e-4 wd 0.05. Warmup 300, cosine, bf16, clip 1.0 per group (vision / heads).
- Validation: every 1 000 steps, R-A + R-B loss on val selects `best.pt`.
- Outputs: `$DATA_DIR/runs/junction_eyes/step1/ckpt/<tag>/{best,last,resume}.pt` and `evals.json`. The run dir is `runs/junction_eyes/step1/train-<tag>/<ts>/`.

## Gates (all on the box, through the pool; raw `$DATA_DIR/runs/junction_eyes/step1/gates/`)

**Identity gate (5.2 / 4.4)**: `ident.json`. 512 navtrain t0 pairs (`px/navtrain_full.s0of12`), trunk at its starting point compared with the frozen trunk, view_39.

| arm kind, dtype | bitwise-equal share | max abs diff | mean abs diff vs the fp32 trunk |
|:--|--:|--:|--:|
| LoRA + depthwise masters, bf16, no-grad and grad pass | 1.0 | 0 | 0.01380 (= frozen bf16) |
| LoRA + depthwise masters, fp32, both passes | 1.0 | 0 | 0 |
| full unfreeze (U), fp32, both passes | 1.0 | 0 | 0 |
| full unfreeze (U), bf16 | 0.167 | 0.5 | **0.01280** (frozen bf16: 0.01380) |
| control: LoRA with B ~ N(0, 1e-3), bf16 / fp32 | | 1.03 / 0.84 | mean 0.035 / 0.029 |

- LoRA is bit-exact in both dtypes.
- U in bf16 is not bitwise equal to the frozen bf16 trunk. Its fp32 masters keep the LayerNorm weights at fp16 precision, whereas the frozen bf16 port rounds them to bf16. So U sits slightly *closer* to the fp32 trunk than the frozen bf16 trunk does. This is precision, not an error.
- Token rms is 1.68. Pass = LoRA bitwise in both dtypes, U bitwise in fp32, and U bf16 no further from fp32 than frozen bf16: **pass**.
- The pixel path reproduces the cached tokens bit-exactly on navtrain (executor 2, `vt_px check --t0`, results/pixels.md).

**G-serve (5.3, decision 282 wording)**: `serve0.json`. The B = 0 model was merged into a copy of `cinque.ort.onnx`.
- All 113 changed initializers are byte-identical to the factory (the merged file's sha256 equals the factory's, `3f12a0cd…`).
- Over a 400-step 20 Hz stream of held-out comma segment `001774ef…`, onnxruntime (eager), the plan is bit-identical to the factory ONNX in **400 / 400 steps**.
- The factory file's sha256 is checked before and after every export.
- The first run found 56 initializers whose -0 entries became +0 (values equal, bytes not). `merged()` now keeps W where the update is exactly zero.
- **Pass**.

**Equivalence gate (5.3)**: `equiv-smoke-L-F0-nav-s0.json`. A 300-step smoke checkpoint (trained LoRA, B ≠ 0) was merged and run on onnxruntime CPU (op-train's onnxruntime has no CUDA provider). Training moved the tokens by mean 0.049 / max 3.4 and the plan by up to 0.44 m.

| quantity | arm: torch fp32 vs merged ONNX | factory: torch fp32 vs factory ONNX (the fp16 floor) |
|:--|--:|--:|
| view_39 on 512 pairs, mean / max abs | 9.0e-4 / 0.038 | 8.9e-4 / 0.041 |
| plan (x, y), first 15 points, 400 stream steps: max / median | 0.0022 m / 0.0005 m | 0.0033 m / 0.0005 m |

The merged model is at the fp16 floor, and the plan is within 0.5 m (it is 0.002 m). **Pass.**

## Smoke (300 steps, one card each; reads no read set)

Real J_nav / J_on rows and labels (step0), executor 2's pixel caches and keep store; `--gn-every 10` adds per-term gradient norms on the vision weights.
- `smoke-realkeep-s0` and `smoke-L-F0-full-s0` are the L-F0 smokes (nav-only and two-source).
- The U row is the U-F0 smoke on the 4-segment mini keep store; the real keep store gives the same order.
- `smoke-L-3V-s0` ran 100 steps on a 2 000-row, 3-view test file.

| run | keep grad norm / head grad norm (steps 100, 200, 300) | keep loss (λ 30) step 100 → 300; monitor set | R-A / R-B val loss 0 → 300 |
|:--|:--|:--|:--|
| L-F0-nav, real keep | 3.86/0.96 = 4.0, 5.80/0.95 = 6.1, 3.39/0.75 = 4.5 | 0.063 → 0.021; monitor 0.012 → 0.069 | 0.42 / 1.20 → 0.105 / 0.572 |
| L-F0, J_nav + J_on, real keep | 2.76/1.20 = 2.3, 3.59/1.97 = 1.8, 3.40/0.93 = 3.7 | 0.056 → 0.038; monitor 0.012 → 0.091 | 0.44 / 1.20 → 0.143 / 0.613 |
| U-F0, mini keep | 3.45/0.78 = 4.4, 2.29/1.18 = 1.9, 1.17/0.79 = 1.5 | 0.019 → 0.009; monitor 0.012 → 0.010 | 0.52 / 1.19 → 0.105 / 0.569 |
| L-3V, 3-view test rows | 3.08/0.47 = 6.5 (step 100) | 0.019 at step 100 | 0.41 / 1.20 → 0.21 / 0.88 |

- **Gradient ratio.** The keep gradient is 1.5 to 6.5 times the head gradient on the vision weights. That is the same order of magnitude, not the d245 failure shape (1 to 5 %). **No weight change proposed: λ_keep stays 30 (vt.py's lam_d).**
- **Keep loss.** At step 0 it sits at the bf16 / teacher floor (0.012). It rises in the first ~100 steps while the heads pull the trunk, then the training-batch keep loss falls (L-F0-nav 0.063 → 0.021, L-F0 0.056 → 0.038, U 0.019 → 0.009).
- **Monitor set (256 fixed keep rows).** On the LoRA arms it has not come back down by step 300 (0.069 / 0.091 against 0.012 at the start), so the keep term restores the training rows faster than it restores the trunk in general. Watch `val/mon_keep` in the full runs. If it keeps rising past a few thousand steps, the weight is too low for 30 k steps. That is a reason to amend B before any read, not after.
- **Earlier bug.** The first smokes ran the keep term at weight 1 (λ was not applied); fixed in 24be1911. The numbers above are with λ 30.

## Throughput and projected wall time per arm

Measured with `je_train.py prof --compile` on one free RTX 6000D per arm (box idle, 22:29). The fetch is hidden (0.09 to 0.17 s per batch on one thread, 3 worker threads), so the loop runs at the GPU's speed: **the bottleneck is GPU compute in the trunk's forward / backward**.

| arm | GPU it/s | loop it/s | peak VRAM | steps | train time | + 31 evals | **wall per job, alone on a card** |
|:--|--:|--:|--:|--:|--:|--:|--:|
| L-F0, L-F0-nav | 3.22 | 3.07 | 12.1 GB | 30 029 | 2.72 h | ~0.2 h | **≈ 2.9 h** |
| U-F0 | 3.09 | 2.95 | 17.5 GB | 30 029 | 2.83 h | ~0.2 h | **≈ 3.0 h** |
| L-W90 | 3.21 | 3.08 | 12.1 GB | 30 029 | 2.71 h | ~0.2 h | **≈ 2.9 h** |
| L-3V (144 + 16 grad pairs per step) | 1.99 | 1.95 | 25.0 GB | 30 029 | 4.28 h | ~0.3 h | **≈ 4.6 h** |

- **Steps.** 12 epochs of J at 48 J rows per step. J fit = P = 90 421 J_nav + 29 694 J_on = 120 115 rows, so 12 · 120 115 / 48 = **30 029 steps** for every arm. The plan estimated ~20 k for 70 to 90 k rows; J turned out larger. L-F0-nav is then 15.9 epochs of its 90 421 rows. The trainer's own default (`--epochs 12`) would give the nav-only arms fewer steps, so the launch lines pass `--steps 30029` explicitly.
- **Shared cards.** The smokes on cards shared with executor 2's render jobs ran at 1.0 to 1.5 it/s. Two arms on one card therefore take about twice as long.
- **The whole set.** The 10 jobs on 6 free cards take about 3 h for the first six, then up to 4.6 h for the last wave: **≈ 7.5 h wall**, or ≈ 6 h if both L-3V seeds start first.

## Launch (on HOLD; nothing below has been submitted)

From `~/data/jev-drive` at a pushed commit (current head; the trainer is unchanged since `146229ef`):

```
PY=$DATA_DIR/envs/op-train/bin/python; L=$DATA_DIR/runs/junction_eyes/step1/pool; S=experiments/junction_eyes/scripts
for seed in 0 1; do for arm in L-3V L-F0 L-F0-nav U-F0 L-W90; do
  v=24; [ $arm = U-F0 ] && v=30; [ $arm = L-3V ] && v=40
  $PY -m jevdrive.cl submit --name JE-$arm --vram $v --cpu 10 --ram 32 --train --log-dir $L/JE-$arm-s$seed -- \
    $PY $S/je_train.py train --arm $arm --seed $seed --tag JE-$arm-s$seed --steps 30029 --compile
done; done
```

- Tags: `JE-<arm>-s<seed>`. Resume after an interruption: the same line with `--resume`.
- Z' (pre-registration B): `$PY -m jevdrive.cl submit --name je-encode-factory --vram 12 --cpu 8 --ram 16 -- $PY $S/je_encode.py --tag factory --preset f0`, which writes `$DATA_DIR/runs/junction_eyes/step1/tokens/factory/`. Not submitted (HOLD).
- Read protocol per arm, after training:
  - `je_encode.py --tag JE-<arm>-s<seed> --preset f0`, or `w90` / `3v` for the field arms;
  - `je_export.py export --tag JE-<arm>-s<seed>` and then `je_export.py gate --tag ...`;
  - E5 and WOD on `step1/onnx/JE-<arm>-s<seed>.onnx`.
- Re-encode speed: ~1 000 pairs/s on a free card (tested: `--preset 3v`, one navtrain shard, 8 608 rows x 3 views in 27 s), so a full
  `--preset f0` (~158 k pairs) takes ~3 min.
- The field arms' merged ONNX is not a servable model: W90 needs the 90° wide-slot input, and 3V needs three views and the heads. Plan 4.1 says so too.
