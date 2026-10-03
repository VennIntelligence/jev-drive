# Scale check: does the NAVSIM height error (decision 104) exist on WOD, and on HUGSIM? (2026-10-04)

Question: WOD's front camera extrinsic z is 1.81 m, as high as NAVSIM's 1.87 m, yet openpilot "behaves" on WOD (plan-speed proxy 0.92 vs 0.81-0.84).
Either WOD's vehicle origin is not on the ground, or the scale error is there and the proxy hides it. Code: `scripts/sc_*.py`; numbers: `scale_check/stats.json`,
`scale_check_hugsim.csv`. All openpilot runs: Cinque (trt), 20 Hz, zero state, no desire, shipped rig; "h1.22" = the virtual 1.22 m camera of
`model_smoke/lib/wod_openpilot_rigs.py` (same rig as decision 104's intervention). CIs: cluster bootstrap over logs / sequences / segments / scenes.

## 1. WOD ground height in the vehicle frame (evidence)

Perception v2 labels (986 train+val segments, 1.71 M vehicle boxes within 4-30 m, >= 50 lidar points): box bottom (z - h/2) in the vehicle frame has pooled
median -0.063 m (IQR -0.24 to +0.08), per-segment median -0.049 m (IQR -0.15 to +0.02). So the **WOD vehicle-frame origin is on the ground (ground z about -0.05 m)**, not
0.35 m above it as NAVSIM's rear-axle origin. Perception FRONT camera z = 2.116 m (constant) -> 2.16 m above ground (IQR 2.09-2.27 per segment).
The **WOD-E2E** FRONT extrinsic z is 1.8065 m in all 1984 sequences, 0.31 m lower than the perception rig (x 1.519 vs 1.539): a different platform. E2E has no labels, so
its ground z is not measurable directly: inference = same vehicle-frame convention, so **true height about 1.86 m** (1.81 + 0.05); section 2 checks this from the images.

## 2. Scale read (model vs reference)

Model lane width = openpilot ego lane lines at 10 m (20 m agrees within 0.03 m), both lines p > 0.5. Model road z = lane-line z in the device frame (what the model thinks the camera height is).
Plan speed = plan point-0 speed / logged speed, speed > 3 m/s. Predicted lane ratio from height alone: 1.22 / h.

| board (true camera height) | rig | lane width m | model road z m | plan speed / logged | lead x / true gap (slope, intercept) |
|---|---|---|---|---|---|
| NAVSIM navtest, 1.87 m (n 12 146 tokens; lane 2 962 / 116 logs) | shipped | 2.28 [2.25, 2.32] (map 3.38: ratio 0.68) | 1.343 [1.339, 1.348] | 0.815 [0.806, 0.822] (8 693 / 134 logs) | median 0.889 [0.882, 0.896] (4 251 / 111 logs); slope 0.75 [0.71, 0.79], icpt +1.2 m |
| NAVSIM 400-token subset | shipped | 2.29 [2.24, 2.37] | 1.349 | 0.836 [0.826, 0.847] | 0.918 [0.905, 0.938] (152 / 70); slope 0.81 [0.74, 0.88] |
| NAVSIM 400-token subset | h1.30 | 3.25 [3.20, 3.31] (ratio 0.96) | 1.287 | 0.998 [0.991, 1.006] | 1.039 [1.023, 1.054] (147 / 68); slope 0.95 [0.87, 1.03] |
| **WOD-E2E** rater+extra, E2E z 1.81 (+0.05) | shipped | **2.22 [2.18, 2.32]** (251 targets / 127 seqs) | **1.343 [1.328, 1.351]** | **0.918 [0.914, 0.923]** (816 / 353 seqs) | no labels |
| WOD-E2E | h1.22 | 3.18 [3.13, 3.24] | 1.298 | 1.102 [1.093, 1.110] | no labels |
| WOD perception sceneflow, 2.16 m (10 segments, few leads) | shipped | 1.99 [1.89, 3.09] (226 / 2 segs) | 1.350 [1.26, 1.40] | 0.862 [0.834, 1.245] (333 / 3 segs) | 0.849 [0.759, 0.910] (325 / 5 segs); 8-15 m 0.96, 15-25 m 0.89, 25-50 m 0.81 |
| WOD perception | h1.22 | 3.23 [3.10, 4.98] (231 / 3) | 1.279 | 1.280 [1.269, 1.606] | 1.108 [0.995, 1.131] (322 / 4) |

Lead rule (fixed before looking): nearest vehicle box ahead, |y| < 1.5 m (pre-registered 1.2, relaxed on all boards after the sceneflow set gave 3 clusters), heading within 0.4 rad,
rear face 8-50 m from the camera; model lead = P > 0.5, selection 0 at t = 0. Lead recall 0.94-0.97 everywhere. Sceneflow has only 10 segments (the only local WOD frames with
images and labels), 5 with an in-lane lead, 2-3 with moving lane reads: its CIs are wide and it is a consistency check, not a measurement.

Findings
- (Evidence) **WOD-E2E reads the same world scale as NAVSIM.** Model road z 1.343 vs 1.343, lane width 2.22 vs 2.28 m. The lane width times height is flat across boards and rigs
  (NAVSIM 4.27 and 4.22 at 1.87 / 1.30 m, sceneflow 4.31 at 2.16 m): the model's lane width is about 4.25 / h. With NAVSIM as the calibration, E2E's height is 1.87 x 2.28 / 2.22 = **1.92 m** if lane
  widths match between nuPlan cities and Waymo cities (assumption), consistent with 1.86 m and not with a height near 1.22-1.5 m. Against the US standard 3.6 m the E2E ratio is 0.62.
- (Evidence) **Lowering the virtual camera to 1.22 m moves WOD-E2E exactly as it moves NAVSIM:** lane width 2.22 -> 3.18 m (x1.43; NAVSIM x1.42 at 1.30 m), plan speed 0.918 -> 1.102 (x1.20; NAVSIM 0.836 -> 0.998, x1.19).
  The scale error is there on WOD; the plan-speed proxy hides it because its starting value is higher on WOD (0.92 vs 0.82-0.84 at the same height; the speed head mixes a prior, and the two speed
  distributions differ), not because the scale differs.
- (Evidence, NAVSIM) lead distance carries the same error but weaker in the median: x / gap 0.89 (0.92 at 8-15 m, 0.87 at 25-50 m), slope 0.75 with a +1.2 m intercept (the offset is probably bumper vs camera
  referencing in the radar labels); at a 1.30 m virtual camera 1.04, slope 0.95. WOD sceneflow at 2.16 m (predicted lane scale 0.56): lead ratio 0.85, falling with distance (0.96, 0.89, 0.81), back to 1.11 at 1.22 m.
  Lead ratio is a diluted read of scale (0.85-0.92 against 0.56-0.68 for lanes); use slope and lanes for the scale.

## 3. HUGSIM 64 scenes by source dataset

Shipped Cinque (native, 10 spins) and it_dw3 (13 spins), 16 scenes per dataset, single seed. Effective camera height = recorded minus the cam_rect lowering as in `board_views.md` (not re-derived).
Plan speed from the shipped run's `zs_steps.jsonl` (scene medians, v > 3 m/s steps only, hence few scenes). Sharpness from the shipped run videos, 5 frames per scene: Laplacian variance of the
800x450 CAM_FRONT render and of openpilot's 512x256 road frame; hf = share of spectral power above 0.25 cycles/px of the render (low = soft). Laplacian variance depends on the scene content, hf less so.

| dataset | h_eff m | predicted scale 1.22/h | native spins | native HD [95% CI] | native RC | it_dw3 spins | it_dw3 HD | plan speed / ego (scenes) | f px | hf render | Lap render / road |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nuScenes | 1.2 | 1.02 | 3 | 0.458 [0.252, 0.673] | 0.561 | 3 | 0.448 [0.255, 0.645] | 1.29 (12) | 626 | 0.0124 | 154 / 168 |
| PandaSet | 1.5 | 0.81 | 2 | 0.283 [0.140, 0.456] | 0.307 | 3 | 0.376 [0.202, 0.553] | 1.22 (7) | 720 | 0.0316 | 416 / 393 |
| KITTI-360 | 1.5 | 0.81 | 3 | 0.212 [0.107, 0.336] | 0.338 | 3 | 0.156 [0.074, 0.247] | 1.22 (11) | 626 | 0.0128 | 607 / 216 |
| Waymo | 1.8 | 0.68 | 2 | 0.157 [0.097, 0.235] | 0.191 | 4 | 0.216 [0.117, 0.347] | 1.08 (5) | 720 | 0.0130 | 334 / 309 |

- (Evidence) Spins do not track height (3, 2, 3, 2 native; 3, 3, 3, 4 it_dw3). HD runs from 0.46 (nuScenes, lowest camera) to 0.16 (Waymo, highest) for the native model; Spearman of scene HD against height
  -0.17 (permutation p 0.18) native, -0.19 (p 0.13) it_dw3 over 64 scenes: the direction fits a height effect, but it is not significant and is confounded by dataset (scene content, difficulty mix, renderer quality).
  it_dw3 shows no such ordering (Waymo above KITTI-360).
- (Evidence) The model does not under-drive on HUGSIM: plan speed / ego speed is 1.08-1.29 on every dataset, not the 0.68-0.81 a height-only story predicts for Waymo / PandaSet / KITTI-360 (it falls slightly with height, 1.29 to 1.08, few scenes).
  This is the closed-loop ego speed (self-consistent history), so it is not comparable to the open-loop NAVSIM / WOD speed ratio.
- (Evidence) Sharpness differs by dataset and not along height: nuScenes, KITTI-360 and Waymo renders are equally soft (hf 0.012-0.013), PandaSet is 2.5x sharper. HUGSIM upsamples 800x450 to the model by 1.45x (nuScenes / KITTI-360, f 626)
  or 1.27x (f 720); KITTI-360's road frame loses most detail (Lap 607 -> 216).

## Reading
1. (Evidence) WOD's origin is on the ground (-0.05 m), so WOD-E2E's 1.81 m is about 1.86 m true height; openpilot reads WOD-E2E at the same scale as NAVSIM (road z 1.343 both, lanes 2.22 vs 2.28 m, same response to the 1.22 m rig): the error is on WOD too and the 0.92 plan-speed proxy hides it.
2. (Evidence) The error follows 1.22 / h on the boards where it can be measured (NAVSIM 1.87 and 1.30 m, WOD perception 2.16 and 1.22 m, WOD-E2E 1.86 and 1.22 m); WOD-E2E's own true height rests on the same-convention inference plus the lane-width calibration (1.92 m, assumption: equal lane widths).
3. (Inference) HUGSIM's dataset split does not isolate a height effect: spins are flat in height, HD falls with height at p 0.13-0.18 only, and plan speed is not low; HUGSIM's problem is probably elsewhere (closed loop, soft renders), though the "native" shipped HD ordering is compatible with a modest height contribution.
Limits: sceneflow is 10 segments (lead / lane CIs from 2-5 clusters); NAVSIM lead truth uses the rear face and camera x 1.62 m; effective HUGSIM heights are the board_views notes.
