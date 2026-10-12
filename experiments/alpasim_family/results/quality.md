# FAMILY-Q: is the openpilot family more sensitive to MTGS reconstruction quality than the samples and garage? (2026-10-12)

Pre-registration: [../plans/2026-10-12-quality-prereg.md](../plans/2026-10-12-quality-prereg.md) (a893df7a, before any regression). Stored
10-decision rollouts on the 1479 scored public scenes, no rerun: YR10m10-F-s0, APY10m10-AB-s0, WA-JEPA (aud1), the LTF / DiffusionDrive
samples and garage 0014 (proxy anchors). CPU only, plus one 96-scene GPU job (the A0 control rerun with `SH30_DUMP=96`, identical
rollouts) to get rendered k = 0 frames for Q5. Code: `scripts/asset_quality.py` (box), `scripts/q5_photometric.py` (box),
`scripts/quality_report.py` (Mac). Tables: `asset_quality_public.csv`, `q5_photometric_s96.csv`, `quality/{slopes,tails,extrapolate}.csv`.
Deviation: 2 000 bootstrap draws for the slopes, not 4 000 (run time); tails and extrapolation 1 000.

Note: `asset_grade_public.csv` is road grade (logged ego pitch / altitude change), not reconstruction quality; it is kept as a covariate.

## Quality measures

| measure | what | public median / p95 |
|---|---|---|
| Q1 `pose_t_m`, `pose_r_deg` | how far the reconstruction moved the logged ego poses (refined `ego2global` vs `ego2global_original`, re-anchored at the scene's first frame, median over the 5 s window) | 0.19 m / 0.58 m; 0.33 / 1.01 deg (34 older assets have no original poses) |
| Q2 `fyfx_dev` | \|fy/fx - 1\| of the refined CAM_F0 intrinsics (the render is that anisotropic) | 0.016 / 0.069 |
| Q3 `k1_dev` | \|k1 refined - k1 raw\| | 0.019 / 0.050 |
| Q4 `bg_bytes` | background checkpoint size (Gaussian count proxy; small = sparse) | 392 MB mean |
| Q5 `q5_ncc`, `q5_l1` (96 scenes) | k = 0 render vs the real nuPlan frame of the same instant (best of frames 16-20): gradient NCC, normalised gray L1 | NCC 0.22 median |

Validation (96 scenes, Spearman with Q5 NCC / L1): pose residual -0.25 / +0.26 (the only measure tied to render quality at about p 0.01),
rotation residual -0.15 / +0.20, background size -0.13 / +0.19, intrinsics and distortion deviation not (+0.23 / -0.12 and +0.07 / -0.02,
the wrong way). Q5 itself is a weak ruler: the best-matching real frame spreads over 16-20, moving traffic dominates the error.

## Slopes per SD of each measure (score), and the family difference

Family difference = mean(YR, APY) slope - mean(LTF, DD, garage) slope; 44-log cluster bootstrap. "Family more sensitive" would be a
negative score difference on the measures where large is worse (positive for `bg_bytes`), with GT distance moving the same way.

| measure | YR | APY | LTF | DD | garage | WA-JEPA | family diff [95 % CI] | GT family diff [95 % CI] |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| pose_t_m (large = worse) | -0.011 | -0.003 | -0.022 | -0.025 | +0.047 | -0.014 | -0.007 [-0.026, +0.014] | -0.141 [-0.265, -0.049] |
| pose_r_deg (large = worse) | -0.006 | -0.002 | -0.057 | -0.049 | +0.019 | -0.011 | **+0.025 [+0.004, +0.046]** | -0.072 [-0.143, -0.003] |
| fyfx_dev (large = worse) | +0.001 | +0.000 | +0.016 | +0.011 | +0.005 | +0.003 | -0.010 [-0.039, +0.007] | -0.028 [-0.084, +0.070] |
| k1_dev (large = worse) | +0.005 | +0.006 | +0.024 | +0.029 | +0.017 | -0.002 | -0.018 [-0.050, +0.005] | -0.005 [-0.069, +0.092] |
| bg_bytes (small = worse) | +0.011 | +0.012 | +0.034 | +0.033 | +0.077 | +0.001 | **-0.036 [-0.055, -0.020]** | +0.008 [-0.080, +0.093] |
| pitch_mean (grade, covariate) | -0.003 | +0.002 | +0.050 | +0.035 | +0.027 | -0.009 | -0.038 [-0.057, -0.020] | +0.018 [-0.047, +0.088] |
| dz_end (grade, covariate) | +0.006 | +0.007 | +0.058 | +0.051 | +0.055 | +0.003 | -0.048 [-0.066, -0.032] | +0.032 [-0.035, +0.100] |

Read: no measure meets the registered line (with or without Bonferroni). Where the family difference is clear, it points the other way:
the samples lose score on scenes whose poses the reconstruction corrected most (LTF -0.057, DD -0.049 per SD of rotation residual) and
gain with denser backgrounds (garage +0.077 per SD), while YR and APY are flat (|slope| <= 0.012 everywhere). Our GT distance even falls on
the high-residual scenes. The grade covariates show the same pattern (samples gain on uphill / rising scenes, ours flat).

## Tails: the worst 10 % of public assets on each measure (about 145-150 scenes)

Score on the tail (difference to the other 90 %, log bootstrap CI):

| tail | YR | APY | LTF | DD | garage | WA-JEPA |
|---|--:|--:|--:|--:|--:|--:|
| pose residual, translation | 0.903 (-0.032 [-0.076, +0.014]) | 0.936 (-0.007) | 0.756 (-0.083 [-0.205, +0.030]) | 0.722 (-0.089) | 0.823 (+0.085) | 0.850 (-0.058) |
| pose residual, rotation | 0.935 (+0.003) | 0.947 (+0.005) | **0.722 (-0.120 [-0.221, -0.022])** | **0.698 (-0.117 [-0.223, -0.020])** | 0.791 (+0.050) | 0.890 (-0.013) |
| intrinsics anisotropy | 0.931 (-0.001) | 0.948 (+0.006) | 0.902 (+0.079) | 0.869 (+0.074) | 0.764 (+0.020) | 0.918 (+0.018) |
| distortion deviation | 0.949 (+0.019) | 0.953 (+0.012) | 0.913 (+0.091) | 0.894 (+0.101) | 0.772 (+0.028) | 0.917 (+0.017) |
| sparsest background | 0.898 (-0.038 [-0.077, +0.004]) | 0.885 (-0.064 [-0.112, -0.015]) | 0.781 (-0.056) | 0.747 (-0.062) | **0.505 (-0.267 [-0.355, -0.177])** | 0.863 (-0.044) |
| steepest downhill (grade) | 0.931 (-0.001) | 0.944 (+0.001) | **0.671 (-0.177 [-0.252, -0.104])** | 0.715 (-0.098) | 0.685 (-0.068) | 0.905 (+0.004) |

Extrapolated score with the whole suite at the public 95th percentile of the measure (log bootstrap CI): rotation residual YR 0.921
[0.896, 0.947], APY 0.939, LTF 0.719 [0.643, 0.794], DD 0.706, garage 0.783; sparse background YR 0.920, APY 0.928, LTF 0.791, DD 0.764,
garage 0.655 [0.605, 0.702]; all of it in `quality/extrapolate.csv`.

**Answer: no.** On every reconstruction-quality measure computed here, the openpilot family is the least sensitive driver set, not the
most. A private suite of lower-quality assets would, by these slopes, cost the samples and garage more than us — the opposite of the
private result (samples hold, ours fall). The tails do not simulate the private gap for any driver: YR, APY and LTF on the worst rotation
residual decile read 0.935 / 0.947 / 0.722, the reverse of their private order. Reconstruction quality as measured here does not explain the
family-specific loss. Caveat: the measures are proxies (only the pose residual is tied to render error, rho about 0.25 on 96 scenes); a
quality defect that none of them captures (e.g. a different renderer build or a texture / exposure shift in the private assets) is not
excluded.
