# FAMILY task 2: calibration and image-geometry sensitivity on 96 public scenes (2026-10-12)

Pre-registration and amendment 1: [../plans/2026-10-12-family-prereg.md](../plans/2026-10-12-family-prereg.md). 96 public scenes (`s96.txt`,
6 per vehicle over all 16 vehicles, 8 logs), one GPU-pool job per arm (about 7 min each, 31 jobs, about 3.5 card-hours), native AlpaSim
0bb4c4b, `+e2e_challenge_nuplan=full`. Calibration arms change only what the driver is told about CAM_F0 (the image is the render);
image arms change the JPEGs (CAM_L0 / F0 / R0, re-encoded at the runtime's quality 95). Wrappers: `scripts/fam_driver.py` (our served
driver, imported unchanged), `scripts/fam_ltf.py` (the shipped LTF sample, imported unchanged), arms in `scripts/fam_perturb.py`. Tables:
`sens/sensitivity.csv` (`scripts/fam_report.py`); box runs `$DATA_DIR/runs/alpasim_family/{pilot,main,def,rnom,apy}/`.

Gates. G0: the control reproduces the stored compiled YR10m10-F-s0 rollouts on these scenes to 4 decimals (0.9527 / 1.003 m both), APY's
control likewise (0.9564 / 1.156 against 0.9563 / 1.157); every start record logs the arm's calibration; no inference or input error in
any arm (the fallback-camera arm included). G1: the LTF control 0.9052 / 2.130 m against the stored anchor 0.9057 / 2.132. Both pass.

Criteria (pre-registered): (a) score change in [-0.20, -0.06] (private YR -0.125), (b) GT ratio in [1.4, 2.4] (private x1.84), (c) the
k = 10 drift rises on >= 12 of 16 vehicles, (d) the LTF control moves <= 0.02 under the same kind of change. Score CIs: scene bootstrap;
the log-clustered ones are in the CSV and agree in sign everywhere.

| arm | perturbation | score | d score [95 % CI] | GT m | GT ratio [CI] | at-fault events / km per event | zeros | dist k=10 m | vehicles up (of 16) | (a)+(b) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| A0 | control (YR10m10-F-s0, sh30, interpreted) | 0.953 | — | 1.00 | — | 2 / 1.21 | 3 | 1.06 |  |  |
| Pp05 | pitch told +0.5 deg (further down) | 0.931 | -0.022 [-0.053, -0.001] | 1.01 | x1.01 [0.87, 1.20] | 4 / 0.63 | 5 | 1.13 | 11 | no |
| Pp1 | pitch +1 | 0.901 | -0.052 [-0.098, -0.015] | 1.38 | x1.37 [1.04, 1.87] | 5 / 0.52 | 7 | 1.57 | 13 | no |
| Pp2 | pitch +2 | 0.804 | -0.148 [-0.214, -0.088] | 1.59 | x1.58 [1.16, 2.23] | 11 / 0.22 | 13 | 2.22 | 15 | **yes** |
| Pp3 | pitch +3 | 0.778 | -0.175 [-0.249, -0.108] | 2.17 | x2.16 [1.56, 3.08] | 13 / 0.19 | 15 | 2.85 | 15 | **yes** |
| Pm1 | pitch -1 | 0.939 | -0.014 [-0.050, +0.021] | 1.11 | x1.10 [0.98, 1.26] | 3 / 0.78 | 4 | 1.21 | 11 | no |
| Pm2 | pitch -2 | 0.973 | +0.021 [+0.002, +0.048] | 1.63 | x1.62 [1.32, 2.08] | 1 / 2.65 | 2 | 1.88 | 15 | no |
| Yp2 | yaw +2 (left) | 0.897 | -0.056 [-0.106, -0.016] | 1.28 | x1.27 [1.13, 1.47] | 4 / 0.60 | 8 | 1.39 | 15 | no |
| Ym2 | yaw -2 | 0.938 | -0.014 [-0.050, +0.020] | 1.13 | x1.13 [1.00, 1.32] | 3 / 0.80 | 4 | 1.20 | 9 | no |
| Yp4 | yaw +4 | 0.859 | -0.094 [-0.156, -0.039] | 1.62 | x1.61 [1.37, 1.96] | 9 / 0.27 | 12 | 1.80 | 16 | **yes** |
| Rp2 | roll +2 | 0.950 | -0.003 [-0.030, +0.024] | 0.94 | x0.93 [0.87, 0.99] | 2 / 1.20 | 3 | 0.99 | 7 | no |
| F095 | fx, fy x0.95 | 0.942 | -0.011 [-0.017, -0.005] | 0.92 | x0.92 [0.86, 0.98] | 2 / 1.16 | 3 | 1.01 | 5 | no |
| F105 | fx, fy x1.05 | 0.945 | -0.007 [-0.040, +0.023] | 1.07 | x1.07 [1.00, 1.16] | 2 / 1.26 | 4 | 1.14 | 11 | no |
| FY09 | fy x0.9 | 0.927 | -0.025 [-0.050, -0.009] | 0.91 | x0.91 [0.82, 0.99] | 3 / 0.75 | 4 | 1.01 | 2 | no |
| CXp40 | cx +40 px | 0.909 | -0.044 [-0.088, -0.009] | 1.15 | x1.14 [1.03, 1.30] | 4 / 0.60 | 7 | 1.26 | 14 | no |
| CYp40 | cy +40 px | 0.957 | +0.005 [-0.027, +0.037] | 1.30 | x1.30 [1.10, 1.58] | 2 / 1.22 | 3 | 1.44 | 13 | no |
| TZp03 | height +0.3 m | 0.696 | -0.256 [-0.318, -0.195] | 0.99 | x0.99 [0.77, 1.29] | 9 / 0.17 | 10 | 1.17 | 8 | no |
| TZm03 | height -0.3 m | 0.776 | -0.177 [-0.263, -0.093] | 5.86 | x5.84 [4.40, 8.16] | 16 / 0.21 | 21 | 14.44 | 16 | no |
| Dnom | distortion told: public median | 0.955 | +0.002 [-0.000, +0.005] | 1.04 | x1.03 [0.97, 1.10] | 2 / 1.23 | 3 | 1.13 | 10 | no |
| Dasset | distortion told: the asset's own | 0.955 | +0.003 [+0.000, +0.006] | 1.04 | x1.03 [0.97, 1.10] | 2 / 1.23 | 3 | 1.13 | 10 | no |
| DEF | MTGS fallback camera | 0.463 | -0.490 [-0.557, -0.420] | 0.77 | x0.77 [0.54, 1.11] | 18 / 0.06 | 19 | 1.17 | 9 | no |
| Rnom | mount ignored: nominal level, 1.45 deg down | 0.950 | -0.003 [-0.034, +0.028] | 0.94 | x0.94 [0.81, 1.10] | 1 / 2.45 | 3 | 1.00 | 7 | no |
| RE | image re-encoded only | 0.952 | -0.000 [-0.001, +0.000] | 1.00 | x1.00 [0.99, 1.00] | 2 / 1.21 | 3 | 1.06 | 7 | no |
| Z09c | image 10 % zoom, K follows (consistent rig) | 0.961 | +0.008 [-0.001, +0.026] | 1.05 | x1.04 [1.01, 1.08] | 1 / 2.43 | 2 | 1.11 | 12 | no |
| Z09m | image 10 % zoom, K unchanged | 0.920 | -0.033 [-0.059, -0.015] | 0.98 | x0.98 [0.89, 1.08] | 3 / 0.75 | 4 | 1.08 | 7 | no |
| APY_A0 | control APY10m10-AB-s0 (ap2) | 0.956 | — | 1.16 | — | 2 / 1.25 | 3 | 1.26 |  |  |
| APY_Pp2 | APY, pitch +2 | 0.889 | -0.067 [-0.126, -0.013] | 2.22 | x1.92 [1.43, 2.64] | 6 / 0.46 | 9 | 3.16 | 15 | **yes** |
| L0 | control LTF sample | 0.905 | — | 2.13 | — | 5 / 0.51 | 6 | 2.57 |  |  |
| Lcal | LTF, told pitch +2, yaw +2, f x1.05 | 0.906 | +0.001 [-0.000, +0.002] | 2.13 | x1.00 [1.00, 1.00] | 5 / 0.51 | 6 | 2.57 | 5 | no |
| LRE | LTF, re-encoded | 0.907 | +0.002 [-0.001, +0.005] | 2.12 | x1.00 [0.99, 1.01] | 5 / 0.51 | 6 | 2.56 | 6 | no |
| LZ | LTF, 10 % zoom (all three cameras) | 0.900 | -0.005 [-0.020, +0.014] | 1.79 | x0.84 [0.78, 0.90] | 5 / 0.49 | 6 | 2.19 | 2 | no |

"pitch +d" = the driver is told the optical axis points d deg further down than the render's; "yaw +d" = further left. The stored control
has only 2 at-fault events on these scenes, so km per event is noise below about 5 events.

## Read

- **What reproduces YR's private shape (a)-(d): a reported pitch 2-3 deg further down than the rendered camera** (P+2: -0.148, GT x1.58,
  15 / 16 vehicles; P+3: -0.175, x2.16), or a yaw 4 deg off (Y+4: -0.094, x1.61, 16 / 16). The loss is uniform drift (dist at k = 10 doubles,
  slope x2.2), mostly offroad zeros. The LTF sample does not read calibration at all (`Lcal` identical to its control up to batching
  noise), so (d) holds by construction. At-fault events overshoot private (km per event x0.18 against x0.67 on private).
- The response is one-sided: P-2 *raises* the score (+0.021) while still doubling GT distance; small errors (0.5-1 deg, focal +-5 %, roll
  2 deg, principal point) cost 0-0.05.
- Camera height is the most sensitive input (+-0.3 m: -0.26 / -0.18; -0.3 m sends 80 % of clips past 4 m), but neither direction has the
  private shape (GT unchanged or x5.8).
- **The distortion the renderer applies and never reports does not matter**: telling the driver the asset's own distortion changes the
  score by +0.003 [+0.000, +0.006]. Not worth an image change.
- The MTGS fallback camera (what a flat-format asset would report) collapses the driver (0.463, 18 at-fault events in 96) — far below
  private 0.80, so it is not what happened on the whole private suite.
- A consistent rig change is absorbed: a 10 % narrower lens with K following costs nothing (+0.008); the same lens with K unchanged -0.033;
  LTF under the 10 % zoom -0.005 (GT x0.84).
- **Amendment-1 ordering test fails.** If a ~2 deg reported-pitch error were the private mechanism, APY10m10-AB-s0 should lose more than
  YR under it (private: -0.295 against -0.125). Under P+2 APY loses -0.067 [-0.126, -0.013] (GT x1.92), less than YR's -0.148. The
  mechanism gets the size for YR but the family ordering backwards on score.
- **Ignoring the reported mount is free on public**: a fixed nominal rotation (level, 1.45 deg down) scores -0.003 [-0.034, +0.028],
  GT x0.94, although public mounts span 2.2 deg of pitch and 4 deg of yaw.

## Is the private suite likely to have a reported / rendered mismatch?

Nothing in the shipped code produces one: the report and the render read the same pickle fields (K, rotation, translation), apart from the
distortion (harmless, above) and the flat-pickle fallback (catastrophic, above). The renderer ignores the runtime's camera override
(AUD1). One image-side mechanism exists for every driver: in the closed loop (physics NONE) the ego is rendered level and at the hand-over
altitude (rollout.asl: ego quaternions about z only, z constant), while the logged car pitches with the road. On public scenes this does
not hurt YR (66 scenes with mean logged pitch <= -1.5 deg: 0.922 against 0.930 on flat scenes; altitude change > 0.5 m: 0.954),
`asset_grade_public.csv`. So a private suite with steeper roads is not supported either.
