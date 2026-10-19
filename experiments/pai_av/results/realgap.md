# REAL-GAP: real PAI-AV footage vs the NuRec render, same clips and times

50 test281 clips, 700 decisions (log replay, 1 Hz dumps with 8 rendered slots, k >= 20); 50 non-exam clips, 900 real decisions. CIs: scene-cluster bootstrap (jevdrive.stats). Measurement only: no test281 real frame is in any training row or cache.

## A. Conversion check (the driver's f-theta -> openpilot path on real frames)

| Check | median [p10, p90] or range |
|---|---|
| camera rotation, PAI-AV extrinsics vs the session's NuRec rig_to_camera (deg) | 0.328 [0.168, 0.671] (max 1.141) |
| camera position difference (m) | 0.106 [0.0819, 0.132] |
| principal point difference (px, native) | 3.02 [0.362, 6.8]; dcy 1.63 [-0.302, 5.95] |
| f-theta radius difference at 20 / 40 / 60 deg (px) | -3.27 [-4.5, -1.26] / -6.77 [-9.24, -2.8] / -9.56 [-13.1, -3.87] |
| native size PAI-AV / NuRec / decoded video | ['[1920, 1080]'] / ['[1920, 1080]'] / ['(1920, 1080)'] |
| NuRec first front-wide frame minus nearest PAI-AV frame time (us) | 0 [0, 0] |
| real frame vs slot time, worst (us) | 14819 |
| image lag scan: best real frame offset (frames -3 .. +3) | [18, 18, 41, 28, 20, 9, 16] |
| road-frame shift render -> real (deg, x / y) | 0.0195 [-0.163, 0.236] / 0.0974 [-0.313, 0.576] |
| road-frame shift real via NuRec spec -> via PAI-AV calib (deg, x / y) | -0.00737 [-0.096, 0.0759] / 0.101 [-0.0464, 0.342] |
| road luma MAE render vs real / real NuRec-spec vs PAI-AV-calib (0-255) | 13.5 [8.76, 19.4] / 4.68 [2.24, 12.3] |

## B. Open-loop plans on render vs real (same ego row, route memory and command; only the frames differ)

Replay check: NTV0-L18-s0 on the dumped render slots vs the plan the driver logged: median 0.0073 m, p99 0.0625 m, max 0.1163 m.

| Model | Input | ADE (m) | FDE 4 s (m) | v4 plan - human (m/s) | abs v4 err (m/s) | lat 4 s (m) |
|---|---|---|---|---|---|---|
| NTV0-L18-s0 | render | 2.533 [1.957, 3.226] | 4.901 [3.875, 6.143] | -0.670 [-1.071, -0.329] | 1.189 [0.930, 1.504] | 0.058 [-0.089, 0.206] |
| NTV0-L18-s0 | real | 2.945 [2.240, 3.785] | 5.663 [4.394, 7.171] | -0.927 [-1.384, -0.530] | 1.376 [1.052, 1.759] | 0.056 [-0.117, 0.228] |
| NTV0-L18-s0 | realN | 2.778 [2.131, 3.569] | 5.365 [4.188, 6.790] | -0.772 [-1.215, -0.393] | 1.303 [1.005, 1.666] | 0.056 [-0.118, 0.228] |
| RSG-L18-s0 | render | 2.292 [1.758, 2.947] | 4.445 [3.496, 5.633] | -0.585 [-0.960, -0.273] | 1.077 [0.835, 1.377] | 0.103 [-0.031, 0.243] |
| RSG-L18-s0 | real | 2.654 [1.996, 3.460] | 5.100 [3.919, 6.556] | -0.808 [-1.239, -0.442] | 1.237 [0.938, 1.604] | 0.104 [-0.047, 0.262] |
| RSG-L18-s0 | realN | 2.505 [1.905, 3.258] | 4.836 [3.748, 6.207] | -0.664 [-1.082, -0.316] | 1.173 [0.897, 1.517] | 0.101 [-0.053, 0.258] |

Paired (NTV0-L18-s0):

| Quantity | mean [95 % CI] |
|---|---|
| real - render ADE | 0.412 [0.215, 0.623] |
| real - render v4 bias | -0.257 [-0.351, -0.168] |
| real - render abs v4 err | 0.186 [0.101, 0.279] |
| plan distance real vs render | 0.897 [0.734, 1.085] |
| plan distance NTV0 vs RSG on render (yardstick) | 0.350 [0.299, 0.406] |
| plan distance real PAI-AV calib vs NuRec spec | 0.401 [0.322, 0.493] |
| plan distance real vs render, day | 0.707 [0.567, 0.859] |
| plan distance real vs render, night | 1.103 [0.812, 1.427] |
| real - render ADE, day | 0.256 [0.067, 0.463] |
| real - render ADE, night | 0.582 [0.230, 0.944] |

## C. Frozen Cinque tokens (t0 slot, 32 x 512)

| Cosine (flattened) | median [p10, p90] |
|---|---|
| render vs real (same clip, time) | 0.888 [0.824, 0.931] |
| real NuRec-spec vs real PAI-AV-calib | 0.984 [0.948, 0.992] |
| real t vs real t + 1 s (yardstick) | 0.819 [0.679, 0.905] |
| render t vs render t + 1 s | 0.848 [0.71, 0.917] |
| real vs real of another clip (null) | 0.311 [0.121, 0.552] |

| Linear CKA (mean-pooled 512 unless said) | value |
|---|---|
| render vs real (paired) | 0.904 |
| render vs real (pairing permuted, null) | 0.029 |
| render vs real, flattened tokens | 0.962 |
| real NuRec-spec vs PAI-AV-calib | 0.993 |

Domain classifier (logistic, scene-grouped 5-fold) render vs real: accuracy 0.966 (0.5 = indistinguishable).

## D. Probe transfer (ridge on the flattened t0 tokens; targets from the PAI-AV egomotion)

| Target | Train | Test real R2 | Test render R2 | render - real squared error [95 % CI] |
|---|---|---|---|---|
| v0 | real non-exam (900) | 0.977 | 0.978 | -0.098 [-0.740, 0.445] |
| v0 | exam render, scene 5-fold | 0.926 | 0.937 | -0.802 [-1.980, 0.105] |
| v0 | exam real, scene 5-fold | 0.939 | 0.934 | 0.395 [-0.265, 1.088] |
| dv3 | real non-exam (892) | 0.570 | 0.480 | 0.548 [0.172, 0.967] |
| dv3 | exam render, scene 5-fold | 0.466 | 0.433 | 0.199 [-0.135, 0.568] |
| dv3 | exam real, scene 5-fold | 0.518 | 0.413 | 0.640 [0.296, 1.016] |
| dyaw3 | real non-exam (892) | 0.749 | 0.755 | -1.569 [-6.224, 3.296] |
| dyaw3 | exam render, scene 5-fold | 0.771 | 0.780 | -2.443 [-9.034, 3.694] |
| dyaw3 | exam real, scene 5-fold | 0.767 | 0.789 | -5.886 [-13.237, 0.292] |
