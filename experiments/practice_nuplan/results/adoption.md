# Adoption: NTVXS-L18-s0 and JTXS-L18-s1 (PN prereg supplement 3)

Native = arena held split, no hold (decision 371). Drift = closed-loop signed lateral offset to the route, arm minus parent, non-turn held 460 scenes, decisions 11-17, both before their first zero (decision 377's measure; NTVX - NTV0 was -0.110 [-0.131, -0.085]).

| arm | parent | line | measured | pass |
|---|---|---|---|:--|
| NTVXS-L18-s0 | NTV0-L18-s0 | nuPlan native arena held vs NTV0-L18-s0 native: >= +0.005, CI low > 0 | +0.0103 [+0.0034, +0.0188] (2228 windows; 0.7803 vs 0.7699; zeros 469 / 494) | **yes** |
| NTVXS-L18-s0 | NTV0-L18-s0 | held 460 @18 zeros <= NTV0-L18-s0 + 3 | 33 vs 32 (mean 0.8814) | **yes** |
| NTVXS-L18-s0 | NTV0-L18-s0 | drift vs NTV0-L18-s0 > -0.05 m | +0.047 [+0.024, +0.071] m (2757 decisions) | **yes** |
| NTVXS-L18-s0 | NTV0-L18-s0 | PAI 281 ACC + GAP vs NTV0-L18-s0 + ACC + GAP: >= +0.01, CI low > 0 | -0.0221 [-0.0424, -0.0034] (279 scenes; 0.7212) | no |
| NTVXS-L18-s0 | NTV0-L18-s0 | PAI zeros <= NTV0-L18-s0's + 2 (same scenes) | 66 vs 59 | no |
| JTXS-L18-s1 | JTN-L18-s1 | nuPlan native arena held vs JTN-L18-s1 native: >= +0.005, CI low > 0 | +0.0016 [-0.0130, +0.0200] (2228 windows; 0.8925 vs 0.8909; zeros 204 / 210) | no |
| JTXS-L18-s1 | JTN-L18-s1 | held 460 @18 zeros <= JTN-L18-s1 + 3 | 38 vs 36 (mean 0.8580) | **yes** |
| JTXS-L18-s1 | JTN-L18-s1 | drift vs JTN-L18-s1 > -0.05 m | +0.045 [+0.017, +0.077] m (2699 decisions) | **yes** |
| JTXS-L18-s1 | NTV0-L18-s0 | PAI 281 ACC + GAP vs NTV0-L18-s0 + ACC + GAP: >= +0.01, CI low > 0 | -0.0393 [-0.0625, -0.0187] (279 scenes; 0.7076) | no |
| JTXS-L18-s1 | NTV0-L18-s0 | PAI zeros <= NTV0-L18-s0's + 2 (same scenes) | 65 vs 58 | no |

Reported, not gated:

| comparison | value |
|---|---|
| native arena NTVXS-L18-s0 - NTV0-L18-s0 | +0.0103 [+0.0034, +0.0188] (2228 w; zeros 469 / 494) |
| native arena NTVXS-L18-s0 - JTN-L18-s1 | -0.1107 [-0.1549, -0.0732] (2228 w; zeros 469 / 210) |
| native arena JTXS-L18-s1 - NTV0-L18-s0 | +0.1226 [+0.0784, +0.1745] (2228 w; zeros 204 / 494) |
| native arena JTXS-L18-s1 - JTN-L18-s1 | +0.0016 [-0.0130, +0.0200] (2228 w; zeros 204 / 210) |
| native arena JTN-L18-s1 - NTV0-L18-s0 | +0.1210 [+0.0826, +0.1649] (2228 w; zeros 210 / 494) |
| PAI 281 NTVXS-L18-s0 - NTV0-L18-s0 (ACC + GAP) | -0.0221 [-0.0424, -0.0034] (zeros 66 / 59); vs RSG + ACC + GAP +0.0290 [-0.0018, +0.0610] |
| PAI 281 JTXS-L18-s1 - JTN-L18-s1 (ACC + GAP) | +0.0051 [-0.0134, +0.0248] (zeros 65 / 68); vs RSG + ACC + GAP +0.0154 [-0.0171, +0.0490] |
| PAI 281 JTXS-L18-s1 - NTV0-L18-s0 (ACC + GAP) | -0.0393 [-0.0625, -0.0187] (zeros 65 / 58); vs RSG + ACC + GAP +0.0154 [-0.0181, +0.0492] |
| drift NTVXS-L18-s0: k 5-10 / k 11-17 / turn k 11-17 | +0.026 [+0.016, +0.035] / +0.047 [+0.024, +0.071] / +0.101 [-0.058, +0.298] |
| drift JTXS-L18-s1: k 5-10 / k 11-17 / turn k 11-17 | +0.024 [+0.014, +0.036] / +0.045 [+0.017, +0.077] / +0.022 [-0.150, +0.202] |
| drift JTXS-L18-s1 vs NTV0: k 5-10 / k 11-17 / turn k 11-17 | +0.038 [+0.016, +0.060] / +0.087 [+0.040, +0.132] / +0.331 [+0.033, +0.703] |
| drift NTVXS-L18-s0 vs JTN-L18-s1: k 5-10 / k 11-17 / turn k 11-17 | +0.012 [-0.003, +0.030] / +0.004 [-0.032, +0.040] / -0.204 [-0.359, -0.033] |
| held 460 NTVXS-L18-s0 - RSG-L18-s0 | +0.0158 [+0.0044, +0.0268] (zeros 33 / 34) |
| held 460 NTVXS-L18-s0 - NTV0-L18-s0 | -0.0062 [-0.0175, +0.0037] (zeros 33 / 32) |
| held 460 NTVXS-L18-s0 - JTN-L18-s1 | +0.0148 [-0.0060, +0.0479] (zeros 33 / 36) |
| held 460 JTXS-L18-s1 - RSG-L18-s0 | -0.0075 [-0.0241, +0.0066] (zeros 38 / 34) |
| held 460 JTXS-L18-s1 - NTV0-L18-s0 | -0.0295 [-0.0476, -0.0151] (zeros 38 / 32) |
| held 460 JTXS-L18-s1 - JTN-L18-s1 | -0.0085 [-0.0251, +0.0158] (zeros 38 / 36) |

NTVXS-L18-s0: nuPlan **adopt**, PAI **not adopted**.
JTXS-L18-s1: nuPlan **not adopted**, PAI **not adopted**.
Stop3 was not read (reference only, not cheap: 36 more arena reads).
