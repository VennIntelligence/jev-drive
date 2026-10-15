# FLOW1 in closed loop: flow head (FMH-F) vs its regression twin (RGH-F), held 460 @18

Prereg `experiments/route_follow/plans/2026-10-15-mode-averaging-prereg.md` (Reading 2). Served by the AlpaSim sh30 driver with the trajectory head (thead.pt); G-eq passed on all four tags (`gate.txt`: served vs bench head poses, mean max 0.003-0.004 m, p99 0.012-0.015 m). Paired by scene, CIs clustered by log, B 10 000.

**F1** (strict held turns completed, 24 turns, 2-seed mean): FM 0.5, RG 11.0, FM - RG -10.5 [-16.5, -5.5]: **refuted**. Per seed: FMH-F-s0 - RGH-F-s0 -9 [-15, -4]; FMH-F-s1 - RGH-F-s1 -12 [-18, -6]. Completed per tag: FMH-F-s0 1 / 24, FMH-F-s1 0 / 24, RGH-F-s0 10 / 24, RGH-F-s1 12 / 24, RSG-L18-s0 16 / 24, RSG-L18-s1 14 / 24.

**F2** (approach-window deficit, per-turn median, 2-seed mean per scene): FM +0.27, RG +0.09, FM - RG +0.18 [+0.07, +0.26] (23 turns): **refuted**. Arc-matched (Amendment 3): FM +0.08, RG +0.05, FM - RG +0.03 [-0.05, +0.19].

**F3** (held 460 @18, FM - RG; mean score, zeros, at-fault zeros, GT distance m; from ol_confirm, see `held18.md`):

- FMH-F - RGH-F: score -0.2325 [-0.2756, -0.1848], zeros +110.0 [+90.4, +128.1], at-fault +30.5 [+14.5, +45.5], GT +0.842 [+0.660, +0.996]
- FMH-F-s1 - RGH-F-s1: score -0.1950 [-0.2452, -0.1417], zeros +92.0 [+69.4, +113.8], at-fault +15.0 [-5.4, +35.4], GT +0.704 [+0.532, +0.861]
- FMH-F - RSG-L18: score -0.4202 [-0.4656, -0.3708], zeros +203.5 [+181.7, +223.7], at-fault +70.0 [+53.0, +86.8], GT +1.887 [+1.702, +2.081]
- RGH-F - RSG-L18: score -0.1877 [-0.2312, -0.1446], zeros +93.5 [+75.0, +113.7], at-fault +39.5 [+28.3, +51.8], GT +1.045 [+0.877, +1.221]

**K = 8 flow samples** (noise rows 0-7, logged at each closed-loop decision; full = sample dpsi >= 0.6 x human 4 s dpsi):

- failing: 65 decisions (16 turns), >= 1 full sample on 0.25 (arc-matched 0.45), mean full samples 0.86 of 8, histogram {0: 49, 1: 1, 2: 1, 3: 9, 4: 2, 6: 3}, median sample sd 4.4 deg: partial
- failed_all: 197 decisions (23 turns), >= 1 full sample on 0.72 (arc-matched 0.82), mean full samples 5.09 of 8, histogram {0: 56, 1: 2, 2: 3, 3: 9, 4: 4, 5: 2, 6: 11, 7: 5, 8: 105}, median sample sd 3.5 deg: mode exists, selection missing
- completed: 7 decisions (1 turns), >= 1 full sample on 0.86 (arc-matched 1.00), mean full samples 6.29 of 8, histogram {0: 1, 4: 1, 8: 5}, median sample sd 2.7 deg: mode exists, selection missing

Amendment 4 (post-hoc, descriptive): on the 52 straight decisions (median sample heading change < 3 deg, > 10 m ahead) of the FM strict-turn rollouts, the 4 s lateral offset (left +) per noise row 0-7 is -0.91, -0.45, -0.81, -0.47, -0.64, -0.38, -0.14, -0.35 m; mean distance from the 8 samples' median 0.43, 0.08, 0.33, 0.09, 0.16, 0.10, 0.34, 0.13 m (row 0 is the served plan).
