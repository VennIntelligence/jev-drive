# AUD1-LHT: does serving the left-hand-traffic flag help APY10m10-AB-s0 on the public Singapore scenes? (2026-10-11)

Follow-up to [aud1_driver_assumptions.md](aud1_driver_assumptions.md) ("Extra: traffic side"). That audit found the checkpoint was trained with
the left-hand flag on Singapore rows but is always served right-hand (`ap2_driver.py:84, 124`; `SH30_LHT` unset in the image). This note
measures the closed-loop effect of serving the flag on the public Singapore scenes. Descriptive; driver code unchanged, nothing submitted.

## What was run

- **Scenes.** The 220 public scenes with `city: sg-one-north` in `$DATA_DIR/datasets/alpasim_nuplan/navtest/configs/*.yaml` (of 1491:
  Las Vegas 481, Boston 467, Pittsburgh 323). They come from 2 logs (`2021.10.06.07.26.10_veh-52` 106 scenes, `2021.10.06.08.16.17_veh-52` 114)
  and 13 log segments. List: [aud1/lht/sg220.txt](aud1/lht/sg220.txt) (box: `$DATA_DIR/runs/alpasim/aud1_lht/lists/sg220.txt`).
  No non-Singapore scene was run.
- **Served configuration.** `run.sh <dir> ap2` with `AP2_TAG=APY10m10-AB-s0 SH30_COMPILE=0` (interpreted passes, as in the image),
  `AP2_COLD` unset (backwarp), `SH30_SYNTH` unset (gpu), every serving switch off (driver log: `serving: JEV_VCONT 0, JEV_LEAD 0, JEV_BASE 0`,
  `APY10m10-AB-s0 (backwarp, route arm False)`). The CF1 runtime overrides (`+e2e_challenge_nuplan=full`, 8 concurrent rollouts). Code at `cbfce0f6`.
- **Arms.** `lht1` = `SH30_LHT=1`; `lht0` = `SH30_LHT=0`, otherwise identical; `stored` = the CF1 run that holds all 220 scenes
  (`cf1/a/runs/APY10m10-AB-s0@fresh-cf1fresh/20261009-203623`, code `c9d2e07b`, flag off). The `lht0` arm was added because the stored run used
  **compiled** passes (`SH30_COMPILE` default 1, inductor), not the image's interpreted ones. On the check scene the interpreted run's plans
  drift from the stored ones by up to 1 m by decision 8. Over the 220 scenes `lht0` and `stored` differ by +0.0002 in score, with identical zeros,
  so the stored run is a valid control in practice. The comparison below uses `lht1` - `lht0`.
- **Flag check first** (2 pool jobs, one scene, `veh-52_00422_00728-fb1d534c`, `aud1_lht/verify/lht{0,1}`). The driver sees identical inputs at
  k = 0, and the flag moves the 4 s plan point there by (0.38, 0.94) m. The audit's open-loop test found 0.94 m on the same scene. So the
  environment path `pool job env -> run.sh -> drivers/ap2.sh -> ap2_driver.py` reaches the model. In closed loop that scene goes from 0
  (corridor exit) to 1.0.
- **Pool jobs.** `1011-222151-752a` (lht1, card 4) and `1011-222151-dd29` (lht0, card 5): 220 rollouts each, 0 failed runs. Each has 219
  sessions and 2190 `drive` calls, with no inference or input errors; the one route-fold scene never starts a session. About 11 min each.
  Total with the check jobs: about 0.4 card-hour.
- **Analysis.** [scripts/aud1_lht_report.py](../scripts/aud1_lht_report.py) (box copy `aud1_lht/code/`). Run-level metrics are defined as in
  `aud1_collect.py` (src/eval at 0bb4c4b). CIs are a cluster bootstrap over the 13 log segments (10 000 draws, seed 0, `jevdrive.stats.paired`
  for the score; the same resample on both arms for the run-level ratios). Two logs are too few clusters for a log-level bootstrap.
  Numbers: [aud1/lht/summary.json](aud1/lht/summary.json), [aud1/lht/per_scene.csv](aud1/lht/per_scene.csv). Box outputs:
  `$DATA_DIR/runs/alpasim/aud1_lht/{sg220,verify,results}/`.

## Results (220 Singapore scenes)

| arm | avg scene score [95 % CI] | zeros: offroad / corridor / route fold | at-fault events (collision / offroad) | at-fault distance km [CI] | GT distance m [CI] | clips >= 4 m |
|:--|--:|--:|--:|--:|--:|--:|
| lht1 (flag on) | **0.916** [0.878, 0.950] | 17: 13 / 3 / 1 | 13 (0 / 13) | 0.413 [0.267, 0.819] | **1.028** [0.848, 1.280] | 5.9 % |
| lht0 (flag off, interpreted) | 0.903 [0.867, 0.939] | 20: 14 / 5 / 1 | 14 (0 / 14) | 0.400 [0.268, 0.763] | 1.323 [1.116, 1.578] | 10.5 % |
| stored CF1 (flag off, compiled) | 0.903 [0.867, 0.939] | 20: 14 / 5 / 1 | 14 (0 / 14) | 0.399 [0.267, 0.761] | 1.316 [1.110, 1.570] | 10.5 % |

Paired, flag on minus flag off (219 clips with metrics):

| quantity | lht1 - lht0 [95 % CI, 13 segments] | lht1 - stored |
|:--|--:|--:|
| avg scene score | **+0.012 [-0.014, +0.034]** | +0.012 [-0.014, +0.035] |
| GT distance (m) | **-0.295 [-0.365, -0.198]** | -0.287 [-0.354, -0.195] |
| at-fault distance (km) | +0.013 [-0.114, +0.172] | +0.014 [-0.113, +0.173] |
| scenes up / down / equal | 10 / 18 / 192 | 12 / 16 / 192 |
| zero -> positive / positive -> zero | 6 / 3 | 6 / 3 |

Details (`per_scene.csv`):
- The score gain comes from the zeros. 6 scenes are recovered (3 corridor exits, 3 offroad), all to 1.0, and 3 scenes that scored 1.0 go
  offroad. On the 197 scenes that are non-zero in both arms the score moves by -0.002 on average (from -0.18 to +0.09). The flag slightly
  shortens progress: 24.5 m travelled per clip against 25.5 m.
- By log: `07.26.10` goes from 0.919 to 0.955 (5 of the 6 recoveries) and `08.16.17` from 0.889 to 0.878 (2 of the 3 new zeros). The sign is
  not stable across the two logs.
- The path metric moves consistently. GT distance falls on both logs (-0.37 m and -0.22 m), and on the 194 clips that stay under 4 m in both
  arms by 0.21 m (median -0.06 m). Clips leaving the logged path by 4 m drop from 23 to 13. With the flag on, the driver stays closer to the
  logged (left-hand) path.
- All at-fault events on these scenes are offroad, both arms. There are no at-fault collisions.

Against the Singapore gap (public Singapore 0.903 against Las Vegas 0.977, `aud1_driver_assumptions.md`): the flag recovers +0.012 of the
0.074 score gap (about 17 %, CI includes 0). It brings the Singapore GT distance from 1.32 m down to 1.03 m, below the 1.13 m all-scene mean.

## Verdict

On the Singapore scenes the flag does not hurt and probably helps. The score difference is +0.012 [-0.014, +0.034] (6 zeros recovered,
3 lost, the second log slightly worse). The at-fault distance does not change (0.41 against 0.40 km). The GT distance, one of the two
leaderboard metrics, falls by 0.30 m [0.20, 0.37], and the clips leaving the path by 4 m nearly halve. That is a clear, consistent fix of
the train / serve mismatch on the path metric. It recovers only a small, non-significant part of the score gap, so Singapore's 0.903 is
mostly not about the flag. **It belongs in the next submission image only as a per-scene signal, not as a global switch.** Serving
`SH30_LHT=1` everywhere would tell the model that every US scene is left-hand. That arm was not run (US scenes were out of scope), and the
audit's one US scene moved by 0.02 m, which is not enough to clear it. The driver has no map or city input: the scene id is empty in
benchmark runs, and veh-52 is only a proxy through its calibration. So putting the flag in the image needs (a) a rule that detects
left-hand traffic from what the driver receives, and (b) a run of that rule on all 1491 public scenes showing no loss on the 1271 US
scenes. Until then the image should stay as it is.
