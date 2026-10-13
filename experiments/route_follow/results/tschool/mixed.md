# RSGmixPAIT-L18 (Amendment 4: `--on-mode mixed`), 2026-10-13

Prereg: [../../plans/2026-10-13-turning-school-prereg.md](../../plans/2026-10-13-turning-school-prereg.md) Amendment 4. Chain: [../../scripts/rf_tschool_mixed.sh](../../scripts/rf_tschool_mixed.sh).
Command (RSGPAIT-L18's, only the mode changed): `rm_train.py train --tag RSGmixPAIT-L18-s{k} --init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on on1_RSGPAIT --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode mixed --on-weight pool=p:1.0306,topo=straight_exit:2,topo=straight_road:2 --seed k`
Mode assignment checked before submission: pool tab modes r = self (17 429 S rows), p = both (5 160 PAI rows); `rm_train.py mixcheck` with these flags: max abs diff self 0.0, both 0.0, pass.

## Verdicts (s0 / s1)

| line | RSGmixPAIT | RSG-L18 | RSGPAIT | RSGboth | verdict |
|:--|:--|:--|:--|:--|:--|
| held 460 @18 mean >= RSG - 0.01 | 0.8338 / 0.8379 | 0.8656 / 0.8676 | 0.9045 / 0.9057 | 0.9020 / 0.8922 | **FAIL** (-0.032; paired on 452 -0.031 [-0.046, -0.018]) |
| held zeros <= RSG-s0 + 3 (37) | 37 / 33 | 34 / 34 | 37 / 36 | 41 / 46 | PASS (37 = line) |
| public 1479 @10 zeros <= RSG + 5 | 49 / 45 | 46 / 44 | 56 / 54 | 58 / 56 | PASS (+3 / +1; paired +2.0 [-6.9, +11.8]) |
| public mean ties RSG | 0.9305 / 0.9318 | 0.9473 / 0.9487 | 0.9500 / 0.9495 | 0.9522 / 0.9539 | **FAIL** (-0.017 [-0.024, -0.011]; vs base YR10m10 -0.001, i.e. the RSG gain is gone) |

Turn completion (rf_follow, 24 held turns): 70.8 % / 62.5 % vs RSG 66.7 / 58.3, RSGPAIT 87.5 / 87.5, RSGboth 91.7 / 87.5 (mixed vs RSG +4.2 [-11, +17] pp for s1: no reliable change). Inside-cut share (> 1.0 m): 58.3 / 62.5 % vs RSG 54.2 / 54.2 (paired +4.2 / +8.3 pp, CI touches 0). GT distance on held 1.11 / 1.08 m = RSG's 1.14 (RSGPAIT 1.75, RSGboth 1.78).

Reading: with the S rows on `self` and imitation only on the 30 % PAI rows the policy does not take up the human-timed behaviour (GT distance, turn completion, held mean gain all stay at the RSG level or below) and it loses RSG's own public-mean gain, while keeping RSG's zero counts. The held mean and the drive-farther effect of RSGPAIT / RSGboth come from `both` on the S rows (as the RSGboth vs RSGPAIT comparison already suggested). The mixed arm fails step 2 on both mean lines; it does not displace RSGPAIT as the best arm of the ladder, and nothing goes to the PAI offline ranking or the Tokyo exam.

Note: the training jobs were submitted without `--gpus 1,2,3,4,5` and ran on card 0 (the rf_tschool.sh template had no such flag; the script now has it); the 10 guard chunks per seed were pinned to 1-5.

Files: mixed_held_raw.txt, mixed_held18_confirm.{md,json}, mixed_follow.md, mixed_inside.json, mixed_guard_confirm.{md,json}, mixed.json.
