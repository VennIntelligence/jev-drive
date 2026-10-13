# RSGbothSB-L18 (side-bias fix 1: right-turn rows outside : inside the human path drawn 1 : 1), 2026-10-13

Prereg: [plans/2026-10-13-rsgboth-prereg.md](../../plans/2026-10-13-rsgboth-prereg.md) Amendment 2. Question from
[research/route-follow/side-bias.md](../../../../research/route-follow/side-bias.md): is RSGboth's right-turn inside cut caused by the pool's
state mix (on1_L18RB0 right-turn rows outside / inside the human path 321 : 77)?

Command = RSGboth-L18-s{k}'s exact command plus `--on on1_L18RB0s --on-weight rside=R_in:2,rside=R_out:0.48`, seeds 0 and 1
(chain [scripts/rf_rsgbsb.sh](../../scripts/rf_rsgbsb.sh), pool jobs on cards 1-5). Pool on1_L18RB0s = on1_L18RB0 plus tab keys `tside` /
`pside` / `rside` ([scripts/rf_sidetag.py](../../scripts/rf_sidetag.py) pool; gate R1 PASS, `sidetag_r1.json`). rm_train `--on-weight` reads
string keys of the pool's tab.npz and multiplies one factor per matching KEY=VALUE term (no AND, hence the combined key `rside`). Dry run
with rm_train's parsing and generator, 1000 batches x 13: R_out / R_in drawn 107 / 98 (uniform 245 / 56), every other class at its uniform
rate; the trainer's own smoke log shows the same draw probabilities (min 0.482 / max 2.010 x uniform). Weighted mean right-turn push at 2 s
-0.38 -> -0.22 m (left -0.14).

## Verdicts

| line | RSGbothSB s0 / s1 (mean) | RSGboth s0 / s1 | RSG s0 / s1 | verdict |
|:--|:--|:--|:--|:--|
| 1. right-turn plan lat 2 s vs RSG, paired by scene (34 scenes) | +0.12 [-0.10, +0.25] / -0.12 [-0.40, +0.02]; mean +0.00 [-0.25, +0.13] | -0.33 [-0.66, -0.15] (mean) | 0 | **PASS: the right-turn inside shift is gone** |
| 1b. vs RSGboth, right / left | right +0.34 [+0.22, +0.46], left +0.00 [-0.04, +0.04], straight 0.00 | | | right moved out by the full RSGboth shift, left untouched |
| 1c. ledger loose R inside peak (29 turns) | SB - RSG -0.11 [-0.18, -0.00]; SB - RSGboth +0.22 [+0.14, +0.33]; cut > 1 m 12 % vs 22 % (RSGboth) vs 9 % (RSG) | -0.61 | -0.28 | about two thirds of RSGboth's -0.32 removed, a residual -0.11 |
| 1d. ledger loose L outside peak (60 turns) | SB - RSGboth -0.00 [-0.04, +0.04]; SB - RSG +0.22 [+0.09, +0.35] | 1.11 | 0.89 | left late / wide unchanged: not this mechanism |
| 2. turn completion >= 85 % (24 held turns) | 75.0 % / 87.5 % (mean 81.3 %) | 91.7 / 87.5 % | 66.7 / 58.3 % | **FAIL on s0 / mean** (s0 right turns 4 / 8: off, short x 2, event; s1 identical to RSGboth) |
| 3. held zeros <= RSG + 3, or < RSGboth (460) | 44 / 39 (41.5) | 41 / 46 | 34 / 34 | s0 FAIL (above both), s1 weak PASS (< RSGboth); mean weak PASS (paired on 452: SB - RSGboth -2.0 [-4.3, 0.0]) |
| held mean (460) | 0.8948 / 0.9076 | 0.9020 / 0.8922 | 0.8656 / 0.8676 | SB - RSGboth (452, seed mean) +0.004 [-0.001, +0.010] |
| 4. public guard 1479 @10: zeros <= RSG + 5, mean >= RSG | zeros 45 / 51 (48), mean 0.9609 / 0.9577 (0.9593) | 58 / 56, 0.9522 / 0.9539 | 46 / 44, 0.9473 / 0.9487 | **PASS** on the seed mean (zeros +3.0 [-4.8, +11.0] vs RSG, -9.0 [-15.8, -3.3] vs RSGboth; mean +0.011 [+0.006, +0.017] vs RSG); s1 alone 51 > 49 |

Reading: the intervention confirms side-bias.md's mechanism causally. Rebalancing only the right-turn rows by path side removes RSGboth's
right-turn inside shift of the plan (paired diff vs RSG +0.00 m, CI includes 0; vs RSGboth +0.34 m) and leaves left turns and straight road
exactly where RSGboth had them. As a side effect the public-guard zeros fall from RSGboth's 57 to 48 (RSG 45), and held zeros on the seed
mean fall by 2. The cost is in right-turn completion on seed 0 (4 of 8 right turns not completed vs 8 / 8 for RSGboth; s1 keeps 8 / 8) and
a right-turn exit that is wider again (ledger exit offset +0.32 m vs RSGboth -0.28, RSG +0.66). One seed of two fails the completion line,
so the fix is a confirmed mechanism, not yet a usable arm.

Caveats: 8 strict right turns and 34 right-turn scenes only, wide CIs; held plan lateral is the 2 s plan point (side-bias.md's metric), the
ledger peaks are over the whole turn. The guard is the 1479 common scenes of the 1491 list.

Files: `held_raw.txt` (460), `held18/confirm.{md,json}` (452 paired), `guard/confirm.{md,json}`, `follow.{md,json}`, `inside.json`,
`side.{md,json}` (rf_sidetag.py read), `held_RSGbothSB-L18-s{0,1}.csv.gz` (rf_sidebias held), `ledger/` (rf_ledger with RSG / RSGboth /
RSGbothSB), `tax/` (rf_tax report of the new tags), `sidetag_r1.json` (R1 + dry run). Box: `$DATA_DIR/runs/route_follow/rsgboth_sb/`.
