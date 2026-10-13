# RSGboth-L18 (RSG-L18 recipe, `--on-mode self` -> `both`), 2026-10-13

Command (recovered from RSG-L18-s0's pool log, `$DATA_DIR/runs/route_follow/pool/rsg-s0/log.txt`; pool job name rsg-s0, pool `on1_L18RB0`, init `RB0-F-s0`), only the mode changed:
`rm_train.py train --tag RSGboth-L18-s{k} --init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode both --on on1_L18RB0 --seed k`
Chain: scripts/rf_rsgboth.sh (jobs in the box's rsgboth/jobs.txt), inside-cut: scripts/rf_inside.py.

## Verdicts

| line | RSGboth s0 / s1 | RSG-L18 s0 / s1 | verdict |
|:--|:--|:--|:--|
| held mean >= RSG - 0.01 | 0.9020 / 0.8922 (460), 0.9179 / 0.9080 (452 confirm) | 0.8656 / 0.8676, 0.8809 / 0.8830 | PASS (paired +0.037 [+0.016, +0.062] / +0.025 [+0.003, +0.048]) |
| held zeros <= RSG + 3 | 41 / 46 (460) | 34 / 34 | FAIL (+7 [0, 13.5] / +12 [4.1, 19.7] on 452; at-fault +3 / +11) |
| turn completion (24 held turns) | 91.7 % [80, 100] / 87.5 % [71, 100] | 66.7 / 58.3 % | up: +25.0 pp [+4.8, +48.0] / +29.2 [+5.9, +51.6]; vs RB0-F-s0 +37.5 [+16, +57] / +33.3 [+16, +48]; outcomes 22/1/0/1 and 21/1/1/1 (completed/event/off/short): the slow-"short" failures are gone |
| inside-cut share on held turns (ego > 1.0 m inside the route in the junction stretch) | 54.2 % / 58.3 % | 54.2 % / 54.2 % | no change: paired diff 0.0 [-16.7, +16.7] / +4.2 [-8.3, +16.7] pp (at 0.5 m: 79 / 79 % vs 75 / 67 %, +4.2 [-8, +21] / +12.5 [0, +25]) |
| G-swap junction median ratio (>= 0.50) | 0.250 [0.126, 0.423] / 0.241 [0.110, 0.378], side 0.70 / 0.71 | not rerun here | FAIL (route-sensitivity of the plan unchanged-to-lower than RCFT 0.31) |
| public 1479 @10 guard, ties RSG / YS | mean 0.9522 / 0.9539 (pair 0.9530) | RSG 0.9480, YS 0.9450 | mean tie/up: +0.0050 [-0.0018, +0.0115] vs RSG, +0.0080 [-0.0006, +0.0160] vs YS |
| guard zeros | 58 / 56 | RSG 46 / 44, YS 47 / 48 | FAIL on zeros: +12 [+3.4, +21.3] vs RSG (each seed +12), +9.5 [-1.5, +21.1] vs YS; at-fault +6.5 [0, 13.9] |

Reading: `both` rows lift held mean (+0.03) and turn completion (+25 to +29 pp, the speed-ratchet "short" outcomes vanish) and GT distance (1.14 -> 1.78 m: it drives much farther) and the public mean, but it costs zeros on both sets (about +7 to +12), with the extra zeros mostly offroad / collision / corridor (held offroad 17 / 19 vs 17 / 13; collision 7 / 11 vs 4 / 6). It does not fix inside-cutting nor route sensitivity (G-swap). Registered lines: mean PASS, zeros FAIL, guard-zeros FAIL, G-swap FAIL.

Caveats: inside-cut here is the closed-loop version (signed lateral from rf_follow decisions, route = lane centre), not the audit's plan-level rf_loss_audit definition; 24 turns only, CIs wide. G-swap for RSG-L18 itself was not re-read, comparison number is RCFT / RB-F from cft/summary.md.

Files: held_raw.txt, held18_confirm.{md,json}, follow.{md,json}, inside.json, gswap_*.json, guard_confirm.{md,json}.
