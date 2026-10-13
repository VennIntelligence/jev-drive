# STOPLINE read (plans/2026-10-13-stopline-prereg.md), 2026-10-13

Arm `RSGB-L18-s{0,1}` = RSG-L18's command + `--on-lon on_lon` (items B, P) on `on1_L18RB0`. Files: `confirm.md` / `confirm.json` (public 1491
at 10 decisions, `rf_stopline.sh confirm`), `lonread-on1_L18RSGallheld.json` (held offline read), `rely-RSGB-L18-s*.json`, `lon_pool-*.json`
(pack gates).

## Training gates (gate 2 fails, both seeds)

| checkpoint | dev ADE @ 2500 (line <= RSG + 0.02 = 0.548) | G-rely (d_junction >= 0.01 and CI of junction - straight > 0) |
|:--|--:|:--|
| RSG-L18-s0 / s1 | 0.528 / 0.527 | read (+0.025 / +0.028) |
| RSGB-L18-s0 | **0.565** | **not read**: d_junction -0.009, diff -0.003 [-0.025, +0.021] |
| RSGB-L18-s1 | **0.558** | **not read**: d_junction +0.003, diff +0.003 [-0.019, +0.027] |

Dev ADE jumps to 0.59-0.60 at step 500 and recovers only to 0.56; the route memory stops being relied on. By the prereg a seed failing gate 2
is reported, not read: everything below is descriptive.

## Public 1491 / 10 decisions (1479 scenes, 136 logs; descriptive)

| checkpoint | mean | zeros | coll / off / corr | at-fault | GT (m) |
|:--|--:|--:|:--|--:|--:|
| RSG-L18 (seed mean) | 0.9480 | 45 | 8 / 30.5 / 7.5 | 38.5 | 0.796 |
| RSGB-L18-s0 / s1 | 0.9587 / 0.9557 | 50 / 51 | 7/31/12 · 10/28/13 | 38 / 38 | 1.228 / 1.134 |
| RSGB-L18 (seed mean) | 0.9572 | 50.5 | 8.5 / 29.5 / 12.5 | 38 | 1.181 |
| YR10m10-F-s0 (base) | 0.9323 | 68 | 7 / 43 / 20 | 50 | 0.941 |

| a - b | mean | zeros | at-fault | GT (m) |
|:--|:--|:--|:--|:--|
| RSGB - RSG | +0.0091 [+0.0035, +0.0151] | +5.5 [-2.4, +13.3] | -0.5 [-7.9, +6.7] | **+0.386 [+0.319, +0.456]** |
| RSGB-s0 - RSG | +0.0106 [+0.0043, +0.0171] | +5.0 [-3.0, +13.6] | -0.5 | +0.432 [+0.361, +0.507] |
| RSGB-s1 - RSG | +0.0076 [+0.0007, +0.0147] | +6.0 [-4.1, +16.0] | -0.5 | +0.339 [+0.276, +0.406] |
| RSGB - base | +0.0249 [+0.0165, +0.0342] | -17.5 [-30.8, -5.0] | -12.0 [-24.0, -0.5] | +0.241 [+0.169, +0.312] |

Lines as registered: P1 (mean >= -0.010) passes, P2 (zeros point <= +3) fails (+5.5; CI includes 0), P3 (base guard) passes. The mean
goes up because the car goes further (progress); corridor zeros 7.5 -> 12.5 and GT distance +0.39 m (CI far from 0; RSG's GT advantage
over the base, -0.145 m, turns into +0.241 m).

## Held offline read (`on1_L18RSGallheld`, 7 837 rows, 447 scenes; 744 rows with red in the 4 s window)

| plans of | B active rows (scenes) | B sum on red rows (m) | P active share | P mean over all rows (m) |
|:--|:--|--:|--:|--:|
| RB0-F-s0 | 10 (3) | 0.132 | 10.3 % | 0.230 |
| RSG-L18-s0 / s1 | 10 (3) / 11 (3) | 0.124 / 0.134 | 10.3 % | 0.229 |
| RSGB-L18-s0 / s1 | 12 (4) / 13 (4) | 0.278 / 0.411 | 4.3 % / 4.6 % | 0.040 / 0.044 |

P did what P does (the progress shortfall on open rows drops by 80 %); B did not: the red-line overrun at held states is 2-3 x larger, not
smaller, though tiny in absolute terms (B is near its floor on nuPlan). Read: by mass the arm is the progress guard (Amendment 1: P ~0.6 of
the total loss, B ~0.02), and the progress push is what moved the closed loop (more progress, more corridor zeros, larger GT distance).
B on nuPlan has 84 active rows in 13 scenes of the training pool: too thin to teach stopping.
