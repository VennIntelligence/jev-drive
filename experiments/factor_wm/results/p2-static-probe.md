# factor_wm p2op static-history probe (decision 146 point 4)

Written 2026-10-06. No training. Script `scripts/fw_p2_static.py` (g1-diag's H1 method, [g1-diag.md](g1-diag.md), run on the P2 family); raw per-clip
numbers on the box in `runs/factor_wm/p2op/static_probe.json`. Arms: P2 = P2-F-s0 (op_parity full run), C = PC-s0 (off-policy control), X = PX-s0
(P2 + on-policy), all seed 0.

**Verdict: not supported as stated (that on-policy training makes X refuse to launch specifically when history is static).** X does refuse
more than P2 and more than C when only the images are static, but the larger part of that gap is C vs P2 (the extra WOD data), and once the ego
history is static too (HUGSIM's actual input) X equals C. On standing-static clips X is as quiet as C, so the "P2 creeps, X fixes it" link is
the off-policy WOD data, not the on-policy labels.

## Method

Readout at t0 of WOD clips, one forward pass per clip: g0b `launch` (40 clips, log launches) and g1s `stay` (40 clips, the log stands still for 8 s).
History variants: `real` = the 10 logged frames and logged ego history; `static_img` = the t0 frame repeated 10 times, ego inputs unchanged
(exactly g1-diag); `static_full` = static images plus the ego history of a standing car (history poses 0, v = 0, a = 0, HUGSIM's 5 s warm-up).
Readouts: action[1] (a shipped-distilled head in the P2 family, so it barely moves); plan speed at 1 s and 3 s; shouldStop = v0 < 0.3 and
action[1] < 0.1 (g1-diag's definition); shouldStop (plan) = v0 < 0.3 and the plan's 0.5 s acceleration < 0.1, the quantity the plan engine and
HUGSIM execute. Cells are means over clips. CIs are 95% bootstrap over clips, clustered by logged sequence, paired.

## Cells

| clip type | history | arm | shouldStop (action[1]) | shouldStop (plan) | action[1] | plan v 1 s | plan v 3 s |
|:--|:--|:--|--:|--:|--:|--:|--:|
| launch | real | P2 | 0.55 | 0.03 | +0.184 | +1.19 | +2.90 |
| launch | real | C | 0.60 | 0.12 | +0.155 | +1.10 | +2.81 |
| launch | real | X | 0.60 | 0.17 | +0.096 | +0.99 | +2.72 |
| launch | static_img | P2 | 0.90 | 0.35 | -0.020 | +0.46 | +1.39 |
| launch | static_img | C | 0.90 | 0.75 | -0.044 | +0.25 | +0.89 |
| launch | static_img | X | 0.90 | 0.97 | -0.062 | +0.06 | +0.50 |
| launch | static_full | P2 | 0.90 | 0.90 | +0.006 | +0.06 | +0.95 |
| launch | static_full | C | 0.93 | 1.00 | -0.048 | -0.04 | +0.40 |
| launch | static_full | X | 0.93 | 1.00 | -0.041 | -0.01 | +0.39 |
| standing-static | real | P2 | 1.00 | 0.75 | -0.036 | +0.08 | +0.44 |
| standing-static | real | C | 1.00 | 1.00 | -0.081 | -0.03 | +0.14 |
| standing-static | real | X | 1.00 | 1.00 | -0.073 | -0.01 | +0.11 |
| standing-static | static_img | P2 | 1.00 | 0.80 | -0.036 | +0.06 | +0.38 |
| standing-static | static_img | C | 1.00 | 1.00 | -0.075 | -0.02 | +0.11 |
| standing-static | static_img | X | 1.00 | 1.00 | -0.063 | -0.01 | +0.08 |
| standing-static | static_full | P2 | 1.00 | 0.80 | -0.036 | +0.06 | +0.38 |
| standing-static | static_full | C | 1.00 | 1.00 | -0.075 | -0.02 | +0.11 |
| standing-static | static_full | X | 1.00 | 1.00 | -0.063 | -0.01 | +0.08 |

## Paired contrasts (X minus reference)

| clip type | history | contrast | action[1] | plan v 1 s | plan v 3 s | shouldStop (plan) |
|:--|:--|:--|:--|:--|:--|:--|
| launch | real | X - P2 | -0.088 [-0.110, -0.067] | -0.196 [-0.251, -0.142] | -0.183 [-0.352, -0.022] | +0.150 [+0.050, +0.275] |
| launch | real | X - C | -0.060 [-0.085, -0.036] | -0.102 [-0.142, -0.066] | -0.088 [-0.166, -0.016] | +0.050 [+0.000, +0.125] |
| launch | static_img | X - P2 | -0.042 [-0.056, -0.029] | -0.401 [-0.468, -0.337] | -0.893 [-1.030, -0.760] | +0.625 [+0.475, +0.775] |
| launch | static_img | X - C | -0.018 [-0.030, -0.007] | -0.186 [-0.248, -0.131] | -0.392 [-0.499, -0.289] | +0.225 [+0.100, +0.350] |
| launch | static_full | X - P2 | -0.046 [-0.059, -0.035] | -0.074 [-0.090, -0.058] | -0.557 [-0.649, -0.467] | +0.100 [+0.025, +0.200] |
| launch | static_full | X - C | +0.007 [+0.002, +0.013] | +0.024 [+0.019, +0.030] | -0.002 [-0.014, +0.012] | +0.000 [+0.000, +0.000] |
| standing-static | real | X - P2 | -0.036 [-0.045, -0.028] | -0.091 [-0.113, -0.071] | -0.324 [-0.441, -0.225] | +0.250 [+0.114, +0.410] |
| standing-static | real | X - C | +0.008 [+0.001, +0.015] | +0.019 [+0.010, +0.029] | -0.024 [-0.057, +0.008] | +0.000 [+0.000, +0.000] |
| standing-static | static_img | X - P2 | -0.027 [-0.033, -0.020] | -0.069 [-0.084, -0.054] | -0.292 [-0.397, -0.203] | +0.200 [+0.075, +0.350] |
| standing-static | static_img | X - C | +0.013 [+0.007, +0.019] | +0.015 [+0.008, +0.021] | -0.029 [-0.050, -0.015] | +0.000 [+0.000, +0.000] |
| standing-static | static_full | X - P2 | -0.027 [-0.033, -0.020] | -0.069 [-0.085, -0.054] | -0.293 [-0.396, -0.204] | +0.200 [+0.075, +0.350] |
| standing-static | static_full | X - C | +0.013 [+0.007, +0.019] | +0.015 [+0.007, +0.021] | -0.029 [-0.051, -0.015] | +0.000 [+0.000, +0.000] |

## Reading

- **Launch, real history:** all three arms launch (plan 3 s speed 2.7-2.9 m/s). X is a little lower than P2 (-0.20 m/s at 1 s) and C (-0.10).
- **Launch, static images only:** every arm drops, X most: plan 1 s speed P2 0.46, C 0.25, X 0.06; 3 s 1.39 / 0.89 / 0.50; plan-stop 0.35 / 0.75 / 0.98.
  The static-minus-real drop at 1 s is X 0.93, P2 0.73, C 0.85. X - C is -0.19 at 1 s and -0.39 at 3 s (CIs exclude 0), versus -0.10 / -0.09 with real
  history. So there is a static-specific extra over C, small in absolute terms (about 0.2 m/s at 1 s).
- **Launch, fully static (HUGSIM-like):** all arms stop at 1 s (P2 0.06, C -0.04, X -0.01). X - C is +0.02 / 0.00 (no difference). P2 keeps 0.95 m/s
  at 3 s against 0.39 for C and X; that P2-C gap is the off-policy data, not on-policy training.
- **Standing-static control:** P2 creeps mildly (3 s plan speed 0.44, plan-stop 0.75-0.80). C and X are at 0.14 / 0.12 and both plan-stop 1.00.
  X - P2 is -0.32 [-0.44, -0.23] at 3 s, X - C -0.02 [-0.06, +0.01]. History variants barely matter here (the logged history is already static).
- action[1] shouldStop is uninformative for the P2 family (0.9-1.0 whenever the history is static, including P2); use the plan-based column.

## Caveats

- 40 + 40 clips, one seed, t0 only (no closed-loop rollout). HUGSIM starts at 1 m/s after the warm-up; the probe sets v = 0 in `static_full`, which
  is the stricter input. The probe does not reproduce P2 launching in HUGSIM (P2 1 stall of 12): at the t0 read P2 also stops at 1 s under `static_full`,
  so what lets P2 start in HUGSIM is not this input read.
- A factor that the probe leaves open: the C arm has the same WOD stay clips and the same label fix, so the "stay" pressure that suppresses X's
  creep is shared with C. What X adds over C is only the on-policy distribution; its HUGSIM launch stalls (9 of 12 vs C 3) are therefore not
  explained by the t0 readout alone. The remaining candidate is the closed-loop dynamics after the first steps (X's lower 1 s and 3 s plan speed in
  `static_img`), which this probe cannot measure.
