# NR1STAND: "do not walk farther than the human when the human stands" (2026-10-19)

Prereg: NR1 prereg supplement 17 + addenda 1-5 ([../../plans/2026-10-17-nr1-prereg.md](../../plans/2026-10-17-nr1-prereg.md)).
Decision 388. One seed, 1 000 steps, all arms against the on-disk control JTX1K-L18-s0, native reads. Box: `$DATA_DIR/runs/defect_fix/stand*`.

Arms (command diffs in [jtx/cmd_diff.txt](jtx/cmd_diff.txt) and [cmd_diff.txt](cmd_diff.txt); dose and gate in [dose.json](dose.json), [gate.json](gate.json), [identity.json](identity.json)):
- **A STANDX1K**: STAND term only (W_S 0.001, margin 1.0 m).
- **B STANDFMX1K**: STAND + FREE-M (`--on-free 0.002 --on-free-mode move --on-free-frac 0.95`).
- **C STANDFRX1K**: STAND + the d365 launch-side FREE term as is (`--on-free 0.004`, mode all).

Per-arm tables with every guard: [jtx/pilot_A.md](jtx/pilot_A.md), [jtx/pilot_B.md](jtx/pilot_B.md), [jtx/pilot_C.md](jtx/pilot_C.md);
arena subset and drift: `jtx/arena_*.json`, `jtx/drift_*.json`; per-window arena scores: `jtx/arena_windows.csv.gz`.

## Guards (control JTX1K -> arm)

| guard (line) | A STANDX1K | B STANDFMX1K | C STANDFRX1K |
|:--|--:|--:|--:|
| held 460 zeros 37 (<= +2) | 39 ok | **45 FAIL** | **52 FAIL** |
| wait other_lead 0.944 (>= -0.03) | **0.889 FAIL** | 0.917 ok | **0.889 FAIL** |
| wait red_queue 0.923 (>= -0.03) | 0.923 ok | 0.923 ok | **0.872 FAIL** |
| launch delay other_lead / red_queue (<= +0.15 s) | ok / ok | ok / ok | ok / ok |
| launched other_lead / red_queue (>= 0.95x) | ok / ok | ok / ok | ok / ok |
| held 460 front collisions 2 (<= +2) | 3 ok | **7 FAIL** | **9 FAIL** |
| arena AF-front windows 1 (<= +2) | 1 ok | 1 ok | **4 FAIL** |
| arena static stratum (331) paired >= 0 | 0.776 -> 0.894 ok | 0.776 -> 0.867 ok | **0.776 -> 0.719 FAIL** |
| arena subset (753) paired >= +0.01 | +0.0505 [+0.000, +0.126] ok | +0.0432 [+0.002, +0.109] ok | **-0.0236 [-0.043, -0.002] FAIL** |
| drift | ok | ok | ok |
| verdict | FAIL (1 guard) | FAIL (2 guards) | FAIL (broad) |

Held 460 mean (control 0.8292): A 0.8167 (paired -0.0125 [-0.022, -0.003]), B 0.8620 (+0.0329 [+0.011, +0.050]), C 0.8710 (+0.0419 [+0.021, +0.077]).
Arena subset zeros (control 88): A 49, B 58, C 113.

## Cue strata (arena subset, zeros control -> arm)

| stratum (n) | A | B | C |
|:--|--:|--:|--:|
| static: red light at the line, no lead (96) | 62 -> 30, +0.333 | 62 -> 38, +0.250 | 62 -> 63, -0.010 |
| static: light of unknown state, no lead (34) | 2 -> 2 | 2 -> 2 | 2 -> **17**, -0.441 |
| static: lead <= 8 m (188) | 2 -> 1 | 2 -> 1 | 2 -> 5 |
| static: lead 8-30 m (12) | 8 -> 2 | 8 -> 3 | 8 -> 8 |
| red-line launch (59), paired | -0.013 [-0.024, -0.001] | -0.004 [-0.015, +0.009] | +0.020 [+0.001, +0.052] |

## k = 9 speed ratio (arena subset, median v / v_human; control 0.845)

| | k2 | k5 | k9 |
|:--|--:|--:|--:|
| A | 0.985 | 0.904 | 0.824 |
| B | 0.992 | 0.973 | **0.948** |
| C | 0.992 | 0.961 | 0.948 |

A does not touch the slow ratchet (k9 diff -0.021 [-0.031, -0.001]); B and C remove it (+0.102 [+0.078, +0.164] and +0.103 [+0.078, +0.149]).
Slow partials (progress < 0.8): control 48, A 50, B 25, C 13; near-1 (0.98-0.999): 12, 14, 6, 3.

## Drift (held 460, left +, arm - control, log-clustered 95 % CI)

| | k5_10 | k11_17 | turn k11_17 |
|:--|--:|--:|--:|
| A | -0.001 [-0.004, +0.002] | -0.004 [-0.017, +0.006] | -0.099 [-0.210, +0.035] |
| B | -0.018 [-0.030, -0.007] | -0.024 [-0.052, +0.004] | -0.155 [-0.298, +0.002] |
| C | -0.019 [-0.031, -0.007] | -0.009 [-0.041, +0.024] | -0.018 [-0.124, +0.105] |

## Frozen-pool PCS (d382 board-mix pool, computed on the Mac on the 716-window subset)

| arm | PCS | arena mean | slow | near-1 |
|:--|--:|--:|--:|--:|
| JTX1K control | 1173 | 0.8648 | 48 | 12 |
| B STANDFMX1K | **1259** | 0.9115 | 25 | 6 |
| C STANDFRX1K | 1191 | 0.8439 | 13 | 3 |
| A STANDX1K | 1204 | 0.9192 | 50 | 14 |

B is best by PCS (+86 against the control), A +31, C +18. Same-pool, subset only; one reading per arm, no CI.

## Reading

- B is the only arm that fixes both the slow ratchet (k9 0.845 -> 0.948; slow partials 48 -> 25) and most of the red-line static zeros (62 -> 38), and it is best by PCS and
  arena subset. It fails two held 460 guards: zeros 37 -> 45 and front collisions 2 -> 7 (cause not isolated).
- A has the largest static gain (red-line zeros 62 -> 30) and clean held guards except wait other_lead (0.944 -> 0.889, below the -0.03 line), but it leaves the ratchet
  (k9 0.824) and held 460 mean is -0.0125.
- C fails broadly: waits, AF-front, static, held zeros. Its unknown-light static zeros go 2 -> 17.
- Not adopted as is. B is the base of the next arm (front-collision fix).
