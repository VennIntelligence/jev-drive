# S-DROP summary (sdrop_report.py; lines of plans/2026-10-10-sdrop-prereg.md section 5)

Question 1: G2 = +3.33 (stage 2, P2H10S - P2H10, seeds 0 + 1); half = 1.67

| arm | dropped | stage 2: arm - P2H10S | stage 2: arm - base | share of G2 kept | verdict |
|:--|:--|:--|:--|--:|:--|
| noA | A (agent hinge) | -1.164 [-2.18, -0.2401] | +2.168 [+0.2522, +4.014] | 0.65 | not needed |
| noB | B (hinge-only rows; C goes with them) | -3.534 [-5.477, -1.641] | -0.2022 [-1.094, +0.6645] | -0.06 | carries |
| noC | C (road hinge on hinge-only rows) | -1.534 [-2.984, -0.0469] | +1.798 [+0.1152, +3.455] | 0.54 | not needed |

Question 2: W2 on navtest tokens over 45 deg, seeds 0 / 1: base [45, 46], P2H10S [58, 65]; tolerance +5. "Only the road hinge on hinge-only rows": **fails**

| arm | W2 | excess over base | verdict |
|:--|:--|:--|:--|
| noA | [54, 63] | [9, 17] | wide |
| noB | [45, 47] | [0, 1] | not wide |
| noC | [55, 54] | [10, 8] | wide |

Attribution by the common rule (E = P2H10S - base on the same two seeds):

| effect | E [95 % CI] | noA | noB | noC |
|:--|:--|:--|:--|:--|
| navhard | 225 groups | combined | +2.219 [+0.2529, +4.364] | carries (0.42) | carries (-0.17) | unresolved (0.45) |
| navhard | 225 groups | stage1 | -0.5701 [-2.207, +1.317] (includes 0) | unresolved (2.14) | unresolved (-0.01) | not needed (1.91) |
| navhard | 225 groups | stage2 | +3.332 [+1.275, +5.482] | not needed (0.65) | carries (-0.06) | not needed (0.54) |
| navtest | all | EPDMS | +0.3768 [+0.2039, +0.566] | not needed (1.03) | carries (0.11) | carries (0.18) |
| navtest | > 45 deg | EPDMS | +0.8072 [+0.102, +1.534] | not needed (1.08) | unresolved (0.38) | carries (0.04) |
| navtest replay | all | DAC fail % | -0.3458 [-0.5251, -0.1934] | not needed (1.11) | carries (0.01) | carries (0.19) |
| navtest replay | T45 (> 45 deg) | DAC fail % | -0.857 [-1.658, -0.07375] | not needed (1.15) | unresolved (0.35) | carries (0.04) |
| navtest replay | T45 (> 45 deg) | inside-cut % | -0.7251 [-1.394, -0.1866] | not needed (1.09) | carries (0.14) | unresolved (0.36) |
| navtest replay | T45 (> 45 deg) | cannot-make-turn % | +0.1648 [-0.4082, +0.7837] (includes 0) | unresolved (1.20) | unresolved (-0.60) | unresolved (2.20) |
| navtest own plan | pooled | agent rate | -0.0002058 [-0.001377, +0.0009528] (includes 0) | unresolved (-1.00) | unresolved (-1.20) | unresolved (0.40) |
| navtest own plan | pooled | boundary rate | -0.006504 [-0.01025, -0.003318] | not needed (0.86) | carries (0.15) | carries (0.34) |
| navtest own plan | > 45 deg | signed lateral at 4 s (m, + = outside) | +0.2235 [+0.1859, +0.2672] | not needed (0.75) | carries (0.14) | not needed (0.55) |
| navtest own plan | > 45 deg | W2 share | +0.01055 [+0.004285, +0.01685] | not needed (0.81) | carries (0.03) | not needed (0.56) |
| hold own plan | pooled | agent rate | -0.01044 [-0.01238, -0.008539] | not needed (0.63) | carries (0.02) | not needed (0.67) |
| hold own plan | on-log | agent rate | -0.0006407 [-0.001661, +0.0002794] (includes 0) | unresolved (0.86) | unresolved (0.14) | unresolved (0.21) |
| hold own plan | off-track | agent rate | -0.01603 [-0.01902, -0.01298] | not needed (0.63) | carries (0.01) | not needed (0.68) |
| hold own plan | pooled | boundary rate | -0.01667 [-0.01901, -0.01453] | not needed (0.94) | carries (0.01) | carries (0.39) |
| hold own plan | on-log | boundary rate | -0.001785 [-0.002916, -0.0009108] | not needed (0.92) | carries (-0.03) | carries (0.31) |
| hold own plan | off-track | boundary rate | -0.02516 [-0.0286, -0.0221] | not needed (0.95) | carries (0.01) | carries (0.40) |
| hold on-log own plan | > 45 deg | signed lateral at 4 s (m, + = outside) | +0.06699 [+0.03099, +0.1013] | not needed (0.70) | carries (0.15) | not needed (0.83) |
| hold on-log own plan | > 45 deg | W2 share | +0.009378 [+0.003698, +0.0158] | not needed (0.77) | carries (0.18) | not needed (1.00) |
| hold four families own plan | > 45 deg | signed lateral at 4 s (m, + = outside) | +0.1932 [+0.1465, +0.2387] | not needed (0.87) | carries (0.08) | carries (0.44) |
| hold four families own plan | > 45 deg | W2 share | -0.006702 [-0.01625, +0.003354] (includes 0) | unresolved (1.13) | carries (-0.49) | carries (-0.62) |

Share kept = (arm - base) / E. Full tables: contrasts.csv, arc.csv, widening.csv.

Arc guard missed: none

Cross-check against bd4_g3.py per-seed pooled rates: at most 0.0 states apart over 24 numbers.
