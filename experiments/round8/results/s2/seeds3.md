# 3-seed recipe means, R8K vs R6DAG (PAI 281, ACC + GAP; 278 common scenes; amendment 1, descriptive, no adoption rule)

| | R6DAG-s0 | R6DAG-s1 | R6DAG-s2 | R8K-s0 | R8K-s1 | R8K-s2 |
|:--|--:|--:|--:|--:|--:|--:|
| OFF | 0.778 | 0.818 | 0.794 | 0.795 | 0.804 | 0.808 |
| ON | 0.795 | 0.783 | 0.783 | 0.773 | 0.771 | 0.797 |
| (OFF + ON) / 2 | 0.787 | 0.801 | 0.788 | 0.784 | 0.787 | 0.803 |

| recipe mean (3 seeds) | R6DAG | R8K | R8K - R6DAG (paired scene bootstrap of the seed means) |
|:--|--:|--:|--:|
| OFF | 0.797 | 0.802 | +0.006 [-0.013, +0.025] |
| ON | 0.787 | 0.780 | -0.007 [-0.024, +0.010] |
| (OFF + ON) / 2 | 0.792 | 0.791 | -0.001 [-0.014, +0.013] |
| seed sd of the read mean, OFF / ON | 0.020 / 0.007 | 0.006 / 0.015 | |

| same-seed pair | OFF R8K - R6DAG | ON R8K - R6DAG |
|:--|--:|--:|
| s0 | +0.017 [-0.011, +0.047] | -0.022 [-0.049, +0.003] |
| s1 | -0.014 [-0.041, +0.012] | -0.012 [-0.041, +0.016] |
| s2 | +0.014 [-0.011, +0.040] | +0.014 [-0.007, +0.036] |
