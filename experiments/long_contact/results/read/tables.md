### P1 own + arc: AUC, two-seed score [95% log-cluster CI]

| label | positives (logs) | blind | vis | nat | priv | blind_nat | best non-priv - blind | nat - vis | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A | 1905 (65) | 0.953 [0.940, 0.967] | 0.984 [0.977, 0.990] | 0.987 [0.982, 0.991] | 1.000 [0.999, 1.000] | 0.983 [0.978, 0.989] | nat +0.033 [+0.023, +0.044] | +0.002 [+0.000, +0.006] | PARTIAL |
| B | 397 (24) | 0.912 [0.874, 0.948] | 0.969 [0.949, 0.984] | 0.952 [0.924, 0.977] | 0.999 [0.997, 1.000] | 0.953 [0.925, 0.980] | vis +0.056 [+0.021, +0.093] | -0.017 [-0.041, +0.008] | READABLE |
| C | 1909 (73) | 0.984 [0.977, 0.990] | 0.991 [0.985, 0.995] | 0.991 [0.982, 0.995] | 1.000 [1.000, 1.000] | 0.987 [0.974, 0.994] | vis +0.007 [+0.002, +0.014] | -0.000 [-0.003, +0.001] | NOT READABLE |
| agent contact (S0 label) | 4854 (98) | 0.833 [0.812, 0.853] | 0.919 [0.903, 0.932] | 0.922 [0.905, 0.935] | 0.993 [0.991, 0.995] | 0.909 [0.889, 0.925] | - | +0.003 [-0.003, +0.010] | - |
| boundary (S0 label) | 4468 (94) | 0.820 [0.796, 0.840] | 0.978 [0.970, 0.984] | 0.974 [0.965, 0.982] | 0.985 [0.978, 0.990] | 0.962 [0.951, 0.972] | - | -0.003 [-0.007, +0.000] | - |

Per seed (seed 0 / seed 1):

| label | blind | vis | nat | priv | blind_nat |
|---|---|---|---|---|---|
| A | 0.948 / 0.948 | 0.983 / 0.981 | 0.985 / 0.984 | 1.000 / 1.000 | 0.981 / 0.981 |
| B | 0.903 / 0.907 | 0.953 / 0.970 | 0.952 / 0.942 | 0.998 / 0.997 | 0.947 / 0.951 |
| C | 0.983 / 0.982 | 0.991 / 0.989 | 0.990 / 0.989 | 1.000 / 1.000 | 0.985 / 0.986 |
| a | 0.827 / 0.828 | 0.913 / 0.908 | 0.914 / 0.914 | 0.992 / 0.991 | 0.901 / 0.906 |
| b | 0.816 / 0.809 | 0.972 / 0.973 | 0.968 / 0.969 | 0.983 / 0.982 | 0.959 / 0.952 |

### P0 own: AUC, two-seed score [95% log-cluster CI]

| label | positives (logs) | blind | vis | nat | priv | blind_nat | best non-priv - blind | nat - vis | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A | 85 (7) | 0.888 [0.830, 0.933] | 0.990 [0.979, 0.993] | 0.989 [0.970, 0.995] | 1.000 [1.000, 1.000] | 0.986 [0.968, 0.992] | vis +0.102 [+0.055, +0.158] | -0.001 [-0.010, +0.004] | READABLE |
| B | 36 (6) | 0.784 [0.534, 0.946] | 0.936 [0.853, 0.982] | 0.868 [0.711, 0.965] | 0.998 [0.995, 1.000] | 0.847 [0.699, 0.990] | vis +0.153 [-0.004, +0.384] | -0.068 [-0.212, +0.053] | PARTIAL |
| C | 111 (9) | 0.949 [0.916, 0.984] | 0.965 [0.905, 0.997] | 0.961 [0.888, 0.997] | 1.000 [0.999, 1.000] | 0.934 [0.807, 0.997] | vis +0.016 [-0.039, +0.064] | -0.004 [-0.019, +0.008] | NOT READABLE |
| agent contact (S0 label) | 911 (64) | 0.779 [0.740, 0.810] | 0.900 [0.864, 0.927] | 0.902 [0.864, 0.930] | 0.995 [0.992, 0.996] | 0.892 [0.857, 0.919] | - | +0.002 [-0.012, +0.017] | - |
| boundary (S0 label) | 1090 (82) | 0.785 [0.747, 0.817] | 0.977 [0.968, 0.984] | 0.972 [0.961, 0.981] | 0.985 [0.975, 0.991] | 0.960 [0.946, 0.970] | - | -0.004 [-0.010, +0.001] | - |

Per seed (seed 0 / seed 1):

| label | blind | vis | nat | priv | blind_nat |
|---|---|---|---|---|---|
| A | 0.854 / 0.890 | 0.985 / 0.986 | 0.984 / 0.983 | 1.000 / 1.000 | 0.984 / 0.979 |
| B | 0.789 / 0.764 | 0.899 / 0.939 | 0.893 / 0.810 | 0.998 / 0.989 | 0.836 / 0.838 |
| C | 0.946 / 0.941 | 0.966 / 0.951 | 0.962 / 0.949 | 1.000 / 1.000 | 0.927 / 0.933 |
| a | 0.763 / 0.777 | 0.892 / 0.885 | 0.885 / 0.894 | 0.994 / 0.993 | 0.880 / 0.887 |
| b | 0.776 / 0.777 | 0.972 / 0.971 | 0.966 / 0.967 | 0.983 / 0.981 | 0.954 / 0.950 |

### Gains (paired, same log-cluster draws)

| population | label | rung | against | AUC difference [95% CI] |
|---|---|---|---|---|
| P1 own + arc | A | vis | blind | +0.031 [+0.021, +0.041] |
| P1 own + arc | A | nat | blind | +0.033 [+0.023, +0.044] |
| P1 own + arc | A | priv | blind | +0.047 [+0.032, +0.060] |
| P1 own + arc | A | blind_nat | blind | +0.030 [+0.020, +0.041] |
| P1 own + arc | A | nat | vis | +0.002 [+0.000, +0.006] |
| P1 own + arc | A | priv | nat | +0.013 [+0.008, +0.018] |
| P1 own + arc | A | blind_nat | blind | +0.030 [+0.020, +0.041] |
| P1 own + arc | A | nat | blind_nat | +0.003 [+0.002, +0.005] |
| P1 own + arc | B | vis | blind | +0.056 [+0.021, +0.093] |
| P1 own + arc | B | nat | blind | +0.040 [+0.004, +0.067] |
| P1 own + arc | B | priv | blind | +0.086 [+0.050, +0.125] |
| P1 own + arc | B | blind_nat | blind | +0.041 [+0.001, +0.074] |
| P1 own + arc | B | nat | vis | -0.017 [-0.041, +0.008] |
| P1 own + arc | B | priv | nat | +0.047 [+0.022, +0.075] |
| P1 own + arc | B | blind_nat | blind | +0.041 [+0.001, +0.074] |
| P1 own + arc | B | nat | blind_nat | -0.001 [-0.024, +0.020] |
| P1 own + arc | C | vis | blind | +0.007 [+0.002, +0.014] |
| P1 own + arc | C | nat | blind | +0.007 [-0.001, +0.014] |
| P1 own + arc | C | priv | blind | +0.016 [+0.010, +0.023] |
| P1 own + arc | C | blind_nat | blind | +0.003 [-0.008, +0.012] |
| P1 own + arc | C | nat | vis | -0.000 [-0.003, +0.001] |
| P1 own + arc | C | priv | nat | +0.009 [+0.004, +0.017] |
| P1 own + arc | C | blind_nat | blind | +0.003 [-0.008, +0.012] |
| P1 own + arc | C | nat | blind_nat | +0.004 [+0.002, +0.008] |
| P1 own + arc | a | vis | blind | +0.086 [+0.070, +0.101] |
| P1 own + arc | a | nat | blind | +0.089 [+0.072, +0.106] |
| P1 own + arc | a | priv | blind | +0.160 [+0.141, +0.180] |
| P1 own + arc | a | blind_nat | blind | +0.076 [+0.062, +0.090] |
| P1 own + arc | a | nat | vis | +0.003 [-0.003, +0.010] |
| P1 own + arc | a | priv | nat | +0.071 [+0.059, +0.087] |
| P1 own + arc | a | blind_nat | blind | +0.076 [+0.062, +0.090] |
| P1 own + arc | a | nat | blind_nat | +0.013 [+0.005, +0.022] |
| P1 own + arc | b | vis | blind | +0.157 [+0.141, +0.176] |
| P1 own + arc | b | nat | blind | +0.154 [+0.138, +0.172] |
| P1 own + arc | b | priv | blind | +0.165 [+0.148, +0.185] |
| P1 own + arc | b | blind_nat | blind | +0.142 [+0.127, +0.159] |
| P1 own + arc | b | nat | vis | -0.003 [-0.007, +0.000] |
| P1 own + arc | b | priv | nat | +0.011 [+0.007, +0.015] |
| P1 own + arc | b | blind_nat | blind | +0.142 [+0.127, +0.159] |
| P1 own + arc | b | nat | blind_nat | +0.012 [+0.009, +0.016] |
| P0 own | A | vis | blind | +0.102 [+0.055, +0.158] |
| P0 own | A | nat | blind | +0.100 [+0.051, +0.161] |
| P0 own | A | priv | blind | +0.112 [+0.067, +0.170] |
| P0 own | A | blind_nat | blind | +0.097 [+0.053, +0.156] |
| P0 own | A | nat | vis | -0.001 [-0.010, +0.004] |
| P0 own | A | priv | nat | +0.011 [+0.005, +0.030] |
| P0 own | A | blind_nat | blind | +0.097 [+0.053, +0.156] |
| P0 own | A | nat | blind_nat | +0.003 [-0.007, +0.014] |
| P0 own | B | vis | blind | +0.153 [-0.004, +0.384] |
| P0 own | B | nat | blind | +0.085 [-0.026, +0.232] |
| P0 own | B | priv | blind | +0.214 [+0.051, +0.465] |
| P0 own | B | blind_nat | blind | +0.063 [-0.125, +0.279] |
| P0 own | B | nat | vis | -0.068 [-0.212, +0.053] |
| P0 own | B | priv | nat | +0.129 [+0.034, +0.287] |
| P0 own | B | blind_nat | blind | +0.063 [-0.125, +0.279] |
| P0 own | B | nat | blind_nat | +0.021 [-0.054, +0.103] |
| P0 own | C | vis | blind | +0.016 [-0.039, +0.064] |
| P0 own | C | nat | blind | +0.012 [-0.054, +0.064] |
| P0 own | C | priv | blind | +0.051 [+0.016, +0.084] |
| P0 own | C | blind_nat | blind | -0.015 [-0.134, +0.064] |
| P0 own | C | nat | vis | -0.004 [-0.019, +0.008] |
| P0 own | C | priv | nat | +0.039 [+0.003, +0.111] |
| P0 own | C | blind_nat | blind | -0.015 [-0.134, +0.064] |
| P0 own | C | nat | blind_nat | +0.027 [-0.003, +0.082] |
| P0 own | a | vis | blind | +0.121 [+0.091, +0.152] |
| P0 own | a | nat | blind | +0.123 [+0.088, +0.157] |
| P0 own | a | priv | blind | +0.216 [+0.185, +0.254] |
| P0 own | a | blind_nat | blind | +0.114 [+0.084, +0.144] |
| P0 own | a | nat | vis | +0.002 [-0.012, +0.017] |
| P0 own | a | priv | nat | +0.093 [+0.065, +0.130] |
| P0 own | a | blind_nat | blind | +0.114 [+0.084, +0.144] |
| P0 own | a | nat | blind_nat | +0.010 [-0.003, +0.022] |
| P0 own | b | vis | blind | +0.191 [+0.164, +0.223] |
| P0 own | b | nat | blind | +0.187 [+0.161, +0.218] |
| P0 own | b | priv | blind | +0.199 [+0.172, +0.231] |
| P0 own | b | blind_nat | blind | +0.174 [+0.149, +0.204] |
| P0 own | b | nat | vis | -0.004 [-0.010, +0.001] |
| P0 own | b | priv | nat | +0.012 [+0.007, +0.018] |
| P0 own | b | blind_nat | blind | +0.174 [+0.149, +0.204] |
| P0 own | b | nat | blind_nat | +0.013 [+0.008, +0.020] |

### Positive control and lateral split

| population | label | subset | score | AUC [95% CI] | n | positives (logs) |
|---|---|---|---|---|--:|---|
| P1 own + arc | a | all | S0 (mean of seeds) | 0.919 [0.901, 0.934] | 153948 | 4854 (98) |
| P1 own + arc | a | all | S0 step seed 0 | 0.911 [0.893, 0.925] | 153948 | 4854 (98) |
| P1 own + arc | a | all | S0 step seed 1 | 0.907 [0.887, 0.924] | 153948 | 4854 (98) |
| P1 own + arc | a | all | S0 blind seed 0 | 0.824 [0.802, 0.845] | 153948 | 4854 (98) |
| P1 own + arc | b | all | S0 (mean of seeds) | 0.978 [0.970, 0.984] | 148037 | 4468 (94) |
| P1 own + arc | b | all | S0 step seed 0 | 0.970 [0.960, 0.978] | 148037 | 4468 (94) |
| P1 own + arc | b | all | S0 step seed 1 | 0.974 [0.965, 0.982] | 148037 | 4468 (94) |
| P1 own + arc | b | all | S0 blind seed 0 | 0.814 [0.787, 0.836] | 148037 | 4468 (94) |
| P1 own + arc | agent split | A-type: vehicle ahead in own lane (A positive) | S0 (mean of seeds) | 0.961 [0.944, 0.974] | 150999 | 1905 (65) |
| P1 own + arc | agent split | A-type: vehicle ahead in own lane (A positive) | S0 blind seed 0 | 0.912 [0.883, 0.938] | 150999 | 1905 (65) |
| P1 own + arc | agent split | A-type: vehicle ahead in own lane (A positive) | ladder vis (lat head) | 0.962 [0.948, 0.975] | 150999 | 1905 (65) |
| P1 own + arc | agent split | A-type: vehicle ahead in own lane (A positive) | ladder nat (lat head) | 0.968 [0.958, 0.979] | 150999 | 1905 (65) |
| P1 own + arc | agent split | A-type: vehicle ahead in own lane (A positive) | ladder priv (lat head) | 0.998 [0.996, 0.999] | 150999 | 1905 (65) |
| P1 own + arc | agent split | A-type: vehicle ahead in own lane (A positive) | ladder vis (A head) | 0.984 [0.977, 0.990] | 150999 | 1905 (65) |
| P1 own + arc | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | S0 (mean of seeds) | 0.901 [0.873, 0.923] | 151438 | 2344 (82) |
| P1 own + arc | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | S0 blind seed 0 | 0.771 [0.735, 0.800] | 151438 | 2344 (82) |
| P1 own + arc | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | ladder vis (lat head) | 0.896 [0.870, 0.918] | 151438 | 2344 (82) |
| P1 own + arc | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | ladder nat (lat head) | 0.898 [0.871, 0.920] | 151438 | 2344 (82) |
| P1 own + arc | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | ladder priv (lat head) | 0.991 [0.989, 0.994] | 151438 | 2344 (82) |
| P1 own + arc | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | ladder vis (A head) | 0.814 [0.788, 0.839] | 151438 | 2344 (82) |
| P1 own + arc | agent split | other object (pedestrian, cyclist, generic object) | S0 (mean of seeds) | 0.858 [0.813, 0.904] | 149699 | 605 (39) |
| P1 own + arc | agent split | other object (pedestrian, cyclist, generic object) | S0 blind seed 0 | 0.757 [0.714, 0.801] | 149699 | 605 (39) |
| P1 own + arc | agent split | other object (pedestrian, cyclist, generic object) | ladder vis (lat head) | 0.870 [0.832, 0.906] | 149699 | 605 (39) |
| P1 own + arc | agent split | other object (pedestrian, cyclist, generic object) | ladder nat (lat head) | 0.868 [0.823, 0.907] | 149699 | 605 (39) |
| P1 own + arc | agent split | other object (pedestrian, cyclist, generic object) | ladder priv (lat head) | 0.984 [0.972, 0.992] | 149699 | 605 (39) |
| P1 own + arc | agent split | other object (pedestrian, cyclist, generic object) | ladder vis (A head) | 0.807 [0.759, 0.849] | 149699 | 605 (39) |
| P0 own | a | all | S0 (mean of seeds) | 0.905 [0.868, 0.934] | 51316 | 911 (64) |
| P0 own | a | all | S0 step seed 0 | 0.898 [0.859, 0.927] | 51316 | 911 (64) |
| P0 own | a | all | S0 step seed 1 | 0.892 [0.855, 0.921] | 51316 | 911 (64) |
| P0 own | a | all | S0 blind seed 0 | 0.770 [0.730, 0.804] | 51316 | 911 (64) |
| P0 own | b | all | S0 (mean of seeds) | 0.978 [0.969, 0.985] | 49417 | 1090 (82) |
| P0 own | b | all | S0 step seed 0 | 0.971 [0.959, 0.980] | 49417 | 1090 (82) |
| P0 own | b | all | S0 step seed 1 | 0.975 [0.964, 0.983] | 49417 | 1090 (82) |
| P0 own | b | all | S0 blind seed 0 | 0.779 [0.742, 0.810] | 49417 | 1090 (82) |
| P0 own | agent split | A-type: vehicle ahead in own lane (A positive) | S0 (mean of seeds) | 0.946 [0.908, 0.973] | 50490 | 85 (7) |
| P0 own | agent split | A-type: vehicle ahead in own lane (A positive) | S0 blind seed 0 | 0.757 [0.637, 0.859] | 50490 | 85 (7) |
| P0 own | agent split | A-type: vehicle ahead in own lane (A positive) | ladder vis (lat head) | 0.933 [0.877, 0.959] | 50490 | 85 (7) |
| P0 own | agent split | A-type: vehicle ahead in own lane (A positive) | ladder nat (lat head) | 0.928 [0.834, 0.956] | 50490 | 85 (7) |
| P0 own | agent split | A-type: vehicle ahead in own lane (A positive) | ladder priv (lat head) | 0.998 [0.995, 0.999] | 50490 | 85 (7) |
| P0 own | agent split | A-type: vehicle ahead in own lane (A positive) | ladder vis (A head) | 0.990 [0.978, 0.994] | 50490 | 85 (7) |
| P0 own | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | S0 (mean of seeds) | 0.906 [0.862, 0.941] | 51082 | 677 (58) |
| P0 own | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | S0 blind seed 0 | 0.774 [0.728, 0.816] | 51082 | 677 (58) |
| P0 own | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | ladder vis (lat head) | 0.900 [0.858, 0.935] | 51082 | 677 (58) |
| P0 own | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | ladder nat (lat head) | 0.903 [0.858, 0.937] | 51082 | 677 (58) |
| P0 own | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | ladder priv (lat head) | 0.996 [0.994, 0.997] | 51082 | 677 (58) |
| P0 own | agent split | other vehicle contact (lateral / go-around / drift / cut-in / crossing) | ladder vis (A head) | 0.820 [0.767, 0.865] | 51082 | 677 (58) |
| P0 own | agent split | other object (pedestrian, cyclist, generic object) | S0 (mean of seeds) | 0.879 [0.831, 0.926] | 50554 | 149 (24) |
| P0 own | agent split | other object (pedestrian, cyclist, generic object) | S0 blind seed 0 | 0.759 [0.688, 0.818] | 50554 | 149 (24) |
| P0 own | agent split | other object (pedestrian, cyclist, generic object) | ladder vis (lat head) | 0.881 [0.833, 0.923] | 50554 | 149 (24) |
| P0 own | agent split | other object (pedestrian, cyclist, generic object) | ladder nat (lat head) | 0.881 [0.824, 0.930] | 50554 | 149 (24) |
| P0 own | agent split | other object (pedestrian, cyclist, generic object) | ladder priv (lat head) | 0.989 [0.976, 0.996] | 50554 | 149 (24) |
| P0 own | agent split | other object (pedestrian, cyclist, generic object) | ladder vis (A head) | 0.828 [0.764, 0.881] | 50554 | 149 (24) |
| P1 own + arc | A | S0 head as a score | S0 agent logit | 0.961 [0.944, 0.974] | 150999 | 1905 |
| P1 own + arc | A | S0 head as a score | S0 boundary logit | 0.429 [0.380, 0.473] | 150999 | 1905 |
| P1 own + arc | B | S0 head as a score | S0 agent logit | 0.731 [0.641, 0.824] | 153633 | 397 |
| P1 own + arc | B | S0 head as a score | S0 boundary logit | 0.602 [0.517, 0.671] | 153633 | 397 |
| P1 own + arc | C | S0 head as a score | S0 agent logit | 0.670 [0.622, 0.715] | 150888 | 1909 |
| P1 own + arc | C | S0 head as a score | S0 boundary logit | 0.750 [0.698, 0.801] | 150888 | 1909 |
| P0 own | A | S0 head as a score | S0 agent logit | 0.946 [0.908, 0.973] | 50490 | 85 |
| P0 own | A | S0 head as a score | S0 boundary logit | 0.403 [0.310, 0.466] | 50490 | 85 |
| P0 own | B | S0 head as a score | S0 agent logit | 0.585 [0.335, 0.813] | 51233 | 36 |
| P0 own | B | S0 head as a score | S0 boundary logit | 0.562 [0.385, 0.725] | 51233 | 36 |
| P0 own | C | S0 head as a score | S0 agent logit | 0.658 [0.454, 0.830] | 50954 | 111 |
| P0 own | C | S0 head as a score | S0 boundary logit | 0.378 [0.238, 0.541] | 50954 | 111 |

### Strata (P1, two-seed score, 2 000 resamples)

| label | stratum | value | positives (logs) | blind | vis | nat | priv | blind_nat |
|---|---|---|---|---|---|---|---|---|
| A | speed band | < 2 | 38 (14) | 0.746 [0.693, 0.810] | 0.981 [0.965, 0.990] | 0.991 [0.982, 0.996] | 1.000 [1.000, 1.000] | 0.970 [0.956, 0.989] |
| A | speed band | 2-5 | 431 (40) | 0.954 [0.941, 0.969] | 0.982 [0.976, 0.991] | 0.981 [0.974, 0.991] | 1.000 [1.000, 1.000] | 0.977 [0.969, 0.986] |
| A | speed band | 5-10 | 1282 (54) | 0.956 [0.939, 0.973] | 0.985 [0.977, 0.992] | 0.988 [0.983, 0.993] | 1.000 [1.000, 1.000] | 0.986 [0.981, 0.991] |
| A | speed band | >= 10 | 154 (14) | 0.930 [0.898, 0.953] | 0.978 [0.956, 0.991] | 0.984 [0.973, 0.992] | 0.998 [0.996, 1.000] | 0.981 [0.965, 0.991] |
| A | states | log | 627 (52) | 0.954 [0.939, 0.970] | 0.987 [0.981, 0.992] | 0.990 [0.985, 0.994] | 1.000 [1.000, 1.000] | 0.987 [0.981, 0.992] |
| A | states | ot1 | 675 (53) | 0.955 [0.942, 0.969] | 0.983 [0.973, 0.991] | 0.985 [0.979, 0.991] | 1.000 [0.999, 1.000] | 0.983 [0.977, 0.989] |
| A | states | yr1 | 603 (57) | 0.946 [0.933, 0.960] | 0.981 [0.975, 0.987] | 0.983 [0.978, 0.988] | 0.999 [0.999, 1.000] | 0.979 [0.974, 0.985] |
| A | query | own | 85 (7) | 0.888 [0.830, 0.932] | 0.990 [0.978, 0.994] | 0.989 [0.969, 0.994] | 1.000 [1.000, 1.000] | 0.986 [0.967, 0.992] |
| A | query | arc | 1820 (65) | 0.945 [0.933, 0.957] | 0.980 [0.972, 0.987] | 0.983 [0.977, 0.988] | 1.000 [0.999, 1.000] | 0.979 [0.972, 0.985] |
| A | user class | other | 434 (38) | 0.961 [0.951, 0.971] | 0.983 [0.970, 0.992] | 0.985 [0.978, 0.993] | 1.000 [1.000, 1.000] | 0.984 [0.979, 0.990] |
| A | user class | 1 obstacle ahead | 894 (36) | 0.933 [0.914, 0.956] | 0.966 [0.958, 0.975] | 0.971 [0.963, 0.979] | 0.999 [0.999, 1.000] | 0.965 [0.956, 0.973] |
| A | user class | 2 turn | 14 (4) | 0.977 [0.898, 0.993] | 0.995 [0.962, 1.000] | 0.994 [0.954, 1.000] | 1.000 [1.000, 1.000] | 0.994 [0.974, 1.000] |
| A | user class | 3 leaving the road | 563 (48) | 0.973 [0.956, 0.984] | 0.991 [0.979, 0.996] | 0.993 [0.987, 0.997] | 1.000 [0.999, 1.000] | 0.991 [0.982, 0.995] |
| A | > 45 deg | yes | 39 (5) | 0.862 [0.802, 0.991] | 0.989 [0.963, 0.999] | 0.989 [0.956, 0.998] | 1.000 [1.000, 1.000] | 0.994 [0.977, 1.000] |
| A | sub-class | A1 stopped | 1064 (52) | 0.954 [0.933, 0.972] | 0.983 [0.970, 0.992] | 0.985 [0.977, 0.992] | 1.000 [0.999, 1.000] | 0.983 [0.974, 0.990] |
| A | sub-class | A2 moving | 841 (39) | 0.952 [0.942, 0.964] | 0.986 [0.983, 0.989] | 0.988 [0.985, 0.991] | 1.000 [1.000, 1.000] | 0.984 [0.981, 0.988] |
| B | speed band | < 2 | 35 (6) | 0.785 [0.532, 0.894] | 0.928 [0.751, 0.981] | 0.857 [0.761, 0.956] | 0.998 [0.996, 1.000] | 0.764 [0.638, 0.963] |
| B | speed band | 2-5 | 152 (13) | 0.929 [0.865, 0.968] | 0.963 [0.921, 0.992] | 0.958 [0.905, 0.993] | 0.999 [0.999, 1.000] | 0.982 [0.971, 0.991] |
| B | speed band | 5-10 | 192 (16) | 0.911 [0.848, 0.968] | 0.980 [0.965, 0.994] | 0.964 [0.921, 0.994] | 0.999 [0.999, 1.000] | 0.966 [0.930, 0.991] |
| B | speed band | >= 10 | 18 (2) | 0.968 [0.958, 0.990] | 0.995 [0.977, 0.998] | 0.995 [0.979, 0.998] | 1.000 [0.999, 1.000] | 0.983 [0.964, 0.987] |
| B | states | log | 142 (21) | 0.893 [0.842, 0.933] | 0.956 [0.913, 0.984] | 0.938 [0.909, 0.968] | 0.998 [0.997, 1.000] | 0.919 [0.876, 0.974] |
| B | states | ot1 | 129 (19) | 0.933 [0.897, 0.963] | 0.982 [0.972, 0.991] | 0.970 [0.940, 0.991] | 0.999 [0.999, 1.000] | 0.979 [0.962, 0.990] |
| B | states | yr1 | 126 (21) | 0.912 [0.870, 0.958] | 0.970 [0.953, 0.986] | 0.951 [0.913, 0.987] | 0.999 [0.999, 1.000] | 0.967 [0.943, 0.987] |
| B | query | own | 36 (6) | 0.784 [0.542, 0.948] | 0.936 [0.848, 0.983] | 0.868 [0.719, 0.964] | 0.998 [0.995, 1.000] | 0.847 [0.692, 0.990] |
| B | query | arc | 361 (24) | 0.917 [0.889, 0.944] | 0.970 [0.953, 0.983] | 0.959 [0.928, 0.980] | 0.999 [0.998, 1.000] | 0.961 [0.942, 0.978] |
| B | user class | other | 160 (14) | 0.857 [0.750, 0.949] | 0.951 [0.907, 0.981] | 0.929 [0.874, 0.976] | 0.999 [0.997, 1.000] | 0.935 [0.880, 0.980] |
| B | user class | 1 obstacle ahead | 63 (11) | 0.906 [0.851, 0.972] | 0.965 [0.911, 0.995] | 0.923 [0.825, 0.988] | 0.997 [0.994, 1.000] | 0.909 [0.765, 0.992] |
| B | user class | 2 turn | 5 (1) | 0.989 [0.984, 0.993] | 0.997 [0.993, 0.999] | 0.997 [0.992, 1.000] | 1.000 [1.000, 1.000] | 0.994 [0.986, 0.999] |
| B | user class | 3 leaving the road | 169 (16) | 0.944 [0.914, 0.973] | 0.980 [0.965, 0.992] | 0.988 [0.982, 0.994] | 0.999 [0.998, 1.000] | 0.983 [0.974, 0.991] |
| B | > 45 deg | yes | 19 (3) | 0.975 [0.963, 0.999] | 0.998 [0.997, 1.000] | 0.993 [0.990, 1.000] | 1.000 [1.000, 1.000] | 0.996 [0.994, 1.000] |
| B | sub-class | red light | 273 (19) | 0.898 [0.820, 0.947] | 0.960 [0.930, 0.982] | 0.936 [0.884, 0.974] | 0.998 [0.997, 1.000] | 0.936 [0.894, 0.974] |
| B | sub-class | stop sign | 124 (11) | 0.943 [0.872, 0.984] | 0.987 [0.965, 0.998] | 0.987 [0.960, 0.998] | 1.000 [0.999, 1.000] | 0.991 [0.981, 0.997] |
| B | decision-relevant | eligible stop line <= 60 m ahead | 397 (24) | 0.853 [0.803, 0.905] | 0.943 [0.918, 0.966] | 0.929 [0.898, 0.960] | 0.994 [0.989, 0.998] | 0.928 [0.892, 0.961] |
| C | speed band | < 2 | 75 (27) | 0.988 [0.979, 0.995] | 0.991 [0.978, 0.998] | 0.990 [0.978, 0.997] | 1.000 [0.999, 1.000] | 0.987 [0.980, 0.993] |
| C | speed band | 2-5 | 864 (54) | 0.977 [0.965, 0.987] | 0.984 [0.967, 0.994] | 0.984 [0.965, 0.994] | 1.000 [1.000, 1.000] | 0.976 [0.947, 0.992] |
| C | speed band | 5-10 | 970 (54) | 0.986 [0.974, 0.994] | 0.995 [0.992, 0.997] | 0.995 [0.993, 0.997] | 1.000 [0.999, 1.000] | 0.993 [0.989, 0.996] |
| C | speed band | >= 10 | 0 | too few | too few | too few | too few | too few |
| C | states | log | 733 (70) | 0.985 [0.979, 0.991] | 0.992 [0.986, 0.996] | 0.991 [0.983, 0.996] | 1.000 [1.000, 1.000] | 0.987 [0.974, 0.994] |
| C | states | ot1 | 597 (59) | 0.983 [0.976, 0.990] | 0.992 [0.985, 0.995] | 0.991 [0.982, 0.996] | 1.000 [1.000, 1.000] | 0.986 [0.972, 0.993] |
| C | states | yr1 | 579 (62) | 0.983 [0.975, 0.990] | 0.990 [0.981, 0.996] | 0.990 [0.982, 0.996] | 1.000 [1.000, 1.000] | 0.987 [0.977, 0.994] |
| C | query | own | 111 (9) | 0.949 [0.915, 0.984] | 0.965 [0.905, 0.997] | 0.961 [0.889, 0.997] | 1.000 [0.999, 1.000] | 0.934 [0.808, 0.997] |
| C | query | arc | 1798 (73) | 0.984 [0.980, 0.989] | 0.992 [0.989, 0.995] | 0.992 [0.988, 0.994] | 1.000 [1.000, 1.000] | 0.988 [0.983, 0.992] |
| C | user class | other | 523 (41) | 0.968 [0.953, 0.981] | 0.976 [0.954, 0.991] | 0.973 [0.947, 0.990] | 0.999 [0.999, 1.000] | 0.958 [0.916, 0.985] |
| C | user class | 1 obstacle ahead | 322 (27) | 0.978 [0.955, 0.998] | 0.998 [0.997, 0.999] | 0.998 [0.996, 0.999] | 1.000 [1.000, 1.000] | 0.996 [0.994, 0.998] |
| C | user class | 2 turn | 995 (61) | 0.981 [0.974, 0.988] | 0.989 [0.985, 0.993] | 0.990 [0.986, 0.993] | 1.000 [1.000, 1.000] | 0.987 [0.983, 0.990] |
| C | user class | 3 leaving the road | 69 (18) | 0.952 [0.928, 0.985] | 0.976 [0.961, 0.991] | 0.986 [0.975, 0.995] | 0.999 [0.998, 1.000] | 0.986 [0.979, 0.994] |
| C | > 45 deg | yes | 1172 (66) | 0.975 [0.951, 0.990] | 0.992 [0.988, 0.995] | 0.992 [0.989, 0.995] | 1.000 [1.000, 1.000] | 0.990 [0.986, 0.993] |
| C | decision-relevant | eligible turn peak <= 60 m ahead | 1909 (73) | 0.968 [0.954, 0.979] | 0.981 [0.970, 0.989] | 0.980 [0.968, 0.988] | 0.999 [0.999, 1.000] | 0.974 [0.957, 0.985] |

### Kinematic baselines and the hard subset (prereg addendum 1; descriptive)

| population | label | score | AUC [95% CI] | n | positives (logs) |
|---|---|---|---|--:|---|
| P1 own + arc | A | rung blind | 0.953 [0.940, 0.967] | 150999 | 1905 (65) |
| P1 own + arc | A | rung blind_nat | 0.983 [0.978, 0.989] | 150999 | 1905 (65) |
| P1 own + arc | A | rung nat | 0.987 [0.982, 0.991] | 150999 | 1905 (65) |
| P1 own + arc | A | rung priv | 1.000 [0.999, 1.000] | 150999 | 1905 (65) |
| P1 own + arc | A | rung vis | 0.984 [0.977, 0.990] | 150999 | 1905 (65) |
| P1 own + arc | A | K1 v0 | 0.588 [0.539, 0.638] | 150999 | 1905 (65) |
| P1 own + arc | A | K2 a0 | 0.718 [0.670, 0.768] | 150999 | 1905 (65) |
| P1 own + arc | A | K3 v0 + a0 | 0.721 [0.671, 0.775] | 150999 | 1905 (65) |
| P1 own + arc | A | K4 plan speed profile | 0.926 [0.898, 0.955] | 150999 | 1905 (65) |
| P1 own + arc | A | K5 K4 + plan curvature | 0.926 [0.897, 0.956] | 150999 | 1905 (65) |
| P1 own + arc | A | plan 4 s arc length | 0.627 [0.585, 0.668] | 150999 | 1905 (65) |
| P0 own | A | rung blind | 0.888 [0.830, 0.932] | 50490 | 85 (7) |
| P0 own | A | rung blind_nat | 0.986 [0.967, 0.992] | 50490 | 85 (7) |
| P0 own | A | rung nat | 0.989 [0.969, 0.994] | 50490 | 85 (7) |
| P0 own | A | rung priv | 1.000 [1.000, 1.000] | 50490 | 85 (7) |
| P0 own | A | rung vis | 0.990 [0.978, 0.994] | 50490 | 85 (7) |
| P0 own | A | K1 v0 | 0.460 [0.323, 0.543] | 50490 | 85 (7) |
| P0 own | A | K2 a0 | 0.625 [0.396, 0.744] | 50490 | 85 (7) |
| P0 own | A | K3 v0 + a0 | 0.617 [0.385, 0.742] | 50490 | 85 (7) |
| P0 own | A | K4 plan speed profile | 0.613 [0.497, 0.727] | 50490 | 85 (7) |
| P0 own | A | K5 K4 + plan curvature | 0.611 [0.483, 0.754] | 50490 | 85 (7) |
| P0 own | A | plan 4 s arc length | 0.390 [0.291, 0.458] | 50490 | 85 (7) |
| P1 & H (no kinematic cue) | A | rung blind | 0.915 [0.874, 0.966] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | rung blind_nat | 0.984 [0.975, 0.995] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | rung nat | 0.985 [0.975, 0.995] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | rung priv | 1.000 [1.000, 1.000] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | rung vis | 0.982 [0.969, 0.994] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | K1 v0 | 0.446 [0.332, 0.591] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | K2 a0 | 0.704 [0.603, 0.816] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | K3 v0 + a0 | 0.671 [0.569, 0.780] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | K4 plan speed profile | 0.878 [0.841, 0.948] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | K5 K4 + plan curvature | 0.873 [0.829, 0.948] | 13693 | 139 (18) |
| P1 & H (no kinematic cue) | A | plan 4 s arc length | 0.572 [0.455, 0.715] | 13693 | 139 (18) |
| P0 & H | A | rung blind | 0.868 [0.763, 0.993] | 4581 | 17 (3) |
| P0 & H | A | rung blind_nat | 0.988 [0.977, 0.999] | 4581 | 17 (3) |
| P0 & H | A | rung nat | 0.990 [0.977, 0.999] | 4581 | 17 (3) |
| P0 & H | A | rung priv | 1.000 [0.999, 1.000] | 4581 | 17 (3) |
| P0 & H | A | rung vis | 0.987 [0.976, 0.996] | 4581 | 17 (3) |
| P0 & H | A | K1 v0 | 0.413 [0.261, 0.621] | 4581 | 17 (3) |
| P0 & H | A | K2 a0 | 0.638 [0.276, 0.970] | 4581 | 17 (3) |
| P0 & H | A | K3 v0 + a0 | 0.600 [0.244, 0.933] | 4581 | 17 (3) |
| P0 & H | A | K4 plan speed profile | 0.619 [0.403, 0.727] | 4581 | 17 (3) |
| P0 & H | A | K5 K4 + plan curvature | 0.611 [0.336, 0.728] | 4581 | 17 (3) |
| P0 & H | A | plan 4 s arc length | 0.387 [0.312, 0.488] | 4581 | 17 (3) |
| P1 & H0 (a0 >= 0) | A | rung blind | 0.911 [0.881, 0.946] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | rung blind_nat | 0.985 [0.977, 0.992] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | rung nat | 0.990 [0.985, 0.995] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | rung priv | 1.000 [1.000, 1.000] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | rung vis | 0.988 [0.982, 0.993] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | K1 v0 | 0.499 [0.416, 0.605] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | K2 a0 | 0.598 [0.470, 0.719] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | K3 v0 + a0 | 0.592 [0.464, 0.725] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | K4 plan speed profile | 0.889 [0.862, 0.925] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | K5 K4 + plan curvature | 0.888 [0.861, 0.922] | 66158 | 218 (26) |
| P1 & H0 (a0 >= 0) | A | plan 4 s arc length | 0.611 [0.512, 0.726] | 66158 | 218 (26) |
| P1 own + arc | B | rung blind | 0.912 [0.874, 0.948] | 153633 | 397 (24) |
| P1 own + arc | B | rung blind_nat | 0.953 [0.925, 0.980] | 153633 | 397 (24) |
| P1 own + arc | B | rung nat | 0.952 [0.923, 0.978] | 153633 | 397 (24) |
| P1 own + arc | B | rung priv | 0.999 [0.997, 1.000] | 153633 | 397 (24) |
| P1 own + arc | B | rung vis | 0.969 [0.949, 0.984] | 153633 | 397 (24) |
| P1 own + arc | B | K1 v0 | 0.533 [0.444, 0.649] | 153633 | 397 (24) |
| P1 own + arc | B | K2 a0 | 0.599 [0.533, 0.674] | 153633 | 397 (24) |
| P1 own + arc | B | K3 v0 + a0 | 0.608 [0.537, 0.688] | 153633 | 397 (24) |
| P1 own + arc | B | K4 plan speed profile | 0.867 [0.816, 0.915] | 153633 | 397 (24) |
| P1 own + arc | B | K5 K4 + plan curvature | 0.883 [0.841, 0.922] | 153633 | 397 (24) |
| P1 own + arc | B | plan min speed after 0.5 s | 0.467 [0.356, 0.570] | 153633 | 397 (24) |
| P1 own + arc | B | plan 4 s arc length | 0.543 [0.425, 0.635] | 153633 | 397 (24) |
| P0 own | B | rung blind | 0.784 [0.542, 0.948] | 51233 | 36 (6) |
| P0 own | B | rung blind_nat | 0.847 [0.692, 0.990] | 51233 | 36 (6) |
| P0 own | B | rung nat | 0.868 [0.719, 0.964] | 51233 | 36 (6) |
| P0 own | B | rung priv | 0.998 [0.995, 1.000] | 51233 | 36 (6) |
| P0 own | B | rung vis | 0.936 [0.848, 0.983] | 51233 | 36 (6) |
| P0 own | B | K1 v0 | 0.697 [0.497, 0.894] | 51233 | 36 (6) |
| P0 own | B | K2 a0 | 0.510 [0.286, 0.729] | 51233 | 36 (6) |
| P0 own | B | K3 v0 + a0 | 0.593 [0.348, 0.775] | 51233 | 36 (6) |
| P0 own | B | K4 plan speed profile | 0.755 [0.436, 0.968] | 51233 | 36 (6) |
| P0 own | B | K5 K4 + plan curvature | 0.764 [0.461, 0.960] | 51233 | 36 (6) |
| P0 own | B | plan min speed after 0.5 s | 0.249 [0.075, 0.536] | 51233 | 36 (6) |
| P0 own | B | plan 4 s arc length | 0.267 [0.090, 0.550] | 51233 | 36 (6) |
| P1 & H (no kinematic cue) | B | rung blind | 0.959 [0.936, 0.979] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | rung blind_nat | 0.991 [0.987, 0.995] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | rung nat | 0.989 [0.981, 0.996] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | rung priv | 0.999 [0.998, 1.000] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | rung vis | 0.991 [0.985, 0.996] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | K1 v0 | 0.601 [0.471, 0.728] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | K2 a0 | 0.731 [0.640, 0.819] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | K3 v0 + a0 | 0.752 [0.671, 0.826] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | K4 plan speed profile | 0.930 [0.877, 0.977] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | K5 K4 + plan curvature | 0.946 [0.909, 0.975] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | plan min speed after 0.5 s | 0.506 [0.391, 0.638] | 14023 | 65 (10) |
| P1 & H (no kinematic cue) | B | plan 4 s arc length | 0.590 [0.485, 0.702] | 14023 | 65 (10) |
| P0 & H | B | (too few positives) | n/a | 4672 | 0 |
| P1 & H0 (a0 >= 0) | B | rung blind | 0.901 [0.832, 0.966] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | rung blind_nat | 0.892 [0.820, 0.977] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | rung nat | 0.912 [0.856, 0.982] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | rung priv | 0.999 [0.997, 1.000] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | rung vis | 0.971 [0.946, 0.993] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | K1 v0 | 0.578 [0.433, 0.754] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | K2 a0 | 0.636 [0.527, 0.767] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | K3 v0 + a0 | 0.677 [0.586, 0.789] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | K4 plan speed profile | 0.840 [0.753, 0.946] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | K5 K4 + plan curvature | 0.862 [0.794, 0.955] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | plan min speed after 0.5 s | 0.466 [0.297, 0.599] | 67040 | 106 (15) |
| P1 & H0 (a0 >= 0) | B | plan 4 s arc length | 0.535 [0.358, 0.670] | 67040 | 106 (15) |
| P1 own + arc | C | rung blind | 0.984 [0.977, 0.990] | 150888 | 1909 (73) |
| P1 own + arc | C | rung blind_nat | 0.987 [0.974, 0.994] | 150888 | 1909 (73) |
| P1 own + arc | C | rung nat | 0.991 [0.982, 0.995] | 150888 | 1909 (73) |
| P1 own + arc | C | rung priv | 1.000 [1.000, 1.000] | 150888 | 1909 (73) |
| P1 own + arc | C | rung vis | 0.991 [0.985, 0.996] | 150888 | 1909 (73) |
| P1 own + arc | C | K1 v0 | 0.562 [0.521, 0.606] | 150888 | 1909 (73) |
| P1 own + arc | C | K2 a0 | 0.587 [0.540, 0.630] | 150888 | 1909 (73) |
| P1 own + arc | C | K3 v0 + a0 | 0.615 [0.579, 0.651] | 150888 | 1909 (73) |
| P1 own + arc | C | K4 plan speed profile | 0.922 [0.903, 0.939] | 150888 | 1909 (73) |
| P1 own + arc | C | K5 K4 + plan curvature | 0.969 [0.960, 0.979] | 150888 | 1909 (73) |
| P1 own + arc | C | plan max v^2 kappa | 0.946 [0.930, 0.960] | 150888 | 1909 (73) |
| P1 own + arc | C | plan speed at its max curvature | 0.699 [0.658, 0.734] | 150888 | 1909 (73) |
| P0 own | C | rung blind | 0.949 [0.915, 0.984] | 50954 | 111 (9) |
| P0 own | C | rung blind_nat | 0.934 [0.808, 0.997] | 50954 | 111 (9) |
| P0 own | C | rung nat | 0.961 [0.889, 0.997] | 50954 | 111 (9) |
| P0 own | C | rung priv | 1.000 [0.999, 1.000] | 50954 | 111 (9) |
| P0 own | C | rung vis | 0.965 [0.905, 0.997] | 50954 | 111 (9) |
| P0 own | C | K1 v0 | 0.480 [0.360, 0.669] | 50954 | 111 (9) |
| P0 own | C | K2 a0 | 0.518 [0.400, 0.623] | 50954 | 111 (9) |
| P0 own | C | K3 v0 + a0 | 0.537 [0.454, 0.622] | 50954 | 111 (9) |
| P0 own | C | K4 plan speed profile | 0.753 [0.692, 0.842] | 50954 | 111 (9) |
| P0 own | C | K5 K4 + plan curvature | 0.894 [0.854, 0.930] | 50954 | 111 (9) |
| P0 own | C | plan max v^2 kappa | 0.836 [0.718, 0.911] | 50954 | 111 (9) |
| P0 own | C | plan speed at its max curvature | 0.538 [0.344, 0.670] | 50954 | 111 (9) |
| P1 & H (no kinematic cue) | C | rung blind | 0.968 [0.949, 0.984] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | rung blind_nat | 0.985 [0.980, 0.992] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | rung nat | 0.989 [0.985, 0.994] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | rung priv | 1.000 [0.999, 1.000] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | rung vis | 0.986 [0.978, 0.993] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | K1 v0 | 0.607 [0.513, 0.722] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | K2 a0 | 0.492 [0.415, 0.553] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | K3 v0 + a0 | 0.575 [0.481, 0.674] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | K4 plan speed profile | 0.873 [0.844, 0.915] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | K5 K4 + plan curvature | 0.947 [0.934, 0.967] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | plan max v^2 kappa | 0.913 [0.864, 0.948] | 13788 | 307 (25) |
| P1 & H (no kinematic cue) | C | plan speed at its max curvature | 0.624 [0.538, 0.695] | 13788 | 307 (25) |
| P0 & H | C | rung blind | 0.968 [0.942, 0.998] | 4623 | 31 (3) |
| P0 & H | C | rung blind_nat | 0.997 [0.991, 1.000] | 4623 | 31 (3) |
| P0 & H | C | rung nat | 0.998 [0.994, 1.000] | 4623 | 31 (3) |
| P0 & H | C | rung priv | 1.000 [1.000, 1.000] | 4623 | 31 (3) |
| P0 & H | C | rung vis | 0.997 [0.992, 1.000] | 4623 | 31 (3) |
| P0 & H | C | K1 v0 | 0.368 [0.280, 0.464] | 4623 | 31 (3) |
| P0 & H | C | K2 a0 | 0.571 [0.252, 0.799] | 4623 | 31 (3) |
| P0 & H | C | K3 v0 + a0 | 0.438 [0.234, 0.651] | 4623 | 31 (3) |
| P0 & H | C | K4 plan speed profile | 0.717 [0.636, 0.794] | 4623 | 31 (3) |
| P0 & H | C | K5 K4 + plan curvature | 0.905 [0.835, 0.956] | 4623 | 31 (3) |
| P0 & H | C | plan max v^2 kappa | 0.930 [0.865, 0.979] | 4623 | 31 (3) |
| P0 & H | C | plan speed at its max curvature | 0.631 [0.534, 0.715] | 4623 | 31 (3) |
| P1 & H0 (a0 >= 0) | C | rung blind | 0.983 [0.976, 0.990] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | rung blind_nat | 0.992 [0.989, 0.994] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | rung nat | 0.994 [0.992, 0.995] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | rung priv | 1.000 [1.000, 1.000] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | rung vis | 0.993 [0.990, 0.996] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | K1 v0 | 0.555 [0.492, 0.618] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | K2 a0 | 0.500 [0.451, 0.553] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | K3 v0 + a0 | 0.533 [0.464, 0.601] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | K4 plan speed profile | 0.904 [0.885, 0.923] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | K5 K4 + plan curvature | 0.961 [0.947, 0.973] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | plan max v^2 kappa | 0.925 [0.906, 0.944] | 65452 | 1003 (60) |
| P1 & H0 (a0 >= 0) | C | plan speed at its max curvature | 0.692 [0.646, 0.732] | 65452 | 1003 (60) |
| P1 own + arc | a | rung blind | 0.833 [0.811, 0.853] | 153948 | 4854 (98) |
| P1 own + arc | a | rung blind_nat | 0.909 [0.889, 0.925] | 153948 | 4854 (98) |
| P1 own + arc | a | rung nat | 0.922 [0.905, 0.936] | 153948 | 4854 (98) |
| P1 own + arc | a | rung priv | 0.993 [0.991, 0.995] | 153948 | 4854 (98) |
| P1 own + arc | a | rung vis | 0.919 [0.903, 0.932] | 153948 | 4854 (98) |
| P1 own + arc | a | K1 v0 | 0.550 [0.514, 0.583] | 153948 | 4854 (98) |
| P1 own + arc | a | K2 a0 | 0.605 [0.571, 0.641] | 153948 | 4854 (98) |
| P1 own + arc | a | K3 v0 + a0 | 0.613 [0.576, 0.652] | 153948 | 4854 (98) |
| P1 own + arc | a | K4 plan speed profile | 0.748 [0.718, 0.779] | 153948 | 4854 (98) |
| P1 own + arc | a | K5 K4 + plan curvature | 0.755 [0.727, 0.786] | 153948 | 4854 (98) |
| P0 own | a | rung blind | 0.779 [0.742, 0.809] | 51316 | 911 (64) |
| P0 own | a | rung blind_nat | 0.892 [0.858, 0.918] | 51316 | 911 (64) |
| P0 own | a | rung nat | 0.902 [0.867, 0.929] | 51316 | 911 (64) |
| P0 own | a | rung priv | 0.995 [0.992, 0.996] | 51316 | 911 (64) |
| P0 own | a | rung vis | 0.900 [0.866, 0.927] | 51316 | 911 (64) |
| P0 own | a | K1 v0 | 0.528 [0.467, 0.589] | 51316 | 911 (64) |
| P0 own | a | K2 a0 | 0.530 [0.476, 0.584] | 51316 | 911 (64) |
| P0 own | a | K3 v0 + a0 | 0.546 [0.487, 0.601] | 51316 | 911 (64) |
| P0 own | a | K4 plan speed profile | 0.514 [0.457, 0.575] | 51316 | 911 (64) |
| P0 own | a | K5 K4 + plan curvature | 0.546 [0.495, 0.599] | 51316 | 911 (64) |
| P1 & H (no kinematic cue) | a | rung blind | 0.789 [0.741, 0.828] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | rung blind_nat | 0.920 [0.894, 0.944] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | rung nat | 0.940 [0.916, 0.961] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | rung priv | 0.995 [0.991, 0.997] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | rung vis | 0.927 [0.904, 0.951] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | K1 v0 | 0.413 [0.336, 0.510] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | K2 a0 | 0.494 [0.381, 0.646] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | K3 v0 + a0 | 0.457 [0.347, 0.599] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | K4 plan speed profile | 0.679 [0.584, 0.792] | 14052 | 498 (40) |
| P1 & H (no kinematic cue) | a | K5 K4 + plan curvature | 0.684 [0.612, 0.786] | 14052 | 498 (40) |
| P0 & H | a | rung blind | 0.740 [0.635, 0.805] | 4684 | 120 (20) |
| P0 & H | a | rung blind_nat | 0.916 [0.862, 0.952] | 4684 | 120 (20) |
| P0 & H | a | rung nat | 0.945 [0.896, 0.974] | 4684 | 120 (20) |
| P0 & H | a | rung priv | 0.997 [0.994, 0.999] | 4684 | 120 (20) |
| P0 & H | a | rung vis | 0.930 [0.891, 0.960] | 4684 | 120 (20) |
| P0 & H | a | K1 v0 | 0.394 [0.266, 0.562] | 4684 | 120 (20) |
| P0 & H | a | K2 a0 | 0.425 [0.298, 0.664] | 4684 | 120 (20) |
| P0 & H | a | K3 v0 + a0 | 0.413 [0.264, 0.665] | 4684 | 120 (20) |
| P0 & H | a | K4 plan speed profile | 0.434 [0.326, 0.621] | 4684 | 120 (20) |
| P0 & H | a | K5 K4 + plan curvature | 0.466 [0.362, 0.597] | 4684 | 120 (20) |
| P1 & H0 (a0 >= 0) | a | rung blind | 0.756 [0.720, 0.787] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | rung blind_nat | 0.863 [0.825, 0.896] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | rung nat | 0.879 [0.837, 0.914] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | rung priv | 0.989 [0.984, 0.993] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | rung vis | 0.878 [0.845, 0.907] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | K1 v0 | 0.511 [0.443, 0.581] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | K2 a0 | 0.521 [0.464, 0.583] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | K3 v0 + a0 | 0.528 [0.459, 0.602] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | K4 plan speed profile | 0.661 [0.620, 0.702] | 67284 | 1344 (64) |
| P1 & H0 (a0 >= 0) | a | K5 K4 + plan curvature | 0.670 [0.629, 0.716] | 67284 | 1344 (64) |
| P1 own + arc | b | rung blind | 0.820 [0.797, 0.840] | 148037 | 4468 (94) |
| P1 own + arc | b | rung blind_nat | 0.962 [0.951, 0.972] | 148037 | 4468 (94) |
| P1 own + arc | b | rung nat | 0.974 [0.965, 0.982] | 148037 | 4468 (94) |
| P1 own + arc | b | rung priv | 0.985 [0.978, 0.990] | 148037 | 4468 (94) |
| P1 own + arc | b | rung vis | 0.978 [0.970, 0.984] | 148037 | 4468 (94) |
| P1 own + arc | b | K1 v0 | 0.539 [0.506, 0.574] | 148037 | 4468 (94) |
| P1 own + arc | b | K2 a0 | 0.517 [0.489, 0.545] | 148037 | 4468 (94) |
| P1 own + arc | b | K3 v0 + a0 | 0.546 [0.514, 0.578] | 148037 | 4468 (94) |
| P1 own + arc | b | K4 plan speed profile | 0.726 [0.704, 0.747] | 148037 | 4468 (94) |
| P1 own + arc | b | K5 K4 + plan curvature | 0.810 [0.790, 0.829] | 148037 | 4468 (94) |
| P0 own | b | rung blind | 0.785 [0.748, 0.817] | 49417 | 1090 (82) |
| P0 own | b | rung blind_nat | 0.960 [0.946, 0.971] | 49417 | 1090 (82) |
| P0 own | b | rung nat | 0.972 [0.961, 0.981] | 49417 | 1090 (82) |
| P0 own | b | rung priv | 0.985 [0.975, 0.991] | 49417 | 1090 (82) |
| P0 own | b | rung vis | 0.977 [0.969, 0.984] | 49417 | 1090 (82) |
| P0 own | b | K1 v0 | 0.527 [0.485, 0.572] | 49417 | 1090 (82) |
| P0 own | b | K2 a0 | 0.534 [0.502, 0.565] | 49417 | 1090 (82) |
| P0 own | b | K3 v0 + a0 | 0.551 [0.515, 0.590] | 49417 | 1090 (82) |
| P0 own | b | K4 plan speed profile | 0.634 [0.599, 0.664] | 49417 | 1090 (82) |
| P0 own | b | K5 K4 + plan curvature | 0.751 [0.718, 0.779] | 49417 | 1090 (82) |
| P1 & H (no kinematic cue) | b | rung blind | 0.723 [0.674, 0.760] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | rung blind_nat | 0.953 [0.930, 0.969] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | rung nat | 0.967 [0.954, 0.982] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | rung priv | 0.980 [0.969, 0.989] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | rung vis | 0.974 [0.961, 0.984] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | K1 v0 | 0.456 [0.377, 0.555] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | K2 a0 | 0.407 [0.331, 0.475] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | K3 v0 + a0 | 0.414 [0.331, 0.519] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | K4 plan speed profile | 0.652 [0.603, 0.697] | 13348 | 448 (43) |
| P1 & H (no kinematic cue) | b | K5 K4 + plan curvature | 0.699 [0.638, 0.758] | 13348 | 448 (43) |
| P0 & H | b | rung blind | 0.692 [0.612, 0.771] | 4462 | 117 (28) |
| P0 & H | b | rung blind_nat | 0.950 [0.920, 0.975] | 4462 | 117 (28) |
| P0 & H | b | rung nat | 0.960 [0.940, 0.985] | 4462 | 117 (28) |
| P0 & H | b | rung priv | 0.984 [0.974, 0.991] | 4462 | 117 (28) |
| P0 & H | b | rung vis | 0.975 [0.961, 0.990] | 4462 | 117 (28) |
| P0 & H | b | K1 v0 | 0.494 [0.402, 0.618] | 4462 | 117 (28) |
| P0 & H | b | K2 a0 | 0.426 [0.332, 0.503] | 4462 | 117 (28) |
| P0 & H | b | K3 v0 + a0 | 0.444 [0.345, 0.575] | 4462 | 117 (28) |
| P0 & H | b | K4 plan speed profile | 0.500 [0.414, 0.569] | 4462 | 117 (28) |
| P0 & H | b | K5 K4 + plan curvature | 0.610 [0.526, 0.683] | 4462 | 117 (28) |
| P1 & H0 (a0 >= 0) | b | rung blind | 0.778 [0.746, 0.806] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | rung blind_nat | 0.951 [0.939, 0.961] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | rung nat | 0.967 [0.959, 0.974] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | rung priv | 0.983 [0.977, 0.987] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | rung vis | 0.971 [0.962, 0.978] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | K1 v0 | 0.506 [0.459, 0.558] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | K2 a0 | 0.435 [0.398, 0.470] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | K3 v0 + a0 | 0.464 [0.419, 0.517] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | K4 plan speed profile | 0.700 [0.670, 0.728] | 64838 | 1924 (73) |
| P1 & H0 (a0 >= 0) | b | K5 K4 + plan curvature | 0.767 [0.738, 0.796] | 64838 | 1924 (73) |
