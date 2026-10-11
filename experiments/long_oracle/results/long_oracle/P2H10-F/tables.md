## navtest: failure sets

| set | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of 12 146 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NC | 188 | 176 | 182 | 77.7 | 38.7 | 59.6 | 74.5 | 70.1 | 95.1 | 1.5 |
| TTC-only | 86 | 81 | 83.5 | 62.9 | 17.4 | 26.9 | 73.7 | 63.5 | 84.4 | 0.7 |

### NC failures by class

| big | n_s0 | n_s1 | n | share % | share % [95% CI] | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | WA-JEPA passes % [95% CI] | arm-specific (fails, WA passes): n | arm-specific share of set % [95% CI] | recovered, main % [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A stopped / slow vehicle ahead | 94 | 93 | 93.5 | 51.4 | 51 [41, 61] | 79.1 | 47.6 | 72.7 | 76.5 | 71.7 | 95.7 | 79 [69, 90] | 74 | 41 [33, 50] | 76 [63, 89] |
| B cut-in | 21 | 21 | 21 | 11.5 | 12 [5, 20] | 57.1 | 76.2 | 97.6 | 90.5 | 85.7 | 100 | 57 [30, 85] | 12 | 7 [2, 12] | 90 [81, 100] |
| C crossing at junction | 15 | 14 | 14.5 | 8.0 | 8 [0, 17] | 62.1 | 48.3 | 75.9 | 75.9 | 72.4 | 93.1 | 62 [40, 100] | 9 | 5 [0, 10] | 76 [53, 100] |
| D side contact | 19 | 17 | 18 | 9.9 | 10 [5, 16] | 97.2 | 2.8 | 2.8 | 72.2 | 72.2 | 88.9 | 97 [90, 100] | 17.5 | 10 [5, 16] | 72 [56, 89] |
| E other | 39 | 31 | 35 | 19.2 | 19 [12, 28] | 82.9 | 7.1 | 24.3 | 60 | 54.3 | 94.3 | 83 [66, 100] | 29 | 16 [10, 24] | 60 [43, 77] |

### NC failures by sub-class

| cls | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 58 | 60 | 59 | 32.4 | 82.2 | 47.5 | 56.8 | 66.9 | 59.3 | 93.2 |
| A2 lead vehicle (moving) | 36 | 33 | 34.5 | 19.0 | 73.9 | 47.8 | 100 | 92.8 | 92.8 | 100 |
| B cut-in | 21 | 21 | 21 | 11.5 | 57.1 | 76.2 | 97.6 | 90.5 | 85.7 | 100 |
| C crossing / turn conflict | 15 | 14 | 14.5 | 8.0 | 62.1 | 48.3 | 75.9 | 75.9 | 72.4 | 93.1 |
| D side contact | 19 | 17 | 18 | 9.9 | 97.2 | 2.8 | 2.8 | 72.2 | 72.2 | 88.9 |
| E VRU | 7 | 7 | 7 | 3.8 | 42.9 | 28.6 | 28.6 | 42.9 | 28.6 | 85.7 |
| E oncoming | 9 | 8 | 8.5 | 4.7 | 76.5 | 5.9 | 52.9 | 58.8 | 58.8 | 100 |
| E static object | 23 | 16 | 19.5 | 10.7 | 100 | 0.0 | 10.3 | 66.7 | 61.5 | 94.9 |

### NC failures by decision-153 fine class

| fine | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| VRU | 7 | 7 | 7 | 3.8 | 42.9 | 28.6 | 28.6 | 42.9 | 28.6 | 85.7 |
| crossing / turn conflict | 15 | 14 | 14.5 | 8.0 | 62.1 | 48.3 | 75.9 | 75.9 | 72.4 | 93.1 |
| cut-in | 21 | 21 | 21 | 11.5 | 57.1 | 76.2 | 97.6 | 90.5 | 85.7 | 100 |
| lead vehicle | 36 | 33 | 34.5 | 19.0 | 73.9 | 47.8 | 100 | 92.8 | 92.8 | 100 |
| oncoming | 9 | 8 | 8.5 | 4.7 | 76.5 | 5.9 | 52.9 | 58.8 | 58.8 | 100 |
| sideswipe (ego across lanes) | 5 | 5 | 5 | 2.7 | 100 | 0.0 | 0.0 | 80 | 80 | 80 |
| static object | 23 | 16 | 19.5 | 10.7 | 100 | 0.0 | 10.3 | 66.7 | 61.5 | 94.9 |
| stopped vehicle ahead | 72 | 72 | 72 | 39.6 | 84.7 | 39.6 | 47.2 | 67.4 | 61.1 | 93.1 |

### NC failures: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 21 | 11 | 10.5 | 16.5 | 59 |
| A2 lead vehicle (moving) | 33.5 | 1 | 0.0 | 0.0 | 34.5 |
| B cut-in | 10.5 | 9.5 | 1 | 0.0 | 21 |
| C crossing / turn conflict | 8 | 3 | 2 | 1.5 | 14.5 |
| D side contact | 8 | 3.5 | 2 | 4.5 | 18 |
| E static object | 12.5 | 2.5 | 2 | 2.5 | 19.5 |
| E VRU | 3 | 3 | 1 | 0.0 | 7 |
| E oncoming | 3.5 | 1 | 2 | 2 | 8.5 |
| all | 100 | 34.5 | 20.5 | 27 | 182 |

### NC failures: class x t0 ego speed band, m/s (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 18 | 29.5 | 11.5 | 0.0 | 59 |
| A2 lead vehicle (moving) | 7 | 9 | 18.5 | 0.0 | 34.5 |
| B cut-in | 2 | 14.5 | 4.5 | 0.0 | 21 |
| C crossing / turn conflict | 2.5 | 9 | 3 | 0.0 | 14.5 |
| D side contact | 0.5 | 5.5 | 9 | 3 | 18 |
| E static object | 0.0 | 4.5 | 15 | 0.0 | 19.5 |
| E VRU | 1 | 2 | 4 | 0.0 | 7 |
| E oncoming | 1 | 4.5 | 3 | 0.0 | 8.5 |
| all | 32 | 78.5 | 68.5 | 3 | 182 |

### NC failures by turn bucket

| turn | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of bucket |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <5 | 105 | 95 | 100 | 76.50 | 52 | 71 | 70.50 | 68.50 | 92 | 1.56 |
| 5-20 | 36 | 33 | 34.50 | 68.12 | 47.83 | 66.67 | 71.01 | 65.22 | 97.10 | 1.33 |
| 20-45 | 20 | 21 | 20.50 | 95.12 | 0.00 | 39.02 | 82.93 | 63.41 | 100 | 1.25 |
| >45 | 27 | 27 | 27 | 81.48 | 7.41 | 24.07 | 87.04 | 87.04 | 100 | 1.78 |

### NC failures by t0 ego speed band

| speed | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of band |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <2 | 34 | 30 | 32 | 82.81 | 85.94 | 93.75 | 65.62 | 59.38 | 96.88 | 1.26 |
| 2-5 | 78 | 79 | 78.50 | 71.97 | 40.13 | 67.52 | 73.25 | 68.15 | 96.18 | 2.27 |
| 5-10 | 73 | 64 | 68.50 | 81.02 | 16.79 | 37.23 | 80.29 | 77.37 | 94.16 | 1.32 |
| >=10 | 3 | 3 | 3 | 100 | 0.00 | 0.00 | 66.67 | 66.67 | 66.67 | 0.31 |

### TTC-only failures by class

| big | n_s0 | n_s1 | n | share % | share % [95% CI] | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | WA-JEPA passes % [95% CI] | arm-specific (fails, WA passes): n | arm-specific share of set % [95% CI] | recovered, main % [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A stopped / slow vehicle ahead | 23 | 19 | 21 | 25.1 | 25 [16, 36] | 71.4 | 26.2 | 52.4 | 76.2 | 71.4 | 92.9 | 71 [49, 93] | 15 | 18 [10, 27] | 76 [59, 93] |
| B cut-in | 10 | 9 | 9.5 | 11.4 | 11 [2, 24] | 26.3 | 10.5 | 0.0 | 57.9 | 57.9 | 57.9 | 26 [5, 75] | 2.5 | 3 [0, 7] | 58 [14, 100] |
| C crossing at junction | 5 | 5 | 5 | 6.0 | 6 [1, 14] | 70 | 0.0 | 10 | 80 | 80 | 80 | 70 [0, 100] | 3.5 | 4 [0, 12] | 80 [68, 100] |
| D side contact | 7 | 6 | 6.5 | 7.8 | 8 [2, 16] | 100 | 0.0 | 0.0 | 100 | 100 | 100 | 100 [100, 100] | 6.5 | 8 [2, 16] | 100 [100, 100] |
| E other | 41 | 42 | 41.5 | 49.7 | 50 [33, 65] | 60.2 | 19.3 | 26.5 | 71.1 | 53.0 | 84.3 | 60 [47, 75] | 25 | 30 [19, 41] | 71 [59, 81] |

### TTC-only failures by sub-class

| cls | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 20 | 16 | 18 | 21.6 | 66.7 | 27.8 | 50 | 72.2 | 66.7 | 91.7 |
| A2 lead vehicle (moving) | 3 | 3 | 3 | 3.6 | 100 | 16.7 | 66.7 | 100 | 100 | 100 |
| B cut-in | 10 | 9 | 9.5 | 11.4 | 26.3 | 10.5 | 0.0 | 57.9 | 57.9 | 57.9 |
| C crossing / turn conflict | 5 | 5 | 5 | 6.0 | 70 | 0.0 | 10 | 80 | 80 | 80 |
| D side contact | 7 | 6 | 6.5 | 7.8 | 100 | 0.0 | 0.0 | 100 | 100 | 100 |
| E VRU | 7 | 9 | 8 | 9.6 | 68.8 | 0.0 | 37.5 | 68.8 | 62.5 | 81.2 |
| E oncoming | 9 | 8 | 8.5 | 10.2 | 17.6 | 0.0 | 17.6 | 52.9 | 41.2 | 76.5 |
| E static object | 25 | 25 | 25 | 29.9 | 72 | 32 | 26 | 78 | 54 | 88 |

### TTC-only failures by decision-153 fine class

| fine | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| VRU | 7 | 9 | 8 | 9.6 | 68.8 | 0.0 | 37.5 | 68.8 | 62.5 | 81.2 |
| crossing / turn conflict | 5 | 5 | 5 | 6.0 | 70 | 0.0 | 10 | 80 | 80 | 80 |
| cut-in | 10 | 9 | 9.5 | 11.4 | 26.3 | 10.5 | 0.0 | 57.9 | 57.9 | 57.9 |
| lead vehicle | 3 | 3 | 3 | 3.6 | 100 | 16.7 | 66.7 | 100 | 100 | 100 |
| oncoming | 9 | 8 | 8.5 | 10.2 | 17.6 | 0.0 | 17.6 | 52.9 | 41.2 | 76.5 |
| static object | 25 | 25 | 25 | 29.9 | 72 | 32 | 26 | 78 | 54 | 88 |
| stopped vehicle ahead | 27 | 22 | 24.5 | 29.3 | 75.5 | 20.4 | 36.7 | 79.6 | 75.5 | 93.9 |

### TTC-only failures: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 3 | 5.5 | 8 | 1.5 | 18 |
| A2 lead vehicle (moving) | 1 | 1 | 0.0 | 1 | 3 |
| B cut-in | 3 | 2.5 | 3 | 1 | 9.5 |
| C crossing / turn conflict | 0.0 | 1.5 | 1 | 2.5 | 5 |
| D side contact | 0.5 | 2.5 | 2.5 | 1 | 6.5 |
| E static object | 1 | 8.5 | 9.5 | 6 | 25 |
| E VRU | 2.5 | 0.0 | 4 | 1.5 | 8 |
| E oncoming | 2 | 5.5 | 1 | 0.0 | 8.5 |
| all | 13 | 27 | 29 | 14.5 | 83.5 |

### TTC-only failures: class x t0 ego speed band, m/s (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 0.0 | 9 | 9 | 0.0 | 18 |
| A2 lead vehicle (moving) | 0.0 | 0.0 | 3 | 0.0 | 3 |
| B cut-in | 3 | 5 | 0.5 | 1 | 9.5 |
| C crossing / turn conflict | 0.5 | 0.0 | 4.5 | 0.0 | 5 |
| D side contact | 0.5 | 3.5 | 2.5 | 0.0 | 6.5 |
| E static object | 2 | 13.5 | 9.5 | 0.0 | 25 |
| E VRU | 2 | 6 | 0.0 | 0.0 | 8 |
| E oncoming | 0.0 | 2.5 | 6 | 0.0 | 8.5 |
| all | 8 | 39.5 | 35 | 1 | 83.5 |

### TTC-only failures by turn bucket

| turn | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of bucket |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <5 | 13 | 13 | 13 | 53.85 | 19.23 | 42.31 | 73.08 | 69.23 | 73.08 | 0.20 |
| 5-20 | 28 | 26 | 27 | 57.41 | 18.52 | 24.07 | 64.81 | 64.81 | 83.33 | 1.04 |
| 20-45 | 30 | 28 | 29 | 74.14 | 10.34 | 27.59 | 75.86 | 53.45 | 89.66 | 1.77 |
| >45 | 15 | 14 | 14.50 | 58.62 | 27.59 | 17.24 | 86.21 | 75.86 | 86.21 | 0.96 |

### TTC-only failures by t0 ego speed band

| speed | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of band |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <2 | 9 | 7 | 8 | 31.25 | 37.50 | 18.75 | 43.75 | 37.50 | 43.75 | 0.32 |
| 2-5 | 39 | 40 | 39.50 | 63.29 | 25.32 | 37.97 | 77.22 | 62.03 | 89.87 | 1.14 |
| 5-10 | 37 | 33 | 35 | 71.43 | 4.29 | 17.14 | 75.71 | 70 | 87.14 | 0.68 |
| >=10 | 1 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100 | 100 | 100 | 0.10 |

### NC failures: class x turn x speed cells with seed-mean n >= 3

| cls | turn | speed | n_s0 | n_s1 | n | WA-JEPA passes % | recovered, main % |
|:--|--:|--:|--:|--:|--:|--:|--:|
| A2 lead vehicle (moving) | <5 | 5-10 | 18 | 17 | 17.5 | 77.1 | 100 |
| A1 stopped vehicle ahead | >45 | 2-5 | 13 | 14 | 13.5 | 63.0 | 74.1 |
| E static object | <5 | 5-10 | 15 | 10 | 12.5 | 100 | 64 |
| A1 stopped vehicle ahead | <5 | <2 | 12 | 11 | 11.5 | 100 | 43.5 |
| A2 lead vehicle (moving) | <5 | 2-5 | 9 | 9 | 9 | 77.8 | 88.9 |
| A1 stopped vehicle ahead | 20-45 | 2-5 | 7 | 8 | 7.5 | 100 | 86.7 |
| A2 lead vehicle (moving) | <5 | <2 | 8 | 6 | 7 | 71.4 | 78.6 |
| B cut-in | <5 | 2-5 | 7 | 7 | 7 | 42.9 | 92.9 |
| B cut-in | 5-20 | 2-5 | 7 | 6 | 6.5 | 84.6 | 84.6 |
| A1 stopped vehicle ahead | <5 | 5-10 | 6 | 6 | 6 | 100 | 66.7 |
| A1 stopped vehicle ahead | 5-20 | 2-5 | 5 | 5 | 5 | 100 | 50 |
| C crossing / turn conflict | <5 | 2-5 | 5 | 5 | 5 | 60 | 50 |
| A1 stopped vehicle ahead | 5-20 | <2 | 5 | 4 | 4.5 | 55.6 | 66.7 |
| E oncoming | <5 | 2-5 | 4 | 3 | 3.5 | 42.9 | 14.3 |
| A1 stopped vehicle ahead | <5 | 2-5 | 3 | 4 | 3.5 | 28.6 | 57.1 |
| D side contact | <5 | 5-10 | 4 | 3 | 3.5 | 100 | 42.9 |
| A1 stopped vehicle ahead | 20-45 | 5-10 | 3 | 3 | 3 | 66.7 | 66.7 |
| C crossing / turn conflict | 5-20 | 2-5 | 3 | 3 | 3 | 33.3 | 66.7 |
| D side contact | <5 | >=10 | 3 | 3 | 3 | 100 | 66.7 |

### WA-JEPA's own failures by turn bucket (stored per-token scores)

| turn | WA-JEPA NC failures | WA-JEPA TTC-only failures |
|:--|--:|--:|
| <5 | 50 | 11 |
| 5-20 | 15 | 23 |
| 20-45 | 2 | 22 |
| >45 | 8 | 11 |
| all | 75 | 67 |

## navtest: same-path longitudinal scaling oracle (non-reactive, no-EC EPDMS)

### Recovery rate by family

| set | family | n | recovered % [95% CI] | clean recovered % [95% CI] |
|:--|--:|--:|--:|--:|
| NC | 0.9-1.1 | 182 | 35 [29, 42] | 31 [25, 38] |
| NC | 0.8-1.1 | 182 | 60 [53, 69] | 56 [48, 64] |
| NC | 0.7-1.1 (main) | 182 | 74 [67, 83] | 70 [62, 79] |
| NC | 0.5-1.1 (extended) | 182 | 95 [92, 98] | 93 [88, 97] |
| NC | slower only (0.7-0.9) | 182 | 73 [65, 81] | 68 [60, 77] |
| NC | faster only (1.1) | 182 | 3 [1, 6] | 3 [1, 6] |
| NC | 0.95-1.05 | 182 | 21 [16, 28] | 19 [14, 26] |
| NC | 0.85-1.15 | 182 | 49 [42, 58] | 45 [38, 53] |
| NC | all scales | 182 | 95 [92, 98] | 93 [88, 97] |
| TTC-only | 0.9-1.1 | 83.5 | 46 [37, 57] | 41 [32, 52] |
| TTC-only | 0.8-1.1 | 83.5 | 63 [53, 73] | 56 [44, 67] |
| TTC-only | 0.7-1.1 (main) | 83.5 | 74 [63, 83] | 63 [51, 76] |
| TTC-only | 0.5-1.1 (extended) | 83.5 | 84 [73, 94] | 74 [61, 85] |
| TTC-only | slower only (0.7-0.9) | 83.5 | 63 [52, 74] | 51 [40, 64] |
| TTC-only | faster only (1.1) | 83.5 | 18 [11, 26] | 15 [8, 23] |
| TTC-only | 0.95-1.05 | 83.5 | 28 [19, 38] | 25 [17, 34] |
| TTC-only | 0.85-1.15 | 83.5 | 63 [54, 73] | 56 [45, 67] |
| TTC-only | all scales | 83.5 | 86 [76, 94] | 75 [65, 86] |

### EP cost on the recovered tokens (a* = best-scoring recovering scale)

| set | tokens | n | EP(a*) - EP(1.0), x100 | no-EC EPDMS(a*) - (1.0), x100 | a*=0.7 | a*=0.8 | a*=0.9 | a*=1.1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| NC | recovered (main) | 135.50 | -3.46 | 86.06 | 38.50 | 42.50 | 50 | 4.50 |
| NC | clean recovered (main) | 127.50 | -3.46 | 91.45 | 36 | 41.50 | 46 | 4 |
| TTC-only | recovered (main) | 61.50 | -3.59 | 33.33 | 9 | 14.50 | 25.50 | 12.50 |
| TTC-only | clean recovered (main) | 53 | -3.16 | 39.26 | 7 | 12 | 22 | 12 |

### NC recovery by sub-class and family (%)

| cls | n_s0 | n_s1 | n | rec 0.9-1.1 | rec 0.8-1.1 | rec 0.7-1.1 (main) | rec 0.5-1.1 (extended) | rec slower only (0.7-0.9) | rec faster only (1.1) | rec 0.95-1.05 | rec 0.85-1.15 | rec all scales | clean 0.7-1.1 (main) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 58 | 60 | 59 | 29.7 | 50.8 | 66.9 | 93.2 | 66.9 | 0.0 | 15.3 | 38.1 | 93.2 | 59.3 |
| A2 lead vehicle (moving) | 36 | 33 | 34.5 | 43.5 | 87.0 | 92.8 | 100 | 92.8 | 0.0 | 31.9 | 72.5 | 100 | 92.8 |
| B cut-in | 21 | 21 | 21 | 45.2 | 69.0 | 90.5 | 100 | 90.5 | 0.0 | 23.8 | 54.8 | 100 | 85.7 |
| C crossing / turn conflict | 15 | 14 | 14.5 | 31.0 | 62.1 | 75.9 | 93.1 | 72.4 | 3.4 | 17.2 | 48.3 | 93.1 | 72.4 |
| D side contact | 19 | 17 | 18 | 44.4 | 63.9 | 72.2 | 88.9 | 61.1 | 16.7 | 33.3 | 58.3 | 88.9 | 72.2 |
| E VRU | 7 | 7 | 7 | 14.3 | 21.4 | 42.9 | 85.7 | 42.9 | 0.0 | 7.1 | 14.3 | 85.7 | 28.6 |
| E oncoming | 9 | 8 | 8.5 | 17.6 | 35.3 | 58.8 | 100 | 58.8 | 0.0 | 0.0 | 29.4 | 100 | 58.8 |
| E static object | 23 | 16 | 19.5 | 33.3 | 51.3 | 66.7 | 94.9 | 61.5 | 12.8 | 20.5 | 46.2 | 94.9 | 61.5 |

### TTC-only recovery by sub-class and family (%)

| cls | n_s0 | n_s1 | n | rec 0.9-1.1 | rec 0.8-1.1 | rec 0.7-1.1 (main) | rec 0.5-1.1 (extended) | rec slower only (0.7-0.9) | rec faster only (1.1) | rec 0.95-1.05 | rec 0.85-1.15 | rec all scales | clean 0.7-1.1 (main) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 20 | 16 | 18 | 47.2 | 58.3 | 72.2 | 91.7 | 61.1 | 16.7 | 38.9 | 58.3 | 91.7 | 66.7 |
| A2 lead vehicle (moving) | 3 | 3 | 3 | 100 | 100 | 100 | 100 | 66.7 | 33.3 | 83.3 | 100 | 100 | 100 |
| B cut-in | 10 | 9 | 9.5 | 47.4 | 52.6 | 57.9 | 57.9 | 26.3 | 36.8 | 10.5 | 68.4 | 68.4 | 57.9 |
| C crossing / turn conflict | 5 | 5 | 5 | 30 | 80 | 80 | 80 | 80 | 10 | 10 | 70 | 80 | 80 |
| D side contact | 7 | 6 | 6.5 | 61.5 | 76.9 | 100 | 100 | 84.6 | 15.4 | 46.2 | 76.9 | 100 | 100 |
| E VRU | 7 | 9 | 8 | 50 | 68.8 | 68.8 | 81.2 | 62.5 | 12.5 | 18.8 | 62.5 | 87.5 | 62.5 |
| E oncoming | 9 | 8 | 8.5 | 23.5 | 41.2 | 52.9 | 76.5 | 47.1 | 5.9 | 11.8 | 41.2 | 76.5 | 41.2 |
| E static object | 25 | 25 | 25 | 44 | 66 | 78 | 88 | 76 | 18 | 26 | 64 | 88 | 54 |

### NC recovery by turn bucket (%)

| turn | n_s0 | n_s1 | n | rec 0.7-1.1 (main) | clean 0.7-1.1 (main) | rec 0.5-1.1 (extended) |
|:--|--:|--:|--:|--:|--:|--:|
| <5 | 105 | 95 | 100 | 70.5 | 68.5 | 92 |
| 5-20 | 36 | 33 | 34.5 | 71.0 | 65.2 | 97.1 |
| 20-45 | 20 | 21 | 20.5 | 82.9 | 63.4 | 100 |
| >45 | 27 | 27 | 27 | 87.0 | 87.0 | 100 |

### Board-level upper bound: a* on the arm's failing tokens only, all other tokens unchanged (points of the 12 146-token mean)

| family | NC (s0 / s1) | TTC (s0 / s1) | EP (s0 / s1) | no-EC EPDMS (s0 / s1) | tokens replaced (seed mean) |
|:--|--:|--:|--:|--:|--:|
| main 0.7-1.1 | +1.000 (+1.004 / +0.996) | +1.284 (+1.317 / +1.251) | -0.050 (-0.053 / -0.048) | +1.131 (+1.153 / +1.110) | 180.5 |
| extended 0.5-1.1 | +1.319 (+1.354 / +1.284) | +1.737 (+1.778 / +1.696) | -0.071 (-0.071 / -0.072) | +1.492 (+1.533 / +1.452) | 232 |

### Uniform scaling of every token (change against a = 1.0 in points; counts are seed means)

| a | dNC | dTTC | dDAC | dEP | dno-EC EPDMS | NC fail | new NC fail | fixed NC fail | TTC-only fail |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 0.31 | -2.63 | 0.48 | -23.69 | -9.27 | 138.50 | 122.50 | 166 | 490.50 |
| 0.60 | 0.55 | -1.20 | 0.79 | -18.26 | -5.80 | 108.50 | 80.50 | 154 | 318.50 |
| 0.70 | 0.72 | 0.14 | 0.94 | -13.16 | -3.14 | 91 | 39 | 130 | 165.50 |
| 0.80 | 0.72 | 0.68 | 0.95 | -8.39 | -1.18 | 91.50 | 13 | 103.50 | 87.50 |
| 0.85 | 0.59 | 0.65 | 0.87 | -6.13 | -0.56 | 109 | 7.50 | 80.50 | 72.50 |
| 0.90 | 0.42 | 0.54 | 0.65 | -3.97 | -0.23 | 129.50 | 5 | 57.50 | 65 |
| 0.95 | 0.24 | 0.22 | 0.35 | -1.92 | -0.06 | 151.50 | 2.50 | 33 | 84 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 182 | 0.00 | 0.00 | 83.50 |
| 1.05 | -0.27 | -0.30 | -0.50 | 1.77 | -0.32 | 215 | 38 | 5 | 90 |
| 1.10 | -0.70 | -0.85 | -0.98 | 3.39 | -0.94 | 267.50 | 91.50 | 6 | 111 |
| 1.15 | -1.36 | -1.53 | -1.65 | 4.85 | -1.97 | 350.50 | 176.50 | 8 | 121 |
