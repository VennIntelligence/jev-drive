## navtest: failure sets

| set | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of 12 146 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NC | 181 | 167 | 174 | 77.0 | 41.4 | 58.9 | 72.1 | 67.2 | 94.3 | 1.4 |
| TTC-only | 90 | 92 | 91 | 67.0 | 20.9 | 30.2 | 72.5 | 64.3 | 83.0 | 0.7 |

### NC failures by class

| big | n_s0 | n_s1 | n | share % | share % [95% CI] | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | WA-JEPA passes % [95% CI] | arm-specific (fails, WA passes): n | arm-specific share of set % [95% CI] | recovered, main % [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A stopped / slow vehicle ahead | 96 | 85 | 90.5 | 52.0 | 52 [42, 62] | 77.9 | 51.4 | 71.3 | 75.1 | 69.6 | 95.0 | 78 [68, 89] | 70.5 | 41 [32, 51] | 75 [61, 88] |
| B cut-in | 21 | 21 | 21 | 12.1 | 12 [6, 20] | 59.5 | 76.2 | 83.3 | 88.1 | 83.3 | 100 | 60 [34, 86] | 12.5 | 7 [3, 13] | 88 [75, 100] |
| C crossing at junction | 15 | 14 | 14.5 | 8.3 | 8 [0, 17] | 65.5 | 34.5 | 72.4 | 72.4 | 72.4 | 93.1 | 66 [38, 100] | 9.5 | 5 [0, 11] | 72 [38, 100] |
| D side contact | 16 | 12 | 14 | 8.0 | 8 [4, 14] | 100 | 0.0 | 0.0 | 67.9 | 67.9 | 89.3 | 100 [100, 100] | 14 | 8 [4, 14] | 68 [52, 86] |
| E other | 33 | 35 | 34 | 19.5 | 20 [13, 27] | 80.9 | 13.2 | 29.4 | 55.9 | 48.5 | 91.2 | 81 [65, 97] | 27.5 | 16 [11, 22] | 56 [38, 74] |

### NC failures by sub-class

| cls | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 60 | 56 | 58 | 33.3 | 81.0 | 49.1 | 56.9 | 65.5 | 56.9 | 92.2 |
| A2 lead vehicle (moving) | 36 | 29 | 32.5 | 18.7 | 72.3 | 55.4 | 96.9 | 92.3 | 92.3 | 100 |
| B cut-in | 21 | 21 | 21 | 12.1 | 59.5 | 76.2 | 83.3 | 88.1 | 83.3 | 100 |
| C crossing / turn conflict | 15 | 14 | 14.5 | 8.3 | 65.5 | 34.5 | 72.4 | 72.4 | 72.4 | 93.1 |
| D side contact | 16 | 12 | 14 | 8.0 | 100 | 0.0 | 0.0 | 67.9 | 67.9 | 89.3 |
| E VRU | 6 | 7 | 6.5 | 3.7 | 30.8 | 38.5 | 38.5 | 46.2 | 30.8 | 84.6 |
| E oncoming | 9 | 9 | 9 | 5.2 | 77.8 | 11.1 | 55.6 | 33.3 | 33.3 | 100 |
| E static object | 18 | 19 | 18.5 | 10.6 | 100 | 5.4 | 13.5 | 70.3 | 62.2 | 89.2 |

### NC failures by decision-153 fine class

| fine | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| VRU | 6 | 7 | 6.5 | 3.7 | 30.8 | 38.5 | 38.5 | 46.2 | 30.8 | 84.6 |
| crossing / turn conflict | 15 | 14 | 14.5 | 8.3 | 65.5 | 34.5 | 72.4 | 72.4 | 72.4 | 93.1 |
| cut-in | 21 | 21 | 21 | 12.1 | 59.5 | 76.2 | 83.3 | 88.1 | 83.3 | 100 |
| lead vehicle | 36 | 29 | 32.5 | 18.7 | 72.3 | 55.4 | 96.9 | 92.3 | 92.3 | 100 |
| oncoming | 9 | 9 | 9 | 5.2 | 77.8 | 11.1 | 55.6 | 33.3 | 33.3 | 100 |
| sideswipe (ego across lanes) | 4 | 4 | 4 | 2.3 | 100 | 0.0 | 0.0 | 75 | 75 | 75 |
| static object | 18 | 19 | 18.5 | 10.6 | 100 | 5.4 | 13.5 | 70.3 | 62.2 | 89.2 |
| stopped vehicle ahead | 72 | 64 | 68 | 39.1 | 83.8 | 41.9 | 48.5 | 65.4 | 58.1 | 92.6 |

### NC failures: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 20.5 | 10 | 11.5 | 16 | 58 |
| A2 lead vehicle (moving) | 31.5 | 1 | 0.0 | 0.0 | 32.5 |
| B cut-in | 11.5 | 8.5 | 1 | 0.0 | 21 |
| C crossing / turn conflict | 7.5 | 3 | 2 | 2 | 14.5 |
| D side contact | 7.5 | 3 | 1 | 2.5 | 14 |
| E static object | 9.5 | 2 | 3 | 4 | 18.5 |
| E VRU | 3.5 | 3 | 0.0 | 0.0 | 6.5 |
| E oncoming | 4 | 1 | 2 | 2 | 9 |
| all | 95.5 | 31.5 | 20.5 | 26.5 | 174 |

### NC failures: class x t0 ego speed band, m/s (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 18 | 29.5 | 10.5 | 0.0 | 58 |
| A2 lead vehicle (moving) | 7 | 7.5 | 18 | 0.0 | 32.5 |
| B cut-in | 2 | 14.5 | 4.5 | 0.0 | 21 |
| C crossing / turn conflict | 3 | 8.5 | 3 | 0.0 | 14.5 |
| D side contact | 1 | 2.5 | 7 | 3.5 | 14 |
| E static object | 0.5 | 7 | 11 | 0.0 | 18.5 |
| E VRU | 1.5 | 2 | 3 | 0.0 | 6.5 |
| E oncoming | 1 | 5 | 3 | 0.0 | 9 |
| all | 34 | 76.5 | 60 | 3.5 | 174 |

### NC failures by turn bucket

| turn | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of bucket |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <5 | 100 | 91 | 95.50 | 75.92 | 54.97 | 69.63 | 69.63 | 67.54 | 90.58 | 1.49 |
| 5-20 | 32 | 31 | 31.50 | 65.08 | 50.79 | 66.67 | 68.25 | 61.90 | 96.83 | 1.22 |
| 20-45 | 22 | 19 | 20.50 | 95.12 | 4.88 | 39.02 | 75.61 | 56.10 | 100 | 1.25 |
| >45 | 27 | 26 | 26.50 | 81.13 | 9.43 | 26.42 | 83.02 | 81.13 | 100 | 1.75 |

### NC failures by t0 ego speed band

| speed | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of band |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <2 | 36 | 32 | 34 | 83.82 | 82.35 | 89.71 | 70.59 | 63.24 | 97.06 | 1.34 |
| 2-5 | 79 | 74 | 76.50 | 71.24 | 40.52 | 64.05 | 67.97 | 62.75 | 95.42 | 2.21 |
| 5-10 | 62 | 58 | 60 | 79.17 | 21.67 | 38.33 | 78.33 | 75 | 92.50 | 1.16 |
| >=10 | 4 | 3 | 3.50 | 100 | 0.00 | 0.00 | 71.43 | 71.43 | 71.43 | 0.36 |

### TTC-only failures by class

| big | n_s0 | n_s1 | n | share % | share % [95% CI] | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | WA-JEPA passes % [95% CI] | arm-specific (fails, WA passes): n | arm-specific share of set % [95% CI] | recovered, main % [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A stopped / slow vehicle ahead | 30 | 29 | 29.5 | 32.4 | 32 [22, 45] | 79.7 | 35.6 | 54.2 | 83.1 | 78.0 | 93.2 | 80 [62, 94] | 23.5 | 26 [16, 37] | 83 [69, 94] |
| B cut-in | 11 | 8 | 9.5 | 10.4 | 10 [2, 21] | 31.6 | 10.5 | 5.3 | 52.6 | 52.6 | 52.6 | 32 [11, 86] | 3 | 3 [1, 6] | 53 [11, 100] |
| C crossing at junction | 4 | 5 | 4.5 | 4.9 | 5 [0, 12] | 77.8 | 0.0 | 0.0 | 77.8 | 77.8 | 77.8 | 78 [0, 100] | 3.5 | 4 [0, 11] | 78 [67, 100] |
| D side contact | 7 | 8 | 7.5 | 8.2 | 8 [2, 16] | 100 | 0.0 | 0.0 | 100 | 100 | 100 | 100 [100, 100] | 7.5 | 8 [2, 16] | 100 [100, 100] |
| E other | 38 | 42 | 40 | 44.0 | 44 [28, 60] | 58.8 | 18.8 | 27.5 | 63.7 | 48.8 | 80 | 59 [43, 73] | 23.5 | 26 [15, 37] | 64 [51, 75] |

### TTC-only failures by sub-class

| cls | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 27 | 26 | 26.5 | 29.1 | 77.4 | 35.8 | 52.8 | 81.1 | 75.5 | 92.5 |
| A2 lead vehicle (moving) | 3 | 3 | 3 | 3.3 | 100 | 33.3 | 66.7 | 100 | 100 | 100 |
| B cut-in | 11 | 8 | 9.5 | 10.4 | 31.6 | 10.5 | 5.3 | 52.6 | 52.6 | 52.6 |
| C crossing / turn conflict | 4 | 5 | 4.5 | 4.9 | 77.8 | 0.0 | 0.0 | 77.8 | 77.8 | 77.8 |
| D side contact | 7 | 8 | 7.5 | 8.2 | 100 | 0.0 | 0.0 | 100 | 100 | 100 |
| E VRU | 8 | 10 | 9 | 9.9 | 77.8 | 5.6 | 33.3 | 72.2 | 66.7 | 83.3 |
| E oncoming | 8 | 9 | 8.5 | 9.3 | 11.8 | 0.0 | 17.6 | 52.9 | 41.2 | 76.5 |
| E static object | 22 | 23 | 22.5 | 24.7 | 68.9 | 31.1 | 28.9 | 64.4 | 44.4 | 80 |

### TTC-only failures by decision-153 fine class

| fine | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| VRU | 8 | 10 | 9 | 9.9 | 77.8 | 5.6 | 33.3 | 72.2 | 66.7 | 83.3 |
| crossing / turn conflict | 4 | 5 | 4.5 | 4.9 | 77.8 | 0.0 | 0.0 | 77.8 | 77.8 | 77.8 |
| cut-in | 11 | 8 | 9.5 | 10.4 | 31.6 | 10.5 | 5.3 | 52.6 | 52.6 | 52.6 |
| lead vehicle | 3 | 3 | 3 | 3.3 | 100 | 33.3 | 66.7 | 100 | 100 | 100 |
| oncoming | 8 | 9 | 8.5 | 9.3 | 11.8 | 0.0 | 17.6 | 52.9 | 41.2 | 76.5 |
| static object | 22 | 23 | 22.5 | 24.7 | 68.9 | 31.1 | 28.9 | 64.4 | 44.4 | 80 |
| stopped vehicle ahead | 34 | 34 | 34 | 37.4 | 82.4 | 27.9 | 41.2 | 85.3 | 80.9 | 94.1 |

### TTC-only failures: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 8 | 7 | 9 | 2.5 | 26.5 |
| A2 lead vehicle (moving) | 1 | 1 | 0.0 | 1 | 3 |
| B cut-in | 3.5 | 2.5 | 2.5 | 1 | 9.5 |
| C crossing / turn conflict | 0.0 | 1.5 | 1 | 2 | 4.5 |
| D side contact | 0.0 | 3 | 3.5 | 1 | 7.5 |
| E static object | 2 | 7 | 8 | 5.5 | 22.5 |
| E VRU | 2.5 | 0.0 | 5.5 | 1 | 9 |
| E oncoming | 0.5 | 5 | 3 | 0.0 | 8.5 |
| all | 17.5 | 27 | 32.5 | 14 | 91 |

### TTC-only failures: class x t0 ego speed band, m/s (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 0.0 | 10.5 | 16 | 0.0 | 26.5 |
| A2 lead vehicle (moving) | 0.0 | 0.0 | 3 | 0.0 | 3 |
| B cut-in | 3 | 4.5 | 1.5 | 0.5 | 9.5 |
| C crossing / turn conflict | 0.0 | 0.0 | 4.5 | 0.0 | 4.5 |
| D side contact | 0.0 | 3.5 | 4 | 0.0 | 7.5 |
| E static object | 1.5 | 12.5 | 8.5 | 0.0 | 22.5 |
| E VRU | 2.5 | 5.5 | 1 | 0.0 | 9 |
| E oncoming | 0.0 | 2.5 | 6 | 0.0 | 8.5 |
| all | 7 | 39 | 44.5 | 0.5 | 91 |

### TTC-only failures by turn bucket

| turn | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of bucket |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <5 | 16 | 19 | 17.50 | 71.43 | 45.71 | 60 | 71.43 | 68.57 | 77.14 | 0.27 |
| 5-20 | 28 | 26 | 27 | 59.26 | 16.67 | 24.07 | 59.26 | 59.26 | 79.63 | 1.04 |
| 20-45 | 32 | 33 | 32.50 | 70.77 | 9.23 | 26.15 | 81.54 | 63.08 | 90.77 | 1.99 |
| >45 | 14 | 14 | 14 | 67.86 | 25 | 14.29 | 78.57 | 71.43 | 78.57 | 0.92 |

### TTC-only failures by t0 ego speed band

| speed | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of band |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <2 | 6 | 8 | 7 | 35.71 | 42.86 | 14.29 | 35.71 | 28.57 | 35.71 | 0.28 |
| 2-5 | 39 | 39 | 39 | 65.38 | 26.92 | 38.46 | 76.92 | 66.67 | 87.18 | 1.13 |
| 5-10 | 45 | 44 | 44.50 | 74.16 | 12.36 | 25.84 | 75.28 | 68.54 | 87.64 | 0.86 |
| >=10 | 0 | 1 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |

### NC failures: class x turn x speed cells with seed-mean n >= 3

| cls | turn | speed | n_s0 | n_s1 | n | WA-JEPA passes % | recovered, main % |
|:--|--:|--:|--:|--:|--:|--:|--:|
| A2 lead vehicle (moving) | <5 | 5-10 | 19 | 15 | 17 | 76.5 | 100 |
| A1 stopped vehicle ahead | >45 | 2-5 | 13 | 12 | 12.5 | 60 | 72 |
| A1 stopped vehicle ahead | <5 | <2 | 12 | 10 | 11 | 100 | 40.9 |
| E static object | <5 | 5-10 | 10 | 9 | 9.5 | 100 | 57.9 |
| A1 stopped vehicle ahead | 20-45 | 2-5 | 9 | 8 | 8.5 | 100 | 76.5 |
| B cut-in | <5 | 2-5 | 8 | 8 | 8 | 50 | 87.5 |
| A2 lead vehicle (moving) | <5 | 2-5 | 8 | 7 | 7.5 | 73.3 | 86.7 |
| A2 lead vehicle (moving) | <5 | <2 | 8 | 6 | 7 | 71.4 | 78.6 |
| B cut-in | 5-20 | 2-5 | 6 | 5 | 5.5 | 81.8 | 81.8 |
| A1 stopped vehicle ahead | <5 | 5-10 | 5 | 6 | 5.5 | 100 | 72.7 |
| A1 stopped vehicle ahead | 5-20 | <2 | 5 | 5 | 5 | 60 | 80 |
| A1 stopped vehicle ahead | 5-20 | 2-5 | 5 | 4 | 4.5 | 100 | 44.4 |
| C crossing / turn conflict | <5 | 2-5 | 5 | 4 | 4.5 | 55.6 | 55.6 |
| A1 stopped vehicle ahead | <5 | 2-5 | 4 | 4 | 4 | 25 | 50 |
| E oncoming | <5 | 2-5 | 4 | 4 | 4 | 50 | 0.0 |
| D side contact | <5 | >=10 | 4 | 3 | 3.5 | 100 | 71.4 |
| A1 stopped vehicle ahead | 20-45 | 5-10 | 3 | 3 | 3 | 66.7 | 66.7 |
| C crossing / turn conflict | 5-20 | 2-5 | 3 | 3 | 3 | 33.3 | 33.3 |
| D side contact | <5 | 5-10 | 3 | 3 | 3 | 100 | 33.3 |
| E static object | 20-45 | 2-5 | 3 | 3 | 3 | 100 | 66.7 |

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
| NC | 0.9-1.1 | 174 | 33 [26, 40] | 31 [24, 37] |
| NC | 0.8-1.1 | 174 | 58 [50, 67] | 53 [45, 62] |
| NC | 0.7-1.1 (main) | 174 | 72 [64, 82] | 67 [59, 76] |
| NC | 0.5-1.1 (extended) | 174 | 94 [92, 97] | 92 [88, 96] |
| NC | slower only (0.7-0.9) | 174 | 70 [62, 80] | 65 [57, 74] |
| NC | faster only (1.1) | 174 | 3 [1, 6] | 3 [1, 6] |
| NC | 0.95-1.05 | 174 | 18 [13, 24] | 17 [12, 24] |
| NC | 0.85-1.15 | 174 | 49 [42, 57] | 45 [38, 53] |
| NC | all scales | 174 | 94 [92, 97] | 92 [88, 96] |
| TTC-only | 0.9-1.1 | 91 | 51 [41, 60] | 46 [36, 56] |
| TTC-only | 0.8-1.1 | 91 | 63 [53, 73] | 56 [44, 67] |
| TTC-only | 0.7-1.1 (main) | 91 | 73 [62, 83] | 64 [52, 76] |
| TTC-only | 0.5-1.1 (extended) | 91 | 83 [72, 92] | 74 [61, 85] |
| TTC-only | slower only (0.7-0.9) | 91 | 66 [55, 77] | 57 [45, 68] |
| TTC-only | faster only (1.1) | 91 | 13 [7, 20] | 12 [6, 19] |
| TTC-only | 0.95-1.05 | 91 | 33 [23, 42] | 31 [21, 41] |
| TTC-only | 0.85-1.15 | 91 | 62 [53, 71] | 54 [44, 64] |
| TTC-only | all scales | 91 | 86 [77, 93] | 76 [66, 87] |

### EP cost on the recovered tokens (a* = best-scoring recovering scale)

| set | tokens | n | EP(a*) - EP(1.0), x100 | no-EC EPDMS(a*) - (1.0), x100 | a*=0.7 | a*=0.8 | a*=0.9 | a*=1.1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| NC | recovered (main) | 125.50 | -3.26 | 86.44 | 33 | 44.50 | 44 | 4 |
| NC | clean recovered (main) | 117 | -3.22 | 92.72 | 30 | 41 | 42 | 4 |
| TTC-only | recovered (main) | 66 | -3.36 | 33.14 | 9 | 12.50 | 34 | 10.50 |
| TTC-only | clean recovered (main) | 58.50 | -3.05 | 37.39 | 8 | 9.50 | 31 | 10 |

### NC recovery by sub-class and family (%)

| cls | n_s0 | n_s1 | n | rec 0.9-1.1 | rec 0.8-1.1 | rec 0.7-1.1 (main) | rec 0.5-1.1 (extended) | rec slower only (0.7-0.9) | rec faster only (1.1) | rec 0.95-1.05 | rec 0.85-1.15 | rec all scales | clean 0.7-1.1 (main) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 60 | 56 | 58 | 23.3 | 49.1 | 65.5 | 92.2 | 65.5 | 0.9 | 15.5 | 37.1 | 92.2 | 56.9 |
| A2 lead vehicle (moving) | 36 | 29 | 32.5 | 40 | 83.1 | 92.3 | 100 | 92.3 | 0.0 | 21.5 | 70.8 | 100 | 92.3 |
| B cut-in | 21 | 21 | 21 | 47.6 | 69.0 | 88.1 | 100 | 88.1 | 0.0 | 23.8 | 57.1 | 100 | 83.3 |
| C crossing / turn conflict | 15 | 14 | 14.5 | 41.4 | 62.1 | 72.4 | 93.1 | 69.0 | 3.4 | 17.2 | 58.6 | 93.1 | 72.4 |
| D side contact | 16 | 12 | 14 | 39.3 | 57.1 | 67.9 | 89.3 | 46.4 | 25 | 28.6 | 50 | 89.3 | 67.9 |
| E VRU | 6 | 7 | 6.5 | 15.4 | 23.1 | 46.2 | 84.6 | 46.2 | 7.7 | 7.7 | 23.1 | 84.6 | 30.8 |
| E oncoming | 9 | 9 | 9 | 22.2 | 33.3 | 33.3 | 100 | 33.3 | 0.0 | 5.6 | 33.3 | 100 | 33.3 |
| E static object | 18 | 19 | 18.5 | 32.4 | 51.4 | 70.3 | 89.2 | 70.3 | 5.4 | 13.5 | 45.9 | 89.2 | 62.2 |

### TTC-only recovery by sub-class and family (%)

| cls | n_s0 | n_s1 | n | rec 0.9-1.1 | rec 0.8-1.1 | rec 0.7-1.1 (main) | rec 0.5-1.1 (extended) | rec slower only (0.7-0.9) | rec faster only (1.1) | rec 0.95-1.05 | rec 0.85-1.15 | rec all scales | clean 0.7-1.1 (main) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 27 | 26 | 26.5 | 69.8 | 71.7 | 81.1 | 92.5 | 75.5 | 9.4 | 47.2 | 73.6 | 94.3 | 75.5 |
| A2 lead vehicle (moving) | 3 | 3 | 3 | 83.3 | 100 | 100 | 100 | 66.7 | 33.3 | 83.3 | 100 | 100 | 100 |
| B cut-in | 11 | 8 | 9.5 | 42.1 | 47.4 | 52.6 | 52.6 | 31.6 | 31.6 | 26.3 | 63.2 | 68.4 | 52.6 |
| C crossing / turn conflict | 4 | 5 | 4.5 | 33.3 | 66.7 | 77.8 | 77.8 | 77.8 | 11.1 | 22.2 | 44.4 | 77.8 | 77.8 |
| D side contact | 7 | 8 | 7.5 | 46.7 | 86.7 | 100 | 100 | 93.3 | 6.7 | 20 | 66.7 | 100 | 100 |
| E VRU | 8 | 10 | 9 | 61.1 | 72.2 | 72.2 | 83.3 | 66.7 | 11.1 | 16.7 | 66.7 | 88.9 | 66.7 |
| E oncoming | 8 | 9 | 8.5 | 29.4 | 41.2 | 52.9 | 76.5 | 52.9 | 11.8 | 29.4 | 41.2 | 76.5 | 41.2 |
| E static object | 22 | 23 | 22.5 | 35.6 | 51.1 | 64.4 | 80 | 62.2 | 11.1 | 26.7 | 48.9 | 80 | 44.4 |

### NC recovery by turn bucket (%)

| turn | n_s0 | n_s1 | n | rec 0.7-1.1 (main) | clean 0.7-1.1 (main) | rec 0.5-1.1 (extended) |
|:--|--:|--:|--:|--:|--:|--:|
| <5 | 100 | 91 | 95.5 | 69.6 | 67.5 | 90.6 |
| 5-20 | 32 | 31 | 31.5 | 68.3 | 61.9 | 96.8 |
| 20-45 | 22 | 19 | 20.5 | 75.6 | 56.1 | 100 |
| >45 | 27 | 26 | 26.5 | 83.0 | 81.1 | 100 |

### Board-level upper bound: a* on the arm's failing tokens only, all other tokens unchanged (points of the 12 146-token mean)

| family | NC (s0 / s1) | TTC (s0 / s1) | EP (s0 / s1) | no-EC EPDMS (s0 / s1) | tokens replaced (seed mean) |
|:--|--:|--:|--:|--:|--:|
| main 0.7-1.1 | +0.918 (+0.984 / +0.852) | +1.247 (+1.342 / +1.153) | -0.046 (-0.051 / -0.041) | +1.073 (+1.142 / +1.005) | 175.5 |
| extended 0.5-1.1 | +1.262 (+1.326 / +1.198) | +1.700 (+1.754 / +1.647) | -0.074 (-0.071 / -0.077) | +1.443 (+1.496 / +1.391) | 230 |

### Uniform scaling of every token (change against a = 1.0 in points; counts are seed means)

| a | dNC | dTTC | dDAC | dEP | dno-EC EPDMS | NC fail | new NC fail | fixed NC fail | TTC-only fail |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 0.24 | -2.60 | 0.29 | -23.69 | -9.50 | 140 | 124.50 | 158.50 | 485 |
| 0.60 | 0.46 | -1.21 | 0.52 | -18.25 | -6.10 | 114 | 83 | 143 | 315 |
| 0.70 | 0.63 | 0.14 | 0.83 | -13.15 | -3.28 | 93.50 | 40.50 | 121 | 164 |
| 0.80 | 0.64 | 0.68 | 0.84 | -8.37 | -1.33 | 93 | 15.50 | 96.50 | 81 |
| 0.85 | 0.55 | 0.65 | 0.79 | -6.12 | -0.66 | 105.50 | 9.50 | 78 | 72.50 |
| 0.90 | 0.36 | 0.55 | 0.61 | -3.96 | -0.28 | 129 | 6.50 | 51.50 | 65 |
| 0.95 | 0.20 | 0.28 | 0.37 | -1.91 | -0.01 | 149 | 2.50 | 27.50 | 78.50 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 174 | 0.00 | 0.00 | 91 |
| 1.05 | -0.30 | -0.37 | -0.45 | 1.77 | -0.33 | 211 | 41 | 4 | 96.50 |
| 1.10 | -0.67 | -0.82 | -0.96 | 3.38 | -0.91 | 257 | 89 | 6 | 111 |
| 1.15 | -1.32 | -1.54 | -1.58 | 4.83 | -1.96 | 338.50 | 171 | 6.50 | 128.50 |

## navhard stage 1 (450 tokens, G frames): counts only

| set | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % |
|:--|--:|--:|--:|--:|--:|--:|
| NC | 18 | 15 | 16.5 | 69.7 | 21.2 | 24.2 |
| TTC-only | 1 | 3 | 2 | 50 | 50 | 50 |

### navhard stage 1, NC failures by sub-class

| cls | n_s0 | n_s1 | n | WA-JEPA passes % |
|:--|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 8 | 6 | 7 | 57.1 |
| A2 lead vehicle (moving) | 1 | 1 | 1 | 100 |
| B cut-in | 1 | 1 | 1 | 100 |
| C crossing / turn conflict | 1 | 1 | 1 | 100 |
| D side contact | 0 | 1 | 0.5 | 100 |
| E VRU | 2 | 2 | 2 | 0.0 |
| E oncoming | 5 | 3 | 4 | 100 |

### navhard stage 1, NC: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 2 | 0.0 | 1.5 | 3.5 | 7 |
| A2 lead vehicle (moving) | 1 | 0.0 | 0.0 | 0.0 | 1 |
| B cut-in | 0.0 | 1 | 0.0 | 0.0 | 1 |
| C crossing / turn conflict | 0.0 | 0.0 | 0.0 | 1 | 1 |
| D side contact | 0.0 | 0.0 | 0.5 | 0.0 | 0.5 |
| E VRU | 2 | 0.0 | 0.0 | 0.0 | 2 |
| E oncoming | 1.5 | 1 | 1.5 | 0.0 | 4 |
| all | 6.5 | 2 | 3.5 | 4.5 | 16.5 |

### navhard stage 1, NC: class x t0 ego speed band (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 2 | 2.5 | 2.5 | 0.0 | 7 |
| A2 lead vehicle (moving) | 0.0 | 0.0 | 1 | 0.0 | 1 |
| B cut-in | 0.0 | 0.0 | 1 | 0.0 | 1 |
| C crossing / turn conflict | 0.0 | 1 | 0.0 | 0.0 | 1 |
| D side contact | 0.0 | 0.5 | 0.0 | 0.0 | 0.5 |
| E VRU | 0.0 | 0.0 | 2 | 0.0 | 2 |
| E oncoming | 0.0 | 0.0 | 4 | 0.0 | 4 |
| all | 2 | 4 | 10.5 | 0.0 | 16.5 |

### navhard stage 1, TTC-only failures by sub-class

| cls | n_s0 | n_s1 | n | WA-JEPA passes % |
|:--|--:|--:|--:|--:|
| E VRU | 0 | 1 | 0.5 | 100 |
| E oncoming | 1 | 2 | 1.5 | 33.3 |

### navhard stage 1, TTC-only: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| E VRU | 0.0 | 0.5 | 0.0 | 0.0 | 0.5 |
| E oncoming | 1 | 0.0 | 0.5 | 0.0 | 1.5 |
| all | 1 | 0.5 | 0.5 | 0.0 | 2 |

### navhard stage 1, TTC-only: class x t0 ego speed band (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| E VRU | 0.0 | 0.0 | 0.5 | 0.0 | 0.5 |
| E oncoming | 1 | 0.0 | 0.5 | 0.0 | 1.5 |
| all | 1 | 0.0 | 1 | 0.0 | 2 |

WA-JEPA on the same 450 stage-1 tokens: 9 NC failures, 4 TTC-only failures (stored per-token scores).
