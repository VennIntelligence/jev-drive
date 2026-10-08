## navtest: failure sets

| set | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of 12 146 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NC | 180 | 173 | 176.5 | 76.2 | 41.6 | 62.6 | 74.2 | 70.3 | 95.5 | 1.5 |
| TTC-only | 88 | 87 | 87.5 | 62.9 | 17.7 | 27.4 | 73.1 | 62.3 | 82.9 | 0.7 |

### NC failures by class

| big | n_s0 | n_s1 | n | share % | share % [95% CI] | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | WA-JEPA passes % [95% CI] | SH30-specific (fails, WA passes): n | SH30-specific share of set % [95% CI] | recovered, main % [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A stopped / slow vehicle ahead | 95 | 92 | 93.5 | 53.0 | 53 [43, 62] | 78.6 | 50.8 | 73.3 | 77.0 | 73.3 | 95.2 | 79 [69, 90] | 73.5 | 42 [34, 50] | 77 [65, 89] |
| B cut-in | 24 | 22 | 23 | 13.0 | 13 [6, 22] | 60.9 | 73.9 | 93.5 | 91.3 | 87.0 | 100 | 61 [35, 87] | 14 | 8 [3, 14] | 91 [82, 100] |
| C crossing at junction | 14 | 13 | 13.5 | 7.6 | 8 [0, 17] | 59.3 | 44.4 | 74.1 | 59.3 | 55.6 | 92.6 | 59 [47, 80] | 8 | 5 [0, 9] | 59 [35, 100] |
| D side contact | 17 | 16 | 16.5 | 9.3 | 9 [5, 15] | 97.0 | 0.0 | 6.1 | 72.7 | 72.7 | 90.9 | 97 [89, 100] | 16 | 9 [5, 15] | 73 [56, 90] |
| E other | 30 | 30 | 30 | 17.0 | 17 [11, 24] | 76.7 | 10 | 31.7 | 60 | 53.3 | 96.7 | 77 [55, 96] | 23 | 13 [8, 20] | 60 [41, 79] |

### NC failures by sub-class

| cls | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 60 | 57 | 58.5 | 33.1 | 81.2 | 48.7 | 57.3 | 67.5 | 61.5 | 92.3 |
| A2 lead vehicle (moving) | 35 | 35 | 35 | 19.8 | 74.3 | 54.3 | 100 | 92.9 | 92.9 | 100 |
| B cut-in | 24 | 22 | 23 | 13.0 | 60.9 | 73.9 | 93.5 | 91.3 | 87.0 | 100 |
| C crossing / turn conflict | 14 | 13 | 13.5 | 7.6 | 59.3 | 44.4 | 74.1 | 59.3 | 55.6 | 92.6 |
| D side contact | 17 | 16 | 16.5 | 9.3 | 97.0 | 0.0 | 6.1 | 72.7 | 72.7 | 90.9 |
| E VRU | 7 | 7 | 7 | 4.0 | 28.6 | 42.9 | 42.9 | 42.9 | 28.6 | 85.7 |
| E oncoming | 8 | 9 | 8.5 | 4.8 | 76.5 | 0.0 | 52.9 | 58.8 | 58.8 | 100 |
| E static object | 15 | 14 | 14.5 | 8.2 | 100 | 0.0 | 13.8 | 69.0 | 62.1 | 100 |

### NC failures by decision-153 fine class

| fine | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| VRU | 7 | 7 | 7 | 4.0 | 28.6 | 42.9 | 42.9 | 42.9 | 28.6 | 85.7 |
| crossing / turn conflict | 14 | 13 | 13.5 | 7.6 | 59.3 | 44.4 | 74.1 | 59.3 | 55.6 | 92.6 |
| cut-in | 24 | 22 | 23 | 13.0 | 60.9 | 73.9 | 93.5 | 91.3 | 87.0 | 100 |
| lead vehicle | 35 | 35 | 35 | 19.8 | 74.3 | 54.3 | 100 | 92.9 | 92.9 | 100 |
| oncoming | 8 | 9 | 8.5 | 4.8 | 76.5 | 0.0 | 52.9 | 58.8 | 58.8 | 100 |
| sideswipe (ego across lanes) | 5 | 3 | 4 | 2.3 | 87.5 | 0.0 | 12.5 | 75 | 75 | 75 |
| static object | 15 | 14 | 14.5 | 8.2 | 100 | 0.0 | 13.8 | 69.0 | 62.1 | 100 |
| stopped vehicle ahead | 72 | 70 | 71 | 40.2 | 84.5 | 40.1 | 47.9 | 68.3 | 63.4 | 93.0 |

### NC failures: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 23 | 10 | 10.5 | 15 | 58.5 |
| A2 lead vehicle (moving) | 33 | 2 | 0.0 | 0.0 | 35 |
| B cut-in | 11.5 | 10.5 | 1 | 0.0 | 23 |
| C crossing / turn conflict | 8.5 | 2 | 2 | 1 | 13.5 |
| D side contact | 6.5 | 3 | 1 | 6 | 16.5 |
| E static object | 9 | 1 | 2 | 2.5 | 14.5 |
| E VRU | 4 | 3 | 0.0 | 0.0 | 7 |
| E oncoming | 3.5 | 1 | 2 | 2 | 8.5 |
| all | 99 | 32.5 | 18.5 | 26.5 | 176.5 |

### NC failures: class x t0 ego speed band, m/s (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 19 | 28.5 | 11 | 0.0 | 58.5 |
| A2 lead vehicle (moving) | 8.5 | 7.5 | 19 | 0.0 | 35 |
| B cut-in | 2 | 15 | 6 | 0.0 | 23 |
| C crossing / turn conflict | 2 | 8.5 | 3 | 0.0 | 13.5 |
| D side contact | 0.0 | 5 | 8.5 | 3 | 16.5 |
| E static object | 0.0 | 5 | 9.5 | 0.0 | 14.5 |
| E VRU | 2 | 2 | 3 | 0.0 | 7 |
| E oncoming | 1 | 4.5 | 3 | 0.0 | 8.5 |
| all | 34.5 | 76 | 63 | 3 | 176.5 |

### NC failures by turn bucket

| turn | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of bucket |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <5 | 102 | 96 | 99 | 74.75 | 58.08 | 74.75 | 70.20 | 68.18 | 92.93 | 1.55 |
| 5-20 | 33 | 32 | 32.50 | 66.15 | 43.08 | 69.23 | 72.31 | 66.15 | 96.92 | 1.25 |
| 20-45 | 18 | 19 | 18.50 | 94.59 | 0.00 | 43.24 | 78.38 | 62.16 | 100 | 1.13 |
| >45 | 27 | 26 | 26.50 | 81.13 | 7.55 | 22.64 | 88.68 | 88.68 | 100 | 1.75 |

### NC failures by t0 ego speed band

| speed | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of band |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <2 | 37 | 32 | 34.50 | 81.16 | 86.96 | 95.65 | 66.67 | 60.87 | 97.10 | 1.36 |
| 2-5 | 75 | 77 | 76 | 71.05 | 40.79 | 66.45 | 71.05 | 65.79 | 95.39 | 2.20 |
| 5-10 | 65 | 61 | 63 | 78.57 | 19.84 | 42.86 | 82.54 | 80.95 | 96.03 | 1.22 |
| >=10 | 3 | 3 | 3 | 100 | 0.00 | 0.00 | 66.67 | 66.67 | 66.67 | 0.31 |

### TTC-only failures by class

| big | n_s0 | n_s1 | n | share % | share % [95% CI] | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | WA-JEPA passes % [95% CI] | SH30-specific (fails, WA passes): n | SH30-specific share of set % [95% CI] | recovered, main % [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A stopped / slow vehicle ahead | 22 | 25 | 23.5 | 26.9 | 27 [17, 38] | 76.6 | 31.9 | 53.2 | 83.0 | 76.6 | 93.6 | 77 [56, 92] | 18 | 21 [11, 31] | 83 [67, 96] |
| B cut-in | 11 | 10 | 10.5 | 12 | 12 [2, 24] | 28.6 | 9.5 | 0.0 | 52.4 | 52.4 | 52.4 | 29 [8, 78] | 3 | 3 [1, 7] | 52 [9, 100] |
| C crossing at junction | 6 | 7 | 6.5 | 7.4 | 7 [1, 15] | 69.2 | 15.4 | 15.4 | 84.6 | 69.2 | 84.6 | 69 [11, 100] | 4.5 | 5 [0, 12] | 85 [71, 100] |
| D side contact | 4 | 6 | 5 | 5.7 | 6 [1, 13] | 100 | 0.0 | 0.0 | 100 | 100 | 100 | 100 [100, 100] | 5 | 6 [1, 13] | 100 [100, 100] |
| E other | 45 | 39 | 42 | 48 | 48 [33, 63] | 58.3 | 14.3 | 25 | 67.9 | 51.2 | 82.1 | 58 [42, 76] | 24.5 | 28 [18, 39] | 68 [53, 82] |

### TTC-only failures by sub-class

| cls | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 19 | 21 | 20 | 22.9 | 75 | 35 | 50 | 80 | 72.5 | 92.5 |
| A2 lead vehicle (moving) | 3 | 4 | 3.5 | 4 | 85.7 | 14.3 | 71.4 | 100 | 100 | 100 |
| B cut-in | 11 | 10 | 10.5 | 12 | 28.6 | 9.5 | 0.0 | 52.4 | 52.4 | 52.4 |
| C crossing / turn conflict | 6 | 7 | 6.5 | 7.4 | 69.2 | 15.4 | 15.4 | 84.6 | 69.2 | 84.6 |
| D side contact | 4 | 6 | 5 | 5.7 | 100 | 0.0 | 0.0 | 100 | 100 | 100 |
| E VRU | 9 | 10 | 9.5 | 10.9 | 73.7 | 0.0 | 31.6 | 84.2 | 73.7 | 89.5 |
| E oncoming | 13 | 10 | 11.5 | 13.1 | 21.7 | 0.0 | 17.4 | 43.5 | 34.8 | 73.9 |
| E static object | 23 | 19 | 21 | 24 | 71.4 | 28.6 | 26.2 | 73.8 | 50 | 83.3 |

### TTC-only failures by decision-153 fine class

| fine | n_s0 | n_s1 | n | share % | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| VRU | 9 | 10 | 9.5 | 10.9 | 73.7 | 0.0 | 31.6 | 84.2 | 73.7 | 89.5 |
| crossing / turn conflict | 6 | 7 | 6.5 | 7.4 | 69.2 | 15.4 | 15.4 | 84.6 | 69.2 | 84.6 |
| cut-in | 11 | 10 | 10.5 | 12 | 28.6 | 9.5 | 0.0 | 52.4 | 52.4 | 52.4 |
| lead vehicle | 3 | 4 | 3.5 | 4 | 85.7 | 14.3 | 71.4 | 100 | 100 | 100 |
| oncoming | 13 | 10 | 11.5 | 13.1 | 21.7 | 0.0 | 17.4 | 43.5 | 34.8 | 73.9 |
| static object | 23 | 19 | 21 | 24 | 71.4 | 28.6 | 26.2 | 73.8 | 50 | 83.3 |
| stopped vehicle ahead | 23 | 27 | 25 | 28.6 | 80 | 28.0 | 40 | 84 | 78 | 94 |

### TTC-only failures: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 6 | 6.5 | 7 | 0.5 | 20 |
| A2 lead vehicle (moving) | 2.5 | 0.0 | 0.0 | 1 | 3.5 |
| B cut-in | 3.5 | 2.5 | 3 | 1.5 | 10.5 |
| C crossing / turn conflict | 0.0 | 1.5 | 1 | 4 | 6.5 |
| D side contact | 0.0 | 2 | 2 | 1 | 5 |
| E static object | 2.5 | 6.5 | 7.5 | 4.5 | 21 |
| E VRU | 3 | 0.0 | 5 | 1.5 | 9.5 |
| E oncoming | 3 | 7 | 1 | 0.5 | 11.5 |
| all | 20.5 | 26 | 26.5 | 14.5 | 87.5 |

### TTC-only failures: class x t0 ego speed band, m/s (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 0.0 | 9.5 | 10.5 | 0.0 | 20 |
| A2 lead vehicle (moving) | 0.0 | 0.0 | 3.5 | 0.0 | 3.5 |
| B cut-in | 3 | 5.5 | 0.5 | 1.5 | 10.5 |
| C crossing / turn conflict | 1 | 1 | 4.5 | 0.0 | 6.5 |
| D side contact | 0.0 | 3.5 | 1.5 | 0.0 | 5 |
| E static object | 1 | 10.5 | 9.5 | 0.0 | 21 |
| E VRU | 2 | 6.5 | 1 | 0.0 | 9.5 |
| E oncoming | 0.0 | 2.5 | 9 | 0.0 | 11.5 |
| all | 7 | 39 | 40 | 1.5 | 87.5 |

### TTC-only failures by turn bucket

| turn | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of bucket |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <5 | 20 | 21 | 20.50 | 58.54 | 21.95 | 46.34 | 70.73 | 65.85 | 75.61 | 0.32 |
| 5-20 | 27 | 25 | 26 | 59.62 | 15.38 | 15.38 | 59.62 | 59.62 | 80.77 | 1.00 |
| 20-45 | 28 | 25 | 26.50 | 73.58 | 11.32 | 26.42 | 83.02 | 56.60 | 90.57 | 1.62 |
| >45 | 13 | 16 | 14.50 | 55.17 | 27.59 | 24.14 | 82.76 | 72.41 | 82.76 | 0.96 |

### TTC-only failures by t0 ego speed band

| speed | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % | recovered, main % | clean recovered, main % | recovered, extended % | rate % of band |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <2 | 7 | 7 | 7 | 14.29 | 28.57 | 28.57 | 42.86 | 42.86 | 42.86 | 0.28 |
| 2-5 | 38 | 40 | 39 | 66.67 | 26.92 | 33.33 | 80.77 | 65.38 | 88.46 | 1.13 |
| 5-10 | 41 | 39 | 40 | 68.75 | 7.50 | 22.50 | 72.50 | 63.75 | 86.25 | 0.77 |
| >=10 | 2 | 1 | 1.50 | 33.33 | 0.00 | 0.00 | 33.33 | 33.33 | 33.33 | 0.15 |

### NC failures: class x turn x speed cells with seed-mean n >= 3

| cls | turn | speed | n_s0 | n_s1 | n | WA-JEPA passes % | recovered, main % |
|:--|--:|--:|--:|--:|--:|--:|--:|
| A2 lead vehicle (moving) | <5 | 5-10 | 17 | 17 | 17 | 76.5 | 100 |
| A1 stopped vehicle ahead | <5 | <2 | 14 | 11 | 12.5 | 100 | 44 |
| A1 stopped vehicle ahead | >45 | 2-5 | 12 | 12 | 12 | 58.3 | 79.2 |
| E static object | <5 | 5-10 | 9 | 9 | 9 | 100 | 61.1 |
| A2 lead vehicle (moving) | <5 | <2 | 8 | 9 | 8.5 | 76.5 | 82.4 |
| A1 stopped vehicle ahead | 20-45 | 2-5 | 8 | 8 | 8 | 100 | 81.2 |
| A2 lead vehicle (moving) | <5 | 2-5 | 8 | 7 | 7.5 | 73.3 | 86.7 |
| B cut-in | <5 | 2-5 | 7 | 7 | 7 | 42.9 | 92.9 |
| B cut-in | 5-20 | 2-5 | 7 | 7 | 7 | 85.7 | 85.7 |
| A1 stopped vehicle ahead | <5 | 5-10 | 7 | 6 | 6.5 | 100 | 76.9 |
| C crossing / turn conflict | <5 | 2-5 | 5 | 6 | 5.5 | 63.6 | 27.3 |
| A1 stopped vehicle ahead | 5-20 | 2-5 | 4 | 5 | 4.5 | 100 | 55.6 |
| A1 stopped vehicle ahead | 5-20 | <2 | 5 | 4 | 4.5 | 55.6 | 66.7 |
| A1 stopped vehicle ahead | <5 | 2-5 | 4 | 4 | 4 | 25 | 50 |
| D side contact | <5 | 5-10 | 4 | 3 | 3.5 | 85.7 | 42.9 |
| B cut-in | <5 | 5-10 | 4 | 3 | 3.5 | 71.4 | 100 |
| E oncoming | <5 | 2-5 | 3 | 4 | 3.5 | 42.9 | 28.6 |
| D side contact | <5 | >=10 | 3 | 3 | 3 | 100 | 66.7 |
| D side contact | >45 | 5-10 | 3 | 3 | 3 | 100 | 100 |
| D side contact | >45 | 2-5 | 3 | 3 | 3 | 100 | 100 |

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
| NC | 0.9-1.1 | 176.5 | 39 [33, 47] | 35 [30, 41] |
| NC | 0.8-1.1 | 176.5 | 61 [53, 70] | 58 [50, 66] |
| NC | 0.7-1.1 (main) | 176.5 | 74 [66, 83] | 70 [61, 79] |
| NC | 0.5-1.1 (extended) | 176.5 | 95 [93, 98] | 93 [89, 97] |
| NC | slower only (0.7-0.9) | 176.5 | 73 [64, 82] | 69 [60, 78] |
| NC | faster only (1.1) | 176.5 | 3 [1, 6] | 3 [1, 5] |
| TTC-only | 0.9-1.1 | 87.5 | 47 [37, 58] | 39 [30, 50] |
| TTC-only | 0.8-1.1 | 87.5 | 61 [51, 72] | 52 [41, 63] |
| TTC-only | 0.7-1.1 (main) | 87.5 | 73 [62, 84] | 62 [49, 74] |
| TTC-only | 0.5-1.1 (extended) | 87.5 | 83 [71, 93] | 75 [63, 86] |
| TTC-only | slower only (0.7-0.9) | 87.5 | 67 [56, 79] | 55 [44, 68] |
| TTC-only | faster only (1.1) | 87.5 | 13 [7, 19] | 11 [6, 17] |

### EP cost on the recovered tokens (a* = best-scoring recovering scale)

| set | tokens | n | EP(a*) - EP(1.0), x100 | no-EC EPDMS(a*) - (1.0), x100 | a*=0.7 | a*=0.8 | a*=0.9 | a*=1.1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| NC | recovered (main) | 131 | -2.98 | 88.35 | 31.50 | 39 | 57 | 3.50 |
| NC | clean recovered (main) | 124 | -3.03 | 93.34 | 29 | 39 | 53 | 3 |
| TTC-only | recovered (main) | 64 | -3.71 | 32.22 | 10.50 | 13 | 31 | 9.50 |
| TTC-only | clean recovered (main) | 54.50 | -3.52 | 38.39 | 9 | 11 | 25.50 | 9 |

### NC recovery by sub-class and family (%)

| cls | n_s0 | n_s1 | n | rec 0.9-1.1 | rec 0.8-1.1 | rec 0.7-1.1 (main) | rec 0.5-1.1 (extended) | rec slower only (0.7-0.9) | rec faster only (1.1) | clean 0.7-1.1 (main) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 60 | 57 | 58.5 | 31.6 | 51.3 | 67.5 | 92.3 | 67.5 | 0.9 | 61.5 |
| A2 lead vehicle (moving) | 35 | 35 | 35 | 45.7 | 85.7 | 92.9 | 100 | 92.9 | 0.0 | 92.9 |
| B cut-in | 24 | 22 | 23 | 54.3 | 76.1 | 91.3 | 100 | 91.3 | 0.0 | 87.0 |
| C crossing / turn conflict | 14 | 13 | 13.5 | 33.3 | 51.9 | 59.3 | 92.6 | 51.9 | 7.4 | 55.6 |
| D side contact | 17 | 16 | 16.5 | 51.5 | 60.6 | 72.7 | 90.9 | 63.6 | 15.2 | 72.7 |
| E VRU | 7 | 7 | 7 | 7.1 | 28.6 | 42.9 | 85.7 | 42.9 | 0.0 | 28.6 |
| E oncoming | 8 | 9 | 8.5 | 29.4 | 41.2 | 58.8 | 100 | 58.8 | 0.0 | 58.8 |
| E static object | 15 | 14 | 14.5 | 41.4 | 51.7 | 69.0 | 100 | 69.0 | 6.9 | 62.1 |

### TTC-only recovery by sub-class and family (%)

| cls | n_s0 | n_s1 | n | rec 0.9-1.1 | rec 0.8-1.1 | rec 0.7-1.1 (main) | rec 0.5-1.1 (extended) | rec slower only (0.7-0.9) | rec faster only (1.1) | clean 0.7-1.1 (main) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 19 | 21 | 20 | 52.5 | 62.5 | 80 | 92.5 | 75 | 7.5 | 72.5 |
| A2 lead vehicle (moving) | 3 | 4 | 3.5 | 100 | 100 | 100 | 100 | 71.4 | 28.6 | 100 |
| B cut-in | 11 | 10 | 10.5 | 38.1 | 42.9 | 52.4 | 52.4 | 28.6 | 23.8 | 52.4 |
| C crossing / turn conflict | 6 | 7 | 6.5 | 46.2 | 84.6 | 84.6 | 84.6 | 84.6 | 0.0 | 69.2 |
| D side contact | 4 | 6 | 5 | 60 | 70 | 100 | 100 | 100 | 0.0 | 100 |
| E VRU | 9 | 10 | 9.5 | 57.9 | 84.2 | 84.2 | 89.5 | 84.2 | 10.5 | 73.7 |
| E oncoming | 13 | 10 | 11.5 | 17.4 | 30.4 | 43.5 | 73.9 | 43.5 | 0.0 | 34.8 |
| E static object | 23 | 19 | 21 | 47.6 | 59.5 | 73.8 | 83.3 | 69.0 | 26.2 | 50 |

### NC recovery by turn bucket (%)

| turn | n_s0 | n_s1 | n | rec 0.7-1.1 (main) | clean 0.7-1.1 (main) | rec 0.5-1.1 (extended) |
|:--|--:|--:|--:|--:|--:|--:|
| <5 | 102 | 96 | 99 | 70.2 | 68.2 | 92.9 |
| 5-20 | 33 | 32 | 32.5 | 72.3 | 66.2 | 96.9 |
| 20-45 | 18 | 19 | 18.5 | 78.4 | 62.2 | 100 |
| >45 | 27 | 26 | 26.5 | 88.7 | 88.7 | 100 |

### Board-level upper bound: a* on SH30's failing tokens only, all other tokens unchanged (points of the 12 146-token mean)

| family | NC (s0 / s1) | TTC (s0 / s1) | EP (s0 / s1) | no-EC EPDMS (s0 / s1) | tokens replaced (seed mean) |
|:--|--:|--:|--:|--:|--:|
| main 0.7-1.1 | +0.984 (+0.955 / +1.013) | +1.268 (+1.243 / +1.293) | -0.047 (-0.046 / -0.048) | +1.125 (+1.099 / +1.151) | 178.5 |
| extended 0.5-1.1 | +1.303 (+1.326 / +1.280) | +1.712 (+1.729 / +1.696) | -0.089 (-0.095 / -0.083) | +1.483 (+1.502 / +1.463) | 231 |

### Uniform scaling of every token (change against a = 1.0 in points; counts are seed means)

| a | dNC | dTTC | dDAC | dEP | dno-EC EPDMS | NC fail | new NC fail | fixed NC fail | TTC-only fail |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 0.25 | -2.66 | -0.05 | -23.70 | -9.77 | 142 | 127.50 | 162 | 485.50 |
| 0.60 | 0.50 | -1.20 | 0.26 | -18.27 | -6.31 | 111.50 | 86 | 151 | 317 |
| 0.70 | 0.68 | 0.05 | 0.52 | -13.16 | -3.54 | 92 | 43 | 127.50 | 171 |
| 0.80 | 0.69 | 0.65 | 0.65 | -8.39 | -1.44 | 91.50 | 18.50 | 103.50 | 85.50 |
| 0.90 | 0.48 | 0.56 | 0.46 | -3.97 | -0.33 | 116 | 5 | 65.50 | 70 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 176.50 | 0.00 | 0.00 | 87.50 |
| 1.10 | -0.76 | -0.88 | -0.96 | 3.39 | -0.98 | 272 | 100.50 | 5 | 112 |

## navhard stage 1 (450 tokens, G frames): counts only

| set | n_s0 | n_s1 | n | WA-JEPA passes % | event at ego < 3 m/s % | plan > 1.1 x logged arc % |
|:--|--:|--:|--:|--:|--:|--:|
| NC | 13 | 14 | 13.5 | 66.7 | 14.8 | 29.6 |
| TTC-only | 1 | 1 | 1 | 0.0 | 100 | 100 |

### navhard stage 1, NC failures by sub-class

| cls | n_s0 | n_s1 | n | WA-JEPA passes % |
|:--|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 5 | 7 | 6 | 58.3 |
| B cut-in | 1 | 1 | 1 | 100 |
| D side contact | 1 | 0 | 0.5 | 100 |
| E VRU | 2 | 2 | 2 | 0.0 |
| E oncoming | 4 | 4 | 4 | 100 |

### navhard stage 1, NC: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 2 | 0.0 | 1 | 3 | 6 |
| B cut-in | 0.0 | 1 | 0.0 | 0.0 | 1 |
| D side contact | 0.0 | 0.0 | 0.5 | 0.0 | 0.5 |
| E VRU | 2 | 0.0 | 0.0 | 0.0 | 2 |
| E oncoming | 1.5 | 1 | 1.5 | 0.0 | 4 |
| all | 5.5 | 2 | 3 | 3 | 13.5 |

### navhard stage 1, NC: class x t0 ego speed band (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 2 | 2 | 2 | 0.0 | 6 |
| B cut-in | 0.0 | 0.0 | 1 | 0.0 | 1 |
| D side contact | 0.0 | 0.5 | 0.0 | 0.0 | 0.5 |
| E VRU | 0.0 | 0.0 | 2 | 0.0 | 2 |
| E oncoming | 0.0 | 0.0 | 4 | 0.0 | 4 |
| all | 2 | 2.5 | 9 | 0.0 | 13.5 |

### navhard stage 1, TTC-only failures by sub-class

| cls | n_s0 | n_s1 | n | WA-JEPA passes % |
|:--|--:|--:|--:|--:|
| E oncoming | 1 | 1 | 1 | 0.0 |

### navhard stage 1, TTC-only: class x turn bucket (seed-mean counts)

| class | <5 | 5-20 | 20-45 | >45 | all |
|:--|--:|--:|--:|--:|--:|
| E oncoming | 1 | 0.0 | 0.0 | 0.0 | 1 |
| all | 1 | 0.0 | 0.0 | 0.0 | 1 |

### navhard stage 1, TTC-only: class x t0 ego speed band (seed-mean counts)

| class | <2 | 2-5 | 5-10 | >=10 | all |
|:--|--:|--:|--:|--:|--:|
| E oncoming | 1 | 0.0 | 0.0 | 0.0 | 1 |
| all | 1 | 0.0 | 0.0 | 0.0 | 1 |

WA-JEPA on the same 450 stage-1 tokens: 9 NC failures, 4 TTC-only failures (stored per-token scores).
