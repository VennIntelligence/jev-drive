## navtest

| metric | model | variant | n | n_valid | score | ci_lo | ci_hi | NC | DAC | EP | TTC | C | DDC | TLC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PDMS | human (log) |  | 12146 | 12146 | 94.6 | 94.5 | 94.7 | 100.0 | 100.0 | 87.0 | 100.0 | 99.9 |  |  |  |  |  |
| PDMS | constant velocity |  | 12146 | 12146 | 20.7 | 20.1 | 21.2 | 68.0 | 57.8 | 19.4 | 50.0 | 100.0 |  |  |  |  |  |
| PDMS | Alpamayo 1.5 | nav | 12146 | 12146 | 44.3 | 43.5 | 45.0 | 76.8 | 70.6 | 42.6 | 62.6 | 96.0 |  |  |  |  |  |
| PDMS | Alpamayo 1.5 | no-nav | 3000 | 3000 | 41.9 | 40.3 | 43.5 | 81.3 | 63.9 | 38.7 | 65.9 | 95.9 |  |  |  |  |  |
| PDMS | openpilot Lebowski | none | 12146 | 12146 | 50.9 | 50.2 | 51.6 | 78.5 | 79.1 | 49.7 | 71.1 | 63.3 |  |  |  |  |  |
| PDMS | openpilot Lebowski | cmd | 12146 | 12146 | 49.4 | 48.6 | 50.1 | 78.1 | 76.9 | 47.5 | 70.8 | 65.0 |  |  |  |  |  |
| PDMS | openpilot Cinque v3 | none | 12146 | 12146 | 52.1 | 51.3 | 52.8 | 78.4 | 75.7 | 55.7 | 67.8 | 68.8 |  |  |  |  |  |
| PDMS | openpilot Cinque v3 | cmd | 12146 | 12146 | 50.7 | 49.8 | 51.4 | 77.9 | 74.1 | 54.2 | 67.3 | 68.9 |  |  |  |  |  |
| PDMS | openpilot small | none | 12146 | 12146 | 47.3 | 46.6 | 48.2 | 70.9 | 79.5 | 51.1 | 61.6 | 58.1 |  |  |  |  |  |
| PDMS | openpilot small | cmd | 12146 | 12146 | 42.3 | 41.6 | 43.2 | 69.6 | 72.8 | 44.8 | 60.1 | 64.6 |  |  |  |  |  |
| EPDMS | human (log) |  | 12146 | 12146 | 94.5 | 94.4 | 94.6 | 100.0 | 100.0 | 87.4 | 100.0 |  | 99.8 | 100.0 | 100.0 | 98.1 | 90.1 |
| EPDMS | constant velocity |  | 12146 | 12146 | 25.9 | 25.2 | 26.6 | 68.1 | 57.8 | 77.7 | 66.9 |  | 84.2 | 98.1 | 82.7 | 97.9 | 55.3 |
| EPDMS | Alpamayo 1.5 | nav | 12146 | 12146 | 43.2 | 42.4 | 43.9 | 76.8 | 70.6 | 84.8 | 73.2 |  | 86.3 | 98.3 | 85.4 | 91.0 | 32.7 |
| EPDMS | Alpamayo 1.5 | no-nav | 3000 | 3000 | 41.6 | 40.1 | 43.2 | 81.4 | 63.9 | 81.6 | 76.7 |  | 82.2 | 98.4 | 83.5 | 88.5 | 29.7 |
| EPDMS | openpilot Lebowski | none | 12146 | 12146 | 45.5 | 44.7 | 46.2 | 78.6 | 79.1 | 85.5 | 77.6 |  | 87.2 | 97.4 | 87.6 | 52.4 | 12.3 |
| EPDMS | openpilot Lebowski | cmd | 12146 | 12146 | 44.3 | 43.6 | 45.0 | 78.2 | 76.9 | 85.0 | 77.2 |  | 86.1 | 97.4 | 86.8 | 54.0 | 12.4 |
| EPDMS | openpilot Cinque v3 | none | 12146 | 12146 | 46.2 | 45.4 | 46.9 | 78.4 | 75.7 | 95.2 | 75.2 |  | 85.1 | 96.9 | 89.6 | 52.5 | 10.0 |
| EPDMS | openpilot Cinque v3 | cmd | 12146 | 12146 | 45.2 | 44.4 | 45.9 | 78.0 | 74.1 | 95.1 | 74.6 |  | 84.1 | 97.0 | 88.4 | 52.3 | 10.0 |
| EPDMS | openpilot small | none | 12146 | 12146 | 42.5 | 41.8 | 43.3 | 71.0 | 79.5 | 95.0 | 68.3 |  | 89.0 | 95.6 | 91.2 | 45.2 | 12.1 |
| EPDMS | openpilot small | cmd | 12146 | 12146 | 39.1 | 38.4 | 39.9 | 69.7 | 72.8 | 93.1 | 67.0 |  | 84.3 | 95.9 | 88.1 | 55.5 | 13.5 |

## navtest by command

| metric | model | variant | command | n | score | EP | DAC | NC |
|---|---|---|---|---|---|---|---|---|
| PDMS | human (log) |  | left | 2501 | 94.4 | 86.6 | 100.0 | 100.0 |
| PDMS | human (log) |  | right | 1575 | 93.0 | 83.3 | 100.0 | 100.0 |
| PDMS | human (log) |  | straight | 8070 | 94.9 | 87.8 | 100.0 | 100.0 |
| PDMS | constant velocity |  | left | 2501 | 18.7 | 11.4 | 35.7 | 78.6 |
| PDMS | constant velocity |  | right | 1575 | 21.9 | 12.7 | 43.4 | 77.0 |
| PDMS | constant velocity |  | straight | 8070 | 21.0 | 23.3 | 67.5 | 63.0 |
| PDMS | Alpamayo 1.5 | nav | left | 2501 | 39.7 | 35.8 | 54.3 | 86.5 |
| PDMS | Alpamayo 1.5 | nav | right | 1575 | 40.8 | 35.3 | 59.1 | 84.5 |
| PDMS | Alpamayo 1.5 | nav | straight | 8070 | 46.4 | 46.1 | 78.0 | 72.3 |
| PDMS | Alpamayo 1.5 | no-nav | left | 1000 | 40.2 | 37.0 | 55.1 | 87.4 |
| PDMS | Alpamayo 1.5 | no-nav | right | 1000 | 41.9 | 35.3 | 59.6 | 85.2 |
| PDMS | Alpamayo 1.5 | no-nav | straight | 1000 | 43.5 | 43.9 | 77.1 | 71.5 |
| PDMS | openpilot Lebowski | none | left | 2501 | 47.4 | 44.8 | 65.1 | 84.4 |
| PDMS | openpilot Lebowski | none | right | 1575 | 53.9 | 49.9 | 73.3 | 86.7 |
| PDMS | openpilot Lebowski | none | straight | 8070 | 51.4 | 51.1 | 84.6 | 75.1 |
| PDMS | openpilot Lebowski | cmd | left | 2501 | 44.1 | 39.6 | 60.5 | 83.3 |
| PDMS | openpilot Lebowski | cmd | right | 1575 | 50.1 | 45.0 | 67.4 | 85.8 |
| PDMS | openpilot Lebowski | cmd | straight | 8070 | 50.8 | 50.4 | 83.8 | 75.0 |
| PDMS | openpilot Cinque v3 | none | left | 2501 | 38.1 | 39.5 | 51.5 | 79.5 |
| PDMS | openpilot Cinque v3 | none | right | 1575 | 48.6 | 48.3 | 60.8 | 84.1 |
| PDMS | openpilot Cinque v3 | none | straight | 8070 | 57.1 | 62.2 | 86.2 | 76.9 |
| PDMS | openpilot Cinque v3 | cmd | left | 2501 | 35.8 | 37.2 | 49.5 | 78.5 |
| PDMS | openpilot Cinque v3 | cmd | right | 1575 | 47.8 | 47.9 | 59.6 | 85.0 |
| PDMS | openpilot Cinque v3 | cmd | straight | 8070 | 55.8 | 60.7 | 84.6 | 76.4 |
| PDMS | openpilot small | none | left | 2501 | 46.1 | 46.7 | 64.0 | 83.0 |
| PDMS | openpilot small | none | right | 1575 | 48.5 | 47.0 | 66.2 | 81.0 |
| PDMS | openpilot small | none | straight | 8070 | 47.5 | 53.3 | 86.9 | 65.2 |
| PDMS | openpilot small | cmd | left | 2501 | 32.5 | 30.8 | 47.1 | 76.8 |
| PDMS | openpilot small | cmd | right | 1575 | 43.8 | 39.3 | 57.6 | 84.2 |
| PDMS | openpilot small | cmd | straight | 8070 | 45.1 | 50.3 | 83.7 | 64.6 |
| EPDMS | human (log) |  | left | 2501 | 92.1 | 87.3 | 100.0 | 100.0 |
| EPDMS | human (log) |  | right | 1575 | 91.9 | 83.7 | 100.0 | 100.0 |
| EPDMS | human (log) |  | straight | 8070 | 95.8 | 88.2 | 100.0 | 100.0 |
| EPDMS | constant velocity |  | left | 2501 | 20.5 | 68.9 | 35.7 | 78.6 |
| EPDMS | constant velocity |  | right | 1575 | 23.1 | 65.1 | 43.4 | 77.1 |
| EPDMS | constant velocity |  | straight | 8070 | 28.1 | 83.0 | 67.5 | 63.0 |
| EPDMS | Alpamayo 1.5 | nav | left | 2501 | 37.4 | 82.0 | 54.3 | 86.5 |
| EPDMS | Alpamayo 1.5 | nav | right | 1575 | 37.7 | 75.0 | 59.1 | 84.5 |
| EPDMS | Alpamayo 1.5 | nav | straight | 8070 | 46.0 | 87.6 | 78.0 | 72.3 |
| EPDMS | Alpamayo 1.5 | no-nav | left | 1000 | 39.4 | 82.3 | 55.1 | 87.5 |
| EPDMS | Alpamayo 1.5 | no-nav | right | 1000 | 38.8 | 74.6 | 59.6 | 85.2 |
| EPDMS | Alpamayo 1.5 | no-nav | straight | 1000 | 46.5 | 88.0 | 77.1 | 71.5 |
| EPDMS | openpilot Lebowski | none | left | 2501 | 39.5 | 84.8 | 65.1 | 84.5 |
| EPDMS | openpilot Lebowski | none | right | 1575 | 44.9 | 82.3 | 73.3 | 86.8 |
| EPDMS | openpilot Lebowski | none | straight | 8070 | 47.5 | 86.3 | 84.6 | 75.2 |
| EPDMS | openpilot Lebowski | cmd | left | 2501 | 36.9 | 83.1 | 60.5 | 83.4 |
| EPDMS | openpilot Lebowski | cmd | right | 1575 | 42.2 | 82.5 | 67.4 | 85.8 |
| EPDMS | openpilot Lebowski | cmd | straight | 8070 | 47.0 | 86.0 | 83.8 | 75.1 |
| EPDMS | openpilot Cinque v3 | none | left | 2501 | 29.6 | 94.6 | 51.5 | 79.6 |
| EPDMS | openpilot Cinque v3 | none | right | 1575 | 39.6 | 92.0 | 60.8 | 84.3 |
| EPDMS | openpilot Cinque v3 | none | straight | 8070 | 52.6 | 96.1 | 86.2 | 76.9 |
| EPDMS | openpilot Cinque v3 | cmd | left | 2501 | 28.5 | 94.7 | 49.5 | 78.6 |
| EPDMS | openpilot Cinque v3 | cmd | right | 1575 | 39.7 | 91.9 | 59.6 | 85.1 |
| EPDMS | openpilot Cinque v3 | cmd | straight | 8070 | 51.4 | 95.8 | 84.6 | 76.4 |
| EPDMS | openpilot small | none | left | 2501 | 37.5 | 90.9 | 64.0 | 83.2 |
| EPDMS | openpilot small | none | right | 1575 | 41.6 | 87.5 | 66.2 | 81.1 |
| EPDMS | openpilot small | none | straight | 8070 | 44.3 | 97.7 | 86.9 | 65.2 |
| EPDMS | openpilot small | cmd | left | 2501 | 29.0 | 87.3 | 47.1 | 77.0 |
| EPDMS | openpilot small | cmd | right | 1575 | 40.5 | 84.3 | 57.6 | 84.2 |
| EPDMS | openpilot small | cmd | straight | 8070 | 42.0 | 96.6 | 83.7 | 64.6 |

## paired differences

| metric | a | b | command | n | a_mean | b_mean | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| PDMS | alpamayo_nav | alpamayo_nonav | all | 3000 | 41.4 | 41.9 | -0.5 | -2.1 | 1.0 |
| PDMS | alpamayo_nav | alpamayo_nonav | left | 1000 | 37.4 | 40.2 | -2.8 | -5.5 | -0.1 |
| PDMS | alpamayo_nav | alpamayo_nonav | straight | 1000 | 44.7 | 43.5 | 1.2 | -1.5 | 4.1 |
| PDMS | alpamayo_nav | alpamayo_nonav | right | 1000 | 42.0 | 41.9 | 0.1 | -2.5 | 2.8 |
| PDMS | alpamayo_nav | lebowski_none | all | 12146 | 44.3 | 50.9 | -6.6 | -7.6 | -5.6 |
| PDMS | alpamayo_nav | lebowski_none | left | 2501 | 39.7 | 47.4 | -7.7 | -9.9 | -5.5 |
| PDMS | alpamayo_nav | lebowski_none | straight | 8070 | 46.4 | 51.4 | -5.0 | -6.2 | -3.8 |
| PDMS | alpamayo_nav | lebowski_none | right | 1575 | 40.8 | 53.9 | -13.1 | -15.9 | -10.6 |
| PDMS | lebowski_cmd | lebowski_none | all | 12146 | 49.4 | 50.9 | -1.5 | -1.9 | -1.1 |
| PDMS | lebowski_cmd | lebowski_none | left | 2501 | 44.1 | 47.4 | -3.3 | -4.8 | -2.0 |
| PDMS | lebowski_cmd | lebowski_none | straight | 8070 | 50.8 | 51.4 | -0.5 | -0.8 | -0.3 |
| PDMS | lebowski_cmd | lebowski_none | right | 1575 | 50.1 | 53.9 | -3.9 | -5.6 | -2.2 |
| PDMS | cinque_cmd | cinque_none | all | 12146 | 50.7 | 52.1 | -1.4 | -1.9 | -1.0 |
| PDMS | cinque_cmd | cinque_none | left | 2501 | 35.8 | 38.1 | -2.2 | -3.8 | -0.7 |
| PDMS | cinque_cmd | cinque_none | straight | 8070 | 55.8 | 57.1 | -1.3 | -1.6 | -1.0 |
| PDMS | cinque_cmd | cinque_none | right | 1575 | 47.8 | 48.6 | -0.8 | -2.6 | 1.1 |
| PDMS | small_cmd | small_none | all | 12146 | 42.3 | 47.3 | -5.0 | -5.6 | -4.4 |
| PDMS | small_cmd | small_none | left | 2501 | 32.5 | 46.1 | -13.7 | -15.6 | -11.8 |
| PDMS | small_cmd | small_none | straight | 8070 | 45.1 | 47.5 | -2.4 | -2.8 | -2.0 |
| PDMS | small_cmd | small_none | right | 1575 | 43.8 | 48.5 | -4.7 | -7.2 | -2.3 |
| PDMS | alpamayo_nav | cv | all | 12146 | 44.3 | 20.7 | 23.6 | 22.9 | 24.3 |
| PDMS | alpamayo_nav | cv | left | 2501 | 39.7 | 18.7 | 21.0 | 19.5 | 22.7 |
| PDMS | alpamayo_nav | cv | straight | 8070 | 46.4 | 21.0 | 25.3 | 24.4 | 26.2 |
| PDMS | alpamayo_nav | cv | right | 1575 | 40.8 | 21.9 | 19.0 | 17.1 | 20.8 |
| EPDMS | alpamayo_nav | alpamayo_nonav | all | 3000 | 39.6 | 41.6 | -2.0 | -3.5 | -0.6 |
| EPDMS | alpamayo_nav | alpamayo_nonav | left | 1000 | 35.6 | 39.4 | -3.8 | -6.3 | -1.2 |
| EPDMS | alpamayo_nav | alpamayo_nonav | straight | 1000 | 44.3 | 46.5 | -2.3 | -5.2 | 0.8 |
| EPDMS | alpamayo_nav | alpamayo_nonav | right | 1000 | 38.8 | 38.8 | 0.0 | -2.2 | 2.4 |
| EPDMS | alpamayo_nav | lebowski_none | all | 12146 | 43.2 | 45.5 | -2.3 | -3.2 | -1.4 |
| EPDMS | alpamayo_nav | lebowski_none | left | 2501 | 37.4 | 39.5 | -2.1 | -4.0 | -0.1 |
| EPDMS | alpamayo_nav | lebowski_none | straight | 8070 | 46.0 | 47.5 | -1.4 | -2.6 | -0.2 |
| EPDMS | alpamayo_nav | lebowski_none | right | 1575 | 37.7 | 44.9 | -7.2 | -9.6 | -4.8 |
| EPDMS | lebowski_cmd | lebowski_none | all | 12146 | 44.3 | 45.5 | -1.2 | -1.5 | -0.9 |
| EPDMS | lebowski_cmd | lebowski_none | left | 2501 | 36.9 | 39.5 | -2.6 | -3.8 | -1.4 |
| EPDMS | lebowski_cmd | lebowski_none | straight | 8070 | 47.0 | 47.5 | -0.5 | -0.7 | -0.3 |
| EPDMS | lebowski_cmd | lebowski_none | right | 1575 | 42.2 | 44.9 | -2.7 | -4.2 | -1.2 |
| EPDMS | cinque_cmd | cinque_none | all | 12146 | 45.2 | 46.2 | -1.0 | -1.4 | -0.6 |
| EPDMS | cinque_cmd | cinque_none | left | 2501 | 28.5 | 29.6 | -1.1 | -2.3 | 0.1 |
| EPDMS | cinque_cmd | cinque_none | straight | 8070 | 51.4 | 52.6 | -1.2 | -1.4 | -0.9 |
| EPDMS | cinque_cmd | cinque_none | right | 1575 | 39.7 | 39.6 | 0.1 | -1.5 | 1.7 |
| EPDMS | small_cmd | small_none | all | 12146 | 39.1 | 42.5 | -3.4 | -3.9 | -2.9 |
| EPDMS | small_cmd | small_none | left | 2501 | 29.0 | 37.5 | -8.5 | -10.1 | -7.0 |
| EPDMS | small_cmd | small_none | straight | 8070 | 42.0 | 44.3 | -2.2 | -2.6 | -1.9 |
| EPDMS | small_cmd | small_none | right | 1575 | 40.5 | 41.6 | -1.1 | -3.1 | 0.9 |
| EPDMS | alpamayo_nav | cv | all | 12146 | 43.2 | 25.9 | 17.3 | 16.5 | 18.0 |
| EPDMS | alpamayo_nav | cv | left | 2501 | 37.4 | 20.5 | 16.9 | 15.4 | 18.4 |
| EPDMS | alpamayo_nav | cv | straight | 8070 | 46.0 | 28.1 | 17.9 | 16.9 | 18.9 |
| EPDMS | alpamayo_nav | cv | right | 1575 | 37.7 | 23.1 | 14.6 | 12.8 | 16.4 |

## failure classes (EPDMS)

| agent | NC_fail | DAC_fail | DDC_fail | TLC_fail | TTC_fail | EC_fail | NC_fail_overshoot>2m | all_overshoot>2m | DAC_fail_turn_cmd | all_turn_cmd | DAC_fail_lat_err_med | lat_err_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpamayo_nav | 24.9 | 29.4 | 17.2 | 1.7 | 26.8 | 55.6 | 84.0 | 56.7 | 50.2 | 33.6 | 3.2 | 0.9 |
| cv | 34.6 | 42.2 | 21.5 | 1.9 | 33.1 | 36.9 | 69.5 | 48.4 | 48.8 | 33.6 | 3.1 | 1.0 |
| lebowski_none | 22.5 | 20.9 | 15.1 | 2.6 | 22.4 | 72.5 | 82.9 | 68.7 | 50.9 | 33.6 | 7.6 | 1.6 |
| cinque_none | 23.3 | 24.3 | 17.2 | 3.1 | 24.8 | 74.4 | 84.6 | 77.9 | 62.2 | 33.6 | 6.9 | 1.3 |
| small_none | 30.0 | 20.5 | 13.2 | 4.4 | 31.7 | 72.6 | 95.7 | 86.3 | 57.6 | 33.6 | 5.6 | 1.5 |

## navhard two-stage

| model | variant | n_tokens | stage1 | stage2 | EPDMS | NC_one | NC_two | DAC_one | DAC_two | DDC_one | DDC_two | TLC_one | TLC_two | EP_one | EP_two | TTC_one | TTC_two | LK_one | LK_two | HC_one | HC_two | EC_one | EC_two |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constant velocity |  | 5912 | 29.0 | 34.2 | 11.5 | 88.9 | 83.2 | 42.9 | 59.1 | 70.7 | 76.5 | 99.3 | 98.1 | 77.5 | 71.4 | 87.3 | 81.1 | 78.7 | 48.0 | 97.1 | 97.2 | 60.4 | 62.0 |
| Alpamayo 1.5 | nav | 5912 | 34.1 | 32.6 | 10.8 | 89.4 | 80.6 | 50.9 | 60.0 | 76.4 | 78.4 | 99.3 | 97.9 | 81.5 | 75.0 | 84.9 | 76.7 | 86.4 | 53.1 | 88.2 | 89.1 | 26.2 | 29.3 |
| openpilot Lebowski | none | 5912 | 31.8 | 30.2 | 10.2 | 82.3 | 79.2 | 60.2 | 61.7 | 73.2 | 73.4 | 97.8 | 97.6 | 86.7 | 81.8 | 80.2 | 74.5 | 79.6 | 49.8 | 49.3 | 57.7 | 5.3 | 16.3 |
| openpilot small | none | 5912 | 28.7 | 31.5 | 10.2 | 79.0 | 76.7 | 57.8 | 67.1 | 72.8 | 75.5 | 97.1 | 97.4 | 93.6 | 86.3 | 74.4 | 73.3 | 85.8 | 52.3 | 36.7 | 52.5 | 5.8 | 19.0 |

## ADE vs log (navtest)

| pred | n | ADE4 | FDE4 | median_ADE4 | progress_ratio |
|---|---|---|---|---|---|
| alpamayo_nav_repeat_main | 12146 | 3.066 | 7.204 | 2.522 | 1.182 |
| alpamayo_nonav_repeat_main | 3000 | 3.005 | 7.076 | 2.454 | 1.146 |
| cvreplay | 12146 | 2.797 | 6.752 | 2.611 | 1.071 |
| logreplay | 12146 | 0.000 | 0.000 | 0.000 | 1.000 |
| cinque_cmd | 12146 | 8.360 | 14.735 | 6.213 | 1.582 |
| cinque_none | 12146 | 8.380 | 14.909 | 6.116 | 1.595 |
| lebowski_cmd | 12146 | 9.136 | 15.728 | 7.202 | 1.474 |
| lebowski_none | 12146 | 9.294 | 16.020 | 7.499 | 1.531 |
| small_cmd | 12146 | 9.572 | 16.603 | 6.455 | 1.705 |
| small_none | 12146 | 10.454 | 18.354 | 8.213 | 1.864 |
