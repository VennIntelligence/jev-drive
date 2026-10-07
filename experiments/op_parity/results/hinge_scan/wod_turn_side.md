| arm             |   n_turn_frames |   median_inward_m |   mean_inward_m |   inside_gt1m |   wide_gt1m |   n_near |   near_median_inward_m |   near_inside_gt1m | metric               |   mean |    ref |   diff |     lo |     hi |
|:----------------|----------------:|------------------:|----------------:|--------------:|------------:|---------:|-----------------------:|-------------------:|:---------------------|-------:|-------:|-------:|-------:|-------:|
| shipped         |              52 |              0.63 |            1.13 |         23.00 |        6.00 |    19.00 |                   0.06 |               4.00 | nan                  | nan    | nan    | nan    | nan    | nan    |
| P2H10           |              52 |              0.06 |            0.19 |         15.00 |        9.00 |    19.00 |                  -0.08 |               5.00 | nan                  | nan    | nan    | nan    | nan    | nan    |
| SH30            |              52 |              0.06 |            0.02 |         12.00 |       11.00 |    19.00 |                   0.07 |               4.00 | nan                  | nan    | nan    | nan    | nan    | nan    |
| SH30 - P2H10    |              52 |            nan    |          nan    |        nan    |      nan    |   nan    |                 nan    |             nan    | mean inward miss (m) |   0.02 |   0.19 |  -0.17 |  -0.26 |  -0.07 |
| SH30 - P2H10    |              52 |            nan    |          nan    |        nan    |      nan    |   nan    |                 nan    |             nan    | inside > 1 m (%)     |  23.08 |  28.85 |  -5.77 | -13.46 |   0.00 |
| SH30 - shipped  |              52 |            nan    |          nan    |        nan    |      nan    |   nan    |                 nan    |             nan    | mean inward miss (m) |   0.02 |   1.13 |  -1.11 |  -1.85 |  -0.41 |
| SH30 - shipped  |              52 |            nan    |          nan    |        nan    |      nan    |   nan    |                 nan    |             nan    | inside > 1 m (%)     |  23.08 |  44.23 | -21.15 | -36.54 |  -7.64 |
| P2H10 - shipped |              52 |            nan    |          nan    |        nan    |      nan    |   nan    |                 nan    |             nan    | mean inward miss (m) |   0.19 |   1.13 |  -0.94 |  -1.70 |  -0.25 |
| P2H10 - shipped |              52 |            nan    |          nan    |        nan    |      nan    |   nan    |                 nan    |             nan    | inside > 1 m (%)     |  28.85 |  44.23 | -15.38 | -30.77 |   0.00 |

CI: percentile bootstrap

turn frames (intent left / right) of the 479 rater frames; miss at 5 s against the top-rated trajectory, + = inside of the turn; per-frame seed means; paired rows bootstrap over sequences
