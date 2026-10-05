| arm   | set      | metric     | op    |   n |   units |   mean |     lo |     hi |   mean_a |   mean_b |
|:------|:---------|:-----------|:------|----:|--------:|-------:|-------:|-------:|---------:|---------:|
| Wfrz  | turn     | A_act      | diff  |   8 |       8 | -0.035 | -0.067 | -0.007 |    0.854 |    0.889 |
| Wfrz  | turn     | A_plan     | diff  |   8 |       8 | -0.006 | -0.014 |  0.001 |    1.034 |    1.040 |
| Wfrz  | turn     | H2         | diff  |   8 |       8 |  0.002 | -0.016 |  0.024 |    0.977 |    0.974 |
| Wfrz  | turn     | Y2         | diff  |   8 |       8 | -0.009 | -0.030 |  0.013 |    1.020 |    1.029 |
| Wfrz  | turn     | peak_ratio | diff  |   8 |       8 | -0.024 | -0.059 |  0.009 |    0.912 |    0.935 |
| Wfrz  | turn     | lead       | diff  |   8 |       8 | -0.219 | -0.506 |  0.019 |   -0.063 |    0.156 |
| Wfrz  | turn     | FDE3       | diff  |   8 |       8 |  0.526 | -0.155 |  1.262 |    2.879 |    2.352 |
| Wfrz  | turn     | lane_w10   | ratio |   2 |       2 |  1.004 |  0.992 |  1.016 |    2.689 |    2.877 |
| Wfrz  | turn     | v_ratio    | ratio |   8 |       8 |  0.993 |  0.970 |  1.016 |    0.984 |    0.990 |
| Wfrz  | straight | lane_w10   | ratio |   4 |       4 |  0.998 |  0.996 |  1.000 |    2.935 |    2.941 |
| Wfrz  | straight | v_ratio    | ratio |   4 |       4 |  1.004 |  1.000 |  1.007 |    1.018 |    1.014 |
| Wfrz  | straight | ADE5       | ratio |   4 |       4 |  1.088 |  0.935 |  1.275 |    0.903 |    0.839 |

CI: percentile bootstrap, n_boot = 10000, seed = 0, alpha = 0.05

split comma1m/fov-pilot@v1:251572ae8695; diff = arm - N, ratio = arm / N per window; cluster bootstrap over segments
