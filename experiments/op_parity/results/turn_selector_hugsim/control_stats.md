| set        | metric      |   SH30 |    tsB |   diff |      lo |     hi |
|:-----------|:------------|-------:|-------:|-------:|--------:|-------:|
| turn23     | jerk_rms    | 0.3519 | 0.4196 | 0.0677 |  0.0244 | 0.1174 |
| turn23     | dsteer_mean | 0.0084 | 0.0107 | 0.0024 |  0.0007 | 0.0045 |
| turn23     | dsteer_p95  | 0.0253 | 0.0330 | 0.0077 |  0.0016 | 0.0142 |
| straight41 | jerk_rms    | 0.1857 | 0.1914 | 0.0057 | -0.0071 | 0.0220 |
| straight41 | dsteer_mean | 0.0030 | 0.0032 | 0.0002 | -0.0000 | 0.0006 |
| straight41 | dsteer_p95  | 0.0074 | 0.0085 | 0.0012 | -0.0000 | 0.0030 |
| all64      | jerk_rms    | 0.2454 | 0.2734 | 0.0280 |  0.0089 | 0.0510 |
| all64      | dsteer_mean | 0.0049 | 0.0059 | 0.0010 |  0.0004 | 0.0019 |
| all64      | dsteer_p95  | 0.0138 | 0.0173 | 0.0035 |  0.0011 | 0.0062 |

CI: percentile bootstrap

v > 1 m/s steps; jerk = d/dt of v * yaw rate (m/s^3, rms per scenario); dsteer = |step change of ego_steer|; seeds averaged per scenario, bootstrap over scenarios
