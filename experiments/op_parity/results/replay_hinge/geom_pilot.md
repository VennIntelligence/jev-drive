| contrast           | stratum       | metric                  |     n |   arm |   ref |   diff |    lo |    hi |
|:-------------------|:--------------|:------------------------|------:|------:|------:|-------:|------:|------:|
| RMP-F-s0 - HP-F-s0 | all           | replay out (DAC fail) % | 12146 |  3.98 |  4.37 |  -0.40 | -0.54 | -0.25 |
| RMP-F-s0 - HP-F-s0 | all           | raw plan out %          | 12146 |  4.36 |  3.99 |   0.37 |  0.15 |  0.61 |
| RMP-F-s0 - HP-F-s0 | all           | replay only %           | 12146 |  1.09 |  1.50 |  -0.41 | -0.59 | -0.24 |
| RMP-F-s0 - HP-F-s0 | all           | raw only %              | 12146 |  1.47 |  1.12 |   0.35 |  0.18 |  0.55 |
| RMP-F-s0 - HP-F-s0 | turn > 20 deg | replay out (DAC fail) % |  3154 |  8.94 |  9.80 |  -0.86 | -1.30 | -0.41 |
| RMP-F-s0 - HP-F-s0 | turn > 20 deg | raw plan out %          |  3154 |  9.92 |  9.04 |   0.89 |  0.25 |  1.52 |
| RMP-F-s0 - HP-F-s0 | turn > 20 deg | replay only %           |  3154 |  2.50 |  3.49 |  -0.98 | -1.48 | -0.48 |
| RMP-F-s0 - HP-F-s0 | turn > 20 deg | raw only %              |  3154 |  3.49 |  2.73 |   0.76 |  0.23 |  1.27 |
| RMP-F-s0 - HP-F-s0 | > 45 deg      | replay out (DAC fail) % |  1517 |  9.95 | 10.94 |  -0.99 | -1.66 | -0.38 |
| RMP-F-s0 - HP-F-s0 | > 45 deg      | raw plan out %          |  1517 | 12.72 | 11.93 |   0.79 |  0.00 |  1.65 |
| RMP-F-s0 - HP-F-s0 | > 45 deg      | replay only %           |  1517 |  2.24 |  2.83 |  -0.59 | -1.32 |  0.13 |
| RMP-F-s0 - HP-F-s0 | > 45 deg      | raw only %              |  1517 |  5.01 |  3.82 |   1.19 |  0.38 |  2.05 |

CI: percentile bootstrap

footprint departures from the scorer's drivable polygons, devkit replay vs raw plan; paired, cluster bootstrap over navtest logs, B 10 000
