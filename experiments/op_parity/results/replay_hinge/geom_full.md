| contrast                | stratum       | metric                  |     n |   arm |   ref |   diff |    lo |    hi |
|:------------------------|:--------------|:------------------------|------:|------:|------:|-------:|------:|------:|
| RMH10-F-s0 - P2H10-F-s0 | all           | replay out (DAC fail) % | 12146 |  3.35 |  3.86 |  -0.51 | -0.71 | -0.32 |
| RMH10-F-s0 - P2H10-F-s0 | all           | raw plan out %          | 12146 |  3.93 |  3.59 |   0.34 |  0.16 |  0.52 |
| RMH10-F-s0 - P2H10-F-s0 | all           | replay only %           | 12146 |  0.77 |  1.27 |  -0.49 | -0.68 | -0.32 |
| RMH10-F-s0 - P2H10-F-s0 | all           | raw only %              | 12146 |  1.35 |  1.00 |   0.35 |  0.18 |  0.54 |
| RMH10-F-s0 - P2H10-F-s0 | turn > 20 deg | replay out (DAC fail) % |  3154 |  7.64 |  8.81 |  -1.17 | -1.67 | -0.63 |
| RMH10-F-s0 - P2H10-F-s0 | turn > 20 deg | raw plan out %          |  3154 |  9.04 |  8.05 |   0.98 |  0.45 |  1.52 |
| RMH10-F-s0 - P2H10-F-s0 | turn > 20 deg | replay only %           |  3154 |  2.00 |  3.20 |  -1.20 | -1.69 | -0.71 |
| RMH10-F-s0 - P2H10-F-s0 | turn > 20 deg | raw only %              |  3154 |  3.39 |  2.44 |   0.95 |  0.49 |  1.40 |
| RMH10-F-s0 - P2H10-F-s0 | > 45 deg      | replay out (DAC fail) % |  1517 |  9.23 | 10.22 |  -0.99 | -1.72 | -0.15 |
| RMH10-F-s0 - P2H10-F-s0 | > 45 deg      | raw plan out %          |  1517 | 12.39 | 11.27 |   1.12 |  0.36 |  1.92 |
| RMH10-F-s0 - P2H10-F-s0 | > 45 deg      | replay only %           |  1517 |  1.91 |  2.83 |  -0.92 | -1.51 | -0.32 |
| RMH10-F-s0 - P2H10-F-s0 | > 45 deg      | raw only %              |  1517 |  5.08 |  3.89 |   1.19 |  0.47 |  1.92 |
| RMH10-F-s1 - P2H10-F-s1 | all           | replay out (DAC fail) % | 12146 |  3.28 |  3.80 |  -0.52 | -0.69 | -0.36 |
| RMH10-F-s1 - P2H10-F-s1 | all           | raw plan out %          | 12146 |  3.95 |  3.67 |   0.28 |  0.11 |  0.47 |
| RMH10-F-s1 - P2H10-F-s1 | all           | replay only %           | 12146 |  0.84 |  1.33 |  -0.49 | -0.64 | -0.36 |
| RMH10-F-s1 - P2H10-F-s1 | all           | raw only %              | 12146 |  1.51 |  1.21 |   0.30 |  0.14 |  0.50 |
| RMH10-F-s1 - P2H10-F-s1 | turn > 20 deg | replay out (DAC fail) % |  3154 |  7.70 |  8.78 |  -1.08 | -1.60 | -0.56 |
| RMH10-F-s1 - P2H10-F-s1 | turn > 20 deg | raw plan out %          |  3154 |  9.07 |  8.31 |   0.76 |  0.23 |  1.34 |
| RMH10-F-s1 - P2H10-F-s1 | turn > 20 deg | replay only %           |  3154 |  2.25 |  3.33 |  -1.08 | -1.54 | -0.68 |
| RMH10-F-s1 - P2H10-F-s1 | turn > 20 deg | raw only %              |  3154 |  3.61 |  2.85 |   0.76 |  0.22 |  1.36 |
| RMH10-F-s1 - P2H10-F-s1 | > 45 deg      | replay out (DAC fail) % |  1517 |  9.36 | 10.15 |  -0.79 | -1.54 | -0.06 |
| RMH10-F-s1 - P2H10-F-s1 | > 45 deg      | raw plan out %          |  1517 | 12.46 | 11.34 |   1.12 |  0.42 |  1.89 |
| RMH10-F-s1 - P2H10-F-s1 | > 45 deg      | replay only %           |  1517 |  2.31 |  3.03 |  -0.73 | -1.29 | -0.15 |
| RMH10-F-s1 - P2H10-F-s1 | > 45 deg      | raw only %              |  1517 |  5.41 |  4.22 |   1.19 |  0.32 |  2.15 |

CI: percentile bootstrap

footprint departures from the scorer's drivable polygons, devkit replay vs raw plan; paired, cluster bootstrap over navtest logs, B 10 000
