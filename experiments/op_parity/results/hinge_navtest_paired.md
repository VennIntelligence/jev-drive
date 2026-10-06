| pair                 |   mean |    lo |    hi |   mean_a |   mean_b |     n |   units |   dDACfail pp |   dDACfail lo |   dDACfail hi |    dEP |   dDAC |    dNC |   dTTC |    dEC |
|:---------------------|-------:|------:|------:|---------:|---------:|------:|--------:|--------------:|--------------:|--------------:|-------:|-------:|-------:|-------:|-------:|
| P2H10 - P2           |   0.46 |  0.33 |  0.61 |    88.67 |    88.21 | 12146 |     136 |         -0.50 |         -0.64 |         -0.37 |   0.01 |   0.50 |   0.03 |   0.05 |   0.08 |
| P2H10 - WA-JEPA      |  -3.04 | -3.74 | -2.32 |    88.67 |    91.71 | 12146 |     136 |          2.03 |          1.37 |          2.76 |  -0.70 |  -2.03 |  -0.83 |  -0.93 |   0.57 |
| P2 - WA-JEPA         |  -3.50 | -4.24 | -2.76 |    88.21 |    91.71 | 12146 |     136 |          2.53 |          1.84 |          3.29 |  -0.72 |  -2.53 |  -0.86 |  -0.97 |   0.49 |
| P2H10-F-s0 - P2-F-s0 |   0.46 |  0.27 |  0.66 |    88.58 |    88.12 | 12146 |     136 |         -0.56 |         -0.75 |         -0.38 | nan    | nan    | nan    | nan    | nan    |
| P2H10-F-s1 - P2-F-s1 |   0.47 |  0.34 |  0.60 |    88.77 |    88.30 | 12146 |     136 |         -0.44 |         -0.57 |         -0.32 | nan    | nan    | nan    | nan    | nan    |

CI: percentile bootstrap

per-token EPDMS x 100 (seed means), cluster bootstrap over navtest logs, B 10000; dDACfail = DAC failure share diff in pp (negative = fewer failures); dX = subscore diffs x 100
