# SIMP-F lead-break read (supplement 3; lead_break.py cf on RB0-F-s0's PAI crash states)

G0: PASS (p99 0.0591 m, rows 411)

## BRK: 324 decisions in 30 scenes

| model / arm | stays behind the lead (95 % scene CI) |
|:--|--:|
| RB0-F-s0/F | +0.148 [+0.095, +0.206] |
| SIMPB1K-L18-s0/F | +0.247 [+0.186, +0.305] |
| SIMPF1K-L18-s0/F | +0.321 [+0.247, +0.389] |
| SIMPN1K-L18-s0/F | +0.170 [+0.112, +0.228] |
| RB0-F-s0/OFF | +0.614 [+0.494, +0.728] |

- SIMPF1K-L18-s0 - SIMPN1K-L18-s0: +0.151 [+0.111, +0.193]; SIMPF1K-L18-s0 - SIMPB1K-L18-s0: +0.074 [+0.043, +0.107]; SIMPB1K-L18-s0 - SIMPN1K-L18-s0: +0.077 [+0.053, +0.104]; share of the gap to RB0-F OFF closed: 0.34

## FOL: 87 decisions in 10 scenes

| model / arm | stays behind the lead (95 % scene CI) |
|:--|--:|
| RB0-F-s0/F | +0.966 [+0.859, +1.000] |
| SIMPB1K-L18-s0/F | +0.966 [+0.859, +1.000] |
| SIMPF1K-L18-s0/F | +0.966 [+0.859, +1.000] |
| SIMPN1K-L18-s0/F | +0.966 [+0.859, +1.000] |
| RB0-F-s0/OFF | +1.000 [+1.000, +1.000] |

- SIMPF1K-L18-s0 - SIMPN1K-L18-s0: +0.000 [+0.000, +0.000]; SIMPF1K-L18-s0 - SIMPB1K-L18-s0: +0.000 [+0.000, +0.000]; SIMPB1K-L18-s0 - SIMPN1K-L18-s0: +0.000 [+0.000, +0.000]; share of the gap to RB0-F OFF closed: 0.00

Rule: BRK SIMPF1K-L18-s0 - SIMPN1K-L18-s0 >= +0.1 with CI low > 0, and |FOL SIMPF1K-L18-s0 - SIMPN1K-L18-s0| <= 0.05: PASS
