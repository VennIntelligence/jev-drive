# SIMP-F lead-break read (supplement 3; lead_break.py cf on RB0-F-s0's PAI crash states)

G0: PASS (p99 0.0591 m, rows 411)

## BRK: 324 decisions in 30 scenes

| model / arm | stays behind the lead (95 % scene CI) |
|:--|--:|
| NTV0-L18-s0/F | +0.182 [+0.124, +0.240] |
| R2C-L18-s0/F | +0.176 [+0.115, +0.236] |
| R2J-L18-s0/F | +0.250 [+0.177, +0.329] |
| R2JH-L18-s0/F | +0.256 [+0.186, +0.332] |
| RB0-F-s0/F | +0.148 [+0.095, +0.206] |
| RB0-F-s0/OFF | +0.614 [+0.494, +0.728] |

- R2JH-L18-s0 - R2C-L18-s0: +0.080 [+0.041, +0.128]; R2JH-L18-s0 - NTV0-L18-s0: +0.074 [+0.035, +0.121]; NTV0-L18-s0 - R2C-L18-s0: +0.006 [+0.000, +0.016]; share of the gap to RB0-F OFF closed: 0.18

## FOL: 87 decisions in 10 scenes

| model / arm | stays behind the lead (95 % scene CI) |
|:--|--:|
| NTV0-L18-s0/F | +0.966 [+0.859, +1.000] |
| R2C-L18-s0/F | +0.966 [+0.859, +1.000] |
| R2J-L18-s0/F | +0.954 [+0.845, +1.000] |
| R2JH-L18-s0/F | +0.966 [+0.859, +1.000] |
| RB0-F-s0/F | +0.966 [+0.859, +1.000] |
| RB0-F-s0/OFF | +1.000 [+1.000, +1.000] |

- R2JH-L18-s0 - R2C-L18-s0: +0.000 [+0.000, +0.000]; R2JH-L18-s0 - NTV0-L18-s0: +0.000 [+0.000, +0.000]; NTV0-L18-s0 - R2C-L18-s0: +0.000 [+0.000, +0.000]; share of the gap to RB0-F OFF closed: 0.00

Rule: BRK R2JH-L18-s0 - R2C-L18-s0 >= +0.1 with CI low > 0, and |FOL R2JH-L18-s0 - R2C-L18-s0| <= 0.05: FAIL
