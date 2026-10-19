# BEST-OF-N gate: FAIL

Arena held split native (no hold, 2246 windows, `jt.sh aread <tag> off`), paired on the common windows without a route-fail, 95 % CI clustered by log (jevdrive.stats). Held 460 @18: `rm_chain.sh read`, paired session-clustered (defect_fix df.cl_read).

- arena `BNX2K-L18-s1` - `BNC2K-L18-s1`: **-0.0197 [-0.0349, -0.0063]** (rule paired >= +0.01 and CI lo > 0: fail)
- held 460 zeros 36 vs control 33 (rule zeros <= control + 2: fail)

| pair | stratum | windows | tag mean | ref mean | paired [CI] | zeros tag / ref | new / fixed zeros |
|:--|:--|--:|--:|--:|:--|--:|--:|
| BNX2K-L18-s1 vs BNC2K-L18-s1 | all | 2228 | 0.8855 | 0.9052 | -0.0197 [-0.0349, -0.0063] | 224 / 174 | 60 / 10 |
| BNX2K-L18-s1 vs BNC2K-L18-s1 | static | 878 | 0.8303 | 0.8895 | -0.0592 [-0.0942, -0.0268] | 149 / 97 | 53 / 1 |
| BNX2K-L18-s1 vs BNC2K-L18-s1 | launch | 200 | 0.8726 | 0.8460 | +0.0266 [+0.0130, +0.0434] | 10 / 11 | 0 / 1 |
| BNX2K-L18-s1 vs BNC2K-L18-s1 | braking | 372 | 0.9264 | 0.9234 | +0.0030 [-0.0109, +0.0164] | 25 / 26 | 2 / 3 |
| BNX2K-L18-s1 vs BNC2K-L18-s1 | turn | 213 | 0.8894 | 0.8980 | -0.0086 [-0.0265, +0.0098] | 21 / 19 | 4 / 2 |
| BNX2K-L18-s1 vs BNC2K-L18-s1 | cruise | 116 | 0.9806 | 0.9778 | +0.0029 [+0.0005, +0.0055] | 0 / 0 | 0 / 0 |
| BNX2K-L18-s1 vs BNC2K-L18-s1 | other | 449 | 0.9391 | 0.9319 | +0.0072 [+0.0015, +0.0136] | 19 / 21 | 1 / 3 |
| BNX2K-L18-s1 vs JTN-L18-s1 | all | 2228 | 0.8855 | 0.8909 | -0.0054 [-0.0160, +0.0059] | 224 / 210 | 37 / 23 |
| BNX2K-L18-s1 vs JTN-L18-s1 | static | 878 | 0.8303 | 0.8474 | -0.0171 [-0.0462, +0.0123] | 149 / 134 | 28 / 13 |
| BNX2K-L18-s1 vs JTN-L18-s1 | launch | 200 | 0.8726 | 0.8564 | +0.0162 [+0.0062, +0.0259] | 10 / 11 | 0 / 1 |
| BNX2K-L18-s1 vs JTN-L18-s1 | braking | 372 | 0.9264 | 0.9219 | +0.0045 [-0.0074, +0.0165] | 25 / 27 | 1 / 3 |
| BNX2K-L18-s1 vs JTN-L18-s1 | turn | 213 | 0.8894 | 0.8983 | -0.0089 [-0.0303, +0.0128] | 21 / 19 | 6 / 4 |
| BNX2K-L18-s1 vs JTN-L18-s1 | cruise | 116 | 0.9806 | 0.9836 | -0.0030 [-0.0049, -0.0012] | 0 / 0 | 0 / 0 |
| BNX2K-L18-s1 vs JTN-L18-s1 | other | 449 | 0.9391 | 0.9383 | +0.0008 [-0.0068, +0.0064] | 19 / 19 | 2 / 2 |
| BNC2K-L18-s1 vs JTN-L18-s1 | all | 2228 | 0.9052 | 0.8909 | +0.0143 [+0.0002, +0.0345] | 174 / 210 | 11 / 47 |
| BNC2K-L18-s1 vs JTN-L18-s1 | static | 878 | 0.8895 | 0.8474 | +0.0421 [+0.0094, +0.0871] | 97 / 134 | 4 / 41 |
| BNC2K-L18-s1 vs JTN-L18-s1 | launch | 200 | 0.8460 | 0.8564 | -0.0103 [-0.0294, +0.0065] | 11 / 11 | 1 / 1 |
| BNC2K-L18-s1 vs JTN-L18-s1 | braking | 372 | 0.9234 | 0.9219 | +0.0014 [-0.0074, +0.0114] | 26 / 27 | 1 / 2 |
| BNC2K-L18-s1 vs JTN-L18-s1 | turn | 213 | 0.8980 | 0.8983 | -0.0003 [-0.0124, +0.0121] | 19 / 19 | 3 / 3 |
| BNC2K-L18-s1 vs JTN-L18-s1 | cruise | 116 | 0.9778 | 0.9836 | -0.0059 [-0.0100, -0.0022] | 0 / 0 | 0 / 0 |
| BNC2K-L18-s1 vs JTN-L18-s1 | other | 449 | 0.9319 | 0.9383 | -0.0064 [-0.0142, -0.0010] | 21 / 19 | 2 / 0 |

Held 460 @18:

| tag | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired vs BNC2K-L18-s1 | paired vs JTN-L18-s1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|
| BNX2K-L18-s1 | 36 | 0.8655 | 6 | 14 | 23 | 5 / 1 / 31 | 0.797 | 14 / 3 | +0.0023 [-0.0115, +0.0207] | -0.0010 [-0.0184, +0.0219] |
| BNC2K-L18-s1 | 33 | 0.8632 | 4 | 16 | 22 | 3 / 1 / 37 | 0.788 | 16 / 4 |  | -0.0033 [-0.0224, +0.0193] |
