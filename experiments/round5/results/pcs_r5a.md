Pool paimix: 75 rows, 275 scenes. Map: PCS = 135.8 x ability + 1767; 100 bootstrap draws.

| subject | scenes scored | local mean | zeros | ability | PCS | 95 % CI | vs NTV0-s0 (paired) | 95 % CI | P(> 0) |
|---|--:|--:|--:|--:|--:|---|--:|---|--:|
| NTV0-s0+AG (reference) | 275 | 0.743 | 58 | +3.15 | **2195** | [2138, 2299] | | | |
| R2C-s0+AG | 275 | 0.740 | 59 | +2.90 | **2161** | [2089, 2239] | -29 (point -34) | [-95, +19] | 0.17 |
| R2J-s0+AG | 275 | 0.687 | 63 | +0.87 | **1885** | [1797, 1938] | -307 (point -310) | [-420, -227] | 0.00 |
| R3MV-s0+AG | 274 | 0.741 | 61 | +1.11 | **1918** | [1849, 1960] | -276 (point -277) | [-375, -217] | 0.00 |
| R5A-s0+AG | 274 | 0.676 | 68 | +1.16 | **1925** | [1847, 1969] | -269 (point -270) | [-358, -210] | 0.00 |

Paired differences between the new reads (same draws): R5A-s0+AG - R3MV-s0+AG +10 [-39, +49]; R5A-s0+AG - R2C-s0+AG -241 [-326, -152]

## PCS value of each loss class if fixed (scenes -> 1, items fixed, MAP; paimix)

Class = pai_fix.py `fclass` of the stored facts. Read the ranking, not the sizes: the ceiling sits above the anchors.

| class | NTV0-s0+AG | R2C-s0+AG | R2J-s0+AG | R3MV-s0+AG | R5A-s0+AG |
|---|--:|--:|--:|--:|--:|
| partial rear-hit | 25 sc, 10.4 pt: **+168** | 25 sc, 10.8 pt: **+101** | 33 sc, 19.1 pt: **+31** | 24 sc, 9.5 pt: **+20** | 27 sc, 18.4 pt: **+38** |
| partial slow | 11 sc, 2.2 pt: **+52** | 10 sc, 1.7 pt: **+33** | 12 sc, 4.1 pt: **+25** | 5 sc, 0.6 pt: **+15** | 8 sc, 2.4 pt: **+12** |
| partial other contact | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** |
| zero overshoot (passes the human's rest point) | 22 sc, 22.0 pt: **+38** | 21 sc, 21.0 pt: **+34** | 21 sc, 21.0 pt: **+6** | 22 sc, 22.0 pt: **+12** | 20 sc, 20.0 pt: **+5** |
| zero route-turn | 15 sc, 15.0 pt: **+28** | 15 sc, 15.0 pt: **+46** | 2 sc, 2.0 pt: **+0** | 2 sc, 2.0 pt: **+0** | 6 sc, 6.0 pt: **+2** |
| zero offroad | 12 sc, 12.0 pt: **+22** | 11 sc, 11.0 pt: **+28** | 20 sc, 20.0 pt: **+17** | 19 sc, 19.0 pt: **+21** | 20 sc, 20.0 pt: **+23** |
| zero other (AF-front, side, drift / lane change) | 9 sc, 9.0 pt: **+19** | 12 sc, 12.0 pt: **+46** | 20 sc, 20.0 pt: **+15** | 18 sc, 18.0 pt: **+25** | 22 sc, 22.0 pt: **+25** |
| all partials | 36 sc, 12.6 pt: **+508** | 35 sc, 12.5 pt: **+277** | 45 sc, 23.1 pt: **+96** | 29 sc, 10.1 pt: **+49** | 35 sc, 20.8 pt: **+68** |
| all zeros | 58 sc, 58.0 pt: **+117** | 59 sc, 59.0 pt: **+177** | 63 sc, 63.0 pt: **+37** | 61 sc, 61.0 pt: **+61** | 68 sc, 68.0 pt: **+62** |

## Pool sensitivity (PCS)

| variant | anchor RMSE | NTV0-s0+AG | R2C-s0+AG | R2J-s0+AG | R3MV-s0+AG | R5A-s0+AG | garage | go-straight | RSG+AG | R3MV - NTV0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paimix | 25 | 2195 | 2161 | 1885 | 1918 | 1925 | 1783 | 1464 | 2107 | -277 |
| paimix-s1 | 8 | 2164 | 2145 | 1931 | 1945 | 1950 | 1808 | 1451 | 2095 | -220 |
| paimix-s2 | 10 | 2176 | 2149 | 1913 | 1920 | 1935 | 1804 | 1453 | 2097 | -256 |
| garagex0.5 | 16 | 2181 | 2147 | 1905 | 1925 | 1934 | 1796 | 1457 | 2101 | -256 |
| garagex2 | 39 | 2188 | 2170 | 1870 | 1919 | 1914 | 1765 | 1476 | 2114 | -269 |
| bandx0.5 | 8 | 2158 | 2146 | 1856 | 1909 | 1921 | 1808 | 1451 | 2095 | -250 |
| bandx2 | 55 | 2207 | 2148 | 1858 | 1871 | 1879 | 1741 | 1493 | 2120 | -335 |
| gsx0.5 | 57 | 2221 | 2156 | 1866 | 1900 | 1897 | 1740 | 1494 | 2120 | -321 |
| gsx2 | 2 | 2149 | 2128 | 1864 | 1907 | 1923 | 1816 | 1447 | 2091 | -242 |
| low-as-gs | 2 | 2135 | 2119 | 1855 | 1900 | 1915 | 1822 | 1445 | 2087 | -235 |
| dedup | 73 | 2260 | 2179 | 1854 | 1884 | 1876 | 1717 | 1514 | 2123 | -376 |
| paimix-1pl | 16 | 2130 | 2117 | 2015 | 2125 | 1994 | 1797 | 1457 | 2101 | -5 |

## The 12 most discriminating items (a >= 4.2): scores of the reads and of the 13 pool band reads

| scene | a | b | NTV0-s0 | R2C-s0 | R2J-s0 | R3MV-s0 | R5A-s0 | R3MV class | band reads scoring 1 (of 13) |
|---|--:|--:|--:|--:|--:|--:|--:|---|--:|
| 2fe53681 | 4.21 | -0.6 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | zero-other | 13 |
| be932173 | 4.19 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| cf451fc1 | 4.19 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| e2524a02 | 4.18 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| c090153b | 4.18 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 64b0bb90 | 4.17 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 04b6e94a | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| bf8a05a6 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 56b32a73 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| b8086e1c | 4.15 | -0.6 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | ok | 13 |
| fc8e80a9 | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 42eda85c | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |

90 items have a >= 3. Scored 0 on them: NTV0-s0+AG 0, R2C-s0+AG 0, R2J-s0+AG 11, R3MV-s0+AG 13, R5A-s0+AG 8; mean score on them: NTV0-s0+AG 1.000, R2C-s0+AG 1.000, R2J-s0+AG 0.861, R3MV-s0+AG 0.846, R5A-s0+AG 0.910.

## Where the ruler looks: items by discrimination a, and what perfection on the high-a items is worth

| read | a >= 3 (zeros / partials / ones) | 1 <= a < 3 | a < 1 | PCS if every imperfect a >= 3 item -> 1 | if every imperfect a >= 1 item -> 1 |
|---|---|---|---|--:|--:|
| NTV0-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 0 / 1 / 25 | 159 items: 58 / 35 / 66 | +0 | +12 |
| R2C-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 4 / 0 / 22 | 159 items: 55 / 35 / 69 | +0 | +55 |
| R2J-s0+AG | 90 items: 11 / 6 / 73 | 26 items: 5 / 5 / 16 | 159 items: 47 / 37 / 75 | +114 | +182 |
| R3MV-s0+AG | 90 items: 13 / 4 / 73 | 26 items: 3 / 2 / 21 | 159 items: 45 / 24 / 89 | +164 | +341 |
| R5A-s0+AG | 90 items: 8 / 1 / 81 | 26 items: 6 / 7 / 13 | 159 items: 54 / 31 / 73 | +48 | +146 |

Zeros on the a >= 3 items (all 13 pool band reads score 1 on nearly all of them), by cause, and whether the garage 0014 scores 1 there:

| read | scene | a | fclass | garage | go-straight |
|---|---|--:|---|--:|--:|
| R2J-s0+AG | d641ffd0 | 3.05 | offroad | 0.00 | 0.00 |
| R2J-s0+AG | 30d50172 | 4.15 | zero-other | 1.00 | 0.00 |
| R2J-s0+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R2J-s0+AG | 8c3d15a9 | 3.10 | zero-other | 0.00 | 0.00 |
| R2J-s0+AG | 55ea9803 | 3.06 | overshoot | 0.00 | 0.00 |
| R2J-s0+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R2J-s0+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R2J-s0+AG | 5793734a | 3.10 | zero-other | 0.00 | 0.00 |
| R2J-s0+AG | 4d3a871e | 3.11 | AF-front | 0.00 | 0.00 |
| R2J-s0+AG | b8086e1c | 4.15 | offroad | 1.00 | 0.00 |
| R2J-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R3MV-s0+AG | b64a6e11 | 4.11 | zero-other | 1.00 | 0.00 |
| R3MV-s0+AG | 300a909c | 3.06 | overshoot | 0.00 | 0.00 |
| R3MV-s0+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R3MV-s0+AG | 0357016f | 4.13 | zero-other | 1.00 | 0.00 |
| R3MV-s0+AG | 6c8ad043 | 4.13 | side | 1.00 | 0.00 |
| R3MV-s0+AG | 8c3d15a9 | 3.10 | zero-other | 0.00 | 0.00 |
| R3MV-s0+AG | 55ea9803 | 3.06 | overshoot | 0.00 | 0.00 |
| R3MV-s0+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R3MV-s0+AG | 2fe53681 | 4.21 | zero-other | 1.00 | 0.00 |
| R3MV-s0+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R3MV-s0+AG | 5793734a | 3.10 | zero-other | 0.00 | 0.00 |
| R3MV-s0+AG | 03e2290d | 3.08 | offroad | 0.00 | 0.00 |
| R3MV-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R5A-s0+AG | 30d50172 | 4.15 | zero-other | 1.00 | 0.00 |
| R5A-s0+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R5A-s0+AG | 2fe53681 | 4.21 | zero-other | 1.00 | 0.00 |
| R5A-s0+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R5A-s0+AG | 4d3a871e | 3.11 | AF-front | 0.00 | 0.00 |
| R5A-s0+AG | 03e2290d | 3.08 | offroad | 0.00 | 0.00 |
| R5A-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R5A-s0+AG | d76703f1 | 4.09 | zero-other | 1.00 | 0.00 |
