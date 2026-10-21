Pool paimix: 75 rows, 275 scenes. Map: PCS = 135.8 x ability + 1767; 100 bootstrap draws.

| subject | scenes scored | local mean | zeros | ability | PCS | 95 % CI | vs NTV0-s0 (paired) | 95 % CI | P(> 0) |
|---|--:|--:|--:|--:|--:|---|--:|---|--:|
| NTV0-s0+AG (reference) | 275 | 0.743 | 58 | +3.15 | **2195** | [2138, 2299] | | | |
| R2C-s0+AG | 275 | 0.740 | 59 | +2.90 | **2161** | [2089, 2239] | -29 (point -34) | [-95, +19] | 0.17 |
| R3MV-s0+AG | 274 | 0.741 | 61 | +1.11 | **1918** | [1849, 1960] | -276 (point -277) | [-375, -217] | 0.00 |
| R5LC-s0+AG | 274 | 0.745 | 61 | +1.21 | **1931** | [1859, 1992] | -266 (point -264) | [-366, -189] | 0.00 |
| R5LC-s1+AG | 275 | 0.721 | 68 | +1.01 | **1905** | [1823, 1959] | -285 (point -291) | [-394, -208] | 0.00 |
| R6REC-s0+AG | 275 | 0.728 | 66 | +1.04 | **1908** | [1837, 1959] | -288 (point -287) | [-377, -205] | 0.00 |
| R6REC-s1+AG | 275 | 0.726 | 66 | +1.36 | **1952** | [1893, 2005] | -242 (point -243) | [-336, -177] | 0.00 |

Paired differences between the new reads (same draws): R6REC-s0+AG - R5LC-s0+AG -22 [-72, +16]; R6REC-s1+AG - R5LC-s1+AG +43 [-9, +124]; R6REC-s1+AG - R6REC-s0+AG +39 [-1, +116]

## PCS value of each loss class if fixed (scenes -> 1, items fixed, MAP; paimix)

Class = pai_fix.py `fclass` of the stored facts. Read the ranking, not the sizes: the ceiling sits above the anchors.

| class | NTV0-s0+AG | R2C-s0+AG | R3MV-s0+AG | R5LC-s0+AG | R5LC-s1+AG | R6REC-s0+AG | R6REC-s1+AG |
|---|--:|--:|--:|--:|--:|--:|--:|
| partial rear-hit | 25 sc, 10.4 pt: **+168** | 25 sc, 10.8 pt: **+101** | 24 sc, 9.5 pt: **+20** | 18 sc, 7.7 pt: **-1** | 22 sc, 8.3 pt: **+39** | 21 sc, 8.3 pt: **+50** | 19 sc, 8.9 pt: **+24** |
| partial slow | 11 sc, 2.2 pt: **+52** | 10 sc, 1.7 pt: **+33** | 5 sc, 0.6 pt: **+15** | 7 sc, 1.1 pt: **+56** | 5 sc, 0.5 pt: **+4** | 5 sc, 0.5 pt: **+4** | 3 sc, 0.4 pt: **-3** |
| partial other contact | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** |
| zero overshoot (passes the human's rest point) | 22 sc, 22.0 pt: **+38** | 21 sc, 21.0 pt: **+34** | 22 sc, 22.0 pt: **+12** | 21 sc, 21.0 pt: **+9** | 22 sc, 22.0 pt: **+7** | 23 sc, 23.0 pt: **+11** | 23 sc, 23.0 pt: **+20** |
| zero route-turn | 15 sc, 15.0 pt: **+28** | 15 sc, 15.0 pt: **+46** | 2 sc, 2.0 pt: **+0** | 3 sc, 3.0 pt: **+1** | 4 sc, 4.0 pt: **+1** | 2 sc, 2.0 pt: **+0** | 2 sc, 2.0 pt: **+0** |
| zero offroad | 12 sc, 12.0 pt: **+22** | 11 sc, 11.0 pt: **+28** | 19 sc, 19.0 pt: **+21** | 20 sc, 20.0 pt: **+22** | 24 sc, 24.0 pt: **+28** | 26 sc, 26.0 pt: **+18** | 22 sc, 22.0 pt: **+33** |
| zero other (AF-front, side, drift / lane change) | 9 sc, 9.0 pt: **+19** | 12 sc, 12.0 pt: **+46** | 18 sc, 18.0 pt: **+25** | 17 sc, 17.0 pt: **+15** | 18 sc, 18.0 pt: **+14** | 15 sc, 15.0 pt: **+10** | 19 sc, 19.0 pt: **+34** |
| all partials | 36 sc, 12.6 pt: **+508** | 35 sc, 12.5 pt: **+277** | 29 sc, 10.1 pt: **+49** | 25 sc, 8.8 pt: **+64** | 27 sc, 8.8 pt: **+54** | 26 sc, 8.8 pt: **+70** | 22 sc, 9.3 pt: **+20** |
| all zeros | 58 sc, 58.0 pt: **+117** | 59 sc, 59.0 pt: **+177** | 61 sc, 61.0 pt: **+61** | 61 sc, 61.0 pt: **+49** | 68 sc, 68.0 pt: **+49** | 66 sc, 66.0 pt: **+38** | 66 sc, 66.0 pt: **+106** |

## Pool sensitivity (PCS)

| variant | anchor RMSE | NTV0-s0+AG | R2C-s0+AG | R3MV-s0+AG | R5LC-s0+AG | R5LC-s1+AG | R6REC-s0+AG | R6REC-s1+AG | garage | go-straight | RSG+AG | R3MV - NTV0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paimix | 25 | 2195 | 2161 | 1918 | 1931 | 1905 | 1908 | 1952 | 1783 | 1464 | 2107 | -277 |
| paimix-s1 | 8 | 2164 | 2145 | 1945 | 1962 | 1943 | 1944 | 1963 | 1808 | 1451 | 2095 | -220 |
| paimix-s2 | 10 | 2176 | 2149 | 1920 | 1942 | 1920 | 1920 | 1936 | 1804 | 1453 | 2097 | -256 |
| garagex0.5 | 16 | 2181 | 2147 | 1925 | 1938 | 1919 | 1924 | 1951 | 1796 | 1457 | 2101 | -256 |
| garagex2 | 39 | 2188 | 2170 | 1919 | 1924 | 1895 | 1900 | 1962 | 1765 | 1476 | 2114 | -269 |
| bandx0.5 | 8 | 2158 | 2146 | 1909 | 1912 | 1878 | 1887 | 1968 | 1808 | 1451 | 2095 | -250 |
| bandx2 | 55 | 2207 | 2148 | 1871 | 1891 | 1870 | 1872 | 1895 | 1741 | 1493 | 2120 | -335 |
| gsx0.5 | 57 | 2221 | 2156 | 1900 | 1916 | 1890 | 1893 | 1933 | 1740 | 1494 | 2120 | -321 |
| gsx2 | 2 | 2149 | 2128 | 1907 | 1910 | 1882 | 1888 | 1956 | 1816 | 1447 | 2091 | -242 |
| low-as-gs | 2 | 2135 | 2119 | 1900 | 1901 | 1872 | 1879 | 1959 | 1822 | 1445 | 2087 | -235 |
| dedup | 73 | 2260 | 2179 | 1884 | 1912 | 1876 | 1888 | 1913 | 1717 | 1514 | 2123 | -376 |
| paimix-1pl | 16 | 2130 | 2117 | 2125 | 2152 | 2122 | 2127 | 2137 | 1797 | 1457 | 2101 | -5 |

## The 12 most discriminating items (a >= 4.2): scores of the reads and of the 13 pool band reads

| scene | a | b | NTV0-s0 | R2C-s0 | R3MV-s0 | R5LC-s0 | R5LC-s1 | R6REC-s0 | R6REC-s1 | R3MV class | band reads scoring 1 (of 13) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| 2fe53681 | 4.21 | -0.6 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | zero-other | 13 |
| be932173 | 4.19 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| cf451fc1 | 4.19 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| e2524a02 | 4.18 | -0.6 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | ok | 13 |
| c090153b | 4.18 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 64b0bb90 | 4.17 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 04b6e94a | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| bf8a05a6 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 56b32a73 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| b8086e1c | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | ok | 13 |
| fc8e80a9 | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 42eda85c | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | ok | 13 |

90 items have a >= 3. Scored 0 on them: NTV0-s0+AG 0, R2C-s0+AG 0, R3MV-s0+AG 13, R5LC-s0+AG 10, R5LC-s1+AG 13, R6REC-s0+AG 9, R6REC-s1+AG 10; mean score on them: NTV0-s0+AG 1.000, R2C-s0+AG 1.000, R3MV-s0+AG 0.846, R5LC-s0+AG 0.885, R5LC-s1+AG 0.841, R6REC-s0+AG 0.887, R6REC-s1+AG 0.885.

## Where the ruler looks: items by discrimination a, and what perfection on the high-a items is worth

| read | a >= 3 (zeros / partials / ones) | 1 <= a < 3 | a < 1 | PCS if every imperfect a >= 3 item -> 1 | if every imperfect a >= 1 item -> 1 |
|---|---|---|---|--:|--:|
| NTV0-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 0 / 1 / 25 | 159 items: 58 / 35 / 66 | +0 | +12 |
| R2C-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 4 / 0 / 22 | 159 items: 55 / 35 / 69 | +0 | +55 |
| R3MV-s0+AG | 90 items: 13 / 4 / 73 | 26 items: 3 / 2 / 21 | 159 items: 45 / 24 / 89 | +164 | +341 |
| R5LC-s0+AG | 90 items: 10 / 3 / 77 | 26 items: 3 / 3 / 19 | 159 items: 48 / 21 / 90 | +110 | +336 |
| R5LC-s1+AG | 90 items: 13 / 4 / 73 | 26 items: 5 / 1 / 20 | 159 items: 50 / 23 / 86 | +225 | +335 |
| R6REC-s0+AG | 90 items: 9 / 5 / 76 | 26 items: 5 / 3 / 18 | 159 items: 52 / 20 / 87 | +147 | +300 |
| R6REC-s1+AG | 90 items: 10 / 1 / 79 | 26 items: 7 / 3 / 16 | 159 items: 49 / 20 / 90 | +106 | +300 |

Zeros on the a >= 3 items (all 13 pool band reads score 1 on nearly all of them), by cause, and whether the garage 0014 scores 1 there:

| read | scene | a | fclass | garage | go-straight |
|---|---|--:|---|--:|--:|
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
| R5LC-s0+AG | d641ffd0 | 3.05 | offroad | 0.00 | 0.00 |
| R5LC-s0+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R5LC-s0+AG | e2524a02 | 4.18 | zero-other | 1.00 | 0.00 |
| R5LC-s0+AG | 8c3d15a9 | 3.10 | zero-other | 0.00 | 0.00 |
| R5LC-s0+AG | 55ea9803 | 3.06 | overshoot | 0.00 | 0.00 |
| R5LC-s0+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R5LC-s0+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R5LC-s0+AG | 4d3a871e | 3.11 | AF-front | 0.00 | 0.00 |
| R5LC-s0+AG | 03e2290d | 3.08 | offroad | 0.00 | 0.00 |
| R5LC-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R5LC-s1+AG | d641ffd0 | 3.05 | offroad | 0.00 | 0.00 |
| R5LC-s1+AG | 30d50172 | 4.15 | zero-other | 1.00 | 0.00 |
| R5LC-s1+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R5LC-s1+AG | e2524a02 | 4.18 | zero-other | 1.00 | 0.00 |
| R5LC-s1+AG | ea8e056b | 4.10 | offroad | 1.00 | 0.00 |
| R5LC-s1+AG | 8c3d15a9 | 3.10 | zero-other | 0.00 | 0.00 |
| R5LC-s1+AG | 55ea9803 | 3.06 | overshoot | 0.00 | 0.00 |
| R5LC-s1+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R5LC-s1+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R5LC-s1+AG | b8086e1c | 4.15 | offroad | 1.00 | 0.00 |
| R5LC-s1+AG | 03e2290d | 3.08 | offroad | 0.00 | 0.00 |
| R5LC-s1+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R5LC-s1+AG | 42eda85c | 4.15 | offroad | 1.00 | 0.00 |
| R6REC-s0+AG | 300a909c | 3.06 | overshoot | 0.00 | 0.00 |
| R6REC-s0+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R6REC-s0+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R6REC-s0+AG | 9ea35e26 | 3.07 | AF-front | 0.00 | 0.00 |
| R6REC-s0+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R6REC-s0+AG | 5793734a | 3.10 | zero-other | 0.00 | 0.00 |
| R6REC-s0+AG | 5c9c5f59 | 3.04 | overshoot | 0.00 | 0.00 |
| R6REC-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R6REC-s0+AG | 82feecd0 | 3.13 | AF-front | 0.00 | 0.00 |
| R6REC-s1+AG | 300a909c | 3.06 | overshoot | 0.00 | 0.00 |
| R6REC-s1+AG | 30d50172 | 4.15 | zero-other | 1.00 | 0.00 |
| R6REC-s1+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R6REC-s1+AG | e2524a02 | 4.18 | zero-other | 1.00 | 0.00 |
| R6REC-s1+AG | 1a89244b | 4.12 | zero-other | 1.00 | 0.00 |
| R6REC-s1+AG | 55ea9803 | 3.06 | overshoot | 0.00 | 0.00 |
| R6REC-s1+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R6REC-s1+AG | 9ea35e26 | 3.07 | side | 0.00 | 0.00 |
| R6REC-s1+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R6REC-s1+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
