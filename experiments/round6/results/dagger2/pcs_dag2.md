Pool paimix: 75 rows, 275 scenes. Map: PCS = 135.8 x ability + 1767; 100 bootstrap draws.

| subject | scenes scored | local mean | zeros | ability | PCS | 95 % CI | vs NTV0-s0 (paired) | 95 % CI | P(> 0) |
|---|--:|--:|--:|--:|--:|---|--:|---|--:|
| NTV0-s0+AG (reference) | 275 | 0.743 | 58 | +3.15 | **2195** | [2138, 2299] | | | |
| R3MV-s0+AG | 274 | 0.741 | 61 | +1.11 | **1918** | [1849, 1960] | -276 (point -277) | [-375, -217] | 0.00 |
| R5LC-s0+AG | 274 | 0.745 | 61 | +1.21 | **1931** | [1859, 1992] | -266 (point -264) | [-366, -189] | 0.00 |
| R5LC-s1+AG | 275 | 0.721 | 68 | +1.01 | **1905** | [1823, 1959] | -285 (point -291) | [-394, -208] | 0.00 |
| DAG-s0+AG | 275 | 0.772 | 53 | +1.59 | **1983** | [1907, 2122] | -210 (point -213) | [-305, -57] | 0.00 |
| DAG-s1+AG | 275 | 0.813 | 43 | +2.01 | **2040** | [1965, 2224] | -153 (point -155) | [-260, +36] | 0.06 |
| DAG2-s0+AG | 275 | 0.786 | 50 | +1.78 | **2008** | [1952, 2109] | -183 (point -187) | [-277, -83] | 0.00 |
| DAG2-s1+AG | 275 | 0.792 | 50 | +1.50 | **1971** | [1908, 2066] | -225 (point -224) | [-326, -105] | 0.00 |

Paired differences between the new reads (same draws): DAG2-s0+AG - DAG-s0+AG +20 [-55, +115]; DAG2-s1+AG - DAG-s1+AG -72 [-198, -23]; DAG2-s1+AG - DAG2-s0+AG -39 [-121, +17]

## PCS value of each loss class if fixed (scenes -> 1, items fixed, MAP; paimix)

Class = pai_fix.py `fclass` of the stored facts. Read the ranking, not the sizes: the ceiling sits above the anchors.

| class | NTV0-s0+AG | R3MV-s0+AG | R5LC-s0+AG | R5LC-s1+AG | DAG-s0+AG | DAG-s1+AG | DAG2-s0+AG | DAG2-s1+AG |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| partial rear-hit | 25 sc, 10.4 pt: **+168** | 24 sc, 9.5 pt: **+20** | 18 sc, 7.7 pt: **-1** | 22 sc, 8.3 pt: **+39** | 22 sc, 8.7 pt: **+5** | 20 sc, 7.7 pt: **+21** | 20 sc, 7.6 pt: **+21** | 16 sc, 6.2 pt: **+16** |
| partial slow | 11 sc, 2.2 pt: **+52** | 5 sc, 0.6 pt: **+15** | 7 sc, 1.1 pt: **+56** | 5 sc, 0.5 pt: **+4** | 6 sc, 0.9 pt: **+79** | 6 sc, 0.8 pt: **+45** | 8 sc, 1.2 pt: **+44** | 7 sc, 1.1 pt: **+34** |
| partial other contact | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** |
| zero overshoot (passes the human's rest point) | 22 sc, 22.0 pt: **+38** | 22 sc, 22.0 pt: **+12** | 21 sc, 21.0 pt: **+9** | 22 sc, 22.0 pt: **+7** | 20 sc, 20.0 pt: **+7** | 19 sc, 19.0 pt: **+8** | 21 sc, 21.0 pt: **+14** | 20 sc, 20.0 pt: **+9** |
| zero route-turn | 15 sc, 15.0 pt: **+28** | 2 sc, 2.0 pt: **+0** | 3 sc, 3.0 pt: **+1** | 4 sc, 4.0 pt: **+1** | 2 sc, 2.0 pt: **+0** | 2 sc, 2.0 pt: **+1** | 2 sc, 2.0 pt: **+1** | 2 sc, 2.0 pt: **+0** |
| zero offroad | 12 sc, 12.0 pt: **+22** | 19 sc, 19.0 pt: **+21** | 20 sc, 20.0 pt: **+22** | 24 sc, 24.0 pt: **+28** | 20 sc, 20.0 pt: **+16** | 12 sc, 12.0 pt: **+12** | 18 sc, 18.0 pt: **+24** | 18 sc, 18.0 pt: **+24** |
| zero other (AF-front, side, drift / lane change) | 9 sc, 9.0 pt: **+19** | 18 sc, 18.0 pt: **+25** | 17 sc, 17.0 pt: **+15** | 18 sc, 18.0 pt: **+14** | 11 sc, 11.0 pt: **+4** | 10 sc, 10.0 pt: **+6** | 9 sc, 9.0 pt: **+14** | 10 sc, 10.0 pt: **+8** |
| all partials | 36 sc, 12.6 pt: **+508** | 29 sc, 10.1 pt: **+49** | 25 sc, 8.8 pt: **+64** | 27 sc, 8.8 pt: **+54** | 28 sc, 9.6 pt: **+167** | 26 sc, 8.4 pt: **+113** | 28 sc, 8.8 pt: **+161** | 23 sc, 7.3 pt: **+104** |
| all zeros | 58 sc, 58.0 pt: **+117** | 61 sc, 61.0 pt: **+61** | 61 sc, 61.0 pt: **+49** | 68 sc, 68.0 pt: **+49** | 53 sc, 53.0 pt: **+28** | 43 sc, 43.0 pt: **+27** | 50 sc, 50.0 pt: **+59** | 50 sc, 50.0 pt: **+44** |

## Pool sensitivity (PCS)

| variant | anchor RMSE | NTV0-s0+AG | R3MV-s0+AG | R5LC-s0+AG | R5LC-s1+AG | DAG-s0+AG | DAG-s1+AG | DAG2-s0+AG | DAG2-s1+AG | garage | go-straight | RSG+AG | R3MV - NTV0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paimix | 25 | 2195 | 1918 | 1931 | 1905 | 1983 | 2040 | 2008 | 1971 | 1783 | 1464 | 2107 | -277 |
| paimix-s1 | 8 | 2164 | 1945 | 1962 | 1943 | 2016 | 2061 | 2023 | 1995 | 1808 | 1451 | 2095 | -220 |
| paimix-s2 | 10 | 2176 | 1920 | 1942 | 1920 | 2005 | 2059 | 2011 | 1971 | 1804 | 1453 | 2097 | -256 |
| garagex0.5 | 16 | 2181 | 1925 | 1938 | 1919 | 1993 | 2034 | 2005 | 1975 | 1796 | 1457 | 2101 | -256 |
| garagex2 | 39 | 2188 | 1919 | 1924 | 1895 | 1971 | 2047 | 2017 | 1971 | 1765 | 1476 | 2114 | -269 |
| bandx0.5 | 8 | 2158 | 1909 | 1912 | 1878 | 1958 | 2038 | 2025 | 1963 | 1808 | 1451 | 2095 | -250 |
| bandx2 | 55 | 2207 | 1871 | 1891 | 1870 | 1956 | 2015 | 1967 | 1929 | 1741 | 1493 | 2120 | -335 |
| gsx0.5 | 57 | 2221 | 1900 | 1916 | 1890 | 1969 | 2034 | 1992 | 1957 | 1740 | 1494 | 2120 | -321 |
| gsx2 | 2 | 2149 | 1907 | 1910 | 1882 | 1951 | 2020 | 2008 | 1955 | 1816 | 1447 | 2091 | -242 |
| low-as-gs | 2 | 2135 | 1900 | 1901 | 1872 | 1940 | 2012 | 2010 | 1946 | 1822 | 1445 | 2087 | -235 |
| dedup | 73 | 2260 | 1884 | 1912 | 1876 | 1989 | 2100 | 2026 | 1981 | 1717 | 1514 | 2123 | -376 |
| paimix-1pl | 16 | 2130 | 2125 | 2152 | 2122 | 2155 | 2225 | 2159 | 2212 | 1797 | 1457 | 2101 | -5 |

## The 12 most discriminating items (a >= 4.2): scores of the reads and of the 13 pool band reads

| scene | a | b | NTV0-s0 | R3MV-s0 | R5LC-s0 | R5LC-s1 | DAG-s0 | DAG-s1 | DAG2-s0 | DAG2-s1 | R3MV class | band reads scoring 1 (of 13) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| 2fe53681 | 4.21 | -0.6 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | zero-other | 13 |
| be932173 | 4.19 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| cf451fc1 | 4.19 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| e2524a02 | 4.18 | -0.6 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| c090153b | 4.18 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 64b0bb90 | 4.17 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | ok | 13 |
| 04b6e94a | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| bf8a05a6 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 56b32a73 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| b8086e1c | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| fc8e80a9 | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 42eda85c | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |

90 items have a >= 3. Scored 0 on them: NTV0-s0+AG 0, R3MV-s0+AG 13, R5LC-s0+AG 10, R5LC-s1+AG 13, DAG-s0+AG 2, DAG-s1+AG 0, DAG2-s0+AG 3, DAG2-s1+AG 5; mean score on them: NTV0-s0+AG 1.000, R3MV-s0+AG 0.846, R5LC-s0+AG 0.885, R5LC-s1+AG 0.841, DAG-s0+AG 0.976, DAG-s1+AG 1.000, DAG2-s0+AG 0.967, DAG2-s1+AG 0.941.

## Where the ruler looks: items by discrimination a, and what perfection on the high-a items is worth

| read | a >= 3 (zeros / partials / ones) | 1 <= a < 3 | a < 1 | PCS if every imperfect a >= 3 item -> 1 | if every imperfect a >= 1 item -> 1 |
|---|---|---|---|--:|--:|
| NTV0-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 0 / 1 / 25 | 159 items: 58 / 35 / 66 | +0 | +12 |
| R3MV-s0+AG | 90 items: 13 / 4 / 73 | 26 items: 3 / 2 / 21 | 159 items: 45 / 24 / 89 | +164 | +341 |
| R5LC-s0+AG | 90 items: 10 / 3 / 77 | 26 items: 3 / 3 / 19 | 159 items: 48 / 21 / 90 | +110 | +336 |
| R5LC-s1+AG | 90 items: 13 / 4 / 73 | 26 items: 5 / 1 / 20 | 159 items: 50 / 23 / 86 | +225 | +335 |
| DAG-s0+AG | 90 items: 2 / 1 / 87 | 26 items: 3 / 4 / 19 | 159 items: 48 / 26 / 85 | +55 | +249 |
| DAG-s1+AG | 90 items: 0 / 1 / 89 | 26 items: 0 / 5 / 21 | 159 items: 43 / 22 / 94 | +19 | +225 |
| DAG2-s0+AG | 90 items: 3 / 0 / 87 | 26 items: 1 / 5 / 20 | 159 items: 46 / 24 / 89 | +25 | +185 |
| DAG2-s1+AG | 90 items: 5 / 2 / 83 | 26 items: 1 / 4 / 21 | 159 items: 44 / 20 / 95 | +88 | +350 |

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
| DAG-s0+AG | 300a909c | 3.06 | overshoot | 0.00 | 0.00 |
| DAG-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| DAG2-s0+AG | 8bfb501e | 4.11 | zero-other | 1.00 | 0.00 |
| DAG2-s0+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| DAG2-s0+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| DAG2-s1+AG | 947ee626 | 4.13 | zero-other | 1.00 | 0.00 |
| DAG2-s1+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| DAG2-s1+AG | 64b0bb90 | 4.17 | overshoot | 1.00 | 0.00 |
| DAG2-s1+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| DAG2-s1+AG | 03e2290d | 3.08 | offroad | 0.00 | 0.00 |
