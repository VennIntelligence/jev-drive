Pool paimix: 75 rows, 275 scenes. Map: PCS = 135.8 x ability + 1767; 100 bootstrap draws.

| subject | scenes scored | local mean | zeros | ability | PCS | 95 % CI | vs NTV0-s0 (paired) | 95 % CI | P(> 0) |
|---|--:|--:|--:|--:|--:|---|--:|---|--:|
| NTV0-s0+AG (reference) | 275 | 0.743 | 58 | +3.15 | **2195** | [2138, 2299] | | | |
| R3MV-s0+AG | 274 | 0.741 | 61 | +1.11 | **1918** | [1849, 1960] | -276 (point -277) | [-375, -217] | 0.00 |
| DAG-s0+AG | 275 | 0.772 | 53 | +1.59 | **1983** | [1907, 2122] | -210 (point -213) | [-305, -57] | 0.00 |
| DAG-s1+AG | 275 | 0.813 | 43 | +2.01 | **2040** | [1965, 2224] | -153 (point -155) | [-260, +36] | 0.06 |
| R7J-s0+AG | 275 | 0.784 | 52 | +2.03 | **2043** | [1961, 2149] | -152 (point -153) | [-265, -35] | 0.01 |
| R7J-s1+AG | 274 | 0.779 | 54 | +2.09 | **2050** | [1971, 2179] | -138 (point -145) | [-246, -6] | 0.03 |
| R8K-s0+AG | 275 | 0.790 | 50 | +1.82 | **2014** | [1951, 2188] | -181 (point -181) | [-275, -24] | 0.02 |
| R8K-s1+AG | 275 | 0.799 | 47 | +2.01 | **2039** | [1963, 2199] | -156 (point -156) | [-250, +32] | 0.06 |

Paired differences between the new reads (same draws): R8K-s0+AG - DAG-s0+AG +25 [-77, +164]; R8K-s1+AG - DAG-s1+AG -1 [-69, +80]

## PCS value of each loss class if fixed (scenes -> 1, items fixed, MAP; paimix)

Class = pai_fix.py `fclass` of the stored facts. Read the ranking, not the sizes: the ceiling sits above the anchors.

| class | NTV0-s0+AG | R3MV-s0+AG | DAG-s0+AG | DAG-s1+AG | R7J-s0+AG | R7J-s1+AG | R8K-s0+AG | R8K-s1+AG |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| partial rear-hit | 25 sc, 10.4 pt: **+168** | 24 sc, 9.5 pt: **+20** | 22 sc, 8.7 pt: **+5** | 20 sc, 7.7 pt: **+21** | 17 sc, 6.4 pt: **+25** | 17 sc, 5.9 pt: **+44** | 17 sc, 7.2 pt: **+14** | 20 sc, 7.9 pt: **+30** |
| partial slow | 11 sc, 2.2 pt: **+52** | 5 sc, 0.6 pt: **+15** | 6 sc, 0.9 pt: **+79** | 6 sc, 0.8 pt: **+45** | 6 sc, 1.1 pt: **+15** | 3 sc, 0.7 pt: **+1** | 4 sc, 0.7 pt: **+42** | 3 sc, 0.5 pt: **+60** |
| partial other contact | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** |
| zero overshoot (passes the human's rest point) | 22 sc, 22.0 pt: **+38** | 22 sc, 22.0 pt: **+12** | 20 sc, 20.0 pt: **+7** | 19 sc, 19.0 pt: **+8** | 22 sc, 22.0 pt: **+33** | 20 sc, 20.0 pt: **+27** | 19 sc, 19.0 pt: **+8** | 19 sc, 19.0 pt: **+9** |
| zero route-turn | 15 sc, 15.0 pt: **+28** | 2 sc, 2.0 pt: **+0** | 2 sc, 2.0 pt: **+0** | 2 sc, 2.0 pt: **+1** | 3 sc, 3.0 pt: **+2** | 2 sc, 2.0 pt: **+2** | 3 sc, 3.0 pt: **+3** | 3 sc, 3.0 pt: **+4** |
| zero offroad | 12 sc, 12.0 pt: **+22** | 19 sc, 19.0 pt: **+21** | 20 sc, 20.0 pt: **+16** | 12 sc, 12.0 pt: **+12** | 16 sc, 16.0 pt: **+40** | 22 sc, 22.0 pt: **+88** | 17 sc, 17.0 pt: **+48** | 16 sc, 16.0 pt: **+32** |
| zero other (AF-front, side, drift / lane change) | 9 sc, 9.0 pt: **+19** | 18 sc, 18.0 pt: **+25** | 11 sc, 11.0 pt: **+4** | 10 sc, 10.0 pt: **+6** | 11 sc, 11.0 pt: **+22** | 10 sc, 10.0 pt: **+23** | 11 sc, 11.0 pt: **+12** | 9 sc, 9.0 pt: **+10** |
| all partials | 36 sc, 12.6 pt: **+508** | 29 sc, 10.1 pt: **+49** | 28 sc, 9.6 pt: **+167** | 26 sc, 8.4 pt: **+113** | 23 sc, 7.5 pt: **+69** | 20 sc, 6.6 pt: **+46** | 21 sc, 7.9 pt: **+108** | 23 sc, 8.4 pt: **+337** |
| all zeros | 58 sc, 58.0 pt: **+117** | 61 sc, 61.0 pt: **+61** | 53 sc, 53.0 pt: **+28** | 43 sc, 43.0 pt: **+27** | 52 sc, 52.0 pt: **+111** | 54 sc, 54.0 pt: **+156** | 50 sc, 50.0 pt: **+77** | 47 sc, 47.0 pt: **+58** |

## Pool sensitivity (PCS)

| variant | anchor RMSE | NTV0-s0+AG | R3MV-s0+AG | DAG-s0+AG | DAG-s1+AG | R7J-s0+AG | R7J-s1+AG | R8K-s0+AG | R8K-s1+AG | garage | go-straight | RSG+AG | R3MV - NTV0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paimix | 25 | 2195 | 1918 | 1983 | 2040 | 2043 | 2050 | 2014 | 2039 | 1783 | 1464 | 2107 | -277 |
| paimix-s1 | 8 | 2164 | 1945 | 2016 | 2061 | 2036 | 2049 | 2027 | 2049 | 1808 | 1451 | 2095 | -220 |
| paimix-s2 | 10 | 2176 | 1920 | 2005 | 2059 | 2019 | 2033 | 2017 | 2039 | 1804 | 1453 | 2097 | -256 |
| garagex0.5 | 16 | 2181 | 1925 | 1993 | 2034 | 2033 | 2054 | 2006 | 2030 | 1796 | 1457 | 2101 | -256 |
| garagex2 | 39 | 2188 | 1919 | 1971 | 2047 | 2059 | 2066 | 2027 | 2052 | 1765 | 1476 | 2114 | -269 |
| bandx0.5 | 8 | 2158 | 1909 | 1958 | 2038 | 2067 | 2067 | 2036 | 2071 | 1808 | 1451 | 2095 | -250 |
| bandx2 | 55 | 2207 | 1871 | 1956 | 2015 | 2007 | 2027 | 1967 | 1985 | 1741 | 1493 | 2120 | -335 |
| gsx0.5 | 57 | 2221 | 1900 | 1969 | 2034 | 2036 | 2045 | 1999 | 2018 | 1740 | 1494 | 2120 | -321 |
| gsx2 | 2 | 2149 | 1907 | 1951 | 2020 | 2046 | 2049 | 2015 | 2045 | 1816 | 1447 | 2091 | -242 |
| low-as-gs | 2 | 2135 | 1900 | 1940 | 2012 | 2054 | 2057 | 2017 | 2050 | 1822 | 1445 | 2087 | -235 |
| dedup | 73 | 2260 | 1884 | 1989 | 2100 | 2054 | 2087 | 2043 | 2075 | 1717 | 1514 | 2123 | -376 |
| paimix-1pl | 16 | 2130 | 2125 | 2155 | 2225 | 2223 | 2219 | 2203 | 2211 | 1797 | 1457 | 2101 | -5 |

## The 12 most discriminating items (a >= 4.2): scores of the reads and of the 13 pool band reads

| scene | a | b | NTV0-s0 | R3MV-s0 | DAG-s0 | DAG-s1 | R7J-s0 | R7J-s1 | R8K-s0 | R8K-s1 | R3MV class | band reads scoring 1 (of 13) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| 2fe53681 | 4.21 | -0.6 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | zero-other | 13 |
| be932173 | 4.19 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| cf451fc1 | 4.19 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| e2524a02 | 4.18 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| c090153b | 4.18 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 64b0bb90 | 4.17 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 04b6e94a | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| bf8a05a6 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 56b32a73 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| b8086e1c | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | ok | 13 |
| fc8e80a9 | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 42eda85c | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |

90 items have a >= 3. Scored 0 on them: NTV0-s0+AG 0, R3MV-s0+AG 13, DAG-s0+AG 2, DAG-s1+AG 0, R7J-s0+AG 3, R7J-s1+AG 3, R8K-s0+AG 3, R8K-s1+AG 1; mean score on them: NTV0-s0+AG 1.000, R3MV-s0+AG 0.846, DAG-s0+AG 0.976, DAG-s1+AG 1.000, R7J-s0+AG 0.967, R7J-s1+AG 0.967, R8K-s0+AG 0.967, R8K-s1+AG 0.989.

## Where the ruler looks: items by discrimination a, and what perfection on the high-a items is worth

| read | a >= 3 (zeros / partials / ones) | 1 <= a < 3 | a < 1 | PCS if every imperfect a >= 3 item -> 1 | if every imperfect a >= 1 item -> 1 |
|---|---|---|---|--:|--:|
| NTV0-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 0 / 1 / 25 | 159 items: 58 / 35 / 66 | +0 | +12 |
| R3MV-s0+AG | 90 items: 13 / 4 / 73 | 26 items: 3 / 2 / 21 | 159 items: 45 / 24 / 89 | +164 | +341 |
| DAG-s0+AG | 90 items: 2 / 1 / 87 | 26 items: 3 / 4 / 19 | 159 items: 48 / 26 / 85 | +55 | +249 |
| DAG-s1+AG | 90 items: 0 / 1 / 89 | 26 items: 0 / 5 / 21 | 159 items: 43 / 22 / 94 | +19 | +225 |
| R7J-s0+AG | 90 items: 3 / 0 / 87 | 26 items: 4 / 4 / 18 | 159 items: 45 / 20 / 94 | +32 | +271 |
| R7J-s1+AG | 90 items: 3 / 0 / 87 | 26 items: 5 / 2 / 19 | 159 items: 46 / 19 / 93 | +47 | +256 |
| R8K-s0+AG | 90 items: 3 / 0 / 87 | 26 items: 4 / 3 / 19 | 159 items: 43 / 19 / 97 | +29 | +281 |
| R8K-s1+AG | 90 items: 1 / 0 / 89 | 26 items: 4 / 2 / 20 | 159 items: 42 / 21 / 96 | +9 | +200 |

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
| DAG-s0+AG | 300a909c | 3.06 | overshoot | 0.00 | 0.00 |
| DAG-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R7J-s0+AG | 300a909c | 3.06 | overshoot | 0.00 | 0.00 |
| R7J-s0+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R7J-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R7J-s1+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R7J-s1+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R7J-s1+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R8K-s0+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R8K-s0+AG | b8086e1c | 4.15 | offroad | 1.00 | 0.00 |
| R8K-s0+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R8K-s1+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
