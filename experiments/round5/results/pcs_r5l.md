Pool paimix: 75 rows, 275 scenes. Map: PCS = 135.8 x ability + 1767; 100 bootstrap draws.

| subject | scenes scored | local mean | zeros | ability | PCS | 95 % CI | vs NTV0-s0 (paired) | 95 % CI | P(> 0) |
|---|--:|--:|--:|--:|--:|---|--:|---|--:|
| NTV0-s0+AG (reference) | 275 | 0.743 | 58 | +3.15 | **2195** | [2138, 2299] | | | |
| R2C-s0+AG | 275 | 0.740 | 59 | +2.90 | **2161** | [2089, 2239] | -29 (point -34) | [-95, +19] | 0.17 |
| R2J-s0+AG | 275 | 0.687 | 63 | +0.87 | **1885** | [1797, 1938] | -307 (point -310) | [-420, -227] | 0.00 |
| R3MV-s0+AG | 274 | 0.741 | 61 | +1.11 | **1918** | [1849, 1960] | -276 (point -277) | [-375, -217] | 0.00 |
| R3MV-s1+AG | 273 | 0.714 | 69 | +1.09 | **1915** | [1828, 1967] | -282 (point -280) | [-384, -225] | 0.00 |
| R5LC-s0+AG | 274 | 0.745 | 61 | +1.21 | **1931** | [1859, 1992] | -266 (point -264) | [-366, -189] | 0.00 |
| R5LC-s1+AG | 275 | 0.721 | 68 | +1.01 | **1905** | [1823, 1959] | -285 (point -291) | [-394, -208] | 0.00 |

Paired differences between the new reads (same draws): R5LC-s0+AG - R3MV-s0+AG +17 [-25, +62]; R5LC-s1+AG - R3MV-s1+AG -5 [-50, +36]; R3MV-s1+AG - R3MV-s0+AG -2 [-40, +20]; R5LC-s1+AG - R5LC-s0+AG -26 [-68, +41]; R5LC-s0+AG - R2C-s0+AG -234 [-334, -149]; R5LC-s1+AG - R2C-s0+AG -260 [-356, -154]

## PCS value of each loss class if fixed (scenes -> 1, items fixed, MAP; paimix)

Class = pai_fix.py `fclass` of the stored facts. Read the ranking, not the sizes: the ceiling sits above the anchors.

| class | NTV0-s0+AG | R2C-s0+AG | R2J-s0+AG | R3MV-s0+AG | R3MV-s1+AG | R5LC-s0+AG | R5LC-s1+AG |
|---|--:|--:|--:|--:|--:|--:|--:|
| partial rear-hit | 25 sc, 10.4 pt: **+168** | 25 sc, 10.8 pt: **+101** | 33 sc, 19.1 pt: **+31** | 24 sc, 9.5 pt: **+20** | 21 sc, 8.7 pt: **+9** | 18 sc, 7.7 pt: **-1** | 22 sc, 8.3 pt: **+39** |
| partial slow | 11 sc, 2.2 pt: **+52** | 10 sc, 1.7 pt: **+33** | 12 sc, 4.1 pt: **+25** | 5 sc, 0.6 pt: **+15** | 4 sc, 0.4 pt: **+13** | 7 sc, 1.1 pt: **+56** | 5 sc, 0.5 pt: **+4** |
| partial other contact | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** |
| zero overshoot (passes the human's rest point) | 22 sc, 22.0 pt: **+38** | 21 sc, 21.0 pt: **+34** | 21 sc, 21.0 pt: **+6** | 22 sc, 22.0 pt: **+12** | 20 sc, 20.0 pt: **+9** | 21 sc, 21.0 pt: **+9** | 22 sc, 22.0 pt: **+7** |
| zero route-turn | 15 sc, 15.0 pt: **+28** | 15 sc, 15.0 pt: **+46** | 2 sc, 2.0 pt: **+0** | 2 sc, 2.0 pt: **+0** | 3 sc, 3.0 pt: **+1** | 3 sc, 3.0 pt: **+1** | 4 sc, 4.0 pt: **+1** |
| zero offroad | 12 sc, 12.0 pt: **+22** | 11 sc, 11.0 pt: **+28** | 20 sc, 20.0 pt: **+17** | 19 sc, 19.0 pt: **+21** | 21 sc, 21.0 pt: **+27** | 20 sc, 20.0 pt: **+22** | 24 sc, 24.0 pt: **+28** |
| zero other (AF-front, side, drift / lane change) | 9 sc, 9.0 pt: **+19** | 12 sc, 12.0 pt: **+46** | 20 sc, 20.0 pt: **+15** | 18 sc, 18.0 pt: **+25** | 25 sc, 25.0 pt: **+42** | 17 sc, 17.0 pt: **+15** | 18 sc, 18.0 pt: **+14** |
| all partials | 36 sc, 12.6 pt: **+508** | 35 sc, 12.5 pt: **+277** | 45 sc, 23.1 pt: **+96** | 29 sc, 10.1 pt: **+49** | 25 sc, 9.1 pt: **+27** | 25 sc, 8.8 pt: **+64** | 27 sc, 8.8 pt: **+54** |
| all zeros | 58 sc, 58.0 pt: **+117** | 59 sc, 59.0 pt: **+177** | 63 sc, 63.0 pt: **+37** | 61 sc, 61.0 pt: **+61** | 69 sc, 69.0 pt: **+89** | 61 sc, 61.0 pt: **+49** | 68 sc, 68.0 pt: **+49** |

## Pool sensitivity (PCS)

| variant | anchor RMSE | NTV0-s0+AG | R2C-s0+AG | R2J-s0+AG | R3MV-s0+AG | R3MV-s1+AG | R5LC-s0+AG | R5LC-s1+AG | garage | go-straight | RSG+AG | R3MV - NTV0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paimix | 25 | 2195 | 2161 | 1885 | 1918 | 1915 | 1931 | 1905 | 1783 | 1464 | 2107 | -277 |
| paimix-s1 | 8 | 2164 | 2145 | 1931 | 1945 | 1942 | 1962 | 1943 | 1808 | 1451 | 2095 | -220 |
| paimix-s2 | 10 | 2176 | 2149 | 1913 | 1920 | 1921 | 1942 | 1920 | 1804 | 1453 | 2097 | -256 |
| garagex0.5 | 16 | 2181 | 2147 | 1905 | 1925 | 1921 | 1938 | 1919 | 1796 | 1457 | 2101 | -256 |
| garagex2 | 39 | 2188 | 2170 | 1870 | 1919 | 1914 | 1924 | 1895 | 1765 | 1476 | 2114 | -269 |
| bandx0.5 | 8 | 2158 | 2146 | 1856 | 1909 | 1902 | 1912 | 1878 | 1808 | 1451 | 2095 | -250 |
| bandx2 | 55 | 2207 | 2148 | 1858 | 1871 | 1871 | 1891 | 1870 | 1741 | 1493 | 2120 | -335 |
| gsx0.5 | 57 | 2221 | 2156 | 1866 | 1900 | 1897 | 1916 | 1890 | 1740 | 1494 | 2120 | -321 |
| gsx2 | 2 | 2149 | 2128 | 1864 | 1907 | 1904 | 1910 | 1882 | 1816 | 1447 | 2091 | -242 |
| low-as-gs | 2 | 2135 | 2119 | 1855 | 1900 | 1896 | 1901 | 1872 | 1822 | 1445 | 2087 | -235 |
| dedup | 73 | 2260 | 2179 | 1854 | 1884 | 1872 | 1912 | 1876 | 1717 | 1514 | 2123 | -376 |
| paimix-1pl | 16 | 2130 | 2117 | 2015 | 2125 | 2123 | 2152 | 2122 | 1797 | 1457 | 2101 | -5 |

## The 12 most discriminating items (a >= 4.2): scores of the reads and of the 13 pool band reads

| scene | a | b | NTV0-s0 | R2C-s0 | R2J-s0 | R3MV-s0 | R3MV-s1 | R5LC-s0 | R5LC-s1 | R3MV class | band reads scoring 1 (of 13) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| 2fe53681 | 4.21 | -0.6 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | zero-other | 13 |
| be932173 | 4.19 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| cf451fc1 | 4.19 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| e2524a02 | 4.18 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | ok | 13 |
| c090153b | 4.18 | -0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | ok | 13 |
| 64b0bb90 | 4.17 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 04b6e94a | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| bf8a05a6 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 56b32a73 | 4.16 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| b8086e1c | 4.15 | -0.6 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | ok | 13 |
| fc8e80a9 | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | 13 |
| 42eda85c | 4.15 | -0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | ok | 13 |

90 items have a >= 3. Scored 0 on them: NTV0-s0+AG 0, R2C-s0+AG 0, R2J-s0+AG 11, R3MV-s0+AG 13, R3MV-s1+AG 17, R5LC-s0+AG 10, R5LC-s1+AG 13; mean score on them: NTV0-s0+AG 1.000, R2C-s0+AG 1.000, R2J-s0+AG 0.861, R3MV-s0+AG 0.846, R3MV-s1+AG 0.806, R5LC-s0+AG 0.885, R5LC-s1+AG 0.841.

## Where the ruler looks: items by discrimination a, and what perfection on the high-a items is worth

| read | a >= 3 (zeros / partials / ones) | 1 <= a < 3 | a < 1 | PCS if every imperfect a >= 3 item -> 1 | if every imperfect a >= 1 item -> 1 |
|---|---|---|---|--:|--:|
| NTV0-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 0 / 1 / 25 | 159 items: 58 / 35 / 66 | +0 | +12 |
| R2C-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 4 / 0 / 22 | 159 items: 55 / 35 / 69 | +0 | +55 |
| R2J-s0+AG | 90 items: 11 / 6 / 73 | 26 items: 5 / 5 / 16 | 159 items: 47 / 37 / 75 | +114 | +182 |
| R3MV-s0+AG | 90 items: 13 / 4 / 73 | 26 items: 3 / 2 / 21 | 159 items: 45 / 24 / 89 | +164 | +341 |
| R3MV-s1+AG | 90 items: 17 / 3 / 70 | 26 items: 3 / 3 / 20 | 159 items: 49 / 20 / 88 | +171 | +304 |
| R5LC-s0+AG | 90 items: 10 / 3 / 77 | 26 items: 3 / 3 / 19 | 159 items: 48 / 21 / 90 | +110 | +336 |
| R5LC-s1+AG | 90 items: 13 / 4 / 73 | 26 items: 5 / 1 / 20 | 159 items: 50 / 23 / 86 | +225 | +335 |

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
| R3MV-s1+AG | b64a6e11 | 4.11 | zero-other | 1.00 | 0.00 |
| R3MV-s1+AG | d641ffd0 | 3.05 | offroad | 0.00 | 0.00 |
| R3MV-s1+AG | c090153b | 4.18 | zero-other | 1.00 | 0.00 |
| R3MV-s1+AG | f4f7641f | 3.07 | overshoot | 0.00 | 0.00 |
| R3MV-s1+AG | 0357016f | 4.13 | zero-other | 1.00 | 0.00 |
| R3MV-s1+AG | e2524a02 | 4.18 | zero-other | 1.00 | 0.00 |
| R3MV-s1+AG | 8c3d15a9 | 3.10 | zero-other | 0.00 | 0.00 |
| R3MV-s1+AG | 55ea9803 | 3.06 | overshoot | 0.00 | 0.00 |
| R3MV-s1+AG | ef64a231 | 4.10 | offroad | 1.00 | 0.00 |
| R3MV-s1+AG | 2fe53681 | 4.21 | zero-other | 1.00 | 0.00 |
| R3MV-s1+AG | 9ea35e26 | 3.07 | side | 0.00 | 0.00 |
| R3MV-s1+AG | 48d0ba22 | 4.12 | offroad | 1.00 | 0.00 |
| R3MV-s1+AG | 5793734a | 3.10 | zero-other | 0.00 | 0.00 |
| R3MV-s1+AG | 7c104b76 | 3.03 | offroad | 0.00 | 0.00 |
| R3MV-s1+AG | 03e2290d | 3.08 | offroad | 0.00 | 0.00 |
| R3MV-s1+AG | a3bad8dd | 3.08 | offroad | 0.00 | 0.00 |
| R3MV-s1+AG | d76703f1 | 4.09 | zero-other | 1.00 | 0.00 |
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
