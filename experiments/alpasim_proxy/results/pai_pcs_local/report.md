Pool paimix: 75 rows, 275 scenes (complete matrix over the pool). Band counts from the board: {'band': 32, 'garage': 35, 'gs': 14, 'low': 23}.

## Anchor fit

| anchor | private PCS (mean) | local 281 mean | ability | fitted PCS | residual | leave-this-anchor-out PCS |
|---|--:|--:|--:|--:|--:|--:|
| garage | 1819 (0.436) | 0.477 | +0.12 | 1783 | -36 | 1765 (-54) |
| go-straight | 1446 (0.140) | 0.164 | -2.23 | 1464 | +18 | 1553 (+107) |
| RSG+AG | 2089 (0.632) | 0.691 | +2.50 | 2107 | +18 | 2197 (+108) |

Map: PCS = 135.8 x ability + 1767. RSG + ACC + GAP is the highest anchor; ability above it is extrapolated.
Out-of-pool scoring check (pool subjects: MAP under the fitted items minus the fitted posterior mean): median +0.03, max |diff| 0.11 (NTV0-s0+AG+hl), = 16 PCS.

## Predicted private PAI PCS

CI: scene bootstrap of the whole pipeline (281 scenes resampled, pool refit, anchors re-mapped; 100 draws), 2.5-97.5 %. 'beyond 2089' = extrapolation above the top anchor.

| subject | family | local mean | exact 1 | zeros | ability | PCS paimix | 95 % CI | min-max over pool variants | vs RSG + ACC + GAP | to top-3 2186 |
|---|---|--:|--:|--:|--:|--:|---|---|--:|--:|
| NTV0-s0+AG+hl | band | 0.746 | 191 | 61 | +3.34 | **2221** | [2155, 2298] | 2151-2332 | +114 | +35 |
| NTV0-s0+AG | band | 0.744 | 184 | 59 | +3.15 | **2195** | [2138, 2299] | 2130-2260 | +89 | +9 |
| NTV0-s1+AG | band | 0.739 | 181 | 62 | +2.91 | **2163** | [2106, 2249] | 2121-2233 | +56 | -23 |
| NTVXS-s0+AG | band | 0.722 | 177 | 66 | +2.91 | **2162** | [2107, 2230] | 2120-2199 | +56 | -24 |
| STANDFMX-s1+AG | band | 0.726 | 187 | 63 | +2.90 | **2161** | [2066, 2257] | 2108-2246 | +54 | -25 |
| STANDFMX-s0+AG | band | 0.697 | 180 | 71 | +2.82 | **2150** | [2075, 2235] | 2097-2206 | +43 | -36 |
| STANDFMXH-s0+AG | band | 0.704 | 173 | 66 | +2.76 | **2142** | [2070, 2226] | 2088-2173 | +35 | -44 |
| JTN-s0+AG | band | 0.714 | 173 | 65 | +2.72 | **2136** | [2089, 2203] | 2095-2157 | +29 | -50 |
| JTN-s1+AG | band | 0.704 | 170 | 68 | +2.65 | **2126** | [2069, 2184] | 2089-2154 | +20 | -60 |
| BNC2K-s1+AG | band | 0.715 | 174 | 65 | +2.56 | **2115** | [2043, 2170] | 2079-2136 | +8 | -71 |
| NTVX-s0+AG | band | 0.729 | 177 | 64 | +2.56 | **2114** | [2043, 2210] | 2091-2154 | +7 | -72 |
| JTXS-s1+AG | band | 0.709 | 170 | 65 | +2.52 | **2109** | [2060, 2178] | 2080-2128 | +2 | -77 |
| RSG+AG | band | 0.691 | 170 | 76 | +2.50 | **2107** | [2088, 2119] | 2087-2123 | +0 | -79 |
| RSGC-s1+ACC | other | 0.693 | 167 | 71 | +1.71 | **1999** | [1919, 2072] | 1929-2067 | -108 | -187 |
| RSGC-s0+ACC | other | 0.683 | 165 | 73 | +1.70 | **1998** | [1924, 2059] | 1933-2053 | -108 | -188 |
| RSG-s1+ACC | other | 0.693 | 167 | 70 | +1.69 | **1997** | [1917, 2064] | 1930-2064 | -110 | -189 |
| RSG+ACC | other | 0.676 | 163 | 75 | +1.65 | **1991** | [1914, 2057] | 1924-2047 | -116 | -195 |
| RSG-s0+ACC (compile) | other | 0.674 | 163 | 76 | +1.65 | **1991** | [1914, 2057] | 1924-2047 | -116 | -195 |
| RSG+ACC+desire | other | 0.674 | 160 | 74 | +1.59 | **1983** | [1912, 2046] | 1915-2041 | -124 | -203 |
| NR1F-s0+ACC | other | 0.665 | 159 | 78 | +1.50 | **1970** | [1913, 2020] | 1909-2034 | -136 | -216 |
| NR1FB10-s0+ACC | other | 0.646 | 155 | 85 | +1.43 | **1962** | [1903, 2004] | 1902-2032 | -145 | -224 |
| NTV0-s0+AG+keywarp | other | 0.676 | 151 | 62 | +0.69 | **1861** | [1793, 1905] | 1834-1971 | -246 | -325 |
| RSG-g10+hl | other | 0.612 | 118 | 76 | +0.25 | **1801** | [1741, 1883] | 1785-1921 | -306 | -385 |
| RCVL-g20 | other | 0.601 | 108 | 75 | +0.17 | **1791** | [1732, 1879] | 1774-1905 | -316 | -395 |
| RCVRL-g20 | other | 0.591 | 106 | 78 | +0.17 | **1790** | [1732, 1876] | 1773-1903 | -317 | -396 |
| RCVRLF-g20 | other | 0.600 | 106 | 71 | +0.14 | **1786** | [1730, 1873] | 1769-1898 | -321 | -400 |
| RCVLF-g20 | other | 0.602 | 106 | 71 | +0.14 | **1786** | [1730, 1877] | 1768-1899 | -321 | -400 |
| garage | third | 0.477 | 122 | 136 | +0.12 | **1783** | [1747, 1820] | 1717-1822 | -323 | -403 |
| RCVR-g20 | other | 0.576 | 98 | 77 | +0.12 | **1783** | [1729, 1869] | 1765-1892 | -324 | -403 |
| JTN-s0 | other | 0.601 | 101 | 58 | +0.09 | **1780** | [1726, 1863] | 1760-1880 | -327 | -406 |
| NTV0-s0 (N0) | other | 0.611 | 102 | 59 | +0.09 | **1780** | [1726, 1866] | 1761-1886 | -327 | -406 |
| RSG-g20 | other | 0.581 | 97 | 72 | +0.09 | **1779** | [1727, 1865] | 1760-1884 | -328 | -407 |
| RSG-g10 | other | 0.571 | 98 | 75 | +0.08 | **1778** | [1727, 1865] | 1759-1884 | -328 | -408 |
| RBSG2-g10 | other | 0.584 | 98 | 68 | +0.08 | **1778** | [1725, 1863] | 1758-1884 | -329 | -408 |
| RSG-g5 | other | 0.563 | 94 | 78 | +0.07 | **1777** | [1726, 1864] | 1757-1884 | -330 | -409 |
| NTV0-s0+keywarp | other | 0.572 | 84 | 59 | +0.06 | **1775** | [1726, 1862] | 1754-1875 | -332 | -411 |
| go-straight | third | 0.164 | 37 | 197 | -2.23 | **1464** | [1445, 1489] | 1445-1514 | -643 | -722 |

Paired bootstrap differences (same draws):

| difference | median | 95 % CI | P(> 0) |
|---|--:|---|--:|
| NTV0-s0+AG - RSG+AG | +83 | [+20, +201] | 1.00 |
| NTV0-s1+AG - RSG+AG | +55 | [-0, +145] | 0.97 |
| STANDFMX-s0+AG - RSG+AG | +34 | [-31, +129] | 0.82 |
| STANDFMX-s0+AG - NTV0-s0+AG | -51 | [-118, +9] | 0.06 |
| JTN-s0+AG - RSG+AG | +26 | [-16, +94] | 0.83 |
| NTVX-s0+AG - RSG+AG | +4 | [-64, +101] | 0.59 |

## Sensitivity to the pool

| variant | rows | scenes | anchor RMSE | NTV0-s0+AG | NTV0-s1+AG | STANDFMX-s0+AG | JTN-s0+AG | NTVX-s0+AG | garage | go-straight | RSG+AG |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paimix | 75 | 275 | 25 | 2195 | 2163 | 2150 | 2136 | 2114 | 1783 | 1464 | 2107 |
| paimix-s1 | 75 | 275 | 8 | 2164 | 2138 | 2122 | 2118 | 2096 | 1808 | 1451 | 2095 |
| paimix-s2 | 75 | 275 | 10 | 2176 | 2144 | 2132 | 2122 | 2100 | 1804 | 1453 | 2097 |
| garagex0.5 | 58 | 275 | 16 | 2181 | 2153 | 2141 | 2124 | 2106 | 1796 | 1457 | 2101 |
| garagex2 | 110 | 275 | 39 | 2188 | 2162 | 2137 | 2141 | 2123 | 1765 | 1476 | 2114 |
| bandx0.5 | 62 | 275 | 8 | 2158 | 2139 | 2113 | 2116 | 2108 | 1808 | 1451 | 2095 |
| bandx2 | 114 | 275 | 55 | 2207 | 2171 | 2179 | 2139 | 2116 | 1741 | 1493 | 2120 |
| gsx0.5 | 68 | 275 | 57 | 2221 | 2188 | 2174 | 2157 | 2130 | 1740 | 1494 | 2120 |
| gsx2 | 89 | 275 | 2 | 2149 | 2129 | 2117 | 2106 | 2096 | 1816 | 1447 | 2091 |
| low-as-gs | 98 | 275 | 2 | 2135 | 2121 | 2106 | 2099 | 2091 | 1822 | 1445 | 2087 |
| allours | 84 | 269 | 11 | 2148 | 2134 | 2097 | 2106 | 2093 | 1804 | 1453 | 2097 |
| dedup | 15 | 275 | 73 | 2260 | 2233 | 2206 | 2156 | 2154 | 1717 | 1514 | 2123 |
| paimix-1pl | 75 | 275 | 16 | 2130 | 2128 | 2097 | 2095 | 2112 | 1797 | 1457 | 2101 |
| paimix, MAP for all | 75 | 275 | 28 | 2209 | 2171 | 2158 | 2141 | 2116 | 1779 | 1467 | 2109 |

## Easy items: garage scores 1, NTV0-s0 + ACC + GAP does not (decision 381 cause classes)

24 easy items lost (5 zeros, 19 partials, 10.6 points); garage = 1 on 119 of 275 scenes, NTV0 = 1 on 181.

| cause (d381 class / sub) | items | points | median a | PCS if these -> 1 (paimix) |
|---|--:|--:|--:|--:|
| partial rear-hit | 12 | 4.4 | 0.16 | +19 |
| partial slow | 7 | 1.2 | 0.22 | +18 |
| zero corridor / cor_drift | 1 | 1.0 | 0.06 | +1 |
| zero corridor / cor_route_end | 3 | 3.0 | 0.09 | +3 |
| zero offroad / off_straight | 1 | 1.0 | 0.09 | +1 |
| **all easy items** | 24 | 10.6 | 0.18 | +45 |
| (non-easy losses: garage < 1 too) | 70 | 59.9 | 0.19 | +640 |

NTV0 + ACC + GAP scores 1 where garage does not on 86 scenes (median a 2.33).

Item discrimination by d381 stratum (paimix): static 0.19, launch 0.54, stop/lead 0.70, straight 1.45, turn 0.28
