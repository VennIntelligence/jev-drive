# PAI-PCS-LOCAL-R5: R2C / R2J / R3MV scored on the d392 ruler (2026-10-21)

The [pai_pcs_local.md](pai_pcs_local.md) ruler (decision 392: zoib 2PL official settings, 75-row pool banded as on the PAI board of 10-19,
275 scenes with no driver RPC error, three-anchor least-squares map PCS = 135.8 x ability + 1767, scene bootstrap) applied to the stored PAI 281
ACC + GAP reads of R2C-L18-s0, R2J-L18-s0 and R3MV-L18-s0 (read roots under `$DATA_DIR/runs/round2/pai281/` and `round5/r5_281/pai281/`; per-scene
causes from `runs/pai_fix/facts/*-281ag.csv`, column `fclass`, as in decision 427). Nothing about the pool changed: the 13 band reads, garage,
go-straight, the 100 bootstrap seeds. The three new reads are scored out of pool (MAP under the fixed items), exactly like every non-pool read in d392.
NTV0-L18-s0 + ACC + GAP is re-scored in the same run as the reference: it reproduces d392 (2195, CI [2138, 2299]; bootstrap draws equal to
d392's `boot.csv.gz` to 0.05 PCS). Code: `scripts/pai_pcs_local.py` (`collect --only`, `r5`) and `scripts/pai_pcs_local_r5.sh`; the fits ran as CPU
pool jobs (PRIO 15; drive-irt 82ddd5a copied to the box as a pure-python tree, since the box env has no `drive_irt`). Artifacts:
`pai_pcs_local_r5/` (`r5_report.md` generated tables, `pai_pcs_local_r5.json`, `boot_r5.csv.gz` 100 draws x all subjects, `rows.csv.gz` per-scene scores of 40 subjects).

## Short answer

| read | local mean (275 scenes) | zeros | predicted PCS | 95 % CI (scene bootstrap) | paired vs NTV0-s0 | 95 % CI | P(> NTV0) |
|---|--:|--:|--:|---|--:|---|--:|
| **NTV0-s0 + ACC + GAP** (reference) | 0.743 | 58 | **2195** | [2138, 2299] | | | |
| R2C-s0 + ACC + GAP | 0.740 | 59 | **2161** | [2089, 2239] | -29 (point -34) | [-95, +19] | 0.17 |
| R2J-s0 + ACC + GAP | 0.687 | 63 | **1885** | [1797, 1938] | -307 (point -310) | [-420, -227] | 0.00 |
| R3MV-s0 + ACC + GAP | 0.741 (274 scenes, 1 RPC error) | 61 | **1918** | [1849, 1960] | -276 (point -277) | [-375, -217] | 0.00 |

- Between the new reads (same draws): R3MV - R2C -250 [-354, -158]; R3MV - R2J +28 [-10, +84]; R2J - R2C -278 [-388, -186].
- **Official board check on NTV0.** The same candidate (NTV0-L18-s0 + g10 + ACC + GAP 0.7,2.7, no holdlead; `p1021a`, decision 426) scored **2154 [2106, 2202]** on
  the board; the ruler predicted **2195 [2138, 2299]**: 41 PCS high, inside the CI. In the same board call RSG + ACC + GAP scored 2108, and the ruler puts it at 2107
  (anchor 2089 in the 10-19 fit), so the NTV0 - RSG gap is +88 on the ruler against +46 on the board. The board-to-ruler ratio of the *gap* is about 0.5-0.6, the
  direction decision 426 already noted (gap of local means +0.0545 against +0.033 on the board).
- **R3MV does not hold up on this ruler even though its mean equals NTV0's** (0.741 vs 0.743; decision 427: -0.004). Predicted 277 PCS below NTV0, at about R2J's level (R3MV - R2J +28 [-10, +84]). The cause is not the mean but where the failures sit (next section). This is a statement about the ruler, which counts imperfect
  scores on band-agreed scenes very heavily; see the caveats before reading it as a board forecast.

## Why: the ruler is a perfection test on 116 band-agreed items

The fitted 2PL puts 90 of the 275 scenes at discrimination a >= 3 (all 13 pool band reads score 1 on essentially all of them, difficulty b about -0.6) and 26 more at 1 <= a < 3.
The other 159 (a < 1) carry the scenes where our whole family fails or splits. In the zoib a non-extreme score has almost no probability at high ability, so one partial
or zero on an a >= 3 item costs ~10 log-likelihood units at ability 3 and caps the MAP ability.
Items by discrimination a (zeros / partials / ones of each read), and what perfection on the high-a items is worth:

| read | a >= 3 (zeros / partials / ones) | 1 <= a < 3 | a < 1 | PCS if every imperfect a >= 3 item -> 1 | if every imperfect a >= 1 item -> 1 |
|---|---|---|---|--:|--:|
| NTV0-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 0 / 1 / 25 | 159 items: 58 / 35 / 66 | +0 | +12 |
| R2C-s0+AG | 90 items: 0 / 0 / 90 | 26 items: 4 / 0 / 22 | 159 items: 55 / 35 / 69 | +0 | +55 |
| R2J-s0+AG | 90 items: 11 / 6 / 73 | 26 items: 5 / 5 / 16 | 159 items: 47 / 37 / 75 | +114 | +182 |
| R3MV-s0+AG | 90 items: 13 / 4 / 73 | 26 items: 3 / 2 / 21 | 159 items: 45 / 24 / 89 | +164 | +341 |


- NTV0 and R2C are perfect on all 90 a >= 3 items. R3MV has **13 zeros and 4 partials** there, R2J 11 and 6. R3MV's 13 zeros are 6 `zero-other` / `side` (drift / lane change), 4 `offroad`, 3 `overshoot`;
  6 of them are on items with a >= 4.1 where the garage 0014 scores 1 (listing below).
- Fixing only R3MV's 13 high-a zeros moves it +45 (ability 1.11 -> 1.44) because 4 partials remain on those items and the ruler charges them as heavily; making every imperfect a >= 3 item a 1 gives +164, and every a >= 1 item +341 (to about NTV0's level).
- Decision 427's trade (route-turn zeros -13, offroad +8, other zeros +8, slow -6) is therefore paid in the wrong currency on this ruler: the 13 route-turn scenes R3MV fixed
  sit on a < 1 items (NTV0's 15 route-turn zeros have a 0.05-0.40, median 0.19: the whole family fails them together), while 10 of the 23 lateral zeros R3MV has and NTV0 does not (offroad / drift / lane change / side) are on a >= 3 items.
### Zeros on the a >= 3 items, by cause (garage 0014 and go-straight scores for reference)

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

## Value of each loss class if fixed

Class = `fclass` of the stored facts (partials 0 < score < 0.98: rear-hit / slow; zeros: overshoot = passes the human's rest point, route-turn = straight at a turn, offroad, other =
AF-front + side + drift / lane change). Setting a class's scenes to 1 and re-estimating the read's ability under the fixed items (MAP), PCS delta. Class values are **not additive**
and not comparable across columns: the ability is a steep function of the few imperfect high-a items, so a class that holds only low-a scenes is worth little however many scenes it has,
and fixing every class saturates the ability at the grid edge (8), which has no meaning. Read the ranking inside a column.
 (scenes -> 1, items fixed, MAP; paimix)

Class = pai_fix.py `fclass` of the stored facts. Read the ranking, not the sizes: the ceiling sits above the anchors.

| class | NTV0-s0+AG | R2C-s0+AG | R2J-s0+AG | R3MV-s0+AG |
|---|--:|--:|--:|--:|
| partial rear-hit | 25 sc, 10.4 pt: **+168** | 25 sc, 10.8 pt: **+101** | 33 sc, 19.1 pt: **+31** | 24 sc, 9.5 pt: **+20** |
| partial slow | 11 sc, 2.2 pt: **+52** | 10 sc, 1.7 pt: **+33** | 12 sc, 4.1 pt: **+25** | 5 sc, 0.6 pt: **+15** |
| partial other contact | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** | 0 sc, 0.0 pt: **+0** |
| zero overshoot (passes the human's rest point) | 22 sc, 22.0 pt: **+38** | 21 sc, 21.0 pt: **+34** | 21 sc, 21.0 pt: **+6** | 22 sc, 22.0 pt: **+12** |
| zero route-turn | 15 sc, 15.0 pt: **+28** | 15 sc, 15.0 pt: **+46** | 2 sc, 2.0 pt: **+0** | 2 sc, 2.0 pt: **+0** |
| zero offroad | 12 sc, 12.0 pt: **+22** | 11 sc, 11.0 pt: **+28** | 20 sc, 20.0 pt: **+17** | 19 sc, 19.0 pt: **+21** |
| zero other (AF-front, side, drift / lane change) | 9 sc, 9.0 pt: **+19** | 12 sc, 12.0 pt: **+46** | 20 sc, 20.0 pt: **+15** | 18 sc, 18.0 pt: **+25** |
| all partials | 36 sc, 12.6 pt: **+508** | 35 sc, 12.5 pt: **+277** | 45 sc, 23.1 pt: **+96** | 29 sc, 10.1 pt: **+49** |
| all zeros | 58 sc, 58.0 pt: **+117** | 59 sc, 59.0 pt: **+177** | 63 sc, 63.0 pt: **+37** | 61 sc, 61.0 pt: **+61** |


For **R3MV** the ranking is: other zeros (drift / lane change / side / AF-front) +25 (18 scenes), offroad +21 (19), partial rear-hit +20 (24), partial slow +15 (5), overshoot +12 (22), route-turn +0 (2 left);
all partials together +49, all zeros together +61. Individually small because every class holds only some of the 20 imperfect a >= 1 items; together (see the band table) they are worth +341.
For NTV0 the d392 numbers reproduce (rear-hit +168, slow +52, overshoot +38).

## Pool sensitivity


| variant | anchor RMSE | NTV0-s0+AG | R2C-s0+AG | R2J-s0+AG | R3MV-s0+AG | garage | go-straight | RSG+AG | R3MV - NTV0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paimix | 25 | 2195 | 2161 | 1885 | 1918 | 1783 | 1464 | 2107 | -277 |
| paimix-s1 | 8 | 2164 | 2145 | 1931 | 1945 | 1808 | 1451 | 2095 | -220 |
| paimix-s2 | 10 | 2176 | 2149 | 1913 | 1920 | 1804 | 1453 | 2097 | -256 |
| garagex0.5 | 16 | 2181 | 2147 | 1905 | 1925 | 1796 | 1457 | 2101 | -256 |
| garagex2 | 39 | 2188 | 2170 | 1870 | 1919 | 1765 | 1476 | 2114 | -269 |
| bandx0.5 | 8 | 2158 | 2146 | 1856 | 1909 | 1808 | 1451 | 2095 | -250 |
| bandx2 | 55 | 2207 | 2148 | 1858 | 1871 | 1741 | 1493 | 2120 | -335 |
| gsx0.5 | 57 | 2221 | 2156 | 1866 | 1900 | 1740 | 1494 | 2120 | -321 |
| gsx2 | 2 | 2149 | 2128 | 1864 | 1907 | 1816 | 1447 | 2091 | -242 |
| low-as-gs | 2 | 2135 | 2119 | 1855 | 1900 | 1822 | 1445 | 2087 | -235 |
| dedup | 73 | 2260 | 2179 | 1854 | 1884 | 1717 | 1514 | 2123 | -376 |
| paimix-1pl | 16 | 2130 | 2117 | 2015 | 2125 | 1797 | 1457 | 2101 | -5 |


- R3MV is below NTV0 in 11 of the 12 2PL variants (-220 to -376) and at -5 under 1PL (fixed discrimination), where ability is close to the mean score. The whole gap is the 2PL
  discrimination of the band-agreed items. R2J behaves the same (1854-1931 under 2PL, 2015 under 1PL, about -115 vs NTV0 there).
- R3MV - R2C is similar in sign (-250 on the default pool; paired CI excludes 0).

## Caveats

- **The strong band is our own family.** The 13 band reads (NTV0 seeds and variants, JTN, RSG, STAND, ...) agree on the 90 high-a scenes, which is what makes them high-a. On the real
  board, other strong teams may also fail those scenes, and then they would not carry a = 4. d392 flagged this; here it is the whole story for R3MV. The prediction is a statement
  about "does the read fail scenes that every read of the NTV0 family solves", not a calibrated board forecast.
- **Out-of-pool scoring.** The three new reads are MAP under fixed items; NTV0 is a pool member (posterior mean, with 13 near-copies in the pool). For pool subjects MAP and posterior mean differ by at most 16 PCS (d392).
  R2C, a near-copy of NTV0, lands at -34 on that footing, so some of that may be the pool-member advantage.
- **Extrapolation.** NTV0's 2195 is above the top anchor (2089); the leave-top-anchor-out error was +108. The board gave 2154. For R3MV and R2J (1885-1918) the predictions sit between the garage (1819) and the top anchor
  and are interpolations.
- One seed per read (the NTV0-s0 / s1 pair differs by 32 PCS); the CI is a scene bootstrap, not a seed one. Sensitivity rows other than `paimix` were fit once.

## Commands

```bash
# box (repo root): reads' per-scene rows (base = d392 rows.csv.gz), then CPU pool jobs: 12 fits + 20 bootstrap shards x 5 draws + report
.venv/bin/python experiments/alpasim_proxy/scripts/pai_pcs_local.py collect --out $T/rows.csv.gz --only "R2C-s0+AG,R2J-s0+AG,R3MV-s0+AG,NTV0-s0+AG" --base experiments/alpasim_proxy/results/pai_pcs_local/rows.csv.gz
bash experiments/alpasim_proxy/scripts/pai_pcs_local_r5.sh submit      # T=$DATA_DIR/tmp/pcs_r5, drive-irt tree in $T/site on PYTHONPATH
```
