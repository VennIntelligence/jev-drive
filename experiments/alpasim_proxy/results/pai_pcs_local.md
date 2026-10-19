# PAI-PCS-LOCAL: a local ruler for the private PAI-board PCS (2026-10-19)

The PAI counterpart of [pcs_local.md](pcs_local.md) (decision 382). The question: can we predict the private PAI-board Policy
Capability Score of a candidate from its local PAI 281 read without spending a submission slot? And what does that predict for
NTV0-L18-s0 + ACC + GAP and the other stored PAI 281 reads?

Pre-registration: [../plans/2026-10-19-pai-pcs-local-prereg.md](../plans/2026-10-19-pai-pcs-local-prereg.md) (0e8d022b). It was
written before any garage or go-straight PAI read. Code: `scripts/pai_pcs_local.py` (`collect` on the box; `fit` / `report` on the
Mac, in `tmp/pcs_local/.venv` with drive-irt 82ddd5a). The generated tables are in [pai_pcs_local/report.md](pai_pcs_local/report.md).
The artifacts sit next to it: `pai_pcs_local.json`, `rows.csv.gz` (per-scene score / type / RPC flag of all 37 subjects),
`boot.csv.gz` (100 bootstrap draws) and `easy_items.csv`. Box reads are under `$DATA_DIR/runs/alpasim_proxy/pai_pcs/`.

## Short answer

1. **The new third-party reads work and behave like the board.**
   - **Garage `model_0014`:** PAI rig `sensor_rig_2.yaml`, official camera set. Local PAI 281 mean **0.477** against 0.436 private.
   - **Go-straight:** AlpaSim starter kit, as it ships. Local mean **0.164** against 0.140 private.
   - **Ours, RSG + ACC + GAP** (stored read, decision 330 `acc_gap/read/s2`): local 0.691 against 0.632 private.
   - The local / private ratio is 1.09-1.17 for all three, so the local 281 scenes are slightly easier than the private set and
     keep the board's order.
   - Garage had 0 driver input or inference errors over 281 sessions.
   - The rig does not matter: the rig-2 and rig-3 smoke runs were bit-identical, controller logs included. The garage driver
     replaces the rig's camera calibration with the simulator's announced f-theta calibration.
2. **Anchor fit.** Three anchors and two map parameters give RMSE 25 PCS. Leaving one anchor out costs -54 (garage), +107
   (go-straight) and +108 (RSG + ACC + GAP).
3. **Prediction for NTV0-L18-s0 + ACC + GAP: 2195, 95 % CI [2138, 2299].**
   - The CI is a scene bootstrap of the whole pipeline.
   - Over 13 pool variants the prediction ranges from 2130 to 2260. Over seeds 42 / 1 / 2 it is 2195 / 2164 / 2176.
   - Against RSG + ACC + GAP on the same draws: **+83 [+20, +201]**, P(> 0) 1.00.
   - **This is an extrapolation.** 2195 lies 106 PCS, or 0.65 ability units, above the highest anchor (2089).
   - Leaving the top anchor out mispredicts it by +108. The upward slope is pinned by one point.
   - **Top-3 (2186) and #1 (2250) are 97 and 161 PCS above our highest anchor.** "NTV0 reaches top-3" (+9 on the point
     estimate) is inside the error of the map.
4. **Full B (STANDFMX-L18-s0 + ACC + GAP): 2150 [2075, 2235].** That is +34 [-31, +129] over RSG + ACC + GAP and **-51 [-118, +9]
   below NTV0-s0** (P(B > NTV0) 0.06). On the PAI ruler, B is not the better PAI candidate.
5. **Easy items do not carry PAI the way they carry nuPlan.**
   - Garage scores exactly 1 on 119 of 275 scenes; NTV0 + ACC + GAP misses 24 of those (5 zeros, 19 partials, 10.6 points).
   - Fixing all 24 is worth only +45.
   - On PAI the garage is the weak family. Discrimination is high where the garage fails and our band splits (median a 2.33 on the
     86 scenes NTV0 aces and the garage does not). It is low on the scenes our whole family fails together.
   - NTV0's PCS is lost mostly to **partials**: rear-hit partials (25 scenes, 10.4 points) are worth **+168** and slow partials
     (11, 2.2 points) **+52**. That is far more per point than its zeros: passing the human's rest point (22 zeros) is worth +38,
     and straight-at-turn (11) +20.

## 1. Reads

All reads use the same 8-chunk table as every stored PAI 281 read (hash 28800209). They ran through `pai_box.sh` with the new
switches (0e8d022b):
- `DRV=garage` / `DRV=starter`, through `PAI_DRV` in `pai_native.sh`;
- `GARAGE_RIG` in `drivers/garage.sh`, default `sensor_rig_0.yaml`, so nuPlan behaviour is unchanged;
- `ALPASIM_FAST=0` (official camera set; fast mode is single-camera and the garage reads the cross-left, front-wide and
  cross-right 120fov cameras);
- `eval.video.render_video=false` (the video is drawn after scoring);
- the preset's controller gain (5). Third-party entries have no reason to set the CLI gain of 10 that our probes set.

| subject | local mean | exact 1 | zeros | private mean | private PCS |
|---|--:|--:|--:|--:|--:|
| garage 0014, rig 2 | 0.477 | 122 | 136 | 0.436 | 1819 |
| go-straight (starter kit) | 0.164 | 37 | 197 | 0.140 | 1446 (host `go_straight_with_delay_v2`) |
| RSG + ACC + GAP (stored) | 0.691 | 170 | 76 | 0.632 | 2089 |

- **Smoke.** Two scenes per rig and for go-straight passed. Rig 2 and rig 3 gave identical scores and controller csv md5s.
- **Pre-registered sanity check passed:** garage mean between go-straight and RSG + ACC + GAP, and 0 % of sessions with driver
  errors.
- **One infrastructure failure.** Go-straight c005 failed once with the runtime unable to spawn threads (EAGAIN at the pids cap,
  16 stacks up). The pool retried it and the retry passed.
- **Caveats.**
  - The host's go-straight is `go_straight_with_delay_v2`, not necessarily the starter kit, so the go-straight anchor is a
    stand-in (decision 325 made the same assumption on nuPlan).
  - **GPU cost: 11.5 stack-hours** over 20 tries (16 chunks, 3 smoke runs, 1 retry). At about 3 stacks per card that is about
    3.8 card-hours. Wall time was about 1 h 10 min.

## 2. The ruler

- **Model.** Zoib 2PL with the official settings (`pcs_local.fit`: epochs 1000, lr 0.1, decay 0.975, 16 particles, seed 42).
- **Items.** The 275 scenes that every pool subject scored without a driver RPC error (complete-matrix rule; 6 of the 281 have an
  RPC error in some ACC + GAP read).
- **Pool weights.** Taken from the 104 PAI-board rows (`leaderboard_pai_20261019.json`), each assigned to the stand-in nearest in
  PCS and behaviour:

| board band | rows | who | stand-in | pool rows |
|---|--:|---|---|--:|
| PCS >= 1940 | 32 | ACM, VAIL, Anonymous host, BASICLAB, ..., ours, Host internal (mean 0.50-0.74) | our 13 ACC + GAP reads x 2 | 26 |
| 1750-1939 | 35 | 17 py123d-garage LTF rows, Ulsan's 0014 copy, participants that score like them | garage 0014 | 35 |
| 1440-1749 | 14 | Alpain mars / jupiter, starter copies, host go-straight (mean 0.14-0.40) | go-straight | 14 |
| < 1440 | 23 | VaVAM / WAM / broken drivers, policy_4 (mean 0.03-0.19) | none, left out (as in decision 382) | 0 |

- **Why the strong band uses our reads.** On PAI our family scores near the top of the board (row 18, team #11). It is the only
  local stand-in for the 32 strong rows; the other 31 rows are policies we cannot read.
- **This is the main limitation.** Decision 382 showed that discrimination is set by whoever is the majority in the pool. Here the
  strong band's failure pattern is ours, so scenes our family fails together get low discrimination. That may undervalue them
  relative to the board, where other strong teams may solve them.
- **Scale.** PCS = 135.8 x ability + 1767, least squares on the three anchors.
- **Subjects outside the pool.** Our 23 other serve configurations are scored as one more submission under the fixed items (MAP).
  For pool subjects, MAP and the posterior mean differ by at most 0.11 ability units (16 PCS).

## 3. Anchor fit

| anchor | private PCS | fitted | residual | leave-this-anchor-out |
|---|--:|--:|--:|--:|
| garage | 1819 | 1783 | -36 | 1765 (-54) |
| go-straight | 1446 | 1464 | +18 | 1553 (+107) |
| RSG + ACC + GAP | 2089 | 2107 | +18 | 2197 (+108) |

With two anchors and two parameters every leave-one-out is an exact two-point line, so these are the only out-of-sample checks
available. They say a prediction 100 PCS beyond the anchors can be off by about 100.

## 4. Predicted private PAI PCS

| candidate | local mean | PCS | 95 % CI (scene bootstrap) | range over pool variants | vs RSG + ACC + GAP | vs top-3 2186 |
|---|--:|--:|---|---|--:|--:|
| NTV0-s0 + ACC + GAP + holdlead | 0.746 | 2221 | [2155, 2298] | 2151-2332 | +114 | +35 |
| **NTV0-s0 + ACC + GAP** | 0.744 | **2195** | [2138, 2299] | 2130-2260 | **+89** | +9 |
| NTV0-s1 + ACC + GAP | 0.739 | 2163 | [2106, 2249] | 2121-2233 | +56 | -23 |
| NTVXS-s0 + ACC + GAP | 0.722 | 2162 | [2107, 2230] | 2120-2199 | +56 | -24 |
| STANDFMX-s1 + ACC + GAP (full B, seed 1) | 0.726 | 2161 | [2066, 2257] | 2108-2246 | +54 | -25 |
| **STANDFMX-s0 + ACC + GAP (full B)** | 0.697 | **2150** | [2075, 2235] | 2097-2206 | +43 | -36 |
| STANDFMXH-s0 + ACC + GAP | 0.704 | 2142 | [2070, 2226] | 2088-2173 | +35 | -44 |
| JTN-s0 + ACC + GAP | 0.714 | 2136 | [2089, 2203] | 2095-2157 | +29 | -50 |
| JTN-s1 + ACC + GAP | 0.704 | 2126 | [2069, 2184] | 2089-2154 | +20 | -60 |
| BNC2K-s1 + ACC + GAP | 0.715 | 2115 | [2043, 2170] | 2079-2136 | +8 | -71 |
| NTVX-s0 + ACC + GAP | 0.729 | 2114 | [2043, 2210] | 2091-2154 | +7 | -72 |
| JTXS-s1 + ACC + GAP | 0.709 | 2109 | [2060, 2178] | 2080-2128 | +2 | -77 |
| RSG + ACC + GAP (anchor, private 2089) | 0.691 | 2107 | [2088, 2119] | 2087-2123 | 0 | -79 |
| RSG + ACC (no GAP; 5 reads) | 0.674-0.693 | 1991-1999 | ~[1915, 2070] | 1924-2067 | -108 to -116 | |
| NR1F / NR1FB10 + ACC | 0.665 / 0.646 | 1970 / 1962 | | | | |
| g10 / g20 / g5 family, no ACC (11 reads) | 0.56-0.61 | 1775-1801 | | | | |

**Paired on the same bootstrap draws:**

| difference | median | 95 % CI | P(> 0) |
|---|--:|---|--:|
| NTV0-s0 + AG - RSG + AG | +83 | [+20, +201] | 1.00 |
| NTV0-s1 + AG - RSG + AG | +55 | [-0, +145] | 0.97 |
| STANDFMX-s0 + AG - RSG + AG | +34 | [-31, +129] | 0.82 |
| STANDFMX-s0 + AG - NTV0-s0 + AG | -51 | [-118, +9] | 0.06 |
| JTN-s0 + AG - RSG + AG | +26 | [-16, +94] | 0.83 |
| NTVX-s0 + AG - RSG + AG | +4 | [-64, +101] | 0.59 |

How to read it:

- **Every ACC + GAP read lands above the RSG anchor.** Every read without GAP lands about 100 below it, even at an equal mean:
  RSGC-s1 + ACC has 0.693 against 0.691. GAP's fewer rear-hit partials are what the zoib pays for.
- **NTV0-s0 and its holdlead variant lead.** Holdlead failed decision 370's mean line (+0.0004 [-0.0136, +0.0134]), but it is +26
  here: rear-hit partials 26 -> 21 and slow 11 -> 8, traded for 2 more zeros, is exactly the trade the zoib pays for. The seed pair
  NTV0-s0 / s1 differs by 32 PCS, which is the seed noise to keep in mind.
- **Extrapolation.**
  - Our three anchors lie at or below 2089, and every ACC + GAP candidate above RSG is extrapolated.
  - NTV0-s0's ability (3.15) is 0.65 units above the top anchor (2.50).
  - The top-3 threshold 2186 is 97 PCS above it, and #1 (2250) is 161 above.
  - The leave-top-anchor-out error is +108. The predicted margin over top-3 (+9) is therefore not evidence that NTV0 makes
    top-3; the prediction is "very likely above our current 2089, likely in the 2130-2260 band".

## 5. Sensitivity to the pool

| variant | rows | anchor RMSE | NTV0-s0 | NTV0-s1 | full B | JTN-s0 | NTVX | garage | go-straight | RSG + AG |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paimix (default) | 75 | 25 | 2195 | 2163 | 2150 | 2136 | 2114 | 1783 | 1464 | 2107 |
| seed 1 / seed 2 | 75 | 8 / 10 | 2164 / 2176 | 2138 / 2144 | 2122 / 2132 | 2118 / 2122 | 2096 / 2100 | 1808 / 1804 | 1451 / 1453 | 2095 / 2097 |
| garage x0.5 / x2 | 58 / 110 | 16 / 39 | 2181 / 2188 | 2153 / 2162 | 2141 / 2137 | 2124 / 2141 | 2106 / 2123 | 1796 / 1765 | 1457 / 1476 | 2101 / 2114 |
| our band x0.5 / x2 | 62 / 114 | 8 / 55 | 2158 / 2207 | 2139 / 2171 | 2113 / 2179 | 2116 / 2139 | 2108 / 2116 | 1808 / 1741 | 1451 / 1493 | 2095 / 2120 |
| go-straight x0.5 / x2 | 68 / 89 | 57 / 2 | 2221 / 2149 | 2188 / 2129 | 2174 / 2117 | 2157 / 2106 | 2130 / 2096 | 1740 / 1816 | 1494 / 1447 | 2120 / 2091 |
| low band as go-straight (+23) | 98 | 2 | 2135 | 2121 | 2106 | 2099 | 2091 | 1822 | 1445 | 2087 |
| all 35 of our reads pooled | 84 | 11 | 2148 | 2134 | 2097 | 2106 | 2093 | 1804 | 1453 | 2097 |
| dedup (one row each) | 15 | 73 | 2260 | 2233 | 2206 | 2156 | 2154 | 1717 | 1514 | 2123 |
| 1PL | 75 | 16 | 2130 | 2128 | 2097 | 2095 | 2112 | 1797 | 1457 | 2101 |

- **The order at the top is stable.** NTV0-s0 + ACC + GAP + holdlead (decision 370) is first in all 14 variants, 14-72 above
  NTV0-s0. NTV0-s0 is second in all 14, and in 1PL it ties with NTV0-s1. Full B s0 stays below NTV0-s0 in every variant.
- **The variants that fit the anchors best put NTV0-s0 at 2135-2150.** That is go-straight x2 and the low band as go-straight
  (RMSE 2); in the default pool the low band has no stand-in.
- **The upward variants put it at 2207-2260.** Those are band x2, go-straight x0.5 and dedup, and they fit the anchors worst
  (RMSE 55-73).
- So the anchor-best estimate is a bit lower than the default: about **2140-2195**.

## 6. Easy items: garage = 1, NTV0-s0 + ACC + GAP < 1

Cause classes are decision 381's, from `$DATA_DIR/runs/route_follow/pai_gap/scenes.csv`. "PCS if -> 1" sets those scenes to 1 in
NTV0's read, with items fixed (MAP).

| cause | items | points | median a | PCS if -> 1 |
|---|--:|--:|--:|--:|
| partial rear-hit | 12 | 4.4 | 0.16 | +19 |
| partial slow | 7 | 1.2 | 0.22 | +18 |
| zero corridor, passes the human's rest point (cor_route_end) | 3 | 3.0 | 0.09 | +3 |
| zero corridor, drift | 1 | 1.0 | 0.06 | +1 |
| zero offroad, straight | 1 | 1.0 | 0.09 | +1 |
| **all easy items** | **24** | **10.6** | 0.18 | **+45** |
| losses where the garage fails too | 70 | 59.9 | 0.19 | (+640 if all -> 1; far beyond the anchors) |

NTV0's PCS by loss class over all its losses (same MAP; the ceiling sits far above the anchors, so read the ranking, not the sizes):

| class | scenes | points | PCS if -> 1 |
|---|--:|--:|--:|
| partial rear-hit | 25 | 10.4 | **+168** |
| partial slow | 11 | 2.2 | **+52** |
| zero corridor, passes the human's rest point | 22 | 22.0 | +38 |
| zero corridor, straight at a turn | 11 | 11.0 | +20 |
| zero corridor, drift | 5 | 5.0 | +9 |
| AF-front | 3 | 3.0 | +8 |
| turn under / over, junction and straight offroad (8 classes) | 17 | 17.0 | +1 to +7 each |

- **Item discrimination by decision 381 stratum:** straight 1.45, stop / lead 0.70, launch 0.54, turn 0.28, static 0.19.
- **Our family fails together.** Of NTV0's 94 lost scenes, only 1 is scored 1 by 6 or more of the other 12 ACC + GAP reads.
- **The PAI levers, per this ruler:**
  - First, partials. Each rear-hit partial is worth about 7 PCS, each slow one about 5, against about 2 per rest-point zero.
  - The rest-point and turn zeros (decision 381's levers 1 and 2) are worth less than their mean points suggest, because the strong
    band fails them as a block.
  - That ranking depends on the pool's strong band being our own family. If other strong board teams solve those scenes, the board
    may value them more than this ruler does.

## Commands

```bash
# box: reads (submitted once; the collect job is chained --after all 16 chunks)
CF=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks; B=$DATA_DIR/runs/alpasim_proxy/pai_pcs
CHUNKS_FROM=$CF DRV=garage SERVE=GARAGE_RIG=sensor_rig_2.yaml ALPASIM_FAST=0 WIZ=eval.video.render_video=false \
  bash experiments/alpasim/scripts/pai_box.sh submit $B/G0014 G0014
CHUNKS_FROM=$CF DRV=starter SERVE= ALPASIM_FAST=0 WIZ=eval.video.render_video=false bash experiments/alpasim/scripts/pai_box.sh submit $B/GS GS
.venv/bin/python experiments/alpasim_proxy/scripts/pai_pcs_local.py collect --out $B/rows.csv.gz
# Mac
X="tmp/pcs_local/.venv/bin/python experiments/alpasim_proxy/scripts/pai_pcs_local.py"
$X fit --rows rows.csv.gz --out out                                     # 13 pool variants, about 20 s each
$X fit --rows rows.csv.gz --out out --variants paimix-s1 --boot 25 --boot0 {0,25,50,75}   # 4 parallel shards
$X report --rows rows.csv.gz --out out --gap scenes.csv
```

A new candidate: add its PAI 281 read root to `SRC` (family `band` if it is a submission-serve ACC + GAP read), collect, fit, report.
