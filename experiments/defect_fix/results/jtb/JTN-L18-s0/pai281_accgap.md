# JTN-L18-s0 on PAI 281 with the submitted PAI probe's serve config (ACC + GAP), against RSG-L18-s0 under the same config (10-18)

Config: `SERVE=JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` (no JEV_HOLD), steering weight 10, CONC 4, `ALPASIM_FAST=1`, code d78d5f2b, one seed, 8 chunk lists of the
reference, 8 pool jobs at priority 13. Reference = the archived ACC-GAP stage-2 read (decision 330, `runs/route_follow/acc_gap/read/s2`, code 94711ca9, identical SERVE / WIZ / CONC / chunks),
reused, not rerun: the served-path diff 94711ca9..HEAD is only default-off flags (JEV_DESIRE, JEV_HOLD files, anchored-memory layout read from the sidecar) plus a t_first fix in the Tokyo-only ap2 driver.
Note: `wt.PAI_REF` (`pai_g10/acc`) is the ACC-only read (no GAP); the GAP read is `acc_gap/read/s2`. Third column = the no-hold JTN read (`pai281.md`, `pai_nohold/JTN-L18-s0`, no ACC / GAP).
280 common scenes, no driver RPC error in any of the three reads (the 281st scene is missing from the reference's scored set). Strata rows tagged "(JTN-no-hold)" are the no-hold read against the same reference
(their column headers read as the first read). Script `scripts/pai281.py --j2`.


J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/JTN-L18-s0`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J [] [], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| JTN+ACC+GAP | 280 | 0.7129 | 65 | 173 | 0.872 | 0.844 |
| JTN-no-hold | 280 | 0.6011 | 58 | 104 | 0.755 | 0.614 |
| RSG+ACC+GAP | 280 | 0.6898 | 76 | 170 | 0.886 | 0.851 |

Paired mean (JTN+ACC+GAP - RSG+ACC+GAP): **+0.0232 [-0.0083, +0.0565]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0393 [-0.0714, -0.0071] (-11 zeros).


Paired mean (JTN-no-hold - RSG+ACC+GAP): **-0.0887 [-0.1366, -0.0401]**; zeros -18.

Paired mean (JTN+ACC+GAP - JTN-no-hold): **+0.1118 [+0.0664, +0.1575]**; zeros +7.

## Zero types

| type | JTN+ACC+GAP | JTN-no-hold | RSG+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 4 | 7 | 7 |
| AF-side | 1 | 2 | 2 |
| corridor | 41 | 36 | 53 |
| offroad | 19 | 13 | 14 |

## Partial scores (0 < s < 0.98)

| type | JTN+ACC+GAP | JTN-no-hold | RSG+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 30 | 76 | 26 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 3 | 0 |
| slow | 12 | 39 | 8 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | JTN+ACC+GAP mean | RSG+ACC+GAP mean | diff | contribution | JTN+ACC+GAP zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-no-hold) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-no-hold) launch | 37 | 0.4513 | 0.7545 | -0.3032 | -0.0401 | 7 | 4 | 3 | 0 |
| launch | 37 | 0.7226 | 0.7545 | -0.0319 | -0.0042 | 2 | 4 | 0 | 2 |
| (JTN-no-hold) moving | 241 | 0.6250 | 0.6814 | -0.0565 | -0.0486 | 50 | 71 | 10 | 31 |
| moving | 241 | 0.7132 | 0.6814 | +0.0318 | +0.0274 | 62 | 71 | 5 | 14 |

## New zeros (zero for JTN+ACC+GAP, scored for RSG+ACC+GAP): 5

| scene | stratum | JTN+ACC+GAP | RSG+ACC+GAP | JTN-no-hold |
|---|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok | 0.784 rear-hit |
| 0de150b0 | moving | 0.000 offroad | 0.947 rear-hit | 0.662 rear-hit |
| 7aeba6e8 | moving | 0.000 corridor | 1.000 ok | 0.000 offroad |
| 29d87ce7 | moving | 0.000 corridor | 1.000 ok | 0.466 rear-hit |
| 1438905d | moving | 0.000 offroad | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for RSG+ACC+GAP, scored for JTN+ACC+GAP): 16

| scene | stratum | JTN+ACC+GAP | RSG+ACC+GAP | JTN-no-hold |
|---|---|---|---|---|
| be9320f3 | moving | 1.000 ok | 0.000 corridor | 0.781 rear-hit |
| 58eb6cd6 | moving | 1.000 ok | 0.000 corridor | 0.655 slow |
| 90f9cd38 | launch | 1.000 ok | 0.000 AF-front | 0.000 AF-front |
| 353e8268 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 62ec6f99 | moving | 1.000 ok | 0.000 AF-front | 0.000 AF-front |
| 272b7bf0 | moving | 0.831 slow | 0.000 corridor | 0.868 rear-hit |
| 89fae1ac | moving | 0.952 slow | 0.000 offroad | 0.584 slow |
| 1f5c9e1f | moving | 0.565 rear-hit | 0.000 corridor | 0.000 offroad |
| 8832a062 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 7c0fb633 | moving | 1.000 ok | 0.000 AF-front | 0.000 AF-front |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 4f568cd4 | moving | 0.827 slow | 0.000 corridor | 0.688 slow |
| d5527d43 | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| bd6a4d25 | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
| 70d5986f | moving | 1.000 ok | 0.000 AF-side | 0.612 rear-hit |
| 3793ab49 | launch | 0.480 rear-hit | 0.000 corridor | 0.000 corridor |
