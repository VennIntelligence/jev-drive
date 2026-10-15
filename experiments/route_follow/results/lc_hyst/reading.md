# LC-HYSTERESIS reading (plans/2026-10-15-lc-hysteresis-prereg.md)

Map lane-change sites with gap <= 6 m (rf_laneoff lc_map, d299 Amendment 3); validated = the human moved 2-6 m to that side. Per site: had the side command in the window, the first L/R -> S flip (decision k, station rel. to the site, ego move so far, d_target, the command input y at the flip), the plan's 2 s lateral toward the side at k_f - 1 and its min over k_f .. k_f + 2, C1 / C2.

## Result (registered lines)

- **C1 on RSG-L18-s0 (the pilot policy): 0 sites** (15 reached sites, 10 with the side command, 5 flips to S; no flip is followed within 2
  decisions by a >= 50 % drop of the plan's 2 s lateral while the ego is >= 1.5 m from the target). Below the pilot gate (>= 3):
  **no JEV_CMD_LATCH pilot.** RV2a: 0. RV2b: 4 (26004ca8, 415e950b, 51242b4d, 6250146a; reported, not the gate).
- **C2: 0 / 0 on RSG** (no C1 flip). Where flips happen (all runs) the command's own input y sits just under the threshold (1.21-1.99 m)
  while the ego has moved little (median ~0.4 m; d_target 3.2-3.6 m at most flips): the flip comes from the 42 m waypoint's y hovering at
  2 m (route geometry, ego yaw), not from the ego having moved 1.5-2.5 m toward the target.

Descriptive (validated sites, figure `figs/lc_hyst_timeline.png`: blue = d_target, red = plan 2 s lateral toward the side, dots = cmd):
- The command is a poor lane-change signal: at 8a143cb2 (the one site RSG completes) the command is R then S throughout, never L; at
  51242b4d it is S for most of the approach and L only briefly around the site; at 1e8e939e it follows the road's bends (R / S / L / R).
- 415e950b shows the user's drift-then-snap without a flip: cmd L held for the whole window, the plan's 2 s lateral rises to ~1 m just past
  the site and falls back to ~0 while L is still served; the snap is the policy, not the command.
- f8ee31cb is the closest case to the hypothesis: the plan peaks at +0.7 m about 22 m past the site, the command turns S at +24 m and the
  plan falls to -1.5 m over the next ~8 decisions (slower than the registered 2-decision window).

## rsg: 23 sites (6 validated), 15 reached; with the side command 10; flipped to S 5; **C1 0** (); C2 0

| scene | val | side | gap | reached | side cmd | flip k (st rel) | move at flip | d_target at flip | cmd y at flip | plan y2 pre / post | C1 | C2 | shift / in lane after |
|:--|:--|:--|--:|:--|:--|:--|--:|--:|--:|:--|:--|:--|:--|
| 1e8e939e | yes | L | 5.7 | True | True | 47 (+23 m) | +1.77 | -0.61 | +1.52 | +1.96 / +1.52 | False | False | +1.3 / True |
| 415e950b | yes | L | 4.3 | True | True | 167 (+8 m) | -3.60 | +3.66 | +1.85 | -0.02 / -0.31 | False | False | +0.3 / False |
| 51242b4d | yes | L | 3.8 | True | True | 164 (+5 m) | +0.35 | +3.32 | +1.75 | +0.52 / +0.30 | False | False | +1.1 / False |
| 8a143cb2 | yes | L | 2.6 | True | False |  |  |  |  |  /  | False | False | +4.0 / True |
| e904e9c0 | yes | R | 4.1 | False | True |  |  |  |  |  /  | False | False |  / False |
| f8ee31cb | yes | L | 3.1 | False | True | 46 (+24 m) | -0.01 | +3.15 | +1.70 | +0.67 / +0.58 | False | False |  / False |
| 000525f6 |  | R | 3.4 | True | False |  |  |  |  |  /  | False | False | +0.0 / False |
| 1e31b2ad |  | R | 3.5 | True | False |  |  |  |  |  /  | False | False | +0.4 / True |
| 26004ca8 |  | L | 3.7 | True | True |  |  |  |  |  /  | False | False | +0.0 / False |
| 2a0efada |  | L | 2.6 | False | False |  |  |  |  |  /  | False | False |  / False |
| 478d6ec4 |  | L | 4.0 | True | False |  |  |  |  |  /  | False | False | +0.0 / False |
| 5ddb96bd |  | L | 2.9 | True | True |  |  |  |  |  /  | False | False | +0.1 / False |
| 6250146a |  | R | 3.3 | True | True |  |  |  |  |  /  | False | False | -0.0 / True |
| 6e6cbe67 |  | R | 4.7 | True | True |  |  |  |  |  /  | False | False | +0.5 / True |
| 788c4ad9 |  | R | 2.2 | False | True |  |  |  |  |  /  | False | False |  / False |
| 98f88a4e |  | R | 2.8 | False | True |  |  |  |  |  /  | False | False |  / False |
| a2c5ae46 |  | L | 4.3 | True | True | 54 (+15 m) | +0.53 | +1.73 | +1.91 | +1.08 / +0.64 | False | False | +3.6 / True |
| a4d2f7cf |  | L | 5.3 | True | True |  |  |  |  |  /  | False | False | +9.1 / False |
| a6941bb6 |  | R | 3.7 | False | False |  |  |  |  |  /  | False | False |  / False |
| bd97de34 |  | R | 2.7 | True | True | 1 (-24 m) | -0.04 | +2.42 | +1.95 | +0.21 / +0.30 | False | False | +0.0 / True |
| c4a58f83 |  | L | 3.8 | True | False |  |  |  |  |  /  | False | False | +0.0 / False |
| ca481583 |  | R | 3.8 | False | False |  |  |  |  |  /  | False | False |  / False |
| e5e47fd2 |  | L | 3.6 | False | True |  |  |  |  |  /  | False | False |  / False |

## rv2a: 23 sites (6 validated), 15 reached; with the side command 10; flipped to S 6; **C1 0** (); C2 0

| scene | val | side | gap | reached | side cmd | flip k (st rel) | move at flip | d_target at flip | cmd y at flip | plan y2 pre / post | C1 | C2 | shift / in lane after |
|:--|:--|:--|--:|:--|:--|:--|--:|--:|--:|:--|:--|:--|:--|
| 1e8e939e | yes | L | 5.7 | True | True | 47 (+23 m) | +1.76 | -0.60 | +1.28 | +1.92 / +1.56 | False | False | +1.2 / True |
| 415e950b | yes | L | 4.3 | True | True |  |  |  |  |  /  | False | False | +0.0 / False |
| 51242b4d | yes | L | 3.8 | True | True | 164 (+6 m) | +0.45 | +3.22 | +1.93 | +0.53 / +0.30 | False | False | +1.0 / False |
| 8a143cb2 | yes | L | 2.6 | True | False |  |  |  |  |  /  | False | False | +3.9 / True |
| e904e9c0 | yes | R | 4.1 | False | True |  |  |  |  |  /  | False | False |  / False |
| f8ee31cb | yes | L | 3.1 | False | True | 45 (+24 m) | -0.01 | +3.15 | +1.77 | +0.58 / +0.56 | False | False |  / False |
| 000525f6 |  | R | 3.4 | True | False |  |  |  |  |  /  | False | False | -0.0 / False |
| 1e31b2ad |  | R | 3.5 | True | False |  |  |  |  |  /  | False | False | +0.4 / True |
| 26004ca8 |  | L | 3.7 | True | True | 178 (+24 m) | +1.43 | +3.35 | +1.99 | +0.23 / -0.23 | False | False | +0.0 / True |
| 2a0efada |  | L | 2.6 | False | False |  |  |  |  |  /  | False | False |  / False |
| 478d6ec4 |  | L | 4.0 | True | False |  |  |  |  |  /  | False | False | +0.0 / False |
| 5ddb96bd |  | L | 2.9 | True | True | 194 (+54 m) | +5.85 | -6.49 | +1.95 | +0.38 / +0.29 | False | False | +0.1 / False |
| 6250146a |  | R | 3.3 | True | True |  |  |  |  |  /  | False | False | -0.0 / True |
| 6e6cbe67 |  | R | 4.7 | True | True |  |  |  |  |  /  | False | False | +0.4 / True |
| 788c4ad9 |  | R | 2.2 | False | True |  |  |  |  |  /  | False | False |  / False |
| 98f88a4e |  | R | 2.8 | False | False |  |  |  |  |  /  | False | False |  / False |
| a2c5ae46 |  | L | 4.3 | True | True | 54 (+15 m) | +0.52 | +1.74 | +1.92 | +1.10 / +0.78 | False | False | +3.8 / True |
| a4d2f7cf |  | L | 5.3 | True | True |  |  |  |  |  /  | False | False | +8.1 / False |
| a6941bb6 |  | R | 3.7 | False | False |  |  |  |  |  /  | False | False |  / False |
| bd97de34 |  | R | 2.7 | True | True | 1 (-24 m) | -0.04 | +2.42 | +1.95 | +0.18 / +0.26 | False | False | +0.0 / True |
| c4a58f83 |  | L | 3.8 | True | False |  |  |  |  |  /  | False | False | +0.0 / False |
| ca481583 |  | R | 3.8 | False | False |  |  |  |  |  /  | False | False |  / False |
| e5e47fd2 |  | L | 3.6 | False | True | 192 (-18 m) | -97.50 | +95.17 | +1.22 | +0.00 / +0.00 | False | False |  / False |

## rv2b: 23 sites (6 validated), 14 reached; with the side command 10; flipped to S 9; **C1 4** (26004ca8, 415e950b, 51242b4d, 6250146a); C2 1

| scene | val | side | gap | reached | side cmd | flip k (st rel) | move at flip | d_target at flip | cmd y at flip | plan y2 pre / post | C1 | C2 | shift / in lane after |
|:--|:--|:--|--:|:--|:--|:--|--:|--:|--:|:--|:--|:--|:--|
| 1e8e939e | yes | L | 5.7 | True | True | 47 (+23 m) | +1.74 | -0.58 | +1.21 | +1.94 / +1.56 | False | False | +1.4 / True |
| 415e950b | yes | L | 4.3 | True | True | 96 (-23 m) | -1.88 | +1.96 | +1.91 | +0.96 / -0.00 | True | True | +8.1 / False |
| 51242b4d | yes | L | 3.8 | True | True | 153 (-0 m) | +0.29 | +3.39 | +1.94 | +0.60 / -0.26 | True | False | +1.0 / False |
| 8a143cb2 | yes | L | 2.6 | True | False |  |  |  |  |  /  | False | False | +4.1 / True |
| e904e9c0 | yes | R | 4.1 | False | True |  |  |  |  |  /  | False | False |  / False |
| f8ee31cb | yes | L | 3.1 | False | True | 40 (+23 m) | -0.01 | +3.15 | +1.80 | +0.62 / +0.30 | True | False |  / False |
| 000525f6 |  | R | 3.4 | True | False |  |  |  |  |  /  | False | False | +0.4 / False |
| 1e31b2ad |  | R | 3.5 | True | False |  |  |  |  |  /  | False | False | +0.5 / True |
| 26004ca8 |  | L | 3.7 | True | True | 194 (+18 m) | +1.84 | +3.58 | +1.75 | +1.01 / +0.49 | True | False | +0.0 / False |
| 2a0efada |  | L | 2.6 | False | False |  |  |  |  |  /  | False | False |  / False |
| 478d6ec4 |  | L | 4.0 | True | False |  |  |  |  |  /  | False | False | +0.1 / False |
| 5ddb96bd |  | L | 2.9 | True | True | 194 (+59 m) | +7.73 | -8.38 | +1.82 | +1.36 / +1.25 | False | False | +0.1 / False |
| 6250146a |  | R | 3.3 | True | True | 127 (+57 m) | -0.67 | +3.20 | +1.99 | +0.38 / +0.16 | True | False | -0.0 / True |
| 6e6cbe67 |  | R | 4.7 | True | True |  |  |  |  |  /  | False | False | -0.0 / True |
| 788c4ad9 |  | R | 2.2 | False | True |  |  |  |  |  /  | False | False |  / False |
| 98f88a4e |  | R | 2.8 | False | True |  |  |  |  |  /  | False | False |  / False |
| a2c5ae46 |  | L | 4.3 | True | True | 61 (+26 m) | +0.80 | +1.46 | +1.99 | +1.22 / +0.85 | False | False | +3.5 / True |
| a4d2f7cf |  | L | 5.3 | False | True |  |  |  |  |  /  | False | False |  / False |
| a6941bb6 |  | R | 3.7 | False | False |  |  |  |  |  /  | False | False |  / False |
| bd97de34 |  | R | 2.7 | True | True | 1 (-24 m) | -0.04 | +2.42 | +1.95 | +0.06 / +0.35 | False | False | +0.0 / True |
| c4a58f83 |  | L | 3.8 | True | True | 191 (-2 m) | -0.78 | -0.39 | +1.99 | +0.59 / +0.37 | False | False | +0.0 / False |
| ca481583 |  | R | 3.8 | False | False |  |  |  |  |  /  | False | False |  / False |
| e5e47fd2 |  | L | 3.6 | False | True |  |  |  |  |  /  | False | False |  / False |

## Validated sites: per-decision timeline (cmd letters from s - 40 m; k of the first window decision)

- rsg 1e8e939e (L, k 4..71): `RRRRRRRRRRRRRRRRRRRRRRRRSSSSSSSLLLLLLLLLLLLSSSRRRRRRRRRRRRRRRRRRRRRR`
- rsg 415e950b (L, k 48..198): `LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLSSSSLLLLLLLLLLLLLLLLLLSSSSSSSSSS`
- rsg 51242b4d (L, k 51..198): `SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSLLLLLLLLLLLLLLLLLLLLLLLLSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS`
- rsg 8a143cb2 (L, k 69..138): `RRRRRRRRRRSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS`
- rsg e904e9c0 (R, k 89..196): `SSSSSSSSSSSSSSSSSSRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR`
- rsg f8ee31cb (L, k 0..90): `LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLSSSSSSSSSSSRRRRRRSSSSSSSSSSSSLLLLLLLLLLLLLLLL`
- rv2a 1e8e939e (L, k 4..71): `RRRRRRRRRRRRRRRRRRRRRRRRSSSSSSSLLLLLLLLLLLLSSSRRRRRRRRRRRRRRRRRRRRRR`
- rv2a 415e950b (L, k 48..198): `LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL`
- rv2a 51242b4d (L, k 51..198): `SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSRRRRRRRRRRRRSSSSSSSSSSSSSSSSSSSSSSSSSSSLLLLLLLLLLLLLLLLLLLLLLLLLLLSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS`
- rv2a 8a143cb2 (L, k 69..138): `RRRRRRRRRRRRRRSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS`
- rv2a e904e9c0 (R, k 89..196): `SSSSSSSSSSSSSSSSSSRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR`
- rv2a f8ee31cb (L, k 0..89): `LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLSSSSSSSSSSSSSRRRRSSSSSSSSSSSSLLLLLLLLLLLLLLLL`
- rv2b 1e8e939e (L, k 4..72): `RRRRRRRRRRRRRRRRRRRRRRRRSSSSSSLLLLLLLLLLLLLSSSRRRRRRRRRRRRRRRRRRRRRRR`
- rv2b 415e950b (L, k 48..198): `LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSLLLLLLLLLLLLLLLLLLLLLLLLLLLLLSSSSRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR`
- rv2b 51242b4d (L, k 50..198): `SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSLLLLLLLLLLSSSSSSSSSSLSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS`
- rv2b 8a143cb2 (L, k 70..140): `RRRRRRRRRRRRRSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS`
- rv2b e904e9c0 (R, k 89..196): `SSSSSSSSSSSSSSSSSSSRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR`
- rv2b f8ee31cb (L, k 0..80): `LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLSSSSSSSSSSSSSSSSSSSSSSSSSSSLLLLLLLLLLLLLL`
