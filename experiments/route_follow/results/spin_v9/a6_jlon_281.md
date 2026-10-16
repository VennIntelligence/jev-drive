> **Amendment 6 passed by user waiver** (L2 literal 8 vs 7 control zeros, one late corridor zero at 19.1 s, scene 62cfe9d3; user 2026-10-16 03:00 JST). This is the regular jlon281 read of the spin_v9 lane (Tokyo, 2026-10-16 06:35-08:56 JST), paired with `v2pai/RSG-L18-s0-gate`.
> The L1-L3 lines printed below are the 84-scene lines of the sv9_read template applied to the 281 read and are not the Amendment 6 criterion for it; the criterion is the union row: zeros and at-fault paired CIs level or better (95 % CI including 0 or below). Union: zeros 71 vs 71, diff +0 [-5, +5]; at-fault 13 vs 12, diff +1 [-2, +5]; mean 0.5855 vs 0.5860, -0.0004 [-0.0166, +0.0165] -> level on both, so by the Amendment text v8b (gate + jlon_v=8) would replace v8 as the PAI default (the decision is the main session's).
> Companion files: `a6_jlon_281_read281.md` (rf_pai281 tables), `a6_jlon_281_turns.md` (strict-36 / exam-72 turns, paired with the gate reference: strict completed 4 vs 4, zeros over the 90 turn scenes 50 vs 51), `a6_jlon_281_line3.txt` (SLOW line-3 numbers for information only: slow ratio 0.598 vs 0.587, handoff 0.867 vs 0.867, collision zeros 13 vs 12, rear-ended 105 vs 108, cor_route_end 18 vs 18).
> Note: the Tokyo `rf_v2pai.sh read` run by `sv9.sh jlon281` printed a turns table with every scene `missing` (its reference pkl label was not mapped); the table here was regenerated with `rf_v2pai.py turns --pkl rsg=<gate follow.pkl>:rsg-l18-s0-gate jl8=<jlon8 follow.pkl>:rsg-l18-s0-jlon8`.

# SPIN v9 read: A6 281: jlon_v=8 vs RSG-L18-s0-gate, arm `/data/runs/route_night/v2pai/RSG-L18-s0-jlon8` vs identity `/data/runs/route_night/v2pai/RSG-L18-s0-gate`

| set | scenes (both) | spins sf any (id) | handoff spins (id) | zeros (id) | new / recovered zeros | zeros diff 95 % CI | at-fault / front / rear (id) | at-fault diff 95 % CI | mean (id) | diff 95 % CI |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| spin set | 38 | 1 (1) | 0 (0) | 15 (15) | 0 / 0 | +0 [+0, +0] | 1 / 0 / 6 (1 / 0 / 6) | +0 [+0, +0] | 0.4514 (0.4516) | -0.0002 [-0.0007, +0.0002] |
| controls | 46 | 0 (0) | 0 (0) | 8 (7) | 1 / 0 | +1 [+0, +3] | 1 / 1 / 19 (1 / 1 / 19) | +0 [+0, +0] | 0.5612 (0.5823) | -0.0210 [-0.0617, -0.0001] |
| union | 281 | 2 (2) | 0 (0) | 71 (71) | 3 / 3 | +0 [-5, +5] | 13 / 6 / 72 (12 / 4 / 72) | +1 [-2, +5] | 0.5855 (0.5860) | -0.0004 [-0.0166, +0.0165] |

- L1 spin set spins 1 <= 2: pass
- L2 controls zeros CI upper +3 <= +2 and mean diff -0.0210 >= -0.01: FAIL
- L3 new offroad zeros with the event <= 4.7 s over the union: 0 : pass
- missing rollouts (no metrics in either root): spin set 0, controls 0, union 0

## Per scene (union; changed score, zero or spin only)

| scene | sets | id score | arm score | id zero / spin (onset s) | arm zero / spin (onset s) | id / arm first event s |
|:--|:--|--:|--:|:--|:--|--:|
| 0f4b8753 |  | 0.333 | 0.337 | - / - | - / - | 8.8 / 8.9 |
| 13d646af |  | 0.733 | 0.738 | - / - | - / - | - / - |
| 5b5f9d86 |  | 0.501 | 0.484 | - / - | - / - | 10.2 / 10.0 |
| 62cfe9d3 | controls | 0.915 | 0.000 | - / - | corridor / - | - / 19.1 |
| 66b15fbe | controls | 0.859 | 0.842 | - / - | - / - | 15.4 / 15.2 |
| 81dc4ece |  | 0.862 | 0.870 | - / - | - / - | - / - |
| 82f505c9 |  | 1.000 | 0.973 | - / - | - / - | 13.1 / 12.7 |
| 8e62a884 |  | 0.473 | 0.469 | - / - | - / - | 14.5 / 14.4 |
| 901c2cff | controls | 0.612 | 0.605 | - / - | - / - | - / - |
| 92af888e |  | 0.494 | 0.439 | - / - | - / - | 8.9 / 8.3 |
| 94c1d722 |  | 0.000 | 0.542 | offroad / - | - / - | 16.1 / - |
| 9d9e7f39 |  | 0.431 | 0.431 | - / - | - / - | - / - |
| a68ad579 | controls | 0.425 | 0.441 | - / - | - / - | - / - |
| b1d93f58 |  | 0.640 | 0.636 | - / - | - / - | - / - |
| b680a689 |  | 0.000 | 0.000 | offroad / - | collision / - | 14.4 / 9.4 |
| b727530b | controls | 0.823 | 0.807 | - / - | - / - | 19.6 / 19.3 |
| baabc57d |  | 0.960 | 0.887 | - / - | - / - | - / - |
| bbc53aa8 | spin set | 0.576 | 0.576 | - / - | - / - | - / - |
| bf56fd00 |  | 0.546 | 0.519 | - / - | - / - | 12.5 / 12.1 |
| c4a58f83 | controls | 0.839 | 0.832 | - / - | - / - | - / - |
| c5da4975 |  | 0.534 | 0.543 | - / - | - / - | - / - |
| ca7accbc |  | 0.739 | 0.688 | - / - | - / - | 18.2 / 17.6 |
| db8693db |  | 0.617 | 0.579 | - / - | - / - | 13.5 / 12.8 |
| df99cdc4 |  | 0.821 | 0.806 | - / - | - / - | 10.7 / 10.4 |
| e848c843 | controls | 0.789 | 0.781 | - / - | - / - | - / - |
| e9384415 |  | 0.948 | 0.958 | - / - | - / - | - / - |
| f9121c36 | controls | 0.728 | 0.727 | - / - | - / - | 9.2 / 9.2 |
| fda39805 |  | 0.000 | 0.987 | corridor / - | - / - | 17.9 / - |
| ffe5b17d | controls | 0.452 | 0.442 | - / - | - / - | - / - |
| 074e5ee3 |  | 0.380 | 0.387 | - / - | - / - | 10.9 / 11.0 |
| 1e31b2ad |  | 0.544 | 0.538 | - / - | - / - | - / - |
| 2a0efada |  | 0.576 | 0.564 | - / - | - / - | - / - |
| 4b5a7c8b |  | 0.317 | 0.317 | - / - | - / - | 7.7 / 7.7 |
| 524089d2 | controls | 0.511 | 0.499 | - / - | - / - | 15.8 / 15.6 |
| 593ebf97 |  | 0.611 | 0.610 | - / - | - / - | 11.4 / 11.4 |
| 7c6529f6 |  | 0.219 | 0.243 | - / - | - / - | 18.1 / 18.3 |
| 903731dd |  | 0.166 | 1.000 | - / - | - / - | 9.0 / - |
| a686c7d0 |  | 0.051 | 0.055 | - / - | - / - | 10.4 / 10.5 |
| b2591a17 |  | 0.523 | 0.504 | - / - | - / - | - / - |
| c92137b4 |  | 0.615 | 0.612 | - / - | - / - | - / - |
| cbb22323 |  | 0.889 | 0.768 | - / - | - / - | - / 15.5 |
| e632cf52 |  | 0.788 | 0.627 | - / - | - / - | 15.5 / 12.2 |
| e682c5d7 |  | 0.967 | 0.974 | - / - | - / - | 17.8 / 17.9 |
| ea468790 | controls | 0.522 | 0.527 | - / - | - / - | - / - |
| ec1d0b2b | controls | 0.137 | 0.142 | - / - | - / - | 9.6 / 9.7 |
| fb138448 |  | 0.927 | 0.928 | - / - | - / - | 14.6 / 14.6 |
| fbf97438 |  | 0.333 | 0.332 | - / - | - / - | 7.8 / 7.8 |
| 009c69f0 |  | 0.952 | 0.907 | - / - | - / - | 15.9 / 15.2 |
| 0cc7fa67 |  | 0.598 | 0.596 | - / - | - / - | 9.4 / 9.4 |
| 2a47ea01 |  | 0.256 | 0.264 | - / - | - / - | 12.5 / 12.6 |
| 3294623f |  | 0.437 | 0.448 | - / - | - / - | - / - |
| 450d5c8d | controls | 0.724 | 0.722 | - / - | - / - | - / - |
| 4f7cb072 | spin set | 0.451 | 0.442 | - / - | - / - | - / - |
| 51242b4d |  | 0.756 | 0.755 | - / - | - / - | 7.9 / 7.9 |
| 5266c7c9 |  | 0.990 | 1.000 | - / - | - / - | - / - |
| 5c4a277a |  | 0.441 | 0.486 | - / - | - / - | 15.0 / 15.5 |
| 6fd8cf30 |  | 0.304 | 0.295 | - / - | - / - | 11.3 / 11.1 |
| 7b5c9fd6 |  | 0.125 | 0.175 | - / - | - / - | 11.9 / 12.4 |
| 88bbf0c9 |  | 0.500 | 0.510 | - / - | - / - | - / - |
| 8b25a6ac |  | 0.748 | 0.721 | - / - | - / - | - / - |
| a1f07c94 |  | 0.752 | 0.748 | - / - | - / - | - / - |
| be0d3628 |  | 0.991 | 0.987 | - / - | - / - | - / - |
| c4100c5d |  | 0.610 | 0.587 | - / - | - / - | - / - |
| d641893f |  | 0.366 | 0.345 | - / - | - / - | 15.4 / 15.3 |
| 21829be5 |  | 0.692 | 0.782 | - / - | - / - | 16.8 / 17.5 |
| 229791a5 |  | 0.000 | 0.989 | collision / - | - / - | 10.5 / 10.5 |
| 28621273 |  | 0.381 | 0.367 | - / - | - / - | 15.9 / 15.7 |
| 3c2822d9 |  | 0.864 | 0.879 | - / - | - / - | - / - |
| 415e950b |  | 0.453 | 0.440 | - / - | - / - | 10.6 / 10.3 |
| 4e454838 |  | 0.899 | 0.849 | - / - | - / - | - / - |
| a2b415d5 |  | 0.735 | 0.715 | - / - | - / - | - / - |
| ddb6a43a |  | 0.352 | 0.361 | - / - | - / - | - / - |
| f18c6340 |  | 0.153 | 0.184 | - / - | - / - | 11.0 / 11.2 |
| 2b577fc2 |  | 0.944 | 1.000 | - / - | - / - | 19.0 / - |
| 47db2eb6 |  | 0.991 | 0.987 | - / - | - / - | - / - |
| 9fa1d6eb |  | 1.000 | 0.000 | - / - | collision / - | 14.6 / 14.5 |
| 000525f6 |  | 0.953 | 0.922 | - / - | - / - | - / - |
| 05855d57 | controls | 0.545 | 0.541 | - / - | - / - | - / - |
| 098e2482 |  | 0.686 | 0.679 | - / - | - / - | 14.4 / 14.3 |
| 132ae6cb | controls | 0.029 | 0.031 | - / - | - / - | 6.1 / 6.2 |
| 1a050ead |  | 0.937 | 0.934 | - / - | - / - | - / - |
| 32be03e4 | spin set | 0.439 | 0.441 | - / - | - / - | - / - |
| 404541da |  | 0.665 | 0.429 | - / - | - / - | 13.3 / 10.0 |
| 4e4a2a51 |  | 0.750 | 0.000 | - / - | offroad / - | - / 8.5 |
| 52bd89ce |  | 0.155 | 0.141 | - / - | - / - | 8.7 / 8.5 |

VERDICT A6 281: jlon_v=8 vs RSG-L18-s0-gate: L1 pass, L2 FAIL, L3 pass
