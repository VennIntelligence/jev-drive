> **UNRELIABLE (route v2 prereg Amendment 18):** this run drove at the default controller gains, not gains 20 (`rf_v2pai.sh run` launched pai_eval in a tmux window that did not inherit WIZ). Not comparable with r281/rsg. The read of record is `RV2a-L18-s0.md` (gains 20).

# Handoff-spin reading (route v2 prereg Amendment 15): rv2a vs rsg on PAI 281

Roots `/data/runs/route_night/v2pai/RV2a-L18-s0` / `/data/runs/route_night/r281/rsg`; spins by rf_pai281's rule; handoff k 17 (force-GT ends; before it both drive the log). rf_v2pai.py spins.

## Answers (main session's questions; descriptive, v2 prereg Amendment 15)

**(a) Not a different plan at the handoff, and not the d285 handoff mechanics either: a closed-loop instability of RV2a at speed.**
At the handoff decision k 17 RV2a's plan equals RSG's in all 13 spin scenes (median |heading diff at 2 s| 0.0 deg, |lateral| 0.01 m;
0 / 13 over 10 deg or 0.5 m). One to three decisions later, with the two tracks still within 0.05 m (they part at k 19-21), RV2a's
2 s heading jumps by 15-50 deg and then oscillates in sign (06b3e399: +2.5, +5.3, +8.0, -46.8, +11.7, +31.7 deg at k 17-25; RSG holds
+2.1 to +2.5), while RSG's plan under the same near-identical states stays steady. So the plan is what breaks, from tiny state changes,
not the controller's first step.

**(b) The route memory is not new at the handoff.** In all 13 spin scenes the first route message arrived 1.7 s before k 17 (18 messages
by then), the memory is full or nearly full (35-40 valid slots, first valid slot at -6 to +4 m) and the ordered start equals the projection
start (gap 0.0 m; >= 5 m in 0 / 13, 21 / 268 elsewhere). The slot-defect fix does not act in these scenes.

**What does separate them (exploratory, not registered):** speed. Every spin is at a fed speed >= 23 m/s at k 17: 0 / 235 scenes below 20 m/s
spin, 2 / 21 at 20-25, 5 / 13 at 25-30, 6 / 12 at >= 30 m/s (RSG: 2, both >= 30 m/s, and both also spin under RV2a). With speed goes the
ordered layout's new `travelled` scalar (min(trav, 200) / 100): 39-66 m at k 17 in the 13 spin scenes vs median 16 m (p90 36 m) in the
other 268; d_turn reads 60 m (no turn) in all 13. The scalar map is zero-initialised and trained on nuPlan rows; whether trav values
this large at the anchor were seen in training is not checked here. Lead: the ordered head's scalars (most likely `travelled`) push the
plan off at highway speed; a check would be RV2a served with the scalars zeroed or trav clipped on these 13 scenes (GPU, ~10 min).
The 10 new non-turn zeros: 8 are spin scenes (06b3e399, 26004ca8, 3fa5e06b, 4a863a41, 4f7cb072, 8120803d, bbc53aa8, e7db8971); the
other two (404541da offroad at ~4-6 m/s, 7750265d collision) diverge late (k 52 / 33) with the same memory state in both runs.

## spin under rv2a: 13 scenes

- k 17 plan differs (|dh2| >= 10 deg or |dy2| >= 0.5 m): 0 / 13; median |dh2| 0.0 deg, |dy2| 0.01 m; median max over k 17-27 |dh2| 42.5 deg
- pre-handoff (k 0-16, identical inputs) max |plan diff| median 0.76 m; tracks diverge (> 0.05 m) at median k 20
- first route message before the k 17 decision: 13 / 13; ordered start >= 5 m from projection start at k 17 (rv2a): 0 / 13

## no spin under rv2a: 268 scenes

- k 17 plan differs (|dh2| >= 10 deg or |dy2| >= 0.5 m): 0 / 268; median |dh2| 0.0 deg, |dy2| 0.00 m; median max over k 17-27 |dh2| 0.1 deg
- pre-handoff (k 0-16, identical inputs) max |plan diff| median 0.22 m; tracks diverge (> 0.05 m) at median k 41
- first route message before the k 17 decision: 268 / 268; ordered start >= 5 m from projection start at k 17 (rv2a): 21 / 268

## Scenes with a spin under either label (k 17 fields)

| scene | spin a/b | score a / b | k17 h2 a / b (deg) | k17 y2 a / b (m) | k17 arc4 a / b | k17 v | pre-k17 max diff m | tracks diverge k | first msg rel k17 (s) | msgs k17 | valid a / b | first slot a / b | gap b (m) |
|:--|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|:--|:--|--:|
| 06b3e399 | 0/1 | 0.86 / 0.00 corridor | +2.5 / +2.5 | +1.49 / +1.50 | 138.9 / 139.1 | 36.8 | 0.61 | 21 | -1.7 | 18 | 40 / 40 | -6 / -6 | -0.0 |
| 26004ca8 | 0/1 | 0.32 / 0.00 offroad | -13.3 / -13.3 | -5.73 / -5.72 | 96.2 / 96.4 | 27.4 | 0.30 | 20 | -1.7 | 18 | 39 / 39 | -4 / -4 | -0.0 |
| 2eaf1872 | 0/1 | 0.00 / 0.00 corridor | +3.8 / +3.8 | +1.43 / +1.42 | 89.6 / 89.4 | 25.8 | 1.02 | 19 | -1.7 | 18 | 37 / 37 | 0 / 0 | +0.0 |
| 32be03e4 | 1/1 | 0.00 / 0.00 corridor | +1.0 / +1.0 | +0.45 / +0.46 | 124.3 / 124.5 | 37.9 | 0.90 | 19 | -1.7 | 18 | 40 / 40 | -6 / -6 | -0.0 |
| 3fa5e06b | 0/1 | 0.59 / 0.00 offroad | +0.2 / +0.2 | +0.07 / +0.07 | 118.7 / 118.6 | 35.0 | 0.60 | 19 | -1.7 | 18 | 40 / 40 | -6 / -6 | -0.0 |
| 4a863a41 | 0/1 | 0.45 / 0.00 offroad | +4.4 / +4.4 | +1.89 / +1.90 | 86.3 / 86.0 | 25.0 | 0.76 | 20 | -1.7 | 18 | 37 / 37 | 0 / 0 | +0.0 |
| 4cb05d65 | 1/1 | 0.00 / 0.00 offroad | +0.0 / +0.0 | +0.02 / +0.02 | 124.4 / 124.4 | 38.5 | 2.19 | 19 | -1.7 | 18 | 40 / 40 | -6 / -6 | +0.0 |
| 4f7cb072 | 0/1 | 0.43 / 0.00 offroad | +8.0 / +8.0 | +3.41 / +3.40 | 81.9 / 81.9 | 23.1 | 13.57 | 20 | -1.7 | 18 | 35 / 35 | 4 / 4 | -0.0 |
| 596af0c7 | 0/1 | 0.41 / 0.00 offroad | -7.1 / -7.0 | -2.18 / -2.15 | 93.1 / 93.3 | 26.5 | 0.35 | 20 | -1.7 | 18 | 38 / 38 | -2 / -2 | +0.0 |
| 8120803d | 0/1 | 0.48 / 0.00 offroad | -2.9 / -2.9 | -1.68 / -1.67 | 108.6 / 108.7 | 29.7 | 0.35 | 20 | -1.7 | 18 | 40 / 40 | -6 / -6 | +0.0 |
| bbc53aa8 | 0/1 | 0.53 / 0.00 offroad | +0.3 / +0.3 | +0.21 / +0.22 | 114.4 / 114.5 | 35.0 | 0.34 | 19 | -1.7 | 18 | 40 / 40 | -6 / -6 | -0.0 |
| e7db8971 | 0/1 | 0.84 / 0.00 corridor | -0.1 / -0.1 | -0.04 / -0.05 | 99.8 / 99.4 | 28.2 | 13.66 | 20 | -1.7 | 18 | 39 / 39 | -4 / -4 | +0.0 |
| fcd8cc77 | 0/1 | 0.38 / 0.00 corridor | -0.2 / -0.2 | -0.10 / -0.09 | 114.6 / 114.8 | 32.5 | 0.88 | 20 | -1.7 | 18 | 40 / 40 | -6 / -6 | +0.0 |

## rv2a's new non-turn zeros (10): at k 17, then 3 s before the zero event (row marked -3s)

| scene | spin a/b | score a / b | k17 h2 a / b (deg) | k17 y2 a / b (m) | k17 arc4 a / b | k17 v | pre-k17 max diff m | tracks diverge k | first msg rel k17 (s) | msgs k17 | valid a / b | first slot a / b | gap b (m) |
|:--|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|:--|:--|--:|
| 06b3e399 | 0/1 | 0.86 / 0.00 corridor | +2.5 / +2.5 | +1.49 / +1.50 | 138.9 / 139.1 | 36.8 | 0.61 | 21 | -1.7 | 18 | 40 / 40 | -6 / -6 | -0.0 |
| 26004ca8 | 0/1 | 0.32 / 0.00 offroad | -13.3 / -13.3 | -5.73 / -5.72 | 96.2 / 96.4 | 27.4 | 0.30 | 20 | -1.7 | 18 | 39 / 39 | -4 / -4 | -0.0 |
| 3fa5e06b | 0/1 | 0.59 / 0.00 offroad | +0.2 / +0.2 | +0.07 / +0.07 | 118.7 / 118.6 | 35.0 | 0.60 | 19 | -1.7 | 18 | 40 / 40 | -6 / -6 | -0.0 |
| 404541da | 0/0 | 0.39 / 0.00 offroad | +0.5 / +0.5 | +0.05 / +0.05 | 20.8 / 20.7 | 5.9 | 6.11 | 52 | -1.7 | 18 | 20 / 20 | 34 / 34 | +0.0 |
| 404541da -3s | 0/0 | 0.39 / 0.00 offroad | +1.3 / +1.2 | +0.07 / +0.06 | 15.1 / 16.1 | 4.0 | 6.11 | 52 | -1.7 | 70 | 33 / 33 | 8 / 8 | +0.0 |
| 4a863a41 | 0/1 | 0.45 / 0.00 offroad | +4.4 / +4.4 | +1.89 / +1.90 | 86.3 / 86.0 | 25.0 | 0.76 | 20 | -1.7 | 18 | 37 / 37 | 0 / 0 | +0.0 |
| 4f7cb072 | 0/1 | 0.43 / 0.00 offroad | +8.0 / +8.0 | +3.41 / +3.40 | 81.9 / 81.9 | 23.1 | 13.57 | 20 | -1.7 | 18 | 35 / 35 | 4 / 4 | -0.0 |
| 7750265d | 0/0 | 1.00 / 0.00 collision | -0.1 / -0.1 | -0.06 / -0.06 | 72.0 / 71.9 | 19.6 | 0.31 | 33 | -1.7 | 18 | 32 / 32 | 10 / 10 | +0.0 |
| 7750265d -3s | 0/0 | 1.00 / 0.00 collision | -0.8 / -0.5 | -0.02 / +0.05 | 12.1 / 12.2 | 5.8 | 0.31 | 33 | -1.7 | 159 | 22 / 21 | -6 / -6 | +0.8 |
| 8120803d | 0/1 | 0.48 / 0.00 offroad | -2.9 / -2.9 | -1.68 / -1.67 | 108.6 / 108.7 | 29.7 | 0.35 | 20 | -1.7 | 18 | 40 / 40 | -6 / -6 | +0.0 |
| bbc53aa8 | 0/1 | 0.53 / 0.00 offroad | +0.3 / +0.3 | +0.21 / +0.22 | 114.4 / 114.5 | 35.0 | 0.34 | 19 | -1.7 | 18 | 40 / 40 | -6 / -6 | -0.0 |
| e7db8971 | 0/1 | 0.84 / 0.00 corridor | -0.1 / -0.1 | -0.04 / -0.05 | 99.8 / 99.4 | 28.2 | 13.66 | 20 | -1.7 | 18 | 39 / 39 | -4 / -4 | +0.0 |
