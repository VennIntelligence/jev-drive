# vred3: offline readings that fix the yellow rule's parameters

Plan: [plans/2026-10-02-pbyp2-vred2.md](../plans/2026-10-02-pbyp2-vred2.md) section 11. Logged answers of Qwen3-VL-4B (the `vred` serving) with the simulator's light state of the same frame: shadow runs `cal`, `cal2`, `cal3` (the car drives as `drive`; seed 1, 19 routes, three repeats of the same traffic seed, so repeats are not independent) and the in-loop `vred` runs. Code: `scripts/vred3_offline.py`.

## 1. Yellow duration

Complete green -> yellow -> red sequences of every light in the logs (143, over 13 routes and 64 lights; samples every 0.5 s, so each duration is known to within a bracket of 1.0 s).

Yellow samples per sequence (0.5 s each; a 3.0 s yellow gives 6, a 2.5 s one 5 or 6, a 3.5 s one 6 or 7): {1: 1, 2: 1, 6: 123, 7: 16, 10: 2}. Duration bracket [0.0, 5.5] s over all sequences; share of sequences whose bracket contains 3.0 s: 86.0%; contains 2.0 s: 0.0%; contains 4.0 s: 11.2%.

|   route |   sequences |   n_yellow_min |   n_yellow_max |
|--------:|------------:|---------------:|---------------:|
|   15102 |           6 |              6 |              6 |
|   15483 |           2 |              6 |              6 |
|   15612 |           4 |              1 |             10 |
|   16508 |           5 |              6 |              6 |
|   16529 |          11 |              6 |              7 |
|   19324 |           5 |              7 |              7 |
|   19832 |           5 |              6 |              6 |
|   24944 |          39 |              6 |              7 |
|    2520 |           5 |              6 |              6 |
|   27297 |          10 |              6 |              6 |
|   27870 |          15 |              6 |              6 |
|   28147 |           2 |              6 |              6 |
|    9196 |          34 |              6 |              6 |

Reading: the sequences that are complete and sampled without gaps give 6 yellow samples, a 3.0 s yellow; the registered `yellow_s` = 3.0 s. Sequences with other counts are listed in `yellow_durations.csv` (gaps in the sampling when the ego stopped answering or left the 60 m window).

## 2. Inferring the yellow from the answer sequence (option b)

Truth frames of yellow (ego light within 50 m): 48; answered `red_or_yellow_for_ego`: 42 (87.5%). Option (a), a yellow option in the question, would need a changed prompt, a changed server and a new Phase A on 48 yellow frames: not run; the inference needs no model change.

True green -> yellow onsets of the ego light within 50 m: 7 (7 detected, i.e. a yellow frame was answered non-green before the yellow ended; 0 never). Detection lag in frames (0 = the first yellow frame is answered non-green): {0: 6, 2: 1}.

Every onset (`lag` = yellow frames answered green before the first non-green answer; deficit = true age with the earliest possible onset minus the age the rule assumes):

| unit          |   route |   t_onset_hi |   tl_dist |   lag_frames |   age_assumed |   age_true_min |   age_true_max |   deficit_max |
|:--------------|--------:|-------------:|----------:|-------------:|--------------:|---------------:|---------------:|--------------:|
| v2-cal-s1-q1  |   24944 |        39.55 |     24.16 |            0 |           0.5 |              0 |            0.5 |             0 |
| v2-cal-s1-q1  |   27297 |         9.55 |      5.34 |            0 |           0.5 |              0 |            0.5 |             0 |
| v2-cal2-s1-q1 |   24944 |        39.55 |      8.16 |            0 |           0.5 |              0 |            0.5 |             0 |
| v2-cal2-s1-q1 |   24944 |        54.55 |     30.7  |            2 |          16.5 |              1 |            1.5 |           -15 |
| v2-cal2-s1-q1 |   27297 |         9.55 |      5.34 |            0 |           0.5 |              0 |            0.5 |             0 |
| v2-cal3-s1-q1 |   24944 |        39.55 |     15.16 |            0 |           0.5 |              0 |            0.5 |             0 |
| v2-cal3-s1-q1 |   27297 |         9.55 |      5.34 |            0 |           0.5 |              0 |            0.5 |             0 |

Age of the yellow at the first non-green answer: the rule assumes `age_assumed` = frame time of the first non-green answer minus the frame time of the last green answer (the yellow may have started right after it). Truth: the onset lies between the last green sample and the first yellow sample. The deficit `true - assumed` is positive when the rule assumes a younger yellow than the truth (not conservative): this happens when the model keeps answering green into the yellow.

| quantity | median | p90 | max |
|:--|--:|--:|--:|
| age assumed [s] | 0.50 | 6.90 | 16.50 |
| true age, mid of the bracket [s] | 0.25 | 0.65 | 1.25 |
| deficit, true (mid) - assumed [s] | -0.25 | -0.25 | -0.25 |
| deficit, true (earliest onset) - assumed [s] | 0.00 | 0.00 | 0.00 |

The registered `lag_margin_s` = 0.5 s is subtracted from the remaining yellow time: it covers the p90 of the earliest-onset deficit (0.00 s) when that is <= 0.5 s.

## 3. What reacting to the first non-green answer costs in false starts

Green frames of the ego light within 50 m: 408 in 26 (run, light) approaches. Answer transitions green -> `red_or_yellow_for_ego` while the truth is green: 11 (2.70 per 100 green frames, 42.3 per 100 approaches); 0 of them single (the next answer is green again: the second answer cancels the tentative stop), 11 persistent. **Before the stop line (`tl_dist` > 0, where the rule can start a stop): 1 transitions, 0 single.** The others are after the front bumper has passed the line, where the commit rule starts nothing.

Distance of the ego to the stop line at these false transitions (m): median -3.8, p10 -4.8, min -4.8; 11 within 15 m.

Cost model: a false first answer starts a tentative R2 hold (the IDM stop constraint) for one answer period (0.5 s plus the answer delay) before the second answer cancels it; a persistent one is the same as a K = 2 hold started 0.5 s earlier. The speed cost of a tentative hold is measured in the batch (`vred3.md`: tentative starts, cancellations).

## 4. What the rule would have decided at the yellow encounters of the vred runs (hypothetical)

The logged speed, distance and answer times of the `vred` in-loop runs at each green -> non-green answer transition with a yellow truth frame. These cars braked under `vred`'s rules and had no R1 cap, so speeds differ from what `vred3` would drive; the table checks the decision function on real states, it is not an outcome.

| unit          |   route |    t |    v |   d_stop_target |   d_line |   age |   remaining | decision   |   stop_need |
|:--------------|--------:|-----:|-----:|----------------:|---------:|------:|------------:|:-----------|------------:|
| v2-vred-s0-q1 |   24944 | 39.9 | 3.11 |           15.66 |    24.62 |  0.85 |        1.65 | stop       |        3.17 |
| v2-vred-s0-q1 |   27297 |  9.9 | 5.32 |            4.84 |     7.25 |  0.85 |        1.65 | stop_hard  |        7.38 |
| v2-vred-s1-q1 |   24944 | 39.9 | 4.52 |           17.66 |    26.14 |  0.85 |        1.65 | stop       |        5.67 |
| v2-vred-s1-q1 |   27297 |  9.9 | 4.66 |            4.84 |     7.59 |  0.85 |        1.65 | stop_hard  |        5.95 |

