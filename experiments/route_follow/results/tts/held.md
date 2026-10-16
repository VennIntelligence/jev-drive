# TTS held 460 @18: RSG-L18-s0-tts vs the stored RSG-L18-s0 read

Paired 452 scenes, log-clustered 95 % CI (10000 draws). Serve = rm_chain.sh read (the stored read's path) + JEV_TTS=1 with the frozen weights (results/tts/weights.json).

| line | value [95 % CI] | registered | result |
|:--|:--|:--|:--|
| held zeros d | +11.0 [+1.1, +20.7] | <= +3 (point) | FAIL |
| held at-fault zeros d | +6.0 [-2.4, +14.0] | <= +3 (point) | FAIL |
| held turns completed d (of 24) | -1.0 [-6.0, +4.0] | >= +3 (or the PAI line) | not met |
| held mean score d | -0.0169 [-0.0377, +0.0032] | reported | - |
| held GT distance d (m) | +0.133 [+0.050, +0.219] | reported | - |

Zeros by kind (a scene can count in several kinds): zero 37 vs 26 (d +11, CI +1.1 .. +20.7); z_atf 27 vs 21 (d +6, CI -2.4 .. +14.0); z_col 9 vs 4 (d +5, CI +0.0 .. +10.2); z_off 18 vs 17 (d +1, CI -4.8 .. +6.9); z_cor 12 vs 6 (d +6, CI +1.2 .. +11.7).
Turns: RSG-L18-s0-tts 15 / 24; RSG-L18-s0 16 / 24.

Stage 2 (PAI 281): not run (held zeros d > +3).

Switch statistics (8129 decisions, 452 scenes): switched 683 (8.4 %), in 231 scenes; reasons {'own': 7446, 'switch': 683}.

| segment | decisions | switched | share | candidate types picked |
|:--|--:|--:|--:|:--|
| approach | 2379 | 358 | 15.0 % | {'gain': 136, 'lat': 146, 'arc': 16, 'gain_arc': 60} |
| junction | 28 | 2 | 7.1 % | {'lat': 2} |
| straight | 5697 | 323 | 5.7 % | {'lat': 207, 'gain': 52, 'arc': 3, 'gain_arc': 61} |
| none | 25 | 0 | 0.0 % | {} |

Picked candidates: {'lat-0.5': 129, 'g0.8': 106, 'lat+0.5': 96, 'lat+1': 69, 'lat-1': 61, 'g1.2': 49, 'g0.8s0.9': 44, 'g1.4': 33, 'g1.4s0.9': 25, 'g1.4s1.1': 20, 'g0.8s1.1': 17, 's0.9': 15, 'g1.2s0.9': 10, 'g1.2s1.1': 5, 's1.1': 4}.
Latency: tts step p50 / p95 / p99 [8.38, 17.89, 25.31] ms; driver total per inference p50 / p95 [37.25, 67.03] ms vs reference [22.97, 42.05] ms.
