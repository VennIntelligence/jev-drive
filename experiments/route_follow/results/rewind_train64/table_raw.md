# Recovery-expert rewind on the nuPlan @18 train turns: RSG-L18-s0 (box, held-read serving), expert XP-L12-E46

Turn scenes read 63; identity outcome completed 21, event 23, off 9, short 10. Stratum A (failed) 42: off 9, short_end 10, zero 23; stratum B (completed) 21: first-violation 4, no violation (control) 17. Gate thresholds: kmax 0.184 1/m, dk 0.0425 1/m^2, alat 4.42 m/s^2, entry speed 5.6 m/s.

Outcome columns are completed / event / off / short (rf_follow; short = still on the route when the rollout ended, its own outcome).

| stratum | Delta | units (h >= 1) | expert completes [95 % CI] | expert c / e / o / s | policy (identity, same length) c / e / o / s | paired expert - policy [95 % CI] | handover within tolerance | settle <= 1 m (n, median s) | v(h + 3 s) >= 2 m/s | rear after h | at-fault zero | gate pass / completed | gate pass / handovers | rows in turn window after h (sum; completed + gated) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A | 1 s | 41 / 42 | 30 = 0.73 [0.58, 0.87] | 30/11/0/0 | 10/23/8/0 | 0.49 [0.31, 0.66] | 35 / 41 | 33, 1.8 | 37 / 40 | 11 | 11 | 7 / 30 | 7 / 41 | 267; 48 |
| A | 2 s | 41 / 42 | 33 = 0.80 [0.66, 0.93] | 33/7/0/1 | 10/23/8/0 | 0.56 [0.36, 0.74] | 40 / 41 | 40, 1.1 | 38 / 41 | 11 | 11 | 15 / 33 | 15 / 41 | 345; 122 |
| A | 3 s | 40 / 42 | 31 = 0.78 [0.62, 0.90] | 31/8/0/1 | 10/22/8/0 | 0.53 [0.33, 0.71] | 39 / 40 | 40, 0.7 | 38 / 40 | 9 | 11 | 23 / 31 | 23 / 40 | 405; 230 |
| A-event | 1 s | 22 / 23 | 12 = 0.55 [0.33, 0.75] | 12/10/0/0 | 0/22/0/0 | 0.55 [0.33, 0.75] | 18 / 22 | 14, 1.9 | 19 / 21 | 7 | 9 | 3 / 12 | 3 / 22 | 157; 26 |
| A-event | 2 s | 22 / 23 | 16 = 0.73 [0.52, 0.91] | 16/5/0/1 | 0/22/0/0 | 0.73 [0.52, 0.91] | 21 / 22 | 21, 1.3 | 21 / 22 | 9 | 8 | 5 / 16 | 5 / 22 | 205; 47 |
| A-event | 3 s | 21 / 23 | 14 = 0.67 [0.48, 0.86] | 14/6/0/1 | 0/21/0/0 | 0.67 [0.48, 0.86] | 20 / 21 | 21, 0.0 | 20 / 21 | 6 | 8 | 8 / 14 | 8 / 21 | 235; 97 |
| A-off | 1 s | 9 / 9 | 9 = 1.00 [1.00, 1.00] | 9/0/0/0 | 1/1/7/0 | 0.89 [0.73, 1.00] | 9 / 9 | 9, 2.2 | 9 / 9 | 1 | 1 | 2 / 9 | 2 / 9 | 63; 11 |
| A-off | 2 s | 9 / 9 | 9 = 1.00 [1.00, 1.00] | 9/0/0/0 | 1/1/7/0 | 0.89 [0.73, 1.00] | 9 / 9 | 9, 1.8 | 9 / 9 | 0 | 1 | 6 / 9 | 6 / 9 | 79; 50 |
| A-off | 3 s | 9 / 9 | 9 = 1.00 [1.00, 1.00] | 9/0/0/0 | 1/1/7/0 | 0.89 [0.73, 1.00] | 9 / 9 | 9, 0.0 | 9 / 9 | 1 | 1 | 8 / 9 | 8 / 9 | 94; 82 |
| A-short | 1 s | 10 / 10 | 9 = 0.90 [0.67, 1.00] | 9/1/0/0 | 9/0/1/0 | 0.00 [-0.33, 0.25] | 8 / 10 | 10, 0.0 | 9 / 10 | 3 | 1 | 2 / 9 | 2 / 10 | 47; 11 |
| A-short | 2 s | 10 / 10 | 8 = 0.80 [0.50, 1.00] | 8/2/0/0 | 9/0/1/0 | -0.10 [-0.50, 0.20] | 10 / 10 | 10, 0.5 | 8 / 10 | 2 | 2 | 4 / 8 | 4 / 10 | 61; 25 |
| A-short | 3 s | 10 / 10 | 8 = 0.80 [0.50, 1.00] | 8/2/0/0 | 9/0/1/0 | -0.10 [-0.50, 0.20] | 10 / 10 | 10, 1.0 | 9 / 10 | 2 | 2 | 7 / 8 | 7 / 10 | 76; 51 |
| B | 1 s | 1 / 4 | 1 = 1.00 [1.00, 1.00] | 1/0/0/0 | 1/0/0/0 | 0.00 [0.00, 0.00] | 1 / 1 | 1, 0.0 | 1 / 1 | 0 | 0 | 1 / 1 | 1 / 1 | 16; 16 |
| B | 2 s | 0 / 4 | 0 = - | 0/0/0/0 | 0/0/0/0 | - | 0 / 0 | 0, nan | 0 / 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0; 0 |
| B | 3 s | 0 / 4 | 0 = - | 0/0/0/0 | 0/0/0/0 | - | 0 / 0 | 0, nan | 0 / 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0; 0 |

Read: units, strata and k_ref from the stored 18-decision train-pool rollouts; h = k_ref - 2 Delta >= 1 (decision 0 is the runtime's forced-log step). Every session of this read runs 30 decisions (15 s; Amendment 2). Policy = RSG-L18-s0 alone at the same length (`id`), so it is no longer 0 / 1 by construction. Handover within tolerance = before h served plans <= 0.15 m and track <= 5 cm from `id`, |v(h) - v_id(h)| <= 0.05 m/s. CI: bootstrap over logs (scenes of one log drawn together), 10 000. Rows = expert decisions k >= h with route station in [entry - 20, exit + 10] m before any zero (rf_follow's junction stretch, 2 Hz).

## Expert outcomes per stratum and Delta

| stratum | Delta | completed | event (collision / offroad / corridor) | off | short |
|:--|--:|--:|--:|--:|--:|
| A | 1 s | 30 | 11 (9 / 2 / 0) | 0 | 0 |
| A | 2 s | 33 | 7 (7 / 0 / 0) | 0 | 1 |
| A | 3 s | 31 | 8 (8 / 0 / 0) | 0 | 1 |
| B | 1 s | 1 | 0 (0 / 0 / 0) | 0 | 0 |
| B | 2 s | 0 | 0 (0 / 0 / 0) | 0 | 0 |
| B | 3 s | 0 | 0 (0 / 0 / 0) | 0 | 0 |

## Stratum B: the turn after the handover, expert against the identity over the same time span (median over handovers)

| Delta | n | median abs lateral id / expert (m) | max abs lateral id / expert (m) | p95-peak kappa id / expert (1/m) | peak a_lat id / expert (m/s^2) | violated item at k_ref |
|--:|--:|--:|--:|--:|--:|:--|
| 1 s | 1 | 2.29 / 0.17 | 2.56 / 0.97 | 0.074 / 0.088 | 2.46 / 1.83 | v_entry 4 |
| 2 s | 0 | nan / nan | nan / nan | nan / nan | nan / nan | v_entry 4 |
| 3 s | 0 | nan / nan | nan / nan | nan / nan | nan / nan | v_entry 4 |

## Stratum C: the expert from the scene start (xp0) on every turn

- expert completes 48 / 63 = 0.76 [0.65, 0.86] (policy 31); outcomes completed 48, event 14, short 1
- on A units 33 / 42, on B units 15 / 21; gate pass (of completed) 46 / 48
- calibration, the human log through the same gate: all items 37 / 63; kmax 63, dk 62, alat 63, v_entry 38

## Rewind check (each D session against the identity of the same scene, before h)

- sessions 123; served plans before h bit-identical 104 / 123 (max 3.9); controller track before h identical 104 / 123 (max 2.6); |v(h) - v_id(h)| max 1.03 m/s; within the registered tolerance 115 / 123
- identity replays inside the D launches (no handover at that Delta): score and outcome equal to id 59 / 66
- the identity at 30 decisions against the stored 18-decision outcome (63 turns): completed->completed 21, event->event 23, off->completed 1, off->event 1, off->off 7, short->completed 9, short->off 1

## Flags (Amendment 2)

- decisions run by the identity: 30 on 59 / 63; fewer (the log ends, the runtime clips): 84ea5053 15, b5895541 26, 55475ba6 28, ef305e77 22
- MTGS camera coverage after the token (span.csv video_after_s): median 9.6 s, >= 14 s on 1 / 63. Past it the renderer draws outside the reconstruction: the policy's frames (identity, and the model before h) are confounded there; the expert reads the map, not the frames.

## Registered lines

- A (primary Delta 2 s): expert completion 0.80 [0.66, 0.93] on 41 units (line >= 0.60): **PASS**; Delta 1 s 0.73 [0.58, 0.87], 3 s 0.78 [0.62, 0.90]
- B (Delta 2 s): gate pass on 0 / 0 handovers = nan (line >= 0.5): **FAIL**; Delta 1 s 1 / 1, 3 s 0 / 0
- C: expert from start 0.76 [0.65, 0.86] (line >= 0.80): **FAIL**

Read errors: {'id 2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9': '(x)\n                 ^^^^^\n  File "/root/autodl-tmp/ujs/runs/route_follow/rewind_train64/code/experiments/route_follow/scripts/rf_rewind.py", line 447, in n_roll\n    recs = drive_records(D)[scene]\n           ~~~~~~~~~~~~~~~~^^^^^^^\nKeyError: \'2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9\'\n', 'xp0 2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9': '(x)\n                 ^^^^^\n  File "/root/autodl-tmp/ujs/runs/route_follow/rewind_train64/code/experiments/route_follow/scripts/rf_rewind.py", line 447, in n_roll\n    recs = drive_records(D)[scene]\n           ~~~~~~~~~~~~~~~~^^^^^^^\nKeyError: \'2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9\'\n', 'D1 2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9': '(x)\n                 ^^^^^\n  File "/root/autodl-tmp/ujs/runs/route_follow/rewind_train64/code/experiments/route_follow/scripts/rf_rewind.py", line 447, in n_roll\n    recs = drive_records(D)[scene]\n           ~~~~~~~~~~~~~~~~^^^^^^^\nKeyError: \'2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9\'\n', 'D2 2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9': '(x)\n                 ^^^^^\n  File "/root/autodl-tmp/ujs/runs/route_follow/rewind_train64/code/experiments/route_follow/scripts/rf_rewind.py", line 447, in n_roll\n    recs = drive_records(D)[scene]\n           ~~~~~~~~~~~~~~~~^^^^^^^\nKeyError: \'2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9\'\n', 'D3 2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9': '(x)\n                 ^^^^^\n  File "/root/autodl-tmp/ujs/runs/route_follow/rewind_train64/code/experiments/route_follow/scripts/rf_rewind.py", line 447, in n_roll\n    recs = drive_records(D)[scene]\n           ~~~~~~~~~~~~~~~~^^^^^^^\nKeyError: \'2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9\'\n'}
