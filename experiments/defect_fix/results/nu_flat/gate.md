# NU-FLAT read-out (plans/2026-10-20-nu-flat-prereg.md; native reads; continuations of B2 `STANDFMXH-L18-s1`, control CTRL `B2C-L18-s1`)

Arena held native: 2106 frozen board-mix windows every read has (18 chunks); d399's subset: 716 of them.

## Overview (full 18 chunks unless noted)

| read | B2C-L18-s1 | B2F-L18-s1 | B2FR-L18-s1 | STANDFMXH-L18-s1 |
|:--|--:|--:|--:|--:|
| arena mean / zeros | 0.9365 / 92 | 0.9377 / 88 | 0.9374 / 74 | 0.9365 / 96 |
| frozen-pool PCS | 1386 | 1392 | 1290 | 1398 |
| easy misses (red-line zeros) | 48 (10) | 46 (11) | 51 (3) | 47 (9) |
| subset: PCS / easy misses (red-line) | 1360 / 20 (9) | 1370 / 18 (9) | 1368 / 15 (3) | 1395 / 17 (7) |
| held 460 mean / zeros / front | 0.8496 / 39 / 5 | 0.8548 / 37 / 4 | 0.8170 / 38 / 5 | 0.8527 / 41 / 4 |

| pair | windows | arena mean diff [log-clustered 95 % CI] | arena zeros | AF-front | PCS diff | held 460 mean diff |
|:--|--:|--:|--:|--:|--:|--:|
| B2C-L18-s1 - STANDFMXH-L18-s1 | 2228 | -0.0003 [-0.0055, +0.0045] | 101 -> 98 | 2 -> 4 | -12 | -0.0031 [-0.0302, +0.0177] |
| B2F-L18-s1 - B2C-L18-s1 | 2228 | +0.0012 [-0.0016, +0.0039] | 98 -> 94 | 4 -> 3 | +7 | +0.0052 [-0.0059, +0.0159] |
| B2F-L18-s1 - STANDFMXH-L18-s1 | 2228 | +0.0009 [-0.0034, +0.0051] | 101 -> 94 | 2 -> 3 | -5 | +0.0021 [-0.0253, +0.0261] |
| B2FR-L18-s1 - B2C-L18-s1 | 2228 | +0.0009 [-0.0055, +0.0086] | 98 -> 77 | 4 -> 2 | -96 | -0.0326 [-0.0561, -0.0181] |
| B2FR-L18-s1 - STANDFMXH-L18-s1 | 2228 | +0.0006 [-0.0058, +0.0075] | 101 -> 77 | 2 -> 2 | -108 | -0.0357 [-0.0792, -0.0090] |

## C `B2C-L18-s1` against `STANDFMXH-L18-s1`: continuation effect (CTRL against stored B2-s1; lines shown for reference, not a verdict)

| read | STANDFMXH-L18-s1 | B2C-L18-s1 | line |
|:--|--:|--:|:--|
| guard: wait b launched other_lead | 0.646 | 0.583 | >= 0.95 x control (binding): FAIL |
| guard: wait b launched red_queue | 0.486 | 0.4 | >= 0.95 x control (binding): FAIL |
| guard: wait b median launch delay other_lead (s) | 0.567 | 0.717 | <= control + 0.15 (binding): ok |
| guard: wait b median launch delay red_queue (s) | 0.917 | 0.792 | <= control + 0.15 (binding): ok |
| guard: wait a other_lead | 0.944 | 0.944 | >= control - 0.03: ok |
| guard: wait a red_queue | 0.949 | 0.949 | >= control - 0.03: ok |
| guard: held 460 zeros | 41 | 39 | <= control + 2: ok |
| guard: held 460 front collisions | 4 | 5 | <= control + 2: ok |
| guard: held 460 right drift, non-turn k 11-17 (arm - control, m) |  | 0.026 | > -0.05: ok |
| guard: subset AF-front windows | 0 | 1 | <= control + 2: ok |
| guard: subset easy partial: free-road ratchet / slow | 0 | 2 | <= control + 2: ok |
| guard: subset easy partial: launch late / not launching | 2 | 2 | <= control + 2: ok |
| guard: subset easy partial: near-miss 0.98-0.999 | 2 | 1 | <= control + 2: ok |
| guard: subset easy partial: rear-ended | 1 | 1 | <= control + 2: ok |
| guard: subset easy partial: stop / lock | 1 | 1 | <= control + 2: ok |
| guard: subset easy zero: corridor, moving non-turn | 3 | 3 | <= control + 2: ok |
| guard: subset easy zero: turn window, offroad / corridor | 1 | 1 | <= control + 2: ok |
| guard: subset slow partial windows | 14 | 14 | <= control + 3: ok |
| guard: subset k = 9 speed ratio | 0.9932 | 1.0083 | >= control - 0.03: ok |
| main: frozen-pool PCS (18 chunks) | 1398 | 1386 | >= control + 15: FAIL |
| description: subset easy red-line zeros (d399 line <= control - 6) | 7 | 9 | description only: ok |

- red-line easy zeros (18 chunks) fixed 2, new 3; easy misses 47 -> 48
- full easy classes (control -> arm): partial: following a lead 1 -> 0; partial: free-road ratchet / slow 3 -> 5; partial: launch late / not launching 5 -> 7; partial: near-miss 0.98-0.999 4 -> 6; partial: rear-ended 1 -> 1; partial: stop / lock 2 -> 3; zero: AF-side collision 1 -> 0; zero: corridor, moving non-turn 7 -> 4; zero: offroad, non-turn 5 -> 5; zero: static, other cue 1 -> 0; zero: static, red line, no lead 9 -> 10; zero: turn window, offroad / corridor 8 -> 7

## F `B2F-L18-s1` against `B2C-L18-s1`: FAIL (launch guard FAIL)

| read | B2C-L18-s1 | B2F-L18-s1 | line |
|:--|--:|--:|:--|
| guard: wait b launched other_lead | 0.583 | 0.542 | >= 0.95 x control (binding): FAIL |
| guard: wait b launched red_queue | 0.4 | 0.371 | >= 0.95 x control (binding): FAIL |
| guard: wait b median launch delay other_lead (s) | 0.717 | 0.767 | <= control + 0.15 (binding): ok |
| guard: wait b median launch delay red_queue (s) | 0.792 | 0.617 | <= control + 0.15 (binding): ok |
| guard: wait a other_lead | 0.944 | 0.944 | >= control - 0.03: ok |
| guard: wait a red_queue | 0.949 | 0.949 | >= control - 0.03: ok |
| guard: held 460 zeros | 39 | 37 | <= control + 2: ok |
| guard: held 460 front collisions | 5 | 4 | <= control + 2: ok |
| guard: held 460 right drift, non-turn k 11-17 (arm - control, m) |  | 0.014 | > -0.05: ok |
| guard: subset AF-front windows | 1 | 1 | <= control + 2: ok |
| guard: subset easy partial: free-road ratchet / slow | 2 | 1 | <= control + 2: ok |
| guard: subset easy partial: launch late / not launching | 2 | 2 | <= control + 2: ok |
| guard: subset easy partial: near-miss 0.98-0.999 | 1 | 0 | <= control + 2: ok |
| guard: subset easy partial: rear-ended | 1 | 1 | <= control + 2: ok |
| guard: subset easy partial: stop / lock | 1 | 1 | <= control + 2: ok |
| guard: subset easy zero: corridor, moving non-turn | 3 | 3 | <= control + 2: ok |
| guard: subset easy zero: turn window, offroad / corridor | 1 | 1 | <= control + 2: ok |
| guard: subset slow partial windows | 14 | 16 | <= control + 3: ok |
| guard: subset k = 9 speed ratio | 1.0083 | 1.0011 | >= control - 0.03: ok |
| main: frozen-pool PCS (18 chunks) | 1386 | 1392 | >= control + 15: FAIL |
| description: subset easy red-line zeros (d399 line <= control - 6) | 9 | 9 | description only: ok |

- red-line easy zeros (18 chunks) fixed 1, new 2; easy misses 48 -> 46
- full easy classes (control -> arm): partial: free-road ratchet / slow 5 -> 6; partial: launch late / not launching 7 -> 8; partial: near-miss 0.98-0.999 6 -> 1; partial: rear-ended 1 -> 1; partial: stop / lock 3 -> 3; zero: corridor, moving non-turn 4 -> 4; zero: offroad, non-turn 5 -> 5; zero: static, red line, no lead 10 -> 11; zero: turn window, offroad / corridor 7 -> 7

## FR `B2FR-L18-s1` against `B2C-L18-s1`: FAIL (launch guard FAIL)

| read | B2C-L18-s1 | B2FR-L18-s1 | line |
|:--|--:|--:|:--|
| guard: wait b launched other_lead | 0.583 | 0.438 | >= 0.95 x control (binding): FAIL |
| guard: wait b launched red_queue | 0.4 | 0.257 | >= 0.95 x control (binding): FAIL |
| guard: wait b median launch delay other_lead (s) | 0.717 | 0.767 | <= control + 0.15 (binding): ok |
| guard: wait b median launch delay red_queue (s) | 0.792 | 0.817 | <= control + 0.15 (binding): ok |
| guard: wait a other_lead | 0.944 | 0.972 | >= control - 0.03: ok |
| guard: wait a red_queue | 0.949 | 1.0 | >= control - 0.03: ok |
| guard: held 460 zeros | 39 | 38 | <= control + 2: ok |
| guard: held 460 front collisions | 5 | 5 | <= control + 2: ok |
| guard: held 460 right drift, non-turn k 11-17 (arm - control, m) |  | -0.001 | > -0.05: ok |
| guard: subset AF-front windows | 1 | 1 | <= control + 2: ok |
| guard: subset easy partial: free-road ratchet / slow | 2 | 2 | <= control + 2: ok |
| guard: subset easy partial: launch late / not launching | 2 | 2 | <= control + 2: ok |
| guard: subset easy partial: near-miss 0.98-0.999 | 1 | 2 | <= control + 2: ok |
| guard: subset easy partial: rear-ended | 1 | 1 | <= control + 2: ok |
| guard: subset easy partial: stop / lock | 1 | 1 | <= control + 2: ok |
| guard: subset easy zero: corridor, moving non-turn | 3 | 2 | <= control + 2: ok |
| guard: subset easy zero: offroad, non-turn | 0 | 1 | <= control + 2: ok |
| guard: subset easy zero: turn window, offroad / corridor | 1 | 1 | <= control + 2: ok |
| guard: subset slow partial windows | 14 | 17 | <= control + 3: ok |
| guard: subset k = 9 speed ratio | 1.0083 | 0.9941 | >= control - 0.03: ok |
| main: frozen-pool PCS (18 chunks) | 1386 | 1290 | >= control + 15: FAIL |
| main: easy red-line zeros (18 chunks) | 10 | 3 | <= control - 3: ok |
| description: subset easy red-line zeros (d399 line <= control - 6) | 9 | 3 | description only: ok |

- red-line easy zeros (18 chunks) fixed 7, new 0; easy misses 48 -> 51
- full easy classes (control -> arm): partial: free-road ratchet / slow 5 -> 14; partial: launch late / not launching 7 -> 9; partial: near-miss 0.98-0.999 6 -> 2; partial: other slow 0 -> 2; partial: rear-ended 1 -> 2; partial: stop / lock 3 -> 4; zero: corridor, moving non-turn 4 -> 3; zero: offroad, non-turn 5 -> 5; zero: static, red line, no lead 10 -> 3; zero: turn window, offroad / corridor 7 -> 7
