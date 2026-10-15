# Recovery-expert rewind: RSG-L18-s0 failures on the PAI strict turns, expert XP-L12-E46 from the rewound state (box, JEV_TRACK 0)

Scenes read 36; identity outcome: completed 5, event 22, off 5, short_rear 2, short_slow 2; failed (rewound) 27 (t_fail from the zero event 22, from leaving 2.5 m 5).

| Delta | n | expert completed | policy (identity) completed | paired diff [95 % CI] | expert settle <= 1 m (median s, n settled) | v(h + 3 s) >= 2 m/s | rear-ended after h | at-fault zero | shape gate (of completed) | lat at h (median m) | v at h (median m/s) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 s | 27 | 23 / 27 = 85.2 [70.4, 96.3] % | 0 / 27 | 85.2 [70.4, 96.3] pp | 1.4 (21) | 27 / 27 | 4 | 5 | 7 / 23 | 0.91 | 5.35 |
| 2 s | 27 | 22 / 27 = 81.5 [66.7, 96.3] % | 0 / 27 | 81.5 [66.7, 96.3] pp | 0.0 (27) | 26 / 27 | 3 | 3 | 15 / 22 | 0.68 | 4.69 |
| 3 s | 26 | 22 / 26 = 84.6 [69.2, 96.2] % | 0 / 26 | 84.6 [69.2, 96.2] pp | 0.0 (26) | 24 / 26 | 2 | 1 | 16 / 22 | 0.83 | 4.93 |

Exploratory (Amendment 1): expert outcome per Delta, and completion among units the expert was not rear-ended in

| Delta | completed | event (collision / offroad / corridor) | off | short_rear | short_slow | completed / (n - short_rear) |
|--:|--:|--:|--:|--:|--:|--:|
| 1 s | 23 | 3 (3 / 0 / 0) | 0 | 0 | 1 | 23 / 27 |
| 2 s | 22 | 2 (2 / 0 / 0) | 0 | 1 | 2 | 22 / 26 |
| 3 s | 22 | 2 (1 / 1 / 0) | 0 | 1 | 1 | 22 / 25 |

Read: units = strict turns the identity (RSG-L18-s0 alone, the same box session) failed (`event` / `off`) whose handover h = k_fail - 10 Delta is >= 17. Policy control = the identity rollout (box sessions are bit-identical, so the policy handed back the rewound state reproduces it; checked per session below). CI: bootstrap over scenes (10 000).

## Reference: the expert from the scene's start (xp0) and the policy alone (id)

- all read scenes: expert 29 / 36, policy 5 / 36
- on the rewound (failed) units: expert from start 24 / 27
- failed units the expert also fails from the start: 1e8e939e (event/offroad), a174faed (event/offroad), e37aa159 (short_rear/-)

## Rewind check (expert sessions against the identity session of the same scene, before h)

- sessions 80; served plans before h bit-identical 80 / 80 (max 0); controller track before h identical 80 / 80 (max 0); |v(h) expert - v(h) identity| max 0 m/s

Box identity vs Tokyo r281/rsg outcome on 36 scenes: equal 33; 09528162 off->event, 132ae6cb completed->off, ea468790 off->completed

## Registered lines

- Delta 1 s: expert 85.2 [70.4, 96.3] % (line >= 60), paired 85.2 [70.4, 96.3] pp (line >= +30); creep check 100 % with v(h + 3 s) >= 2 m/s (line >= 80); shape gate 7 / 23 completed
- Delta 2 s: expert 81.5 [66.7, 96.3] % (line >= 60), paired 81.5 [66.7, 96.3] pp (line >= +30); creep check 96 % with v(h + 3 s) >= 2 m/s (line >= 80); shape gate 15 / 22 completed
- Delta 3 s: expert 84.6 [69.2, 96.2] % (line >= 60), paired 84.6 [69.2, 96.2] pp (line >= +30); creep check 92 % with v(h + 3 s) >= 2 m/s (line >= 80); shape gate 16 / 22 completed
- RECOVERS at Delta 1 s, 2 s, 3 s
