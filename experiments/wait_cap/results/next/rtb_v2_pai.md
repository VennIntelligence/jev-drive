# V2 (RTB1AR) PAI 281 in the submission config (prereg 2026-10-17 amendment 3, exploratory, one read)

Decision 339 left V2's PAI 281 unread. Amendment 3 asked for it, and for a second base seed, which the user cancelled before any read (seeds behave alike); this file holds the one read that was kept: RTB1ARO-L18-s0 (base RTB1AR-F-s0, RSG-L18 continuation, 2 500 steps, seed 0) on the 281 PAI scenes in the submission config (`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`, `controller.gains.rel_front_steering_angle_weight=10`, ALPASIM_FAST=1), paired against the stored RSG-L18-s0 read `route_follow/pai_g10/acc` (session-clustered CI). The public and held reads of the same arm are in [rtb_v.md](rtb_v.md). Prereg: [plans/2026-10-17-retain-anticopy-draft.md](../../plans/2026-10-17-retain-anticopy-draft.md) amendment 3. Code: `scripts/nx.sh rtb-wide`, `scripts/nx.py v2pai`.

| arm | scenes | zeros | mean | paired vs RSG-L18-s0 [CI] | at-fault | offroad | front / side / rear collisions | stop17 zeros |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RTB1ARO-L18-s0 | 281 | 79 | 0.6664 | -0.007 [-0.038, 0.023] | 12 | 17 | 6 / 6 / 39 | 15 / 17 |
| RTB1O-L18-s0 | 281 | 76 | 0.6765 | 0.003 [-0.034, 0.038] | 12 | 17 | 6 / 6 / 37 | 15 / 17 |
| RSG-L18-s0 (stored) | 281 | 76 | 0.6736 | - | 9 | 16 | 5 / 4 / 40 | 16 / 17 |
| RSG-L18-s0 re-read (cl-compile lane) | 281 | 76 | 0.6736 | 0.000 [0.000, 0.000] | 9 | 16 | 5 / 4 / 40 | 16 / 17 |
| RSG-L18-s1 (cl-compile lane) | 281 | 71 | 0.6910 | 0.017 [0.001, 0.035] | 7 | 17 | 4 / 3 / 41 | 16 / 17 |


Base dev ADE, the DONE value of each training (the amendment-2 stop rule is <= the RTB1-F-s0 value): RTB1AR-F-s0 0.5519, RTB1AR-F-s1 0.5561, RTB1-F-s0 0.5556, RTB1-F-s1 0.5571.

RTB1AR-F-s1 (the second V2 base seed) had just finished training when the cancel landed: its checkpoint is kept on the box and no read was taken of it (the arrival replay and the stop rule never ran).

## Check against the decision 339 refutation conditions

R1 (the second base seed has public rear-ended >= 22) cannot be read: the second base seed was cancelled by the user before any read. R2 (PAI 281 at-fault / zeros clearly above RSG-L18-s0; flags fixed in amendment 3 before the read: zeros >= C + 5, at-fault >= C + 4, paired CI upper bound < 0): RTB1ARO-L18-s0 has 79 zeros (+3 against RSG-L18-s0's 76), 12 at-fault (+3 against 9) and a paired mean of -0.007 [-0.038, 0.023], so none of the three flags is crossed (zeros +3 against the +5 that flags, at-fault +3 against +4) and R2 is not met; stop17 is 15 / 17 against 16 / 17. For scale: the cl-compile lane's read of RSG-L18-s0 in the same config reproduces the stored read (76 zeros, 9 at-fault, paired 0.000 [0.000, 0.000]), so a PAI 281 read of one checkpoint repeated exactly in this comparison; RSG-L18-s1 reads 71 zeros, 7 at-fault, 0.017 [0.001, 0.035], which is the spread between the two RSG seeds.
