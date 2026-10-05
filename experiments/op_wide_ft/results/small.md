# op_wide_ft small set: 9 of the 25 B2D turns (gate read)

took = the car left through the commanded exit. ratio = peak desired curvature towards the commanded side over the turn span / needed 1 / R_min (entered turns; rft_split s_peak / need); forced-turn ratios exclude 28180 (needs 0.213 > op-path MAX_CURVATURE 0.2). turn-in = arc of the car past the turn start when the action first reaches 0.5 / R_min (negative = before). Desire off, zones off, curv (op-path), spec camera 1.86 m, seed 2, one run per cell.

| arm | took | choice | forced | entered | late | turn-in m, median (n) | ratio forced (median) | ratio choice (median) | collisions |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| shipped@doff | 2 / 9 | 1 | 1 | 9 | 1 | 3.00 (4) | 0.48 | 0.21 | 3 |
| wf-w58-s0@58 | 0 / 9 | 0 | 0 | 8 | 5 | 8.50 (6) | 0.94 | 0.70 | 1 |
| wf-w116-s0@116 | 0 / 9 | 0 | 0 | 8 | 4 | 7.15 (4) | 0.36 | 0.65 | 1 |

Paired (route-cluster bootstrap, jevdrive.stats.paired):

| contrast | took | ratio forced | ratio choice | turn-in m |
|---|---|---|---|---|
| wf-w116-s0@116 - wf-w58-s0@58 | +0.000 [+0.000, +0.000] (n 9) | +0.808 [-0.815, +3.569] (n 3) | +0.184 [-0.231, +0.863] (n 4) | -0.567 [-2.000, +1.000] (n 3) |

Per turn (took / cause / ratio / turn-in m):

| turn | forced | R_min | shipped@doff | wf-w58-s0@58 | wf-w116-s0@116 |
|---|--:|--:|---|---|---|
| 10255:0 | 0 | 6.4 | no / not chosen / 0.13 / - | no / late / 0.76 / 8.5 | no / late / 0.75 / 7.8 |
| 15102:0 | 0 | 11.5 | no / not chosen / 0.21 / - | no / late / 0.69 / 12.2 | no / not chosen / 0.40 / - |
| 25051:0 | 1 | 6.2 | no / wrong side / 0.48 / - | no / late / 0.69 / 7.0 | no / not chosen / 0.36 / - |
| 27994:0 | 1 | 7.0 | no / wrong side / 0.15 / - | no / late / 1.06 / 11.5 | no / not chosen / 0.24 / - |
| 28008:0 | 1 | 33.7 | yes / took / 3.03 / -2.5 | no / in time, crawl / stop / 0.94 / 2.5 | no / late / 4.51 / 3.5 |
| 28008:1 | 0 | 8.1 | no / not chosen / 0.02 / - | no / never entered / - / - | no / never entered / - / - |
| 28180:0 (capped) | 1 | 4.7 | no / in time, crawl / stop / 3.76 / 2.5 | no / not chosen / 0.18 / - | no / not chosen / 0.29 / - |
| 34183:0 | 0 | 13.4 | no / late / 0.51 / 12.5 | no / not chosen / 0.24 / - | no / late / 1.45 / 8.0 |
| 5423:0 | 0 | 6.2 | yes / took / 5.53 / 3.5 | no / late / 0.72 / 8.5 | no / late / 0.55 / 6.5 |
