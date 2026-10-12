# road_probe, real frames: is the junction blur in the frozen encoder or in the reconstructed images?

Pre-registration: [plans/2026-10-12-road-probe-prereg.md](../plans/2026-10-12-road-probe-prereg.md) section 8 (pushed before any real-frame label, training or read). Script `scripts/rp_real.py`; box `$DATA_DIR/runs/road_probe/real/` (labels, pred, read `20261012-185658`). Figure [figs/rp_real.png](../figs/rp_real.png).

## Gates

- G_frame_navtest: in_lane 1.00, in_road 1.00, share 1.00, pass True
- G_frame_navtrain: in_lane 1.00, in_road 1.00, share 1.00, pass True
- G_lab: straight 5044, share 0.78, pass True
- G_sign: L20 1.61, R20 -1.75, pass True
- G_repro: held_turn20 0.90, stored 0.88, seeds [0.9274196884846924, 0.9958332619622572], pass True

navtest 12146 tokens (136 log segments, 44 drive logs), 4635 turn decisions, of which 1542 in the 16 held drive logs (the same roads as the reconstructed held read); junction distance known for 0.58 of all tokens; both 20 m boundaries defined on 0.33 of the turns. Fit sizes: A-s0 68728 rows / 27 units, A-s1 68728 rows / 27 units, B-s0 91773 rows / 142 units, B-s1 91773 rows / 142 units, C-s0 19906 rows / 27 units, blind-s0 91773 rows / 142 units.

## Turn decisions: error at 20 m by distance to the junction

| predictor | in junction | 0-10 m | 10-20 m | 20-40 m | 40-60 m | none within 60 m | near (in + 0-20) | far (20-60) | near / far |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| reconstructed frames, probe (decision 275, held AlpaSim) | 1.11 (278) | 1.15 (218) | 1.68 (162) | 0.47 (144) | 0.42 (114) | 0.85 (802) | 1.19 [0.97, 1.53] | 0.44 [0.32, 0.60] | 2.70 [1.84, 4.22] |
| (a) same probe on real frames, same 16 logs | 1.58 (106) | 0.93 (90) | 2.90 (40) | 0.46 (166) | 0.50 (78) | 1.04 (404) | 1.29 [0.89, 2.65] | 0.49 [0.35, 0.82] | 2.66 [1.61, 6.08] |
| (b) probe trained on navtrain, all navtest | 1.11 (426) | 1.53 (272) | 1.06 (242) | 0.44 (906) | 0.56 (230) | 0.62 (1012) | 1.18 [0.94, 1.46] | 0.46 [0.38, 0.55] | 2.58 [1.93, 3.38] |
| (b) on the same 16 logs | 1.01 (106) | 1.15 (90) | 3.48 (40) | 0.54 (166) | 0.54 (78) | 0.70 (404) | 1.23 [0.78, 1.62] | 0.54 [0.35, 0.86] | 2.27 [1.22, 3.92] |
| blind floor (navtrain, ego + command) | 1.44 (426) | 1.36 (272) | 1.57 (242) | 0.55 (906) | 1.18 (230) | 0.87 (1012) | 1.47 [1.19, 1.74] | 0.64 [0.51, 0.81] | 2.29 [1.73, 2.94] |
| (c) navtrain, 27 units only (exploratory) | 1.16 (426) | 1.68 (272) | 1.53 (242) | 0.52 (906) | 0.56 (230) | 0.64 (1012) | 1.38 [1.13, 1.69] | 0.54 [0.41, 0.68] | 2.57 [1.92, 3.62] |
| shipped heads, raw | 6.25 (426) | 5.25 (272) | 2.34 (242) | 0.68 (906) | 1.17 (230) | 2.69 (1012) | 4.60 [3.41, 5.68] | 0.74 [0.62, 0.87] | 6.19 [4.54, 7.87] |
| shipped heads, 0.689 undone | 5.65 (426) | 5.24 (272) | 2.23 (242) | 0.55 (906) | 1.16 (230) | 2.31 (1012) | 4.35 [3.24, 5.37] | 0.61 [0.52, 0.82] | 7.14 [4.82, 9.11] |
| blind floor on reconstructed (decision 275) | 2.10 (278) | 1.96 (218) | 3.47 (162) | 1.42 (144) | 1.11 (114) | 2.01 (802) | 2.33 [1.87, 3.00] | 1.37 [1.03, 1.87] | 1.70 [1.18, 2.49] |
| shipped heads on reconstructed, raw | 4.73 (278) | 4.49 (218) | 5.20 (162) | 1.39 (144) | 3.01 (114) | 3.19 (802) | 4.75 [4.11, 5.60] | 1.58 [0.93, 4.18] | 3.01 [1.13, 5.21] |

Cells: median error in m (pooled L / R errors in the bucket); brackets: 95% CI from resampling log segments (real) or scenes (reconstructed), B = 2000.

## Straight decisions (descriptive)

| predictor | in junction | 0-10 m | 10-20 m | 20-40 m | 40-60 m | none within 60 m | near (in + 0-20) | far (20-60) | near / far |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| reconstructed frames, probe (decision 275, held AlpaSim) | 0.42 (410) | 0.44 (432) | 0.39 (484) | 0.27 (1136) | 0.26 (1022) | 0.22 (4682) | 0.42 [0.36, 0.54] | 0.27 [0.23, 0.30] | 1.57 [1.27, 2.03] |
| (a) same probe on real frames, same 16 logs | 0.32 (114) | 0.34 (158) | 0.32 (98) | 0.22 (324) | 0.14 (330) | 0.18 (1324) | 0.33 [0.24, 0.43] | 0.18 [0.14, 0.22] | 1.82 [1.25, 2.64] |
| (b) probe trained on navtrain, all navtest | 0.39 (366) | 0.35 (532) | 0.41 (440) | 0.25 (1126) | 0.19 (1318) | 0.20 (4058) | 0.39 [0.33, 0.43] | 0.21 [0.18, 0.25] | 1.86 [1.50, 2.20] |
| (b) on the same 16 logs | 0.39 (114) | 0.43 (158) | 0.34 (98) | 0.18 (324) | 0.14 (330) | 0.18 (1324) | 0.39 [0.31, 0.48] | 0.16 [0.13, 0.19] | 2.48 [1.86, 3.17] |
| blind floor (navtrain, ego + command) | 0.50 (366) | 0.45 (532) | 0.52 (440) | 0.37 (1126) | 0.28 (1318) | 0.32 (4058) | 0.49 [0.42, 0.53] | 0.33 [0.28, 0.37] | 1.49 [1.28, 1.78] |
| (c) navtrain, 27 units only (exploratory) | 0.41 (366) | 0.45 (532) | 0.50 (440) | 0.26 (1126) | 0.20 (1318) | 0.22 (4058) | 0.45 [0.39, 0.53] | 0.23 [0.19, 0.28] | 1.94 [1.55, 2.39] |
| shipped heads, raw | 0.49 (366) | 0.51 (532) | 0.53 (440) | 0.47 (1126) | 0.47 (1318) | 0.49 (4058) | 0.51 [0.47, 0.55] | 0.47 [0.45, 0.50] | 1.10 [0.97, 1.19] |
| shipped heads, 0.689 undone | 0.40 (366) | 0.43 (532) | 0.51 (440) | 0.34 (1126) | 0.23 (1318) | 0.26 (4058) | 0.44 [0.39, 0.51] | 0.27 [0.23, 0.31] | 1.61 [1.33, 1.96] |
| blind floor on reconstructed (decision 275) | 0.70 (410) | 0.71 (432) | 0.58 (484) | 0.49 (1136) | 0.46 (1022) | 0.41 (4682) | 0.67 [0.58, 0.80] | 0.48 [0.42, 0.57] | 1.42 [1.15, 1.71] |
| shipped heads on reconstructed, raw | 0.65 (410) | 0.75 (432) | 0.59 (484) | 0.59 (1136) | 0.59 (1022) | 0.60 (4682) | 0.67 [0.57, 0.81] | 0.59 [0.55, 0.64] | 1.13 [0.97, 1.37] |

## Verdict (registered: (b) on all navtest turn decisions)

Line: near < 0.6 m -> reconstruction artefact; else near / far > 2 -> same shape on real frames (the encoder); else undetermined.

- b_navtest: **same shape on real frames**
- b_held_units: **same shape on real frames**
- a_held_units (descriptive): **same shape on real frames**
- reconstructed (descriptive): **same shape on real frames**

## Reading

- **Verdict (registered row (b), all navtest turns): the blur is in the frozen encoder, not in the reconstructed images.** Near the junction
  (inside it or within 20 m of it) the head trained on navtrain is off by 1.18 m [0.94, 1.46], against 0.46 m [0.38, 0.55] when the junction
  is still 20-60 m away. The ratio is 2.58 [1.93, 3.38]; its lower bound sits just under the 2x line. On reconstructed frames the same
  numbers are 1.19 / 0.44 / 2.70. The level matches too, not only the shape. The near-junction error is the same on real and reconstructed
  frames, so it is not a property of the AlpaSim images.
- (a) (the decision 275 probe moved to real frames on the same 16 drive logs) does about as well on real frames as on reconstructed ones:
  near 1.29, far 0.49, straight 0.33 / 0.18. The domain shift from MTGS images to real ones costs nothing that can be measured here.
- **Surprise: the blind floor (ego + command, no image) shows the same shape on real frames** (near 1.47, far 0.64, ratio 2.29). Part of
  the junction bump is how hard the label is: at a junction the route lane cut by the fixed line 20 m ahead is an oblique crossing of a
  connector, and the logged path's own line there varies between drivers. The tokens beat the blind floor by only 0.29 m near the junction
  (1.18 vs 1.47), against 0.18 m away from it (0.46 vs 0.64). On reconstructed frames the blind floor was much worse (2.33 near). So "the
  eyes see little more than ego + command inside junctions" holds on real frames too.
- Data: (c), trained on navtrain with only 27 drive logs, gets 1.38 near against 1.18 with 142 logs. Some of the near-junction gap is data,
  but even 142 real-frame logs stay above 1.1 m.
- The shipped heads are 4.6 m off near junctions on real frames too (0.74 m away from them; undoing the 0.689 squeeze does not help in
  junctions). This matches decision 272's reading on reconstructed frames.
- Coverage: both 20 m boundaries are defined on only 0.33 of navtest turns (the route lane is undefined on the cutting line inside most
  junction connectors). The 10-20 m bucket on the 16-log subset has 40 errors and a CI that spans 1-5 m, so read it as noise.
