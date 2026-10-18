# RTJA parallel diagnostic: was JTN-L18-s0 pulled inward by the route memory on the parking-lot loop?

2026-10-18. Descriptive only, not a gate (prereg `plans/2026-10-17-nr1-prereg.md`, supplement 12, "parallel diagnostic"). CPU only, stored
reads only. Definitions are those of the alignment audit ([../../../route_mem/results/align_audit.md](../../../route_mem/results/align_audit.md),
`rm_align.py` docstring). Generated tables for every zero of the loop log, all six reads: [memlead_tables.md](memlead_tables.md).

## Answer

**No. In the loop log JTN was not pulled inward by the memory before its own failures.** None of its loop-log zeros that RSG does not share
(6 on held 460, 8 on arena stop3, 9 on arena off) had near memory ahead. In 20 of those 23 the car had no valid near slot at all in the
3 s before the failure: arena windows last 5 s, and the near range stays empty until the car has driven about 42 m. The other 3 (held)
had near slots exactly in place (lead 0.0 m). The far slots (labels 24-30 m) did run ahead by 10.8-20.7 m in 4 distinct windows (ddc1271e
held; d0bf0c6c, f81c213c, 7f59a577 arena). But their points lie on the route, -0.9 to +0.6 m from it laterally, not on the inside of the
loop. JTN's plan still cut +1.5 to +5.8 m inside. RSG got the same memory in the same 3 s (first-slot lead 10.8-20.7 m, far lateral
-0.9 to -0.1 m), and its plan stayed at -0.3 to +1.5 m, and it scored 1.0. With the memory switched off (held `JTN-L18-s0-nomem`), JTN
still fails ddc1271e (offroad), 67077701 and 078bc102.
Memory that is both ahead and on the inside does occur in this log, in two kinds of window, and JTN and RSG share the zero in both:
- The loop entry (00181_00574 0c2636bf / c44ff9c8 / 24077398 / dea2f8b6; 00922_01296 145206f6): near slots 22-28 m ahead and 12-21 m
  inside, plan and failure inside for both models.
- The U-turn (01949_02501 3b4031de / c706b74f / a97b6329, the "57 m" scene): near slots 51-56 m ahead and 27-30 m inside. There both
  plans go 1.0-2.2 m outside and both cars fail outside: they under-turn, as in the audit's 3b4031de.

So JTN's own loop-log failures are early inside cuts of the policy's lateral behaviour, the same mechanism the audit found for the U-turn
cuts. They are not a misplaced memory. Over all reads, zeros have near memory 10 m or more ahead in the 3 s before the failure about as
often as RSG's do: JTN 3 / 26 held, 9 / 60 stop3; RSG 4 / 26, 10 / 64. Rollouts with any such lead fail about twice as often, for both
models alike, which matches the audit's geometry confound: the lead occurs only at turns and loops.

## Data and method

| read | stored runs | rollouts | zeros (with a failure event) | loop-log windows / zeros |
|:--|:--|--:|--:|--:|
| JTN held 460 @18 | `runs/route_mem/read/JTN-L18-s0/c*/run` | 452 (+8 route failures without drive records) | 26 | 113 / 13 |
| JTN held, memory off | `runs/route_mem/read/JTN-L18-s0-nomem/c*/run` (SH30_RMEM=0) | 452 | scores only | |
| JTN arena held split, stop3 | `runs/defect_fix/jt/arena/JTN-L18-s0@stop3/c*/run` | 2 228 | 60 | 475 / 28 |
| JTN arena held split, no hold | `runs/defect_fix/jt/arena/JTN-L18-s0@off/c*/run` | 2 228 | 247 | 475 / 58 |
| RSG held 460 @18 | `runs/route_mem/align/rsg_held.*` (the audit's extract of `route_mem/read/RSG-L18-s0`) | 452 | 26 | 113 / 12 |
| RSG arena stop3 | `runs/alpasim_proxy/arena/hold/reads/stop3/c*/run` | 2 228 | 64 | 475 / 36 |
| RSG arena no hold | `runs/alpasim_proxy/arena/reads/RSG-L18-s0/c*/run`, loop log only | 475 | 94 | 475 / 94 |

- `rm_align.py extract` on every read (pool CPU job 1018-201200-3e32; box outputs `$DATA_DIR/runs/defect_fix/rtja/memlead/`). Gate: the
  served memory equals the offline rebuild from the regenerated route messages on every memory-on decision (max 2.8e-4 m, masks 100 % equal).
  All JTN and RSG reads serve the projected layout (200-number `rmem`).
- `rm_align.py memlead` (new subcommand) per rollout, over the decisions in [t_f - 3 s, t_f] (zeros) or the last 3 s (non-zeros):
  - near lead = max lead of the slots labelled <= 20 m; first-slot lead = max lead of the first valid slot (any label); fold = rmem.merge's
    fold-back.
  - At the decision with the max near lead: near pull = the near slots' mean lateral from the route at their labels.
  - At the decision with the max first-slot lead: far pull = the first 3 valid slots' lateral, from the geo dumps.
  - plan10 = the plan's lateral from the route at +10 m; fail side = the car's 3 m front point at t_f.
  - All laterals are signed toward the route's net turn (+ = the inside of the loop / turn).
  - The route is AlpaSim's own route polyline, which follows the logged path.
  - The reference model's decisions over the same absolute 3 s window of the same scene.
- inward = near lead >= 10 m, near pull >= 1 m inside, plan10 inside, fail side inside. far-inward is the same with the first-slot lead and
  the far pull. The 1 m pull threshold was added after the first table: a +0.6 m far pull in f81c213c is the memory lying on the route.

## JTN's loop-log zeros that RSG does not share (all three reads)

"near" = valid near slots (max over the window); "-" = no valid slot of that kind. Laterals in m, + = inside of the loop.

| window | read | JTN zero | t_f s | near | near lead | first-slot lead | far slots at | far pull | JTN plan10 | fail side | RSG plan10 same 3 s | RSG score | JTN memory off |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 00032_00170-84b38cda | held | collision | 6.5 | 8 | +0.0 | +0.0 | 8 | +0.0 | -3.77 | +0.44 | +0.01 | 0.62 | 0.62 |
| 00181_00574-67077701 | held | collision | 7.5 | 4 | +0.0 | +0.2 | 36 | -0.0 | -1.58 | -2.06 | -2.94 | 1.00 | 0.00 corridor |
| 00181_00574-ddc1271e | held | offroad | 7.0 | 0 | - | **+18.0** | 24 | -0.1 | **+5.75** | +0.81 | +0.27 | 1.00 | 0.00 offroad |
| 00922_01296-8d3ac506 | held | corridor | 7.0 | 0 | - | +9.0 | 34 | -0.0 | -2.60 | -1.66 | -0.50 | 1.00 | 1.00 |
| 01590_01725-078bc102 | held | offroad | 4.5 | 0 | - | +6.9 | 34 | -0.0 | -1.78 | -2.04 | -0.73 | 1.00 | 0.00 offroad |
| 01949_02501-e072351f | held | offroad | 8.5 | 2 | -0.0 | -0.0 | 22 | +0.0 | -1.23 | -0.81 | -0.65 | 1.00 | 1.00 |
| 00181_00574-c4e83619 | stop3, off | offroad | 3.5 | 0 | - | +2.5 | 42 | +0.0 | +0.98 | +1.54 | -0.51 | 1.00 | |
| 00181_00574-d0bf0c6c | stop3, off | offroad | 2.5 | 0 | - | **+14.6** | 30 | -0.9 | **+4.33** | +1.02 | -0.28 | 1.00 | |
| 00181_00574-e35d372f | stop3, off | offroad | 2.5 | 0 | - | +4.5 | 40 | -0.0 | +3.16 | +0.65 | +0.68 | 1.00 | |
| 00181_00574-f81c213c | stop3, off | offroad | 2.5 | 0 | - | **+20.7** | 24 | +0.6 | **+3.47** | +0.79 | +1.45 | 1.00 | |
| 00922_01296-7f59a577 | stop3, off | corridor | 5.0 | 0 | - | **+10.8** | 28 | -0.1 | +1.50 | +1.69 | +1.14 | 1.00 | |
| 00922_01296-ecd00b39 | stop3, off | corridor | 5.0 | 0 | - | +9.0 | 34 | +0.0 | +1.91 | +1.99 | +0.73 | 1.00 | |
| 01949_02501-9c334d39 | stop3, off | offroad | 4.5 | 0 | - | +0.2 | 40 | +0.0 | -1.76 | -2.40 | -0.11 | 1.00 | |
| 01949_02501-cbabac00 | stop3, off | offroad | 5.0 | 0 | - | +2.5 | 36 | +0.3 | -2.10 | -1.27 | -1.83 | 1.00 | |
| 00181_00574-af6e81e1 | off only | offroad | 4.5 | 0 | - | +9.0 | 36 | +0.0 | +1.16 | +1.28 | -0.95 | 1.00 | |

- stop3 / off rows: the numbers are the stop3 read's. The off read has the same failure time, memory lead, far pull and failure side; its
  plan10 differs by up to 0.7 m.
- RSG's first-slot lead in the same 3 s equals JTN's to within 0.1 m on every arena window. On held it is 13.5 vs 18.0 m (ddc1271e) and
  6.7 vs 9.0 m (8d3ac506). The two cars receive essentially the same memory; only the plans differ.
- None of these 15 windows has a fold-back (max 0.0 m) in the window.

## Loop-log windows where the memory is ahead and on the inside (both models zero)

| window | reads (JTN zero) | near lead | near pull | JTN plan10 | JTN fail side | RSG near lead / plan10 same 3 s | RSG |
|:--|:--|--:|--:|--:|--:|:--|:--|
| 00181_00574-0c2636bf | stop3, off (corridor) | +24.5 | +18.4 | +1.87 | +6.57 | +25.8 / +11.34 (stop3) | zero |
| 00181_00574-c44ff9c8 | stop3, off (offroad) | +23.6 | +18.9 | +2.77 | +2.54 | +23.6 / +3.02 | zero |
| 00181_00574-24077398 | off (corridor) | +21.9 | +11.8 | +3.58 | +5.75 | +22.0 / +5.41 | zero |
| 00181_00574-dea2f8b6 | off (corridor) | +22.2 | +14.2 | +1.19 | +1.31 | +22.2 / +2.41 | zero |
| 00922_01296-145206f6 | off (corridor) | +28.3 | +20.9 | +0.58 | +0.13 | +28.1 / +1.01 | zero |
| 00181_00574-f9ed38d9 | held (offroad) | +15.1 | +7.6 | -0.02 | -1.34 | +15.4 / +0.65 | zero (RSG counted inward) |
| 01949_02501-3b4031de (U-turn) | held (offroad) | +55.6 | +28.3 | -1.04 | -2.49 | +55.6 / -1.36 | zero |
| 01949_02501-c706b74f (U-turn) | stop3, off (offroad) | +51.3 | +29.9 | -2.07 | -2.59 | +51.3 / -2.23 | zero |
| 01949_02501-a97b6329 (U-turn) | off (offroad) | +55.6 | +26.7 | -1.87 | -2.84 | +55.4 / -1.71 | zero |

## All zeros vs non-zeros (context)

3 s before the failure (zeros) or last 3 s (non-zeros). "unique" = the reference model scores > 0 on the window. Full table per read in
[memlead_tables.md](memlead_tables.md).

| read | subset | n | no near slots | near lead >= 10 m | first-slot lead >= 10 m | fold > 1 m | inward | far-inward |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| JTN held | zeros, loop log | 13 | 8 | 2 | 3 | 0 | 0 | 0 |
| | zeros unique, loop log | 6 | 3 | 0 | 1 | 0 | 0 | 0 |
| | zeros, other logs | 13 | 3 | 1 | 3 | 0 | 1 | 1 |
| | non-zeros | 426 | 33 | 14 | 19 | 4 | - | - |
| JTN arena stop3 | zeros, loop log | 28 | 25 | 3 | 8 | 0 | 2 | 2 |
| | zeros unique, loop log | 8 | 8 | 0 | 3 | 0 | 0 | 0 |
| | zeros, other logs | 32 | 24 | 6 | 11 | 2 | 4 | 5 |
| | non-zeros | 2 168 | 1 373 | 200 | 476 | 7 | - | - |
| JTN arena off | zeros, loop log | 58 | 51 | 7 | 17 | 0 | 5 | 5 |
| | zeros unique, loop log | 9 | 9 | 0 | 3 | 0 | 0 | 0 |
| | zeros, other logs | 189 | 153 | 34 | 86 | 2 | 32 | 35 |
| | non-zeros | 1 981 | 1 218 | 168 | 392 | 7 | - | - |
| RSG held | zeros, loop log | 12 | 9 | 2 | 2 | 0 | 1 | 1 |
| | zeros, other logs | 14 | 4 | 2 | 5 | 0 | 2 | 2 |
| | non-zeros | 426 | 28 | 14 | 16 | 5 | - | - |
| RSG arena stop3 | zeros, loop log | 36 | 31 | 3 | 6 | 0 | 2 | 2 |
| | zeros, other logs | 28 | 20 | 7 | 13 | 0 | 4 | 7 |
| | non-zeros | 2 164 | 1 377 | 203 | 480 | 8 | - | - |

Zero rate in rollouts with any near-slot lead >= 10 m before the outcome, against none:
- JTN: held 0.107 (28 rollouts) vs 0.054; stop3 0.047 (212) vs 0.025; off 0.198 (212) vs 0.102.
- RSG: held 0.143 (28) vs 0.052; stop3 0.051 (216) vs 0.026.
- Both models show the same ratio. Exposure is the turn and loop geometry (audit T3), so this is not a causal effect.

Notes:
- 26 of JTN's 32 "inward" zeros in other logs on the no-hold read are windows of one segment (`2021.09.16.19.49.00_veh-42_00990_01609`),
  mostly corridor zeros. The plan is only +0.1 to +0.5 m inside while the memory lies 18 m inside, so the rule fires but nothing was
  pulled. 28 of those 32
  windows score > 0 under stop3.
- The RSG no-hold read was extracted for the loop log only, so the off read has no RSG column for other logs.
- Windows that are offroad from t = 0 (t_f 0.0 in the tables) have no decision before the failure and are listed without numbers.

## Limits

- Stored closed-loop reads, no counterfactual replay: "the plan cut inside while the memory lay on the route" rules out a lateral pull
  by the memory points, not every influence of the memory. A far slot shown 15-20 m ahead still tells the policy that the route continues
  further along than it really does. Against that: RSG received the same memory and did not cut, and JTN memory-off still fails 3 of the 6
  held unique zeros (ddc1271e, the held representative, among them). JTN memory-off recovers e072351f and 8d3ac506, but in both the plan
  went outside, so they are not inward pulls.
- The memory-off read exists only for held 460. The arena windows were not re-read without memory (no GPU job, by the brief).
- 1 seed per model.

## Reproduce

```
# box, repo root, AlpaSim venv; pool CPU job (job body $DATA_DIR/runs/defect_fix/rtja/memlead/job.sh)
O=$DATA_DIR/runs/defect_fix/rtja/memlead; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; X=experiments/route_mem/scripts/rm_align.py
$AP $X extract --out $O --workers 16 --set jtn_held="$DATA_DIR/runs/route_mem/read/JTN-L18-s0/c*/run" \
  jtn_nomem="$DATA_DIR/runs/route_mem/read/JTN-L18-s0-nomem/c*/run" jtn_ar_stop3="$DATA_DIR/runs/defect_fix/jt/arena/JTN-L18-s0@stop3/c*/run" \
  jtn_ar_off="$DATA_DIR/runs/defect_fix/jt/arena/JTN-L18-s0@off/c*/run" rsg_ar_stop3="$DATA_DIR/runs/alpasim_proxy/arena/hold/reads/stop3/c*/run"
$AP $X extract --out $O --workers 16 --scene 2021.10.06.08.16.17_veh-52 \
  --set rsg_ar_off="$DATA_DIR/runs/alpasim_proxy/arena/reads/RSG-L18-s0/c*/run" rsg_nomem="$DATA_DIR/runs/route_mem/read/RSG-L18-s0-nomem/c*/run"
# any machine with the parquets and geo pickles (rsg_held.* from runs/route_mem/align)
python $X memlead --dir $O --ref-dir $DATA_DIR/runs/route_mem/align --res $O/res \
  --reads "JTN held 460=jtn_held:rsg_held:jtn_nomem" "JTN arena stop3=jtn_ar_stop3:rsg_ar_stop3" "JTN arena off=jtn_ar_off:rsg_ar_off" \
  "RSG held 460=rsg_held:jtn_held:rsg_nomem" "RSG arena stop3=rsg_ar_stop3:jtn_ar_stop3" "RSG arena off (loop log only)=rsg_ar_off:jtn_ar_off"
```
