# P4 turn-gallery fill (2026-10-14 night): camera clips for every definable gallery row

Prereg: [P4 prereg Amendment 5](../../plans/2026-10-14-p4-self-branch-prereg.md#amendment-5-2026-10-14-night-before-any-of-its-rollouts-read-only-camera-fill-of-the-turn-gallery).
This is read-only viewing material: no lines, no verdict, no rows. Every new root holds NO_ROWS, and rollout.asl is deleted.
Gallery: Tokyo `~/Desktop/turn_gallery/index.html` ([turn_gallery.py](../../scripts/turn_gallery.py)).
Family per scene: the identity plus g1 / g1.25 / g1.5 raw at the D40 and D0 handovers.
- Self g1-raw is the identity itself.
- A D0 that shares D40's decision is run once.

## Scenes

The gallery has 71 rows. Before the fill, 5 had P4 clips (403d558a, 3232fdb7, 37d660b1, 33c8976b, 1e8e939e). Of the 66 without:
- **17 define a strict turn** (`plan_strict36.json`) and got clips:
  - **5 train turns** (a174faed, ddc3e8df, e5e47fd2, eaba3f6a, f9e361aa). The box P4 run held every cell.
    - Box `p4/video` already had camera re-runs of all 27 train turns (89 sessions). Tokyo's copy was stale (11 scenes, synced from the box
      during this fill).
    - 16 new box re-runs. That is 3 more than the amendment's 13: for a174faed, e5e47fd2 and f9e361aa the existing re-run was a completing
      speed-profile cell, not D40 g1.25-raw.
  - **12 exam turns.**
    - 11 are from the exam-ceiling run p4_exam: 22fa5c92, 2f227f2c, 382a7247, 3ce13eed, 94c1d722, 9f393c8a, a1b166cc, ce161092,
      d90fe123, e37aa159, fd01e9ab. Of these, 10 are zero rows and d90fe123 is the section-C completed turn. 54 Tokyo re-runs.
    - 302c5c99 (hard_exam only) is a new Tokyo job of the reduced family: 5 rollouts with VIDEO=1, D30 shares D40's handover.
- **49 are not definable** (no strict turn: corridor and non-turn zeros). Their rows are unchanged.

After the fill, **22 / 71 rows** have a P4 cell: **10** show a completing cell, and 12 show D40 g1.25-raw marked "no cell completes" /
"no raw cell completes".

## Checks

- **Box:** 16 / 16 re-runs bit-identical to the stored `p4/run` sessions (served plan at every decision, controller track, score):
  [box_fill/vcollect.txt](p4_fill/box_fill/vcollect.txt).
- **Tokyo, against p4_exam:** 54 / 54 re-runs bit-identical to the stored sessions: [exam_video/vcollect.txt](p4_fill/exam_video/vcollect.txt).
  - The first collect reported 52 / 54. The 2 differing sessions were d90fe123 D40 g1.25 / g1.5 raw.
  - Cause: vcollect's reference glob also matched p4_exam's renamed renderer-OOM attempt `d90fe123-b0.oom`, which overwrote the stored
    b0 sessions (no score, no track).
  - Fixed in rf_p4.py (79ebfab1).
- **Identity S1 against r281/rsg:** 11 / 11 exam re-runs and the 302c5c99 phase A, so 12 / 12 bit-identical:
  [exam_video/s1.txt](p4_fill/exam_video/s1.txt), [new302/s1.txt](p4_fill/new302/s1.txt).
- **Train turns have no S1:** box and Tokyo rollouts are not bit-identical, so the train-turn branches start from the box identity. The
  gallery shows that identity behind "more" as "P4 id (box)".

## Per scene

Outcomes are from the stored reads (the re-runs are bit-identical), and 302c5c99 is from [new302/read.json](p4_fill/new302/read.json).
- `done` = P2 completion rule; `x` = not completed; `(id)` = self g1-raw = the identity.
- D0 h `(= D40)` = the same handover decision, run once.
- Shown cell: the gallery's main P4 cell when a family cell completes. Otherwise D40 g1.25-raw.
- Train turns a174faed, e5e47fd2 and f9e361aa also have an earlier completing speed-profile cell behind "more".

| scene | set | class | side / topo | identity | D40 h | D40 g1 / g1.25 / g1.5 | D0 h | D0 g1 / g1.25 / g1.5 | shown cell |
|--|--|--|--|--|--|--|--|--|--|
| 22fa5c92 | exam | coll_veh_oncoming | L / straight_exit | collision | 17 | (id) / x / x | 72 | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| 2f227f2c | exam | cor_straight_at_turn | R / straight_road | corridor | 17 | (id) / x / x | 88 log | done / done / done | D0-log-g1.25-raw |
| 302c5c99 | exam | cor_straight_at_turn | L / straight_road | corridor | 17 | (id) / x / x | 75 | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| 382a7247 | exam | coll_veh_oncoming | L / straight_exit | done | 17 | (id) / done / x | 51 | (id) / x / x | D40-self-g1.25-raw |
| 3ce13eed | exam | cor_turn_under | R / curve_single | corridor | 17 | (id) / x / x | 59 | (id) / done / done | D0-self-g1.25-raw |
| 94c1d722 | exam | off_junc_outside | R / no_straight_exit | offroad | 22 | (id) / x / x | 134 | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| 9f393c8a | exam | coll_veh_side | R / curve_single | collision | 17 | (id) / x / x | 17 (= D40) | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| a174faed | train | cor_straight_at_turn | R / straight_exit | corridor | 17 | (id) / x / x | 97 | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| a1b166cc | exam | cor_turn_under | L / other_no_lane | corridor | 17 | (id) / done / x | 76 | (id) / x / x | D40-self-g1.25-raw |
| ce161092 | exam | off_curve_outside | L / straight_road | offroad | 59 | (id) / x / x | 109 | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| ddc3e8df | train | cor_turn_under | R / curve_single | corridor | 17 | (id) / done / x | 17 (= D40) | (id) / done / x | D40-self-g1.25-raw |
| e37aa159 | exam | cor_turn_under | L / curve_single | corridor | 17 | (id) / x / x | 129 | (id) / done / x | D0-self-g1.25-raw |
| e5e47fd2 | train | cor_straight_at_turn | L / straight_road | corridor | 17 | (id) / x / x | 48 | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| eaba3f6a | train | cor_straight_at_turn | R / straight_exit | corridor | 17 | (id) / x / x | 53 | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| f9e361aa | train | cor_straight_at_turn | R / straight_exit | corridor | 23 | (id) / x / x | 63 | (id) / x / x | none completes (D40-self-g1.25-raw shown) |
| fd01e9ab | exam | cor_turn_under | L / no_straight_exit | corridor | 17 | (id) / x / x | 53 | (id) / done / x | D0-self-g1.25-raw |
| d90fe123 | exam | completed | L / straight_road | done | 61 | (id) / done / done | 106 | (id) / x / x | D40-self-g1.25-raw |

## Completions by gain and handover (new scenes)

Distinct rollouts only. A D0 that shares D40's handover (ddc3e8df, 9f393c8a) counts once, under D40. Self g1 = identity, reported apart.

| set | handover | g1 (log handovers only) | g1.25 | g1.5 |
|--|--|--:|--:|--:|
| train | D40 | 0 / 0 | 1 / 5 | 0 / 5 |
| train | D0 | 0 / 0 | 0 / 4 | 0 / 4 |
| exam | D40 | 0 / 0 | 3 / 12 | 1 / 12 |
| exam | D0 | 1 / 1 | 4 / 11 | 2 / 11 |

- **Identity completes:** train 0 / 5; exam 2 / 12. The two are 382a7247 (it completes, then zeroes in an oncoming collision) and d90fe123
  (section C).
- **Turns where the identity fails and a family cell completes:**
  - train: 1 / 5 (ddc3e8df, g1.25 at D40 = D0);
  - exam: 5 / 10 (2f227f2c on the D0 log handover, all three gains; 3ce13eed D0 g1.25 / g1.5; a1b166cc D40 g1.25; e37aa159 D0 g1.25;
    fd01e9ab D0 g1.25).
- **g1.25 carries the completions:** 8 of the 12 completing gain-cells (both sets, D40 + D0) are at g1.25.
- **302c5c99** (new, straight_road left turn, straight at turn): nothing completes, every cell leaves the corridor.
- **Nothing here changes a recorded conclusion:**
  - The 11 p4_exam turns and the 5 train turns re-run stored rollouts.
  - 302c5c99 is not among the exam-ceiling's 26 turns.

## Cost

- **Tokyo:** 59 rollouts. 54 are re-runs, at 444-850 s per scene (5-6 rollouts in one launch, ~125 s per rollout with video, 2 launches per
  3090). The other 5 are the 302c5c99 job (615 s).
  - 8 110 launch-seconds = 2.25 launch-hours, ~1.1 card-hours.
  - Wall 21:07-21:53 JST.
  - Card 0 peaked at 23.2 / 24 GB with 2 launches (no OOM).
- **Box:** 16 rollouts in 5 pool jobs (3 launches each, `P4_VRAM=20`, `CL_HIST_KEY=alpasim-read4`; peaks 6-20 GB).
  - 1 844 job-seconds = 0.51 job-hours, ~0.13 card-hours at 4 jobs per card.
  - Wall ~18 min including the pilot.
- **Total:** ~1.25 card-hours.

## Files and roots

- [box_fill/](p4_fill/box_fill/): select (`vselect --family --no-id` minus the cells already in `p4/video`), manifest, vcollect, log.
  - Box root `$DATA_DIR/runs/route_follow/p4/video_fill`.
  - Tokyo copy `/data/runs/route_follow/p4_video_fill`.
- [exam_video/](p4_fill/exam_video/): `rf_p4exam.sh vinit / vstage` from p4_exam; Tokyo `/data/runs/route_night/p4_fill/exam_video`.
- [new302/](p4_fill/new302/): `rf_p4.sh examplan` + `rf_p4exam.sh stage` with `VIDEO=1 P4_D=40,0 P4_ONLY=g1-raw,g1.25-raw,g1.5-raw`,
  then `rf_p4.py vcollect --own`; Tokyo `/data/runs/route_night/p4_fill/new`.
- Code served: p4_exam's frozen tree (re-runs), `/data/runs/route_night/p4_fill/code` at 42b66a72 (302c5c99 job and runners). The box
  re-runs served the P4 run's worktree (652d48b7).
- The gallery reads the four manifests in this order: `p4_video`, `p4_video_fill`, `p4_fill/exam_video`, `p4_fill/new`.
