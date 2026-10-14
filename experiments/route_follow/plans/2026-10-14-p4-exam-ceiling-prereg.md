# Pre-registration: P4 exam ceiling read, the P4 branch family on the hard_exam turn scenes (read-only), 2026-10-14

Context: [P4 prereg](2026-10-14-p4-self-branch-prereg.md) with Amendments 1-2 and its smoke
[results/p4_self/smoke.md](../results/p4_self/smoke.md). P4 searches the turning school's phase-2 branches on 27 training scenes (box,
running). This read runs the same branch family, unchanged, on the hard_exam turn scenes, to give an **upper bound** that is quoted next to
RSGSELF's PASS1 result later: how many exam turns the branch family can complete at all. Written, committed and pushed before any rollout.
Rules fixed by the user; amendments are appended, never edited in place.

## What this read is not

- It changes **no** design value of the training prereg (v_e, cells, handovers, gates, lines). Nothing measured here feeds back into P4,
  the row builder or a training choice.
- **No row is ever built from it.** Output root `/data/runs/route_night/p4_exam` (Tokyo). It must never be passed to `rf_selfpool.py`:
  the root holds a `NO_ROWS` marker and `rf_selfpool.py` refuses any root with that marker, also with `--allow-exam` (it already refuses
  hard_exam scenes without that flag). Every rollout.asl is deleted after its scene's read (drive.jsonl, controller CSV, metrics,
  results-summary, read.json kept), so the rollouts cannot be replayed into rows.
- No pass / fail mark.

## Scenes

The 26 hard_exam route-follow turn scenes, `/data/runs/route_night/exam_turn26.txt` (Tokyo), with the same strict-turn definability rule as
P4's 27 (`rf_p4.py plan`: AlpaSim's MAP route generated along the log, strict turn >= 60 deg within 30 m, exit + 10 m inside the log, P2's
completion target inside the log). Plan: `rf_p4.sh examplan` (code 807e7602), committed as
[results/p4_self/exam_ceiling/plan.json](../results/p4_self/exam_ceiling/plan.json). **15 are definable, 11 dropped:**

| dropped | reason |
|:--|:--|
| 00fabf60, 2466266e, 7381c9fa, 9b6c68b4, ba40be62, cbd87234, e904e9c0, f8ee31cb, fc9fde87 | no strict turn inside the log |
| cdf95fd1 | exit + 10 m beyond the log (227 + 10 > 234) |
| de9db5dc | exit + 10 m beyond the log (42 + 10 > 38) |

The 15: 8 left, 7 right; straight_exit 6, straight_road 3, curve_single 3, no_straight_exit 2, other_no_lane 1 (v_e 4.6 and the all-turn
gate, as P4's rule for classes outside the table); human entry speed median 4.89 m/s. Distinct handover decisions by the log's timing: 1
(1 scene), 2 (3), 3 (2), 4 (2), 5 (7): about 15 + 56 x 8 = **~463 rollouts** (the policy's own reach decides the actual count).

## Design (= P4, unchanged)

Phase A = the policy alone (identity cell), serving identical to `/data/runs/route_night/r281/rsg`: RSG-L18-s0, wizard gains 20,
`JEV_VCONT=1.0 JEV_LEAD=1`, VHIST off, holdlead off, `+e2e_challenge=dev`, harmonizer off, plus the instrument's
`send_recording_ground_truth=true` and `PAI_DUMP`; reach per D in {40, 30, 20, 10, 0} from phase A's own stations (distinct handover
decisions only; duplicates run once); phase B = per reached handover the 8 non-identity cells (lateral gain 1 / 1.25 / 1.5 x longitudinal
raw / ve / ve08), per unreached distance the 9 cells with src = log (fallback). One rollout at a time per launch (`P4_CONC=1`: the NRE
renderer race of Amendment 1; never two rollouts of one scene in one launch). Runner `rf_p4exam.sh` (one `rf_p4.sh job` per scene), served
from a frozen Tokyo clone `/data/runs/route_night/p4_exam/code` at this prereg's commit (recorded in `<root>/code.json`).

**Phase-A gate (S1 type, as the smoke):** every scene's phase A identity rollout against r281/rsg's rollout of the scene (`rf_p4.py s1`:
every served plan, the controller track and the score bit-identical). The count is reported; a scene that is not bit-identical is
reported and flagged in the table (its branches are not then on the reference's states).

## Measures and lines (ceiling; no pass / fail)

`rf_p4.py read` per rollout as in P4 (completion = P2's rule; realized entry speed, shape measures, crawl, fallback), merged with
`rf_p4.py table --ceiling`. Reported, each **over 26** (a dropped scene counts as not covered: its turn is not definable) and over the 15:

1. **Identity**: turns the policy alone completes (phase A).
2. **Ceiling, any cell**: turns with >= 1 completing cell at any handover; branch cells only (P4's rule, the identity is no branch) and
   including the identity.
3. **Coverage definition** (P4's line): turns with completing branch cells at >= 2 distinct handover decisions, each with realized entry
   speed <= class v_e + 1 m/s; self-handovers-only count next to it.
4. **Shape gate** (decision 293 gate as `rf_p4.py read` computes it: class p95 kappa / dkappa, a_lat <= 4.42, entry speed <= v_e + 1):
   turns with gate-passing completions at >= 1 and >= 2 distinct handovers.

Also: the per-scene x handover x cell table, per-D and per-cell completion counts, src = log (fallback) count, crawl counts (approach and
inside), phase-A bit-identity count, timing.

## Running

Tokyo, both cards (checked free: 49 / 18 MiB at 14:20 JST); the user's tmux windows in `jev` are left alone. State lines go to
`/data/runs/route_night/CHAIN_STATUS`. The user asked for three launches per card. Measured in the smoke (`f9e361aa-b0/usage.jsonl`): one
Tokyo launch peaks at **11.2 GB VRAM** (renderer 7.9, driver 3.1, physics 0.4 GB) and ~7.5 GB RAM, so three on a 24 GB card (33.6 GB) is
expected not to fit. Stage 1 tests it; the choice is recorded in an amendment before stage 5:

- **Stage 1**: 1 scene (largest expected cell count first), its phase B as 3 launches side by side on card 0 (`P4_PAR=3`). If every
  launch completes with all rollouts scored and S1 holds, stages 5 and rest run 3 workers per card; otherwise 2 per card (4 launches).
- **Stage 5**: 5 scenes, workers = cards x slots, one scene job per worker (`P4_PAR=1`), largest scenes first. ETA restated after it.
- **Stage rest**: the remaining 9.
- Checklist after each stage: every job DONE, rollouts = cells, reach.json present, phase A bit-identical to r281/rsg, self branches'
  plans before h identical to phase A (`pre_h_diff` 0), no renderer failures, disk within the estimate. A failure stops the chain and is
  reported.

**Estimate.** ~463 rollouts; smoke ~99 s per rollout alone on a 3090 (+ ~45 s start-up per launch). At 4 launches with an assumed 1.3x
contention: 463 / 4 x 130 s = **~4.2 h**; at 6 launches (if they fit) ~3 h. Longest scene job (41 rollouts, one launch) ~1.5 h, below the
per-slot total, so largest-first packing is close to even. Disk: ~0.5 GB per rollout until its scene's read, then the .asl files are deleted
(peak ~4-6 scenes x 20 GB = ~120 GB; /data 837 GB free).

## Not claimed

15 definable turns of 26, one seed, one checkpoint, a deterministic simulator; a completing cell is one draw of a chaotic closed loop. An
upper bound on what the branch family reaches on the exam turns, not a result of any trained policy.

## Commands

```bash
# Tokyo (done before this commit): plan.json + NO_ROWS
BACKEND=tokyo bash experiments/route_follow/scripts/rf_p4.sh examplan /data/runs/route_night/p4_exam
# frozen clone at this commit, then the stages in tmux jev
git clone -q ~/mycode/jev-drive /data/runs/route_night/p4_exam/code
X=/data/runs/route_night/p4_exam/code/experiments/route_follow/scripts/rf_p4exam.sh
tmux new-window -d -t jev -n p4x1 "CARDS=0 SLOTS=1 P4_PAR=3 bash $X stage /data/runs/route_night/p4_exam s1 1"
tmux new-window -d -t jev -n p4x5 "CARDS='0 1' SLOTS=<2|3> bash $X stage /data/runs/route_night/p4_exam s5 5"
tmux new-window -d -t jev -n p4xr "CARDS='0 1' SLOTS=<2|3> bash $X stage /data/runs/route_night/p4_exam rest rest"
```

## Amendment 1 (2026-10-14 15:15 JST, after stage 1, before stage 5): three launches per card fit

- Stage 1 (94c1d722, frozen clone 05f3c5c5, card 0, phase B as 3 launches side by side): card 0 peaked at **22.6 GB** of 24 GB with three
  launches (the renderer takes less than the 7.9 GB it held alone in the smoke), GPU ~100 %, RAM ~15 GB for the three. Each launch ran one
  rollout per ~100-120 s, about the same as one launch alone, so three per card give ~3x throughput. Job: phase A 161 s, phase B 40 cells
  in 1 667 s (14 / 13 / 13 per launch), 41 rollouts in 1 835 s.
- Checklist: 41 / 41 rollouts scored, reach.json present, 5 distinct self handovers (h 22 / 30 / 41 / 89 / 134), phase A bit-identical to
  r281/rsg (1 / 1), self branches' plans before h identical to phase A 40 / 40, every rollout.asl removed after the read (1.3 GB left).
- **Change:** stages 5 and rest run **3 workers per card on both cards** (6 launches, `SLOTS=3`, `P4_PAR=1`), as the user asked.
  Estimate restated: ~422 rollouts left / 6 launches x ~120 s = ~2.3 h plus the tail of the last 41-rollout job (one launch, ~1.4 h if it
  starts late); ~2.5-3 h.
- Fix outside the frozen clone: `rf_p4.sh examplan` wrote `code.json` from the working repo (807e7602) before the clone existed; it was
  rewritten by hand before the first rollout to the clone (05f3c5c5, dirty 0), and `examplan` no longer writes it.

## Amendment 2 (2026-10-14 17:00 JST, after stage 5, before stage rest): one renderer OOM, its launch rerun

- Stage 5 (abd45a30, ce161092, d90fe123, e0f6ebe1, e37aa159; 3 workers per card, 5 busy): 4 scenes DONE (41-42 rollouts each, jobs
  4 522-5 759 s), phase A bit-identical to r281/rsg for all 5, self branches' plans before h identical to phase A in every DONE scene,
  rollout.asl removed after each read.
- **d90fe123's phase B launch** started at 15:17 together with two other launches on card 0; its renderer logged `OOM during operation ...
  attempting memory cleanup` in the first rollout and recovered, but every later rollout then waited exactly 600 s in
  `get_available_cameras` (~12.5 min per rollout, 9 of 40 in 100 min). No other launch (stages 1 and 5, 14 launches) logged an OOM. The
  degraded launch was stopped at 16:57 and moved aside (`d90fe123-b0.oom`, not read, .asl deleted); the scene's phase B is rerun from
  scratch in stage rest (phase A kept: it ran before the OOM and is bit-identical).
- Stage rest keeps 3 workers per card (6 launches; 9 scenes + d90fe123, ~257 rollouts). Every launch's compose.log is watched for
  `OOM during`; a launch that logs it is stopped and its scene's phase B rerun the same way (counted in the README).
- **ETA restated:** rollouts ran 53-77 s each in the healthy stage-5 launches (~110 s per rollout per launch with start-up, read and the
  4-5 per card mix): ~257 / 6 x 110 s = ~1.3 h, bounded below by the largest job (41 rollouts, ~1.2-1.5 h): finish ~18:30-19:00 JST.
