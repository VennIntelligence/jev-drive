# WAIT-SET: a local closed-loop set where the logged human waits at standstill

Why: the private nuPlan board is unfiltered and waiting-rich (decision 325); on d325's uniform 5.5 s windows of the raw test logs the human
waits at a red light for the whole window in 24 % of them (public 0.4 %), and a car that rolls about 3 m where the human stands still is
zeroed by the corridor rule (experiments/wait_cap/results/next/census_red_rear.md). No local exam (held 460, public 1491) contains such a
window. This set does: public scenes whose human starts from a standstill, with the trajdata window moved so that the human is still
waiting in it. Code: [`scripts/wait_set.py`](../scripts/wait_set.py) (definitions in its docstring, fixed before any model read),
[`scripts/wait_set.sh`](../scripts/wait_set.sh) (build, reads, gate, report).

## Read a new tag (one command)

```bash
# box, repo root: one pool job (both variants, 316 windows, public 10-decision driver path + the wait-set cache), then the outcome table
bash experiments/alpasim_proxy/scripts/wait_set.sh read <tag> [after-job-ids]      # <tag>: checkpoint in runs/op_parity/runs, or P0
bash experiments/alpasim_proxy/scripts/wait_set.sh report                           # tables of every read -> $DATA_DIR/runs/alpasim_proxy/wait_set/report/
```

Outputs: `$DATA_DIR/runs/alpasim_proxy/wait_set/reads/<tag>/{run, table/<tag>.csv}` (one row per window: score, zero type, roll while
the human waits, wait success, launch time and delay). The driver path is exactly the public reads' (`ol_chain.sh readlist`: `run.sh sh30`,
`c0b_chain.OVERRIDES`, pool `--vram 20 --cpu 6 --ram 30`, `CL_HIST_KEY=alpasim-read4`); the only addition is
`runtime.scene_provider.trajdata.cache_location=$DATA_DIR/datasets/alpasim_nuplan_wait/v1`.

## Set (frozen, `results/wait_set/set.json`, split `alpasim_nuplan/waitset@v1`)

199 public scenes start with the human at rest (< 0.3 m/s at the scene's first frame). Per scene the movement onset (first frame > 0.25 m
from the rest point) and the standstill before it are read from the raw log; variant a = the 5.5 s window that ends 0.5 s before the onset
(needs >= 6.0 s of standstill), variant b = the window whose 3.0 s mark is the onset (needs >= 3.0 s).

| | candidates | standstill too short | undrawn vehicle in the ego's lane or as the lead | camera coverage | kept | held-split |
|:--|--:|--:|--:|--:|--:|--:|
| a (static) | 199 | 39 | 13 | 0 | 147 | 44 |
| b (launch at 3.0 s) | 199 | 9 | 21 | 0 | 169 | 51 |

Classes (a / b): red light first in line 35 / 41, red light queued 39 / 36, other with a lead 36 / 48, other free 37 / 44. Window starts
lie 2.0-5.5 s (a, median 3.9) and -1.0..+2.5 s (b, median 0.9) before the public window's. Coverage: every window lies inside its
reconstruction's camera video (start >= 0.55 s after the video's first frame, end >= 4.2 s before its last), the logged ego is within
0.49 m of a camera frame everywhere; the human's path in a is <= 0.33 m; in b the human passes 1 m/s at 3.0-5.05 s (median 3.2) and drives
1.6-10.2 m in the window (31 windows under 5 m, where the score has no progress term). Reference: on the public windows of the same source
scenes RSG-L18-s0 scores 0.921 (a sources) / 0.918 (b sources), V2 0.901 / 0.907 (census_red_rear per-scene tables).

Validation before any model read: the rebuilt cache equals the shipped one on all 171 source scenes' public windows (171 / 171 exact);
all 316 cached windows pass the track check (a: ego displacement <= 0.28 m; b: <= 0.38 m before the onset); smoke on 4 windows: go-straight
and human replay 1.0, the replay passes 1 m/s 0.02-0.07 s after the logged human.

Amendment 2 (before any model read): the full validation found 4 windows (1 a, 3 b) whose rollout fails before scoring for every driver,
the human replay included (the runtime's route sanity check, "route folds back on itself", as in the 12 public route-fold scenes); they
were dropped: set v2 = 312 windows (146 a, 166 b) from 168 source scenes, split `alpasim_nuplan/waitset@v2`.

## Validation (set v2, before any model read)

| arm | variant | n | score | zeros | roll while the human waits: median / p90 / max m | launch: never / median delay |
|:--|:--|--:|--:|--:|:--|:--|
| go-straight (starter kit) | a | 146 | 0.993 | 1 (offroad) | 0.003 / 0.28 / 3.12 | |
| go-straight (starter kit) | b | 166 | 0.212 | 0 | 0.006 / 0.26 / 1.59 | 166 / - |
| human replay (force-GT whole run) | a | 146 | 1.000 | 0 | 0.008 / 0.07 / 0.33 | |
| human replay (force-GT whole run) | b | 166 | 1.000 | 0 | 0.22 / 0.24 / 0.47 | 5 / -0.03 s |

- Gate (go-straight >= 0.99 on a, replay >= 0.98 on a and b): PASS. The one go-straight zero is a window where the starter kit holds the
  small speed it measures at the hand-over and creeps 3.1 m off the road (11 of 146 a windows show > 0.5 m of such creep); the replay is
  fine there.
- The replay's b roll (0.22 m) is the human's own creep inside the 0.25 m rest radius before the onset: the floor for "roll while the human
  waits" in b. Its launch delay (median -0.03 s, p90 +0.02 s) is the resolution of the delay metric (10 Hz controller trace against the
  20 Hz log); 5 b windows have a human who stays under 1 m/s (no launch reference).
- Variant b's score barely punishes a late launch (go-straight never launches and still has no zero; 31 windows have a human path < 5 m):
  read b by launch delay and early launches, a by zeros and roll.

## Reads

In the pool (job ids and the collect / anchor-check commands: the WAIT-SET handoff in [../README.md](../README.md)). Gate: go-straight
>= 0.99 on variant a and human replay >= 0.98 on a and b, else no model read runs.
