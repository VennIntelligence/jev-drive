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

## Reads

In the pool (job ids and the collect / anchor-check commands: the WAIT-SET handoff in [../README.md](../README.md)). Gate: go-straight
>= 0.99 on variant a and human replay >= 0.98 on a and b, else no model read runs.
