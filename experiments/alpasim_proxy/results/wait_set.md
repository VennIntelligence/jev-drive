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

(results follow)
