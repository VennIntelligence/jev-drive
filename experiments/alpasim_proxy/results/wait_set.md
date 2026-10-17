# WAIT-SET: a local closed-loop set where the logged human waits at standstill

Why: the private nuPlan board is unfiltered and waiting-rich (decision 325); on d325's uniform 5.5 s windows of the raw test logs the human
waits at a red light for the whole window in 24 % of them (public 0.4 %), and a car that rolls about 3 m where the human stands still is
zeroed by the corridor rule (experiments/wait_cap/results/next/census_red_rear.md). No local exam (held 460, public 1491) contains such a
window. This set does: public scenes whose human starts from a standstill, with the trajdata window moved so that the human is still
waiting in it. Code: [`scripts/wait_set.py`](../scripts/wait_set.py) (definitions in its docstring, fixed before any model read),
[`scripts/wait_set.sh`](../scripts/wait_set.sh) (build, reads, gate, report).

## Read a new tag (one command)

```bash
# box, repo root: one pool job (both variants, 312 windows of set v2, public 10-decision driver path + the wait-set cache), then the outcome table
bash experiments/alpasim_proxy/scripts/wait_set.sh read <tag> [after-job-ids]      # <tag>: checkpoint in runs/op_parity/runs, or P0
bash experiments/alpasim_proxy/scripts/wait_set.sh report                           # tables of every read -> $DATA_DIR/runs/alpasim_proxy/wait_set/report/
```

If the job ends rc 1 at its collect step (the ego launched in one of the 3 variant-b windows whose human never passes 1 m/s, so there is no launch
reference: `TypeError ... 'float' and 'NoneType'`), the run itself is complete: `experiments/defect_fix/scripts/df_wscollect.py <tag>` (AlpaSim venv; a CPU pool job)
writes the table with those 3 references as NaN, then `report`.

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

## Reads (set v2, 312 windows; collected 2026-10-17)

One pool job per model, the public reads' driver path (`wait_set.sh models`): **P0** (the shipped openpilot Cinque weights, no training; waits in closed loop,
decision 336), **RSG-L18-s0**, **V2** = RTB1ARO-L18-s0, and the five private anchors (garage `model_0014`, our YR10m10-F-s0 and APY10m10-AB-s0 (AP2 driver), the
shipped LTF and DD samples), with the go-straight and human-replay validation reads above as references. One read per model, so the intervals are over
windows, not seeds. Per-window outcomes of these ten reads: [wait_set/reads_all.csv.gz](wait_set/reads_all.csv.gz) (`wait_set.py anchors --csv` reads it
as it is; the defect-fix lane's reads of its own tags are not in it).

**Collection notes.** (1) `wait_set.py collect` stopped on the 3 variant-b windows without a human launch time once the ego launched there: RSG, V2, YR, LTF, DD and
APY ended rc 1 after a complete run, and the report job (1017-164839-7fc1) failed on the dependency. Their tables came from `df_wscollect.py` (collect unedited,
launch reference NaN for those windows; it looked for the run directories one level too high and printed "no finished run", fixed) and `wait_set.sh report` was
run by hand; `wait_set.py` is unchanged. (2) The first APY10m10-AB-s0 read scored 0 on all 312 windows: the AP2 driver died at every drive on `k <= s.t_first`
(sh30_driver's session has started `t_first` at None since JEV_ACC; ap2_driver read it with `getattr(s, "t_first", now)`). Fixed in `experiments/alpasim/lib/ap2_driver.py`,
re-read as 1017-172558-8b65 at priority 10 (the failed read is kept on the box under `wait_set/failed/`). (3) The held-split subset is 94 windows (44 a, 50 b), not 95:
95 was set v1; amendment 2 dropped one held b window.

### Variant a: the human never moves (146 windows; the score is 1.0 unless a zero fires)

| model | n | zeros | coll / off / corr / fail | roll median / p90 (m) | roll > 3 m | wait success |
|:--|--:|--:|:--|:--|--:|--:|
| P0 (shipped Cinque) | 146 | 1 | 0 / 0 / 1 / 0 | 0.02 / 5.02 | 22 | 0.849 |
| RSG-L18-s0 | 146 | 7 | 4 / 0 / 3 / 0 | 4.07 / 11.07 | 77 | 0.233 |
| V2 = RTB1ARO-L18-s0 | 146 | 6 | 2 / 1 / 3 / 0 | 3.33 / 11.27 | 75 | 0.411 |
| anchor garage 0014 | 146 | 7 | 0 / 6 / 1 / 0 | 0.01 / 6.29 | 28 | 0.747 |
| anchor YR10m10-F-s0 | 146 | 6 | 3 / 0 / 3 / 0 | 4.50 / 11.00 | 79 | 0.219 |
| anchor DD sample | 146 | 3 | 0 / 0 / 3 / 0 | 4.83 / 15.21 | 103 | 0.007 |
| anchor LTF sample | 146 | 6 | 2 / 0 / 4 / 0 | 6.36 / 13.78 | 112 | 0.034 |
| anchor APY10m10-AB-s0 | 146 | **28** | 25 / 0 / 3 / 0 | 9.19 / 14.10 | 135 | 0.000 |
| go-straight (starter kit) | 146 | 1 | 0 / 1 / 0 / 0 | 0.00 / 0.28 | 3 | 0.952 |
| human replay | 146 | 0 | 0 / 0 / 0 / 0 | 0.01 / 0.07 | 0 | 1.000 |

Zeros by type: collision = at-fault contact, offroad, corridor (4 m from the human's path), fail = the rollout failed before scoring. Roll = the largest distance
from the start position while the human waits (whole run in a); wait success = roll <= 1 m and no zero.

### Variant b: the human launches at 3.0 s (166 windows)

| model | n | score | zeros | early | never | launch delay median / p90 (s) | late > 1 s |
|:--|--:|--:|--:|--:|--:|:--|--:|
| P0 (shipped Cinque) | 166 | 0.365 | 3 | 23 | 129 | -0.53 / +1.54 | 8 |
| RSG-L18-s0 | 166 | 0.938 | 3 | 101 | 3 | -0.51 / +0.41 | 3 |
| V2 = RTB1ARO-L18-s0 | 166 | 0.927 | 3 | 100 | 5 | -0.63 / +0.53 | 5 |
| anchor garage 0014 | 166 | 0.368 | 16 | 33 | 98 | -0.26 / +1.53 | 14 |
| anchor YR10m10-F-s0 | 166 | 0.936 | 4 | 98 | 3 | -0.53 / +0.42 | 1 |
| anchor DD sample | 166 | 0.932 | 1 | 122 | 13 | -1.58 / +0.47 | 7 |
| anchor LTF sample | 166 | 0.954 | 0 | 135 | 8 | -1.63 / +0.37 | 6 |
| anchor APY10m10-AB-s0 | 166 | 0.981 | 3 | 149 | 0 | -1.08 / -0.44 | 0 |
| go-straight (starter kit) | 166 | 0.211 | 0 | 0 | 166 | - | 0 |
| human replay | 166 | 1.000 | 0 | 2 | 5 | -0.03 / +0.02 | 0 |

Delay = ego launch (first time faster than 1 m/s) minus the human's, over the windows where the ego launches; early = the ego passes 1 m/s before the human's onset
(3.0 s); never = no launch in the run (2.5 s after the onset). The b score does not punish an early launch by itself (RSG, V2, YR, DD, LTF, APY: 0-4 zeros of 166) and only partly a late one
(go-straight never launches and scores 0.211): read b by delay.

### Held-split windows only (94: 44 a, 50 b)

| model | a: zeros | a: coll / off / corr / fail | a: roll median / p90 (m) | a: wait success | b: score | b: zeros | b: early / never | b: delay median / p90 (s) |
|:--|--:|:--|:--|--:|--:|--:|:--|:--|
| P0 (shipped Cinque) | 1 | 0 / 0 / 1 / 0 | 0.02 / 4.18 | 0.841 | 0.381 | 1 | 8 / 37 | -0.53 / +0.52 |
| RSG-L18-s0 | 2 | 0 / 0 / 2 / 0 | 5.97 / 11.79 | 0.227 | 0.969 | 0 | 33 / 1 | -0.68 / +0.34 |
| V2 = RTB1ARO-L18-s0 | 2 | 0 / 0 / 2 / 0 | 7.71 / 11.79 | 0.386 | 0.958 | 0 | 32 / 1 | -0.76 / +0.53 |
| anchor garage 0014 | 3 | 0 / 2 / 1 / 0 | 0.01 / 5.81 | 0.705 | 0.322 | 6 | 10 / 30 | -0.36 / +1.52 |
| anchor YR10m10-F-s0 | 2 | 0 / 0 / 2 / 0 | 5.60 / 11.30 | 0.227 | 0.962 | 0 | 29 / 2 | -0.71 / +0.42 |
| anchor DD sample | 1 | 0 / 0 / 1 / 0 | 6.00 / 14.29 | 0.023 | 0.922 | 0 | 35 / 6 | -1.66 / +0.50 |
| anchor LTF sample | 2 | 0 / 0 / 2 / 0 | 6.88 / 13.36 | 0.023 | 0.941 | 0 | 40 / 5 | -1.68 / +0.24 |
| anchor APY10m10-AB-s0 | **8** | 6 / 0 / 2 / 0 | 9.23 / 14.26 | 0.000 | 1.000 | 0 | 44 / 0 | -1.13 / -0.33 |
| go-straight (starter kit) | 0 | 0 / 0 / 0 / 0 | 0.00 / 0.23 | 0.977 | 0.203 | 0 | 0 / 50 | - |

Paired score differences over all windows (log-clustered bootstrap): V2 - RSG a +0.007 [-0.022, +0.037], b -0.011 [-0.034, +0.011]; P0 - RSG a +0.041
[+0.013, +0.075], b -0.573 [-0.650, -0.496].

### Anchor check: public score + the variant-a zero rate against the private board

`wait_set.py anchors` (six anchors: the five and go-straight, whose public number is N1's 300-scene sample). z_a = share of variant-a windows scored 0;
linear fit private = public - k z_a; mixture fit private = (1 - p) public + p (1 - z_a); the LOO columns refit without that anchor.

| anchor | private | public | zero rate a | pred linear | pred mixture | LOO linear | LOO mixture |
|:--|--:|--:|--:|--:|--:|--:|--:|
| G0014 | 0.848 | 0.740 | 0.048 | 0.674 | 0.848 | 0.665 | 0.848 |
| YR10m10-F-s0 | 0.800 | 0.925 | 0.041 | 0.869 | 0.942 | 0.872 | 0.943 |
| starter (go-straight) | 0.799 | 0.522 | 0.007 | 0.513 | 0.761 | 0.512 | 0.687 |
| DD-sample | 0.772 | 0.797 | 0.021 | 0.769 | 0.890 | 0.769 | 0.903 |
| LTF-sample | 0.738 | 0.824 | 0.041 | 0.768 | 0.892 | 0.769 | 0.901 |
| APY10m10-AB-s0 | 0.640 | 0.935 | 0.192 | 0.673 | 0.871 | 0.869 | 0.882 |

| Kendall tau-b against private | public alone | zero rate a (1 - z_a) | linear | mixture | LOO linear | LOO mixture |
|:--|--:|--:|--:|--:|--:|--:|
| six anchors (k = 1.367, p = 0.508) | -0.47 | +0.14 | +0.20 | -0.07 | -0.33 | +0.07 |
| the five without go-straight (k = 1.413, p = 0.351) | -0.60 | +0.11 | +0.40 | -0.40 | -0.20 | -0.20 |

Mean absolute error: six anchors linear 0.099, mixture 0.114, LOO linear 0.134, LOO mixture 0.132, public alone 0.153; five anchors 0.060 / 0.129 / 0.092 / 0.154, public
alone 0.128.

### Reading

- **Waiting.** P0 stays at its rest point (median roll 0.02 m, wait success 0.85), as do the garage checkpoint (0.75) and go-straight. RSG, V2 and YR leave it in about
  half of the windows (77 / 75 / 79 of 146 roll more than 3 m; wait success 0.23 / 0.41 / 0.22), the LTF and DD samples in most (112 / 103; success 0.03 / 0.01), APY
  in nearly all (135; median 9.2 m). At b they launch before the human: median 0.5-0.6 s early for RSG, V2 and YR, 1.1 s for APY, 1.6 s for the samples.
- **The scorer rarely turns it into a zero.** Variant-a zero rates are 4-5 % for RSG, V2, YR, LTF and the garage, 2 % for DD, 0.7 % for P0 and go-straight. V2 and RSG
  do not differ (+0.007 [-0.022, +0.037] on a); P0 is +0.041 [+0.013, +0.075] above RSG on a and -0.573 below it on b (it does not launch in 129 of 166 windows).
- **APY is the exception**: 28 zeros in 146 (25 at-fault contacts, 3 corridor; 8 of 44 on the held-split subset), it never waits and launches early in 149 of 166 b windows.
  The anchor most hurt on private is the only one the waiting set separates.
- **Public + waiting set does not order the anchors like private.** Public alone ranks them backwards (tau -0.47; -0.60 on the five, decision 262). With z_a the in-sample
  fits reach +0.20 (linear) and -0.07 (mixture), the leave-one-out predictions -0.33 and +0.07 (five anchors: +0.40 / -0.40 and -0.20 / -0.20); the leave-one-out error
  (0.13 / 0.13; five anchors 0.09 / 0.15) stays near the public-alone error (0.15; 0.13), more than half of the 0.21 spread of the private scores. The fit is carried by APY alone: k = 1.4 would mean a zero in a static window costs more than a whole scene even if every
  private window were a wait, and the mixture needs p = 0.35-0.51 waiting windows against 0.30 in d325's uniform windows; left out of the fit, its private score is predicted
  at 0.81-0.93 (0.640 on the board). The other anchors have 1-7 zeros of 146 (go-straight 1, DD 3, YR and LTF 6, garage 7), which is no order. The waiting set flags a waiting failure (APY) but does not
  predict the size of the private gap or rank the rest.
