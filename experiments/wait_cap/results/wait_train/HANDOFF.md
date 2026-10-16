# WAIT-TRAIN handoff (2026-10-16 22:05 JST)

Prereg: `experiments/wait_cap/plans/2026-10-16-wait-train-prereg.md` (pushed 7c648469 before any model output; Amendments A = yield rows from
decision 331, B = log batches + control trained with the RSG-L18 recipe). Code: `experiments/wait_cap/scripts/wt.sh` (chain), `wt.py` (steps and gates),
`experiments/wait_cap/lib/wc_arc.py` (loss). Box paths below: `O = $DATA_DIR/runs/wait_cap/train`. Job ids: `$O/jobs.txt`; stage lines: `$O/stages.txt`
and `$O/rows.txt`; per-job logs: `$O/pool/<job name>/{log.txt,STATUS,DONE|ERROR}`.

## State at handoff

- Done: batch A (55 logs, 2 680 rows, leak check clean, teacher check max |diff| 0.0078), batch B (archives that landed later), and the control
  trainings WTCP-L18-s0 (1 000 steps) and WTC-L18-s0 / s1 (2 500 steps, RSG-L18 recipe). The control reads were queued or running: waitcap / dev /
  ledger-F plans, and held 460 @18 (`wt-held-WTC-L18-s{0,1}-{0..5}`). No ERROR in the chain.
- Waiting: the download (tmux `jev:wcdl`, log `/root/autodl-tmp/ujs/runs/download/navsim/20261016-192944/log.txt`). 6 archives were missing
  (20, 24, 103, 138, 159, 196), all retrying, none failed.

## What submits what (all pool jobs, owner `wait-train`)

1. `wt-dl-20 .. wt-dl-196` (1016-215916-6a0b .. 1016-215918-7090): one per missing archive, `--when-exists <state>/<archive>.done`, chained.
2. `wt-finish` (1016-215919-f21b, after the last wait): `wt.sh finish` submits the last batch (`rows-batch C`: select / index / prep / rmnav /
   teacher), then `wt.sh pack A,B,C` (`wt-pack` -> `wt-gg-off` / `wt-gg-w0` 60-step identity runs -> `wt-bitcmp`), then `wt.sh pilot`
   (`wt-train-WTJ2P-L18-s0`, `wt-train-WTJ5P-L18-s0` after bitcmp -> their reads -> `wt-gate-pilot`).
3. `wt-gate-pilot` (`wt.py gate --submit`): on PASS it runs `wt.sh full <j2|j5>`, which submits the J trainings (2 seeds), their reads, the J held
   reads, and `wt-clgate-held` (after J held + the control held reads).
4. `wt-clgate-held` PASS -> `wt.sh stage-sub`: the public static scene for 4 tags + PAI stop17 for J-s0 (gains 10 + JEV_ACC=1) -> `wt-clgate-sub`.
5. `wt-clgate-sub` PASS -> `wt.sh stage-pub`: public 1491 @10, 4 tags x 10 chunks -> `wt-clgate-pub`.
6. `wt-clgate-pub` PASS -> `wt.sh stage-pai`: PAI 281 for J-s0 (the reference's 8 chunk lists) -> `wt-clgate-pai` (final; reported, no line).

## Verdict files (box; each `wt.py` gate also writes a copy into the box's working tree `experiments/wait_cap/results/wait_train/`)

| stage | file | pass field |
|---|---|---|
| rows / pool | `$O/batches/<B>/{cells.md,select.json}`, `$O/pack.json` | the pack asserts inside the job |
| identity | `$O/identity.json` | `identical` (must be true; else the control is void, prereg B) |
| pilot | `$O/pilot_gate.{json,md}` | `verdict`, `pick` (the chosen dose tag); also the per-arm `ledger_F` secondary reads |
| held | `$O/cl_held.json` | `passed` |
| per-scene | `$O/cl_sub.json` | `passed` (static scene nonzero for >= 1 J seed, or J stop17 zeros < reference) |
| public guard | `$O/cl_pub.json` | `passed` (paired lower bound > -0.01 and zeros J <= C + 5) |
| PAI 281 | `$O/cl_pai.json` | reported only (mean, zeros, stop17 zeros, paired CI vs `runs/route_follow/pai_g10/acc`) |

A failed closed-loop line writes `$O/STOPPED` and submits nothing further. A failed pilot submits nothing; `pilot_gate.json` holds the verdict.

## Registered stop conditions (prereg sections 5-7)

- The bitcmp is not identical: stop. Retrain C on the merged pool at weight 0 before any pilot reading.
- Pilot (per dose; all five must hold):
  - wait W >= 0.60
  - go G >= 0.85
  - W_L into the lead box <= 0.05
  - dev ADE <= C + 0.01
  - lateral at 2 s, median |diff| vs C <= 0.05 m
  - Launch-lock (dW >= 0.20 and dD < 0.5 dW) or both doses failing: stop, no rescue variant.
- held: zeros (2-seed mean) J <= C + 3. Per-scene: see the table. Public: CI lower bound > -0.01 and zeros J <= C + 5. PAI: report.

## By hand

- Download "FAILED" in the log: re-run once in the box repo:
  `scripts/tmux_run.sh wcdl env PIGZ=-p4 WAIT_LIST=/home/ujs/data/tmp/wc_dl/list.tsv ROUTE=clash JOBS=12 scripts/download_navsim.sh waittrain`
  (it resumes). The `wt-dl-*` jobs fire by themselves once the `.done` files appear. `jev:tvdl` (the full-trainval CAM_F0 download) is not this chain's.
- The per-scene stage is automatic, but its inspection is descriptive. Look at the static public scene's rollouts (`$O/sub/<tag>/run`) and the
  stop17 per-scene table in `cl_sub.json`.
- After the verdicts: copy the result files from the box into `experiments/wait_cap/results/wait_train/` on the Mac. Write the decision entry
  (next free number; `research/decisions.md` index line + `research/decisions/<N>.md`, Chinese), then update `experiments/wait_cap/README.md` and
  `experiments/INDEX.md`. The main report is the three capabilities (wait share W, go share G, PAI stop17); scores are guards. List the ledger
  S_L / Y_P / Y_V F reads as secondary.
- Row counts for the prereg's 补记 0: `$O/batches/*/cells.md` and `$O/pack.json` (append; never edit the text above it).

## How to tell it stalled

- `ls $O/pool/*/ERROR` is non-empty: a job failed and its dependents never start. Read that job's `log.txt`.
- The `wt-dl-*` jobs stay queued while the download log shows no new line for > 1 h: the download died. Check `tmux list-windows -t jev` and the log tail.
- `python -m jevdrive.cl top` shows `wait-train` jobs queued with idle cards for > 30 min: check `--after` ids against failed jobs.
- After `wt-finish` is done, `wt-bitcmp` and `wt-gate-pilot` should exist in `$O/jobs.txt` within ~1 h. After a PASS gate, the next stage's job names
  appear in `$O/jobs.txt` within minutes.

## Amendment C (2026-10-17): WAIT + desire composed arm WD replaces the WAIT-only formal

Prereg Amendment C + 补记 C1-C3 (`plans/2026-10-16-wait-train-prereg.md`). Changes to the chain above:

- `wt-gate-pilot` no longer runs `wt.sh full`: it writes the picked dose (j2 | j5 | STOP) to `$O/pilot_pick.txt`. `wt.sh full` refuses unless
  `WT_FALLBACK=1` (exit 3).
- Done before the WAIT pilot: `wd_code_identity.json` (code 520e9407 vs the D1 / D2 code with both off: bit-identical, RSG hash), D2 teacher
  `runs/desire_cap/train/teacher/navtrain_full.s*of12.npz` (G-proto 0.0), 40-step smoke `WDSMOKE-s0` (scratch).
- Waiting on `pilot_pick.txt`: `wd-launch` (1017-011520-986b) -> `wt.sh wd-pilot`: `wd-rows` (dt.py rows on on1_L18RBW, D0's rule) ->
  `wd-gg-w0` + `wd-bitcmp` (vs WTGG-off -> `wd_identity.json`) -> `WD<J2|J5>P1-L18-s0` / `WD<J2|J5>P3-L18-s0` (D2 weight 1 / 3, 1 000 steps)
  -> WAIT reads + `dc.py plans` -> `wd-gate-pilot` (`wt.py wdgate --submit`, verdict `$O/wd_pilot_gate.{json,md}`):
  PASS -> `$O/WD_PASS`, `wt.sh wd-full <lam>`: WD<dose>-L18-s{0,1} 2 500 steps, reads, held 460 @18, `wd-follow` (rf_follow on J + C held reads,
  `$O/wd_heldturn/follow`), `wd-clgate-held` (zeros J <= C + 3 AND held turns 2-seed mean J >= C - 2 on `held_turns24.txt`) -> stage-sub ->
  stage-pub -> stage-pai exactly as above. FAIL -> `$O/FALLBACK`, WAIT-only formal (`full <dose>`).
- `wd-deadline` (1017-011521-c597): sleeps until 24 h after `pilot_gate.json`; no `WD_PASS` and no `FALLBACK` -> writes `FALLBACK`, WAIT-only formal.
  WAIT pilot STOP -> neither the composed pilot nor any formal runs.
- Stall signs: `ls $O/pool/wd-*/ERROR`; `wd-rows` waits on `lib/route_desire.py` in the box tree (it is there since f7e5919e).
