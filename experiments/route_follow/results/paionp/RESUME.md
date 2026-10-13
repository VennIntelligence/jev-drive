# PAIONP: state at the box drain (2026-10-13 ~23:00 JST) and how to continue

Prereg: [../../plans/2026-10-13-paionp-prereg.md](../../plans/2026-10-13-paionp-prereg.md) (Amendment 1). Code: scripts/rf_paionp.py, scripts/rf_paionp.sh.

## Done

- **Pool copied** from Tokyo to the box, md5-equal file by file: `cache/on1_PAIRB0{,@warp}` (tab, agent, route, teacher, front; road.npz was not
  copied, it is all zeros), `route_mem/feat/on1_PAIRB0.npz`, source cache tab -> `$DATA_DIR/runs/route_follow/paionp/src/cache_pol_tab.npz`.
- **fut vs plan** ([correction.md](correction.md), [correction.json](correction.json)). 22 695 rows. imit is valid at all 8 poses on 19 261 of them;
  the other 3 434 have a zero-filled fut and are dropped. No usable fut crosses a road edge. Turn-approach rows: 1 543 in 22 scenes (left 789, right 754).
  Straight rows: 16 275. Other rows: 1 443. |fut - plan| at 2 s on turn-approach rows: median 3.40 m [1.59, 6.42], so G0 passes. Inside the turn,
  fut is +0.75 m [-1.12, +3.02] further toward the inside at 2 s (88 % of rows > 0.5 m). 20-0 m before the entry, fut covers 4.3 m less arc by 4 s
  (the human is slower: a braking signal). On straight rows fut covers +3.2 m [-2.4, +17.9] more arc: the ego lags the human there (ds0 median -0.4 m,
  IQR down to -11 m).
- **R1** ([r1_PAIRB0.json](r1_PAIRB0.json), 4 096 rows): RB0-F-s0 over-tolerance share 0.034 (line 0.01, FAIL by the line). Median 1.3 cm,
  p99 6.9 cm, max 12.5 cm. Control RB0-F-s1 0.974. Amendment 1 records this as cross-hardware / route-feature noise, and training went ahead.
- **Pools**: `on1_PAIonp` (19 261 rows, tags `turn`, `side`) and `on1_RSGPAIonp` = on1_L18RB0 (`self`, 17 429) + on1_PAIonp (`both`).
  `--on-weight pool=p:0.3590,turn=approach:2`: PAI rows are 30 % of the draws, and approach rows are 14.8 % of the PAI draws.
- **Smoke** (60 steps): dev ADE at step 0 is 0.5276, equal to the init. Mixed rows: 17 429 self / 19 261 both.
- **Training done**: `RSGPAIonp-L18-s0` (dev ADE 0.5335) and `-s1` (0.5324), in `$DATA_DIR/runs/op_parity/runs/RSGPAIonp-L18-s{0,1}/ckpt-final.pt`.

## In flight at the drain (running pool jobs left to finish), cancelled ones

- Running at the drain: held reads of s0 (`rm-read-RSGPAIonp-L18-s0-{0,2,3,4,5}`; chunk 1 was already done), the s0 guard
  (`pon-g10-RSGPAIonp-L18-s0-{0..9}`), and the P1 open-loop read (`pon-p1ol`).
- Cancelled (never started): the six s1 held reads, both `rm-cover`, the s1 guard (10 jobs) and `pon-report`.

## Continue after the reboot (box, repo root, DATA_DIR exported)

```bash
cd ~/data/jev-drive && git pull && export DATA_DIR=/root/autodl-tmp/ujs
O=$DATA_DIR/runs/route_follow/paionp; M=$DATA_DIR/runs/route_mem/read; G=$DATA_DIR/runs/route_follow/guard; W=$DATA_DIR/runs/onpol_long
# 1. which reads finished (each held chunk: $M/<tag>/c<i>/run; each guard chunk: $G/read10/<tag>/b<i>; pool logs: DONE / ERROR)
ls $M/RSGPAIonp-L18-s0/ $G/read10/RSGPAIonp-L18-s0/ 2>/dev/null; ls $O/pool/*/ | grep -B1 -E "DONE|ERROR"
# 2. s1 held reads + cover (also s0 if any s0 chunk is missing or ERROR: move its read dir aside first)
PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read RSGPAIonp-L18-s1
# 3. guard chunks still missing (s1: all ten; s0: only the ones without a finished b<i>)
for i in $(seq 0 9); do T=RSGPAIonp-L18-s1; .venv/bin/python -m jevdrive.cl submit --owner paionp --name pon-g10-$T-$i --tries 1 --priority 14 --no-check \
  --log-dir $O/pool/pon-g10-$T-$i --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- \
  bash experiments/onpol_long/scripts/ol_chain.sh readlist $T $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i; done
# 4. s0 cover (if step 2 did not resubmit s0), then the report after every read / cover job id above (--after id,id,...)
#    rm_chain's own cover command: see `submit-read` in experiments/route_mem/scripts/rm_chain.sh
.venv/bin/python -m jevdrive.cl submit --owner paionp --name pon-report --tries 1 --priority 14 --no-check --log-dir $O/pool/pon-report \
  --after <ids> --vram 0.5 --gpus 1,2,3,4,5 --pin-strict --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_paionp.sh report
# 5. if pon-p1ol did not finish: rerun its line from rf_paionp.sh `submit` (rf_pai_openloop.py --p1 ... --name p1_paionp --out experiments/route_follow/results/paionp)
```

Then do the verdict with the prereg lines: s0 held mean >= 0.8566, held zeros <= 37, guard zeros <= 50. Write the Tokyo exam_turn26 request line
only if all three pass; the coordinator decides whether it runs.
