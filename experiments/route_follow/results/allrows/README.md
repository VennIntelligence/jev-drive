# allrows: first round under the new split (nuPlan all 1491 = training, PAI = exam), 2026-10-13

Prereg: [plans/2026-10-13-allrows-prereg.md](../../plans/2026-10-13-allrows-prereg.md). Chain: `scripts/rf_allrows.sh`. Files: `train_gates.md`,
`confirm.md / .json` (line P: public 1491 at 10 decisions, ol_confirm vs the stored YR10m10-F-s0 rollouts), `deficit.md / .json` (line D, rf_deficit.py).

| line | RSGall-L18 (A, +45 % rows) | RSGallV-L18 (B, + speed-history drop 0.5) |
|:--|:--|:--|
| G-rely | pass, both seeds | pass, both seeds |
| dev ADE (≤ RSG + 0.01) | pass (0.5283 / 0.5285) | s0 at the line (0.5379), s1 over (0.5395); not excluded (that would need both seeds to fail) |
| P: score diff vs RSG-L18, CI lower > −0.01 | −0.0034 [−0.0073, +0.0001]: pass | −0.0055 [−0.0104, −0.0011]: **fail** |
| P: zeros diff ≤ 0 | +3.0 [−2.1, +8.6]: **fail** | +8.5 [+2.6, +15.1]: **fail** |
| D: plan 2 s deficit vs RSG-L18 (CI lower > 0, median above) | +0.001 [−0.004, +0.006] (reported) | **+0.143 [+0.125, +0.162], median −0.247 vs −0.433: pass** |

Reading: the extra 460 scenes' rows do not help on public nuPlan (zeros +3, mostly offroad, at-fault +4, both CIs include 0). The speed-history drop
takes out about 75 % of the plan's shortness (rollout-mean deficit −0.191 -> −0.049 m) but costs zeros (+8.5, offroad and corridor) and lateral GT
(+0.056 m [+0.030, +0.082]). Neither arm replaces RSG-L18 on nuPlan. Selection for the PAI exam (prereg rule: fewest public zeros, then mean score):
**RSGall-L18-s1** (47) and **RSGall-L18-s0** (49), copied to Tokyo `/data/runs/op_parity/runs/<tag>/` (md5 checked); line X is the coordinator's.
