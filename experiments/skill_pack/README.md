# skill_pack: NAVSIM skill pack N0-N4

status: concluded
decisions: 64, 68, 69, 70, 71, 72, 73, 75, 88, 94, 97, 104
index: Navtest PDMS 84.2 to 91.59 (N3), flat at N4

**Question.** Can a scorer head over native plan plus extra slots raise NAVSIM PDMS, weights unchanged?

**Conclusion.** PDMS 84.2 -> N0 84.94 -> N1 87.32 -> N2 90.60 -> N3 91.59 -> N4 91.43 (-0.17 [-0.42, +0.09]); metric alignment, navhard only N4 above native (decisions 64, 68-72, 75).

**navhard off-road diagnosis (2026-10-03).** [results/navhard-offroad/report.md](results/navhard-offroad/report.md), decision 88.

**History alignment as an input rule (2026-10-03 night).** Removing the history rotation everywhere is rejected (navhard 32.67 / 31.99 vs 33.33, navtest -8 / -10 PDMS: stage 2 up, real scenes down); a navtrain-calibrated confidence selector between the shipped and the aligned rollout (post hoc) gives navhard 34.31 (+0.98 [+0.03, +1.97]) at navtest -0.09 [-0.18, -0.01]. [results/history-align/report.md](results/history-align/report.md), plan [plans/2026-10-04-history-align-plan.md](plans/2026-10-04-history-align-plan.md).

**Tracker-lag pre-compensation (2026-10-04, scorer-adapter trick, rejected).** Submitting poses that make the NAVSIM LQR + bicycle realise the plan (tracking error 1.0-1.4 m -> 0.07-0.09 m) loses: full alpha 1 navhard -5.70 [-8.06, -3.30] / navtest -5.91 (native), N4 -10.53 / -8.47; post hoc path variant at the navtrain alpha 0.25 navhard +0.42 [-1.01, +1.81], navtest -0.18. The exact plan (ideal tracker) also scores below the lagged one; losses are comfort plus DAC / LK on real scenes, gains only on stage-2 displaced starts. [results/tracker-precomp/report.md](results/tracker-precomp/report.md), plan [plans/2026-10-04-tracker-precomp-plan.md](plans/2026-10-04-tracker-precomp-plan.md).

**Road-edge diagnosis (2026-10-04, root cause found, fix not scored).** NAVSIM's CAM_F0 sits ~1.87 m above ground (ego origin is the rear axle, 0.35 m up) vs ~1.22 m for openpilot's training cameras, so openpilot reads the image correctly but scales the world to ~0.7 (lane 0.67-0.70 of map at every distance, plan speed 0.81). A virtual camera at 1.30 m gives lane ratio 0.97, speed 1.00 (400 tokens); output-side plan rescale loses on navhard (-2.6 to -13.8). Map narrowness explains about a third of edge disagreements. [results/roadedge/report.md](results/roadedge/report.md), plan [plans/2026-10-04-roadedge-diagnosis-plan.md](plans/2026-10-04-roadedge-diagnosis-plan.md).

**Read more.** research/leaderboard-skill-pack.md, research/navhard-deficit-breakdown.md

<!-- files:begin -->
## Files

- `skill_pack_n0.py` (archive): a switch between openpilot
- `skill_pack_n1.py` (archive): retrain E6's Hydra-style
- `navsim_heads.py` (jevdrive): fit on navtrain, predict navtest and
- `navsim_raise.py` (archive): NAVSIM score raising on top of skill …
- `navsim_qwen.py` (jevdrive): Qwen `L18_*` features on NAVSIM tokens
- `skill_pack_n0.sh` (archive): prep -> score T -> select -> score …
- `skill_pack_n1.sh` (archive): GIMM frames -> Cinque `temporal` …
- `navsim_raise_n3.sh` (archive): N2's configuration on 19 968 + 40 000 …
- `navsim_raise_n4.sh` (archive): N3's configuration on every navtrain row
- `skill_pack_nav_decomp.py` (archive): NAVSIM navtest per-token PDMS loss …

[archive/](archive/) 19 one-off code · [results/](results/) 92 result files · [plans/](plans/) 3 live plans · [scripts/](scripts/) 29 entry points
<!-- files:end -->
