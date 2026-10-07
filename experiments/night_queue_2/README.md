# night_queue_2: P6 behaviour exam, desire, heads, backbones

status: concluded
decisions: 47, 48, 49, 50, 52, 53
index: P6 v0 holds, placement null misses gate; V-JEPA 2 flips 47%

**Question.** Does the P6 bypass exam hold, and who carries bypass?

**Conclusion.** P6 v0 holds (PDM-Lite bypasses 9 classes 100%) but placement null 0.889 misses 0.90 (decisions 52). Desire shifts 0.61-0.68 m (49); V-JEPA 2 flips pedestrians 47.0%, Qwen 41.5%, openpilot 3.0% (48).

**Read more.** research/decisions/047.md, experiments/night_queue_2/results/, `git show bcbdde4:todos/2026-09-26-night-queue-2.md`

<!-- files:begin -->
## Files

- `p6.py` (jevdrive): the behaviour-mode exam
- `night2_n2.py` (jevdrive): does openpilot carry what a bypass …
- `night2_n3.py` (jevdrive): leaderboard head x reaction (fc65452 …
- `night2_n4.py` (jevdrive): the fast channel without the lift
- `n5_depth.py` (lib): camera-conditioned monocular metric …
- `n6_backbones.py` (lib): the backbone rows of the P5 v1 …
- `night2_c_figs.py` (archive): Figures of night queue 2, executor C …
- `night2_n2.sh` (archive): Night queue 2, N2 on P5 v1
- `p6_gen.sh` (archive): PDM-Lite drives every world
- `n6_fit.sh` (archive): ridge_late + pair-Delta fits for one …

[archive/](archive/) 16 one-off code · [results/](results/) 45 result files · [figs/](figs/) 10 figures · [lib/](lib/) 2 library
<!-- files:end -->
