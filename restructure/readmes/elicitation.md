# elicitation: Carrying CARLA reaction to real data

status: concluded
decisions: 42, 44
index: Zero-shot elicitation harmful: WOD RFS -1.02, NAVSIM PDMS -8.2
key: jevdrive/elicit_e1.py, jevdrive/elicit_i3.py, jevdrive/elicit_e2.py, jevdrive/elicit_e3.py, jevdrive/elicit_e4.py, jevdrive/elicit_e5.py, jevdrive/elicit_e6.py, jevdrive/elicit_seeds.py, scripts/elicit_e1_navsim.sh, scripts/elicit_e6_score.py

**Question.** Does the paired-difference reaction head transfer to real data zero-shot, via log twins or edit pairs?

**Conclusion.** No: WOD RFS -1.02 [-1.21, -0.82], NAVSIM PDMS -8.2 [-8.9, -7.5]; log twin (E3) and inpainting (E2) pairs fail (decisions 42, 44).

**Read more.** research/results/elicitation/, `git show bcbdde4:todos/2026-09-26-elicitation-program.md`

<!-- files:begin -->
<!-- files:end -->
