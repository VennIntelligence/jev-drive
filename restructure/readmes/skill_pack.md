# skill_pack: NAVSIM skill pack N0-N4

status: concluded
decisions: 64, 68, 69, 70, 71, 72, 73, 75
index: Navtest PDMS 84.2 to 91.59 (N3), flat at N4
key: jevdrive/skill_pack_n0.py, jevdrive/skill_pack_n1.py, jevdrive/navsim_heads.py, jevdrive/navsim_raise.py, jevdrive/navsim_qwen.py, scripts/skill_pack_n0.sh, scripts/skill_pack_n1.sh, scripts/navsim_raise_n3.sh, scripts/navsim_raise_n4.sh, scripts/skill_pack_nav_decomp.py

**Question.** Can a scorer head over native plan plus extra slots raise NAVSIM PDMS, weights unchanged?

**Conclusion.** PDMS 84.2 -> N0 84.94 -> N1 87.32 -> N2 90.60 -> N3 91.59 -> N4 91.43 (-0.17 [-0.42, +0.09]); metric alignment, navhard only N4 above native (decisions 64, 68-72, 75).

**Read more.** research/leaderboard-skill-pack.md, research/navhard-deficit-breakdown.md

<!-- files:begin -->
<!-- files:end -->
