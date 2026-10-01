# cosmos: Cosmos-Transfer2.5 re-render of CARLA pairs

status: concluded
decisions: 56, 63
index: Cosmos v1 no-go (80% diff outside ped); G4 made 2004 pairs
key: jevdrive/cosmos_full.py, jevdrive/cosmos_pilot.py, scripts/cosmos_openpilot.py, jevdrive/cosmos_v2.py, jevdrive/cosmos_eval.py, jevdrive/cosmos_white.py, jevdrive/cosmos_report.py, scripts/cosmos_full.sh, scripts/cosmos_gen.sh, scripts/cosmos_infer.py

**Question.** After Cosmos re-rendering, do CARLA pair sides differ only by the pedestrian?

**Conclusion.** v1 no-go: 80% of the difference lies outside the pedestrian (decisions 56). v2 passes the ring check on 1/10 pairs; G4 made 2004 pairs anyway (123 GPU-h); readability follows pedestrian size (63).

**Read more.** research/results/cosmos/, research/feature-adapter-domain-shift.md, `git show bcbdde4:todos/2026-09-28-cosmos-pilot.md`

<!-- files:begin -->
<!-- files:end -->
