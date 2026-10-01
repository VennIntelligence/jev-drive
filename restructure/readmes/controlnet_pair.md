# controlnet_pair: ControlNet pedestrian pairs on real WOD

status: concluded
decisions: 59
index: Paused, not no-go: deletion unclean, insertion fake-ish on 19 scenes
key: jevdrive/cn_pair.py, scripts/cn_pair.sh, scripts/cn_pair_infer.py, scripts/cn_pair_md.py

**Question.** Can one ControlNet-style generator make both pair sides so artifacts cancel?

**Conclusion.** Paused, not no-go (decisions 59): only Edge Distilled tried, no automatic checks. Restart if real-appearance pairs beyond cosmos G4 are needed, with a detector-recall check (< 5%) first.

**Read more.** research/controlnet-pair-pilot.md, research/results/cn_pair/, `git show bcbdde4:tmp/2026-09-29-controlnet-pair-handoff.md`

<!-- files:begin -->
<!-- files:end -->
