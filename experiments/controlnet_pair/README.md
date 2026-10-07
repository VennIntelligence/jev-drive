# controlnet_pair: ControlNet pedestrian pairs on real WOD

status: concluded
decisions: 59
index: Paused, not no-go: deletion unclean, insertion fake-ish on 19 scenes

**Question.** Can one ControlNet-style generator make both pair sides so artifacts cancel?

**Conclusion.** Paused, not no-go (decisions 59): only Edge Distilled tried, no automatic checks. Restart if real-appearance pairs beyond cosmos G4 are needed, with a detector-recall check (< 5%) first.

**Read more.** experiments/controlnet_pair/results/, `git show bcbdde4:tmp/2026-09-29-controlnet-pair-handoff.md`

<!-- files:begin -->
## Files

- `cn_pair.py` (archive): both members of a pedestrian pair are …
- `cn_pair.sh` (archive): ControlNet pair pilot, one stage
- `cn_pair_infer.py` (archive): with its run log
- `cn_pair_md.py` (archive): Review section (pulled from the box

[archive/](archive/) 4 one-off code · [results/](results/) 1 result files · [figs/](figs/) 46 figures
<!-- files:end -->
