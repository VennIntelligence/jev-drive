# cosmos: Cosmos-Transfer2.5 re-rendering of CARLA pedestrian pairs

status: concluded
decisions: 56, 63
headline: Cosmos v1 no-go (80% of diff outside pedestrian); v2 passes ring check on 1/10 pairs; G4 run made 2004 pairs, 123 GPU-h

**Question.** After Cosmos-Transfer2.5 re-renders both sides of an x+ / x- CARLA pair into photoreal video, do the two differ only by the pedestrian, and is the result usable as training data?

**Conclusion.** v1 (translate each side separately) is no-go: openpilot `temporal` difference lies 80% outside the pedestrian region (decisions 56). v2 (x- translated once, pedestrian pixels repainted with anchoring) gives recall 0.982, zero outside-mask difference by construction, but only 1/10 pairs pass the registered ring criterion (a post-hoc void of its second clause was recorded as a deviation); the user approved G4 anyway and the full run produced 2 004 pairs, about 123 GPU-h, as training data for op-adapt B. Cosmos does not restore pedestrian readability: CARLA 75 pairs already read at stage 3 AUC 0.815 vs 0.746 in Cosmos; readability is set by pedestrian size (<500 px: 0.57 both) (decisions 63).

**Read more.** research/feature-adapter-domain-shift.md, research/survey-counterfactual-video-gen.md, research/results/cosmos/, `git show bcbdde4:todos/2026-09-28-cosmos-pilot.md`, `git show bcbdde4:tmp/2026-09-29-cosmos-full-handoff.md`

<!-- files:begin -->
<!-- files:end -->
