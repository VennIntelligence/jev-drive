# op_probe: where P2 loses drivable-area information on its navtest DAC failures

status: live
decisions: (pending) (inputs: 104, 112, 144, 145, 146)
index: P2 vs WA-JEPA DAC failures: probes per stage (vision / temporal / head)
key: experiments/op_probe/plans/2026-10-06-dac-localize-prereg.md, experiments/op_probe/scripts/opb_labels.py, experiments/op_probe/scripts/opb_feats.py, experiments/op_probe/scripts/opb_wajepa.py, experiments/op_probe/scripts/opb_score.py, experiments/op_probe/scripts/opb_probe.py, experiments/op_probe/results/dac-localize.md, lib/parity_adapter.py, experiments/op_parity/scripts/pp_train.py

**Question.** On the navtest tokens where op_parity P2 fails drivable-area compliance and WA-JEPA passes, is the road geometry (a) absent
from Cinque's frozen vision features, (b) present there but lost in the temporal / hidden path, or (c) present before the plan head but not
used by it?

**Design.** Pre-registered in [plans/2026-10-06-dac-localize-prereg.md](plans/2026-10-06-dac-localize-prereg.md): ridge probes for a
drivable signed-distance raster (scorer's map layers) at each stage of P2 (vision tokens, after the adapter bias, temporal summary, plan
head hidden) and of WA-JEPA (encoder context, trajectory head hidden), trained on navtrain, read on navtest DAC-fail vs pass tokens;
plan decoders per stage scored with the devkit's `pdm_score`; input ablations on P2.

**Result (2026-10-06).** [results/dac-localize.md](results/dac-localize.md). Graded (a) plus a smaller (c); (b) not supported. Navtest-wide DAC
failures (stratified, paired): thin heads on Cinque's vision tokens fail 1.7 pp [0.7, 2.8] more than the same heads on WA-JEPA's front encoder;
a drivable-hinge head on P2's plan-head hidden recovers 0.9 pp [0.1, 1.7] of P2's 2.6 pp gap; 1.9 pp [1.1, 2.8] remains at the head (P2-H vs
WA-H). Probes: road SDF R^2 0.66 (Cinque vision) vs 0.78 (WA front encoder); on P2's failures its representations place its plan 0.75 m further
inside the road than it is (WA all-view 0.11 m). The pre-registered probe rule names no class (vision has coarse road geometry; AUC readout
confounded), so the verdict rests on the addendum-1 decoder readouts (medium).

**Joint diagnosis P2 vs WA-JEPA (2026-10-06).** [results/joint/index.html](results/joint/index.html) (paper figures in
`results/joint/figs/`, numbers in `results/joint/stats.json`; code `scripts/opj_build.py` box, `scripts/opj_figs.py` Mac). WA-JEPA leads on every
navtest scene type; the gap is a turning problem (EPDMS x100 straight -1.4, left / right turns -8.2 / -10.8; P2 DAC failures 1.6% -> 11.3% from
< 5 deg to > 45 deg, WA 1.0% -> 3.2%) and carries into HUGSIM only on turning routes (HD -0.14 [-0.28, -0.01], straight routes -0.01; P2 ahead on
nuScenes +0.12 [0.04, 0.22]). Mirror of decision 147: on each model's own DAC failures no stage of that model adds anything over ego state
(decoder lift P2-V on F -0.2 pp, WA-Cf on R -2.6 pp), while the other model's stages gain +26 / +28 pp. P2's DAC failures are in the plan and cut
the inside of turns (+0.36 m); 60% of WA's arise only in the LQR replay and 82% are < 0.3 m deep.

**Next.** Decision entry by main. Candidates: footprint SDF hinge in P2's plan-pathway fine-tune (cheap, about -0.9 pp DAC failures); encoder
unfreeze with a dense drivable-SDF auxiliary head; WA / V-JEPA front tokens as adapter memory (trade test).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
