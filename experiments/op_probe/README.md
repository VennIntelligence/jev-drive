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

**Next.** Small read (400 navtest tokens), then the full read: [results/dac-localize.md](results/dac-localize.md).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
