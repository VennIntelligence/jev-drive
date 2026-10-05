# cosmos: Cosmos-Transfer2.5 re-render of CARLA pairs

status: concluded
decisions: 56, 63
index: Cosmos v1 no-go (80% diff outside ped); G4 made 2004 pairs

**Question.** After Cosmos re-rendering, do CARLA pair sides differ only by the pedestrian?

**Conclusion.** v1 no-go: 80% of the difference lies outside the pedestrian (decisions 56). v2 passes the ring check on 1/10 pairs; G4 made 2004 pairs anyway (123 GPU-h); readability follows pedestrian size (63).

**Read more.** experiments/cosmos/results/, research/feature-adapter-domain-shift.md, `git show bcbdde4:todos/2026-09-28-cosmos-pilot.md`

<!-- files:begin -->
## Files

- `cosmos_full.py` (archive): Cosmos-Transfer G4 full generation
- `cosmos_pilot.py` (lib): CARLA counterfactual pairs re-rendered …
- `cosmos_openpilot.py` (lib): openpilot Cinque on the Cosmos pilot …
- `cosmos_v2.py` (archive): tune the generator so that a translated …
- `cosmos_eval.py` (lib): Each step runs in the env that has its …
- `cosmos_white.py` (archive): where do objects come out pure white?
- `cosmos_report.py` (archive): checks 1-4 against the pass lines …
- `cosmos_full.sh` (archive): the one-shot lane driver on the
- `cosmos_gen.sh` (archive): drive the selected P5 v1 worlds again …
- `cosmos_infer.py` (archive): Cosmos-Transfer2.5 on the pilot clips

[archive/](archive/) 13 one-off code · [results/](results/) 47 result files · [figs/](figs/) 13 figures · [lib/](lib/) 3 library
<!-- files:end -->
