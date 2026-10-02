# op_adapt_h: layer-3 history and offset pairs for Cinque

status: live
decisions: 92, 94, 96 (inputs)
index: Layer-3 pilot: history-perturbation and offset-recover pairs (running)

**Question.** Does a light adaptation of Cinque on two pair types (history inconsistent with the future -> follow the logged
future; heading / lateral offset -> recover to the logged path), trained on navtrain + WOD train + CARLA frames together, cut
the history-yaw extrapolation of decision 92 without costing navtest PDMS, and does it reduce the HUGSIM low-speed spins
without the de-rotation rule?

**Conclusion.** Pending.

**Next.** Pilot `pilot` vs control `pilot_ctl`, readouts (a)-(e) of the pre-registration.

**Read more.** [plans/2026-10-04-op-adapt-H-prereg.md](plans/2026-10-04-op-adapt-H-prereg.md) (design, data, lines; iterations appended).

Why a new topic instead of extending op_adapt_l: op_adapt_l trains on cached stage-3 trunks of fixed frames (log imitation on
behaviour slices of WOD). The pairs here change the images (warps of every history frame, offsets), so stage 1-3 runs online on
rendered frames from three domains; data path, question and readouts differ. The model, checkpoint format, loss scales and
readouts of op_adapt_l are reused by import (`experiments.op_adapt_l.lib.op_adapt_l`), so its readout scripts load these
checkpoints unchanged.

<!-- files:begin -->
## Files

- `2026-10-04-op-adapt-H-prereg.md` (plans): op-adapt H：第 3 层适配 pilot …
- `op_adapt_h.py` (lib): layer-3 adaptation of openpilot Cinque …
- `h_train.py` (scripts): op-adapt H trainer and port readouts
- `h_prep.py` (scripts): every domain in one layout so that the …
- `h_nav_pool.py` (scripts): a fresh navtrain token pool on op_lb's …

[results/](results/) 0 result files · [figs/](figs/) 0 figures · [plans/](plans/) 1 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 3 entry points
<!-- files:end -->
