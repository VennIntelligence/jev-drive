# op_adapt_h: layer-3 history and offset pairs for Cinque

status: live
decisions: 98 (inputs: 92, 94, 96)
index: Fake-yaw following -71..-78%, navtest +0.82; HUGSIM spins 8 to 6 only

**Question.** Does a light adaptation of Cinque on two pair types (history inconsistent with the future -> follow the logged
future; heading / lateral offset -> recover to the logged path), trained on navtrain + WOD train + CARLA frames together, cut
the history-yaw extrapolation of decision 92 without costing navtest PDMS, and does it reduce the HUGSIM low-speed spins
without the de-rotation rule?

**Conclusion.** Pilot (2 500 steps, seed 0, port readouts against shipped Cinque): (a) decision-92 probe G at 0.5-3 m/s
18.4 -> 4.1 deg (nav), 12.9 -> 3.7 (WOD), 8.9 -> 2.1 (CARLA), all CIs < 0, control unchanged; (b) navtest PDMS +0.82 [+0.47,
+1.16] (control +1.43); (c) navhard two-stage EPDMS 33.55 vs 33.59 (control 30.12); (d) HUGSIM spin set 6 / 10 vs same-code
base 8 / 10 (line <= 4 missed; control 7); (e) WOD start +0.007, stop +0.046. Open-loop history-yaw extrapolation is
trainable away across three domains at no navtest cost; the closed-loop spins barely move. Results in results/pilot1, results/it1.
Iteration 2 (results/it2): it_lowrate HUGSIM spins 9/10, navtest +0.40 [+0.15, +0.65], navhard combined 34.8 vs 33.6; it_dw3 navhard 35.4. Pilot on all 64 HUGSIM scenarios (results/hugsim64): 15 spins vs 10 base, non-spin HD delta +0.075 [+0.007, +0.151]; pilot + derot3 on the 10 spin scenarios: 1/10 spins.

One-driver table (results/one_driver.md): it_dw3 + selector is the best row on all three boards: navtest 84.70 (+0.52 vs shipped), navhard 35.76 (+2.43 [+0.17, +4.58]), HUGSIM 64 spins 4 vs 10, non-spin HD +0.056 [+0.001, +0.115]; it_dw3 alone: navhard CI crosses 0, HUGSIM spins 13.

**Next.** `it_lowrate` HUGSIM / navhard (queued); if spins stay >= 6, the closed-loop gain is not the open-loop G at 10 deg/s: measure G at 1-3 deg/s and the launch lean.

**Read more.** [../hugsim/results/launch_lean.md](../hugsim/results/launch_lean.md) (why HUGSIM spins did not drop: the cut of G is 20-25% at 0.5 deg/s, ~75% at 10; at the HUGSIM launch the local / large-signal gain is unchanged, 0.92 / 0.8 of shipped; `scripts/h_rate_probe.py`); [plans/2026-10-04-op-adapt-H-prereg.md](plans/2026-10-04-op-adapt-H-prereg.md) (design, data, lines; iterations appended).

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

[results/](results/) 35 result files · [figs/](figs/) 0 figures · [plans/](plans/) 1 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 12 entry points
<!-- files:end -->
