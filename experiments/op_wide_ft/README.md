# op_wide_ft: fine-tune openpilot with a 116 deg wide slot

status: concluded
decisions: (pending) (inputs: 127, 128, 130, 133, 134, 135, 136)
index: Fine-tune with 116 deg wide: B2D small set 0/9 both arms, ol exit -0.002; gate stop
key: experiments/op_wide_ft/plans/2026-10-05-wide-ft-prereg.md, experiments/op_wide_ft/scripts/wide_run.py, experiments/op_wide_ft/scripts/wide_chain.sh, experiments/op_wide_ft/scripts/wide_lane.py, experiments/op_wide_ft/scripts/wide_report.py, experiments/op_wide_ft/scripts/wide_ol.py, experiments/op_wide_ft/scripts/wide_real.py, experiments/op_route_ft/scripts/rft.py, jevdrive/openpilot/frames.py

**Question.** Once fine-tuned with its wide slot widened to 116 deg (same 512x256 tensor, wide model focal 455 -> 160, horizon row fixed, road
camera unchanged), does openpilot use the wide view to turn on sharp / 90 deg B2D junction turns where the shipped model under-turns (decision 127)?

**Design.** Two arms identical except the wide FOV of the CARLA rows: the op_route_ft rc-bear-fix recipe (bear route adapter, decision-130 action
target, 4000 steps, seed 0) on the 8 607 CARLA exit-pair poses re-rendered once at the `spec` rig, with the wide frame cut from the same wide-sensor
capture at 58.7 deg (W58) and 116 deg (W116); same rows, same targets (teacher = the original on the 58.7 deg input), same seed. Real (navtrain /
WOD) rows keep 58.7 deg in both arms (their cameras have no vertical coverage for 116 deg), so W116 is a mixed-FOV model. Pre-registration with
the gate and decision lines: [plans/2026-10-05-wide-ft-prereg.md](plans/2026-10-05-wide-ft-prereg.md).

**Conclusion.** No (pre-registered gate: stop, clear negative). B2D small set (9 turns, desire off, zones off, spec, seed 2): W58 0 / 9,
W116 0 / 9 (shipped 2 / 9); open loop on CARLA dev junctions W116@116 - W58@58 exit-correct -0.002 [-0.015, +0.011]; the W116 model reads its
wide frame only slightly (+0.017 with 116 vs 58.7 deg input). Real data agrees: comma1M dA_act +0.016 [-0.002, +0.035] with lead -0.16 s,
PhysicalAI-AV A_H -0.08 [-0.18, +0.01]. Both fine-tuned arms degrade comma1M straights (ADE x2.6 vs shipped). (decision pending)

## Results (2026-10-05)

- **Data.** 8 607 CARLA exit-pair poses re-rendered at the `spec` rig; each wide-sensor capture (118.9 deg) cut twice: focal 455 (58.7 deg) and
  160 (116 deg); road frames byte-identical across arms. [figs/inputs.png](figs/inputs.png): three dev poses 10 m before a 3-exit junction with a
  sharp exit; columns road / wide 58.7 / wide 116, green = horizon row 151.8; look at how much of the cross street each wide frame holds.
- **Pilot** (W116 400 steps): losses down, CARLA dev exit 0.389 vs shipped 0.193, drift <= 0.042 m: pass ([results/pilot.json](results/pilot.json)).
- **Gate** ([results/gate.json](results/gate.json), [results/small.md](results/small.md), [results/ol2x2.md](results/ol2x2.md)):
  (a) took W116 0 / 9 vs W58 0 / 9 (line +2); (b) turn-in median 7.2 vs 8.5 m past the turn start (paired -0.57 [-2.0, +1.0] m, n 3; line 2 m
  earlier), peak-curvature ratio median forced 0.36 vs 0.94, choice 0.65 vs 0.70; (c) open-loop exit-correct -0.002 [-0.015, +0.011] (line +0.05).
  All fail: the 25-turn run, the W116@58 closed-loop diagnostic and the navhard guard were not run.
- **Open-loop 2 x 2** (1 505 dev rows, 119 junctions): exit-correct shipped 0.159 / 0.193 (@58 / @116), W58 0.586 / 0.579, W116 0.567 / 0.584.
- **comma1M** (42 turns, 30 straights, hindsight route; [results/comma.md](results/comma.md)): B116 - A58 dA_act +0.016 [-0.002, +0.035], lead
  -0.16 s [-0.27, -0.06]; same model, FOV only (B116 - B58) dA_act -0.024 [-0.042, -0.007]; straights W116 vs W58 lane width 1.000, speed 1.005,
  ADE 1.08 (pass); fine-tune vs shipped (A58 - S58) dA_act -0.109, straight ADE x2.63 [2.20, 3.07], speed x1.03.
- **PhysicalAI-AV** (decision-136 cases, 10 turns; 116 deg frame covers 0.87-0.91 of the f-theta image; [results/pai.md](results/pai.md)): A_H
  B116 - A58 -0.079 [-0.178, +0.010], B116 - B58 -0.128 [-0.217, -0.046]; shipped 0.490 (= decision 136).
- **Guardrails** (open loop): drift median <= 0.05 m both arms; negatives CARLA N1 0.472 / 0.491, navtrain 0.191 / 0.195, WOD 0.255 / 0.309 m
  (W58 / W116; WOD 0.004 over the "W58 + 0.05" line).
- **GIF** (B2D 10255, right, R_min 6.4 m): [figs/b2d_10255_w58.gif](figs/b2d_10255_w58.gif), [figs/b2d_10255_w116.gif](figs/b2d_10255_w116.gif);
  left chase camera, right the model's actual road (top) and wide (bottom) inputs. Look at: the 116 deg wide frame shows the whole cross street on
  the approach, yet both arms turn in late and miss the exit. Every W116 unit's interface.json records `rig.wide = sensor-f160`.

**Limits.** Single seed, one closed-loop run per cell, 9 turns. Real rows stayed at 58.7 deg (no vertical coverage), so 116 deg appeared only on
CARLA rows (22 / 48 per batch). The base recipe (W58 = rc-bear-fix at 4000 steps) itself takes 0 / 9 (rc-bear, old target, 2 / 9), so a FOV
effect had little to act on. No comma1M / PhysicalAI training rows.

**Read more.** [results/small.md](results/small.md), [results/ol2x2.md](results/ol2x2.md), [plans/2026-10-05-wide-ft-prereg.md](plans/2026-10-05-wide-ft-prereg.md), [../op_fov/README.md](../op_fov/README.md) (zero-shot
wide FOV, decision 135), [../alpamayo_turns/README.md](../alpamayo_turns/README.md) (decision 136), [../op_route_ft/README.md](../op_route_ft/README.md)
(the fine-tune recipe and the B2D turn harness).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
