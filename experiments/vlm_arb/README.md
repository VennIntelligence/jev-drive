# vlm_arb: VLM slow channel and arbitration table

status: live
decisions: 84, 85, 86, 87, 89, 91, 95, 105
index: Zero-shot Qwen3-VL-4B light reading: red-light infractions 13 to 6 (privileged 5), DS +5.0; fixed bypass pbyp2, stop-line R2 vred2, yellow rule vred3

**Question.** Can a frozen vision-language model answering four discrete categorical decision questions about camera frames replace privileged simulator state in the op-drive arbitration layer without catastrophic latency or rule-interaction penalties?

**Conclusion.** Diagnostic batch (19 routes x 2 seeds): junction slow-down +5.4 DS [+1.6, +10.1] vs the speed-matched control; privileged red light +4.3 [-3.0, +11.2]; privileged bypass -28.8 on non-target routes. openjev fails the light and block lines (ego red recall 57.8%), so no VLM arm ran. Offline, Qwen3-VL-4B reads ego red 93-94% zero-shot and reaches 68-125 ms with one forward pass at reduced resolution (decisions 84, 85).

**vred (2026-10-02).** Zero-shot Qwen3-VL-4B (one pass, 1153 tokens, per-card servers, L = 0.35 s measured under load) driving R2 + R5: red-light infractions 13 -> 6 (`pred` 5), DS +5.0 [+0.7, +10.6] vs `drive`, no harm on routes without a light; diagnostic read on 19 routes, confirmation lines not evaluated ([results/vred.md](results/vred.md), [plans/2026-10-02-vlm-vred.md](plans/2026-10-02-vlm-vred.md)). Remaining infractions: yellow-onset late stop, stop target beyond the light's stop line, light turning red after the crossing.

**Bypass, offline (2026-10-02).** 22 of 26 privileged-bypass misfires on non-target routes are a route-projection clamp bug (blocker behind the route start or beyond its end), not queues; a time-based static detector fires on 8/8 obstacle runs with 1 false trigger: [results/bypass_misfire.md](results/bypass_misfire.md), [results/bypass_timedet.md](results/bypass_timedet.md), definitions in [plans/2026-10-02-bypass-offline.md](plans/2026-10-02-bypass-offline.md).

**pbyp2 / vred2 / vred3 (2026-10-02, diagnostic).** `pbyp2` (bypass with the projection fix, static >= 5 s, light memory, same-direction gap check): 0 misfire activations on the 15 other routes (`pbyp`: 26), DS -3.4 [-13.0, +6.4] against `drive` there (`pbyp`: -28.8); obstacle routes DS 42.1 [26.0, 69.2] (`drive` 27.8) because the frozen gap rule stalls the car on 3 of 4 under continuous traffic; with no gap check (`pbyp2ng`, 8 runs) 74.1 [48.2, 100.0] ([results/pbyp2.md](results/pbyp2.md)). `vred2` (R2 at the light's stop line): median stop 0.16 m short of it (`vred`: 13 of 23 stops beyond), ego-green recall while holding 28.6% -> 94.9%, but red-light infractions 7 (`vred` 6) and DS 66.5 ([results/vred2.md](results/vred2.md)). `vred3` (+ approach-only slow-down, yellow rule, commit): 7 infractions, DS 67.6 ([results/vred3.md](results/vred3.md)). Plan, parameters, deviations D25-D32, inputs by source: [plans/2026-10-02-pbyp2-vred2.md](plans/2026-10-02-pbyp2-vred2.md).

**pbyp2 / vred2 / vred3 (2026-10-02).** Bypass misfires gone after the projection fix (0 vs 26), but the same-direction gap check blocks 3 of 4 obstacle routes; without it obstacle-route DS 27.8 -> 74.1 with collisions against same-direction traffic. Moving R2's stop target to the stop line and the yellow rule did not reduce red-light infractions (7 vs 6): [results/pbyp2.md](results/pbyp2.md), [results/vred2.md](results/vred2.md), [results/vred3.md](results/vred3.md), plan [plans/2026-10-02-pbyp2-vred2.md](plans/2026-10-02-pbyp2-vred2.md) (decision 87).

**Offline reads on the logs (2026-10-02).** Release-policy replay on the logged red-light answers (K=2 releases falsely in 8 of 62 hold episodes, 3 of the 6 original `vred` infractions followed a false release; a cumulative-evidence rule halves the release delay at fewer false releases): [results/release_replay.md](results/release_replay.md). The official route's command change is within 3.1 m of the true junction entrance on all 13 junctions but 2.5 to 10 m after the light's stop line: [results/route_junction_error.md](results/route_junction_error.md). The vehicles that hit the pulling-out car braked in 8 of 19 contacts and never enough; the adjacent lane carries a vehicle every 3.2 s: [results/pullout_collisions.md](results/pullout_collisions.md). Definitions: [plans/2026-10-02-offline-analyses-definitions.md](plans/2026-10-02-offline-analyses-definitions.md).

**Stop-position probe (2026-10-03).** Frozen Cinque features detect a light / stop sign / stop line nearby (AUC 0.91-0.98) but cannot measure the distance; the route's junction point is the better anchor: [results/stoppos.md](results/stoppos.md) (decision 89).

**4B vs 8B slow channel, offline (2026-10-03).** Same 1 263 frames (21 routes), same code path: the 8B reads ego red worse than the 4B (78% vs 90% at 1153 tokens, red answered green 17% vs 5%), is better on moving leads (+26 points), costs 1.4-1.7x latency and 1.9x memory; a one-prompt directive scores 25% / 32% strict against 66% / 64% for the separate questions combined; a second frame plus the previous directive lifts the directive to about 48% but not yellow onset or stopped-lead reading; ego light is linearly readable from layer ~20 in both models, sign / block / lead not at all: [results/vlm_4b_vs_8b.md](results/vlm_4b_vs_8b.md), plan [plans/2026-10-03-vlm-4b-vs-8b.md](plans/2026-10-03-vlm-4b-vs-8b.md).

**vmerge (2026-10-03, diagnostic, 19 routes x 2 seeds).** One arm merging the vred light rows (R2 at the estimated stop line, R5, cusum release), the stop-sign hold (R3) and the time-based bypass: DS 79.1 vs `drive` 65.9 and `vred` 70.9; paired `vmerge - drive` +13.2 [+1.0, +27.0], obstacle routes +57.2 [+36.8, +77.6], light routes -3.7 [-17.5, +8.8] (`vmerge - vred` light -10.0 [-20.0, -2.5], red-light infractions 8 vs 6); no stop-sign infraction (`drive` 2). Ablations: no bypass (`vmerge - vmnobyp`) +14.6 [+4.3, +27.3], no cusum (`vmerge - vmnocusum`) +2.6 [-2.3, +8.4] (within noise), no R1 (`vmerge - vmnor1`) -3.9 [-10.0, +2.1] (within noise). Follow-up `vmj` (R2 target back at the junction entrance, R5 T_max 50 s; 6 light routes + 17280, 14 runs): light-route red-light infractions 7 -> 3, DS +10.0 [+2.5, +20.0] vs `vmerge`, equal to `vred` on those routes: [results/vmerge.md](results/vmerge.md), plan [plans/2026-10-03-vmerge.md](plans/2026-10-03-vmerge.md).

**vmerge2 (2026-10-03, diagnostic, 19 routes, seeds 0-3).** vmerge + the two vmj switches on every route (registered as the plan's section vmerge2). Seeds 0 and 1: DS 76.7, paired `vmerge2 - drive` +10.8 [+0.3, +22.2], `vmerge2 - vred` +5.7 [-4.6, +17.8], `vmerge2 - vmerge` -2.5 [-9.0, +4.1]; obstacle routes +44.8 [+25.3, +58.9] vs drive, light routes +2.4 [+0.0, +7.2] vs drive and -3.9 [-11.6, +0.0] vs vred; red-light infractions 5 (vmerge 8, drive 13). With seeds 2, 3 (vmerge2 and a rerun of the unchanged drive arm): `vmerge2 - drive` +9.2 [-0.2, +19.8], obstacle +41.4 [+31.0, +51.9], light -3.6 [-9.0, +0.0], i.e. the vmj light-route gain did not hold on the new seeds (15612 seeds 1 / 2 DS ~22, 15483 seed 3 DS 34, all with vehicle_blocked), and vmerge2's per-route DS sd over seeds is 13.8 against 3.4 for drive. Vehicle collisions 21 in 76 runs (drive 12): [results/vmerge2.md](results/vmerge2.md); why they rose in vmerge (R1 junction conflicts shared with drive, a stop-sign hold hit by a turning car on 17280, bypass pull-outs beside traffic): [results/vmerge_collisions.md](results/vmerge_collisions.md).

**vmerge3 (2026-10-04, diagnostic, 19 routes x seeds 0, 1, all 3 arms finished, no crashes).** vmerge2 with the bypass made non-privileged (obstacle from openpilot's lead head + a
15 m prior extended by front-radar returns, side and offset from openpilot's lane lines / road edges, gap from four radars: Bench2Drive
allows 4) and a release check on R2 / R3 releases; ablations `vm3priv` (privileged bypass + release check) and `vm3norel`. The release
check was meant to be a Qwen question on the wide camera: offline on 23k saved frames the best prompt has test AUC 0.875 but FPR 0.66 at the
dev threshold (line FPR <= 0.15: fails): [results/vm3_cross_offline.md](results/vm3_cross_offline.md); the check reads the front radar instead
(deviations D1-D8 in the plan; D8 = answers count at t_q + 0.35 s, verified on all 7139 answers of the vm3 arms). Plan
[plans/2026-10-04-vmerge3.md](plans/2026-10-04-vmerge3.md), numbers [results/vmerge3.md](results/vmerge3.md).
Seeds 0 and 1, DS (paired, route-cluster CI): vmerge3 68.3, vm3priv 75.4, vm3norel 70.6, vmerge2 76.7, drive 65.9.
- **Bypass line: no.** `vmerge3 - drive` on the 4 obstacle routes +12.9 [+3.6, +24.3] DS (line >= +20.7); all routes +2.4 [-6.3, +9.9].
  The perceived bypass activates on all 4 obstacle routes and measures side / offset within ~0.15 m of the map, but passes only part of the
  time: 19324 29 / 97, 2520 23.5 / 41, 19832 33 / 59, 24497 22 / 22 (vmerge2 60-100 on the same routes); dense streams never open the 3 + 3v m gap.
- **Collision line: yes.** Vehicle collisions per run vmerge3 0.158 (6 / 38) = drive 0.158 (vmerge2 0.289, vm3priv 0.263, vm3norel 0.237). It is
  bought with the unrealised bypass, not shown to be safe: the extra collisions of vmerge2 were bypass pull-outs, and fewer pull-outs happen.
- **Cost of de-privileging:** `vmerge3 - vm3priv` -7.1 [-18.0, +1.1] all routes, -32.2 [-63.6, -6.4] on obstacle routes (vm3priv keeps +45.1 over drive there).
- **Release check (radar):** `vmerge3 - vm3norel` -2.3 [-10.1, +4.5] DS, light routes -9.0 [-19.0, 0.0]; vehicle collisions 17280 / 27297: no help on 17280
  (seed 1 hit in vmerge2, vm3priv, vm3norel and vmerge3 alike; vmerge3 DS 60 vs 17-19), 27297 had no vehicle collision in any arm on seeds 0, 1; the check
  held the car 8 s twice on 27043 (timeouts, R6 re-hold) and costs speed (v_mean 1.62 vs 2.17 for vmerge2).

**Next.** Replace the CARLA-map inputs (junction entrance, stop line) by the official route commands; shorten the green-release delay; a gap policy for the bypass.

**Read more.** [plans/2026-10-02-vlm-arb.md](plans/2026-10-02-vlm-arb.md), [results/report.md](results/report.md), [results/phase_a.md](results/phase_a.md), [results/lightsweep.md](results/lightsweep.md), [plans/2026-10-02-vlm-thin.md](plans/2026-10-02-vlm-thin.md) + [results/vlm_thin.md](results/vlm_thin.md) (offline: one-pass scoring, resolution and a cut language model with a thin head for Qwen3-VL-4B's light reading; speed and accuracy), [research/decisions/082.md](../../research/decisions/082.md).
- Small-VLM first look (d182): Qwen3.5-2B/4B and Gemma 4 E2B vs Qwen3-VL-4B on the d85 light reading, none better at equal latency: [results/small_vlm.md](results/small_vlm.md).

<!-- files:begin -->
## Files


[archive/](archive/) 3 one-off code · [results/](results/) 146 result files · [plans/](plans/) 9 live plans · [scripts/](scripts/) 67 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
