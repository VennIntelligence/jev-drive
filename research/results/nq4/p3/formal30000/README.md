# P3 formal readout, registered scenes 000-009, default 30000 iterations

Registration: `todos/2026-09-26-night-queue-4.md`, P section, [P3] 18:50 entry. Every scene ran the same formal chain as
scene 000 (`scripts/p3/gpu_enable.py scene`: drivestudio OmniRe without SMPL, default 30000 iterations, then
`ds.py render` of real / x+ / x-), and one pooled readout re-ran index, openpilot `temporal`, the I3-fitted `ridge_late`
exam and the report over all ten scenes (`gpu_enable.py readout --tag formal10`, 23:25:57 CST 2026-09-27; exam dir in
`exam_dir.txt`). Thresholds, tau, iterations, scene selection, examinees and readout code are unchanged.

## Registered gate (pooled over the ten scenes)

| examinee | tau (I3) | null frames | false flips | rate | 95% CI (scene bootstrap) | gate <= 7% |
|:--|--:|--:|--:|--:|:--|:--|
| `ridge_late` Cinque (primary) | 0.527 | 230 | 10 | 4.35% | [1.30%, 7.39%] | passes |
| `ridge_late` Lebowski (descriptive) | 0.734 | 230 | 4 | 1.74% | [0.00%, 4.35%] | (descriptive) |

`verdict.json` holds the pooled verdict; the ghosting half of the gate is a human judgement: scene 000 was accepted by
the user on 2026-09-27 and scenes 001-009 on 2026-09-28 (deletions essentially fine), so both halves of the registered
gate hold. A proposed exam-item filter (pending user approval) finds no should-react frame in these ten scenes; see
[../filter/](../filter/) and the P section of the night-queue-4 todo. The scene-000-only verdict (3/23, written before scenes 1-9 ran)
and its files are kept in `scene0/`.

## Per scene

`scenes.csv` / `scenes.md` (PSNR is x+ against the log image, full frame and inside pedestrian boxes; `del_diff_px_*`
count x+ vs x- pixels differing by more than 8/255 inside / outside the 12 px-padded deleted boxes).
Technical checks: every stage rc0; 0 deleted tracks missing as nodes; determinism max |d| 0.0 in every scene; the ten
prior-reproduction head checks max 4.8e-4 < 1e-3 (`head_checks.csv`, identical to the scene-000 run).
Scene 004 has full-image PSNR 24.52 dB, below the 25 dB smoke-checklist line; it stays in the gate as registered.
Scene 000's figure, render metadata and head checks are byte-identical to the scene-000-only run.

## Four-panel figures (for the human ghosting review)

Front camera at f0 - 1 s, f0, f0 + 1 s; columns real, x+, x- and |x+ - x-|. Look at whether the x- column still shows
a pedestrian-shaped residue where corridor pedestrians were deleted (the |x+ - x-| column shows where the deletion acted);
pedestrians outside the corridor are not deleted by design.

All ten were reviewed and accepted by the user (scene 000 on 2026-09-27, 001-009 on 2026-09-28). Motion review clips
(10 Hz, log / x+ / x- / |x+ - x-|) replace single frames for later reviews: `$DATA_DIR/runs/nq4/p3/filter/clips/` on the box.

- [p3_000](nq4-p3-p3_000.png) - [p3_001](nq4-p3-p3_001.png) - [p3_002](nq4-p3-p3_002.png)
- [p3_003](nq4-p3-p3_003.png) - [p3_004](nq4-p3-p3_004.png) - [p3_005](nq4-p3-p3_005.png) - [p3_006](nq4-p3-p3_006.png)
- [p3_007](nq4-p3-p3_007.png) - [p3_008](nq4-p3-p3_008.png) - [p3_009](nq4-p3-p3_009.png)

## Descriptive: openpilot native plan (not part of the gate)

`native_plan.csv` / `native_plan.md`: Cinque and Lebowski's own plan from the same modeld forward, same frames and flip
rule (v2 = 2 s longitudinal speed, null = x+ vs real, pair = x+ vs x-). No native-plan tau exists on the I3 null, so tau
is borrowed from each model's I3 `ridge_late` examinee and labelled as borrowed; |delta| median / p90 / p95 are reported
per scene and pooled. The plan capture re-ran openpilot into a separate stream set; its `temporal` equals the gate's
stored streams bit for bit for all 30 streams x 2 models (`native_identity.csv`).

## Other files

`null_gate_all_scopes.csv` (exam output, pooled and per scene), `meta/` (render metadata with node mapping and
render_stats per scene), `timing.json` (train / render wall minutes and card per scene), `SHA256SUMS_formal10.txt`
(digests on the box before transfer; `figs/` and `meta/` paths there are the staging layout, figures now sit here).
