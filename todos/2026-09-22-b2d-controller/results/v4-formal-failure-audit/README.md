# Formal v4 failure evidence (final)

This folder owns only read-only analysis of `/data/runs/b2d/controller/formal-v4`. The live simulator, runtime files and controller configurations are managed elsewhere. Human causal observations are maintained in [../../agents/v4-formal-failures.md](../../agents/v4-formal-failures.md).

Final compact products are [selected-60-attribution.csv](final-v1/selected-60-attribution.csv), [all-attempt-attribution.csv](final-v1/all-attempt-attribution.csv), [manifest](final-v1/manifest.json), and [verification](final-v1/verification.json). There are 60 selected route cases and 63 total attempts. Raw editions are archived separately at `/data/runs/b2d/controller/formal-v4-failure-audit/snapshots`, final edition **034**. [relocation-manifest.json](relocation-manifest.json) records the original-to-archive mapping, inventory hash, 166 files and 218,331,316 verified bytes. The ignored `snapshots` symlink preserves original path access; do not commit or dereference it into the repository.

The writer is stopped. No more editions are needed for this completed campaign. The commands below document reproduction; use a new output directory if regenerating evidence. Historical manifests, failed analysis edition 012 and all earlier versions were preserved unchanged.

Run from `/data/worktrees/jev-drive-controller-v2`:

```bash
/data/envs/carla/bin/python \
  todos/2026-09-22-b2d-controller/results/v4-formal-failure-audit/audit.py \
  --run /data/runs/b2d/controller/formal-v4 \
  --out /data/runs/b2d/controller/formal-v4-failure-audit-reproduction-v1 \
  --watch-seconds 0
```

The initial watcher was stopped at completed edition 033 to avoid repeating large raw contexts. All editions remain preserved. The final single-pass invocation produced edition 034 after the campaign ended. Only one writer may use this output directory. Each changed set of completed attempt events creates the next exclusive `edition-NNN` directory. Existing editions are never overwritten. The final root campaign event creates a final edition even if the attempt set has not changed. The helper imports the campaign's **archived** report code for official status and basic telemetry semantics. It does not run the evaluator or import CARLA.

Each edition includes:

- `all-attempt-cases.csv`: all completed attempts seen, completion and strict subset SR separately, collision categories, blocked/deviation, unknown causal labels, source attempt paths, pose degradation and control quality, physical low-speed intervals and full/precollision CTE.
- `contexts.json`: unmodified official record, captured critical criterion events with array index, exact frame and ±20-frame control context with original JSONL line number; also all invalid-pose/nontracking records and dropout/stall interval boundaries.
- `summary.json`: interim counts, exact attempt signature and campaign-ended flag. It is not a final G4 qualification report.
- `manifest.json`: hashes and byte sizes of completed attempt source files and helper/report identities. Starting with edition 002, the exact current helper is copied into each new edition. `audit-v1.py` preserves the original helper bytes used by earlier editions; its SHA256 matches their manifests.

No failure is automatically attributed to the controller or to other traffic. `blocked_deviation_cause` and `unfinished_cause` stay `unknown` pending manual review. Collision responsibility is separately unknown because actor trajectories/impulses are not recorded here. This prevents event counts from becoming unsupported causal conclusions.

The low-speed/high-target detector can be broken into short fragments by safe-controller ticks that have no target. The current helper therefore also records uninterrupted `abs(speed)<0.5 m/s` intervals of at least five seconds, irrespective of target, and explicitly counts `invalid_motion` separately from `invalid_pose`. These are descriptive intervals including all startup/parking samples, not new acceptance gates.

First-harness-finished selection and ten-requested-route denominators belong to the report owner's final G4 calculation. This archive retains earlier failed attempts and does not select a best run. No full 220-route score or comfort score is inferred.

`finalize.py --edition <final-edition> --out <fresh-final-directory>` refuses an unfinished campaign or fewer than 60 distinct group/route keys. It produces an all-attempt attribution CSV and a selected 60-route CSV, with counterpart types, captured event frames/indices, raw paths, pre-contact fault counts, and physical stall boundaries. It preserves unknown causal labels and marks rows without any harness-finished attempt. Fixtures verified rejection of unfinished input, preservation of 60 cases including missing telemetry, and refusal to overwrite.

Edition 012 is intentionally incomplete and carries `FAILED.md`: the initial missing-telemetry guard was insufficient for an infrastructure crash. `audit-v2.py` preserves that helper version. Edition 013 and later use explicit missing-telemetry fields; a `manifest.json` marks a completed edition. Readers should choose the latest `edition-*/manifest.json`, not merely the newest directory.

The low-speed detector uses the signed forward speedometer component, not a full three-dimensional velocity magnitude. Post-contact lateral displacement can coexist with a small longitudinal reading. Its interval is therefore descriptive; it does not alone prove total immobilization. The severe 25424 and 2091 observations are additionally supported by route-progress and position/event context. Likewise, a small speed at contact does not establish that the other actor was at fault.
