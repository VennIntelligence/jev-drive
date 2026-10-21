# HARM-MIX: Harmonizer-ON twins of the PAI human rows (built 2026-10-22, no training)

Plan (written before any render): [../../plans/2026-10-22-harmonizer-mix.md](../../plans/2026-10-22-harmonizer-mix.md). Code: `scripts/hmix.sh`,
`scripts/hmix.py`, `lib/twin_tokens.py`, `rm_train.py --twin`. Box run dir `$DATA_DIR/runs/round7/hmix/`.

**What exists.** A twin store `on1_PAIHTW` holds **103 904 rows**: on1_PAIH 39 299, PAIH05 35 855, PAIH15 28 750. They come from 246 of the 717
non-held scenes, about 34 % of the non-held PAI human rows (launch rows 40 %, straight cruise 41 %; [select.json](select.json)). The store has the
row names plus a zero-copy token index over the ON pools `on1_PAIHON` / `PAIHON05` / `PAIHON15` (about 28 GB, which must outlive the store). Every
scene was rendered with pai_rows' H recipe plus `HARMONIZER=1`. Training switch: `rm_train --twin PAIHTW:<p>`. A drawn pool row that has a twin
takes the ON tokens with probability p; everything else (labels, ego, route, teacher, name, time scale) stays the OFF row's own.

## Gates ([gates.json](gates.json); lines fixed in the plan)

| Gate | Line | Result |
|---|---|---|
| A alignment | every ON row has its OFF row; no OFF row in those scenes is missing; every non-token field bit-equal | **pass**: 39 299 / 35 855 / 28 750 rows matched, 0 missing; 27 / 29 / 29 of 29 / 31 / 31 fields bit-equal (tab, agent, road, route, route features, teacher di / pi); the other two, `served` and teacher out, differ, as they must (token-dependent: teacher out median abs diff 0.034) |
| B Harmonizer loaded | every ON chunk's renderer log: `enable_harmonizer: true` and the checkpoint loaded; the OFF check without it | **pass**: 10 / 10 ON chunks; the OFF chunk is not enabled |
| C pose alignment | OFF re-render of 6 selected scenes equals stored on1_PAIH bit for bit | **pass**: 957 / 957 rows, tokens, ego, labels, `served`, teacher all identical (before the lane: on1_HARMOFFA vs on1_PAIH 6 223 / 6 223 identical) |
| D token distance | newest pair relative L2 median in [0.22, 0.36], p90 <= 0.5, p10 >= 0.1, no row ON = OFF | **pass**: median 0.273 / 0.278 / 0.267 (p10 0.21 / 0.22 / 0.20, p90 0.36 / 0.35 / 0.36), 0 identical rows; 1 - cos 0.037; by stratum 0.22-0.29 (junction turns lowest, 0.22-0.26); decision 435: 0.29 |

## Trainer switch

| Check | Result |
|---|---|
| Off vs pre-change code (R6DAG-L18-s0's argv, 60 steps eager, commit 287194b8 in a worktree vs HEAD without `--twin`) | **bit-identical**: 274 scalars, every weight, dev ADE 0.53824, on_hash_60 f5368348b47e2019 ([identity.json](identity.json)) |
| `--twin PAIHTW:0` vs no switch (store loaded, p = 0) | **bit-identical** ([identity_p0.json](identity_p0.json)) |
| `--twin PAIHTW:0.5`, 60 steps | 688 drawn rows with a twin (expected 668 = 60 x 11.1 a batch), 361 swapped, **share 0.525** ([smoke.json](smoke.json)) |
| `--twin PAIHTW:1.0`, 60 steps | 688 / 688 swapped (share 1.0) |

Under R6DAG's draw weights (`round6/dag/w/R6DAG.npz`), rows with a twin carry 0.155 of the pool draw mass, about 11.1 of the 72 pool rows a
batch. At p the batch therefore has about 11.1 p ON rows out of 136. Not built: the paired consistency term (it needs a second forward inside the
loss path) and DAgger-row twins (no exact replay of student states; see the plan).

Cost: 10 ON chunks + 1 OFF chunk of renders (about 3 h wall at PRIO 24 beside the joint arm), plus cache / pool / stretch / gates and 7 x 60-step
runs. Disk kept about 28 GB. 32 GB of rows, caches and stretch parts were deleted
([cleanup log](../../../cl_infra/cleanup/20261022_hmix.md)).
