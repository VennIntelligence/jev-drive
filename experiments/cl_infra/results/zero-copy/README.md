# zero-copy: merged on-policy pools without copying the tokens (2026-10-20)

`rf_pait.py merge` copied every source pool's `@warp/front.npy` into a new file. Now it writes `@warp/front.parts.json`
by default ([`jevdrive/tokens.py`](../../../../jevdrive/tokens.py), [docs/lib.md](../../../../docs/lib.md) `jevdrive.tokens`):
the row ranges of the source files, opened as one read-only array (`CatMM`). `--copy` gives the old layout. The small per-row
files, the feat file and `teacher.npz` are still written as files.
Check lane: [`zc_merge.sh`](../../scripts/zc_merge.sh) + [`zc_check.py`](../../scripts/zc_check.py) (pool jobs, priority 1;
the test pool `on1_JTPHZ` and the two 60-step runs were removed afterwards).

## Verification: on1_JTPHZ (zero-copy) against on1_JTPH (copied), the same 8 sources in the same order

Files: [merge.json](merge.json) (rf_pait output), [du.txt](du.txt), [rows.json](rows.json), [pinned.json](pinned.json),
[identity.json](identity.json).

| check | result |
|:--|:--|
| `r2.py fix` (as ROUND2 ran it; its front sample reads through `open_front`) | prefix identical to on1_JT, no mismatch |
| tab / agent (after fix) / road / drivable / route / feat / teacher: every key, value, dtype and shape | equal |
| every token row (528 547 rows, 517 chunks of 1 024): `np.load` of on1_JTPH vs `open_front(on1_JTPHZ)` | 0 differing chunks (239 s, 8 processes) |
| 72 random batches of 96 rows (uniform, sorted, repeated, straddling all 7 part boundaries): CatMM, slot 7, `pp_train.open_front`, `Tokens` host pageable, `RowCache` (CPU) | all bitwise equal |
| the same on the GPU: `Tokens` host mode with the pinned gather (`--prefetch` > 1), on-card `RowCache` | all bitwise equal |
| R2J-L18-s0's command, 60 steps, eager, seed 0, `--on on1_JTPH` vs `--on on1_JTPHZ` (`df.py bitcmp`) | identical: 278 scalars max diff 0, weights max diff 0, dev ADE 0.54585 both, on_hash_60 43bfc742366790f8 both |

Cost of one merge (the same 8 sources, 528 547 rows):

| | copied (ROUND2's `r2-merge`, file times) | zero-copy (`zc-merge`) |
|:--|--:|--:|
| merge wall time | 562 s (of which about 420 s writing `front.npy`) | 84.6 s |
| bytes written | 173.2 GB (`front.npy` 138.6, road 13.1, drivable 13.2, agent 3.3, teacher 3.3, tab 0.8, route 0.5, feat 0.4) | 34.7 GB (`front.parts.json` 1.2 kB) |

So each merge saves the token copy: about 139 GB of disk for a JTPH-sized pool (262 kB per row) and about 8 min of
single-threaded CPU and disk writing. What it still writes is mostly road + drivable (26 GB of `sdf` rasters) and the
teacher / agent arrays. These could be indexed the same way later, but their readers `np.load` the whole npz, so they
were left as files.

## Readers

Every non-archive reader of `<pool>@warp/front.npy` that can be given an `on1_*` pool now opens it with `open_front`. For an
existing `front.npy` that is `np.load(f, mmap_mode="r")`, so the code paths of running jobs do the same thing as before.
- Training: `pp_train.open_front` / `Tokens` (pageable gather, pinned gather through `CatMM.take`, GPU-resident load) /
  `RowCache`, so `on_train.py`, `rm_train.py` and `lib/retain_train.py` work. The `on_train.sh` / `on_train_y.sh` pool
  guard accepts `front.parts.json`.
- Pool tools and gates: `rf_pait.py` (pack G-r1, merge `--copy` sources, r1), `r2.py fix`, `df.py`, `si.py`, `wt.py`,
  `pr.py`, `stoppt.py`, `rf_{apppool,appgate,cfpool,cf,gates,v2gate,v2cpool,redpool,paionp,selfpool,loss_audit,loninv}.py`,
  `rf_recov_roll.py` report, `ol_pool_merge.py`, `cp_train.py`, `rp_real.py`, `head1_train.py`, `je_slots.py`,
  `vt_r.py`, `vt_px.py`, `es_bank.py`, `path_req.py`, `margin_critic.py`, `rep.py`, `opb_feats.py`.
- Left as they were (they read only fixed non-merged caches such as `lb_navtest`, navtrain shards, `idx_*`, `b2d_*`, or named
  rollout pools): `turn_sel*`, `nc_slow`, `turn_dewater`, `sh30_check`, `pai_keywarp_check`, `m1_replay`, `ap2_*`,
  `serving_trace_reader_check`, `pp_full_check`, `b2d_validate`, `leadstart_probe`, `slp_probe`, `contact_head`,
  `lc_train`, `rp.py` (TRAIN_SETS), `rf_stack_c2.py` (`on1_L18RSG`), `je0.py`, `rf_tts.py`. Shell `-f .../front.npy`
  guards in `ol_chain.sh`, `rf_recov_roll.sh`, `rf_round2.sh`, `pn.sh`, `bn.sh` check rollout pools, which are never
  merges. A plain `np.load` of a zero-copy pool raises FileNotFoundError; it never returns other rows.

## Dependency rule

A zero-copy pool needs its sources unchanged: each part's size and mtime are recorded and checked at open. Before deleting
any `on1_*@warp`, run `python -m jevdrive.tokens refs $DATA_DIR/runs/op_parity/cache` ([docs/storage.md](../../../../docs/storage.md)).

## Copied merged pools that a zero-copy index could replace

Swapped 2026-10-21 (user-approved, all five except L18RB0J; 316.2 GB freed, every row verified bitwise first): [swap_20261021.md](swap_20261021.md).

| pool | rows | `front.npy` | sources (all present, row counts match) |
|:--|--:|--:|:--|
| on1_JTPH | 528 547 | 138.6 GB | L18RB0, L18RTB1, L18YS, L18Y, NR1TV, PAIH, PAIH05, PAIH15 |
| on1_JTCP | 253 648 | 66.5 GB | JT's 5 + CPAI570 |
| on1_JTX | 211 392 | 55.4 GB | JT's 5 + PN0..PN5 |
| on1_JT | 169 606 | 44.5 GB | L18RB0, L18RTB1, L18YS, L18Y, NR1TV |
| on1_PPM160 | 43 063 | 11.3 GB | L18RB0, PPAI160 |
| on1_L18RB0J | 29 605 | 7.8 GB | not replaceable: its `cf` rows came from `on1_CF`, which is gone |

Total that could be replaced: 316.3 GB. To replace one: once no job reads it, check it with `zc_check.py rows`
(build the zero-copy twin under a new name and compare), then delete its `front.npy` and write the manifest
(`link_front`) in its place. Jobs that still have the old file open keep its inode, so the space comes back only when they
exit. Each pool needs the user's approval first.
