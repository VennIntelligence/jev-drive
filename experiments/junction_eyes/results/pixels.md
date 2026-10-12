# Pixels (executor 2): caches, checks, keep set, slot pairs

Code: `experiments/vis_train/scripts/vt_px.py` (new `--t0` / `--root`; the full protocol and outputs unchanged), `scripts/je_keep.py`, `scripts/je_slots.py`.
Box root `$DATA_DIR/runs/junction_eyes/`; disk left 872 GB (was 988 GB at the start).

## 1. t0 image pairs (J_nav, navtest)
`px/<data>/frames-t0.npy`, (N, 2, 2, 6, 128, 256) uint8, [prev, cur] x [road, wide], 786 432 bytes per row, rows in the `tab.npz` order of
`$DATA_DIR/runs/op_parity/cache/<data>`. Rendering decodes only the key frame(s) the t0 pair needs (for the t0 pair: the t0 key) and warps
only the needed time, so it is byte-identical to frames 6:8 of the full 8-frame row (see check). Render speed: navtest 12 146 rows in 16 s on 70 cores.

| data | rows | size |
|:--|--:|--:|
| `lb_navtest` | 12 146 | 9.6 GB |
| `navtrain_full.s{0..11}of12` | 103 288 (8 607 each) | 81 GB (all of navtrain) |

**All navtrain tokens were rendered**, not only a J_nav subset: the straight filler pool (25.9 k tokens with jdist > 60 m and turn40 < 10 deg) is
smaller than the 77.4 k candidates, so J_nav is nearly all of navtrain. Executor 1 selects rows by the row index of `labels/rp_real` /
`step0/rows/J_nav.parquet` (`tok_file` shard + `tok_row`), the same order.

`vt_px.py check --t0` (frozen encoder on the stored pairs against `front.npy[:, 7]` of `<data>@warp`; `--pixels n` re-renders n rows through the
original full path and compares the t0 bytes):

| data | rows checked | mean abs diff | max abs diff | front rms | full-path pixel rows differing |
|:--|--:|--:|--:|--:|--:|
| lb_navtest | 512 | 2.7e-4 | 0.031 | 1.71 | 0 of 48 |
| each of 12 navtrain shards | 256 | 0.0 (bit-equal) | 0.0 | 1.68-1.69 | 0 of 16 |

Raw: `results/pixels/check-t0-*.json`.

## 2. Keep set and teacher cache
See [keep_set.md](keep_set.md). 13 431 rows / 20 391 frames / 60 segments held out; teacher cache of all heads (18 452 outputs) per frame.

## 3. Slot pairs of AlpaSim decisions
Layout (N, 2, 2, 6, 128, 256) = the t0 encoding pair (slot 6, slot 7) (`on_rows.make_front`: prev of slot 7 = slot 6). Files in `slots/`.

| set | status | file | check |
|:--|:--|:--|:--|
| held-452 (8 129 rows, `held452.parquet` order) | done, 6.4 GB | `slots/held452_pairs.npy` | taken straight from the dumps' `cur`; 500 rows through `pp_prep.encoder` vs the cached slot-7 token of `road_probe/tok/held`: mean abs diff 0.0, max 0.0 |
| zero-98 (457 rows, `zero98.parquet` order) | done, 0.36 GB | `slots/zero98_pairs.npy` (+ `.ok.npy`) | stored driver messages replayed through `sh30_driver.Driver` (exact path), pair captured from `Core.plan`; vs `road_probe/tok/zero` slot-7 tokens: mean abs 8.8e-4, max 0.031 (fp16 noise), 457 of 457 |
| J_on (33 899 rows, `J_on.parquet` rid order) | done, **reconstructed slot pairs**, 26.7 GB, all rows ok | `slots/jon_pairs.npy` + `slots/jon_pairs.ok.npy` | validation below; accepted by the coordinator |

### J_on: the dumps have no frames
The onpol_long roll dumps had `cur` stripped after tokenizing (`experiments/onpol_long/scripts/ol_chain.sh strip`); only the t0 JPEG, ego, mu, poses
remain. G-dump as written ("npz has all 8 slots") cannot pass. Reconstruction (coordinator decision): rerun the driver's synthesis on the JPEGs of
decisions k-3 .. k (`sh30_core.pack_fast` -> `stack_keys` -> `lattice_gpu`) with the ego history from the dump's `ego` vector (poses exact, newest
velocity exact, older body velocities estimated from the poses: chord, or forward-only), cold rows with fewer keys handled as the driver does.

Validation on 600 random held-452 decisions (cur and JPEG both exist; `results/pixels/validate*.json`):

| quantity | chord velocities | forward-only velocities |
|:--|--:|--:|
| slot 7 pixels: mean abs diff / share of bytes differing | 9.0e-4 / 0.03 % | same |
| slot 6 pixels: mean abs / share differing / max | 0.044 / 3.7 % / 57 | 0.038 / 2.9 % / 81 |
| token (slot 6, slot 7 pair) mean abs diff vs tokens of `cur` | **8.5e-3** | **8.0e-3** |
| token max abs / per-row max p50 / p99 | 4.4 / 0.098 / 1.42 | 5.3 / 0.059 / 2.46 |
| fp16 noise floor of the encoder (same pixels, other batch) mean / max | 8.5e-4 / 0.031 | same |
| token rms | 1.69 | |

Slot 7 is the t0 key and reproduces; slot 6 is a warp along an estimated ego track and differs in 3-4 % of bytes. The token difference is about 10x the fp16 noise floor (0.5 % of the token rms) with a heavy tail, so the reconstruction is **not within fp16
noise**. Decision (coordinator): **accept, chord-velocity variant, no re-roll**. J_on is a training and probe-fit row set only; the three read sets reproduce
(held-452 exact, zero-98 fp16 noise, navtest-16 from the pixel path). All 33 899 J_on rows were reconstructed (`jon_pairs.ok.npy` all True). The G-dump
rule was replaced by the held-452 validation above (cur exists there); `step0/gdump.json` records held-452 pass (exact), zero-98 pass (fp16 noise) and
J_on "reconstructed slot pairs, accepted by coordinator" with these numbers. Any use of the J_on pairs must be worded "reconstructed slot pairs".
