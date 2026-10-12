# Keep set (comma1M, native camera) and its Factory-Cinque teacher cache

Built by `scripts/je_keep.py` (executor 2). Run dir `$DATA_DIR/runs/junction_eyes/keep/` (12 GB). Layout follows `lib/je_data.KeepStore`.

## Deviations from plan 3.2 / 4.3 (recorded in pre-registration A, amendment 3)
- Only **126 of the 600 segment dirs** have both cameras at 1928x1208 (census: `{(1928x1208, no ecam): 334, (1164x874, no ecam): 73, (1344x760, no ecam): 67, (1928x1208, ecam): 126}`);
  the wide camera is a model input, so the other 474 cannot make a model frame pair and are not used.
- Events = the frozen op_fov scan (`experiments/op_fov/results/events.json`: 42 turns, 75 straights). After the hold-out, 24 turn events and
  39 straight windows remain; straight windows are drawn (numpy `default_rng(0)`) until their rows match the turn rows.
- **13 431 rows** (6 631 turn + 6 800 straight) on **20 391 frames**, not about 20 k rows. Rows = every 20 Hz frame from onset - 4 s to
  onset + 10 s (turns) or the 10 s straight window; every window also carries 6 s of warm-up frames before its first row.

## Hold-out (E5, never read by a training job)
60 of the 126 usable segments: 12 turn-bearing + 48 others, `numpy default_rng(0)`. Splits `comma1m/junction-eyes-heldout@v1` (60) and
`comma1m/junction-eyes-keep@v1` (66 source segments); selection, windows and rows in `keep_selection.json`. Held-out segments are **not in the
store** (the `held` mask of rows.npz is all False). Held-out segments:

- `001774ef60cf6c43657cd317035fae58` (turn-bearing)
- `0045b4fe97931624e169d500d6009784` (turn-bearing)
- `004ab781e7af5f89bcdcdd056adfa822`
- `00cc336c4bd634b927105c47bfb856ee`
- `01596da3bb1a94d58fa809fc434a7c4a` (turn-bearing)
- `016adcfb8ffcd390b54edcc59d80a615`
- `032d32c6d40eda1fa5a7980723dd17d5`
- `0344df1713f4031803321b9d608d7a51`
- `03c3757f950a7026e2625cb2a10a5f60`
- `0538b6f37f3a7e31150ff5a4c1607c11` (turn-bearing)
- `057dd840f3d18abbe297c6cdd978c77f`
- `060ab59cd773b7483a8f4a01d166ad74` (turn-bearing)
- `061c581ed96a43f36fc130075bf5e434` (turn-bearing)
- `063dbf20fc63d1fa99526e12a75ec248`
- `08428c1342e8e3ac1c01cdef4116591b` (turn-bearing)
- `0aedbb1bc44cbd6db55e0d6e769f5ed8`
- `0b9a7a45d817fcad06924d0c9e3b417e` (turn-bearing)
- `0c68e6844d1c98a9b5b63dcbe83a2440`
- `0d36c43158d2679619588598a1aaece8`
- `0d578680e60a28f29a80e8e0c94cbeff`
- `0e6027003cb6d0d9cb8157780703e4ae`
- `0f4a47fd20b3949831eb615a57d27460`
- `0f88d742ee06d296d58b4ecd90277b68` (turn-bearing)
- `0f99e7a215237a3edd70d24a88f012b5`
- `0fd58429aefa55a0ceea07209dd1a8c5`
- `1034014d1efd115cca7b6c4ee879fd46`
- `1101381e695bae4222e94f28715031af`
- `11181bd8916a29074831e1ff0c749ffb` (turn-bearing)
- `1196526f7f98386c1837c46032a3cf1e`
- `11fccc016a06cd6fb3d3274a49e2d6c8`
- `1275419274af5b2251ae5a214d276b76`
- `130b834e6741239a732c8834d748e30b`
- `135dbf22b2ede7f75110ca094ca6887b`
- `14aa3e8c82a2449fcb28b10931a43572`
- `15b7720f7ea2d39e2c955818be35ece4`
- `165cb09ab5539683f074cbfb11945c79`
- `16a6e9e7867243a7c28440121e31ce54`
- `17b97f197b313b9ec9563876f20fda0c`
- `182c92057a5210d1c1599d4996050418`
- `184cb38c4dfc44424066c0042b3f76ff`
- `1889e51db9cebbc08d1eaa5dc20f0c47`
- `194f287a210311c93bcd23f60e0e7105`
- `19818063e68797f897cc102652b871e2`
- `1991767a29543439e2f0935326ff6b98` (turn-bearing)
- `19d3e9f18d0a150946c7fd7ee95cbb9a`
- `19dd6dff22e8fe7c2f557b572cf48461`
- `1adc63bb3bc21374be79c9d989a263d3`
- `1b685b24a7468e637298d4db1e6705df`
- `1c3640dad62dea0e77b25d697cb0cb96`
- `1d159c2c857e8ca373dacadca46576f9` (turn-bearing)
- `1de0144b44d1d3ff78ebd4667ece14aa`
- `1dec8c6f2b3dc2fa692a8ce62ab5307d`
- `1f16198f8c134dada1f51191d18fd443`
- `1fb96ca4942226c6b5114730200fcc73`
- `2050513ce84db4a9def3959fff92355d`
- `208a283cf4ca211af00602b1902340e4`
- `210f60d9b92edca646d1aa8789a47bcf`
- `21fbdfaea8489f528b65ce18c3547ddc`
- `228098f9ab1d1fb63385a7199c35961e`
- `2292c6ba675bfce21f7ac72b13b543ea`

## Files in `$DATA_DIR/runs/junction_eyes/keep/`
| file | content |
|:--|:--|
| `frames.npy` | (20391, 2, 6, 128, 256) uint8, 8.0 GB. Every 20 Hz model frame [road, wide] of every window (road f 910, wide f 455 from the segment's own calibration; `frames.segment_model_frames`, streamed). Windows are contiguous: window w covers `off .. off + n` (`keep_selection.json`) |
| `rows.npz` | `fidx` (13431, 10) int64: frames t0 - 36 .. t0 in steps of 4 (0.2 s), so slot j pair = (fidx[j], fidx[j+1]) and the t0 pair = (fidx[8], fidx[9]); `teacher` (13431, 18452) f32 = factory raw `outputs` at t0; `seg`, `held` (all False), `action_t` (0.275, 0.525); extras `win`, `vframe` (frame index in the segment video), `turn` (bool), `temporal` (select_4, 512), `vision` (mean, 512) |
| `teacher_raw.npy` (20391, 18452) f32, `teacher_temporal.npy`, `teacher_vision.npy` (20391, 512) f32 | the same teacher for every stored frame, aligned with `frames.npy`; `teacher_slices.json` = head name -> [start, stop] in the output vector (plan, lane_lines, road_edges, lead, pose, desire_state, meta, ...) |
| `selection.json`, `census.json` | windows (kind, seg, start, onset, end, hi, off, n, row0, nrows), counts |

Teacher: `op_adapt.load("cinque")` (fp32 port) + `op_adapt.Stepper`, desire 0, traffic (1, 0), action_t (0.275, 0.525), stepped at 20 Hz through
**every** frame of each window from a zero state (real history, 120-frame warm-up before the first row). 58 windows, 3 pool jobs, about 7 s per window.
