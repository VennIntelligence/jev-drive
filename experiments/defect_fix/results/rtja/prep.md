# RTJA preparation (NR1 prereg supplement 12, (a) - (c))

Chain: `bash experiments/defect_fix/scripts/jt.sh rtja-prep` (code 870d6eb5), 2026-10-18 20:14 box time; ids in `$DATA_DIR/runs/defect_fix/jt/{jobs,stages}.txt`.

- **(a) anchored features: pass.** `rm_build.py anchor` (PYTHONHASHSEED 0) for navtrain 12 shards + yr1 rows, idx_waitcap / idx_selfiter
  -> `on1_WAIT` / `on1_NR1TV` (`--cat`), pools L18RB0 / L18RTB1 / L18YS / L18Y (`--pool`, 8 workers each) and the merge
  `on1_JT=on1_L18RB0:rb0,on1_L18RTB1:rtb1,on1_L18YS:ys,on1_L18Y:y,on1_NR1TV:tv`. rm_train reads `feat/<d>.a.npz` for every data dir of the
  store (navtrain, yr1, the `--on` pool) and `feat/on1_WAIT.a.npz` for the retention pool; `rtja.py acheck` on all of them
  ([acheck.json](acheck.json)): names and `ok` equal the projected files row by row, width 203, finite. `on1_JT.a`: 169 606 rows, tokens equal
  `feat/on1_JT.npz` and the cache tab row by row, ok share 0.992.
- **(b) smoke RTJASM-L18-s0 (kd-lam 1, 300 steps): pass** ([smoke.json](smoke.json)). No non-finite scalar; kd 0.0066, d2 0.076; act
  pool agent 0.070 / road 0.149 / onc 0.0048 / soft 0.0097 / side 0.020; sidecar head `order: true, anchor: true` (RTB1AR-F-s0 converted
  by `rmem.order_state`). Speed: 2.57 steps/s averaged over the 300 steps including compile; 0.20 s/step steady (steps 100 - 300).
- **(c) chunked wait-set read: pass.** `wait_set.sh ccheck RSG-L18-s0`: chunks a_c0 (37 windows) and b_c0 (42) read as separate lists,
  79 / 79 windows identical to the stored whole read in score, zero, wait_ok, launch_s, fail ([wait_chunkcheck_RSG.json](wait_chunkcheck_RSG.json)).
  The arms read the set with `wait_set.sh readc` (4 chunks per variant).

No fallback: the arms train on the anchored layout.
