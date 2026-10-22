# Cleanup 2026-10-22: nuplan_dag lane's own intermediates

The lane is concluded (decision 445; plan experiments/nuplan_dag/plans/2026-10-22-nuplan-dag.md). Only this lane's own files were deleted.
The main session gave the instruction (the user's standing rule: delete what is certainly deletable). GB = 10^9 bytes; 32 GB freed in total (df before / after).

| when (box time) | what | size | why safe |
|:--|:--|--:|:--|
| 10-22 12:15 | `op_parity/cache/on1_JTXD`, `on1_JTXD_add` (+ `@warp`) | 14 GB | the NDAG training pool and its joined add part; both trainings and the 48 reads are done. `jevdrive.tokens refs` shows that no other manifest indexes them. No queued or running job names them |
| 10-22 12:15 | `op_parity/cache/on1_NDP`, `on1_ND001` .. `on1_ND024` (+ `@warp`) | 7 GB | the chunk pools. Their `front.npy` files were indexed only by `on1_JTXD@warp` / `on1_JTXD_add@warp` (refs output), which were deleted in the same step |
| 10-22 12:15 | `route_mem/feat/on1_{NDP,ND0xx,JTXD,JTXD_add}{,.a}.npz` (54 files) | 0.3 GB | route features of the pools above only |
| 10-22 12:15 | `onpol_nuplan/rows/{NDP,ND001..ND024}` | 0.1 GB | row meta of the pools above. The only links found under `find -type l` / `realpath` point from these dirs into the deleted pools; no link from elsewhere resolves into them |
| 10-22 12:15 | `runs/nuplan_dag/{roll,teach,w}` | 9.3 GB | B2 rollouts (dumps already stripped), teacher label files and the NDAG / NDAG0 draw files. All are consumed by the deleted pools / done trainings |

Kept:
- the checkpoints `NDAG-L18-s0` / `s1` (`op_parity/runs`, `route_mem/train-NDAG-*`);
- the read outputs (arena `defect_fix/jt/arena/NDAG-*@off`, held 460 `route_mem/read/NDAG-*`, and the Part 1 reads of R8K-L18-s2 / R6DAG-L18-s1);
- `runs/nuplan_dag/{res,p1,p2,pool,lists,stage}` (summaries, extracts, job logs; 70 MB).
