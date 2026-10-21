# Storage

Read this when you need to decide where a model, dataset, checkpoint or output goes on the box.

## Layout

`$DATA_DIR` (= `~/data` = `/root/autodl-tmp/ujs`) is the local data disk. Code reads paths from env, never hard-coded.

| Dir | Holds |
|---|---|
| `cache/` | HF (`$HF_HOME`), torch, pip, uv, modelscope caches |
| `models/` | models that do not come from the HF cache (local copies, converted weights) |
| `datasets/` | extracted datasets we read at training time |
| `processed/` | our own preprocessed data (features, tokens, caches) |
| `ckpt/` | our own checkpoints |
| `runs/` | experiment outputs and logs |

Merged on-policy pools (`runs/op_parity/cache/on1_*@warp` with a `front.parts.json` and no `front.npy`, docs/lib.md
`jevdrive.tokens`) hold no tokens of their own: they index their source pools' `front.npy`. Deleting or rebuilding a source
breaks them (loudly, at open), so before removing any `on1_*@warp` run `python -m jevdrive.tokens refs
$DATA_DIR/runs/op_parity/cache` and keep every file it lists while a pool that uses it is live.

HF models live in the HF cache, so `from_pretrained("Qwen/Qwen3-VL-4B-Instruct")` works without a path.
Add new models to `scripts/download_models.sh`.

Small-VLM first look (2026-10-08, d182): `models/gemma-4-E2B-it` 9.9 GB kept (ModelScope copy, bf16); `Qwen/Qwen3.5-2B` (4.5 GB) and `Qwen/Qwen3.5-4B` (9.2 GB) were
downloaded the same way and deleted 2026-10-08 after the first look (re-download with `small_vlm_pdl.py`, ~10 min each); `envs/svlm-extra` is a `--target` dir with flash-linear-attention / fla-core / einops
that the Qwen3.5 runs put on `PYTHONPATH` (the shared `.venv` is untouched). hf-mirror gave ~1 MB/s in total, ModelScope caps
one connection at ~0.6 MB/s but scales with connections: `experiments/vlm_arb/scripts/small_vlm_pdl.py` (16 MB ranges, 64
streams) reached 14-15 MB/s.

## Tiers

- Re-downloadable (HF models, public datasets): not precious. The repo keeps the ids and download scripts.
- Ours (`processed/`, `ckpt/` worth keeping): precious. Once we have any, turn on AutoDL file storage
  (`/root/autodl-fs`, shared by all instances in the same region) and keep an offsite backup too.
- Hot data for training: the local data disk. Grow it when training starts, shrink it after.

## Public data

`/autodl-pub/data` (read-only; West-B and West-D both checked) already has nuScenes (full, 549G of `.tgz`), KITTI,
SemanticKITTI, cityscapes and more. They are archives: extract what you need into `datasets/`.
`scripts/extract_nuscenes.sh mini` or `trainval [member ...]` -> `datasets/nuscenes`
(default member `samples/CAM_FRONT`, ~6 GB; the full trainval is 300 GB+).
Do not download these again.

## Region

The box is in West-E since 2026-09-28 (West-B until 2026-09-20, then West-D). autodl-fs only mounts inside one region,
so rent further boxes in West-E too. autodl-fs is not turned on (`/root/autodl-fs` does not exist; checked 2026-09-28).
`/autodl-pub/data` in West-E carries the same datasets (nuScenes, KITTI, ...).

Last verified: 2026-09-20

## Cleanup candidates (scan 2026-10-08)

Disk 5.0T, 4.5T used, 529G free after the 2026-10-08 cleanup (user-approved, script and log in `tmp/cleanup_1008.*` on
the data disk), which removed about 339G and kept small samples:

| Removed | Kept |
|---|---|
| `runs/cosmos_full/{pairs,gen,clips,stage10}` (cosmos no-go, d56/d63) | `pairs_sample/` (20 pair dirs), `stage1/`, logs |
| `runs/cosmos/{out,clips,gen}` | `anchor/`, `gen-cabincam/`, small dirs |
| `models/cosmos` (Cosmos-Reason1-7B, siglip2 naflex, Predict2.5), `envs/cosmos-transfer` | nothing; `vlm_arb_models.py`'s `cosmos-reason1-7b` arm needs a re-download |
| `datasets/nuscenes/sweeps` camera and lidar dirs (re-extract from `/autodl-pub`) | `CAM_BACK` (0.3G), radar |
| `runs/factor_wm/clips/{g0a,g0b,g1s,train}`, `runs/factor_wm/p2op/{logged,roll}` (line dropped, d177) | json indexes, reports, `onnx/`, `runs/` |

2026-10-09 cleanup (user-approved, script and log in `tmp/cleanup_1009.*` on the data disk): 473G -> 892G free.
Removed `runs/openpilot_rigs/frames` (138G, model_smoke) and the per-episode `scene.ply` / `ground.ply` dumps under
`runs/{bench/hugsim,op_parity/hugsim,op_adapt_H/hugsim,op_guard,hugsim-derot}` (11058 files, ~300G; scores, logs and
videos untouched). `runs/hugsim-exam` keeps its ply (35G): `experiments/hugsim/scripts/` spin attribution and
`export_review_cases.py` read them. HUGSIM writes two ply per episode (~27 MB each), so new bench runs refill this.

2026-10-10 cleanup (user-approved, scripts and log in `tmp/cleanup_1010*` on the data disk): 268G -> 827G free.
CARLA / Bench2Drive bulk data of the closed CARLA lines (374G): `runs/b2d_collect/data`,
`runs/op_route_cmd/{carla_pairs_s10000,carla_pairs_s10000ol,carla_pairs_s2000,carla_render_s200}`,
`runs/op_route_ft/{bank,carla_near,synth,gif,plan_track_gif}`, `runs/nq4/{k,gk,cx}`, `runs/b2d_privileged_ceiling/arms`,
`processed/carla_*`, `datasets/bench2drive-mini`. Kept: `runs/op_route_ft/{runs,onnx}` (trained weights, 15G), small
results and logs, `third_party/carla` (30G), the leaderboard-model envs, `runs/op_parity/cache/b2d_v2` (24G).
Full rollout logs of concluded AlpaSim runs (185G): `rollout.asl` under `runs/alpasim/{pai2,c0,cf1,fix1,ot3,c0c,ot2,m1,c0b}`,
one `.asl` per `rollouts` dir kept as a sample (16G); scores, metrics parquet, csv, logs, videos and `aggregate/`
untouched, so these runs can be re-read but no longer replayed. `runs/alpasim/ap2` (120G) is the AP2 token cache, not
rollouts; untouched.

2026-10-11 cleanup (user-approved, script and log in `tmp/cleanup_1011.*` on the data disk; the disk had filled to 280G
free after `runs/vis_train/{px,px_side}` grew to 534G). Removed only material of concluded or paused lines (plan.md section 5)
with no live reader in `jevdrive/`, `lib/` or non-archive topic scripts, about 560G: the CARLA parts of paused lanes
(`runs/vlm_arb/arms`, `runs/op_img_cmd/{carla,ft/bank/skycarla}`, the `*carla*` entries of `runs/op_adapt_H/{fixbank,samples,bank}`,
`processed/op_adapt/p5`, `ckpt/nq4_p3*`, `runs/{jfa,rig122}`); the dropped world-model line (`runs/wl`, `processed/wl*_gen`,
`envs/comma-wm`, `models/comma`); envs and weights of concluded third-party lines (drivestudio, r3d2, p3-wodprep, statepol,
qwen-drive, autovla, drivor, gtrs, sparsedrivev2, gdino, efficientsam3, hugsim-ltf, depth, openjev); the early frozen-feature
caches (`processed/waymo_e2e/features/{qwenvid_train_t4,qwen_front3,qwen_grid_p2}`, `processed/drive_backbones/op_trainval`,
`processed/{elicit_e2,waymo_ds,waymo_ds_veh,night2,hugsim_pairs,hugsim_pairs_10hz}`); concluded runs (`runs/{op_adapt_r2,
infra-accept,scale_check,unified,nq3,waymo_heads,op_cosmos_probe}`); and the per-episode ply of concluded HUGSIM runs and of
`runs/bench/hugsim` (refilled since 10-09). Kept on request: every `datasets/` entry (`waymo_perception`, `womd`); kept by
earlier decisions: `runs/cosmos*` samples, `runs/hugsim-exam`, `runs/hugsim-wajepa` (the ply fallback of
`experiments/lowboard_diag/scripts/lbd_hugsim.py`). Each new HUGSIM bench run writes ply again (~27 MB x 2 per episode).

2026-10-11b cleanup (user-approved, script and log in `tmp/cleanup_1011b.*` on the data disk): 535G -> 1474G free, 939G
removed, all of it concluded-line material or B2D generated caches. The VT lane (decision 248): `runs/vis_train/{px,px_side}`
(546G, pixel caches; `scripts/vt_px.py` / `vt_side.py` rebuild them), `native/{onnx,bias}`, `tokens`; the per-snapshot
checkpoints `runs/op_parity/runs/VT-*-kNN`, `smoke-vt-*`, `UF-U*`, `smoke-U*`, `time-U*` (163G; the final `VT-*-s{0,1}`,
`VTCP2-*` and `VTR-*` stay, so every read in reads.md can be repeated), the arm-R token banks `runs/op_parity/mem/vtr_*` (50G).
TRT engines of concluded models in `runs/op_interp/trt_cache` and `models/openpilot/trt_cache` (85G; the engines of `cinque`,
`SH30-F`, `P2H10-F`, `P2-F`, `shipped` and the `pp-*-chk` stay; any other model rebuilds its engine on first use). B2D generated
caches: `runs/op_parity/cache/b2d_v2*` (24G), `runs/op_adapt_L/b2d`, `processed/op_route_cmd`. B2D infrastructure untouched
(`third_party/carla`, the leaderboard envs, `checkpoints/scout`, `runs/op_route_ft`, `runs/op_wide_ft`, `runs/b2d` results).

2026-10-14 cleanup (user-approved, script and log in `tmp/cleanup_1014.*`, full scan in `tmp/du_1014_d3.txt` on the data
disk; the box came back from a restart with no GPU, disk now 6.4T): 1093G -> 1705G free, 612G removed, all regenerable material
of concluded lines with no live reader. `runs/op_adapt_H/{fixbank,bank,bank2,bank3}` (222G; `runs/`, `onnx/`, `samples/` stay),
`runs/vis_train/px_side` (182G, rebuilt on 10-11 by eyes_sense / junction_eyes right after cleanup 1011b; `vt_side.py` rebuilds
it), `runs/junction_eyes/{px,px90,slots,keep,keep_w90,step1}` (189G; `step0/` and the repo results stay; slots need a GPU to
rebuild), `runs/op_img_cmd/ft` (50G), `runs/op_adapt_L/{t,gate_curve,readout,followup,pilot}` (38G; `runs/` stays), the ply
of `runs/bench/hugsim` (35G, 1328 files), `rollout.asl` of `runs/lowboard_diag/pai2` (25G), `runs/op_parity/mem/vtr_W0` (11G),
`models/openpilot/taps` (9G), HF copies of diffusiongemma / Qwen3-VL-8B / InternVL2 / Wan2.1 / comma world model / R3D2 /
Cosmos-Reason2 / Qwen3.5 (about 45G) and the top-level scratch files. Tier 2 of that scan (ask first, about 420G):
`processed/op_adapt/{navtrain,wodtrain,nusc,wodval}` 198G, the onpol_long old-arm warp caches `runs/op_parity/cache/on1_L18{R2,R2s,R3,YA,Y,P}*`
68G, `runs/op_adapt_H/samples` 48G, `runs/op_lb` side caches 45G, `runs/hugsim-exam` 41G, `runs/road_probe/roll` 18G,
`datasets/nuscenes` + `processed/nuscenes` 58G, `processed/wod_zeroshot` 22G, `datasets/comma1M` 21G, HF Alpamayo-1.5-10B 21G,
`envs/{ultralytics,sam3}` + `models/sam3.1` 16G, `runs/op_parity/{s2_thinhead/ft,wa_xboard}` 29G. Kept on purpose: every
`datasets/` entry, `runs/alpasim/ap2/cache` (120G, read by `route_follow/scripts/rf_cf.py`), all live-lane rollouts.

2026-10-16 cleanup (user-asked; both machines; manifests [box](../experiments/cl_infra/cleanup/2026-10-16-box.md), [Tokyo](../experiments/cl_infra/cleanup/2026-10-16-tokyo.md), raw per-file lists `cleanup_2026-10-16.txt` next to each data root): box 1.1T -> 1703G free (651.7 GiB: rollout.asl of closed arms' reads, P4 and closed-arm on-policy rolls with their driver dumps, `runs/junction_eyes/px_side` (the last hard link of the vis_train side cache), leaked nvimgcodec temp dirs in `tmp/`, pilot / gate-smoke checkpoint weights, smoke pools); Tokyo /data 188G -> 1075G free (886.6 GiB of rollout.asl of closed PAI 281 reads and smokes). Reference and candidate reads, the STACK inputs and the RECOVERY-ENVELOPE round (no decision line yet) were kept; the class C lists in the manifests are for the user.

2026-10-18 cleanup (user-asked, class A only; manifest [box](../experiments/cl_infra/cleanup/2026-10-18-box.md), scripts and logs in `tmp/cleanup_1018/` on the data disk; the 10-17 passes are in [2026-10-17-box.md](../experiments/cl_infra/cleanup/2026-10-17-box.md)): box 1099G -> 1262G free (163.2 GiB of rollout.asl: ARENA reads of the closed NR1HV2 / NR1FB10 arms, held / PAI / public reads of closed NR1 / JT / JTB / nuPlan ACC + GAP arms, the read-throughput pilot, the jtn-diag camera reruns); 49.4 GiB of JT / JTB arena asl follow in a pool job chained after the running JTN-L18-s1 reads. ~960G had landed in 31 h, 517G of it ARENA reads; the anchor reads (313G) are kept, the class C list (about 160G, mostly pools and NR1F / NR1HV reads) is for the user.

2026-10-19 cleanup (user-asked, class A only, to fetch PAI curated-train scenes; manifest [box](../experiments/cl_infra/cleanup/2026-10-19-box.md), scripts and logs in `tmp/cleanup_1019/` on the data disk): box 416G -> 640G free (223.7 GiB, 8 200 rollout.asl, one sample per `rollouts` dir kept: `defect_fix/jt/arena` reads of the closed JT00 / JT10 / JTB / NTVX / NTVXS / JTXS / BN* / FREEM1K / STOPPT* arms 105 GiB (incl. the 49 GiB the 10-18 chained job left), ARENA reads NR1HV-L18-s0 / NR1F-L18-s0 74 GiB, held reads of the same closed arms in `route_mem/read` 18 GiB, the WARP-LOOP static re-dump asl `defect_fix/warp_loop/sdump` 27 GiB). Kept: NTV0, JTN, every STAND* read (B2 candidate STANDFMXH), JTX1K / JTNS1K controls, P0, RB0-F, the RSG / RTB reads that `rm_align.py` / `nx_*.py` open; class C (about 125G: sdump npz 21G, `warp_loop/arena` 61G, the 10-17 reference reads, RTJA / JTNC1K / JTNN1K / NR1 side arms) is for the user.

2026-10-20 cleanup (user-asked, second round to fetch 200 PAI curated-train scenes; manifest [box](../experiments/cl_infra/cleanup/2026-10-20-box.md), scripts and logs in `tmp/cleanup_1020/` on the data disk): box 1076G -> 1583G free (486.8 GiB, 12 173 rollout.asl from a full `runs/` sweep, one sample per `rollouts` dir kept; the 640G the main session saw had already grown to 1076G from the parallel WOD-E2E slim): WARP-LOOP `warp_loop/arena` 59 GiB, PAI-INPUT-LOOP 57, RECOVERY-ENVELOPE PAI read 28, the REDLINE / NATIVE-GAP / LEADSTART / LAMP dumps' asl 68, ALPASIM-LONG family and held reads 39, `wait_cap/next` + `train` 32, ONPOL-LONG rolls and reads 32, BEST-OF-N and PRACTICE-NUPLAN rolls 32, `cl_infra/compile` + `pack_check` 24, the closed `pai_accgap` arms 18, the round-1 class C `route_mem/read` side arms 28, and smaller closed reads. `pai_pcs_local.py` reads only summaries, so its sources lost nothing. Kept: the PAI candidate / JTN / RB0-F / anchor reads, `acc_gap/read/s2` (private anchor), reads that `ev.py` / `rm_align.py` / `nx_*.py` / `bd1_diag.py` open by path; class C (the four ARENA anchor reads APY / LTF / DD / YR 254 GB, plus 28 GiB of live-README or open-question reads and the 20G `warp_loop/sdump` npz) is for the user.

2026-10-20 Tokyo cleanup (user-approved per item; manifest [Tokyo](../experiments/cl_infra/cleanup/2026-10-20-tokyo.md)): /data 1.8T -> 2.0T free, / 1.2T -> 1.3T free; 10-16 image-smoke asl 54 GiB, `runs/op_parity/cache` training pools 82.6 GiB, `runs/b2d/controller-eval/v2` 23.2 GiB, Docker build cache 114 GB. `~/.cache/uv` kept. The 10-16 class B / C `route_night` / `steer_long` / `onpol_pai` dirs had already been removed on 10-16 afternoon (unrecorded until now).

2026-10-20 WOD-E2E slim (user-approved for PAI scene fetching, withdrawn by the user after it had already finished; [docs/waymo-e2e.md](waymo-e2e.md) "Where it lives"): `datasets/waymo_e2e/front3` (FRONT, FRONT_LEFT, FRONT_RIGHT, 725 GB) rewritten to `front1` (FRONT only, 235 GB) by `scripts/download_waymo_e2e.sh reslim`, 622 shards in 5.7 min with 24 workers, no failures; box 640G -> 1.4T free. Index rebuilt, `processed/wod_zeroshot/sets.json` spans remapped; side-camera (openpilot road / wide) WOD readers no longer run, re-fetch from the bucket to restore them.

2026-10-20b cleanup (user-asked, class A only, morning pass for more PAI curated-train NuRec scenes; manifest [box](../experiments/cl_infra/cleanup/2026-10-20b-box.md), scripts and logs in `tmp/cleanup_1020b/` on the data disk): box 571G -> 695G free (124.3 GiB: the PAI-PARTIALS serve-sweep asl, d398, 25 GiB with one sample per `rollouts` dir kept; the closed-line training pools `op_parity/cache/on1_BNX` + `@warp` (d379) 52, `on1_JTP` + `@warp` (JT code-smoke pool, d352) 36, `on1_JTPAI` (STOPPT, d386) 11, none hard-linked elsewhere). The 1583G -> 571G drop since 10-19 23:14 was the NuRec curated-train fetch (`datasets/nurec/all-usdzs` +981 G, up to 218 GB/h until `nurec2` stopped at 07:34) plus the d392 `pai_pcs` reads (+145 G). What asl is left (about 910 GB deletable under the sample rule) is almost all protected anchor / candidate / `pai_pcs` / `pai_g10` reads; class C (the four ARENA anchors 254 GB, `op_lb` 89, navtrain metric caches 59, old HUGSIM outputs 86, closed-line row sources, pools and dumps about 175) is for the user.

2026-10-20c cleanup (user-approved right after 10-20b; same manifest, section "2026-10-20c pass"; `delete2.py` in tmux `jev:cleanup1020c`): box 686G -> 1805G free (1201.2 GB = 1118.7 GiB). rollout.asl, one sample per `rollouts` dir kept, every summary / score / csv / parquet kept, of the four ARENA anchors (254 GB), `pai_pcs` GS / G0014 (139), `pai_g10` (102), the ARENA RSG / starter / G0014 reads (115), `anchors` (34), the candidate / STAND reads (151) and the small class C reads (30); whole `op_lb`, the navtrain metric caches, `bench/ol`, `op_probe`, HF Alpamayo-1.5-10B, `pai_input_loop/msgs`, `lat/L18NR1`, `on1_L18NR1`, `wod_r2` (286); the npz / large files of `warp_loop/sdump`, `road_probe/roll`, `op_adapt_L`, six closed `onpol_long/roll` arms, `route_follow/p4` (89). These reads can be re-scored but no longer replayed or re-rendered. Videos were kept (0.6 GB in scope). Kept: HUGSIM (`hugsim-exam`, `op_parity/hugsim`, `bench/hugsim/_onnx`, `datasets/hugsim`); held for live lanes: `practice_nuplan/roll/JTN-L18-s0`, `on1_PPM160` / `on1_PPAI160`, `native_gap/dump`, the CURATED-PAI REF asl `pai_accgap/NTV0-L18-s0`, `datasets/nurec/_dl` (the restarted NuRec fetch resumes its parts).

2026-10-21 inventory (nothing deleted; [disk_inventory_20261021.md](../experiments/cl_infra/results/disk_inventory_20261021.md)): 558 G free at the scan, 864 G after the user-approved zero-copy swap of five merged pools (316 GB). Candidates ranked for per-item approval: safe 35 GB, probably safe 289, ask 975. No NuRec or dataset duplicates; since the swap, `python -m jevdrive.tokens refs` pins the L18 / NR1TV / PN / PAIH / CPAI570 / PPAI160 source pools.

2026-10-21 cleanups (user-approved S3 pass, then a certain-only pass; manifests [S3 rows](../experiments/cl_infra/cleanup/20261021_s3.md), [S3 roll + certain-only](../experiments/cl_infra/cleanup/20261021b.md), lists in `tmp/cleanup_1021b/` on the data disk): 629 -> 704 G (S3 rows, cache, on1_PAIO / JTPHO), then 182 -> 277 GB free: `round5/s3/roll` 54 GB and the `round5/d3/roll` asl / jsonl 14 GB (both keep every yaml: round6 dag.sh / rmc.py use `A/runs/*/sim/generated-user-config-0.yaml` as templates), the CURATED-PAI material of the closed d402 line (`on1_JTCP`, `on1_CPAI570` + `@warp`, `practice_pai/curated/rows`, 72 GB; CPAI570 is no longer a pinned source) and the PAI-ROWS-N gate copy `on1_PAIHN10` 2.7 GB. The NOT CERTAIN list (REC rows / cache 174 GB, DAgger, JTPHD, JTPHX, diagnostics dumps, reads) is in the manifest.

2026-10-21c (user-asked whole-disk inventory, certain-only deletion, forecast to 10-31; manifest [20261021c](../experiments/cl_infra/cleanup/20261021c.md), scan in `tmp/cleanup_1021c/` on the data disk): 356.5 -> 393.0 GB free (36.7 GB: 18 old-rev git worktrees under `runs/route_follow/` and `tmp/`, the ROUTE v2 arm-c pool `on1_L18RB0J` (d302), `long_contact/{native,priv,read,train}` (d256; `ctx` stays), `op_adapt/train-lam*` (op_adapt_r1, d55)). The DAgger lane removed `round6/dag2/s*/{rows,cache}` (129 GB) the same hour. The USER DECISION table (navsim trainval blobs 471, closed-arm PAI read asl 106, `ap2/cache` 129, HUGSIM outputs, B2D / CARLA 100; `b2dc-train@v2` was already deleted 10-10) and the 10-31 forecast are in the manifest.

Live but regenerable (ask first):
`runs/op_adapt_H/{fixbank,bank,samples,bank2}` non-CARLA parts ~300G (H line, last trained 2026-10-03, no live reader; `runs/`
and `onnx/` stay for the bench `adapt_h` family), `runs/op_img_cmd/ft/{bank,runs}` 48G, `runs/op_adapt_L` 52G (`t/H`, `gate_curve/*.npy`, `readout`),
`runs/op_lb` side caches (`lb_hq_*`, `hq`, `g_*`, `lb_guardneg`, `lb_h1train`, `lb_imgtrain`) 42G, `runs/op_parity/runs/smoke-vt-*`
24G, `processed/op_adapt/nusc` 41G, `datasets/comma1M` 21G, `runs/{vlm_thin,vlm_cmp,opctrl,opctrl_long}` 31G.
Paused CARLA / Bench2Drive material deliberately kept (`third_party/carla`, the leaderboard-model envs,
`runs/op_route_ft`) is about 85G. Also regenerable: `runs/bench/hugsim` ply (30G, refilled by every bench run), `runs/lowboard_diag/pai2`
rollout.asl (27G), the HF copies of diffusiongemma / Qwen3-VL-8B / InternVL2 / unidepth (40G), `envs/{ultralytics,sam3}` with
`models/{sam3.1,ultralytics,lama}` and `models/openpilot/{taps,lebowski*}` (28G, readers all concluded). Re-scan before deleting anything.
