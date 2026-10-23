# PAI probe image R8K-L18-s2 + g10 + ACC + GAP 0.7,2.7: build, gate, audit, submission commands (2026-10-23)

Read this before you push or submit the R8K-L18-s2 PAI probe. Why this probe: decisions 453 / 455. Under the new scorer, R8K-s2 is
locally +0.045 over NTV0 (3-seed recipe mean, paired), but the PCS ruler cannot rank the DAgger family against NTV0, so the board has to
decide. Serve: exactly that of the submitted `p1021a` (NTV0-L18-s0, decision 426): `JEV_DRIVER=pai JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1
JEV_GAP=0.7,2.7`, no holdlead, `JEV_TRACK` unset (v8 gate), steering weight 10 passed at submit. Only the checkpoint changes. The
template for this page is [pai_image_ntv0.md](pai_image_ntv0.md). Small files: [pai_image/audit_r8ks2/](pai_image/audit_r8ks2/)
(`run.sh` = the chain, `stages.txt`, build manifest, layers, test / replicas `result.json`, smoke `compare.md` / `cmp_box.md`, `meta/`
env / config / layer / history dumps of this image and p1021a). Tokyo artefacts: `/data/runs/alpasim/pai_image/audit_r8ks2/`.
Nothing was pushed, tagged to a registry or submitted by an agent.

## Provenance and which code is baked

- Checkpoint: box `runs/op_parity/runs/R8K-L18-s2/` (trained 10-22 05:27, `r8k.sh s2`, round8 amendment 1). `ckpt-final.pt` sha256
  `03b0764983e14fc0cca0d76832d38fad1d95e2db3131294664309ac12fd81748`, `route_mem.json` `3394bd55564d2fbeaf57ddd4adb38d06da23fcb98decc4cf42c5eb8d2988dc9c`
  (init NTV0-L18-s0, pool `on1_JTPHDK`, 9 500 steps; same head layout as NTV0, so no other checkpoint is needed at load). Tokyo already
  held an identical copy (`/data/runs/op_parity/runs/R8K-L18-s2`, same sha256), so nothing was copied. `adapter.pt` is not baked: it is
  written by training and never loaded by the serve (as for NTV0).
- Reads: box `runs/round8/s2/pai281/R8K-L18-s2-ag` (Harmonizer off) and `-agH` (on), `config.txt` `SERVE=JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1
  JEV_GAP=0.7,2.7`, `WIZ=...steering_angle_weight=10`, `CONC=4`, `ALPASIM_FAST=1`, `code=b251b657`. During the chunks (10-22 05:27-07:55 +0800)
  the box repo was at f68d1aa8 (pulled 05:14), which changes no served file against b251b657; the next closure changes landed after 13:25.
- **Baked code: b251b657, the code of the reads, not HEAD.** Same reasoning as the NTV0 page. HEAD (95aaf308) differs from b251b657 in three
  baked files, all behind off-by-default switches but not the code of the read: `sh30_core.py` (`SH30_SLOT`), `rmem.py` (window layout),
  `jevdrive/op_interp.py`. Against p1021a's 0d417ee4, b251b657 changes `pai_driver.py` (`JEV_CMD`), `serve_fix.py` (`JEV_ACC_ALAT / REL / VSET`),
  `sh30_core.py` (`SH30_WARP`), `rmem.py` (`cmd_heading`), `pp_train.py` (zero-copy token pools) and adds `jevdrive/tokens.py`. All switches
  are unset in this image and their defaults reproduce the old serve bit for bit; `get_version` carries no new suffix.
- **Build fix.** Since 381488ba `pp_train.py` imports `jevdrive.tokens`, which was not in `docker/closure.txt`, so any PAI build at b251b657 or
  HEAD would fail `verify.py`. 95aaf308 adds it to the closure (and gives `pai_smoke.sh` a `HARMONIZER=1` switch). The build ran in a
  worktree at b251b657 with only `closure.txt` taken from 95aaf308 (staged, so the label stays `b251b657`); `closure.txt` itself is not in the image.

## Image

| field | value |
|---|---|
| local name | `jev-alpasim:r8k-l18-s2-b251b657-pai-vc1.0-lead-acc-gap0.7_2.7` (Tokyo) |
| image id | `sha256:63adee3b5831334cfaea1107b5fcf057d4cbb0e1b7a8ce578af62646b61dce87` |
| size | 8 110 825 512 B (7.55 GiB), 18 layers |
| built | 2026-10-23 21:16 JST, `cd <worktree b251b657> && env DATA_DIR=/data TRACK=pai TAG=R8K-L18-s2 EXTRA=R8K-L18-s2 ACC=1 GAP=0.7,2.7 OUT=... experiments/alpasim/docker/build.sh` |
| `get_version` | `jev-pai-R8K-L18-s2-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate`, git_hash `b251b657` (the same string in all 6 native chunk logs of the box reads) |
| registry digest | none yet: `docker push` prints it |

**Env vs p1021a** (`drive:p1021a` = `sha256:37183c04...0558`), sorted `Config.Env`:

```
< JEV_TAG=NTV0-L18-s0          > JEV_TAG=R8K-L18-s2       (the checkpoint)
< SH30_GIT_HASH=0d417ee4       > SH30_GIT_HASH=b251b657   (build commit, informational)
```

Everything else is equal: `JEV_DRIVER=pai JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_VHIST= JEV_HOLD= SH30_COMPILE=0`, `JEV_TRACK`
unset; CMD `python /app/serve.py`, no entrypoint, user 10001:10001, workdir /app, port 6789, SIGTERM, no labels.

**Layers vs p1021a.** Layers 1-4 (Debian / Python base) are shared; 5-18 differ. The 6.97 GB package layer came from the build cache of a
10-20 build, not from p1021a's 10-19 build: `pip freeze` is identical and every package file is byte-identical, only the `uv_cache.json` /
`RECORD` install timestamps differ. Real content differences: the checkpoint directory, `tags.json`, the code files above. So the push
uploads about 8 GB (vennai-org's repository holds the 10-17 RSG layers, which share only the base).

## Gate (Tokyo, both RTX 3090s, one chain: [run.sh](pai_image/audit_r8ks2/run.sh))

| check | value | limit | |
|---|--:|--:|---|
| `verify.py` in the build (CPU only) | 3 ROUTE-MEM branches load (R8K-L18-s2, RSG, RBSG2) | | pass |
| no-GPU runtime start | serve line, then `Found no NVIDIA driver` after 5 s, exit 1; p1021a does exactly the same (the track always gives a GPU) | | as p1021a |
| `test.sh` card 0: read-only root, write errors / changed files | 0 / 0, 0 tracebacks | 0 | pass |
| cold start | 18.4 s | runtime waits 240 s | pass |
| driver GPU memory | 1.47 GiB | 16 GiB | pass |
| `/tmp` / `/run` high-water (probe) | 2.8 MiB / 0 | 2 GiB / 64 MiB | pass |
| `drive`, client p50 / p90 / p99 / max | 48 / 55 / 117 / 188 ms, 800 calls | target 100 | as p1021a |
| SIGTERM -> exit | 0.6 s, code 0 | | pass |
| 4 replicas on one 3090 | 1.43 GiB per replica (card 5.8 GiB), cold start max 21.9 s, 1 version, p50 / p90 / max 163 / 170 / 426 ms | 16 GiB | pass |
| smoke-side | GPU peak 1.43 GiB, `/tmp` 33 MiB, 0 tracebacks, 0 inference errors over 19 985 `drive` calls (server p50 56-58 / p90 75-78 / max 221 ms); the driver container cannot reach pypi.org | | pass |

## Smoke (official containerised PAI stack, steering weight 10, image as built, no driver env) vs the box's stored reads

Same chunks as the NTV0 page (identical scene lists; the box reads' results-summary + drive.jsonl copied to Tokyo
`pai_image/native/r8ks2_{off,on}_c00{0,1,7}`). Scores are the archived (old) scorer on both sides. Comparison: `scripts/pai_image_cmp.py`.

| chunk (scenes) | Harmonizer | image mean (zeros) | native mean (zeros) | paired diff [95 % CI] | equal / within 0.01 / within 0.05 | zero agreement | decision-0 plan diff, median / max | zero flips |
|---|---|--:|--:|--:|--:|--:|--:|---|
| c000 (36), card 0 | off | 0.7660 (8) | 0.7662 (8) | -0.0002 [-0.0006, +0.0000] | 33 / 36 / 36 | 36 / 36 | 0.6 mm / 2.10 m | none |
| c001 (36), card 0 | off | 0.8154 (6) | 0.8431 (5) | -0.0277 [-0.0834, +0.0003] | 32 / 35 / 35 | 35 / 36 | 1.8 mm / 16.6 m (bd97de34) | ba62e10d 0 (offroad) vs 1.0 |
| c007 (29), card 0 | off | 0.6865 (9) | 0.7103 (8) | -0.0238 [-0.0714, +0.0001] | 27 / 28 / 28 | 28 / 29 | 1.1 mm / 3.1 cm | 268949d2 0 (at-fault collision) vs 0.69 |
| pooled (101) | off | 0.7608 (23) | 0.7776 (21) | -0.0168 [-0.0434, +0.0000] | | 99 / 101 | | 2, both image lower |
| c007 (29), card 1 | on | stopped at 12 / 29 scenes after 100 min | | | | | | not completed |

Reading. The same picture as p1021a's smoke (pooled -0.029 there). On the first decision both hosts serve the same plan to a median
0.6-1.8 mm, so code, checkpoint and switches agree. The two flips are equal at decision 0 (0.1 mm) and part by more than 0.1 m only at
decisions 36 (ba62e10d) and 76 (268949d2): the closed loop amplifying cross-host numerics (box torch `+cu130` on RTX 6000D, image `+cu126` on
3090). ba62e10d flipped the same way in p1021a's c001 smoke, so it is a property of host x scene. bd97de34 is the known first-frame stop / go
knife edge (16.6 m apart at decision 0, 0.4911 vs 0.4910). The Harmonizer-on smoke was stopped on purpose (Harmonizer is ~7x slower on a 3090,
about 2.3 h were left); it was supplementary, because the board's `e2e_challenge/ec2.yaml` runs the renderer without the Harmonizer.

## Independent audit (fresh Opus agent, 2026-10-23): GO

All 10 checks pass: layers / config (content differences only checkpoint, tags.json, code; package layer differs in install timestamps only),
checkpoint sha256 in image = box, driver family `pai` with the intended switches and the read's `get_version`, all baked files = `git show
b251b657`, env diff = `JEV_TAG` + `SH30_GIT_HASH` only, CPU behaviour as p1021a, no secrets (history and filesystem scan), uid 10001,
outward tag `p1023a` clean, smokes consistent with cross-host drift. Concerns, none blocking:

1. The Harmonizer-on smoke was stopped at 12 / 29 scenes, so there is no Harmonizer-on result (the board's ec2 preset renders without it).
2. As with p1021a, every flip goes image-lower (pooled about -0.017, CI touching 0). The organisers' H100s are a third host; the local R8K
   margins (+0.045 over NTV0) can shrink there.
3. The chain's own comparison step called `pai_image_cmp.py` without its `label=` arguments; the `cmp_box.md` files were regenerated by hand
   and `run.sh` is fixed. `pai_smoke.sh` printed a cosmetic `unbound variable` at c007 teardown (the `/tmp` sampler racing the container stop).
4. Candidate status: decision 443 says R8K "not adopted" under its registered rules, while the status block names R8K-L18-s2 as the best single
   checkpoint by on / off average; the probe is the user's call. vennai-org's October quota was not re-read (sessions expired): run `limits` first.

**Outward names.** Registry tag and submission name: **`p1023a`**, repository `teams/vennai-org`. Mapping, kept only here: `p1023a` =
`jev-alpasim:r8k-l18-s2-b251b657-pai-vc1.0-lead-acc-gap0.7_2.7` = `sha256:63adee3b...1dce87` = R8K-L18-s2 + g10 + ACC + GAP 0.7,2.7. As in
p1021a, internal strings remain inside the image and only a neutral rebuild removes them: env (`JEV_*`, `JEV_TAG=R8K-L18-s2`, `SH30_GIT_HASH=b251b657`),
history (`ARG TAG=R8K-L18-s2`, `ARG GIT_HASH=b251b657`, `COPY jev-drive /app/jev-drive`), paths (`/app/jev-drive`,
`/app/data/runs/op_parity/runs/R8K-L18-s2/`), and `get_version` `jev-pai-R8K-L18-s2-...-trackgate`, which the runtime records in every rollout.

**Verdict: GO** for an official PAI probe under vennai-org.

## Commands for the user (Tokyo)

Not run by an agent. Profile `acct3` = vennai-org, but on 10-21 it was re-pointed to the ujs-iv token for p1021a, and all sessions were
expired on 10-23: sign in again **with the vennai-org account** and check `me` before anything else. One official slot.

```bash
# Tokyo (ssh ujs@100.108.238.8). ~/bin/ap <profile> = the competitor CLI (or docker) with that profile's config / ECR login
~/bin/ap acct3 auth-url; ~/bin/ap acct3 configure-token         # sign in with the vennai-org HF account, paste the token
~/bin/ap acct3 me                                               # must show team vennai-org (Venn Intelligence Foundation); stop if it shows ujs-iv
~/bin/ap acct3 limits                                           # remaining official slots this UTC month (expected 2); stop if 0
~/bin/ap acct3 ecr-login                                        # prints the team repository URI
IMG=jev-alpasim:r8k-l18-s2-b251b657-pai-vc1.0-lead-acc-gap0.7_2.7
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/vennai-org:p1023a
docker image inspect "$IMG" --format '{{.Id}}'                  # must print sha256:63adee3b5831334cfaea1107b5fcf057d4cbb0e1b7a8ce578af62646b61dce87
docker tag "$IMG" drive:p1023a; docker tag "$IMG" "$URI"
~/bin/ap acct3 docker push "$URI"                               # rerun on a transient error; ~8 GB
cat /data/runs/alpasim/pai_image/gains10.json                   # {"rel_front_steering_angle_weight": 10}
~/bin/ap acct3 submit --track pai --controller-gains /data/runs/alpasim/pai_image/gains10.json "$URI"   # official, one slot
~/bin/ap acct3 status <submission_id>                           # SUCCEEDED; rollout version jev-pai-R8K-L18-s2-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate
~/bin/ap acct3 leaderboard --track pai                          # one call: read p1023a, p1021a (ujs-iv) and rsg-l18-s0-acc-gap-v1 together
```

Same form as the vennai-org RSG submission (decision 356) and p1021a (decision 426). Optional first: `submit --track pai-warmup ...` (status
only, own quota of 3 per 30 days).
