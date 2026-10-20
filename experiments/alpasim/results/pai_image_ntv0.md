# PAI probe image NTV0-L18-s0 + g10 + ACC + GAP 0.7,2.7: re-audit, second smoke, submission commands (2026-10-21)

Read this before you push or submit the NTV0 PAI probe. Candidate: decisions 366 / 370 / 392 (NTV0-L18-s0 + `JEV_VCONT=1.0 JEV_LEAD=1
JEV_ACC=1 JEV_GAP=0.7,2.7`, no holdlead, steering weight 10 at submit). This is the serve of the submitted `rsg-l18-s0-acc-gap-v1`
(decision 356) with the checkpoint swapped. The image, its first audit and its first smoke (chunk c007) are in
[pai_image.md](pai_image.md), section "NTV0 ACC-GAP image". This page adds an independent re-audit and a second smoke on two more chunks.
Tokyo artefacts: `/data/runs/alpasim/pai_image/audit_ntv0_v2/`.

**Not rebuilt, on purpose.** The image built on 2026-10-19 from 0d417ee4 (`new_tag.sh`, `TRACK=pai ACC=1 GAP=0.7,2.7`) is still on
Tokyo, and every baked code file equals the code of the box's 281 read (2e2d0e46). A rebuild at HEAD would bake three files that changed
after 0d417ee4: `serve_fix.py` (`JEV_ACC_VSET` / `REL` / `ALAT`, d398), `sh30_core.py` (`SH30_WARP`) and `pp_train.py`. The new switches
are off by default and bit-identical when unset, but the rebuilt image would no longer be the code of the read. So the old image is the
better-matched artefact.

**Serve conflict to note.** Decision 398 (10-20) chose `+ JEV_VHIST=holdlead` for the PAI probe, overriding decision 370. The user asked
for this config without holdlead on 10-21, and this image has no holdlead. A holdlead variant would be one more `ENV` layer (`VHIST=holdlead`).

## Image

| field | value |
|---|---|
| local name | `jev-alpasim:ntv0-l18-s0-0d417ee4-pai-vc1.0-lead-acc-gap0.7_2.7` (Tokyo) |
| image id | `sha256:37183c048a06965ef588d1b7c6810865018956bf926ebf8e0718f0da616a0558` |
| size | 8 110 811 511 B (7.55 GiB), 18 layers |
| built from | 0d417ee4, `env DATA_DIR=/data TRACK=pai ACC=1 GAP=0.7,2.7 ... new_tag.sh NTV0-L18-s0 ...` (2026-10-19) |
| `get_version` | `jev-pai-NTV0-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate` |
| registry digest | none yet: it exists only after the push. `docker push` prints it, and `docker buildx imagetools inspect <URI>` shows it |

**Env vs the submitted RSG image** (`jev-alpasim:rsg-l18-s0-ea61e9df-pai-vc1.0-lead-acc-gap0.7_2.7`, `sha256:328433cb...`, = `rsg-l18-s0-acc-gap-v1`,
audit 3ce87714), sorted `Config.Env`, re-read 2026-10-21:

```
< JEV_TAG=RSG-L18-s0          > JEV_TAG=NTV0-L18-s0        (the checkpoint)
< SH30_GIT_HASH=ea61e9df      > SH30_GIT_HASH=0d417ee4     (build commit, informational)
                              > JEV_HOLD=                  (empty = off; serve.py execs pai_driver.py as before)
```

All serving switches are equal: `JEV_DRIVER=pai JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_VHIST= SH30_COMPILE=0`, with
`JEV_TRACK` unset (v8 gate). The rest of the config is equal too: CMD `python /app/serve.py`, no entrypoint, user 10001:10001, workdir
/app, port 6789, SIGTERM. Only the checkpoint changes the served behaviour. The other two lines are a commit label and an empty switch.
The box read's `config.txt`: `SERVE=JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`, `WIZ=...rel_front_steering_angle_weight=10`,
`CONC=4`, `ALPASIM_FAST=1`, `code=2e2d0e46`. `ALPASIM_FAST` is a simulator-side setting of the box runner, not a driver switch. The
image does not carry it, and the official stack does not use it.

## Audit (2026-10-21, re-run on the image itself)

| check | result | |
|---|---|---|
| image id / size / config | `sha256:37183c04...0558`, 7.55 GiB < 40 GiB, CMD / user / port / signal as RSG image | pass |
| env vs submitted RSG image | only `JEV_TAG`, plus `SH30_GIT_HASH` and an empty `JEV_HOLD=` (above) | pass |
| checkpoint in image = box checkpoint of the 281 reads | `ckpt-final.pt` `5fa544acf89b...aa155b`, `route_mem.json` `f5c259003a92...d57816`, both equal to `/root/autodl-tmp/ujs/runs/op_parity/runs/NTV0-L18-s0/` on the AutoDL box (sha256 re-read 10-21) | pass |
| baked code = code of the read | all 31 `.py` / `.json` files under `/app/jev-drive` equal `git show 2e2d0e46:<path>` (sha256) | pass |
| secrets | `docker history --no-trunc`: 0 hits (hf_ / AKIA / ghp_ / sk- / nvapi- / user@ URL / proxy / box address); filesystem: no .env / .netrc / .git-credentials / challenge.json / ssh key / .git, no token or private-key pattern in /app /etc /home | pass |
| limits (`test.sh`, card 0, organisers' flags + `--network none`) | listening after 19.4 s (runtime waits 240 s); driver GPU 1.47 GiB (16 GiB); `/tmp` 2.8 MiB (2 GiB), `/run` 0 (64 MiB); 0 read-only errors, 0 changed files, 0 tracebacks; 800 `drive` calls p50 / p99 48.5 / 115 ms, 0 inference errors; SIGTERM -> exit 0 in 0.9 s; `pass: true` ([test/result.json](pai_image/audit_ntv0_v2/test/result.json)) | pass |
| smoke-side limits | driver GPU peak 1.47 GiB, `/tmp` 33 MiB over 36 scenes, 0 driver tracebacks, 0 inference errors over 14 233 `drive` calls (server p50 57-59 / p90 75-77 / max 383 ms); the reach of pypi.org from the driver container fails | pass |

**Verdict: GO** (one concern below). The image serves the code and checkpoint of the box's d366 PAI 281 read, with the same serve switches. The
steering weight is passed at submit. Nothing was rebuilt, and the image id is unchanged since the 10-19 audit (also GO).

## Smoke (Tokyo, official containerised PAI stack, steering weight 10, image as built, no driver env)

`pai_smoke.sh` with `WIZ=controller.gains.rel_front_steering_angle_weight=10` runs on two further chunks of the box read
`defect_fix/pai_accgap/NTV0-L18-s0` (identical scene lists, copied with their native results-summary + drive.jsonl to Tokyo
`pai_image/native/ntv0accgap_c00{0,1}`). c000 ran on card 1, c001 on card 0. The comparison is `scripts/pai_image_cmp.py`.

| chunk (scenes) | image mean (zeros) | native mean (zeros) | paired diff [95 % CI] | equal / within 0.01 / within 0.05 | zero agreement | decision-0 plan diff, median / max | zero flips |
|---|--:|--:|--:|--:|--:|--:|---|
| c000 (36), card 1 | 0.6931 (10) | 0.6923 (10) | +0.0009 [-0.0000, +0.0024] | 31 / 35 / 36 | 36 / 36 | 0.7 mm / 4.2 cm | none |
| c001 (36), card 0 | 0.7666 (6) | 0.7939 (5) | -0.0273 [-0.0834, +0.0012] | 27 / 34 / 35 | 35 / 36 | 1.3 mm / 19.3 m (bd97de34) | ba62e10d 0 (offroad) vs 1.0 |
| c007 (29), 10-19, card 1 ([pai_image.md](pai_image.md)) | 0.5403 (13) | 0.6090 (11) | -0.0687 [-0.1724, +0.0005] | 25 / 27 / 27 | 27 / 29 | 0.6 mm / 1.28 m | 2bbf1efc, 3232fdb7 0 (corridor) vs 1.0 |
| pooled (101) | 0.6754 (29) | 0.7046 (26) | -0.0292 [-0.0686, +0.0005] | | 98 / 101 | | 3, all image 0 / native 1 |

Files: [smoke_c000/](pai_image/audit_ntv0_v2/smoke_c000/), [smoke_c001/](pai_image/audit_ntv0_v2/smoke_c001/) (`compare.md` per scene,
`cmp_box.md` decision by decision), [run.sh](pai_image/audit_ntv0_v2/run.sh) (the chain).

Reading. Within decision 215's cross-host drift (the box runs torch `+cu130` on an RTX 6000D, the image runs `+cu126` on an RTX 3090). On
the first decision, where both hosts see the same frame and ego state, served plans differ by a median 0.7-1.3 mm, so code, checkpoint
and switches agree. The new flip ba62e10d is equal at decision 0 (0.3 mm), and its plans part by more than 0.1 m only at decision 42: the
closed loop amplifies the numerics, like c007's two flips (parting at decisions 4 and 24). bd97de34 is a first-frame stop / go knife edge,
like 35e082a4 and 1a5da90a in the earlier smokes: 19.3 m apart at decision 0, yet the same score on both hosts (0.4908). On one host the
image is deterministic (c007 same-host repeat, 29 / 29 bit-identical, 10-19). All rollouts serve the version above, and the driver's start
line shows `JEV_ACC=1 JEV_GAP=0.7,2.7`, so the serve is live.

**Concern (not blocking).** On all three chunks the flips go one way (image lower), and the pooled paired diff is -0.029 with a CI that
touches 0. The RSG image showed the opposite sign on c007 (+0.063, 2 flips the other way, pai_image.md), so the sign is a property of
host x scene, not of the image. Still, the organisers' H100s are a third host, and NTV0's single-host margin over RSG (d366 +0.0545
[+0.024, +0.088] on 280 scenes) can shrink there.

## Board, quota, account

**Account.** Per the user (10-21), the probe goes under **ujs-iv** (`UJS IV Squad`, the nuPlan team of decision 374), not acct3 /
vennai-org. **On 2026-10-20 16:50 UTC no CLI session was alive.** `~/bin/ap ls` shows acct3 / lao-siji / ujs-iv all EXPIRED. The
default `~/.alpasim` reports `missing_or_expired_session`, `ap ujs-iv limits` returns HTTP 401, and the ujs-iv token was last written on
10-19. The user has to re-authenticate (`auth-url` -> HF sign-in -> `configure-token`, 12 h session). So nothing was pushed or submitted.

**Quota (read from the board without a session, 2026-10-20 16:48 UTC; `limits` needs a session).** Official: 3 per team per UTC
calendar month, shared by PAI and nuPlan, no carry-over; accepted submissions count even if evaluation fails (docs/alpasim.md "Dates and
quota"). October entries: ujs-iv 2 (both nuPlan, 10-19), so **1 left**; vennai-org 1 (2 left); lao-siji 2 (1 left). A PAI probe under ujs-iv
spends ujs-iv's last October slot. The board closes 2026-10-31 (UTC), so ujs-iv could then submit no nuPlan final (B2 `STANDFMXH-L18-s1`,
d391) before the close. Warm-ups (`pai-warmup`): 3 per rolling 30 days, unscored, separate.

**Best or latest.** Not settled by any rule read. The board API lists every official submission as its own row: 108 submission rows plus
1 benchmark row on PAI, with teams on several rows (acmlab-dow at ranks 1 and 3). Rows are ranked by PCS (`policy_capability_score`, zoib).
A new submission adds a row and does not replace an old one, so on the public board a team's best row counts. For the final, the rules say
only "Public leaderboard closes; one final valid container per track selected" (2026-10-31; at least the top 3 teams per track get final
runs). Who selects it (the team, or best / latest by default) is not stated in any source read (README `e2e_challenge`, docs/alpasim.md).
Ask the organisers before relying on either.

**Board at read time (PAI, 2026-10-20 16:48 UTC, PCS refits on every call).** #1 acmlab-dow 2251, #2 snu-vail 2223, #3 acmlab-dow 2213;
vennai-org `rsg-l18-s0-acc-gap-v1` rank 18, PCS 2091 [2045, 2138], mean 0.632. nuPlan: ujs-iv `jtn-l18-s1-holdstop3` rank 31, 1458;
`jtn-l18-s1-native` rank 35, 1321.

**Naming (user rule, 10-21).** Nothing outward-facing carries internal names. Registry tag and submission name: **`p1021a`** (the board
shows the image tag as the entry's name; `submit` takes no other name). Repository: `teams/ujs-iv` (fixed by the registry). Local
neutral tag on Tokyo: `drive:p1021a` -> `sha256:37183c048a06...0558`, the same image id as the internal tag. Mapping, kept only here:
`p1021a` = `jev-alpasim:ntv0-l18-s0-0d417ee4-pai-vc1.0-lead-acc-gap0.7_2.7` = NTV0-L18-s0 + g10 + ACC + GAP 0.7,2.7.
The image has no labels (`Config.Labels` null, no author). Some internal strings remain that only a rebuild could remove, so they were
kept. The organisers can see them in the image itself:
ENV values (`JEV_TAG=NTV0-L18-s0`, `SH30_GIT_HASH=0d417ee4`, `JEV_GAP=0.7,2.7`, ...), `docker history` (`ARG TAG=NTV0-L18-s0`,
`ARG GIT_HASH=0d417ee4`, `COPY jev-drive /app/jev-drive`), the paths `/app/jev-drive` and `/app/data/runs/op_parity/runs/NTV0-L18-s0/`,
and the driver's `get_version` string `jev-pai-NTV0-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate`. The runtime records that string
in every rollout, and the submission status may show it. Removing these means a new build with a neutral `TAG` / version string, then a
new test and smoke.

## Commands for the user (Tokyo)

Not run by an agent. Push and submit only after the user has decided on the ujs-iv slot (above). Exactly one official submission.

```bash
# Tokyo (ssh ujs@100.108.238.8). ~/bin/ap <profile> = the competitor CLI (or docker) with that profile's config / ECR login
~/bin/ap ujs-iv auth-url; ~/bin/ap ujs-iv configure-token        # session expired on 10-20: sign in with HF, paste the token
~/bin/ap ujs-iv me                                               # must show team ujs-iv (UJS IV Squad); stop if not
~/bin/ap ujs-iv limits                                           # remaining official slots this UTC month (expected 1)
~/bin/ap ujs-iv ecr-login                                        # prints the team repository URI
IMG=drive:p1021a
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/ujs-iv:p1021a
docker image inspect "$IMG" --format '{{.Id}}'                   # must print sha256:37183c048a06965ef588d1b7c6810865018956bf926ebf8e0718f0da616a0558
docker tag "$IMG" "$URI"; ~/bin/ap ujs-iv docker push "$URI"      # rerun on a transient error; ~8 GB on a repo without these layers
cat /data/runs/alpasim/pai_image/gains10.json                    # {"rel_front_steering_angle_weight": 10}
~/bin/ap ujs-iv submit --track pai --controller-gains /data/runs/alpasim/pai_image/gains10.json "$URI"   # official, one slot
~/bin/ap ujs-iv status <submission_id>                           # SUCCEEDED; first rollout version jev-pai-NTV0-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate
~/bin/ap ujs-iv leaderboard --track pai                          # one call: read p1021a and vennai-org rsg-l18-s0-acc-gap-v1 together
```

ujs-iv's ECR repository has nuPlan layers only. The base / package layers come from the same Dockerfile but were built on 10-19 from other
commits, so expect most of the 8 GB to upload. A `pai-warmup` first (status only, own quota) is optional: d356's identical serve passed
32 / 32 there.

## Pushed and submitted (2026-10-21 ~02:50 JST, session, on the user's handover)

Clash GLOBAL on `AWS Japan 01`. Profile `acct3` now holds the ujs-iv token (`me` -> team_id ujs-iv; `limits` before submit: 2 of 3
used, 1 left). Pushed `teams/ujs-iv:p1021a`, registry digest `sha256:db8168ab51afc7d91ba2adcd78f37366337aa26245e76bb87e6692e705baa8a2`
(local image id `sha256:37183c04...0558`). Official PAI submission with `gains10.json` (`rel_front_steering_angle_weight` 10):
submission id `e616a394-8885-4514-a3ab-bd302507fd02`, QUEUED 4 / 4. This spends ujs-iv's last October slot. Logs:
Tokyo `/data/runs/alpasim/pai_image/push_p1021a/`.
