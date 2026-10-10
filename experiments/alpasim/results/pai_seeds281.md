# PAI track, 281 curated-validation scenes: P2H10S-F (two seeds) against base, paired (2026-10-11)

Descriptive read, not registered; it promotes nothing (same standing as decision 235, whose 60 scenes it extends to 281). All runs: Tokyo box,
`pai_eval.sh` / `pai_run.sh`, served configuration `JEV_VCONT=1.0 JEV_LEAD=1`, CONC 4, unharmonised renderer, one rollout per scene. Scenes:
the 281 of `/data/runs/alpasim/pai_full/s_{b1a,b1b,b1c,q40,b2,n1,n2,n3}.tsv`. Base is `P2H10-F-s0` (one base seed).
Read-out: `pai_eval_report.py --rows base=... s0=... s1=... --mean both=s0,s1 --pair s0-base s1-base both-base s1-s0`; its output is
[pai281/paired281.md](pai281/paired281.md) (with the throughput of every run dir), one row per scene in [pai281/paired281_scenes.tsv](pai281/paired281_scenes.tsv).

**Caveat on the absolute scores.** `pai_core.slots` feeds rendered 10 Hz frames to all history slots of checkpoints trained on warped frames
([../../body1/results/served_plan_length.md](../../body1/results/served_plan_length.md), a diagnosis of 2026-10-11 without a second reader, fix
not applied). Every row below is served that way, so the absolute means are low by an unknown amount; the paired differences have the same input
on both sides.

| Row | Scenes | Mean scene score | Zeros | at-fault collision | offroad | left corridor | other zero |
|---|--:|--:|--:|--:|--:|--:|--:|
| base `P2H10-F-s0` | 281 | 0.3746 | 144 | 23 | 42 | 80 | 1 |
| `P2H10S-F-s0` | 281 | 0.4629 | 119 | 18 | 33 | 70 | 0 |
| `P2H10S-F-s1` | 281 | **0.4837** | **110** | 14 | 29 | 72 | 0 |
| mean of the two seeds (per scene) | 281 | 0.4733 | 99 (zero in both) | | | | |

Zero kinds are the scorer's flags on zero-score scenes; two flags can be set on one scene, so a row's kinds can sum to more than its zeros.

| Paired difference | Scenes | Mean | Scene-bootstrap 95 % CI | Better / worse / within 0.01 |
|---|--:|--:|---|---|
| s0 - base | 281 | +0.0883 | [+0.0472, +0.1304] | 60 / 38 / 183 |
| s1 - base | 281 | +0.1091 | [+0.0655, +0.1536] | 62 / 49 / 170 |
| two-seed mean - base | 281 | +0.0987 | [+0.0594, +0.1394] | 69 / 47 / 165 |
| s1 - s0 | 281 | +0.0208 | [-0.0098, +0.0516] | 38 / 50 / 193 |

Bootstrap: 10 000 resamples of scenes, seed 0. The CI covers scene sampling only: there is one base seed, and decision 235 measured a
base-to-base spread of up to 0.029 on 60 served scenes, so the base seed is an unmeasured term of these differences on this scene set.

Where the difference comes from (decomposition of the paired mean over the 281 scenes):

| | base zero -> arm scores | arm zero <- base scored | both score | both zero |
|---|---|---|---|---|
| s0 - base | 37 scenes, +0.1154 | 12 scenes, -0.0276 | 125 scenes, +0.0005 | 107 |
| s1 - base | 45 scenes, +0.1382 | 11 scenes, -0.0278 | 126 scenes, -0.0013 | 99 |

As in decision 235, the whole difference is zero-score scenes becoming scored ones (net 25 and 34 scenes); on scenes both drivers pass the
score is unchanged. All three zero kinds are lower in both seeds. The two seeds are not separable (CI includes 0); 99 scenes are zeros in both
seeds, 93 of them also in base.

On the 139 scenes that had a base rollout before this run (the earlier subset read): base 0.3245, s0 0.4086 (+0.0841 [+0.0193, +0.1503]),
s1 0.4440 (+0.1195 [+0.0544, +0.1873]), two-seed mean +0.1018 [+0.0419, +0.1646].

Run dirs (Tokyo box, `/data/runs/alpasim/`): base `pai_full/runs/{full1_ab,base_q40_c1}` (139 scenes) and `pai_base/runs/*` (142); s0
`pai_full/runs/p2h10s-f-s0_*`; s1 `pai2c/runs/s1_b1a_c4` (33) and `pai_s1/runs/*` (248).
