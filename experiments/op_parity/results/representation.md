# op_parity representation fix (four_dirs item 4): stage 0 decoders and stage 1 policy pilot

Pre-registration: [plans/2026-10-07-representation-design.md](../plans/2026-10-07-representation-design.md) section 6 (thresholds fixed before
any readout; declared deviations in 6.5, written before scoring). Code: `scripts/rep.py` (vj21 / x4 / decode / mem / report / s1gate /
s1report), `scripts/rep_chain.sh` (stage 0), `scripts/rep_s1_chain.sh` (stage 1), `pp_train.py --mem`. Tables: `results/representation/`.
Stage 2 (full run + HUGSIM) is not launched: it needs review.

**Caveat that holds for every WA-Cf number below.** WA-Cf is WA-JEPA's front encoder after WA-JEPA fine-tuned it end to end on navtrain
(about 139 epochs, planning loss + JEPA loss) on top of nuPlan video SSL. It is a diagnostic upper bound for "a better front representation",
not a method: using it as one would be Cinque plus a competitor's NAVSIM-trained encoder.

## Result

**Gate verdicts.** Stage 0: reproduction pass, positive control pass, **VJ21 fail** (closure 0.35 < 0.5), X4 closure 0.14 (< 0.2:
(a) would need the whole encoder). Stage 1 (arms H0 / MW; MV not run because VJ21 failed): **MW passes every pre-registered criterion** on
the 2-seed means: navtest EPDMS +0.95 [+0.47, +1.37] (gate >= +0.30, CI low > 0), T20 DAC failures -1.33 pp [-2.45, -0.18] (gate <= -1.0,
CI high < 0), EP +0.09, NC + TTC failures -0.63 pp, S5 EPDMS +0.36, plan speed ratio x0.999 of H0, drift_off 0.044 m. With its memory masked
at test time MW falls to 0.60 below H0, so the policy reads the channel. Pre-registered reading for "only MW passes": the gain needs a
NAVSIM-supervised encoder; (b-WA) is not a method; the next step is (c1) / (a) with WA-Cf as the teacher and, by X4, the whole encoder.
Stage 2 not launched (review).

Details: **Stage 0.** The positive control passes: adding WA-Cf to Cinque's vision tokens cuts the T20 (logged turn > 20 deg, 3 154 navtest tokens)
DAC failure rate of the same thin decoder from 10.59% to 6.37% (-4.22 pp [-5.63, -2.83]). The original V-JEPA 2.1 ViT-L (VJ21, frozen,
same encoder path as WA-Cf, only the weights differ) closes 0.35 [0.12, 0.61] of that gap when added to Cinque's tokens (-1.46 pp
[-2.59, -0.43]); the gate needs 0.5, so **VJ21 fails and stage 1 ran without the MV arm**. On its own VJ21 is no better than Cinque's tokens
(11.79% vs 10.59%, +1.20 pp [-0.58, +2.82]). Cinque's own pre-head feature map (X4, `conv2d_36`) closes 0.14 [-0.06, 0.37]: below 0.2, so a
stage-4-only encoder adaptation has no information headroom here; (a) would need the whole encoder.

## Stage 0: offline thin decoders (6.1)

Same decoder for every arm: opb_probe's 2-layer MLP (1024) on z-scored [source, E], hinge lambda 10 / margin 0.3, 4 000 steps, batch 512,
seed 0, trained on navtrain s2-s4 minus the dev logs (25 415 tokens; `navsim/op-parity-s234-train@v1`). DAC failure = devkit `pdm_score`
DAC < 1 (`opb_score.py`, v2 navtest metric cache) on every token of the stratum. CI: cluster bootstrap over the 136 navtest logs, B 4 000;
closure c(X) = (f(V) - f(X)) / (f(V) - f(V+WA)) with its CI from the same resampled logs.

| stratum | arm | n | DAC fail % [95% CI] | f(V) - f(arm), pp | closure c(arm) |
|:--|:--|--:|:--|:--|:--|
| T20 | E (ego only) | 3154 | 19.91 [17.21, 22.24] | -9.32 [-11.65, -6.77] | -2.21 [-3.68, -1.34] |
| T20 | V (Cinque `view_39`) | 3154 | 10.59 [8.73, 12.46] |  |  |
| T20 | WA (WA-Cf) | 3154 | 6.88 [5.49, 8.41] | +3.71 [+1.88, +5.54] | 0.88 [0.60, 1.09] |
| T20 | V+WA | 3154 | 6.37 [5.19, 7.68] | +4.22 [+2.83, +5.63] | 1 (definition) |
| T20 | VJ21 | 3154 | 11.79 [9.58, 13.88] | -1.20 [-2.82, +0.58] | -0.29 [-0.86, 0.12] |
| T20 | V+VJ21 | 3154 | 9.13 [7.31, 10.93] | +1.46 [+0.43, +2.59] | 0.35 [0.12, 0.61] |
| T20 | X4 (`conv2d_36` t0) | 3154 | 9.99 [8.09, 11.94] | +0.60 [-0.25, +1.42] | 0.14 [-0.06, 0.37] |
| T45 | E | 1517 | 20.17 [16.60, 23.34] | -7.65 [-11.00, -3.87] | -1.84 [-4.32, -0.78] |
| T45 | V | 1517 | 12.52 [10.39, 14.74] |  |  |
| T45 | WA | 1517 | 8.83 [6.67, 11.32] | +3.69 [+0.85, +6.34] | 0.89 [0.33, 1.23] |
| T45 | V+WA | 1517 | 8.37 [6.67, 10.40] | +4.15 [+2.10, +6.06] | 1 (definition) |
| T45 | VJ21 | 1517 | 13.58 [10.92, 16.26] | -1.05 [-3.38, +1.29] | -0.25 [-1.25, 0.26] |
| T45 | V+VJ21 | 1517 | 11.40 [9.21, 13.51] | +1.12 [-0.33, +2.67] | 0.27 [-0.11, 0.67] |
| T45 | X4 | 1517 | 12.20 [9.72, 14.82] | +0.33 [-1.14, +1.69] | 0.08 [-0.33, 0.52] |
| S5 (< 5 deg) | V | 6400 | 2.17 [1.60, 2.86] |  |  |
| S5 (< 5 deg) | V+VJ21 | 6400 | 1.98 [1.38, 2.72] | +0.19 [-0.13, +0.51] |  |

Gates (pre-registered; [representation/stage0.json](representation/stage0.json)):

| rule | threshold | result | verdict |
|:--|:--|:--|:--|
| reproduction: stratified full-navtest V / WA vs decision 147 (5.12 / 3.41) | within 0.3 pp | 5.12 / 3.41 (diff -0.003 / -0.002) | pass |
| reproduction: stratified T20 V / WA vs section 1 (10.9 / 6.0) | within 0.7 pp | 10.88 / 6.00 | pass |
| positive control f(V) - f(V+WA) | >= 2.5 pp, CI low > 0 | +4.22 [+2.83, +5.63] | pass |
| VJ21: c(V+VJ21) >= 0.5, f(V) - f(V+VJ21) CI low > 0, S5 not worse by > 0.3 pp | all three | c 0.35; +1.46 [+0.43, +2.59]; S5 -0.19 pp | **fail** (closure) -> stage 1 runs H0 / MW only |
| X4 | >= 0.5: (c2) / stage-4 (a) viable; < 0.2: (a) needs the whole encoder | c 0.14 | (a) needs the whole encoder |
| attribution (report only): c(VJ21 alone) vs c(WA alone) | - | T20 -0.29 vs 0.88; T45 -0.25 vs 0.89 | consistent with the gap coming from WA's nuPlan video SSL + navtrain fine-tune, not from generic V-JEPA 2.1 pretraining |

The decoders are bit-for-bit the decision-147 decoders for E / V / WA (same train rows, seed and loop; final losses identical), so the
reproduction rows check the new scoring sets, not a refit. The section-1 T20 point estimates (V 10.9, WA-Cf 6.0) came from 721 scored
tokens re-weighted; on all 3 154 T20 tokens they are 10.59 and 6.88: the representation gap on turns is 3.7 pp, not 4.9.

Checks. VJ21 path equivalence: WA's weights + scene projector through the front-only path reproduce the cached WA-Cf on 200 navtest tokens
(mean per-token cosine 0.99991, min 0.993; gate 0.999). The V-JEPA 2.1 file (`vjepa2_1_vitl_dist_vitG_384.pt`, key `ema_encoder`) loads all
302 encoder tensors (ratio 1.0000, none left at WA's values). VJ21 PCA 1024 -> 512 per token keeps 98.8% of the variance (raw 1024-d arm
trained but not scored, deviation 4). X4 t0 slot through the frozen stage 4 + head reproduces the W front cache (mean abs 0.0009, RMS 1.7).
Decoder train fit (imitation loss): V 0.0109, WA 0.0097, VJ21 0.0167, X4 0.0066, V+WA 0.0065, V+VJ21 0.0087 ([representation/fits.csv](representation/fits.csv)):
X4 fits navtrain as well as V+WA but does not transfer to navtest turns.

Secondary readouts (report only):

| readout | E | V | WA | VJ21 | X4 | V+WA | V+VJ21 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| junction look-ahead AUC (future path enters a junction, current pose not in one; 3 078 / 12 146 positives; 5-fold log cross-fit) | 0.643 [0.620, 0.666] | 0.805 [0.783, 0.826] | 0.850 [0.832, 0.869] | 0.823 [0.803, 0.842] | 0.808 [0.787, 0.830] | - | - |
| failing share on the P2H10 T20 failure set (297 tokens, 69 logs; selection-biased toward Cinque-feature errors, deviation 8) | 48.5% | 52.9% | 31.6% | 45.8% | 48.5% | 37.0% | 49.2% |

On the junction probe VJ21 sits between Cinque and WA-Cf (+0.018 over V, overlapping CIs); the failure-set shares repeat the selection-bias
pattern of the design note (E below V on P2H's own failures).

## Stage 1: policy pilot (6.2)

Recipe: H0 = the P2H pilot recipe (`--arm P2 --frames warp --host --hinge-lam 10`, 3 000 steps x batch 64, warmup 100, d_frac 0.25) on
navtrain s2-s4 (`navsim/op-parity-s234`, 25 415 train / 408 dev tokens); MW = H0 + `--mem wa_cf` (32 WA-Cf front tokens of the current frame
on the adapter's side channel, LN(512) + Linear(512 -> 256) + cam / time / slot embeddings, memory masked on 25% of rows from its own rng
stream, so H0 and MW see the same rows, anchors and order). Seeds 0 / 1. navtest through `jevdrive.bench` (v2 devkit, full 12 146 tokens);
per-token seed means, paired cluster bootstrap over the 136 logs, B 4 000 ([representation/stage1.csv](representation/stage1.csv)).

Seed-0 early stop ([representation/stage1_gate_s0.json](representation/stage1_gate_s0.json)): MW-s0 - H0-s0 T20 DAC failures -1.08 pp
[-2.15, +0.11] (rule: drop >= 0.4 pp), EPDMS +0.98 [+0.49, +1.40] (rule: >= 0): continue to seed 1.

2-seed means (x 100; diff vs H0 with 95% CI):

| arm | EPDMS (token mean) | T20 DAC fail % | T45 DAC fail % | S5 EPDMS | EP | NC + TTC fail % | DAC fail % (all) | speed ratio / H0 | drift_off (m) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| H0 | 87.92 | 9.56 | 10.15 | 92.54 | 87.03 | 2.46 | 4.23 | 1 (1.003) | 0.049 |
| MW | 88.86, +0.95 [+0.47, +1.37] | 8.23, -1.33 [-2.45, -0.18] | 10.02, -0.13 [-1.60, +1.62] | 92.90, +0.36 [-0.03, +0.74] | 87.12, +0.09 [-0.04, +0.22] | 1.84, -0.63 [-0.90, -0.34] | 3.75, -0.48 [-0.89, -0.07] | 0.999 | 0.044 |
| MW, memory masked at test | 87.31, -0.60 [-0.85, -0.37] | 9.59, +0.03 [-0.51, +0.57] | 10.42, +0.26 [-0.50, +1.14] | 92.04, -0.50 [-0.80, -0.24] | 87.06, +0.03 [-0.07, +0.12] | 2.77, +0.31 [+0.13, +0.50] | 4.57, +0.33 [+0.15, +0.53] | 0.999 | - |

Devkit summary EPDMS (average row, with EC): H0 87.93 / 87.90, MW 88.91 / 88.82, MW memory masked 87.47 / 87.15 (seed 0 / 1). Dev ADE to the
log: H0 0.62 / 0.62 m, MW 0.42 / 0.43 m.

| criterion (2-seed means) | threshold | MW | MV |
|:--|:--|:--|:--|
| navtest EPDMS, arm - H0 | >= +0.30, CI low > 0 | +0.95 [+0.47, +1.37] pass | not run (stage 0 VJ21 gate failed) |
| T20 DAC failures, arm - H0 | <= -1.0 pp, CI high < 0 | -1.33 [-2.45, -0.18] pass | - |
| guard: EP | >= -0.2 | +0.09 pass | - |
| guard: NC + TTC failures | rise <= 0.2 pp | -0.63 pass | - |
| guard: S5 EPDMS | >= -0.2 | +0.36 pass | - |
| guard: plan speed ratio / drift_off | 1.00 +- 5% / <= 0.10 m | 0.999 / 0.044 pass | - |
| attribution (MV - H0) / (MW - H0) | report | - | - |

Reading (pre-registered in 6.2): only MW passes, so the gain needs a NAVSIM-supervised encoder; (b-WA) is not reported as a method. The
pre-registered next step is (c1) / (a) with WA-Cf as the teacher; by the stage-0 X4 readout (closure 0.14) through the whole encoder, not
stage 4 alone. Two things beyond the gates: the MW gain is not only DAC on turns, NC + TTC failures fall by 0.63 pp (the gap item 3 targets),
and masking the memory puts MW 0.60 below H0 (not back at H0), so the adapter's other pathways co-adapted with the memory; a distilled
encoder (c1) would avoid that dependency. T45 does not move (-0.13 [-1.60, +1.62]); the T20 gain sits in the 20-45 deg turns.

## Stage 2 (pre-declared in 6.3, not launched: review)

What it would take for MW (diagnostic only, since WA-Cf is not a method): WA-Cf front tokens for the other 77 k navtrain tokens and the navhard
stage 1 / 2 frames (the front-only path measured here runs about 60-95 tokens/s per card, so about 20-30 card-min, not the 5 card-h of the
full-model extractor); 2 seeds x the full P2H10 recipe (10 000 steps x 128, about 1 h each); navtest + navhard G through the bench; HUGSIM 64
`spec_plan_smooth` x 2 seeds with a WA-JEPA front-encoder process inside the HUGSIM server (wajepa env, 0.5 s front renders, warm-up frames for
the 5 s standstill), batch-1 latency and a 50-frame cosine >= 0.999 equivalence check first: about one engineering day plus about 3-4 card-h.
Risks: the adapter channel changes closed-loop behaviour (P2's ego channel cut stuck runs 24 -> 0 but raised fg collisions 1 -> 10); MW's
masked-memory read is 0.60 below H0, so any render-domain shift of the WA encoder on 3DGS frames lands on a policy that now depends on it;
the HUGSIM criterion has no power (+-0.08 HD half-width over 64 scenarios). Per the pre-registered reading, the more useful next step is a
(c1) / whole-encoder (a) pre-registration with WA-Cf as the teacher, which keeps Cinque's single encoder and deploys on the device.

## Deviations (plan 6.5, declared before scoring)

1. V-JEPA 2.1 file is 5.15 GB, loaded by WA-JEPA's own loader (`ema_encoder`).
2. Equivalence check in bf16 autocast (the cached WA-Cf's precision), rows of the cached shard `lb_navtest.s0of3`.
3. S5 guard read as a population rate over all 6 400 < 5 deg tokens.
4. Scoring in three batches; the optional raw-1024 VJ21 arm trained but not scored; Drive-JEPA arm not run.
5. T20 stratified reproduction uses section 1's weights (PP 11 494 / 1 500).
6. X4 stored as (2048, 4, 8) of the t0 slot only, flattened per cell.
7. Junction probe recipe fixed (logistic, AdamW 400 full-batch steps, L2 1e-3, 5 log folds).
8. D1 u D2 share read on the P2H10 T20 failure set (stage 0 scores only T20).
9. Stage 1 memory via `pp_train.py --mem` (arm `P2+wa_cf`, side channel n_cam = n_t = 1, own drop stream p = 0.25); memory-off read =
   bench `:noside`; splits `navsim/op-parity-s234-train@v1:08fc1d5edd65` / `-dev@v1:542e3107daf9`.

Run-time notes: the first VJ21 extraction was cancelled and rerun after a fix that only reorders request building by log (identical
outputs, 7x faster); the agent-hinge recipe did not become the default (decision 158), so H0 is the P2H pilot recipe as pre-registered.

## Cost

Stage 0: GPU about 0.5 card-h (VJ21 extraction 2 x ~7 min incl. the cancelled run, X4 12 min, decoders + probe 6 min); CPU scoring 38.3 k
`pdm_score` calls, about 40 min wall on 40 + 24 + 16 cores. Stage 1: GPU 4 trainings x ~3.5-6 min + 6 bench plan exports, about 0.5 card-h; CPU 6 bench navtest scorings (6 shards x 12 cores, about
10 min each, partly queued behind other lanes; the cgroup quota is 75 cores although `nproc` shows 208). Both stages within the plan's
estimates (0.5 + 1 GPU-h, ~1 + ~1 h CPU).
