# P1 level (a) pre-registration: how many feature directions does WA-JEPA's planner read? (2026-10-22, before any GPU job)

Decision row: 454 (reserved). Paper: d452, research/paper-outline section 3.2 (P1).

## Subject and feature site

WA-JEPA released checkpoint (`$DATA_DIR/models/wajepa/model_state_dict.pt`, navtest EPDMS 91.71 in our earlier read,
`runs/top10_t2/navsim/wajepa/20260926-122804`). Feature = output of `MultiViewCausalFutureMaskedJEPA._encode_scene_context`:
V-JEPA 2.1 ViT-L history tokens -> SceneTokenProjector (LayerNorm + Linear 1024 -> 512) -> `context_scene` [B, 4096, 512]
(4 cameras x 2 tubelets x 16 x 32 patches). It is the only image input of the flow planner, read at all 4 denoising steps.
The hook (`lib/usage_probe.Probe`) wraps that method on the model instance; third-party code is unchanged.

## Arms

`off` (hook installed, no projection) and k in {8, 16, 32, 64, 128, 256, 512 (= full rank, a numerical control)}:
x -> mu + (x - mu) U_k U_k^T per 512-d token row, fp32, U_k = top-k eigenvectors of the token covariance.
The encoder runs once per navtest token; the planner runs once per arm on the cached feature; the flow noise is re-seeded
per call by the model, so every arm sees the same noise (paired).

## PCA fit

navtrain only: logs of the archives `navtrain_{current,history}_3` (downloaded for this; the navtrain blobs were deleted
10-21), tokens whose four history frames are on disk, at most 3000 tokens spread over those logs, every one of the 4096 token
rows of each sample. Never navtest. Explained variance per k is reported on the fit (navtrain) and held out (navtest, with
the navtrain mean and directions).

## Identity gate (stop if it fails)

On 24 navtest tokens: G1 hook off == no hook, G2 our DataLoader feature path == `agent.compute_trajectory`, G3 sweep arm
`off` (cached feature) == no hook, G4 two plain calls identical; all bit-for-bit. Reported, not gating: max abs difference to
the stored 91.71 plans. On the full run: arm `off` must score within 0.3 EPDMS of 91.71, and arm 512 within 0.1 of `off`
(fp32 rounding only); otherwise stop and investigate before any read-out.

## Scoring and statistics

`python -m jevdrive.bench run --model PROBE-wajepa-ctx-<arm> --bench navtest` (v2 devkit one-stage, non-reactive, the
harness that reproduced 91.71; all 12 146 tokens), sub-scores NC DAC DDC TLC EP TTC LK HC EC. Paired differences arm - off,
per-token, percentile bootstrap clustered by log (bench report: B = 10 000, seed 0).

## Read-out lines (fixed now)

- retained(k) = EPDMS(k) / EPDMS(off). k95 = smallest k with retained >= 0.95; k99 likewise with 0.99 (point estimates).
  CI-backed versions: smallest k whose paired CI lower bound of EPDMS(k) - EPDMS(off) is >= -0.05 x EPDMS(off)
  (resp. -0.01 x). Both are reported; the CI-backed one is the headline.
- Interpretation for C2: CI-backed k95 <= 32 (<= 6 % of the 512 channel directions) = "the planner reads a small subspace";
  k95 >= 256 = "the planner uses most of the space" (contradicts C2 at this site); in between = partial, reported as such.
- Sub-scores: which ones fall first as k shrinks (descriptive).

## Caveats fixed in advance

- PCA directions rank variance, not what the planner reads: a high-variance direction can be ignored and a low-variance one
  can be decisive. k95 is therefore an upper bound on "how many directions are needed" only along this ordering; level (c)
  (causal attribution) would answer it directly and is out of scope tonight.
- The site is after the planner's own projector (512-d), not the raw 1024-d V-JEPA tokens; the projector is trained with the
  planner, so this is the subspace the driving head consumes.
- Token-level PCA: one subspace shared by all tokens / views / times; spatial structure is untouched.

## Level (b) (only if (a) finishes cleanly; pre-registered separately before it runs)
