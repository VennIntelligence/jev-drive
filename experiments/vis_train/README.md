# vis_train: training Cinque's vision weights on navtrain

status: live
decisions: 145, 160, 204
index: VT: trainable vision branch / in-place unfreeze from SH30, six arms training (no read yet)

**Question.** Does the navtest turn off-road gap (and anything else on the boards) move when the vision weights of the driver are really
trained on navtrain, instead of staying frozen as in every method tried so far?

**Conclusion.** None yet: the arms are training (2026-10-11). The lane's output is the trend of the read-outs against steps and weight
displacement, at about 1/12 of WA-JEPA's fine-tuning dose.

**Arms** (all continue `SH30-F-s<seed>` on the same row stream, batch 64, constant learning rate after a 300-step warmup; pre-registration
and its amendments in `plans/`):

| Arm | Seeds | Vision | Control |
|:--|:--|:--|:--|
| F | 0, 1 | frozen, no branch | the same-steps baseline |
| A0 | 0, 1 | frozen; the adapter memory channel reads Cinque's own t0 token | "a channel was added" |
| A | 0, 1 | trainable copy of the vision encoder on the t0 image pair, its 32 x 512 tokens through the memory channel | A0 |
| B | 0, 1 | A + prediction of the frozen tokens 1.0 s and 2.0 s ahead (weight 0.5) | A0, A |
| C | 0 | the encoder itself trained in place, no anchor rows, no distillation of the non-plan heads | F0 |
| F0 | 0 | frozen, no anchor rows, no distillation | F |
| W (second wave) | 0, 1 | A with three t0 views: the shared-weight branch encodes CAM_F0 / CAM_L0 / CAM_R0 (at least +-82 deg together), 3 x 32 tokens through the memory channel; test-time `:sideoff` masks the two side views | A |

A0 / A / B / W carry the branch's own head (thin decoder on [tokens, ego]; decision 204), trained jointly, unused at inference.

**How it runs.** `scripts/vt_chain.sh <arm> <seed> <steps>` (one tmux window per arm and seed) puts the whole chain into the GPU pool:
the training job (`scripts/vt.py train`, resumable from its last snapshot), a navtest read per snapshot (`VT-<arm>-s<seed>-k<NN>`) and the
final reads (`VT-<arm>-s<seed>`: navtest, navhard on protocol W, test-time masked / shuffled memory for A and B, HUGSIM 64 for the plain
P2 arms F and F0), all through `python -m jevdrive.bench` (family `vt`: `vt.py plans` reads the cached slot tokens and the pixel cache).
`scripts/vt.py ident` is the identity gate, `scripts/vt.py tokens` dumps branch tokens for the decision-160 Stage-0 probe.

**Throughput.** Pixels are rendered once into `$DATA_DIR/runs/vis_train/px/` (`scripts/vt_px.py`) and read through `lib/pixel_store.py`;
the profile, the before / after numbers and the step counts they set are in the throughput section of `plans/state.md` (the reference
is decision 145's U2: online CPU rendering, 0.7 steps per second at batch 64).

**Next.** Reads at every snapshot against the registered lines and the no-regression rule (pre-registration amendment 2), the
displacement trend, the Stage-0 probe on the branch tokens, then the decision entry and the Chinese report in `research/vis_train/`.

**Read more.** `plans/2026-10-10-vis-train-prereg.md` (design, checks, reads, stop rules, amendments), `plans/state.md` (working state:
jobs, tags, run dirs, check results, how to resume).

<!-- files:begin -->
## Files

- `2026-10-10-vis-train-prereg.md` (plans): VT：在 navtrain 上真正训练视觉参数，navtest …
- `state.md` (plans): VT 工作状态（builder 与吞吐工程师共用；各写各的小节）
- `vt.py` (scripts): vis_train trainer and read-out paths
- `vt_chain.sh` (scripts): one self-advancing chain per arm and …
- `vt_px.py` (scripts): the protocol-W model frames of every …
- `pixel_store.py` (lib): Pre-rendered protocol-W pixels for …
- `pp_train.py` (experiments/op_parity/scripts): Cinque fine-tuned on navtrain with …
- `pp_unfreeze.py` (experiments/op_parity/scripts): how much of P2's 3.55 EPDMS gap to …
- `parity_adapter.py` (lib): Input-parity adapter for openpilot …
- `navsim.py` (jevdrive/bench): navtest EPDMS and navhard two-stage

[plans/](plans/) 2 live plans · [scripts/](scripts/) 5 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `plans/` live plan notes. Refresh the file list and INDEX.md with `python tools/topic_index.py`.
