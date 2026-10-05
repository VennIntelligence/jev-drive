# wm_policy step 0: can worldmodel-4B generate with no ego future?

Answer: **no in-distribution way exists.** Pose can be dropped in distribution; the future anchor frames cannot.
No scored run was made, no prereg committed, no GPU time used.

## What the shipped code does

Sources read: `openpilot.distill/rl/{server,env}.py` and the torchpackage's bundled
`torchtitan/experiments/worldmodel/{model,model_for_inference,schedulers}.py` (worldmodel-4B, fp8_nvfp4).

- **Input format is fixed**: latents `(B, 15, 32, 16, 32)` = 5 future anchors `f0..f4` + 9 history `h0..h8` + 1
  noised target `x`. `pos_embed` covers exactly 15 x 128 tokens; prefill always caches the first 14 frames clean
  (`t = 0`), so the anchors are always attended to by the target (block-causal mask: `x` sees everything).
- **Anchors are logged future video.** `rl.env.Env` takes them from the episode at `end_index..end_index+4`;
  `future_fidxs` starts at `9 + (end - start)` and counts down by one per step, so over a rollout anchors sit at fidx
  10..49 (0.2-9.8 s after the target frame, which is always fidx 9). The fidx embedder has exactly 50 entries,
  matching that range.
- **Pose conditioning**: per-frame `augments_pos_ref_augment` (dx, dy, dz), `ref_augment_from_augments_euler`
  (roll, pitch, yaw) and `pose_mask` (0 = pose given, 1 = masked). `env.py` gives only the target frame's pose
  (`pose_mask[-1] = 0`, from the actor's curvature / accel through `Physics`); anchors and history are masked.
- **cfg null branch** (`pack_cfg_inputs`): duplicates the decode tokens; the unconditional copy has all poses zeroed
  and `pose_mask = 1`, **with the same anchor and history latents**. Velocity = `u + cfg * (c - u)`, cfg 2.0.
  So "unconditional" in this model means *pose-free but anchored*.
- **Plan head**: `plan_head(x[:, -1, :])`, the last token of the decode sequence (with cfg on, the last token of the
  *conditional* copy), at the last Euler step. Output 2 x 33 x 15 (mean | log-sigma; supercombo layout, 10 s horizon).
  It is an inverse-dynamics head: it sees history, the target and the anchors.
- **Paper** (arXiv 2504.19077, sec. 2.5): every world model there is future-anchored; anchors are never noised
  during training (`tau = 0` for anchor latents) and no anchor dropout is described; the anchored model "can only
  be used offline" and yields trajectories "that converge to a goal state at F". No anchor-free variant is trained.

## Readings considered

| Reading | In distribution? | Ego future hidden? |
|---|---|---|
| All poses masked (`pose_mask = 1`, zero pose), anchors from the log | yes (this is the cfg null branch) | **no**: anchors show the logged future 0.2-9.8 s ahead |
| cfg null branch alone (cfg = 0, all masked) | yes | no, same anchors |
| Anchors at the far end (fidx 45-49, 8-9.8 s ahead) | yes | no: reveals route choice and stop / go, the exact cases a policy test needs |
| Anchors replaced by noise, zeros, or copies of the last history frame | **no**: anchors are always clean real frames in training | yes, but the read would measure OOD behaviour, and a stale-copy anchor encodes "the car does not move" |
| Shift the window (history in anchor slots, target at fidx > 9) | **no**: target is always fidx 9 in the shipped schedule | yes |

So the only in-distribution "no ego future" mode removes the pose but keeps future video, which leaks exactly the
quantity being scored (logged path ADE, stop / go). WM-uncond as specified (history only) cannot be run as shipped.

## Options (not run; for the main session)

1. Stop here (the brief's default for this outcome).
2. A labelled-OOD diagnostic, about 0.5 GPU-h: on the 8-10 pilot clips read the plan head under (a) pose-free +
   logged anchors at fidx 10-14 / 24-28 / 45-49, (b) pose-free + noise anchors, (c) pose-free + stale-copy anchors,
   against Cinque and the log. This measures how much of the plan comes from the anchors (anchor dependence), not
   whether the world model is a policy; result (b) / (c) would be OOD by construction.
3. Fine-tune with anchor dropout: a new training run, outside "run as it ships".
