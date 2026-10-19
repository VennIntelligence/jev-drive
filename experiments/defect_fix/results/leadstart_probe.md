# LEADSTART-PROBE: when a stopped lead moves, at which decision does the policy see it? (2026-10-19)

Diagnosis, no training. Prereg [plans/2026-10-19-leadstart-probe-prereg.md](../plans/2026-10-19-leadstart-probe-prereg.md), committed
before any feature or probe output. Code: `scripts/leadstart_probe.{py,sh}` (select / tvsel / tvindex / feats / probe / report). Pool jobs:
`$O/jobs.txt` (dumps 1019-110200-143c .. 1019-110205-c9e1, tvsel 1019-110206-22d6, tvindex 1019-110207-56a5, tvprep 1019-110207-ec23,
feats 1019-110208-b5af, probe 1019-130055-ae35; priority 11). Box outputs: `$DATA_DIR/runs/defect_fix/leadstart_probe/` (lead.json,
tv/rows.npz, feats_{R,T}.npz, tok_{R,T}.npy, probe/{probe.json, scores_*.npz}; dumped frames deleted). Full tables:
[leadstart_probe.json](leadstart_probe.json); `leadstart_probe.py report --src <dir>` prints them.

**Answer.** The policy sees the lead start at the **first decision after it has moved 0.3 m**, in the render and in real frames alike.
- At that decision (time since onset 0-0.5 s) the frozen tokens separate "lead has moved" from "lead still standing" at AUC 0.984
  [0.97, 0.99] on the render (R) and 0.947 [0.94, 0.95] on real trainval frames (T).
- The lead head is already there too: KIN + lead head reaches 0.96 (R) / 0.93 (T), and ACC's single vLead number reaches 0.945 (R) /
  0.82 (T). The tokens add only +0.02 over the head in every bin (CI above 0, below the +0.05 bar). So nothing is hidden in the tokens
  beyond what the head already reads.
- Even the decision *before* the 0.3 m crossing is partly readable (exploratory, below): the lead's sub-threshold creep shows.
- By the registered rule this is **"both readable", so not an input problem.** The render does not hide the start (R ≥ T), and the
  2 Hz format does not blind the eye. The late launch of d363 / d365 is a training-signal problem on a cue the policy already reads.
- One input limit is real: the lead head's **distance** readout barely moves in the first decision. dRel rises +0.18 m (R, RSG head)
  against a logged +0.30 m, and only 20 % of the events clear the readout's own jitter (p90 0.43 m between standing decisions).
  - In the second decision it rises +0.65 m against a logged +0.81 m, and 63 % of the events clear the jitter.
  - The speed readout (vLead +0.24 m/s, AUC 0.945) is the faster signal. An ACC that waits for the distance to grow loses about one
    decision (0.5 s).

## Setup (as registered)

| source | windows / events | logs | rows (ego standing, k >= 1) | positives by bin 0-0.5 / 0.5-1 / 1-1.5 / 1.5-2 s |
|---|--:|--:|--:|---|
| R1 wait set v2 lead classes, JTN-L18-s0 closed loop (a 68 + b 64 windows) | 132 | 53 | 942 | 93 / 73 / 41 / 16 |
| R2 arena held, launch 48 + static 488 windows with a standing lead <= 15 m, JTN-L18-s1 native | 533 | 27 | 4 264 | 69 / 57 / 41 / 23 |
| R = R1 + R2 | 665 | 64 | 5 206 | 162 / 130 / 82 / 39 |
| T trainval real frames, `navsim/tv-clean-train-logs@v1`, split `navsim/leadstart_tv@v1` | 1 537 events | 584 | 12 856 | 1 506 / 1 336 / 997 / 702 |

- Dropped at selection: R1 lead already moving at w0 (18 windows), no lead <= 15 m (8); R2 lead already moving (66).
- T events are ego-standing runs behind a lead that was stopped and then crossed 0.3 m. 2 517 of their rows sit in census G_L frames
  (the census counts 1 403 G_L events). 999 events have a light line within 60 m.
- Gate passed: the dumping model on the recomputed tokens, with the dumped route memory, reproduces the dumped plan to max 0.033 m
  (p99 0.018). Logged boxes of standing leads drift 0.0-0.1 m, so the 0.3 m threshold is clean.

## Primary: TOK AUC by time since the 0.3 m onset (95 % log CI)

| source | set | 0-0.5 s | 0.5-1 s | 1-1.5 s | 1.5-2 s |
|---|---|--:|--:|--:|--:|
| R1 | KIN | 0.752 | 0.926 | 0.937 | 0.912 |
| R1 | LEAD (best) | 0.956 [0.93, 0.98] | 0.983 | 0.972 | 0.986 |
| R1 | **TOK** | **0.982 [0.97, 0.99]** | 0.999 | 1.000 | 0.999 |
| R2 | KIN | 0.624 | 0.728 | 0.747 | 0.707 |
| R2 | LEAD (best) | 0.951 [0.93, 0.97] | 0.982 | 0.955 | 0.891 |
| R2 | **TOK** | **0.979 [0.95, 1.00]** | 0.990 | 0.982 | 0.950 |
| R | ACC-v (vLead alone) | 0.945 | 0.964 | 0.940 | 0.906 |
| R | LEAD (best) | 0.962 [0.95, 0.98] | 0.987 | 0.959 | 0.917 |
| R | **TOK** | **0.984 [0.97, 0.99]** | 0.993 | 0.989 | 0.970 |
| R | TOK - LEAD | +0.022 [+0.01, +0.04] | +0.006 [-0.00, +0.01] | +0.030 [+0.00, +0.06] | +0.053 [-0.01, +0.12] |
| T | KIN | 0.695 | 0.756 | 0.751 | 0.699 |
| T | ACC-v | 0.823 | 0.837 | 0.787 | 0.720 |
| T | LEAD (best) | 0.930 [0.92, 0.94] | 0.953 | 0.939 | 0.909 |
| T | **TOK** | **0.947 [0.94, 0.95]** | 0.974 | 0.968 | 0.948 |
| T | TOK - LEAD | +0.017 [+0.01, +0.02] | +0.021 | +0.029 | +0.039 [+0.03, +0.05] |

The two lead heads (RSG-L18-s0, JTN-L18-s1) agree to within 0.005 everywhere.

Every source is readable from its first bin (AUC >= 0.75, lower CI > 0.60). "Beyond the head" (+0.05 with lower CI > 0) fails everywhere.

KIN is above 0.5 because both the ego row and the key count shift once the lead goes. The model (R) and the human (T) creep a few
cm/s below the 0.3 m/s standing cut, and the k = 1-2 cold starts fall early in the windows.

At each fold's 10 % FPR threshold, TOK calls "moved" on 98 % (R) / 92 % (T) of first-bin positives; the lead head on 94 % / 82 %.

## Slots: where the motion enters (secondary 1)

Per-slot TOK, bin 0-0.5 s, against the source keyframe of each slot (sh30_core: slot 7 = K0, 6 = K0 warped, 4-5 = K-0.5 warped,
1-3 = K-1.0 (2 is the real keyframe), 0 = K-1.5 warped). An image pair spans two keyframes only at slots 1, 4 and 6.

| slot | 0 | 1 | 2 (real K-1.0) | 3 | 4 | 5 | 6 | 7 (real K0) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| pair spans keyframes | - | K-1.5 -> K-1.0 | no | no | K-1.0 -> K-0.5 | no | K-0.5 -> K0 | no |
| R | 0.656 | 0.846 | 0.684 | 0.697 | 0.947 | 0.674 | **0.980** | 0.691 |
| T | 0.681 | 0.918 | 0.743 | 0.745 | 0.945 | 0.716 | **0.920** | 0.703 |

- The motion is read from the **cross-keyframe pairs** (slots 6, 4, 1). The real keyframe slots 2 and 7 hold only a single image's
  position: 0.69-0.74 at onset, rising to 0.80-0.90 later as the gap grows.
- The registered reading "slot j rises only once its source keyframe is past the onset" holds for slots 6 / 7 against 0-3, 5. It fails
  for slots 4 and 1: they are already high at 0-0.5 s. Their pairs (K-1.0 -> K-0.5, K-1.5 -> K-1.0) end before the 0.3 m crossing, so
  what they carry happens before it. The exploratory check below shows what that is.

## Exploratory (not registered): before the 0.3 m crossing

The last decisions before the onset against decisions more than 1.5 s before it (R also includes the windows where the lead never moves):

| source | rows | KIN | LEAD-RSG | ACC-v | TOK | TOK-s4 | TOK-s6 | TOK-s7 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| R, -0.5..0 s | 189 | 0.54 | 0.85 | 0.86 | 0.87 | 0.77 | 0.84 | 0.61 |
| R, -1..-0.5 s | 200 | 0.48 | 0.66 | 0.68 | 0.60 | 0.61 | 0.59 | 0.57 |
| T, -0.5..0 s | 1 496 | 0.59 | 0.84 | 0.90 | 0.81 | 0.82 | 0.84 | 0.63 |
| T, -1..-0.5 s | 1 451 | 0.52 | 0.58 | 0.83 | 0.57 | 0.60 | 0.60 | 0.57 |

- The lead's sub-0.3 m creep, and the cars ahead of it that go first, are visible one decision before the crossing.
- On real frames the head's vLead flags it 1 s ahead (0.83). This is plausibly the queue ahead, or brake lights going off.
- So "the lead starts" is visible to the policy at the decision it happens, or earlier. There is no input delay to wait out beyond the
  2 Hz keyframe step.

## Lead-head change in the first second (ACC's signal)

Change from each event's last decision before the onset (within 1 s), median [log CI]; jitter = p90 of |dRel change| between two
consecutive standing decisions before the onset.

| source / head | bin | n | dRel change m | logged lead displacement m | vLead change m/s | events above jitter |
|---|---|--:|--:|--:|--:|--:|
| R / RSG (jitter 0.43 m) | 0-0.5 s | 162 | +0.18 [+0.13, +0.23] | +0.30 | +0.24 [+0.16, +0.29] | 20 % |
| R / RSG | 0.5-1 s | 130 | +0.65 [+0.48, +0.85] | +0.81 | +0.47 | 63 % |
| R / JTN1 (0.38 m) | 0-0.5 s | 162 | +0.20 [+0.15, +0.26] | +0.30 | +0.27 | 29 % |
| T / RSG (0.35 m) | 0-0.5 s | 1 490 | +0.12 [+0.11, +0.13] | +0.26 | +0.09 [+0.08, +0.10] | 20 % |
| T / RSG | 0.5-1 s | 1 320 | +0.35 [+0.32, +0.38] | +0.64 | +0.15 | 50 % |

- **Yes, the distance readout changes in the first second, but it under-reads and lags.** The change is about 60 % (render) / 50 % (real)
  of the true displacement, and in the first decision it stays inside its own jitter for 70-80 % of events.
- The speed readout moves earlier and is what separates the classes.
- lead_prob does not change: it stays saturated at "present".

## T light confound (secondary 2) and gap (secondary 4)

- T without a light line: TOK 0.930 / 0.965 / 0.963 / 0.948 against LEAD 0.902 / 0.930 / 0.921 / 0.907. With a light: 0.954 / 0.978 / 0.971 /
  0.945. A green light turning on is not what the tokens read: the no-light subset is as readable.
- R1 variant b only: TOK 0.982 / 0.999 / 1.000 / 1.000 (as R1).
- By lead gap at the start, T 10-15 m (20 / 15 / 13 / 10 positives): TOK 0.78 / 0.82 / 0.85 / 0.78. Small displacements at 10-15 m are
  the only weak cell. The render has almost no windows there.

## What it decides

- **Registered reading: readable in the render and in real frames from the first decision after the onset, not beyond the lead head.**
  The on-time launch is not blocked by the input. The policy's own lead head and tokens show the start at the decision it happens, and
  at the decision before.
  - Consistent with d363: JTN's plan responds after the onset and only carries a standstill shortening.
  - Consistent with d365: what is missing is a training signal that conditions the launch on this readout, not a visible cue.
- A launch rule or loss can key on what the policy already computes: lead-head vLead, or a probe on the cross-keyframe slots 4 / 6.
  It should **not** key on the lead distance growing, which costs about one decision.
- For the render-vs-real question: the render does not hide the start (R reads it at least as well as T). The trainval frames add
  nothing new about visibility. They are a large clean pool of labelled lead starts (1 537 events) for such a signal.
- Evidence: strong for "readable at the first decision" (four sources agree, tight CIs, T has 573 logs at bin 0). Medium for the
  slot attribution and the pre-onset reading (exploratory). Each read is one closed-loop seed. The R2 static windows dominate R's
  negatives.
