# LAMP-PIXELS: is the lamp colour in the AlpaSim render at all? (2026-10-19)

Diagnosis, no policy training. Prereg [plans/2026-10-19-lamp-pixels-prereg.md](../plans/2026-10-19-lamp-pixels-prereg.md), committed before
any output. Code: `scripts/lamp_pixels.{py,sh}` (select / frames / cnn / ft / tok / eval / report). Pool chain 1019-124832-e111 ..
1019-124837-64c7 (7 dump reads, priority 12, about 55 min wall). Box: `$DATA_DIR/runs/defect_fix/lamp_pixels/` (tv / arena parquet, render
JPEGs + serve packs, models, scores; 7.6 GB, own dump npz deleted). Full tables: [lamp_pixels.json](lamp_pixels.json); `lamp_pixels.py report`.

**Answer.** By the registered rule the branch is **(m) intermediate**, neither (a) nor (b). Both gates pass, so the readers are strong
enough to conclude. On the 883 arena held windows at decision 0 (red vs green, compared only within the same speed x distance cell), our
lamp reader scores 0.852 on the real photos and 0.769 [0.64, 0.85] on the render of the same instants:
- **(a) is rejected.** The render keeps most of the lamp colour. It loses 0.083 [-0.15, -0.03] against the paired real frames, and at the
  line itself (< 10 m) it keeps 0.82.
- **(b) is rejected too.** Our dedicated reader adds nothing over the frozen Cinque tokens. On the render: CNN 0.769 against TOK 0.774,
  difference -0.005 [-0.14, +0.12]. On the real frames: TOK 0.867 against CNN 0.852. Native resolution adds nothing either: the 4x
  downscaled CNN-L matches CNN-N everywhere.

So the frozen eye already carries the lamp about as well as a native-resolution reader trained for it. A new lamp pathway into the planner
would not add an input the policy lacks.

At the 241 red-line no-lead waits (the d372 loss), the render shows red. The reader calls 95 % of them red (real 100 %). The policy is
given a red lamp it can read, and it still rolls. So this is the training signal, not the input.

## Setup (as registered)

- Real training: NAVSIM trainval CAM_F0, route connector state from the tv census (`ln_type == 2`, `ln_conn`, line -3..60 m ahead of the front
  bumper, red / green at t0).
  - Balanced red = green in each of 12 cells (speed 0 / 0.3 / 3 / 8 m/s x line distance 10 / 25 m).
  - Train: 30 744 frames, 903 logs (`navsim/tv-clean-train-logs`). Hold: 5 562 frames, 95 logs (`tv-clean-hold-logs`).
- Arena: held windows with a known connector 0-60 m ahead. 8 454 window x decision rows, labels from the raw nuPlan test DB state at that
  instant.
  - Paired real frame: the 2 Hz OpenScene CAM_F0 nearest in time (p99 0.10 s).
  - Render: JTN-L18-s1 native `SH30_DUMP` 1920x1080 JPEG. 235 windows were newly dumped; the rest reuse the native_gap / redline_probe dumps.
  - Primary set: k = 0 (force-GT, the human's pose), 883 windows: 778 red / 105 green, 31 logs (green from 24).
- Readers:
  - CNN-N: ImageNet ResNet-18, fully fine-tuned on 1920x640 upper crops at native resolution, no hue jitter, 3 epochs.
  - CNN-L: the same on a 4x-downscaled crop.
  - TOK: Cinque `view_39` on the single-frame serve pack (prev = cur), LN -> 16 -> MLP probe.
  - All trained on TV only and applied unchanged. FT: CNN-N fine-tuned on the render k = 0 rows, 5 log folds.
- AUC: pairs within the same cell. CI: log-cluster bootstrap, B = 2000, with paired differences on the same draws.

## Gates and primary (k = 0, cell-stratified, 95 % log CI)

| reader | TV hold | arena real | arena render | render - real |
|---|--:|--:|--:|--:|
| **CNN-N** | **0.946** [0.93, 0.96] | 0.852 [0.75, 0.92] | 0.769 [0.64, 0.85] | -0.083 [-0.15, -0.03] |
| CNN-L (1/4 res) | 0.950 | 0.848 | 0.777 | |
| TOK (frozen Cinque) | 0.924 | **0.867** [0.76, 0.94] | **0.774** [0.64, 0.89] | |
| CNN-N fine-tuned on render (log folds) | | | 0.796 [0.66, 0.90] | |

- G1 (TV hold ≥ 0.85): 0.946, passed. G2 (paired real ≥ 0.80): 0.852, passed.
- R = 0.769, inside (0.65, 0.85), so the branch is **(m)**. CNN-N - TOK on the render: -0.005 [-0.14, +0.12], so the (b) margin
  (≥ +0.05) is not met either.
- Natural (unstratified) AUCs are higher: real 0.923, render 0.844, TOK render 0.772. They are shown for reference only, because the
  speed / distance confound inflates them.

By line distance (cell-stratified, CNN-N real / render; TOK render):

| line ahead | pairs | real | render | TOK render |
|---|--:|--:|--:|--:|
| -3 to 10 m (at the line) | 5 009 | 0.862 | 0.821 | 0.849 |
| 10 to 25 m | 1 143 | 0.905 | 0.682 | 0.665 |
| 25 to 60 m | 2 332 | 0.803 | 0.699 | 0.668 |

How to read it:
- The render loses the lamp mainly at 10-60 m: about 0.10-0.22 below real, which looks like blur or baking of small distant lamps.
- At the line, where the 239 red-line windows stand, the render keeps it (0.82-0.86).

## Secondary

**Tracking (within-window red / green pairs, 60 windows with a state change, 21 logs).** CNN-N real 0.632, render 0.589,
difference -0.043 [-0.22, +0.12]; TOK real 0.702 / render 0.613. The real frames are weak here too: at a red-to-green switch the ego
stands at the line, the near lamps are above the crop or the frame, and the label's switch instant is a hard target. This test is
uninformative. It does not show that the lamp is frozen.

**Baked lamp shares** (CNN-N called red at tau = 0.101, the TV-hold balanced-accuracy threshold, 0.875):

| rows | n | logs | real | render |
|---|--:|--:|--:|--:|
| red wait (C), k0 | 635 | 28 | 0.97 | 0.79 |
| red wait (C), no lead, k0 | 241 | 17 | 1.00 | 0.95 |
| launch (G), k0 (log red) | 64 | 24 | 0.75 | 0.86 |
| launch (G), after the log turned green | 82 | 14 | 0.23 | 0.49 |
| log green, k0 | 105 | 24 | 0.23 | 0.25 |

How to read it:
- Where the log is green at the window start, the render shows green as often as the real photo does (0.25 against 0.23).
- At the d372 red-line waits the render shows red (0.95).
- In the launch windows, after the log turns green the render still shows red about half the time (0.49 against 0.23 real). That is
  d340's frozen red, but partial, not total.

## What it decides

- **Not (a): the cue is in the simulator** for the windows that cost the 85 points. The red at the line is rendered and readable.
- **Not (b): no new lamp pathway is warranted.** The frozen tokens read red vs green as well as our native-resolution CNN (real 0.867 vs
  0.852, render 0.774 vs 0.769). The 4x downscale costs nothing. So the bottleneck is not resolution, tokenisation or a missing
  dedicated eye. Per the prereg, no pathway proposal is written for (m).
- **Correction to d373's reading.** d373 inferred "the lamp is not readable" from TOK - LEAD < 0.05 and from P vs G 0.531. But G windows
  are red at k = 0 in the log too (the human launches later). So P vs G is not a lamp contrast. Its evidence was "the tokens add
  nothing beyond standing + no lead for the P-vs-N task", not "the tokens cannot see the lamp". With lamp state as the label, matched for
  speed and distance, the same frozen eye reads it at 0.87 (real) / 0.77 (render).
- **Consequence.** Holding at a red line with no lead is a training-signal problem on features the policy already has: standing, no lead,
  and red lamp. A red-line hold rule that also reads the lamp would mis-hold in about half of the post-green launch decisions (render
  red 0.49), so the d372 cost of about 13 points on the launch windows still mostly applies.
- **Evidence: medium.** There are 105 green k = 0 windows from 24 logs, and the CIs are wide (render 0.64-0.85). Each reader is one seed;
  the frames are deterministic.
