# PCS-LOCAL: the leaderboard's Policy Capability Score refit on the arena held split (2026-10-19)

Diagnostic, no training, no new closed-loop reads. The question: can we reproduce the AlpaSim nuPlan Policy Capability Score (PCS)
locally on the ARENA held split, and which scenes cost us PCS? Context is decision 374. On the private board, JTN-L18-s1 native
scored mean 0.9155 / PCS 1325, and with stop3 it scored 0.950 (the best mean on the board) / PCS 1463. The garage 0014 checkpoint
scores 0.848 / 1543; the leader is 0.944 / 1899.

Code: `experiments/alpasim_proxy/scripts/pcs_local.py`, with subcommands `build`, `fit`, `report`, `subcurve` and `score`. Small
artifacts are in [pcs_local/](pcs_local/):

- the input matrix: `scores.csv.gz`, `types.csv.gz`, `windows.csv.gz`;
- the generated tables: `report.md` (board-mix pool), `report_boardmix-g15.md` and `report_official.md`;
- `pcs_by_variant.csv` and `subcurve.csv`;
- the frozen board-mix item fits: `boardmix.npz`, `boardmix-g15.npz`, `items_boardmix.csv.gz`.

The board snapshot from 10-19 is `../results/leaderboard_nuplan_20261019.json` (130 entries).

## Short answer

1. **Fitting the official model on our own arena reads does not reproduce the board.** That pool has 33 subjects, 29 of them
   our own family. It ranks every one of our stop3 variants first (JTN-s1 + stop3 1874) and the garage near the bottom (1174),
   with Kendall tau +0.29 over the 8 subjects that have a private PCS. Changing seeds, switching to 1PL, dropping the third-party
   subjects or deduplicating to 12 subjects does not help. **IRT discrimination depends on who is in the pool**: when our family is
   the majority, the windows our family fails get low discrimination.
2. **A pool that copies the board's composition does reproduce the pattern.** The board-mix pool is garage x30, LTF x10, DD x5,
   go-straight x2, plus our four board entries. Fitted ability is mapped to board PCS by least squares on the 6 anchors. The JTN
   pair stays out of sample. It gives native **1251** and stop3 **1333** against garage **1553**; the board has 1325 / 1463 / 1543.
   So the "highest mean, lower PCS than garage" pattern comes back, with stop3 above native (+82; board +138). Tau over the 8 is
   +0.71 to +0.86. With garage weighted 15 instead of 30, the result is 1322 / 1446 / 1554, within 17 PCS of the board on both
   probes. That weight was not tuned on the probes, but the 6 anchors cannot choose between weights 15, 30 and 60 (RMSE 142-145
   for all three).
3. **What costs us PCS is non-1 partial scores on windows the strong pool scores exactly 1. Zeros and static windows cost little.**
   Under the zoib likelihood, a strong subject's partial score is penalised far more than a zero: the Beta part's density at
   y < 1 collapses as a·theta grows. On a typical launch item (a 3.2) in the board-mix pool, an exact 0 costs 2.7 PCS, a 0.5
   costs 25, a 0.9 costs 8 and a 0.99 costs 4.4. For JTN-s1 + stop3:

   | change | windows | mean-score points | PCS gain (board-mix / g15) |
   |---|--:|--:|--:|
   | slow-progress partials -> 1 | 105 | 16.8 | **+145 / +223** |
   | partials >= 0.98 -> 1 | 17 | 0.2 | +10 |
   | every zero -> 1 | 57 | 57 | +23 / +40 |
   | launch-window partials (hold cost) -> 1 | 74 | 40.9 | +18 / +27 |
   | static windows | 0 to fix | - | 0 (our static wins over garage are worth only 6-13) |

   Slow progress means progress below 0.8 of the human arc on a moving window (braking / other / cruise / turn). A score of
   exactly 1 needs `progress_clipped_rel >= 0.8` (`scene_score.py`, `progress_saturation_threshold` 0.8). Bringing those 105
   windows up to the human's speed adds 0.008 to the mean and would lift stop3 to about the garage's PCS (1469 / 1656 against
   1553). For native, the static-window corridor zeros cost the most (115 windows, +82 / +119). That is what stop3 bought on the
   board (+138).
4. **Selector.** Do not replace the arena mean with a local PCS fitted on our own reads; at any subject count it ranks the wrong
   way. Add **the frozen board-mix PCS** next to the arena mean: `pcs_local.py score --rows <arena collect>` scores a new read as
   one more submission against the frozen items. Use it to break ties when arena means overlap, and track the number of non-1
   scores on windows the garage aces. Within our family it agrees with the mean (Spearman 0.86-0.92), but it reorders the top. On
   board-mix, JTB-s0 + stop3 scores 1387 (100 partials) and JTN-s1 + stop3 1333 (204), while RSG + stop3, the best mean at 0.945,
   scores 1233 (253 partials).

## 1. The model and the official settings

Sources: drive-irt at 82ddd5a (`irt/beta_irt.py`, `guide.py`, `prior.py`, `elo_scale.py`) and
`e2e_challenge/local_evaluation/evaluate.py` at alpasim 0bb4c4b. Both were read and installed in an isolated uv env on the Mac
(`tmp/pcs_local/.venv`, torch 2.14, CPU).

- **Likelihood (Molenaar zoib 2PL).** Each item n has discrimination a_n = exp(log_a), difficulty b_n, Beta log-precision o_n,
  and thresholds g0_n < g1_n, with g1_n = g0_n + softplus. With exact 0 meaning y <= 1e-4 and exact 1 meaning y >= 1 - 1e-4:
  - P(0) = sigmoid(g0 - a theta)
  - P(1) = sigmoid(a theta - g1)
  - an interior value has probability (1 - P(0) - P(1)) times a Beta density with alpha = exp((eta + o)/2) and
    beta = exp((-eta + o)/2), where eta = a theta - b.

  Difficulty enters only the interior. **A 0.95 is a different outcome from a 1. For a strong subject on an item with high a,
  an interior value is the most expensive outcome**, because alpha grows like exp(a theta / 2). Fitted precision stays close
  to its prior (median exp(o/2) = 1.0 in every pool), so the Beta shape is set by eta alone.
- **Priors.**
  - theta and b: Normal(mu, 1/u), with mu ~ N(0, 1e6) and u ~ Gamma(1, 1).
  - log_a, o, g0 and g1_raw: N(0, 1).
  - Inference: mean-field SVI with Adam.
- **Official settings (evaluate.py).** epochs 1000, lr 0.1, lr_decay 0.975 per epoch, 16 particles, seed 42, clip_norm 10,
  and the average of the last 2.5 % of iterates. 2PL is the default.
- **Matrix rules.**
  - Repeated rollouts of a scene are averaged first.
  - A run whose scene set differs from the canonical set is excluded, so the matrix is complete.
  - **Sufficiency guard:** zoib needs at least S + 5N observations, otherwise the evaluator silently falls back to the
    arithmetic mean. Every fit here passes; 2106 windows means at least 6 subjects.
- **Scale.** PCS is a positive affine map of ability, fixed by two anchor subjects and their target scores
  (`score_scale` / `EloScale`). The board's anchors are not published. The host's `policy_4` at 600 / 606 suggests a low anchor
  at 600. Locally the map is a least-squares fit on the 6 private anchors of decision 355: garage 1543, go-straight 1300,
  DD 1080, YR 1030, LTF 1014, APY 779 (board 10-19). JTN-s1 native and + stop3 are left out of the fit as the out-of-sample check.
- **Board ranking.** The board ranks by the posterior rank upper bound, which this analysis does not need.

## 2. Score matrix

Rows come from `arena.py collect` over every arena read root on the box (`$DATA_DIR/runs/alpasim_proxy/pcs_local/rows_*.csv.gz`;
the roots are listed in `sources.txt`). The window set is the 2228 valid held windows, with the arena's route-fail windows
dropped. The per-window score is the rollout `score` in `results-summary.json`, the same scene score the board uses (one rollout
per window). The official complete-matrix rule forces the canonical set to the garage's 2106 held windows.

Subjects: 36 read, 33 complete.

- **Ours (29).**
  - RSG, NR1F, NR1HV, NR1HV2, YR, APY.
  - RSG + full / lead / leash3 / stop3, and RSG + stop3 + ACC.
  - JTN-s0 and JTN-s1, each native and + stop3; JTN-s0 + stop3 + ACC; JTN-s1 + warp-const.
  - JTB-s0, NTV0-s0 and NTVX-s0, each native and + stop3.
  - JT00-s0 and JT10-s0, each native and + leash3.
  - BNC2K-s1, BNX2K-s1.
  - Partial reads, so excluded: NR1FB10 (1753 windows), FREEM1K and JTNS1K (753).
- **Third-party (4):** garage 0014, LTF, DD, go-straight.
- **Missing.**
  - No V2, P0 or NTV0-s1 read exists on the arena held split.
  - **No host-like strong subject exists either: none of the ~30 py123d-garage variants that make up the top of the board
    (0.87-0.94 private).** The garage 0014 checkpoint is the only public one.
  - That subject pool is what calibrates the items on the board. It is the main limitation here, and the reason for the
    board-mix replication below.

## 3. Does it reproduce the board?

Board-mapped PCS for each pool. The tau columns cover the 8 subjects with a private PCS; the 6 anchors are the least-squares
basis.

| pool | rows (subjects) | tau, 8 | tau, 6 anchors | RMSE, anchors | JTN-s1 | JTN-s1 + stop3 | garage | go-straight |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| official (all 33 complete) | 33 | +0.29 | -0.07 | 235 | 1617 | 1874 | 1174 | 1077 |
| seeds 1 / 2 / 3 | 33 | +0.29 | -0.07 | 234 | 1623-1675 | 1848-1926 | 1178-1185 | 1072-1076 |
| 1PL (fix_discrimination) | 33 | +0.36 | +0.07 | 219 | 1587 | 1683 | 1275 | 1035 |
| no third-party | 29 | +0.29 | -0.07 | 236 | 1645 | 1964 | 1168 | 1083 |
| dedup (one per family) | 12 | +0.29 | -0.07 | 240 | 1020 | 1005 | 1117 | 1143 |
| all 33 + garage to 30 rows | 62 (33) | +0.57 | +0.47 | 162 | 1965 | 2198 | 1502 | 1026 |
| **board-mix** | 51 (8) | **+0.71** | +0.47 | 145 | **1251** | **1333** | **1553** | 1035 |
| board-mix, seeds 1 / 2 | 51 (8) | +0.86 / +0.71 | +0.73 / +0.47 | 142 / 144 | 1249 / 1253 | 1335 / 1336 | 1556 / 1553 | 1044 / 1034 |
| board-mix, garage x15 | 36 (8) | +0.71 | +0.47 | 143 | 1322 | 1446 | 1554 | 1030 |
| board-mix, garage x60 | 81 (8) | +0.86 | +0.73 | 142 | 1198 | 1261 | 1558 | 1050 |
| board-mix 1PL | 51 (8) | +0.36 | +0.07 | 208 | 1635 | 1703 | 1328 | 1012 |
| **board** | 130 | | | | **1325** | **1463** | **1543** | **1300** |

How to read it:

- **Composition decides the result, not subject count or seed.** Every pool where our family is the majority puts our stop3
  variants 300-800 PCS above the garage. That holds for 1PL and seeds; at 1874-2198 the top abilities are prior-bounded.
- **Pools shaped like the board put the garage on top.** The board-mix pools order garage > stop3 > native, as the board does.
  Seed noise is ±3 PCS. The garage weight moves the probes by about ±100 (stop3 1261-1446), and the anchors cannot pin that
  weight.
- **1PL loses the pattern even with the board's composition.** In the board-mix 1PL pool both JTN reads land about 300 above
  the garage. Per-item discrimination is the mechanism.
- **The arena mean ranks the 8 just as well:** tau +0.71, the same as the board's own mean scene score against its PCS. Local
  PCS beats the mean on ordering the anchors only in two of the board-mix fits.
- **Go-straight is underpredicted in every pool** (about 1035 against 1300). The private suite is richer in static windows
  than the arena (d325), and go-straight fails the arena's launch / turn windows (0.53 / 0.13). No local pool fixes this.
- **VI artifact in the own-family pools.** The fitted posterior mean of the top abilities stops near 4.0, while the MAP under
  the same items is higher (up to +3.3). In board-mix the difference is at most 0.18. Counterfactuals use the MAP with items held
  fixed (section 5).

## 4. Item parameters by stratum

These are the fitted item parameters per cell. Cells are the arena strata, with static split by a lead within 8 m at the window
start. Full tables are in the reports; median discrimination a is shown.

| cell | windows | a, official pool | a, board-mix | share a < 0.5 (board-mix) | garage mean | ours mean | JTN-s1 | JTN-s1 + stop3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| static-free | 376 | 0.94 | 1.73 | 0.16 | 0.835 | 0.632 | 0.686 | 0.995 |
| static-lead | 457 | 2.54 | 2.02 | 0.06 | 0.937 | 0.924 | 0.991 | 0.998 |
| launch | 181 | 0.45 | **0.13** | 0.60 | 0.526 | 0.822 | 0.853 | 0.746 |
| turn | 203 | 2.11 | **0.15** | 0.54 | 0.476 | 0.875 | 0.898 | 0.889 |
| braking | 346 | 2.38 | 1.85 | 0.33 | 0.815 | 0.917 | 0.925 | 0.948 |
| cruise | 113 | 2.88 | 2.57 | 0.10 | 0.967 | 0.985 | 0.983 | 0.981 |
| other | 430 | 2.84 | 2.42 | 0.24 | 0.850 | 0.937 | 0.936 | 0.941 |

In the board-mix pool, discrimination tracks the garage: Spearman of a against the garage's score is +0.75 over items (official
pool +0.42).

- **Turns and launches, our strengths, earn almost nothing there.** The garage fails 106 of 203 turns and 35 of 181 launches
  (another 74 launches partial), so those items have a < 0.5.
- **Static windows carry real discrimination** (a 1.7-2.0), because the garage aces them.
- **The opposite holds in our own pool.** Static-free gets the lowest a after launch (0.94, against 2.1-2.9 for the moving
  cells), because our majority fails it. That is hypothesis (a) of the brief, and it holds in whichever pool is the majority.

## 5. Decomposition of our deficit

Board-mix pool, items fixed, one subject's ability re-estimated by MAP. One submission among 130 barely moves the items.

**JTN-s1 + stop3, PCS 1324, 229 below the garage.** Replacing its scores on one cell, with every other cell unchanged:

| cell | ours 0 / partial / 1 | garage 0 / partial / 1 | take garage's scores | max(ours, garage) | zeros -> 1 | partials -> 1 |
|---|---|---|--:|--:|--:|--:|
| other | 15 / 62 / 353 | 52 / 50 / 328 | +54 | +66 | +5 | +59 |
| braking | 16 / 28 / 302 | 58 / 62 / 226 | +15 | +25 | +4 | +21 |
| cruise | 0 / 18 / 95 | 3 / 8 / 102 | +15 | +16 | 0 | +16 |
| turn | 18 / 22 / 163 | 106 / 3 / 94 | +11 | +22 | +11 | +11 |
| launch | 5 / 74 / 102 | 35 / 74 / 72 | +10 | +18 | +1 | +18 |
| static-free | 2 / 0 / 374 | 62 / 0 / 314 | -6 | 0 | 0 | 0 |
| static-lead | 1 / 0 / 456 | 29 / 0 / 428 | -3 | 0 | 0 | 0 |

By outcome class, with each class set to 1:

| outcome | windows | mean points | PCS gain | per window | garage = 1 on |
|---|--:|--:|--:|--:|--:|
| partial, slow progress (moving windows) | 105 | 16.8 | **+145** | +1.38 | 60 |
| 0 offroad | 50 | 50.0 | +21 | +0.43 | 11 |
| partial, launch window (hold cost: late / no launch, creep, rear-hit) | 74 | 40.9 | +18 | +0.25 | 9 |
| partial >= 0.98 | 17 | 0.2 | +10 | +0.59 | 11 |
| partial, struck from behind | 8 | 2.1 | +6 | +0.72 | 5 |
| 0 corridor, moving window | 6 | 6.0 | +1 | +0.11 | 0 |
| 0 at-fault collision | 1 | 1.0 | +1 | +0.65 | 1 |

The marginal value of a single window is about +1 PCS where the garage scores 1, and about 0 where the garage scores 0 or a
partial. That holds for both our zeros and our partials (median +1.3 / +0.9 / +0.8 for an exact 0 / partial < 0.9 /
partial >= 0.9 on garage-1 windows). Gains are superadditive: each fix raises theta, which makes the remaining partials more
expensive. Combined counterfactuals:

| JTN-s1 + stop3 | windows | mean after | PCS, board-mix | PCS, garage x15 |
|---|--:|--:|--:|--:|
| as read | - | 0.944 | 1324 | 1433 |
| slow-progress partials -> 1 | 105 | 0.952 | 1469 | 1656 |
| + partials >= 0.98 and rear-struck -> 1 | 130 | 0.954 | 1575 | 1867 |
| + offroad zeros -> 1 | 180 | 0.977 | 1742 | 2190 |
| launch-window partials only -> 1 | 74 | 0.964 | 1342 | 1460 |
| every zero only -> 1 | 57 | 0.972 | 1347 | 1473 |

**JTN-s1 native, PCS 1247.** Setting the 115 static-window corridor zeros to 1 is worth +82 (+119 with garage x15); the garage
aces 82 of those windows. Slow-progress partials are worth +39 / +41, offroad zeros +15, and launch partials +2. All 193 zeros
together are worth +129 / +202. The native-to-stop3 step comes out at +82 / +124 locally against +138 on the board. stop3 bought
the static windows and paid with 31 more launch partials, which cost only 18 because launch has low discrimination.

The official own-family pool tells the same story at larger scale. For stop3, slow-progress partials -> 1 is worth +398 and
launch partials +97; for native, static corridor zeros are worth +389. The ordering of the levers is the same in every pool.

## 6. Conclusions

- **What to optimise for PCS: exact 1s on moving windows that strong policies ace.** In practice, progress of at least 0.8 of
  the human's on braking / other / cruise / turn windows. That is the slow ratchet of decisions 372 / 375 (plan speed ratio
  0.89 over 3.5 s for JTN1). It costs only 0.008 of mean, but it is worth about 145-220 PCS, which is the whole gap to the
  garage. Near-misses (0.98-0.999) also count. Zeros on windows strong policies fail, and our turn / launch advantages, barely
  move PCS.
- **Static windows.**
  - With stop3, nothing is left to gain there.
  - Natively, standing still at the rest point is worth +80-120. That is the same target as decision 372, valued at about
    +0.03 of mean.
  - A training fix for slow progress that also adds launch partials is cheap in PCS (launch has a ≈ 0.1).
  - A fix that turns moving-window 1s into 0.9s is very expensive.
- **Selector.**
  - Keep the arena held mean as the primary ruler. A local PCS needs a pool shaped like the board, which we can only imitate,
    and its calibration rests on 2 out-of-sample probes and a garage weight the anchors cannot fix.
  - Report the **frozen board-mix PCS** next to every arena held read (`pcs_local.py score`; items fixed, so candidates do not
    move each other).
  - Also report the count of non-1 scores on windows the garage aces.
  - When arena means overlap within CI, prefer the higher board-mix PCS.
  - **Subject count is not the constraint; composition is.** Adding our 25 other subjects to board-mix moves stop3 from 1333
    to 1732 and lowers tau from 0.71 to 0.64. A 12-subject deduplicated pool gives tau 0.29.
  - The pool needs our family to be a small minority (here 4 of 51 rows) and the strong-host pattern to be heavily weighted.
  - The one subject worth adding would be a second host-like garage checkpoint on the 18 held arena chunks, if one ever
    becomes public. None is now, so no read is proposed.

## Commands

```bash
# box (repo root): rows of every arena read root
P=$DATA_DIR/third_party/alpasim/.venv/bin/python
$P experiments/alpasim_proxy/scripts/arena.py collect --reads <root> --out $DATA_DIR/runs/alpasim_proxy/pcs_local/rows_<i>.csv.gz
# Mac: isolated env, then build / fit / report
uv venv -p 3.11 tmp/pcs_local/.venv && VIRTUAL_ENV=tmp/pcs_local/.venv uv pip install <drive-irt 82ddd5a src> pandas tabulate
X="tmp/pcs_local/.venv/bin/python experiments/alpasim_proxy/scripts/pcs_local.py"
$X build --rows tmp/pcs_local --out tmp/pcs_local/data
$X fit --data tmp/pcs_local/data --out tmp/pcs_local/out            # 15 pools, about 70 s each on CPU
$X subcurve --data tmp/pcs_local/data --out tmp/pcs_local/out
$X report --data tmp/pcs_local/data --out tmp/pcs_local/out [--primary boardmix-g15|official]
$X score --rows <arena collect of a new held read>                  # frozen board-mix PCS of a candidate
```
