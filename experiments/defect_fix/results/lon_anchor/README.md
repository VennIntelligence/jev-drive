# LON-ANCHOR: station-anchored two-sided longitudinal term (A), and the SIMP recipe (B) (2026-10-20)

- Prereg: [../../plans/2026-10-20-lon-anchor-prereg.md](../../plans/2026-10-20-lon-anchor-prereg.md). Each part was committed before its
  outputs: base 0331347b / 9bafbe89, supplement 1 1338aef1 (+ addendum ebceff90), supplement 2 ebceff90, supplement 3 e91535e9.
- Code:
  - `lib/nr1_anchor.py`: the label rule and the term (`rm_train --on-anchor W`).
  - `lib/nr1_simp.py`: the merged region and clear hinges (`--simp-region`, `--simp-clear`), wired through `nr1_lat.attach`, which
    `on_train.setup` calls.
  - `rm_train --flat-brk FILE` (SIMP-F).
  - Lane scripts: `scripts/lon_anchor.py`, `lon_anchor.sh`, `simp.py`, `simp.sh`.
  - Every switch defaults to off and is bitwise identical when off.
- Box: `$DATA_DIR/runs/defect_fix/lon_anchor/` and `.../simp/`. Labels are in `$DATA_DIR/runs/op_parity/cache/on1_JTCP/lon_anchor_v2.npz`
  (v1 `lon_anchor.npz` is the registered rule, kept).

## A: labels and gates

- **Label (supplement 1).** The ego is at station s_e on the human's logged path, and t_h is the first time the human was there. The
  target is `s_t,j = S_h(t_h + 0.5 j) - s_e` for j = 1..8.
  - Plan progress is the station of each plan pose projected on the human's path, minus the origin's station.
  - The loss is smooth L1 (beta 0.5) over the 8 poses, two-sided, with gradient only along each pose's heading.
  - Two exclusions:
    - **lead**: the ego's lead is closer than the human's lead was at t_h by more than 0.5 m (the human's lead unknown counts as
      a skip).
    - **clock**: the target moves >= 2 m in 4 s while the human stood at the same clock time, or the target ends standing while
      the human was moving at that clock time.
  - Counts are in [labels_v2.json](labels_v2.json).
    - The first table counts S rows and is computed from that file; the second table counts S rows that have a target, from
      [gate_v2.md](gate_v2.md).
    - Label outcome and clock exclusions per pool:

      | pool | S rows | labelled (ok) | lead | lead_unknown | clock_go | clock_stop | log_end | no_reach |
      |---|--:|--:|--:|--:|--:|--:|--:|--:|
      | nuPlan | 87 012 | 77 479 | 6 091 | 0 | 1 885 | 1 486 | 24 | 47 |
      | PAI | 84 042 | 56 554 | 10 564 | 164 | 4 620 | 1 120 | 8 249 | 2 771 |

    - Clock exclusions by row kind:

      | rows | launch | stop | turn | move |
      |---|--:|--:|--:|--:|
      | nuPlan clock_go | 254 | 1 067 | 79 | 485 |
      | nuPlan clock_stop | 1 252 | 234 | 0 | 0 |
      | PAI clock_go | 1 378 | 1 046 | 267 | 1 135 |
      | PAI clock_stop | 127 | 743 | 0 | 0 |

    - PAI clock exclusions by scene class (clock_go / clock_stop): launch 1 378 / 127, stop_nolead 1 911 / 834, turn_sharp 1 328 / 159,
      bend 3 / 0.
- **G1.**
  - As registered it FAILs: 4.65 % against a 1 % line ([gate.md](gate.md)). The test plan jumped onto the human's line, and that lateral
    jump counted as arc.
  - As amended in supplement 1 (the human's path started at the ego, projected progress) it PASSes: 0.102 % (nuPlan 0.037 %, PAI 0.191 %)
    ([gate_v2.md](gate_v2.md)).
- **G4 (switch off).** Bitwise identical. Control command, 60 steps eager, 40777a4e against ebceff90 ([identity.json](identity.json)).
- **Stored plans against the target** (reported, not gated):
  - The target is longer than the plan (the term pushes faster) on 73 % of nuPlan rows and 54 % of PAI rows. It is shorter on 20 % / 40 %.
  - On PAI stop_nolead it is longer on 39 % and shorter on 54 %.
  - With an ego lead within 15 m: nuPlan 51 % longer / 36 % shorter; PAI 27 % / 61 %.
- **Draw LONP** ([w_LONP.json](w_LONP.json)).
  - 16 nuPlan S rows + 16 curated PAI rows per batch, from 480 cp scenes.
  - PAI 281 and practice-pai (val160) hit 0 rows.
  - `pai_fix/scenes/exam_scenes.txt` (118) and `train_exclude.csv` (216) hit 0 cp rows.
  - Five cp scenes (780 rows) linked through `drives_val160.csv` are also excluded. That is a superset of the two required files.
  - The first chain's draw read `onbox_train_20261020.txt`, an inventory of training scenes, as an exclusion list, which dropped every cp
    row. That draw was cancelled before any training finished and rerun (supplement 1 addendum; `*_nocp.json`).
- **Dose.** The 0.1 gradient rule gave W1 = 0.0011 (median g_anchor / g_imit = 0.352 at W 0.004) ([calib.json](calib.json)). Doses are
  0.00055 / 0.0011 / 0.0022.

## A: pilots (1 000 steps from NTV0-L18-s0, seed 0) and the inertia read

Checkpoints are in `$DATA_DIR/runs/op_parity/runs/<tag>/ckpt-final.pt`:
- `LONC1K-L18-s0` (control)
- `LONAH1K-L18-s0` (x0.5)
- `LONA1K-L18-s0` (x1)
- `LONAD1K-L18-s0` (x2)

**Registered read** ([inertia.md](inertia.md); full cells in [inertia_report.md](inertia_report.md)). All three doses FAIL.

| tag | STOP ratio | TURN ratio | OPEN v24 slope | STOP plan has a stop | v24 - human, STOP / OPEN (m/s) | plan stop station - human rest (STOP, m) |
|---|--:|--:|--:|--:|--:|--:|
| control | 0.891 | 0.926 | 0.778 | 0.032 | +0.49 / +0.31 | -0.65 |
| x0.5 | 0.890 | 0.923 | 0.782 | 0.027 | +0.63 / +0.54 | -0.46 |
| x1 | 0.889 | 0.915 | 0.781 | 0.027 | +0.71 / +0.65 | -0.21 |
| x2 | 0.885 | 0.915 | 0.766 | 0.011 | +1.09 / +0.82 | +0.85 |

What the read shows:
- The term does not make the plan react to the stop or curve cue. The stop and turn ratios stay at the control's level.
- It makes the plan faster everywhere, more so at higher dose. At x2 the plan's stop point on stop approaches moves past the human's rest
  point, and the share of plans that contain a stop falls.
- The nuPlan push-faster tilt dominates: 73 % of labelled nuPlan rows have a longer target.
- The term's loss does not fall over the 1 000 steps (x1: 0.235 to 0.236 per row), while its gradient stays near 0.1 of the imitation
  gradient. Dev ADE: 0.536 for the control, 0.537 at x1, 0.548 at x2.
- Training: about 3-4 min per 1 000-step pilot.

## B: SIMP (supplement 2) and SIMP-F (supplement 3): built, not trained

- **Arms.**
  - `SIMPN1K-L18-s0`: the control, the NTV0 recipe from RB0-F-s0.
  - `SIMPA1K` / `SIMPB1K` / `SIMPC1K-L18-s0`: SIMP at PAI share 0 / 0.5 / 1.0.
  - `SIMPF1K-L18-s0`: SIMPB plus the flattened history.
  - All are 1 000 steps on on1_JTCP, with draws [w_SIMPW00.json](w_SIMPW00.json), [w_SIMPW05.json](w_SIMPW05.json),
    [w_SIMPW10.json](w_SIMPW10.json) (same exclusions as LONP).
  - Launch with `simp.sh arms <LON dose>`.
- **Switch-off identity.**
  - 1338aef1 against ebceff90 without the SIMP flags: identical ([simp_identity.json](simp_identity.json)).
  - d4f3b4f8 against e91535e9 without `--flat-brk`: identical ([simp_identity_F.json](simp_identity_F.json)).
- **Human path as the plan: FAIL as registered** ([simp_gate.md](simp_gate.md)). The line was 2 % each.
  - Merged terms: region 7.62 %, clear 3.94 %.
  - The terms they replace, on the same rows: road hinge 11.69 %, lat_onc 1.21 %, agent hinge 2.21 %, lat_side 2.41 %.
  - So region fires less than the road hinge it replaces, and clear about as often as agent + side together. The 2 % line sat below
    the current recipe's own level.
  - `simp.sh train` refuses until the main session decides.
- **Trigger rates of the clear brake gradient on stored plans** (the lead part, whose along-heading gradient may only shorten the plan):
  7.41 % of S rows (nuPlan 7.71 %, PAI 7.10 %).
- **SIMP-F flattened history** ([simp_flat.json](simp_flat.json)).
  - 12 850 brake-needed navtrain tokens (lead_break nav, 781 logs). 10 750 of them already show deceleration; 2 100 are initiation rows.
  - 12 651 of the 101 499 navtrain training rows match.
  - In the smoke run about 9 rows per step (of 83 navtrain rows) were flattened.
- **Smoke.** 60 steps of the F command PASS ([simp_smoke.json](simp_smoke.json)). The region, clear, LON-ANCHOR and flat-history scalars
  are all present and finite.
- **After the pilots** (queued by `simp.sh arms`): inertia on all five arms, `lead_break.py cf --name simp` on N / B / F, and the
  supplement 3 rule. That rule needs BRK stays-behind SIMPF - SIMPN >= +0.10 with the CI low > 0, and |FOL change| <= 0.05.
