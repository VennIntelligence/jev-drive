# VHIST: the slowness causal split at serving time (held 460 at 18 decisions), 2026-10-13

Prereg: [plans/2026-10-13-vhist-prereg.md](../../plans/2026-10-13-vhist-prereg.md) (5f4ee0ba, before any read). Switch: `JEV_VHIST`
([experiments/alpasim/lib/vhist.py](../../../alpasim/lib/vhist.py), nuPlan and PAI drivers, default off). Reads: box
`runs/route_mem/read/{RSG-L18-s0, RSG-L18-s0-vhold, RSGallV-L18-s1, RSGallV-L18-s1-vzero}` (`READ_ENV=JEV_VHIST=<m> READ_SFX=<s> rm_chain.sh
submit-read`). Tables: `vhist.md` / `vhist.json` (`rf_vhist.py report`), `deficit.md` (rf_deficit, unchanged). Held 460 is training-side
now: a mechanism read only.

Smoke (5 held scenes of c0, RSG-L18-s0): switch off = 90 / 90 plans bit-identical to the stored read; `hold` / `zero` 0 rule mismatches over 90
logged records each (`rf_vhist.py same / check`).

| label | mean | zeros (coll / off / corr) | GT (m) | rear-ended | progress score | deficit median / mean of rollout means (m) |
|:--|--:|:--|--:|--:|--:|:--|
| RSG (RSG-L18-s0, stored) | 0.8656 | 34 (4 / 17 / 6) | 1.135 | 38 | 0.902 | -0.258 / -0.257 |
| RSG + hold | 0.8141 | **70 (39 / 17 / 14)** | **1.738** | 16 | 0.930 | -0.068 / -0.036 |
| allV (RSGallV-L18-s1) | 0.8785 | 40 (5 / 17 / 10) | 1.467 | 16 | 0.927 | -0.106 / -0.080 |
| allV + zero | 0.8710 | 40 (2 / 17 / 13) | 1.162 | 27 | 0.918 | -0.176 / -0.161 |

Lines: **M fails** (deficit median -0.068 >= -0.25 passes, but zeros +36 [+19, +58] against <= +5 and GT +0.603 m [+0.41, +0.83] against
<= +0.05). **S: share removed 0.86** (>= 0.60: the serving hold removes as much shortness as the training-side drop or more; allV removes
0.69 on these scenes). **Z: zero does not help** (deficit -0.081 [-0.126, -0.040], i.e. shorter again; zeros +0; GT -0.305 m [-0.45, -0.16]).

Reading: the mechanism is confirmed: holding only the fed speed history (frames unchanged) takes out 86 % of the plan shortness and cuts
rear-end truncations 38 -> 16, so the ratchet runs through that input. But the hold as specified is unsafe: all 39 collision zeros are
front collisions (RSG: 4), i.e. the car runs into its lead; the hold had no lead release (B's third release, left out to avoid a second
switch) and was active on 31 % of decisions (median lift 0.7 m/s, p90 2.7; released for a turn on 55 %, for a stop plan on 1 %). Zeroing on
RSGallV is the other direction: it returns allV's GT to RSG's level (1.162 vs 1.135) but gives back half of allV's deficit gain.
Note on the baseline: the prereg's -0.433 is the public 1491 / 10-decision median; RSG's held 18-decision median is -0.258, so line M's
deficit threshold was already nearly met by the reference; S (shares on the same scenes) is the comparison that carries.

PAI (prereg rule): M fails and Z does not help -> **no VHIST run on the PAI exam**. A next arm, if wanted: hold + a lead release (JEV_LEAD's
filtered lead inside openpilot's following distance) on the same held read before any PAI time.
