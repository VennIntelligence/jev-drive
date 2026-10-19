# PAI-PARTIALS: serve-side reach of the rear-hit / slow partials, and a small serve sweep (pre-registration, 2026-10-20)

Written after part 1 (a CPU read-out of stored reads) and before any sweep read. Question: the PAI PCS ruler (decision 392) says
NTV0-L18-s0 + ACC + GAP loses most PCS on 25 rear-hit partials (+168 if all -> 1) and 11 slow partials (+52). How much of that can the
serve layer reach, and which serve configuration is best on the ruler?

## Part 1 (done before this addendum; `scripts/pai_partials.py decomp` / `value`, results/pai_partials/)
Per partial scene of the stored read A0 (`defect_fix/pai_accgap/NTV0-L18-s0`), per inference decision from the handoff to the rear hit
(slow: the rollout's end): deficit = the human's time-matched speed 1.5 s ahead minus the served 1.5 s speed (> 0 only); the part the
plan had and the serve removed (serve-removed, by binding term), the rest is the plan's (split: ACC stop guard, brake guard, launch, cruise).

- Rear-hit: 895 deficit m, serve-removed 32 m (real-lead cut 22, phantom 3, curve 7, set 6); plan 863 m: cruise 581, brake guard 227,
  stop guard 44, launch 34. The ACC set speed (max(handoff speed, plan 2-4 s speed)) is below the human's on 807 of the plan metres.
- Slow: 1020 m, serve-removed 56 (curve 35, lead 13); plan: cruise 578, brake guard 317.
- No scene has serve-removed >= plan metres. PCS worth of the serve-removed share (each scene's loss x serve / deficit): rear-hit +23,
  slow +2. The rest is reachable by the serve only by serving faster than the model asks (set speed, brake guard, fed speed history).
- Caveat: a per-decision split; a cut early in a scene lowers later plans (decision 376), so the closed-loop serve share may be larger.
  The sweep is the test.

## Arms (one seed each, PAI 281, same 8-chunk table as A0: `$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks`)
All: NTV0-L18-s0, base serve `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`, WIZ `controller.gains.rel_front_steering_angle_weight=10`
(the CLI cap), CONC 4, ALPASIM_FAST=1, pool via `pai_box.sh`, root `$DATA_DIR/runs/alpasim_proxy/pai_partials/<arm>`.
New serve knobs (`serve_fix.py`, unset = bit-identical): `JEV_ACC_VSET` (gain on the plan's 2-4 s speed inside v_set), `JEV_ACC_REL`
(stop-guard arc m, brake-guard deceleration m/s^2; default 2.0,1.0), `JEV_ACC_ALAT` (curve limit, default 2.4).

| arm | added to the base serve | acts on |
|---|---|---|
| R0 | nothing (re-read of A0) | read-to-read noise floor |
| HL | `JEV_VHIST=holdlead` (stored read, decision 370; not re-run) | plan cruise (fed speed history) |
| V1 | HL + `JEV_ACC_VSET=1.1` | plan cruise / set speed below the human |
| V1b | HL + `JEV_ACC_VSET=1.2` | same, dose |
| V2 | HL + `JEV_ACC_REL=2.0,2.0` | brake guard (plan brakes where the human drives) |
| V3 | HL + `JEV_GAP=0.5,2.0 JEV_ACC_ALAT=3.0` | the serve-removed terms (real-lead cut, curve) |
| J | HL + VSET 1.1 + REL 2.0,2.0 + GAP 0.5,2.0 + ALAT 3.0 | joint |

Smoke first: J on 2 scenes (logs must show v_set = 1.1 x plan speed where above v_hand, the suffix in the version id); then all arms at once.

## Read-out and selection (one rule)
- Per arm: mean, zeros by type, rear-hit / slow partial counts, paired mean diff vs A0 (`pai281.py`, log-cluster bootstrap 10 000, seed 0,
  RPC-error scenes dropped), local PAI PCS on the d392 ruler (paimix; arms scored out of pool, MAP; scene bootstrap 100 draws as d392)
  and the paired PCS diff vs A0 on the same draws; the 13 pool variants' range.
- Guard: at-fault front + side zeros <= A0's + 2. An arm failing the guard is not recommended whatever its PCS.
- Noise: |PCS(R0) - PCS(A0)| is the read noise N.
- Recommendation: the guard-passing arm with the highest paimix PCS, if it beats A0 by more than N and beats HL by more than N; if no arm
  beats HL by more than N, HL (decision 392's first) stays; if HL itself is within N of A0, the plain A0 serve is equally good.
- Stated limit: the arms are tuned and selected on the same 281 scenes the ruler is fitted on, and every reading is above the top anchor
  (2089); absolute PCS is not reliable, only order and paired diffs are reported as evidence.
