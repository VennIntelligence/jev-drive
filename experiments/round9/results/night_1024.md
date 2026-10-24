# Night log 2026-10-23/24 (clean NTV1, round 9)

Plan (main session, user asleep): step 0 pick the lr variant (P1 vs P1L) -> step 1 command + PAI human rows
(N1 / N2a / N2b) -> step 2 self-rolled DAgger + PAI S rows on the 719 PAI train scenes -> step 3 NTV1-D from shipped
weights; seed 1 + Harmonizer-on only if NTV1-D >= 0.86 (new scorer, Harmonizer off). No images, pushes, submissions,
deletions. Target to beat: R8K-L18-s2, PAI 281 ACC + GAP Harmonizer off, 0.874 new scorer.

- 00:27 Running: P1L (PCL1L-F-s0, card 0), P1 seed 1 (PCL1-F-s1, card 1); cards 2-5 idle.
- 00:27 Method search done (research/method-search/notes.md): gradient-alignment mixing unlikely to work; ranked
  "frontier anchoring" first; flagged that S rows also carry distill_ho (NTV0NS removed both).
- 00:30 Filled idle cards with two diagnostics: S-split (P0-NH hinges off / P0-ND distill_ho off, plan
  plans/2026-10-24-s-split.md) and NTV0 <-> R8K-s2 weight interpolation a = 0.5 / 0.75 (plans/2026-10-24-interp.md).
- 01:40 P1L done (decision 461): PCL1L-F-s0 0.816 new / 0.737 old, gap share 98 %, = NTV0-s0 (-0.002) from one clean
  run; vs P1 +0.023 (CI spans 0). Recipe R := P1L. Open: net/adapter lr drop vs route-branch lr rise not separated.
- 01:45 Step 1 dispatched: N1 (R + heading-all), N2a (+ PAI human rows whole run), N2b (+ PAI human rows from T),
  plan plans/2026-10-24-ntv1-step1.md. Body-schema research (research/body-schema/) running in parallel.
- 01:35 (box) Session restart: resumed step-1, body-schema, P1 s1, S-split, interp agents; hourly check re-created.
- ~01:45 (box) Step 1 submitted (97d889d1, 47 jobs): N1 / N2a / N2b; N2a and N2b differ only before T (PAI human rows
  58/batch pre-T in N2a); post-T both 51 + 13 + 32 S + 32 PAI human (RB0P draw: MV-LC + F1 / F2), pool on1_JTPH with
  NTV1M weights; new off-by-default flags --on-pre-k / --on-pre-wfile / --on-mix-log. ~6.5 GPU-h.
- Body-schema agent asked to pause new jobs and report an honest prior-art map (user concern: repackaged old methods).
- Body-schema verdict: existing method + small change (ThinkTwice / iPad / guidance), already tried here (223-227, 323,
  324). Queued bsl eval / report jobs cancelled (user); bsl-train-f0..3 left to finish. Old agent stopped; fresh
  agent: reafference cancellation (copycat / causal-confusion literature first, then CPU erasure check).
- P1 seed 1 done (decision 462): PCL1-F-s1 0.791 new / 0.733 old; P1 reproduces (s1 - s0 -0.002); 2-seed P1 0.792 vs
  2-seed NTV0 0.784 new (+0.007 [-0.025, +0.040]), 0.725 vs 0.743 old. NTV0-s1 is weak on the new scorer (0.751;
  its new-scorer facts were imputed, behavior_stats note) - treat the NTV0 2-seed new mean with care.
- Interp done (decision 463): INTP75-NR (0.25 NTV0-s0 + 0.75 R8K-s2, no training) 0.879 new / 0.814 old vs R8K-s2
  0.872 / 0.807 (+0.007 [-0.017, +0.032]); negative flips on NTV0 full-score scenes 14 -> 8, keeps 33/40 R8K fixes.
  INTP50 loses most of the gain (0.836). Single read each; not a clean-lineage candidate, but the strongest local
  number so far and the baseline any anchoring method must beat (< 8 flips at ~R8K mean). For the user in the morning.
- Copycat / reafference literature survey (sub-agent of the stopped erasure task) done: erasure = Wen 2020 / Chuang
  2022 with another remover, and erasing ego speed has a logic hole (speed is needed) -> consistent with the user's
  objection. Passed to the binding agent (research/body-schema/binding.md).
- Step 1 try 1 stopped at a gate: N2a composition smoke, pre-T PAI share 0.9999999999949 vs check tolerance 1e-12
  (check-script bug; all other gates passed). Tolerance fixed to 1e-9 (f6a0d360), try 1 kept in
  ntv1_step1.try1/, all 47 jobs resubmitted. No training / read GPU time lost.
- S-split done (decision 464): the consequence hinges carry the S gain. NH (hinges off, distill_ho kept) 0.716 new,
  -0.087 vs P0, = NTV0NS level (19 % of the gap kept); ND (distill_ho off, hinges kept) 0.819 new, +0.017 [-0.015,
  +0.049] vs P0, route-turn zeros -6. Supports "body schema = consequence hinge"; NTV1 may drop distill_ho on S rows
  (candidate change for a later step, one seed so far).
- Binding study done (research/body-schema/binding.md, CPU only, uncommitted): representation DOES carry ego state
  (speed R2 0.989) and own executed action (inverse-dynamics R2 0.785); the break is cue -> plan (lead head reads
  ~85 % of a pushed braking cue, plan moves ~19 % of the human response) and speed x cue binding (near stops the plan's
  4 s end depends on speed 2.9 / stop distance 0.02-0.08; human 0.01 / 0.985; kappa 0.72-0.86 vs 0.003), same in all
  5 models. Proposals (not run): cue-anchored longitudinal readout (existing technique moved inside the net), speed
  twins in token space (incremental). Left for the user: stop / lead longitudinal classes are worth little on the new
  PAI scorer, so this is a method question, not tonight's scoring lane.
- 02:17 (box) Step 1 trainings at 36 min on cards 0-2; cards 3-5 idle 27 min. Started step 2a early on them: roll
  P1L (clean, decision 461) on the 719 PAI train scenes with the submission serve and build a clean self-DAgger pool
  with R6DAG's round-1 teacher (plan plans/2026-10-24-ntv1-step2.md). P1L as student instead of waiting for the
  step-1 winner: saves ~1 h, still self-lineage; teacher fixed to round 1 so step 3 is attributable.
- Step 1 done (decision 465): N1 (P1L + heading-all) 0.797 new, -0.016 [-0.054, +0.022] vs P1L (line -0.015: "costs,
  report"; old scorer -0.012 would pass) -> no measurable cost, no gain. N2a 0.722 / N2b 0.779: PAI human rows hurt
  in both placements (route-turn zeros down 6-8, lateral zeros up 13-28); earlier entry hurts more. PAI human
  imitation rows stay out of NTV1.
- Main-session call for step 3: base = P1L (not N1). Reasons: the command change bought nothing; the step-2a DAgger
  pool was rolled by P1L with P1L's serve; keeping P1L makes step 3 a single attributable change. Arms when the pool
  lands: D1 = P1L + clean self-DAgger rows; D2 = D1 + --ho-distill 0 (decision 464 ND), 1 seed each.
- 04:17 (box) All 6 cards idle 6-36 min; step 2a is in its CPU pool build (dgc-pool), 13 gate jobs queued behind.
  Filled cards with two read-only diagnostics (plans/2026-10-24-diag-reads.md): P1L Harmonizer-ON read, and an
  INTP75 same-checkpoint re-read for the negative-flip noise floor.
- Step 2a done: on1_PAIDC 112 620 rows / 92 561 usable / 719 scenes, P1L as student, round-1 teacher; T1-T4 pass
  (T4 via registered fallback, 2.6 % rows to weight 0); P1L on the 719 train scenes 0.674 old vs R5LC 0.663.
  Two faults in the agent's own merge-check script (fixed, tail rerun). ~114 GiB on box, 622 GB free.
- Step 3 dispatched (plans/2026-10-24-ntv1-step3.md): D1 = P1L + own DAgger rows (fixed batch, option B, 8 DAgger
  rows replace 8 navtrain after T), D2 = D1 + --ho-distill 0. Candidate rule: >= 0.86 new -> seed 1 + Harmonizer-ON.
- 05:17 (box) Step 3 D1 / D2 training (8 min in); P1L Harmonizer-ON reads at 56 min; card 4 idle. Added a read of
  the 10k snapshot PCL1-F-s0x10000 (shared pre-S state of P1 / P1L) to split the single run into offline part vs
  S segment (plans/2026-10-24-snap10k.md).
- Diag reads done (decision 466): P1L Harmonizer ON 0.804 new (ON - OFF -0.009, same size as NTV0 -0.006 / R8K
  -0.007, all CIs span 0); P1L ON vs NTV0 ON -0.005. INTP75 re-read bit-identical on all 278 scenes (Harmonizer off,
  same chunk lists): closed-loop reads are deterministic, so seed / chunk-list noise is the only noise left.
- 10k snapshot read (decision 467): PCL1-F-s0x10000 (= RB0L x10000 bit for bit) 0.736 new; S segment adds +0.057 (P1)
  / +0.081 (P1L); the offline 9 500 steps without S cost -0.038 (RB0L - RB0-F), offroad 23 -> 38.
- 06:17 (box) Step 3 D1 / D2 at step ~10 800 (post-T ~2.4 it/s, ~1 h left); cards 1-5 idle 30 min. Filled with the
  P1L lr split (P1La net/adapter drop only, P1Lb route-branch schedule only; plans/2026-10-24-p1l-split.md).
- 08:17 (box) Step 3 first reads (old scorer, 276 scenes): D1 0.798 (P1L 0.732, +0.066 [+0.025, +0.108]; route-turn
  zeros 10 -> 3, launches 32 -> 36) ~ R8K-s2 0.803; D2 (distill_ho off) 0.746, worse than D1. Candidate rule fired
  (dgt-ratio-cand-s1 running): D1 seed 1 + Harmonizer-ON reads in the lane. Waiting for the agent's new-scorer report.
- P1L lr split done (decision 469): P1La (net/adapter only) 0.795, P1Lb (route branch only) 0.792, both = P1 (0.793),
  ~0.02 below P1L. Either both changes are needed together, or P1L - P1 (+0.023, CI spans 0) was mostly seed spread;
  one seed per arm cannot tell. P1L stays the base (D1 is built on it), but its lr gain is not established.
- 09:17 (box) D1 seed 1 training (84 min); cards 1-5 idle. Added a second-exam check: D1-s0, R8K-s2, P1L on PAI
  val160 (never used for choosing), plans/2026-10-24-val160.md.
- val160 second exam (decision 470, 152 scenes, never used for choosing): D1 0.828 / R8K-s2 0.854 / P1L 0.841 new;
  D1 - R8K -0.026 [-0.071, +0.019], D1 - P1L -0.012 [-0.067, +0.043]; on the 80 never-selected scenes all three
  ~0.90. D1's +0.066 over P1L on test281 does NOT replicate on val160 -> treat D1's 281 gain as unconfirmed until
  seed 1 is in; ordering vs R8K matches 281 (R8K ahead).
- Step 3 done (decision 471): D1 (P1L + own DAgger rows) s0 0.867 / s1 0.837 new, 2-seed 0.852 off / 0.837 on;
  vs P1L off +0.040 [+0.002, +0.081]; vs R8K-s2 off -0.020, on -0.028. D2 (distill_ho off) -0.038 vs D1: with
  DAgger rows distill_ho is needed (opposite of 464 without DAgger). Highway: not slow by speed (median 1.00).
- Night ends 11:20 (box). Summary for the user in the conversation; open calls: submission (1 vennai-org slot left),
  DAgger iteration 2, method direction.
- 11:25 Cards idle after step 3; under the standing "iterate to a stronger clean model" authorisation started step 4
  = DAgger iteration 2 (roll D1-s0 on the 719 train scenes, same round-1 teacher, aggregate with iteration 1, D3 from
  shipped weights; plans/2026-10-24-ntv1-step4.md). Candidate rule: D3 >= 0.875 -> seed 1 + val160.
- 12:38 Step 4a in its CPU cache build; cards idle. Added val160 reads of D1-s1 and INTP75-NR (submission-relevant:
  D1 2-seed and INTP75 vs R8K-s2 on the second exam).
