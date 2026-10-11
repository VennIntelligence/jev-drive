# onpol_pai: on-policy self-consequence rows, PAI track

status: live
decisions: 261 (inputs: 219, 220, 226, 235, 237, 241, 247, 259)
index: stopped before the 281 read (user, 10-12 02:24 JST); open loop only: SELF term rate 0.215 vs 0.307

**Question.** At the states the frozen-encoder driver `P2H10S-F-s0` visits in its own AlpaSim PAI rollouts, do labels of the consequence
of its own plan (ON-SELF) do better on the PAI board than the path back to the log at the same states (ON-IMIT, RoaD / DAgger)?

**Conclusion.** No closed-loop answer: the 281-scene read was stopped by the user's decision on 2026-10-12 02:24 JST with 144 of 281
scenes of ON-SELF rolled and none of ON-IMIT or the fine-tune control; nothing of it was read (decision 261). Open loop on 1 977 held
rows, one seed: ON-SELF lowers the summed consequence-term rate from 0.307 to 0.215, ON-IMIT to 0.204 without being trained on the terms.

**What exists.** Checkpoints `P2H10S-SELF-ON-F-s0`, `P2H10S-IMIT-ON-F-s0`, `P2H10S-FT-F-s0` (Tokyo, `/data/runs/op_parity/runs/`);
160 training scenes rolled on-policy (31 772 rows) and 80 along the log (15 884 rows) with labels from the simulator
(`/data/runs/onpol_pai/{pass1,pass2,rows}`); ON-BOTH stopped at the pilot (navtrain dev ADE 0.586, line 0.575).

**Standing, stated plainly.** The input-path gate G-id failed as registered (24.4 % of rows over 0.03 m) and after one amendment
(5.28 %); training went ahead on a judgement that the residual is half-precision rounding at PAI speeds. Three terms of ON-SELF were
amended after an independent audit and before any training (corridor lateral-only, progress guard restricted, road-edge reach). All of
it is in the pre-registration's "As run".

**Next.** Only with the user's word: restart the queue from `/data/runs/onpol_pai/read281` (finished chunks are kept; command in
`PAUSED.md` there), then `scripts/onp_lines.py` for lines H and V. The Tokyo box is reserved for another lane (10-12).

**Read more.** [plans/2026-10-11-onpol-pai-prereg.md](plans/2026-10-11-onpol-pai-prereg.md) (design, lines, "As run" with every gate
and deviation), [results/rows_pol.md](results/rows_pol.md) (row composition), [results/rows_format.md](results/rows_format.md).

<!-- files:begin -->
## Files

- `2026-10-11-onpol-pai-prereg.md` (plans): ONPOL-PAI pre-registration
- `onp_lines.py` (scripts): the pre-registered lines H and V on the …

[results/](results/) 7 result files · [plans/](plans/) 1 live plans · [lib/](lib/) 3 library · [scripts/](scripts/) 12 entry points
<!-- files:end -->
