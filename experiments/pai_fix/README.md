# pai_fix: targeted PAI exam for stop and turn fixes

status: live
decisions: none
index: PAI-FIX exam: val160 STOP 12 / STOP-L 11 / TURN 37 + guard 12, T281 column 46; ~0.4 card-h a read

**Question.** Can a small PAI closed-loop exam tell, in well under 0.5 card-hours a read, whether a candidate checkpoint fixes NTV0-L18-s0 +
ACC + GAP's two main PAI losses: driving past the human's stopping point (d381: 23 zeros) and entering route turns too fast and under-turning
(11 straight-at-turn plus junction / curve offroad)?

**Conclusion.** (filled in below once the reference reads are in)

**Next.** Screen the ~3 station-anchored variants with `pai_fix.sh read <tag> vl` and `... ag`; send 1-2 to the full PAI 281 read.

**Read more.** [plans/2026-10-20-pai-fix-prereg.md](plans/2026-10-20-pai-fix-prereg.md) (selection rules, metrics, read plan; written
before any exam read), [results/calib/calib.md](results/calib/calib.md) (stored-read calibration), [results/reference.md](results/reference.md).

<!-- files:begin -->
## Files

- `README.md` (README.md): targeted PAI exam for stop and turn …
- `2026-10-20-pai-fix-prereg.md` (plans): PAI-FIX 预登记：定向 PAI 小考（停车越点 + …
- `exam.csv` (scenes): 
- `train_exclude.csv` (scenes): 
- `pai_fix.sh` (scripts): a small targeted PAI closed-loop exam
- `pai_fix.py` (scripts): the targeted PAI exam
- `calib.md` (results): PAI-FIX calibration on stored PAI 281 …

[results/](results/) 2 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
