# long_oracle: longitudinal operating point and collision class

status: concluded
decisions: pending (draft in results/long_oracle.md; inputs: 196, 203, 226, 232, 241, 247, 249-251)
index: no speed scale beats served on navtest (0.95 ties, 1.05 -0.33); target class A (41% arm-specific)

**Question.** Is there a longitudinal operating point for the current driver (`P2H10S-F`) better than the one it serves, and which collision
class should a longitudinal fix target?

**Conclusion.** No: on navtest no uniform speed scale 0.7-1.15 beats the served plan (0.95 +0.02 [-0.13, +0.18] / -0.04 [-0.21, +0.12], 1.05
already -0.30 / -0.37) and a speed-band operating point cross-fitted over logs is +0.14 / +0.10 with CIs including 0, so by the registered stop
rule no PAI sweep was run; the class to target is A, a stopped or slow vehicle ahead (41% [32, 51] of NC failures are A failures WA-JEPA passes),
with "should not launch" (A1, t0 < 2 m/s) outside any speed-scaling fix (draft decision in the results doc).

**Read more.** [results/long_oracle.md](results/long_oracle.md); [plans/2026-10-11-long-oracle-prereg.md](plans/2026-10-11-long-oracle-prereg.md);
the method is decision 196's ([../op_parity/results/nc_taxonomy.md](../op_parity/results/nc_taxonomy.md)), its script reused through `NC_TAX_*`
overrides.

<!-- files:begin -->
## Files


[results/](results/) 44 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
