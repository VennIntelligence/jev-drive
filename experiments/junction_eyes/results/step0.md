# junction_eyes step 0: gate read (registered line NOT read)

Pre-registration: [plans/2026-10-12-prereg-A.md](../plans/2026-10-12-prereg-A.md) (amendments 1-4). Gate read `step0/read/20261012-212924`
on the box, copied to [step0_gates.json](step0_gates.json). The read stops at the gates: no R-A / R-B / R-C quantity and no Z-0 line value
has been computed.

| gate | result | registered line |
|:--|:--|:--|
| G-lab (i) coverage, turns, s = 20 | 0.945 navtest-16 / 0.943 held-452 / 0.973 zero-98: pass | >= 0.9 (first label pass 0.81 / 0.83 / 0.80 failed; labeler fixed, amendment 4) |
| G-lab (ii) sign, straight, s = 20 | L > C > R on all three sets: pass | ordered |
| G-lab (iii) navsim side vs AlpaSim loader, 20 scenes | median 0.000 m, p90 0.010 m, R-B cells 100 %: pass | < 0.2 m |
| G-split | 0 overlap at token / segment / drive log / unit: pass | 0 |
| **G-repro** (old ruler on P, 2 seeds) | held-452 turn 20 m **0.934** (seeds 1.074 / 0.993): in range; navtest near **1.107** (seeds 1.314 / 1.063): **below [1.126, 1.398]**: **FAIL** | both in range |
| G-ref (WA-Cf re-extraction) | pending (WA jobs re-queued after a host-memory stall) | mean cosine >= 0.999 |
| G-dump (executor 2) | pending (expected about 00:00) | fp16 noise |

G-repro fails in the "better than rp_real" direction: the old ruler retrained on P (J_nav + J_on, junction-enriched) reads the navtest
near-junction bucket 0.02 m under the registered interval, with a 0.25 m spread between the two seeds. As registered, the step-0 line is
not read; the decision is with the coordinator / user.
