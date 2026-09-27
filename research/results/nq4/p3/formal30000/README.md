# P3 formal scene 000, default 30000 iterations

The complete registered scene-zero chain finished normally at 11:12:57 CST on 2026-09-27. Training, real/plus/minus rendering, indexing, both openpilot models, finalize, exam and report all returned rc0. There is no active P3 owner or pending executable GPU step.

**The registered primary gate failed:** Cinque null false flips are 3/23 = 13.043%, above the unchanged 7% threshold, using the unchanged I3 tau 0.5268521547 m/s. Lebowski is 0/23, but is not a replacement for the registered primary examinee. Scenes 1–9 were not launched and no GO was written.

Full-image PSNR is 29.894 dB (registered minimum 25 dB); pedestrian-box PSNR is 26.989 dB. All four requested nodes were removed with zero missing; repeated-view max absolute difference is 0.0. All ten prior-reproduction checks are below the original 1e-3 tolerance (maximum 0.000480652). These successful technical checks do not override the failed primary null gate.

The four-panel image is provided for human ghosting review; no visual PASS was assigned. meta.json records the real checkpoint, selected tracks and all node mappings. No thresholds, training length, scene selection or learned baseline were altered.

GPU6 became idle because the chain completed. Its old P3 reservation was not reclaimed automatically after the gate stopped further work; the registry now releases both P3 GPUs and CPUs180–189. The controller owner applies this release and clears only the stale P3 scheduling row; GPU1 G/K jobs are untouched. Scientific or human review may decide later work, but the GPUs are not held during that wait.

Source exam: DATA/runs/nq4/p3-exam/20260927-111241. Every formal stage rc was read as0, and config.yaml records num_iters30000. All five collected files match the remote SHA256 digests in SHA256SUMS.json. The PNG passed all chunk CRCs and complete-IEND validation. verdict.json separates technical completion, the primary scientific failure, pending human review and resource release.
