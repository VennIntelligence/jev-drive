# P3 scene 000: registered 3000-iteration technical smoke

Technical pipeline completed on 2026-09-27: train, real/plus/minus render, index, Cinque/Lebowski features, exam and report. These artifacts are isolated from the default 30000-iteration formal scene-zero run, which started afterwards. This is not a feasibility PASS or permission to train scenes 1–9.

The short-run Cinque null false-flip rate is 5/23 = 21.739%, above the unchanged 7% gate, at the unchanged I3 tau of 0.5268521547 m/s. Lebowski is 3/23 = 13.043%. Full-image PSNR is 25.524 dB; pedestrian-box PSNR is 23.942 dB. Four requested nodes were deleted, none were missing, and repeated-view max absolute difference was 0.0. All ten stored-prior reproduction checks satisfy the original 1e-3 tolerance; maximum difference is 0.000480652.

The four-panel figure is retained for human ghosting review; no visual judgment was made. meta.json contains the checkpoint path, selected tracks, frames, calibration and all 60 node-to-Waymo mappings. No scientific threshold or scene selection changed.

Technical fixes: the current env bin directory exposes installed ninja for nvdiffrast JIT; locally trained OmniRe checkpoints explicitly use PyTorch compatibility loading; prior-only MC.fit_fold calls return the unchanged prior before preparing unrequested Q-based arms. Default/full-arm fitting is unchanged. The actual ten-fold reproduction checks validate the latter change.

Source exam: DATA/runs/nq4/nq4_p3_short3000-exam/20260927-095714. All five collected result files match the remote SHA256 digests; the PNG also passed every chunk CRC and complete-IEND validation. See SHA256SUMS.json.
