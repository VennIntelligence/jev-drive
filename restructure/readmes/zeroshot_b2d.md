# zeroshot_b2d: Zero-shot Bench2Drive, Alpamayo 1.5 and openpilot

status: concluded
decisions: 33
headline: n = 5 smoke: Alpamayo 1.5 DS 60.8, SR 2/5; openpilot DS 2.7 voided as adapter bug, ability unknown

**Question.** Can open models (Alpamayo 1.5, openpilot) drive Bench2Drive zero-shot with native-camera inputs and a fixed controller.

**Conclusion.** n = 5 smoke: Alpamayo 1.5 DS 60.8, RC 70.1, SR 2/5; openpilot DS 2.7 was voided as an adapter bug (plan origin at the camera, 95.6% throttle while the model said stop), so whether openpilot can drive is unknown (decisions 33, corrected in place 2026-09-25). The Alpamayo stall diagnosis found the Zoo PID read standing "reverse" plans as forward speed, with 71 of 80 long stalls starting within 3 s of a collision; fix F1 (forward-only plan) was chosen and the F1 full run was paused at 17/220. openpilot in B2D closed loop continued in openpilot lanes, not here.

**Read more.** research/openpilot-and-open-driving-models.md, research/trajectory-to-control.md, research/openpilot-closedloop-integration.md, `git show bcbdde4:todos/2026-09-24-zeroshot-exam/bench2drive.md`, `git show bcbdde4:todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md`, `git show bcbdde4:todos/2026-09-24-zeroshot-exam/openpilot-migration.md`.

<!-- files:begin -->
<!-- files:end -->
