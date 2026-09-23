# Recovery smoke G3 audit

**PASS**: official2390 Completed, route/driving100, penalty1.0;213control and raw-motion frames357–569 are complete and aligned.

One raw compass NaN at frame538 (simulation9.10s) triggers one gyro-prediction tick with compass age0.05s. The next frame has a valid compass; no fault-brake or invalid control occurs. Controller.step p99 is0.26515ms. The simulator-side cause of the NaN remains unproven.

Runtime1ee2eb2, all48 archived source hashes match their manifest. The server uses physicalGPU1, RTX3090 UUIDb90dd90e-394b-7800-f23f-5892a8e3d0f1.

[audit.json](audit.json) retains all checks, official infractions, exact raw source paths/SHA256 and limits. [frames.csv](frames.csv) provides the complete per-frame raw compass, motion, pose status and controls; [gpu-identity.txt](gpu-identity.txt) records physical GPU mapping. Official completion while moving is not a parking-hold test.
